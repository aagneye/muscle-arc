#!/usr/bin/env python3
"""Local web server for the manual measurement tool (tools/labeler/index.html).

Standard library only. Binds to 127.0.0.1; reach it from a laptop through an SSH
tunnel:  ssh -N -L 8765:127.0.0.1:8765 azureuser@<vm>   then open http://localhost:8765

Routes
  GET  /                      tools/labeler/index.html
  GET  /manifest.json         data/labeling/manifest.json
  GET  /images/<file>         data/labeling/images/<file>
  GET  /api/labels            all saved records {image_id: record}
  POST /api/labels/<image_id> save one record (JSON body)

Saves are atomic (write tmp + rename); every ``backup_every`` saves a copy goes
to data/labeling/backups/. Never commit data/labeling (public repo).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MIME = {".html": "text/html; charset=utf-8", ".json": "application/json", ".png": "image/png", ".js": "text/javascript"}


def safe_child(base: Path, name: str) -> Path | None:
    """Resolve ``name`` inside ``base``; None if it escapes base or does not exist."""
    base = base.resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        return None
    return p


class LabelStore:
    def __init__(self, path: Path, backup_every: int = 10, allowed_ids: set[str] | None = None) -> None:
        self.path = path
        self.backup_every = backup_every
        self.allowed_ids = allowed_ids
        self._lock = threading.Lock()
        self._n = 0

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def save(self, image_id: str, record: dict) -> None:
        if self.allowed_ids is not None and image_id not in self.allowed_ids:
            raise ValueError(f"unknown image_id {image_id!r}")
        with self._lock:
            data = self.load()
            data[image_id] = {**record, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=0))
            os.replace(tmp, self.path)
            self._n += 1
            if self._n % self.backup_every == 0:
                bdir = self.path.parent / "backups"
                bdir.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
                (bdir / f"labels_{stamp}_{self._n}.json").write_text(json.dumps(data))


def make_handler(store: LabelStore, data_dir: Path, tool_html: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            if path in ("/", "/index.html"):
                return self._send(200, tool_html.read_bytes(), MIME[".html"])
            if path == "/manifest.json":
                return self._send(200, (data_dir / "manifest.json").read_bytes(), MIME[".json"])
            if path == "/api/labels":
                return self._send(200, json.dumps(store.load()).encode(), MIME[".json"])
            if path.startswith("/images/"):
                f = safe_child(data_dir / "images", path[len("/images/"):])
                if f is not None:
                    return self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = unquote(self.path.split("?")[0])
            if not path.startswith("/api/labels/"):
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                record = json.loads(self.rfile.read(n))
                store.save(path[len("/api/labels/"):], record)
            except (ValueError, json.JSONDecodeError) as exc:
                return self._send(400, str(exc).encode(), "text/plain")
            self._send(200, b'{"ok": true}', MIME[".json"])

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "labeling")
    args = ap.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    ids = {it["image_id"] for it in manifest["images"]}
    store = LabelStore(args.data_dir / "labels.json", allowed_ids=ids)
    handler = make_handler(store, args.data_dir, ROOT / "tools" / "labeler" / "index.html")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"label server on http://127.0.0.1:{args.port}  ({len(ids)} images, {len(store.load())} saved)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
