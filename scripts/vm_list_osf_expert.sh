#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
curl -sS 'https://api.osf.io/v2/nodes/xbawc/files/osfstorage/677817635e3d93d497e6e40f/' -o /tmp/osf_expert.json
python3 - <<'PY'
import json
from pathlib import Path
d=json.load(open('/tmp/osf_expert.json'))
for x in d.get('data',[]):
    a=x['attributes']
    print(a['kind'], a['name'], a.get('size'), x['links'].get('download'))
PY
