#!/usr/bin/env bash
python3 -c 'import json; x=json.load(open("/tmp/sub.json")); items=x if isinstance(x,list) else x.get("submissions") or x.get("data") or []; print(items[0].get("url")); print(items[0].get("urlNullable"))'
