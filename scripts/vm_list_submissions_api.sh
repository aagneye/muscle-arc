#!/usr/bin/env bash
set -euo pipefail
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/umud-challenge-muscle-architecture-in-ultrasound-data?pageSize=5" \
  | head -c 2000
echo
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://www.kaggle.com/api/v1/competitions/submissions/list/umud-challenge-muscle-architecture-in-ultrasound-data?pageSize=5" \
  | head -c 2000
echo
