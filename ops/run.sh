#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Deploy the job API that Orainge Studio's AI panel calls"
modal deploy workers/modal_app.py | tee ops-out/deploy.txt
url=$(grep -o 'https://[a-z0-9-]*--orainge-ai-api\.modal\.run' ops-out/deploy.txt | head -1 || true)
if [ -z "$url" ]; then
  echo "The deploy didn't print the job API's URL"
  exit 1
fi
echo "AI_WORKERS_URL=$url"

echo
echo "== The worker token (checked inside Modal; never printed)"
modal run ops/check_api.py::token_check | tee ops-out/token.txt
if grep -q "ORAINGE_WORKER_TOKEN is fine" ops-out/token.txt; then
  echo
  echo "== Check it end to end, the way the server uses it: a prompt, four pictures, a preview"
  modal run ops/check_api.py --url "$url"
else
  echo
  echo "== Until the token is replaced, the API should answer at once with the reason"
  curl -sS -m 150 -w "\nHTTP %{http_code} after %{time_total} s\n" "$url/" || true
fi
