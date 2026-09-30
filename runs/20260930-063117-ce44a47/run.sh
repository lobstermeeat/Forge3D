#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail
url=https://lobstermeeat--orainge-ai-api.modal.run

echo "== One request to the job API without the token (it should answer 401 at once)"
curl -sS -m 150 -o ops-out/api-answer.txt -w "HTTP %{http_code} after %{time_total} s\n" "$url/" || echo "No answer"
head -c 300 ops-out/api-answer.txt 2>/dev/null || true
echo

echo
echo "== The job API's logs from the last 2 hours"
modal app logs orainge-ai --since 2h --tail 300 || true

echo
echo "== The worker token (checked inside Modal; never printed)"
modal run ops/check_api.py::token_check
