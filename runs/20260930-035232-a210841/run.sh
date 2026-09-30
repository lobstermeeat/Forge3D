#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log.
set -euo pipefail

echo "== Modal token"
modal token info
echo
echo "== Modal secrets (names only)"
modal secret list
echo
echo "== Status"
python workers/modal_app.py status
