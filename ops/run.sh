#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Deploy Phase 4: cleaner textures, the memory retry, picture scores and warm-up"
modal deploy workers/modal_app.py | tee ops-out/deploy.txt
url=$(grep -o 'https://[a-z0-9-]*--orainge-ai-api\.modal\.run' ops-out/deploy.txt | head -1 || true)
[ -n "$url" ] || { echo "The deploy didn't print the job API's URL"; exit 1; }

echo
echo "== Phase 2 again from the same 20 pictures: the same seeds (phase2b) and the next seeds (phase2c)"
modal run ops/rerun_set.py --source phase2 --names phase2b,phase2c --out ops-out/private
du -sh ops-out/private/*

echo
echo "== The job API end to end, with scores and warm-up"
modal run ops/check_api.py --url "$url" --prompt "a wooden rocking chair" --out ops-out/private/api-check
