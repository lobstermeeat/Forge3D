#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Deploy: restarted runs carry on, and exports get the memory retry"
modal deploy workers/modal_app.py | tail -3

echo
echo "== Finish the Phase 4 runs that stopped (the finished ones are skipped)"
modal run ops/rerun_set.py --source phase2 --names phase2b,phase2c --out ops-out/private/all
mkdir -p ops-out/private/finished
for run in phase2b-05-sci-fi-laser-pistol-white phase2b-11-stack-of-old-books-with-a-candle \
  phase2b-13-cute-cartoon-car-bright-blue phase2b-18-electric-guitar phase2c-10-astronaut-helmet \
  phase2c-20-hot-air-balloon; do
  set_name=${run%%-*}
  [ -d "ops-out/private/all/$set_name-runs/$run" ] && cp -r "ops-out/private/all/$set_name-runs/$run" ops-out/private/finished/
done
rm -rf ops-out/private/all
du -sh ops-out/private/finished

echo
echo "== The worker's memory retries in the last hour"
modal app logs orainge-ai --since 1h --tail 2000 2>/dev/null | grep -i "out of GPU memory\|retrying" | tail -20 || true
