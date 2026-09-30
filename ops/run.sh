#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

# The picture chosen for each prompt, from the contact sheets of step 1
picks="1=4,2=3,3=2,4=4,5=4,6=3,7=2,8=4,9=2,10=3,11=2,12=4,13=2,14=4,15=4,16=2,17=3,18=4,19=2,20=1"

echo "== Phase 2, step 2: a preview and a final from each picked picture"
modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/phase2.txt --name phase2 --picks "$picks"

echo
echo "== The runs, for the gallery and scoring"
mkdir -p ops-out/private/phase2-runs
cp -r orainge-outputs/phase2/. ops-out/private/phase2-runs/
du -sh ops-out/private/phase2-runs

echo
echo "== Status"
python workers/modal_app.py status | tail -45
