#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Phase 2, step 1: four pictures for each of the twenty prompts"
modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/phase2.txt --name phase2 --pictures-only

echo
echo "== Contact sheets for picking"
python -m pip install --quiet "pillow>=10.1"
python ops/contact_sheets.py orainge-outputs/phase2 ops-out/private/phase2-pictures
du -sh ops-out/private/phase2-pictures
