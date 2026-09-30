#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Image to 3D on six sample renders (FLUX isn't needed for photos)"
modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/renders.txt --name renders
bash ops/gallery.sh "Image to 3D: sample renders"

echo
echo "== Status"
python workers/modal_app.py status
