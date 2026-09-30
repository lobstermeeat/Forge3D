#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log.
set -euo pipefail

echo "== Build the images (20-40 minutes the first time)"
python workers/modal_app.py build

echo
echo "== Download FLUX.1 [schnell] (needs its terms accepted on Hugging Face)"
modal run --detach workers/modal_app.py::download_models --which reference || echo "FLUX download failed (exit $?)"

echo
echo "== Download TRELLIS.2 (DINOv3 needs Meta's approval; finished files are kept either way)"
modal run --detach workers/modal_app.py::download_models --which trellis2 || echo "TRELLIS.2 download failed (exit $?)"

echo
echo "== Status"
python workers/modal_app.py status
