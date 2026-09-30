#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== FLUX.1 [schnell] weights (the terms are accepted now)"
modal run --detach workers/modal_app.py::download_models --which reference

echo
echo "== Text to 3D: the starter set"
modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/starter.txt --name starter
bash ops/gallery.sh orainge-outputs/starter "Orainge Starter Set" \
  "Eight prompts, each a different kind of game asset. FLUX.1 [schnell] draws four pictures per prompt (the outlined one went on to 3D), then TRELLIS.2 builds the preview and the final. View in 3D opens the final to turn, zoom and check its wireframe."

echo
echo "== Status"
python workers/modal_app.py status
