#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== FLUX.1 [schnell] weights (needs its terms accepted on Hugging Face)"
flux=yes
modal run --detach workers/modal_app.py::download_models --which reference || flux=no

echo
echo "== Image to 3D: six sample renders (finished runs are only collected again)"
modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/renders.txt --name renders
bash ops/gallery.sh orainge-outputs/renders "Orainge 3D First Look" \
  "Six product shots of free sample models (Khronos, CC0), turned into 3D by TRELLIS.2 on Modal. Each card shows the input photo, then the final and the preview from six angles. View in 3D opens the final to turn, zoom and check its wireframe."

if [ "$flux" = yes ]; then
  echo
  echo "== Text to 3D: the starter set"
  modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/starter.txt --name starter
  bash ops/gallery.sh orainge-outputs/starter "Orainge Starter Set" \
    "Eight prompts, each a different kind of game asset. FLUX.1 [schnell] draws four pictures per prompt (the outlined one went on to 3D), then TRELLIS.2 builds the preview and the final. View in 3D opens the final to turn, zoom and check its wireframe."
else
  echo
  echo "FLUX isn't available yet, so the starter set waits"
fi

echo
echo "== Status"
python workers/modal_app.py status
