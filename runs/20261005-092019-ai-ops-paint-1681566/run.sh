#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paint: Phase 8, the view painter, in staging apps. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
#
# Run 4: the six objects' shapes made again (now with to_glb's RGBA kept, so every export gets see-through
# glass), then painted by klein's best variant from run 3 (edit-view) and by Qwen-Image-Edit-2511 (8-step
# Lightning) three ways: the render alone, with the picture, with the picture and the nearest finished view.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-paint
status=0
modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13 --fresh \
  --variants edit-view,qie-edit-view,qie-ref,qie-ref-nb --out ops-out/private/paint4 \
  >ops-out/private/modal-paint4.log 2>&1 || status=$?
grep -aE "^\[paint\]|^\[painter\]|^\[shapes\]|\[forge3d\] (glass|projection)|Traceback|Error|error" ops-out/private/modal-paint4.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-400 | tail -250 || true
du -sh ops-out/private/paint4 2>/dev/null || true
exit $status
