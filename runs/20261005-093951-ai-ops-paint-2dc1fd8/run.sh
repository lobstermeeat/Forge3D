#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paint: Phase 8, the view painter, in staging apps. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
#
# Run 5: the same six shapes (kept from run 4), painted by Qwen-Image-Edit-2511 from the render alone, with the
# views' colours now matched jointly; once into the final's 2048 base colour and once into a 4096 one.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-paint
status=0
modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13 \
  --variants qie-edit-view,qie-edit-view-4k --out ops-out/private/paint5 \
  >ops-out/private/modal-paint5.log 2>&1 || status=$?
grep -aE "^\[paint\]|^\[painter\]|^\[shapes\]|\[forge3d\] (glass|projection)|Traceback|Error|error" ops-out/private/modal-paint5.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-400 | tail -250 || true
du -sh ops-out/private/paint5 2>/dev/null || true
exit $status
