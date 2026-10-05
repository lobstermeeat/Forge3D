#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paint: Phase 8, the view painter, in staging apps. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
#
# Run 11: the painter as production will run it: each try at a view further into the schedule (pure noise, then
# the render noised to 0.95, then 0.9; qie-v9) and the colours held to the picture only where it sees the surface
# face on, on run 6's thirteen shapes (for blind test 3 against today's finals).
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-paint
status=0
modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13,p2,p3,p4,p5,p6,p7,p8 \
  --variants qie-v9 --attempts 3 --out ops-out/private/paint11 \
  >ops-out/private/modal-paint11.log 2>&1 || status=$?
grep -aE "^\[paint\] (app|making|painting|done|[a-z0-9]+ qie)|^\[shapes\]|Traceback|Error|error" ops-out/private/modal-paint11.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-300 | tail -80 || true
du -sh ops-out/private/paint11 2>/dev/null || true
exit $status
