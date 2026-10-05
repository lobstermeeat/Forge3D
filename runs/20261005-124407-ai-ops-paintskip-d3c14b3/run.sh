#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paintskip: Phase 8, the view painter, in a staging app of its own. Nothing is deployed; production's
# orainge-ai keeps running what it runs.
#
# Run 9: each view started from its render noised part way (QwenPainter's skip 2 and 3: qie-s2, qie-s3) on the
# objects whose views the editing model turned to a catalogue angle in run 7 (the shield 06, the sneaker p2, the
# controller p3, the watch p7), and two that went well (the BMW, the cartoon car 13). Run 8 (ai-ops-paint) is the
# same painter without skip on the same shapes.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-skip
status=0
modal run ops/exp_paint.py::check --only bmw,06,13,p2,p3,p7 \
  --variants qie-s2,qie-s3 --attempts 3 --out ops-out/private/paint9 \
  >ops-out/private/modal-paint9.log 2>&1 || status=$?
grep -aE "^\[paint\] (app|making|painting|done|[a-z0-9]+ qie)|^\[shapes\]|Traceback|Error|error" ops-out/private/modal-paint9.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-300 | tail -80 || true
du -sh ops-out/private/paint9 2>/dev/null || true
exit $status
