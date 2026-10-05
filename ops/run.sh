#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paintskip: Phase 8, the view painter, in a staging app of its own. Nothing is deployed; production's
# orainge-ai keeps running what it runs.
#
# Run 10: production's candidate, each try at a view further into the schedule (qie-v9: pure noise, then the
# render noised to 0.95, then to 0.9), on all thirteen shapes; and skip 1 alone (qie-s1) on the four objects whose
# views run 7 lost to a catalogue angle (the shield 06, the sneaker p2, the controller p3, the watch p7).
set -euo pipefail
mkdir -p ops-out/private
status=0
# Side by side, each in a staging app of its own
ORAINGE_APP_NAME=orainge-p8-skip modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13,p2,p3,p4,p5,p6,p7,p8 \
  --variants qie-v9 --attempts 3 --out ops-out/private/paint10 \
  >ops-out/private/modal-paint10.log 2>&1 &
first=$!
ORAINGE_APP_NAME=orainge-p8-skip1 modal run ops/exp_paint.py::check --only 06,p2,p3,p7 \
  --variants qie-s1 --attempts 3 --out ops-out/private/paint10s1 \
  >ops-out/private/modal-paint10s1.log 2>&1 &
second=$!
wait $first || status=$?
wait $second || status=$?
grep -aE "^\[paint\] (app|making|painting|done|[a-z0-9]+ qie)|^\[shapes\]|Traceback|Error|error" ops-out/private/modal-paint10.log ops-out/private/modal-paint10s1.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-300 | tail -80 || true
du -sh ops-out/private/paint10 ops-out/private/paint10s1 2>/dev/null || true
exit $status
