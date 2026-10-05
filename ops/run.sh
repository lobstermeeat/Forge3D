#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paint: Phase 8, the view painter, in staging apps. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
#
# Run 6: paint_views as it now is (brightness and saturation matched, held to the picture's own colour; each
# texel mostly from its best view; no dark bottom), baked at 4096 and also shipped at 2048. The six shapes of
# runs 4 and 5, and seven real products (p2-p8: drawn four times each with today's template, the best taken,
# then shaped). Then five of them again with flatter light in the prompt.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-paint
status=0
modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13,p2,p3,p4,p5,p6,p7,p8 \
  --variants qie-v6 --out ops-out/private/paint6 \
  >ops-out/private/modal-paint6.log 2>&1 || status=$?
modal run ops/exp_paint.py::check --only bmw,13,p2,p3,p5 \
  --variants qie-v6-flat --out ops-out/private/paint6flat \
  >ops-out/private/modal-paint6flat.log 2>&1 || status=$?
for log in ops-out/private/modal-paint6.log ops-out/private/modal-paint6flat.log; do
  echo "== $log"
  grep -aE "^\[paint\]|^\[painter\]|^\[shapes\]|\[forge3d\] (glass|projection)|Traceback|Error|error" "$log" |
    grep -avE "it/s\]|s/it\]" | cut -c1-400 | tail -200 || true
done
du -sh ops-out/private/paint6 ops-out/private/paint6flat 2>/dev/null || true
exit $status
