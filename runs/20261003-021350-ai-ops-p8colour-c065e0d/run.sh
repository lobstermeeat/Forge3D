#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p8colour: Phase 8's colour drift (ops/exp_colour.py) in a staging app: the BMW and six Phase 2
# pictures, a final and three texture options each, made as production makes them, with what the
# projection got kept for measuring. Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p8-colour
OUT=ops-out/private/colour
mkdir -p "$OUT"
status=0
modal run ops/exp_colour.py::check --only bmw,02,03,05,09,13,20 --out "$OUT" >ops-out/private/modal-colour.log 2>&1 || status=$?
grep -aE "^\[colour\]|\[forge3d\] (projection|export)|Traceback|Error|error" ops-out/private/modal-colour.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-600 | tail -120 || true
du -sh "$OUT"
exit $status
