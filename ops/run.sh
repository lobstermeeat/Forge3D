#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p8glass: glass in TRELLIS.2's finals (ops/exp_glass.py) in a staging app. Measure: the founder's
# BMW (its picture drawn again) and Phase 2's helmet, cartoon car, bubble tea, potion and dragon, made as
# production makes them, with to_glb's RGBA textures saved from before unpremultiply. Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p8-glass
OUT=ops-out/private/glass/measure
mkdir -p "$OUT"
status=0
modal run ops/exp_glass.py::check --mode measure --only bmw,10,13,14,15,16 --textures bmw --count 3 \
  --out "$OUT" >ops-out/private/modal-glass.log 2>&1 || status=$?
grep -aE "^\[glass\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-glass.log |
  grep -avE "it/s\]|s/it\]" | tail -200 || true
du -sh "$OUT"
exit $status
