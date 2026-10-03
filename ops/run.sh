#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p8glass: glass in TRELLIS.2's finals (ops/exp_glass.py) in a staging app. Verify: the founder's
# BMW and Phase 2's helmet, cartoon car, bubble tea, potion and dragon, each final mesh exported before
# (the colour only divided by alpha) and after glass; texture options both ways for the BMW, the cartoon
# car, the bubble tea and the dragon; a textures job through production's handle_job and a preview for the
# BMW. Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p8-glass
OUT=ops-out/private/glass/verify
mkdir -p "$OUT"
status=0
modal run ops/exp_glass.py::check --mode verify --only bmw,10,13,14,15,16 --textures bmw,13,14,16 --count 3 \
  --jobs bmw --preview bmw --out "$OUT" >ops-out/private/modal-glass.log 2>&1 || status=$?
grep -aE "^\[glass\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-glass.log |
  grep -avE "it/s\]|s/it\]" | cut -c1-1500 | tail -300 || true
du -sh "$OUT"
exit $status
