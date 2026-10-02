#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-tex: Phase 7's texture experiments (ops/exp_texture.py) in a staging app. Nothing is
# deployed; production's orainge-ai keeps running what it runs.
#
# This run: rolls. Each of the 20 pictures gets its final and three more textures on the same shape
# (new noise each), rendered from ig2mv's cameras, plus ig2mv's views, for pickers scored afterwards.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p7-tex
OUT=ops-out/private/rolls
mkdir -p "$OUT"

# The whole `modal run` output (the containers' logs stream into it) is kept privately; the log shows
# the lines that matter
status=0
modal run ops/exp_texture.py::rolls --out "$OUT" --count 3 >ops-out/private/modal-rolls.log 2>&1 || status=$?
grep -aE "^\[tex\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-rolls.log |
  grep -avE "it/s\]|s/it\]" | tail -300 || true
du -sh "$OUT"
exit $status
