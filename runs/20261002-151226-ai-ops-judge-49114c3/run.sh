#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-judge: Phase 7's judge experiment (ops/exp_judge.py) in a staging app. Nothing is deployed;
# production's orainge-ai keeps running what it runs.
#
# This run, the judge's second check: each of the 20 pictures at Phase 4's seeds (phase2) gets its final
# and three more textures on the same shape, from new noise (seed + 4000, 5000, 6000; the rolls used 1000
# to 3000); Qwen3-VL 8B picks among them, in two orders.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p7-judge
OUT=ops-out/private/judge3
mkdir -p "$OUT"

# The whole `modal run` output (the containers' logs stream into it) is kept privately; the log shows
# the lines that matter
status=0
modal run ops/exp_judge.py::judges --out "$OUT" --prefix phase2 --count 3 --first 4 --tag t10roll --sizes 8b --orders 2 >ops-out/private/modal-judge.log 2>&1 || status=$?
grep -aE "^\[judge\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-judge.log |
  grep -avE "it/s\]|s/it\]|^\[forge3d\] projection:" | tail -300 || true
du -sh "$OUT"
exit $status
