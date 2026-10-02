#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-judge: Phase 7's judge experiment (ops/exp_judge.py) in a staging app. Nothing is deployed;
# production's orainge-ai keeps running what it runs.
#
# Each of the 20 pictures at the next seeds (phase2e) gets its final and three more textures on the same
# shape; the self-hosted judges (Qwen3-VL 8B and 30B-A3B) pick among them, in two orders each.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p7-judge
OUT=ops-out/private/judge
mkdir -p "$OUT"

# The whole `modal run` output (the containers' logs stream into it) is kept privately; the log shows
# the lines that matter
status=0
modal run ops/exp_judge.py::judges --out "$OUT" --count 3 >ops-out/private/modal-judge.log 2>&1 || status=$?
grep -aE "^\[judge\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-judge.log |
  grep -avE "it/s\]|s/it\]|^\[forge3d\] projection:" | tail -300 || true
du -sh "$OUT"
exit $status
