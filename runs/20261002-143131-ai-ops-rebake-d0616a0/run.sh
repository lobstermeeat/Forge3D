#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-rebake: the cached texture layout on a real GPU (ops/exp_rebake.py), then the textures job
# with it (ops/exp_texjob.py), in staging apps. Nothing is deployed.
set -euo pipefail
OUT=ops-out/private
mkdir -p "$OUT"
status=0
ORAINGE_APP_NAME=orainge-p7-rebake modal run ops/exp_rebake.py::check --only 04,12 --out "$OUT/rebake" >"$OUT/modal-rebake.log" 2>&1 || status=$?
grep -aE "^\[rebake\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" "$OUT/modal-rebake.log" | grep -avE "it/s\]|s/it\]" | tail -150 || true
echo
echo "== The textures job with the cached layout"
ORAINGE_APP_NAME=orainge-p7-texjob modal run ops/exp_texjob.py::check --only 04,12 --count 3 --out "$OUT/texjob" >"$OUT/modal-texjob.log" 2>&1 || status=$?
grep -aE "^\[texjob\]|Traceback|Error|error" "$OUT/modal-texjob.log" | grep -avE "it/s\]|s/it\]" | tail -100 || true
du -sh "$OUT"
exit $status
