#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p8pics: which way FLUX.1 [schnell]'s reference pictures face (ops/exp_pictures.py) in a staging
# app, on Phase 2's twenty prompts and eight real products. Round three: template v11 with seed set a, and
# production's template, v6 and v11 with fresh seeds (set c), all scored with the view check; then FLUX.2
# [klein] 4B with production's template and v11 (ops/exp_klein.py, its own staging app). Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p8-pics
OUT=ops-out/private/pictures
mkdir -p "$OUT"
status=0
modal run ops/exp_pictures.py::check --plan "a=v11;c=v0,v6,v11" --out "$OUT/run3" >ops-out/private/modal-pictures.log 2>&1 || status=$?
grep -aE "^\[pictures\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-pictures.log | grep -avE "it/s\]|s/it\]|aclose|asynchronous generator" | tail -300 || true
echo
echo "== FLUX.2 [klein] 4B"
modal run ops/exp_klein.py::check --plan "a=v0,v11" --out "$OUT/klein" >ops-out/private/modal-klein.log 2>&1 || status=$?
grep -aE "^\[pictures\]|Traceback|Error|error" ops-out/private/modal-klein.log | grep -avE "it/s\]|s/it\]|aclose|asynchronous generator" | tail -200 || true
du -sh "$OUT"
exit $status
