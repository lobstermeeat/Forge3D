#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p8pics: which way FLUX.1 [schnell]'s reference pictures face (ops/exp_pictures.py) in a staging
# app: production's template and six variants with seed set a, production's again with seed set b, on
# Phase 2's twenty prompts and eight real products. Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p8-pics
OUT=ops-out/private/pictures
mkdir -p "$OUT"
status=0
modal run ops/exp_pictures.py::check --plan "a=v0,v1,v2,v3,v4,v5,v6;b=v0" --out "$OUT/run1" >ops-out/private/modal-pictures.log 2>&1 || status=$?
grep -aE "^\[pictures\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-pictures.log | grep -avE "it/s\]|s/it\]" | tail -400 || true
du -sh "$OUT"
exit $status
