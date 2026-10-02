#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-texjob: texture options on a real GPU (ops/exp_texjob.py) in a staging app: the Trellis2 worker
# class makes a final and then a "textures" job for the same picture and seed, with the judge (Judge8B)
# picking among the four. Nothing is deployed.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p7-texjob
OUT=ops-out/private/texjob
mkdir -p "$OUT"
status=0
modal run ops/exp_texjob.py::check --only 04,12,14 --count 3 --judge --out "$OUT" >ops-out/private/modal-texjob.log 2>&1 || status=$?
grep -aE "^\[texjob\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" ops-out/private/modal-texjob.log |
  grep -avE "it/s\]|s/it\]" | tail -200 || true
du -sh "$OUT"
exit $status
