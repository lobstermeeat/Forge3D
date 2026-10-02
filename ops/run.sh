#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-tex: Phase 7's texture experiment (ops/exp_texture.py) in a staging app. Nothing is
# deployed; production's orainge-ai keeps running what it runs.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p7-tex
OUT=ops-out/private/tex
mkdir -p "$OUT"

# The whole `modal run` output (the containers' logs stream into it) is kept privately; the log shows
# the lines that matter
texture() {
  local log="$1"
  shift
  local status=0
  modal run ops/exp_texture.py::experiment --out "$OUT" "$@" >"$log" 2>&1 || status=$?
  grep -aE "^\[tex\]|\[forge3d\]|\[orainge\]|Traceback|Error|error" "$log" |
    grep -avE "it/s\]|s/it\]" | tail -200 || true
  return $status
}

echo "== Smoke test: the arcade machine (04), with the replay check"
if ! texture ops-out/private/modal-smoke.log --only 04 --replay 04; then
  echo "The smoke test failed; stopping before the other objects"
  tail -150 ops-out/private/modal-smoke.log | grep -av "it/s\]" || true
  exit 1
fi

echo
echo "== The other eleven objects"
texture ops-out/private/modal-rest.log --only 06,08,11,12,14,18,01,02,13,16,20 || echo "The run exited with an error"
du -sh "$OUT"
