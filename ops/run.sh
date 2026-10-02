#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p6h: Phase 6's re-test on the next seeds (a second try), in its own staging app
# (ops/exp_retest.py), beside ai-ops-p6g. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p6-retest-h
OUT=ops-out/private/all
mkdir -p "$OUT"

# The whole `modal run` output (the workers' logs stream into it) is kept privately; the log shows
# the lines that matter
retest() {
  local log="$1"
  shift
  local status=0
  modal run ops/exp_retest.py::retest --source phase2 --out "$OUT" "$@" >"$log" 2>&1 || status=$?
  grep -aE "^\[retest\]|runs in phase2|Starting|Done:|failed|: final-|\[orainge\]|could not be loaded|out of GPU memory|falling back|Traceback|Error" "$log" |
    grep -avE "it/s\]|s/it\]" | tail -150 || true
  return $status
}

# ai-ops-p6g checks the weights in the shared volume first; starting later keeps the two from
# fetching the same files at once
sleep 300

echo "== Smoke test: 04 on its next seed"
if ! retest ops-out/private/modal-smoke.log --names phase2h --offset 1 --only 04 --fresh ||
  ! python3 ops/finals.py "$OUT" --expect pixal3d; then
  echo "The smoke test's finals were not Pixal3D's; stopping before the whole set"
  tail -120 ops-out/private/modal-smoke.log | grep -av "it/s\]" || true
  exit 1
fi

echo
echo "== Phase 2 again from the same pictures, on the next seeds (phase2h)"
retest ops-out/private/modal-set.log --names phase2h --offset 1 || echo "The set's run exited with an error"
python3 ops/finals.py "$OUT" || true
du -sh "$OUT"
