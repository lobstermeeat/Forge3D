#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-p6g: Phase 6's re-test on Phase 4's seeds, in a staging app (ops/exp_retest.py). Nothing is
# deployed; production's orainge-ai keeps running what it runs.
set -euo pipefail
export ORAINGE_APP_NAME=orainge-p6-retest-g
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

echo "== Smoke test: 04 (the multi-view weights) and 06 (thin, so the single-view weights)"
if ! retest ops-out/private/modal-smoke.log --names phase2g --only 04,06 --fresh ||
  ! python3 ops/finals.py "$OUT" --expect pixal3d; then
  echo "The smoke test's finals were not Pixal3D's; stopping before the whole set"
  tail -120 ops-out/private/modal-smoke.log | grep -av "it/s\]" || true
  exit 1
fi

echo
echo "== Phase 2 again from the same pictures, on Phase 4's seeds (phase2g)"
retest ops-out/private/modal-set.log --names phase2g || echo "The set's run exited with an error"
python3 ops/finals.py "$OUT" || true
du -sh "$OUT"
