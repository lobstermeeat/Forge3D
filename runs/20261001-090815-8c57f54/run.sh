#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Deploy: torch's GPU cache is freed before remeshing and after the projection"
modal deploy workers/modal_app.py | tail -3

echo
echo "== Finish the Phase 5 run that ran out of memory while remeshing (phase2e-17; phase2d-17 is done and skipped)"
modal run ops/rerun_set.py --source phase2 --names phase2d,phase2e --only 17 --out ops-out/private/all
python3 - <<'PY'
import json, pathlib
for progress in sorted(pathlib.Path("ops-out/private/all").glob("*-runs/*/progress.json")):
    final = json.loads(progress.read_text())["steps"].get("final", {})
    projection = final.get("projection") or {}
    print(f"{progress.parent.name}: {final.get('status')} pipeline={final.get('pipeline')} "
          f"painted={projection.get('applied')} ({projection.get('reason', '')}) gpu_s={final.get('gpu_seconds')}")
PY

echo
echo "== Worker log: memory in the last hour"
modal app logs orainge-ai --since 1h --tail 3000 2>/dev/null | grep -i "out of GPU memory\|falling back" | tail -20 || true
