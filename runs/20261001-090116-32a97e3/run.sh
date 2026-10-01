#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
set -euo pipefail

echo "== Deploy Phase 5: smoothed normals, the out-of-memory fallback and the picture painted on"
modal deploy workers/modal_app.py | tail -3

# One line per final: the pipeline that made it, and whether the picture was painted on
summary() {
  python3 - "$1" <<'EOF'
import json, pathlib, sys
for progress in sorted(pathlib.Path(sys.argv[1]).glob("*-runs/*/progress.json")):
    final = json.loads(progress.read_text())["steps"].get("final", {})
    projection = final.get("projection") or {}
    print(
        f"{progress.parent.name}: {final.get('status')} pipeline={final.get('pipeline')} "
        f"painted={projection.get('applied')} ({projection.get('reason', '')}) "
        f"iou={projection.get('iou')} projection_s={projection.get('seconds')} gpu_s={final.get('gpu_seconds')}"
    )
EOF
}

echo
echo "== Smoke test: the shield on its Phase 4 seed, to see the projection work on a GPU"
modal run ops/rerun_set.py --source phase2 --names phase2d --only 06 --out ops-out/private/all
summary ops-out/private/all
if grep -qs '"reason": "error' ops-out/private/all/phase2d-runs/*/progress.json; then
  echo "The projection failed on the GPU; stopping before the whole set"
  modal app logs orainge-ai --since 30m --tail 600 2>/dev/null | grep -i "forge3d\|error\|traceback" | tail -60 || true
  exit 1
fi

echo
echo "== Phase 2 again from the same pictures: phase2d on Phase 4's seeds, phase2e on the next ones"
modal run ops/rerun_set.py --source phase2 --names phase2d,phase2e --out ops-out/private/all
summary ops-out/private/all
du -sh ops-out/private/all

echo
echo "== Worker log: memory fallbacks and failed steps in the last three hours"
modal app logs orainge-ai --since 3h --tail 6000 2>/dev/null |
  grep -i "out of GPU memory\|falling back\|shading normals failed\|projection: {\"applied\": false" | tail -60 || true
