"""
One line per final in a re-run's output (ops-out/.../<name>-runs/<run>/progress.json): its status, the
model that made it and the Pixal3D recipe's notes (weights, the thin measurement, the levelling), the
pipeline, the projection and the GPU time.

    python3 ops/finals.py ops-out/private/all
    python3 ops/finals.py ops-out/private/all --expect pixal3d   # exit 1 unless every final is Pixal3D's
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def describe(final: dict) -> str:
    thin = final.get("thin") or {}
    level = final.get("level") or {}
    projection = final.get("projection") or {}
    parts = [
        f"{final.get('status')}",
        f"model={final.get('model')}",
        f"weights={final.get('weights')}",
        f"thin={thin.get('ratio')}",
        f"level={level.get('elevation')}" if level.get("applied") else "level=no",
        f"pipeline={final.get('pipeline')}",
        f"painted={projection.get('applied')} ({projection.get('reason', '')})",
        f"gpu_s={final.get('gpu_seconds')}",
    ]
    if final.get("fallback"):
        parts.append(f"fallback={final['fallback']!r}")
    if final.get("error"):
        parts.append(f"error={str(final['error'])[:200]!r}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--expect", default="", help="the model every final must have been made by")
    args = parser.parse_args()
    bad = 0
    found = 0
    for progress in sorted(pathlib.Path(args.folder).glob("*-runs/*/progress.json")):
        final = json.loads(progress.read_text())["steps"].get("final", {})
        found += 1
        print(f"{progress.parent.name}: {describe(final)}")
        if args.expect and (final.get("status") != "done" or final.get("model") != args.expect):
            bad += 1
    if args.expect and (bad or not found):
        print(f"{bad} of {found} finals were not made by {args.expect}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
