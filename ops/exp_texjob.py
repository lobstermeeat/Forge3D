"""
A check of texture options on a real GPU, in a staging app (ORAINGE_APP_NAME; nothing is deployed): the
Trellis2 worker class as production runs it, asked for a final and then a "textures" job with the same
picture and seed, the way the Studio's server asks. Results come back inline (no R2 in a staging run).

    ORAINGE_APP_NAME=orainge-p7-texjob modal run ops/exp_texjob.py::check --only 04,12 --count 3
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import sys
import time

import modal

if modal.is_local() and os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app


def read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


def saved(asset: dict, target: pathlib.Path) -> int:
    data = base64.b64decode(asset["base64"]) if asset.get("base64") else b""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


@app.local_entrypoint()
def check(only: str = "04,12", count: int = 3, out: str = "ops-out/private/texjob", judge: bool = False) -> None:
    print(f"[texjob] app {prod.APP_NAME}")
    root = pathlib.Path(out)
    pictures = runs_by_number(prod.outputs, "phase2")
    worker = prod.Trellis2()
    report = []
    for number in [n.strip().zfill(2) for n in only.split(",") if n.strip()]:
        source = pictures[number]
        state = json.loads(read(prod.outputs, f"{source}/progress.json"))
        picture = read(prod.outputs, f"{source}/{state['input']}")
        seed = int(state["seed"])
        base = {"image_base64": base64.b64encode(picture).decode(), "seed": seed, "request_id": f"p7-texjob-{number}"}
        entry: dict = {"source": source, "seed": seed}
        prompt = state.get("prompt") or ""
        textures = {"count": count, **({"judge": True, "prompt": prompt} if judge else {})}
        for mode, extra in (("final", {}), ("textures", textures)):
            clock = time.time()
            result = worker.generate.remote({"id": f"{mode}-{number}", "input": {**base, "mode": mode, **extra}})
            seconds = round(time.time() - clock, 1)
            if result.get("error"):
                entry[mode] = {"error": result["error"], "seconds": seconds}
                print(f"[texjob] {source} {mode}: error: {result['error']}")
                continue
            if mode == "final":
                size = saved(result["glb"], root / f"{source}/final-{seed}.glb")
                entry[mode] = {"seconds": seconds, "bytes": size, "triangles": result.get("triangles"),
                               "pipeline": result.get("pipeline"), "timings": result.get("timings")}
            else:
                textures = []
                for texture in result.get("textures", []):
                    size = saved(texture["glb"], root / f"{source}/texture-{texture['texture_seed']}.glb")
                    textures.append({"texture_seed": texture["texture_seed"], "bytes": size, "triangles": texture.get("triangles"),
                                     "projection": (texture.get("projection") or {}).get("reason")})
                entry[mode] = {"seconds": seconds, "textures": textures, "texture_errors": result.get("texture_errors"),
                               "pipeline": result.get("pipeline"), "timings": result.get("timings"),
                               "keys": [t["glb"].get("key") for t in result.get("textures", [])],
                               "exports": [(t.get("export") or {}).get("path") for t in result.get("textures", [])],
                               "own_texture": result.get("own_texture"), "judge": result.get("judge"),
                               "judge_error": result.get("judge_error")}
            print(f"[texjob] {source} {mode}: {json.dumps(entry[mode], default=str)}")
        (root / source / "picture.png").parent.mkdir(parents=True, exist_ok=True)
        (root / source / "picture.png").write_bytes(picture)
        report.append(entry)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    bad = [e["source"] for e in report if "error" in e.get("textures", {}) or not e.get("textures", {}).get("textures")]
    print(f"[texjob] done; without textures: {bad or 'none'}")
    if bad:
        raise SystemExit(1)
