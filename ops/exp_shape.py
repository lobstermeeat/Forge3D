"""
TRELLIS.2 with more than production's finals ask of it, on objects the painter tested, in a staging app
(ORAINGE_APP_NAME; nothing is deployed). Each object's picture and final seed come from the painter's cache in
the orainge-outputs volume (p8-paint/<name>/picture.png and meta.json), so the shapes compare with today's finals
(p8-paint/<name>/today.glb: production's 1024_cascade, 100,000 faces, 2048 texture) on the same picture and noise:

- prod: production's preset on this GPU (an H100 samples differently from production's L40S);
- t4k, f300: production's 1024_cascade with a 4096 texture, at 100,000 and 300,000 faces;
- faces: production's 1024_cascade, exported as TRELLIS.2's README example does (1,000,000 faces, 4096 texture);
- max: '1536_cascade' with that same export.

Both go through production's handle_job (the picture's projection, shading normals, glass, gltfpack) with the
final preset swapped, on an H100 (1536³ wants more than an L40S's 48 GB at times).

    ORAINGE_APP_NAME=orainge-p9-shape modal run ops/exp_shape.py::check --only bmw,p5,p2,p6
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import pathlib
import sys
import time

import modal

if modal.is_local() and os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
# Here: the checkout's workers/. In the container: /root, where the image puts modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app

CACHE_DIR = "p8-paint"  # the painter experiment's objects: picture.png, meta.json (seed), today.glb
OUT_DIR = "p9-shape"
VARIANTS = {
    # Production's final preset on this GPU, so the others compare on the same hardware's samples
    "prod": {"pipeline_type": "1024_cascade", "max_faces": 100_000, "texture_size": 2048},
    "t4k": {"pipeline_type": "1024_cascade", "max_faces": 100_000, "texture_size": 4096},
    "f300": {"pipeline_type": "1024_cascade", "max_faces": 300_000, "texture_size": 4096},
    "faces": {"pipeline_type": "1024_cascade", "max_faces": 1_000_000, "texture_size": 4096},
    "max": {"pipeline_type": "1536_cascade", "max_faces": 1_000_000, "texture_size": 4096},
}

image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")


class _Memory:
    """Storage for handle_job that keeps what it is given."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        self.data[key] = data
        return {"key": key, "bytes": len(data)}


@app.cls(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=65536,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=2,
)
class BigShapes:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[shape] TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def make(self, job: dict) -> dict:
        import torch

        from forge3d_worker import settings
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.service import handle_job

        name, variant = job["name"], job["variant"]
        folder = pathlib.Path(prod.OUTPUTS) / CACHE_DIR / name
        picture = (folder / "picture.png").read_bytes()
        seed = int(json.loads((folder / "meta.json").read_text())["seed"])
        # handle_job reads the final's preset from this dict (service imports the same object)
        original = settings.PRESETS["final"]
        settings.PRESETS["final"] = dataclasses.replace(original, **VARIANTS[variant])
        torch.cuda.reset_peak_memory_stats()
        storage = _Memory()
        clock = time.time()
        try:
            result = handle_job(
                {
                    "id": f"p9-shape-{name}-{variant}",
                    "input": {
                        "image_base64": base64.b64encode(picture).decode("ascii"),
                        "mode": "final",
                        "seed": seed,
                        "request_id": f"p9-shape-{name}-{variant}",
                    },
                },
                self.runtime,
                storage,
                pack_glb,
            )
        finally:
            settings.PRESETS["final"] = original
        seconds = round(time.time() - clock, 1)
        if result.get("error"):
            print(f"[shape] {name} {variant}: {result['error']}")
            return {"name": name, "variant": variant, "error": result["error"]}
        glb = storage.data[result["glb"]["key"]]
        meta = {
            "name": name,
            "variant": variant,
            "settings": VARIANTS[variant],
            "seed": seed,
            "triangles": result.get("triangles"),
            "pipeline": result.get("pipeline"),
            "timings": result.get("timings"),
            "projection": result.get("projection"),
            "glass": result.get("glass"),
            "bytes": len(glb),
            "seconds": seconds,
            "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
            "gpu": self.gpu,
        }
        out = pathlib.Path(prod.OUTPUTS) / OUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{variant}.glb").write_bytes(glb)
        (out / f"{variant}.json").write_text(json.dumps(meta, indent=2, default=str))
        prod.outputs.commit()
        prod.share_caches()
        print(f"[shape] {name} {variant}: {json.dumps(meta, default=str)}")
        return {"name": name, "variant": variant, "meta": meta, "glb": glb}


def _read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


@app.local_entrypoint()
def check(only: str = "bmw", variants: str = "faces,max", out: str = "ops-out/private/shape") -> None:
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    names = [name.strip() for name in only.split(",") if name.strip()]
    wanted = [variant.strip() for variant in variants.split(",") if variant.strip()]
    unknown = [variant for variant in wanted if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
        for file in ("today.glb", "picture.png", "meta.json"):
            (root / name / file).write_bytes(_read(prod.outputs, f"{CACHE_DIR}/{name}/{file}"))
    jobs = [{"name": name, "variant": variant} for name in names for variant in wanted]
    print(f"[shape] app {prod.APP_NAME}: {len(jobs)} shapes: {jobs}")
    report: dict = {}
    failed = 0
    for job, made in zip(jobs, BigShapes().make.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            report.setdefault(job["name"], {})[job["variant"]] = {"error": f"{type(made).__name__}: {reason}"}
            print(f"[shape] {job['name']} {job['variant']}: failed: {type(made).__name__}: {reason}")
            continue
        if made.get("error"):
            failed += 1
            report.setdefault(job["name"], {})[job["variant"]] = {"error": made["error"]}
            continue
        (root / job["name"] / f"{job['variant']}.glb").write_bytes(made["glb"])
        report.setdefault(job["name"], {})[job["variant"]] = made["meta"]
    (root / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[shape] done: {len(jobs) - failed} of {len(jobs)} shapes")
    if failed:
        raise SystemExit(1)
