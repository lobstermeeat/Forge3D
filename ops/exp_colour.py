"""
Phase 8: how far texture options' colours drift from the picture, on a real GPU, in a staging app
(ORAINGE_APP_NAME; nothing is deployed).

For each object (the BMW, its picture drawn again by the reference worker at its seed, and picked Phase 2
pictures at their seeds) a final and three texture options are made the way production makes them:
service.handle_job with a "final" job, then a "textures" job for the same picture and seed, on
Trellis2Runtime, packed with pack_glb. For every export this keeps, under --out/<object>/<export>/:

- pre.npz, pre-basecolor.png, pre-mr.png: to_glb's mesh as the projection gets it (after unpremultiply):
  vertices, faces and UVs, its base colour and metallic-roughness textures (geometry.json instead of
  pre.npz when the export's geometry is an earlier one's: a rebaked texture's);
- model.glb: the packed GLB the job returned;
- report.json: the projection's full report (the job's result only has its summary);

and <object>/picture.png and <object>/cutout.png (the background-removed picture the projection paints
from). With --paired, each export's projection runs twice, first without the colour match on a copy
(before-basecolor.png, before-report.json), then as production runs it. summary.json has each job's result
without its files.

    ORAINGE_APP_NAME=orainge-p8-colour modal run ops/exp_colour.py::check --only bmw,02,03,05,09,13,20
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
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
# Here: the checkout's workers/. In the container: /root, where the image puts modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app

# The founder's BMW: the reference worker's picture at this seed, made into 3D at the TRELLIS.2 seed
BMW_PROMPT = "make a bmw car m3 model blue"
BMW_PICTURE_SEED = 1627471494
BMW_SEED = 663008479
TEXTURES = 3

colour_image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")
STARTED = time.time()


class Memory:
    """A job's storage: keeps what it is given, by key."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        self.files[key] = data
        return {"key": key, "url": None}


def png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def plain(value):
    """``value`` with whatever json can't write (tensors, numpy) made into plain values."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


@app.cls(
    image=colour_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=3,
)
class Colour:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[colour] TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def run(self, job: dict) -> dict:
        import copy

        import numpy as np
        from PIL import Image

        from forge3d_worker import projection, service
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import CUTOUT

        runtime = self.runtime
        name, seed, paired = job["name"], int(job["seed"]), bool(job.get("paired"))
        files: dict[str, bytes] = {f"{name}/picture.png": job["picture"]}
        exports: list[dict] = []  # one per export, in order: what the projection got and said
        geometries: dict[str, str] = {}  # vertex hash -> the export whose pre.npz has it

        def before_projection(glb, mesh) -> None:
            index = len(exports)
            entry: dict = {"index": index}
            material = glb.visual.material
            vertices = np.asarray(glb.vertices, dtype=np.float32)
            faces = np.asarray(glb.faces, dtype=np.int32)
            uv = np.asarray(glb.visual.uv, dtype=np.float32)
            digest = hashlib.sha256(vertices.tobytes() + faces.tobytes() + uv.tobytes()).hexdigest()[:16]
            entry["geometry"] = digest
            if digest not in geometries:
                geometries[digest] = f"export-{index}"
                buffer = io.BytesIO()
                np.savez_compressed(buffer, vertices=vertices, faces=faces, uv=uv)
                entry["files"] = {"pre.npz": buffer.getvalue()}
            else:
                entry["files"] = {"geometry.json": json.dumps({"same_as": geometries[digest]}).encode()}
            entry["files"]["pre-basecolor.png"] = png(material.baseColorTexture)
            if getattr(material, "metallicRoughnessTexture", None) is not None:
                entry["files"]["pre-mr.png"] = png(material.metallicRoughnessTexture)
            entry["factors"] = {
                "metallic": getattr(material, "metallicFactor", None),
                "roughness": getattr(material, "roughnessFactor", None),
            }
            cutout = getattr(mesh, CUTOUT, None)
            if isinstance(cutout, Image.Image) and f"{name}/cutout.png" not in files:
                files[f"{name}/cutout.png"] = png(cutout)
            exports.append(entry)

        stock = projection.project_picture

        def project_picture(mesh, picture, device=None, debug=None):
            entry = exports[-1] if exports else {}
            if paired:
                twin = copy.deepcopy(mesh)
                with colour_match(projection, False):
                    twin, before = stock(twin, picture, device=device)
                entry.setdefault("files", {})["before-basecolor.png"] = png(twin.visual.material.baseColorTexture)
                entry["before_report"] = plain(before)
            mesh, report = stock(mesh, picture, device=device, debug=debug)
            entry["report"] = plain(report)
            return mesh, report

        runtime.before_projection = before_projection
        projection.project_picture = project_picture
        storage = Memory()
        image = base64.b64encode(job["picture"]).decode()
        results = {}
        try:
            for mode, extra in (("final", {}), ("textures", {"count": TEXTURES})):
                started = time.time()
                payload = {"image_base64": image, "mode": mode, "seed": seed, "request_id": f"p8-colour-{name}", **extra}
                result = service.handle_job({"id": f"{mode}-{name}", "input": payload}, runtime, storage, pack_glb)
                result["seconds"] = round(time.time() - started, 1)
                results[mode] = result
                print(f"[colour] {name} {mode}: {json.dumps(plain(result), default=str)[:2000]}")
        finally:
            projection.project_picture = stock
            runtime.before_projection = None

        # Exports in order: the final, then texture 1 to 3; each with its packed GLB from the job's storage
        keys = [f"ai/p8-colour-{name}/final-{seed}.glb"] + [
            f"ai/p8-colour-{name}/final-{seed}-texture-{k}.glb" for k in range(1, TEXTURES + 1)
        ]
        tags = ["final"] + [f"texture-{k}" for k in range(1, TEXTURES + 1)]
        summary = {"name": name, "seed": seed, "exports": [], "results": plain(results)}
        for tag, key, entry in zip(tags, keys, exports):
            for file, data in entry.pop("files", {}).items():
                files[f"{name}/{tag}/{file}"] = data
            if key in storage.files:
                files[f"{name}/{tag}/model.glb"] = storage.files[key]
            for field in ("report", "before_report"):
                if field in entry:
                    files[f"{name}/{tag}/{field.replace('_', '-')}.json"] = json.dumps(entry[field], indent=1).encode()
            summary["exports"].append(
                {"tag": tag, "geometry": entry["geometry"], "factors": entry["factors"],
                 "projection": projection.summary(entry.get("report") or {})}
            )
        if len(exports) != len(tags):
            summary["warning"] = f"{len(exports)} exports for {len(tags)} models"
        prod.share_caches()
        summary["container_seconds"] = round(time.time() - STARTED, 1)
        summary["gpu"] = self.gpu
        return {"summary": summary, "files": files}


@contextlib.contextmanager
def colour_match(projection, on: bool):
    """The projection with its colour match on or off (projection.COLOUR_MATCH), as it was afterwards."""
    had = getattr(projection, "COLOUR_MATCH", None)
    if had is None:
        yield  # this code has no colour match: the projection as it is
        return
    projection.COLOUR_MATCH = on
    try:
        yield
    finally:
        projection.COLOUR_MATCH = had


def read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the picked pictures)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


def bmw_picture() -> bytes:
    """The BMW's picture, drawn again by the reference worker (FLUX.1 schnell) at its seed."""
    job = {"prompt": BMW_PROMPT, "count": 1, "seed": BMW_PICTURE_SEED, "request_id": "p8-colour-bmw"}
    result = prod.FluxSchnell().generate.remote({"id": "p8-colour-bmw", "input": job})
    if result.get("error"):
        raise RuntimeError(f"the reference worker failed: {result['error']}")
    picture = result["images"][0]
    print(f"[colour] bmw picture: seed {picture['seed']}, score {picture.get('score')}, issues {picture.get('issues')}")
    return base64.b64decode(picture["base64"])


@app.local_entrypoint()
def check(only: str = "bmw,02,03,05,09,13,20", out: str = "ops-out/private/colour", paired: bool = False) -> None:
    print(f"[colour] app {prod.APP_NAME}: a final and {TEXTURES} texture options each{', paired' if paired else ''}")
    root = pathlib.Path(out)
    pictures = runs_by_number(prod.outputs, "phase2")
    jobs = []
    for item in [part.strip() for part in only.split(",") if part.strip()]:
        if item == "bmw":
            jobs.append({"name": "bmw", "seed": BMW_SEED, "picture": bmw_picture(), "paired": paired})
            continue
        number = item.zfill(2)
        if number not in pictures:
            print(f"[colour] no Phase 2 run for {number}")
            continue
        source = pictures[number]
        state = json.loads(read(prod.outputs, f"{source}/progress.json"))
        picture = read(prod.outputs, f"{source}/{state['input']}")
        jobs.append({"name": source, "seed": int(state["seed"]), "picture": picture, "paired": paired})
    listed = ", ".join("{} (seed {})".format(j["name"], j["seed"]) for j in jobs)
    print(f"[colour] {len(jobs)} objects: {listed}")
    summaries, failed = [], 0
    for job, made in zip(jobs, Colour().run.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[colour] {job['name']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"name": job["name"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for name, data in made["files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
        for export in made["summary"]["exports"]:
            print(f"[colour] {job['name']} {export['tag']}: {json.dumps(export['projection'])}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[colour] done: {len(jobs) - failed} of {len(jobs)} objects")
    if not jobs or failed:
        raise SystemExit(1)
