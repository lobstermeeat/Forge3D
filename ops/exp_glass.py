"""
Glass in TRELLIS.2's finals, on a real GPU in a staging app (ORAINGE_APP_NAME; nothing is deployed).

TRELLIS.2 gives glass a low alpha, and the export divides colour by alpha (pipeline.unpremultiply) before it
writes an opaque material, so windows came out pale and opaque. This measures what the texture pass returns
before that division and checks the fix.

    ORAINGE_APP_NAME=orainge-p8-glass modal run ops/exp_glass.py::check --mode measure --only bmw,10,13,14,15,16

Objects are "bmw" (the founder's car: its picture drawn again by the reference worker from the prompt and
seed below, then TRELLIS.2 at the final's seed) and Phase 2 prompt numbers (their picked pictures and seeds,
from the runs phase2-NN-<slug> in orainge-outputs). Every model is made with production's code as the
Trellis2 worker class runs it (handle_with_models and handle_job: the final preset at the seed, exported and
packed); --textures lists the objects that also get a textures job (texture options, --count of them).

measure: what each export's to_glb returned is saved from before unpremultiply: the mesh (vertices, faces,
UVs, normals; npz), its RGBA base colour and metallic-roughness textures (PNG), the picture's full-frame
cutout the projection paints from and the voxel size, so the rest of the export can be replayed on a CPU.
The packed GLBs are production's models as they stand ("before").

verify: each object's final mesh is made once (Trellis2Runtime.generate, the final preset at its seed) and
exported twice, as production exports and packs it: with glass off (Trellis2Runtime.glass_textures = False:
the colour only divided by alpha, as before) into before/, and with glass (glass.py) into after/. Then --count
textures of the same shape for the objects in --textures, each exported both ways too: texture 1 with glass
first (to_glb in full, its layout kept), the others rebaked on that layout. Objects in --jobs also get a
textures job through production's handle_job (after-job/), and --preview objects a preview with glass.

Everything lands in --out (keep it under ops-out/private/: pictures and models stay private).
"""

from __future__ import annotations

import base64
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

# The founder's car: what the Studio asked the reference worker for, and the final's seed
BMW_PROMPT = "make a bmw car m3 model blue"
BMW_PICTURE_SEED = 1627471494
BMW_SEED = 663008479

glass_image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")


@app.cls(
    image=glass_image,
    gpu=prod.TRELLIS2_GPU,
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=3,
)
class Glass:
    @modal.enter()
    def load(self) -> None:
        import torch

        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import Trellis2Runtime
        from forge3d_worker.service import handle_job
        from forge3d_worker.storage import storage_from_env

        # What production's Trellis2.load does, with TRELLIS.2 making the finals (the default)
        prod._require_weights("trellis2")
        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        storage = storage_from_env()  # no R2 in a staging run: models come back inline
        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        pool = prod.ModelPool(self.runtime)
        handlers = {"trellis2": handle_job}
        self.handle = lambda job: prod.handle_with_models(job, pool, "trellis2", handlers, storage, pack_glb)
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[glass] TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def measure(self, job: dict) -> dict:
        """
        The final (and a textures job when ``job["textures"]``) made by production's handle_job, with every
        export's to_glb result saved before unpremultiply. Returns the files and a summary.
        """
        name = job["name"]
        runtime = self.runtime
        captures: list[dict] = []
        stock = type(runtime)._textured

        def textured(mesh, preset):
            glb = stock(runtime, mesh, preset)
            captures.append(snapshot(glb, mesh))  # before _export unpremultiplies it in place
            return glb

        files: dict[str, bytes] = {}
        summary: dict = {"name": name, "seed": job["seed"], "gpu": self.gpu, "load_seconds": self.load_seconds}
        base = {"image_base64": job["image_base64"], "seed": job["seed"], "request_id": f"p8-glass-{name}"}
        runtime._textured = textured
        try:
            clock = time.time()
            result = self.handle({"id": f"final-{name}", "input": {**base, "mode": "final"}})
            summary["final_s"] = round(time.time() - clock, 1)
            if result.get("error"):
                raise RuntimeError(f"the final failed: {result['error']}")
            files[f"{name}/final-{job['seed']}.glb"] = base64.b64decode(result["glb"]["base64"])
            summary["final"] = {k: result.get(k) for k in ("triangles", "pipeline", "projection", "timings", "model")}
            files.update(captured(captures, name, f"final-{job['seed']}"))
            summary["final"]["alpha"] = captures[-1]["stats"]
            print(f"[glass] {name} final: {json.dumps(summary['final'], default=str)}")
            count = int(job.get("textures") or 0)
            if count:
                captures.clear()
                clock = time.time()
                # As the Studio asks: a final that fell back to "512" has its shape made with it
                fell_back = {"pipeline": "512"} if result.get("pipeline") == "512" else {}
                textures_job = {**base, "mode": "textures", "count": count, **fell_back}
                result = self.handle({"id": f"textures-{name}", "input": textures_job})
                summary["textures_s"] = round(time.time() - clock, 1)
                if result.get("error"):
                    raise RuntimeError(f"the textures job failed: {result['error']}")
                summary["textures"] = []
                for texture, capture in zip(result.get("textures", []), captures):
                    label = f"final-{job['seed']}-texture-{texture['texture_seed']}"
                    files[f"{name}/{label}.glb"] = base64.b64decode(texture["glb"]["base64"])
                    files.update(captured([capture], name, label))
                    entry = {k: texture.get(k) for k in ("texture_seed", "triangles", "projection", "export")}
                    entry["alpha"] = capture["stats"]
                    summary["textures"].append(entry)
                    print(f"[glass] {name} texture: {json.dumps(entry, default=str)}")
                summary["texture_errors"] = result.get("texture_errors")
        finally:
            del runtime._textured
        return {"summary": summary, "files": files}


    @modal.method()
    def verify(self, job: dict) -> dict:
        """The final (and texture options) exported before and after glass, from the same meshes."""
        return verify_job(self, job)


def verify_job(worker, job: dict) -> dict:
    """The final (and texture options) exported before and after glass, from the same meshes (Glass.verify)."""
    import torch
    from PIL import Image

    from forge3d_worker.service import texture_seed
    from forge3d_worker.settings import PRESETS

    name, seed = job["name"], int(job["seed"])
    runtime = worker.runtime
    preset = PRESETS["final"]
    picture = Image.open(io.BytesIO(base64.b64decode(job["image_base64"])))
    picture.load()
    files: dict[str, bytes] = {}
    summary: dict = {"name": name, "seed": seed, "gpu": worker.gpu, "load_seconds": worker.load_seconds}
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()

    def both(mesh, label: str, glass_first: bool = False) -> dict:
        ways = (("after", True), ("before", False)) if glass_first else (("before", False), ("after", True))
        entry = {}
        for way, on in ways:
            made = export_once(worker, mesh, preset, on)
            files[f"{name}/{way}/{label}.glb"] = made.pop("packed")
            entry[way] = made
        print(f"[glass] {name} {label}: {json.dumps(entry, default=str)}")
        return entry

    clock = time.time()
    mesh = runtime.generate(picture, preset, seed)
    summary["generate_s"] = round(time.time() - clock, 1)
    summary["pipeline"] = runtime.pipeline_used
    summary["final"] = both(mesh, f"final-{seed}")
    del mesh
    summary["textures"] = []
    for number in range(1, int(job.get("textures") or 0) + 1):
        texture = texture_seed(seed, number)
        mesh = runtime.retexture(seed=texture)
        entry = both(mesh, f"final-{seed}-texture-{texture}", glass_first=number == 1)
        summary["textures"].append({"texture_seed": texture, **entry})
        del mesh
    if job.get("textures_job"):
        # Texture options as the Studio asks for them: production's handle_job, glass and all
        base = {"image_base64": job["image_base64"], "seed": seed, "request_id": f"p8-glass-{name}"}
        fell_back = {"pipeline": "512"} if summary["pipeline"] == "512" else {}
        clock = time.time()
        request = {**base, "mode": "textures", "count": int(job["textures_job"]), **fell_back}
        result = worker.handle({"id": f"textures-{name}", "input": request})
        summary["textures_job_s"] = round(time.time() - clock, 1)
        if result.get("error"):
            summary["textures_job"] = {"error": result["error"]}
        else:
            summary["textures_job"] = []
            for texture in result.get("textures", []):
                path = f"{name}/after-job/final-{seed}-texture-{texture['texture_seed']}.glb"
                files[path] = base64.b64decode(texture["glb"]["base64"])
                kept = ("texture_seed", "triangles", "glass", "export", "projection")
                summary["textures_job"].append({k: texture.get(k) for k in kept})
            summary["texture_errors"] = result.get("texture_errors")
        print(f"[glass] {name} textures job: {json.dumps(summary['textures_job'], default=str)}")
    if job.get("preview"):
        mesh = runtime.generate(picture, PRESETS["preview"], seed)
        made = export_once(worker, mesh, PRESETS["preview"], True)
        files[f"{name}/after/preview-{seed}.glb"] = made.pop("packed")
        summary["preview"] = made
        print(f"[glass] {name} preview: {json.dumps(made, default=str)}")
        del mesh
    if gpu:
        summary["peak_gpu_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
    return {"summary": summary, "files": files}


def export_once(worker, mesh, preset, glass: bool) -> dict:
    """One export of ``mesh`` with glass on or off, packed as production packs it."""
    from forge3d_worker.compress import pack_glb

    runtime = worker.runtime
    runtime.glass_textures = glass
    clock = time.time()
    try:
        raw, triangles = runtime.export(mesh, preset)
    finally:
        runtime.glass_textures = True
    export_s = round(time.time() - clock, 2)
    packed = pack_glb(raw, preset.texture_size)
    return {
        "packed": packed,
        "triangles": triangles,
        "export_s": export_s,
        "glass": runtime.last_glass,
        "path": (runtime.last_export or {}).get("path"),
        "projection": runtime.last_projection,  # already the projection's summary
        "bytes": len(packed),
    }


def snapshot(glb, mesh) -> dict:
    """What to_glb returned for ``mesh``, copied (the export changes its material in place afterwards)."""
    import numpy as np
    from PIL import Image

    from forge3d_worker.pipeline import CUTOUT

    material = glb.visual.material
    colour = material.baseColorTexture
    colour = colour.copy() if isinstance(colour, Image.Image) else Image.fromarray(np.asarray(colour))
    rough = material.metallicRoughnessTexture
    rough = rough.copy() if isinstance(rough, Image.Image) else Image.fromarray(np.asarray(rough))
    arrays = {
        "vertices": np.asarray(glb.vertices, dtype=np.float32),
        "faces": np.asarray(glb.faces, dtype=np.int32),
        "uv": np.asarray(glb.visual.uv, dtype=np.float32),
        "normals": np.asarray(glb.vertex_normals, dtype=np.float32),
    }
    cutout = getattr(mesh, CUTOUT, None)
    return {
        "arrays": arrays,
        "colour": colour,
        "rough": rough,
        "cutout": cutout.copy() if isinstance(cutout, Image.Image) else None,
        "meta": {
            "voxel_size": float(mesh.voxel_size),
            "double_sided": bool(getattr(material, "doubleSided", False)),
            "alpha_mode": getattr(material, "alphaMode", None),
            "colour_mode": colour.mode,
            "texture_size": list(colour.size),
        },
        "stats": alpha_stats(colour, rough),
    }


def alpha_stats(colour, rough) -> dict:
    """How much of the texture has alpha below 1, and how glossy those texels are (inpainted gutters included)."""
    import numpy as np

    rgba = np.asarray(colour.convert("RGBA"), dtype=np.float32) / 255
    alpha = rgba[..., 3]
    gb = np.asarray(rough.convert("RGB"), dtype=np.float32)[..., 1:3] / 255
    if gb.shape[:2] != alpha.shape:
        return {"note": "the textures differ in size"}
    roughness, metallic = gb[..., 0], gb[..., 1]
    out = {f"below_{edge}": round(float((alpha < edge).mean()), 4) for edge in (0.98, 0.75, 0.5, 0.3)}
    for label, where in (("opaque", alpha >= 0.98), ("low", alpha < 0.5)):
        if where.any():
            out[f"{label}_roughness"] = round(float(np.median(roughness[where])), 3)
            out[f"{label}_metallic"] = round(float(np.median(metallic[where])), 3)
            out[f"{label}_luma"] = round(float(np.median(rgba[..., :3][where] @ [0.2126, 0.7152, 0.0722])), 3)
    return out


def captured(captures: list[dict], name: str, label: str) -> dict[str, bytes]:
    """Files for one export's snapshot: mesh.npz, colour.png (RGBA), rough.png, cutout.png, meta.json."""
    import numpy as np

    files = {}
    for capture in captures:
        stem = f"{name}/raw/{label}"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **capture["arrays"])
        files[f"{stem}/mesh.npz"] = buffer.getvalue()
        for key in ("colour", "rough", "cutout"):
            image = capture[key]
            if image is None:
                continue
            buffer = io.BytesIO()
            image.save(buffer, "PNG", compress_level=6)
            files[f"{stem}/{key}.png"] = buffer.getvalue()
        files[f"{stem}/meta.json"] = json.dumps({**capture["meta"], "alpha": capture["stats"]}, indent=2).encode()
    return files


def read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the picked pictures)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


def bmw_picture() -> bytes:
    """The founder's picture drawn again: the reference worker, the Studio's prompt, its seed, one picture."""
    clock = time.time()
    job = {"prompt": BMW_PROMPT, "seed": BMW_PICTURE_SEED, "count": 1, "request_id": "p8-glass-bmw"}
    result = prod.FluxSchnell().generate.remote({"id": "p8-glass-bmw", "input": job})
    if result.get("error"):
        raise SystemExit(f"[glass] the reference worker failed: {result['error']}")
    image = result["images"][0]
    print(f"[glass] bmw picture: seed {image['seed']}, score {image.get('score')}, {round(time.time() - clock, 1)} s")
    return base64.b64decode(image["base64"])


def names(text: str) -> set[str]:
    """'bmw, 13' -> {'bmw', '13'}"""
    return {n.strip().lower().lstrip("0") or "0" for n in text.split(",") if n.strip()}


@app.local_entrypoint()
def check(
    mode: str = "measure",
    only: str = "bmw,10,13,14,15,16",
    textures: str = "bmw",
    count: int = 3,
    jobs: str = "",
    preview: str = "",
    out: str = "ops-out/private/glass",
) -> None:
    if mode not in ("measure", "verify"):
        raise SystemExit(f"[glass] unknown mode {mode!r}")
    print(f"[glass] app {prod.APP_NAME}: {mode} {only}; textures for {textures or 'none'} ({count} each)")
    root = pathlib.Path(out)
    wanted = [n.strip().lower() for n in only.split(",") if n.strip()]
    with_textures, with_jobs, with_preview = names(textures), names(jobs), names(preview)
    pictures = runs_by_number(prod.outputs, "phase2")
    work = []
    for item in wanted:
        if item == "bmw":
            picture, seed, name = bmw_picture(), BMW_SEED, "bmw"
        else:
            number = item.zfill(2)
            if number not in pictures:
                print(f"[glass] no Phase 2 run for {number}")
                continue
            name = pictures[number]
            state = json.loads(read(prod.outputs, f"{name}/progress.json"))
            picture, seed = read(prod.outputs, f"{name}/{state['input']}"), int(state["seed"])
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "picture.png").write_bytes(picture)
        key = item.lstrip("0") or "0"
        work.append({"name": name, "seed": seed, "image_base64": base64.b64encode(picture).decode(),
                     "textures": count if key in with_textures else 0,
                     "textures_job": count if key in with_jobs else 0,
                     "preview": key in with_preview})
    listed = ", ".join("{} (seed {})".format(job["name"], job["seed"]) for job in work)
    print(f"[glass] {len(work)} objects: {listed}")
    summaries, failed = [], 0
    method = Glass().measure if mode == "measure" else Glass().verify
    for job, made in zip(work, method.map(work, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[glass] {job['name']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"name": job["name"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for path, data in made["files"].items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[glass] done: {len(work) - failed} of {len(work)} objects")
    if not work or failed:
        raise SystemExit(1)
