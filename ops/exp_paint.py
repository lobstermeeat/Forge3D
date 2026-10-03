"""
Phase 8's multi-view painter (workers/trellis2/forge3d_worker/paint.py) on a real GPU, in a staging app
(ORAINGE_APP_NAME; nothing is deployed).

For each object, the final as production makes it (the Trellis2 worker's handle_job with the final's preset at
the picture's seed), exported two ways from the same to_glb mesh:

- today: production's export as it is (unpremultiply, the picture's projection, smoothed normals, gltfpack);
- painted: views round the mesh repainted by FLUX.2 [klein] 4B (Apache-2.0) from renders of its own texture
  with the picture as the reference, checked against the render's outline, baked into the texture
  (paint.paint_views), then the same export from there (the picture's projection on top, smoothed normals,
  gltfpack).

Two GPU classes in two images, so the TRELLIS.2 image isn't rebuilt: Shapes (TRELLIS.2 on an L40S) makes the
final and today's export and keeps the to_glb mesh in the orainge-outputs volume under p8-paint/<name>/ (new
paths only), so the painter can run again without remaking it; Painter (klein in bf16 on an L40S) paints and
exports. klein's weights go to the orainge-models volume under p8-paint/ (a new folder). The BMW's picture is
drawn again by the reference worker (FLUX.1 [schnell]) at its prompt and seed.

    ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw
    ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13
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
# Here: the checkout's workers/. In the container: /root, where the images put modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app

# FLUX.2 [klein] 4B, Apache-2.0 (weights, Qwen3-4B text encoder and VAE all in the one repository), pinned
KLEIN = ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294")
KLEIN_DIR = f"{prod.MODELS}/p8-paint/FLUX.2-klein-4B-{KLEIN[1][:7]}"
# Where Shapes keeps each object's to_glb mesh for the painter (new paths in the shared outputs volume)
CACHE_DIR = "p8-paint"
# The founder's BMW: the prompt, the picked picture's FLUX.1 [schnell] seed and the final's TRELLIS.2 seed
BMW = {"name": "bmw", "prompt": "make a bmw car m3 model blue", "picture_seed": 1627471494, "seed": 663008479}
TEXTURE_SIZE = 2048  # the final preset's

# The editing model's instructions. Image 1 is the render to repaint, image 2 the picture, image 3 (when there
# is one) the painted view nearest to this one
PROMPT = (
    "Repaint image 1 as a clean, photorealistic studio product photo of the {subject} shown in image 2, seen "
    "from exactly the same viewpoint as image 1. Keep the outline, size, position and proportions of every part "
    "exactly as they are in image 1: do not move, add, remove or reshape anything, and keep the background where "
    "it is. Give every part the true colours, materials, markings and fine details of the {subject} in image 2, "
    "sharp and clean, with no blotches, smears or streaks. Soft, even, diffused studio lighting from all around, "
    "with no cast shadows, no strong reflections and no bright highlights. Plain light grey background."
)
NEIGHBOUR = " Image 3 shows the same {subject} already repainted from another angle: match its colours and finish exactly."

painter_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "diffusers==0.40.0",
        "transformers==5.18.0",
        "accelerate==1.15.0",
        "huggingface_hub==1.33.0",
        "safetensors==0.8.0",
        "trimesh==4.12.2",
        "pillow==12.3.0",
        "numpy>=2.0,<2.4",
    )
    # gltfpack, exactly as the Trellis2 image has it, for the same packing
    .run_commands(
        "curl -fsSL -o /tmp/gltfpack.zip https://github.com/zeux/meshoptimizer/releases/download/"
        f"v{prod.GLTFPACK_VERSION}/gltfpack-ubuntu.zip && unzip /tmp/gltfpack.zip -d /usr/local/bin"
        " && chmod +x /usr/local/bin/gltfpack && rm /tmp/gltfpack.zip"
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(prod.WORKERS / "trellis2" / "forge3d_worker", "/root/forge3d_worker")
    .add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")
)
shapes_image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")
weights_image = prod.download_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")

STARTED = time.time()


def subject_of(prompt: str) -> str:
    """What the object is, from its prompt: "make a bmw car m3 model blue" -> "bmw car m3 model blue"."""
    text = " ".join((prompt or "").lower().split())
    text = re.sub(r"^(please\s+)?(make|create|generate|draw|build|give me|i want)\s+(me\s+)?", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s*\b(3d\s+)?model\s+of\s+(a|an|the)\s+", " ", text).strip()
    return text or "object"


def prompt_for(subject: str, neighbour: bool) -> str:
    return PROMPT.format(subject=subject) + (NEIGHBOUR.format(subject=subject) if neighbour else "")


def _jpeg(image, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@app.function(
    image=weights_image,
    volumes={prod.MODELS: prod.models},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    timeout=3600,
)
def fetch_weights() -> dict:
    """klein's weights into a new folder of the models volume (once), with what its licence files say."""
    folder = pathlib.Path(KLEIN_DIR)
    marker = folder / ".complete.json"
    if marker.exists():
        return {**json.loads(marker.read_text()), "cached": True}
    from huggingface_hub import snapshot_download

    started = time.time()
    snapshot_download(
        KLEIN[0],
        revision=KLEIN[1],
        local_dir=str(folder),
        token=os.environ.get("HF_TOKEN") or None,
        # diffusers' layout only: not the single-file copy of the transformer, nor the sample images
        ignore_patterns=["flux-2-klein-4b.safetensors", "*.jpg", "*.png"],
    )
    files = {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob("*") if p.is_file() and ".cache" not in p.parts}
    licence = (folder / "LICENSE.md").read_text(errors="replace") if (folder / "LICENSE.md").exists() else ""
    readme = (folder / "README.md").read_text(errors="replace") if (folder / "README.md").exists() else ""
    info = {
        "repo": KLEIN[0],
        "revision": KLEIN[1],
        "bytes": sum(files.values()),
        "files": len(files),
        "license_md_head": " ".join(licence.split()[:12]),
        "readme_license": [line.strip() for line in readme.splitlines() if "license" in line.lower()][:8],
        "seconds": round(time.time() - started, 1),
    }
    marker.write_text(json.dumps(info))
    prod.models.commit()
    return info


class _Memory:
    """Storage for handle_job that keeps what it is given."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        self.data[key] = data
        return {"key": key, "bytes": len(data)}


@app.cls(
    image=shapes_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=1800,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=3,
)
class Shapes:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        self.calls = 0
        print(f"[shapes] TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def make(self, job: dict) -> dict:
        """The final as production makes it, today's export of it, and its to_glb mesh kept for the painter."""
        import torch

        from forge3d_worker import paint
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import CUTOUT
        from forge3d_worker.service import handle_job

        name, seed, picture = job["name"], int(job["seed"]), job["picture"]
        self.calls += 1
        captured: dict = {}

        def keep(glb, mesh) -> dict:
            # After unpremultiply (and the floaters, if the preset dropped them), before the projection
            started = time.time()
            captured["mesh"] = paint.pack_mesh(glb)
            captured["voxel_size"] = float(mesh.voxel_size)
            cutout = getattr(mesh, CUTOUT, None)
            if cutout is not None:
                captured["cutout"] = _png(cutout)
            captured["seconds"] = round(time.time() - started, 2)
            return {"kept": True}

        self.runtime.before_projection = keep
        torch.cuda.reset_peak_memory_stats()
        storage = _Memory()
        clock = time.time()
        try:
            result = handle_job(
                {
                    "id": f"p8-paint-{name}",
                    "input": {
                        "image_base64": base64.b64encode(picture).decode("ascii"),
                        "mode": "final",
                        "seed": seed,
                        "request_id": f"p8-paint-{name}",
                    },
                },
                self.runtime,
                storage,
                pack_glb,
            )
        finally:
            self.runtime.before_projection = None
        seconds = round(time.time() - clock, 1)
        if result.get("error"):
            return {"name": name, "error": result["error"]}
        if "mesh" not in captured or "cutout" not in captured:
            return {"name": name, "error": "the to_glb mesh or the cutout wasn't captured"}
        today = storage.data[result["glb"]["key"]]
        meta = {
            "name": name,
            "seed": seed,
            "prompt": job.get("prompt"),
            "source": job.get("source"),
            "voxel_size": captured["voxel_size"],
            "triangles": result.get("triangles"),
            "pipeline": result.get("pipeline"),
            "timings": result.get("timings"),
            "projection": result.get("projection"),
            "seconds": seconds,
            "keep_s": captured["seconds"],
            "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
            "gpu": self.gpu,
            "load_s": self.load_seconds,
            "call": self.calls,
        }
        folder = pathlib.Path(prod.OUTPUTS) / CACHE_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "mesh.npz").write_bytes(captured["mesh"])
        (folder / "cutout.png").write_bytes(captured["cutout"])
        (folder / "picture.png").write_bytes(picture)
        (folder / "today.glb").write_bytes(today)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        prod.outputs.commit()
        prod.share_caches()
        print(f"[shapes] {name}: {json.dumps(meta, default=str)}")
        return {"name": name, "meta": meta}


@app.cls(
    image=painter_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs},
    timeout=2400,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=3,
)
class Painter:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        started = time.time()
        self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_DIR, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        self.calls = 0
        print(f"[painter] klein on {self.gpu} in {self.load_seconds} s")

    def _editor(self, subject: str, steps: int):
        import torch

        def paint(render, picture, neighbour, seed):
            images = [render, picture] + ([neighbour] if neighbour is not None else [])
            generator = torch.Generator("cuda").manual_seed(int(seed))
            return self.pipe(
                image=images,
                prompt=prompt_for(subject, neighbour is not None),
                height=render.height,
                width=render.width,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=generator,
            ).images[0]

        return paint

    @modal.method()
    def paint(self, job: dict) -> dict:
        import torch
        from PIL import Image

        from forge3d_worker import paint, projection
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import shade

        name = job["name"]
        self.calls += 1
        prod.outputs.reload()
        folder = pathlib.Path(prod.OUTPUTS) / CACHE_DIR / name
        meta = json.loads((folder / "meta.json").read_text())
        mesh = paint.unpack_mesh((folder / "mesh.npz").read_bytes())
        cutout = Image.open(io.BytesIO((folder / "cutout.png").read_bytes()))
        cutout.load()
        subject = subject_of(meta.get("prompt") or "")
        options = dict(job.get("options") or {})
        steps = int(options.pop("steps", 4))
        torch.cuda.reset_peak_memory_stats()
        clock = time.time()
        result = paint.paint_views(mesh, cutout, self._editor(subject, steps), device="cuda", **options)
        paint_s = round(time.time() - clock, 1)

        # Production's export from here: the picture's projection on top, smoothed normals, gltfpack
        clock = time.time()
        mesh.visual.material.baseColorTexture = result.texture
        _, report = projection.project_picture(mesh, cutout)
        torch.cuda.empty_cache()
        glb = shade(mesh, meta["voxel_size"])
        raw = glb.export(file_type="glb")
        packed = pack_glb(raw, TEXTURE_SIZE)
        export_s = round(time.time() - clock, 1)

        files = {
            f"{name}/painted.glb": packed,
            f"{name}/sheet.jpg": _jpeg(paint.sheet(result.views, 320), 88),
            f"{name}/texture-painted.jpg": _jpeg(result.texture, 90),
            f"{name}/texture-final.jpg": _jpeg(mesh.visual.material.baseColorTexture, 90),
            f"{name}/reference.jpg": _jpeg(paint.picture_reference(cutout), 92),
        }
        for number, view in enumerate(result.views):
            stem = f"{name}/views/{number:02d}-{view.camera.name}"
            files[f"{stem}-render.jpg"] = _jpeg(view.render)
            if view.painted is not None:
                files[f"{stem}-painted.jpg"] = _jpeg(view.painted)
        summary = {
            "name": name,
            "subject": subject,
            "options": {**options, "steps": steps},
            "paint": result.report,
            "projection": projection.summary(report),
            "projection_today": meta.get("projection"),
            "seconds": {"paint_views": paint_s, "export": export_s, **result.report.get("timings", {})},
            "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "gpu": self.gpu,
            "load_s": self.load_seconds,
            "call": self.calls,
            "container_s": round(time.time() - STARTED, 1),
            "shape": meta,
        }
        print(f"[painter] {name}: {json.dumps({k: summary[k] for k in ('seconds', 'peak_gpu_gb', 'call')})}")
        return {"summary": summary, "files": files}


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the picked pictures)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


def _read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _cached(name: str) -> bool:
    try:
        return any(entry.path.endswith("meta.json") for entry in prod.outputs.listdir(f"{CACHE_DIR}/{name}"))
    except Exception:  # noqa: BLE001 - not there yet
        return False


def _write(root: pathlib.Path, name: str, data: bytes) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


@app.local_entrypoint()
def check(
    only: str = "bmw",
    out: str = "ops-out/private/paint",
    fresh: bool = False,
    around: int = 8,
    elevation: float = 15.0,
    attempts: int = 2,
    min_iou: float = 0.9,
    steps: int = 4,
    bottom: bool = True,
) -> None:
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[paint] app {prod.APP_NAME}: {only}")
    weights = fetch_weights.spawn()
    names = [n.strip().lower() for n in only.split(",") if n.strip()]
    pictures = runs_by_number(prod.outputs, "phase2")
    jobs, report = [], {"objects": {}}
    for name in names:
        if name == "bmw":
            job = {"name": "bmw", "seed": BMW["seed"], "prompt": BMW["prompt"], "source": "founder"}
        else:
            number = name.zfill(2)
            if number not in pictures:
                print(f"[paint] no Phase 2 picture for {name}")
                continue
            source = pictures[number]
            state = json.loads(_read(prod.outputs, f"{source}/progress.json"))
            job = {"name": number, "seed": int(state["seed"]), "prompt": state.get("prompt"), "source": source}
        job["cached"] = _cached(job["name"]) and not fresh
        jobs.append(job)

    # The shapes the painter needs that aren't kept yet
    todo = [job for job in jobs if not job["cached"]]
    for job in todo:
        if job["name"] == "bmw":
            clock = time.time()
            drawn = prod.FluxSchnell().generate.remote(
                {
                    "id": "p8-paint-bmw-picture",
                    "input": {"prompt": BMW["prompt"], "count": 1, "seed": BMW["picture_seed"], "request_id": "p8-paint-bmw"},
                }
            )
            if drawn.get("error"):
                raise SystemExit(f"[paint] the BMW's picture: {drawn['error']}")
            image = drawn["images"][0]
            job["picture"] = base64.b64decode(image["base64"])
            report["bmw_picture"] = {
                "seed": image.get("seed"),
                "score": image.get("score"),
                "issues": image.get("issues"),
                "seconds": round(time.time() - clock, 1),
                "gpu_seconds": drawn.get("seconds"),
                "prompt": drawn.get("prompt"),
            }
            _write(root, "bmw/picture-redrawn.png", job["picture"])
        else:
            state = json.loads(_read(prod.outputs, f"{job['source']}/progress.json"))
            job["picture"] = _read(prod.outputs, f"{job['source']}/{state['input']}")
    if todo:
        print(f"[paint] making {len(todo)} shapes: {', '.join(job['name'] for job in todo)}")
        for job, made in zip(todo, Shapes().make.map(todo, return_exceptions=True, order_outputs=True)):
            if isinstance(made, BaseException) or made.get("error"):
                reason = made.get("error") if isinstance(made, dict) else f"{type(made).__name__}: {made}"
                print(f"[paint] {job['name']}: shape failed: {reason}")
                report["objects"][job["name"]] = {"error": f"shape: {reason}"}
                continue
            report["objects"][job["name"]] = {"shape": made["meta"]}
    info = weights.get()
    report["weights"] = info
    print(f"[paint] weights: {json.dumps(info)}")

    # Today's export and the picture, from the volume (kept by Shapes, this run or an earlier one)
    ready = [job for job in jobs if "error" not in report["objects"].get(job["name"], {})]
    for job in ready:
        for file in ("today.glb", "picture.png"):
            _write(root, f"{job['name']}/{file}", _read(prod.outputs, f"{CACHE_DIR}/{job['name']}/{file}"))
    options = {"around": around, "elevation": elevation, "attempts": attempts, "min_iou": min_iou, "steps": steps, "bottom": bottom}
    paint_jobs = [{"name": job["name"], "options": options} for job in ready]
    print(f"[paint] painting {len(paint_jobs)}: {json.dumps(options)}")
    failed = 0
    for job, made in zip(paint_jobs, Painter().paint.map(paint_jobs, return_exceptions=True, order_outputs=True)):
        entry = report["objects"].setdefault(job["name"], {})
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            entry["error"] = f"paint: {type(made).__name__}: {reason}"
            print(f"[paint] {job['name']}: failed: {entry['error']}")
            continue
        for name, data in made["files"].items():
            _write(root, name, data)
        entry["paint"] = made["summary"]
        views = made["summary"]["paint"]["views"]
        print(
            f"[paint] {job['name']}: {made['summary']['paint']['accepted']} of {len(views)} views in, "
            f"IoU {[v['attempts'][-1]['iou'] for v in views]}, {made['summary']['seconds']}"
        )
    (root / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[paint] done: {len(paint_jobs) - failed} of {len(jobs)} objects painted")
    if not paint_jobs or failed:
        raise SystemExit(1)
