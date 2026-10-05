"""
Phase 8's multi-view painter (workers/trellis2/forge3d_worker/paint.py) on a real GPU, in a staging app
(ORAINGE_APP_NAME; nothing is deployed).

For each object, the final as production makes it (the Trellis2 worker's handle_job with the final's preset at
the picture's seed), exported two ways from the same to_glb mesh:

- today: production's export as it is (unpremultiply, the picture's projection, smoothed normals, gltfpack);
- painted: views round the mesh repainted by an image editor from renders of its own texture (with the
  picture as a reference in some variants), checked against the render's outline, baked into the texture
  (paint.paint_views), then the same export from there (the picture's projection on top, smoothed normals,
  glass, gltfpack). The editors: FLUX.2 [klein] 4B (Apache-2.0; variants without a prefix) and
  Qwen-Image-Edit-2511 with the 8-step Lightning LoRA (both Apache-2.0; the "qie-" variants).

Three GPU classes in three images, so the TRELLIS.2 image isn't rebuilt: Shapes (TRELLIS.2 on an L40S) makes the
final and today's export (with see-through glass since Phase 8) and keeps the to_glb mesh, with to_glb's RGBA
base colour for the glass, in the orainge-outputs volume under p8-paint/<name>/ (new paths only), so the painters
can run again without remaking it; Painter (klein in bf16 on an L40S) and QwenPaint (Qwen-Image-Edit-2511 in
bf16 on an H100, everything resident: 58 GB of weights) paint and export. klein's weights go to the
orainge-models volume under p8-paint/, Qwen-Image-Edit-2511's are its folders Qwen-Image-Edit-2511 and
Qwen-Image-Edit-2511-Lightning (ops/p8_download.py). The BMW's picture is drawn again by the reference worker
(FLUX.1 [schnell]) at its prompt and seed.

    ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw
    ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw,04,06,08,12,13
    ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw --variants qie-edit-view,qie-ref
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
# Qwen-Image-Edit-2511 and its Lightning LoRA, as ops/p8_download.py put them in the models volume (the revisions
# workers/painter/scripts/download_weights.py pins)
QWEN_DIR = f"{prod.MODELS}/Qwen-Image-Edit-2511"
QWEN_LORA_DIR = f"{prod.MODELS}/Qwen-Image-Edit-2511-Lightning"

# The editing model's instructions. Run 1 (render, picture and the nearest painted view as three references, the
# PROMPT_V1 below) kept the picture's viewpoint instead of the render's in 9 views of 10, so these keep the
# picture out, make it small, or start from the render's own latent (img2img)
PROMPT_V1 = (
    "Repaint image 1 as a clean, photorealistic studio product photo of the {subject} shown in image 2, seen "
    "from exactly the same viewpoint as image 1. Keep the outline, size, position and proportions of every part "
    "exactly as they are in image 1: do not move, add, remove or reshape anything, and keep the background where "
    "it is. Give every part the true colours, materials, markings and fine details of the {subject} in image 2, "
    "sharp and clean, with no blotches, smears or streaks. Soft, even, diffused studio lighting from all around, "
    "with no cast shadows, no strong reflections and no bright highlights. Plain light grey background."
)
NEIGHBOUR = " Image 3 shows the same {subject} already repainted from another angle: match its colours and finish exactly."
EDIT = (
    "Turn this rough 3D render of a {subject} into a clean, photorealistic studio product photo of the same "
    "{subject}. Keep exactly the same camera angle, framing, outline, proportions and position of every part: do "
    "not move, add, remove or reshape anything. Replace the blotchy, smeared surface with clean, crisp, realistic "
    "materials and fine details in the same colours. Soft, even, diffused studio lighting from all around, with no "
    "cast shadows and no strong reflections. Plain light grey background."
)
EDIT_REF = (
    "Turn image 1, a rough 3D render of a {subject}, into a clean, photorealistic studio product photo of it, "
    "keeping exactly the camera angle, framing, outline and position of every part of image 1: do not move, add, "
    "remove or reshape anything. Image 2 is a photo of the same {subject} from a different angle: use it only for "
    "the true colours, materials and fine details, never for the viewpoint or the layout. Replace the blotchy, "
    "smeared surface with clean, crisp, realistic materials. Soft, even, diffused studio lighting from all around, "
    "with no cast shadows and no strong reflections. Plain light grey background."
)
# Run 2's "edit" held every outline and drew clean, photo-real views, but invented a screen and a door on the
# arcade machine's plain back. This one says which side the view shows (taking the picture as the front) and
# the picture's main colours, and asks to keep plain surfaces plain
EDIT_VIEW = (
    "Turn this rough 3D render of a {subject}, seen {side}, into a clean, photorealistic studio product photo of "
    "the same {subject} from exactly the same viewpoint. Keep the camera angle, framing, outline, proportions and "
    "position of every part exactly as they are: do not move, add, remove or reshape anything. Replace the "
    "blotchy, smeared surface with clean, crisp, realistic materials and fine details in the same colours{colours}. "
    "Where the render shows a plain surface, keep it plain: do not invent screens, buttons, doors, handles, text, "
    "logos or patterns that the render doesn't show. Soft, even, diffused studio lighting from all around, with "
    "no cast shadows and no strong reflections. Plain light grey background."
)
# The same asks for Qwen-Image-Edit-2511, which names its inputs "Picture 1", "Picture 2", ... in the prompt
QIE_EDIT_VIEW = (
    "Turn Picture 1, a rough 3D render of a {subject} seen {side}, into a clean, photorealistic studio product "
    "photo of the same {subject} from exactly the same viewpoint. Keep the camera angle, framing, outline, "
    "proportions and position of every part exactly as they are in Picture 1: do not move, add, remove or reshape "
    "anything. Replace the blotchy, smeared surface with clean, crisp, realistic materials and fine details in the "
    "same colours{colours}. Where Picture 1 shows a plain surface, keep it plain: do not invent screens, buttons, "
    "doors, handles, text, logos or patterns that Picture 1 doesn't show. Soft, even, diffused studio lighting from "
    "all around, with no cast shadows and no strong reflections. Plain light grey background."
)
QIE_REF = (
    "Turn Picture 1, a rough 3D render of a {subject} seen {side}, into a clean, photorealistic studio product "
    "photo of it from exactly the same viewpoint as Picture 1. Keep Picture 1's camera angle, framing, outline, "
    "proportions and the position of every part exactly: do not move, add, remove or reshape anything. Picture 2 "
    "is a photo of the same {subject} from another angle: use it only for the true colours, materials, logos and "
    "fine details of the parts both pictures show, never for the viewpoint or the layout. Replace the blotchy, "
    "smeared surface with clean, crisp, realistic materials{colours}. Where Picture 1 shows a plain surface that "
    "Picture 2 doesn't, keep it plain. Soft, even, diffused studio lighting from all around, with no cast shadows "
    "and no strong reflections. Plain light grey background."
)
QIE_NEIGHBOUR = (
    " Picture 3 shows the same {subject} already finished from a nearby angle: match its colours, materials and "
    "finish exactly."
)

# How each variant asks klein: "base" is Flux2KleinPipeline with the listed references (the render first);
# "img2img" is Flux2KleinInpaintPipeline with the whole frame as the mask, which starts from the render's latent
# noised to ``start`` (the render is also its first reference) with the picture as ``image_reference``. klein's
# own 4-step schedule at a megapixel runs 1.0, 0.97, 0.91, 0.77: ``start`` 0.91 is its last two steps; 0.8 is an
# own three-step schedule (0.8, 0.6, 0.35), shifted back through the scheduler's exponential time shift
STRATEGIES = {
    "v1": {"pipeline": "base", "picture": 768 * 768, "neighbour": True, "prompt": PROMPT_V1},
    "edit": {"pipeline": "base", "picture": 0, "prompt": EDIT},
    "edit-view": {"pipeline": "base", "picture": 0, "prompt": EDIT_VIEW},
    "edit-ref": {"pipeline": "base", "picture": 384 * 384, "prompt": EDIT_REF},
    "i2i-ref-91": {"pipeline": "img2img", "picture": 512 * 512, "start": 0.91, "prompt": EDIT_REF},
    "i2i-ref-80": {"pipeline": "img2img", "picture": 512 * 512, "start": 0.8, "prompt": EDIT_REF},
    "i2i-80": {"pipeline": "img2img", "picture": 0, "start": 0.8, "prompt": EDIT},
    # Qwen-Image-Edit-2511 (QwenPaint): the render alone; with the picture; with the picture and the nearest view
    "qie-edit-view": {"model": "qwen", "picture": 0, "prompt": QIE_EDIT_VIEW},
    # The same, baked into a 4096 base colour (the 2048 one upsampled first, same layout) and shipped at 4096
    "qie-edit-view-4k": {"model": "qwen", "picture": 0, "prompt": QIE_EDIT_VIEW, "texture": 4096},
    "qie-ref": {"model": "qwen", "picture": 768 * 768, "prompt": QIE_REF},
    "qie-ref-nb": {"model": "qwen", "picture": 768 * 768, "neighbour": True, "prompt": QIE_REF, "neighbour_text": QIE_NEIGHBOUR},
}
# klein's shifted 4-step schedule (1 MP), and the schedules the "start" values ask for
SCHEDULES = {0.91: None, 0.8: (1.0, 0.8, 0.6, 0.35)}


def unshifted(sigmas, mu: float) -> list:
    """The sigmas to give the scheduler so that its exponential time shift by ``mu`` turns them into ``sigmas``."""
    import math

    return [1.0 if s >= 1 else 1.0 / (1.0 + math.exp(mu) * (1.0 / s - 1.0)) for s in sigmas]


def shrink(image, pixels: int, multiple: int = 16):
    """``image`` scaled down to at most ``pixels`` in all, sides multiples of ``multiple``."""
    import math

    from PIL import Image

    scale = min(1.0, math.sqrt(pixels / (image.width * image.height)))
    width = max(multiple, int(image.width * scale) // multiple * multiple)
    height = max(multiple, int(image.height * scale) // multiple * multiple)
    return image if (width, height) == image.size else image.resize((width, height), Image.Resampling.LANCZOS)


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
qwen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    # workers/painter/requirements.txt's pins, plus trimesh for the mesh
    .pip_install(
        "diffusers==0.37.1",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
        "huggingface_hub==0.36.2",
        "accelerate==1.12.0",
        "peft==0.18.1",
        "safetensors==0.7.0",
        "pillow==12.1.1",
        "numpy<2.3",
        "trimesh==4.12.2",
    )
    .run_commands(
        "curl -fsSL -o /tmp/gltfpack.zip https://github.com/zeux/meshoptimizer/releases/download/"
        f"v{prod.GLTFPACK_VERSION}/gltfpack-ubuntu.zip && unzip /tmp/gltfpack.zip -d /usr/local/bin"
        " && chmod +x /usr/local/bin/gltfpack && rm /tmp/gltfpack.zip"
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(prod.WORKERS / "trellis2" / "forge3d_worker", "/root/forge3d_worker")
    .add_local_dir(prod.WORKERS / "painter" / "painter_worker", "/root/painter_worker")
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


def prompt_for(strategy: str, subject: str, neighbour: bool, view: dict | None = None) -> str:
    spec = STRATEGIES[strategy]
    view = view or {}
    names = list(view.get("colours") or [])
    colours = ""
    if names:
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        colours = f" (its main colours are {listed})"
    extra = spec.get("neighbour_text", NEIGHBOUR).format(subject=subject) if neighbour and spec.get("neighbour") else ""
    text = spec["prompt"].format(subject=subject, side=view.get("side") or "from the front", colours=colours)
    return text + extra


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
            # to_glb's RGBA base colour: its alpha (TRELLIS.2's opacity) is what glass.split_glass reads
            rgba = getattr(self.runtime, "last_rgba", None)
            if rgba is not None:
                captured["rgba"] = _png(rgba)
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
            "glass": result.get("glass"),
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
        if "rgba" in captured:
            (folder / "rgba.png").write_bytes(captured["rgba"])
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
        self._inpaint = None
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        self.calls = 0
        print(f"[painter] klein on {self.gpu} in {self.load_seconds} s")

    @property
    def inpaint(self):
        """img2img through klein's inpainting pipeline on the same weights, made when a variant first needs it."""
        if self._inpaint is None:
            from diffusers import Flux2KleinInpaintPipeline

            self._inpaint = Flux2KleinInpaintPipeline.from_pipe(self.pipe)
            self._inpaint.set_progress_bar_config(disable=True)
        return self._inpaint

    def _editor(self, subject: str, strategy: str, steps: int):
        import torch
        from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import compute_empirical_mu
        from PIL import Image

        spec = STRATEGIES[strategy]

        def paint(render, picture, neighbour, seed, view):
            generator = torch.Generator("cuda").manual_seed(int(seed))
            reference = shrink(picture, spec["picture"]) if spec.get("picture") else None
            use_neighbour = bool(spec.get("neighbour")) and neighbour is not None
            prompt = prompt_for(strategy, subject, use_neighbour, view)
            self.prompts.append({"view": view.get("name"), "seed": int(seed), "prompt": prompt})
            if spec["pipeline"] == "base":
                images = [render] + ([reference] if reference is not None else []) + ([neighbour] if use_neighbour else [])
                return self.pipe(
                    image=images,
                    prompt=prompt,
                    height=render.height,
                    width=render.width,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    generator=generator,
                ).images[0]
            # img2img: the whole frame repainted, from the render's latent noised to the start
            schedule = SCHEDULES[spec["start"]]
            options = {"num_inference_steps": 4, "strength": 0.5}
            if schedule is not None:
                mu = compute_empirical_mu((render.height // 16) * (render.width // 16), len(schedule))
                options = {"num_inference_steps": len(schedule), "strength": (len(schedule) - 1) / len(schedule),
                           "sigmas": unshifted(schedule, mu)}
            return self.inpaint(
                prompt=prompt,
                image=render,
                image_reference=reference,
                mask_image=Image.new("L", render.size, 255),
                guidance_scale=1.0,
                generator=generator,
                **options,
            ).images[0]

        return paint

    @modal.method()
    def paint(self, job: dict) -> dict:
        return paint_job(self, job, self._editor, 4)


def paint_job(owner, job: dict, make_editor, default_steps: int) -> dict:
    """
    One painting: the cached shape of ``job["name"]`` painted by ``make_editor(subject, variant, steps)``'s editor,
    then exported as production exports (the picture's projection on top, smoothed normals, glass, gltfpack),
    three ways (painted, robust, views only). ``owner`` is the painter class (gpu, load_seconds, calls, prompts).
    """
    import torch
    from PIL import Image

    from forge3d_worker import glass, paint, projection
    from forge3d_worker.compress import pack_glb
    from forge3d_worker.pipeline import shade

    name, variant = job["name"], job.get("variant", "edit")
    owner.calls += 1
    prod.outputs.reload()
    folder = pathlib.Path(prod.OUTPUTS) / CACHE_DIR / name
    meta = json.loads((folder / "meta.json").read_text())
    mesh = paint.unpack_mesh((folder / "mesh.npz").read_bytes())
    cutout = Image.open(io.BytesIO((folder / "cutout.png").read_bytes()))
    cutout.load()
    subject = subject_of(meta.get("prompt") or "")
    options = dict(job.get("options") or {})
    steps = int(options.pop("steps", default_steps))
    # The base colour the views go into: the final's own (2048), or upsampled first, the UV layout unchanged
    size = int(STRATEGIES[variant].get("texture") or TEXTURE_SIZE)
    base = mesh.visual.material.baseColorTexture
    if max(base.size) < size:
        mesh.visual.material.baseColorTexture = base.resize((size, size), Image.Resampling.LANCZOS)
    torch.cuda.reset_peak_memory_stats()
    owner.prompts = []
    clock = time.time()
    result = paint.paint_views(mesh, cutout, make_editor(subject, variant, steps), device="cuda", **options)
    paint_s = round(time.time() - clock, 1)

    rgba = None
    if (folder / "rgba.png").exists():
        rgba = Image.open(io.BytesIO((folder / "rgba.png").read_bytes()))
        rgba.load()
    glass_reports = []

    def export(texture, project: bool):
        """Production's export from the painted texture: the picture's projection on top, smoothed normals, glass, gltfpack."""
        started = time.time()
        copy = paint.unpack_mesh((folder / "mesh.npz").read_bytes())
        copy.visual.material.baseColorTexture = texture
        report = None
        if project:
            _, report = projection.project_picture(copy, cutout)
            torch.cuda.empty_cache()
        glb = shade(copy, meta["voxel_size"])
        if rgba is not None:
            glb, glass_report = glass.split_glass(glb, rgba, meta["voxel_size"])
            glass_reports.append(glass.summary(glass_report))
        packed = pack_glb(glb.export(file_type="glb"), size)
        return packed, report, copy.visual.material.baseColorTexture, round(time.time() - started, 1)

    packed, report, final_texture, export_s = export(result.texture, True)
    robust, robust_report, _, _ = export(result.robust or result.texture, True)
    unprojected, _, _, _ = export(result.texture, False)

    out = f"{name}/{variant}"
    files = {
        f"{out}/painted.glb": packed,
        f"{out}/robust.glb": robust,
        f"{out}/views-only.glb": unprojected,
        f"{out}/sheet.jpg": _jpeg(paint.sheet(result.views, 320), 88),
        f"{out}/texture-painted.jpg": _jpeg(result.texture, 90),
        f"{out}/texture-final.jpg": _jpeg(final_texture, 90),
        f"{out}/prompts.json": json.dumps(owner.prompts, indent=1).encode(),
        f"{name}/reference.jpg": _jpeg(paint.picture_reference(cutout), 92),
    }
    for number, view in enumerate(result.views):
        stem = f"{out}/views/{number:02d}-{view.camera.name}"
        files[f"{stem}-render.jpg"] = _jpeg(view.render)
        if view.painted is not None:
            files[f"{stem}-painted.jpg"] = _jpeg(view.painted)
    summary = {
        "name": name,
        "variant": variant,
        "strategy": {k: v for k, v in STRATEGIES[variant].items() if k not in ("prompt", "neighbour_text")},
        "prompt": prompt_for(variant, subject, False, {"side": "<side>", "colours": result.report.get("colours")}),
        "subject": subject,
        "options": {**options, "steps": steps, "texture_size": size},
        "paint": result.report,
        "projection": projection.summary(report),
        "projection_robust": projection.summary(robust_report),
        "projection_today": meta.get("projection"),
        "glass": glass_reports[0] if glass_reports else None,
        "seconds": {"paint_views": paint_s, "export": export_s, **result.report.get("timings", {})},
        "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu": owner.gpu,
        "load_s": owner.load_seconds,
        "call": owner.calls,
        "container_s": round(time.time() - STARTED, 1),
        "shape": meta,
    }
    print(f"[painter] {name} {variant}: {json.dumps({k: summary[k] for k in ('seconds', 'peak_gpu_gb', 'call')})}")
    return {"summary": summary, "files": files}


@app.cls(
    image=qwen_image,
    gpu="H100",
    cpu=8.0,
    memory=65536,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs},
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=60,
    max_containers=2,
)
class QwenPaint:
    """Qwen-Image-Edit-2511 with the 8-step Lightning LoRA (workers/painter), bf16 on an H100, all resident."""

    @modal.enter()
    def load(self) -> None:
        import torch

        from painter_worker.qwen import QwenPainter

        started = time.time()
        self.revision = _revision(QWEN_DIR)
        self.painter = QwenPainter(QWEN_DIR, QWEN_LORA_DIR)
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        self.calls = 0
        self.prompts = []
        print(
            f"[painter] Qwen-Image-Edit-2511@{self.revision} on {self.gpu} in {self.load_seconds} s, "
            f"{self.painter.lora_layers} LoRA layers, {self.painter.steps} steps"
        )

    def _editor(self, subject: str, strategy: str, steps: int):
        from PIL import Image

        from forge3d_worker import paint as painting
        from painter_worker.qwen import from_square

        spec = STRATEGIES[strategy]
        backdrop = tuple(int(round(255 * c)) for c in painting.BACKGROUND)

        def square(image):
            """On a square of the renders' own backdrop, centred as from_square() expects."""
            image = image.convert("RGB")
            side = max(image.size)
            if image.width == image.height:
                return image
            canvas = Image.new("RGB", (side, side), backdrop)
            canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            return canvas

        def paint(render, picture, neighbour, seed, view):
            reference = shrink(picture, spec["picture"]) if spec.get("picture") else None
            use_neighbour = bool(spec.get("neighbour")) and neighbour is not None
            prompt = prompt_for(strategy, subject, use_neighbour, view)
            images = [square(render)]
            if reference is not None:
                images.append(square(reference))
            if use_neighbour:
                images.append(square(neighbour))
            painted = self.painter.paint(images, prompt, seed=int(seed), steps=steps)
            self.prompts.append(
                {"view": view.get("name"), "seed": int(seed), "prompt": prompt, "pictures": len(images),
                 "seconds": self.painter.last_seconds, "peak_gb": self.painter.last_peak_gb}
            )
            return from_square(painted, render.size)

        return paint

    @modal.method()
    def paint(self, job: dict) -> dict:
        return paint_job(self, job, self._editor, self.painter.steps)


def _revision(folder: str) -> str:
    """The Hub commit snapshot_download recorded for a folder (its model_index.json's metadata), or "?"."""
    try:
        meta = pathlib.Path(folder) / ".cache" / "huggingface" / "download" / "model_index.json.metadata"
        return meta.read_text().split()[0][:12]
    except Exception:  # noqa: BLE001 - only for the log
        return "?"


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
    paint: str = "",
    variants: str = "edit",
    out: str = "ops-out/private/paint",
    fresh: bool = False,
    around: int = 8,
    elevation: float = 15.0,
    attempts: int = 2,
    min_iou: float = 0.9,
    max_novelty: float = 0.2,
    steps: int = 4,
    qwen_steps: int = 8,
    bottom: bool = True,
) -> None:
    """
    Makes (or reuses) the shapes of ``only`` (bmw and Phase 2 numbers), then paints those in ``paint`` (all of
    ``only`` when empty) once per variant in ``variants`` (STRATEGIES): klein's variants on Painter, the "qie-"
    ones on QwenPaint, both at once. ``steps`` is klein's, ``qwen_steps`` Qwen-Image-Edit-2511's (8 with the
    Lightning LoRA).
    """
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    chosen = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in STRATEGIES]
    if unknown or not chosen:
        raise SystemExit(f"[paint] variants must be among {', '.join(STRATEGIES)}, not {unknown or variants!r}")
    print(f"[paint] app {prod.APP_NAME}: shapes {only}; paint {paint or only}; variants {chosen}")
    klein = [v for v in chosen if STRATEGIES[v].get("model", "klein") == "klein"]
    weights = fetch_weights.spawn() if klein else None
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
    if weights is not None:
        info = weights.get()
        report["weights"] = info
        print(f"[paint] klein's weights: {json.dumps(info)}")

    # Today's export and the picture, from the volume (kept by Shapes, this run or an earlier one)
    ready = [job for job in jobs if "error" not in report["objects"].get(job["name"], {})]
    for job in ready:
        for file in ("today.glb", "picture.png"):
            _write(root, f"{job['name']}/{file}", _read(prod.outputs, f"{CACHE_DIR}/{job['name']}/{file}"))
    options = {
        "around": around, "elevation": elevation, "attempts": attempts, "min_iou": min_iou, "max_novelty": max_novelty,
        "steps": steps, "bottom": bottom,
    }
    wanted = {n.strip().lower().zfill(2) if n.strip().lower() != "bmw" else "bmw" for n in (paint or only).split(",") if n.strip()}
    paint_jobs = [
        {
            "name": job["name"],
            "variant": variant,
            "options": {**options, "steps": qwen_steps} if STRATEGIES[variant].get("model") == "qwen" else options,
        }
        for job in ready
        if job["name"] in wanted
        for variant in chosen
    ]
    print(f"[paint] painting {len(paint_jobs)}: {json.dumps(options)}, Qwen-Image-Edit-2511 at {qwen_steps} steps")

    # klein's and Qwen-Image-Edit-2511's paintings side by side, each on its own GPUs
    from concurrent.futures import ThreadPoolExecutor

    groups = {
        "klein": [job for job in paint_jobs if STRATEGIES[job["variant"]].get("model", "klein") == "klein"],
        "qwen": [job for job in paint_jobs if STRATEGIES[job["variant"]].get("model") == "qwen"],
    }
    runners = {"klein": lambda: Painter(), "qwen": lambda: QwenPaint()}

    def run(model: str) -> list:
        jobs_here = groups[model]
        if not jobs_here:
            return []
        return list(zip(jobs_here, runners[model]().paint.map(jobs_here, return_exceptions=True, order_outputs=True)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pair for result in pool.map(run, list(groups)) for pair in result]
    failed = 0
    for job, made in outcomes:
        entry = report["objects"].setdefault(job["name"], {}).setdefault("paint", {})
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            entry[job["variant"]] = {"error": f"{type(made).__name__}: {reason}"}
            print(f"[paint] {job['name']} {job['variant']}: failed: {entry[job['variant']]['error']}")
            continue
        for name, data in made["files"].items():
            _write(root, name, data)
        entry[job["variant"]] = made["summary"]
        views = made["summary"]["paint"]["views"]
        print(
            f"[paint] {job['name']} {job['variant']}: {made['summary']['paint']['accepted']} of {len(views)} views in, "
            f"IoU {[v['attempts'][-1]['iou'] for v in views]}, {made['summary']['seconds']}"
        )
    (root / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[paint] done: {len(paint_jobs) - failed} of {len(paint_jobs)} paintings")
    if not paint_jobs or failed:
        raise SystemExit(1)
