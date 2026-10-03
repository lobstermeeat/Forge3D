"""
FLUX.2 [klein] 4B against FLUX.1 [schnell] for the reference pictures, on a real GPU, in a staging app of its
own (ORAINGE_APP_NAME + "-klein"; nothing is deployed and production's app isn't run). It draws
ops/exp_pictures.py's prompts, seeds and templates with klein instead of schnell, scored the same way
(production's score_picture), with each picture's seconds and the GPU's peak memory, for comparison only.

FLUX.2 [klein] 4B is Apache-2.0 (its model card's metadata and LICENSE.md at KLEIN's revision; the [klein] 9B
weights and FLUX.2 [dev] are non-commercial and are never fetched). Its diffusers weights go to a folder of their
own in the orainge-models volume, KLEIN_DIR, on first use (download_klein, CPU only); nothing production reads
is touched. It runs with diffusers' Flux2KleinPipeline at its distilled settings (4 steps, no guidance) at
production's 1024 x 1024 on the same GPU type as schnell.

    ORAINGE_APP_NAME=orainge-p8-pics modal run ops/exp_klein.py::check --plan "a=v0,v11"
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import modal

if modal.is_local() and os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent) if (HERE.parent / "exp_pictures.py").is_file() else "/root")
import exp_pictures as pictures  # noqa: E402 - the prompts, seeds, templates and drawing loop
from exp_pictures import prod  # noqa: E402

app = modal.App(f"{prod.APP_NAME}-klein")

KLEIN = ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294")
KLEIN_DIR = f"{prod.MODELS}/FLUX.2-klein-4B"
KLEIN_MARKER = f"{prod.MODELS}/.flux2-klein-4b-weights"

# What the container imports: production's modal_app and scoring, and exp_pictures beside this file
local_code = [
    (prod.WORKERS / "modal_app.py", "/root/modal_app.py"),
    (HERE.parent / "exp_pictures.py", "/root/exp_pictures.py"),
]
# diffusers has Flux2KleinPipeline since 0.37; 0.39 is the last that allows transformers 4.x's huggingface_hub
klein_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(prod.TORCH[0], index_url=prod.TORCH_INDEX)
    .pip_install(
        "diffusers==0.39.0",
        "transformers==4.56.2",
        "huggingface_hub==0.35.3",
        "accelerate==1.10.1",
        "safetensors==0.6.2",
        "pillow>=10.3",
        "numpy>=1.26",
    )
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(prod.WORKERS / "flux-schnell" / "reference_worker", "/root/reference_worker")
)
download_image = prod.download_image
for source, target in local_code:
    klein_image = klein_image.add_local_file(source, target)
    download_image = download_image.add_local_file(source, target)


@app.function(
    image=download_image,
    volumes={prod.MODELS: prod.models},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    timeout=3600,
)
def download_klein() -> str:
    """FLUX.2 [klein] 4B's diffusers weights at KLEIN's revision into KLEIN_DIR, once (CPU only)."""
    from huggingface_hub import snapshot_download

    marker = pathlib.Path(KLEIN_MARKER)
    if marker.exists() and marker.read_text() == KLEIN[1]:
        return "already there"
    marker.unlink(missing_ok=True)
    snapshot_download(
        KLEIN[0],
        revision=KLEIN[1],
        local_dir=KLEIN_DIR,
        token=os.environ.get("HF_TOKEN") or None,
        # The diffusers layout only: not the 7.8 GB single-file copy of the transformer, nor the sample images
        ignore_patterns=["flux-2-klein-4b.safetensors", "*.jpg"],
    )
    license_text = (pathlib.Path(KLEIN_DIR) / "LICENSE.md").read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("FLUX.2 [klein] 4B's LICENSE.md is not the Apache License 2.0 at this revision")
    marker.write_text(KLEIN[1])
    prod.models.commit()
    return "downloaded"


@app.cls(
    image=klein_image,
    gpu=prod.FLUX_GPU,
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=2,
)
class Klein:
    @modal.enter()
    def load(self) -> None:
        import diffusers
        import torch
        from diffusers import Flux2KleinPipeline

        prod.models.reload()
        marker = pathlib.Path(KLEIN_MARKER)
        if not marker.exists() or marker.read_text() != KLEIN[1]:
            raise RuntimeError("FLUX.2 [klein] 4B's weights are missing: download_klein first")
        clock = time.time()
        pipe = Flux2KleinPipeline.from_pretrained(KLEIN_DIR, torch_dtype=torch.bfloat16).to("cuda")
        self.load_seconds = round(time.time() - clock, 1)
        self.gpu = torch.cuda.get_device_name()
        version = diffusers.__version__
        print(f"[pictures] FLUX.2 [klein] 4B (diffusers {version}) on {self.gpu} in {self.load_seconds} s")

        def generate(prompt: str, seed: int):
            # Its distilled settings (model card): 4 steps, guidance 1 (none); production's size and seeding
            return pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]

        self.generate = generate

    @modal.method()
    def draw(self, job: dict) -> dict:
        return pictures.draw_with(self.generate, job, self.gpu, self.load_seconds)


@app.local_entrypoint()
def check(plan: str = "a=v0", only: str = "", out: str = "ops-out/private/klein") -> None:
    """--plan "a=v0,v11": klein's pictures for those seed sets and templates (see exp_pictures.py)."""
    print(f"[pictures] FLUX.2 [klein] 4B at {KLEIN[1]}: {download_klein.remote()}")
    pictures.run_plan(plan, only, out, Klein(), "klein")
