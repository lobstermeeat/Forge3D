"""
Puts the multiview worker's models under MODELS_ROOT (the orainge-models volume), pinned.

- huanngzh/mv-adapter: mvadapter_i2mv_sdxl.safetensors only, MV-Adapter's image-to-multiview
  adapter for SDXL (Apache-2.0)
- stabilityai/stable-diffusion-xl-base-1.0, the model it adapts (CreativeML Open RAIL++-M): the
  diffusers-format fp16 files only, about 7 GB of the repository's 70
- madebyollin/sdxl-vae-fp16-fix, SDXL's VAE fine-tuned to decode in fp16 without overflowing (MIT)
- ZhengPeng7/BiRefNet for background removal (MIT), shared with the TRELLIS.2 worker at the same
  revision (tests keep them equal), and fetched only if that worker's download hasn't put it there

Revisions are pinned so an upstream change can't silently alter the views.
"""

from __future__ import annotations

import os
import shutil

ROOT = os.environ.get("MODELS_ROOT", "/models")

MVADAPTER = ("huanngzh/mv-adapter", "6de4033df6b53366f3c009d22f5ec434bb55e59f")
ADAPTER_FILE = "mvadapter_i2mv_sdxl.safetensors"
SDXL = ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b")
VAE = ("madebyollin/sdxl-vae-fp16-fix", "207b116dae70ace3637169f1ddd2434b91b3a8cd")
BIREFNET = ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4")

# Folder names match multiview_worker/generator.py
ADAPTER_DIR = os.path.join(ROOT, "mv-adapter")
SDXL_DIR = os.path.join(ROOT, "stable-diffusion-xl-base-1.0")
VAE_DIR = os.path.join(ROOT, "sdxl-vae-fp16-fix")
BIREFNET_DIR = os.path.join(ROOT, "BiRefNet")

SDXL_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    # Replaced by the fp16-fix VAE at run time, but kept so the folder is a complete fp16 pipeline
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
VAE_FILES = ["config.json", "diffusion_pytorch_model.safetensors"]


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None  # none of these is gated; a token only raises rate limits
    hf_hub_download(MVADAPTER[0], ADAPTER_FILE, revision=MVADAPTER[1], local_dir=ADAPTER_DIR, token=token)
    snapshot_download(SDXL[0], revision=SDXL[1], local_dir=SDXL_DIR, allow_patterns=SDXL_FILES, token=token)
    snapshot_download(VAE[0], revision=VAE[1], local_dir=VAE_DIR, allow_patterns=VAE_FILES, token=token)
    if all(os.path.isfile(os.path.join(BIREFNET_DIR, name)) for name in ("config.json", "birefnet.py", "model.safetensors")):
        print("BiRefNet: already there (the TRELLIS.2 worker's download)")
    else:
        snapshot_download(BIREFNET[0], revision=BIREFNET[1], local_dir=BIREFNET_DIR, token=token)

    # huggingface_hub leaves download bookkeeping in .cache; nothing reads it at runtime
    for directory in (ADAPTER_DIR, SDXL_DIR, VAE_DIR, BIREFNET_DIR):
        cache = os.path.join(directory, ".cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
    print("Models ready in", ROOT)


if __name__ == "__main__":
    main()
