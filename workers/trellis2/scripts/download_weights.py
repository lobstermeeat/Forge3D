"""
Bakes every model the worker needs into the image (run during `docker build`).

- microsoft/TRELLIS.2-4B plus the sparse-structure decoder it borrows from
  microsoft/TRELLIS-image-large (both MIT)
- facebook/dinov3-vitl16-pretrain-lvd1689m, the image encoder (DINOv3 License,
  gated: the Hugging Face token must belong to an account Meta approved)
- ZhengPeng7/BiRefNet for background removal (MIT). The stock TRELLIS.2 config
  uses briaai/RMBG-2.0, which is CC BY-NC 4.0 (no commercial use), so it is
  never downloaded; pipeline.json is rewritten to point at local copies instead.

Revisions are pinned so an upstream change can't silently alter production output.
"""

from __future__ import annotations

import json
import os
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.environ.get("MODELS_ROOT", "/models")
TRELLIS_DIR = os.path.join(ROOT, "TRELLIS.2-4B")
DINO_DIR = os.path.join(ROOT, "dinov3-vitl16")
BIREFNET_DIR = os.path.join(ROOT, "BiRefNet")

TRELLIS2 = ("microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d")
TRELLIS1 = ("microsoft/TRELLIS-image-large", "25e0d31ffbebe4b5a97464dd851910efc3002d96")
DINOV3 = ("facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51")
BIREFNET = ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4")

SS_DECODER = "ckpts/ss_dec_conv3d_16l8_fp16"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        raise SystemExit("HF_TOKEN is required: DINOv3 is gated (request access on its model page)")
    snapshot_download(TRELLIS2[0], revision=TRELLIS2[1], local_dir=TRELLIS_DIR)
    for ext in ("json", "safetensors"):
        hf_hub_download(TRELLIS1[0], f"{SS_DECODER}.{ext}", revision=TRELLIS1[1], local_dir=TRELLIS_DIR)
    snapshot_download(DINOV3[0], revision=DINOV3[1], local_dir=DINO_DIR, token=token)
    snapshot_download(BIREFNET[0], revision=BIREFNET[1], local_dir=BIREFNET_DIR)

    config_path = os.path.join(TRELLIS_DIR, "pipeline.json")
    with open(config_path) as f:
        config = json.load(f)
    args = config["args"]
    args["models"]["sparse_structure_decoder"] = SS_DECODER
    args["image_cond_model"]["args"]["model_name"] = DINO_DIR
    args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": BIREFNET_DIR}}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # huggingface_hub leaves download bookkeeping in .cache; nothing reads it at runtime
    for directory in (TRELLIS_DIR, DINO_DIR, BIREFNET_DIR):
        cache = os.path.join(directory, ".cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
    print("Models ready in", ROOT)


if __name__ == "__main__":
    main()
