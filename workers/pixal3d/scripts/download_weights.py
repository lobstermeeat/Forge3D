"""
Puts the Pixal3D worker's weights in the models volume, every file pinned.

- TencentARC/Pixal3D (MIT since 2026-05-21; revision of 2026-08-31, which added the multi-view set):
  the single-view and multi-view flow models (four each, about 5.5 GB apiece) and their configs.
  Its decoders are byte-identical to TRELLIS.2's (the same LFS sha256; the shape and texture
  decoders' configs are the same blobs, the sparse-structure decoder's differs by one whitespace
  byte), so the ones the TRELLIS.2 worker downloaded are used where they are: the pipeline configs
  are rewritten to point at them.
- Ruicheng/moge-2-vitl (MIT): MoGe-2, the single picture's camera.
- NAF (valeoai, Apache-2.0): its 2.6 MB upsampler weights, from the release the code fetches.
- DINOv3 ViT-L/16 and BiRefNet: the TRELLIS.2 worker's copies (facebook/dinov3-vitl16-pretrain-lvd1689m
  and ZhengPeng7/BiRefNet). Pixal3D names a DINOv3 mirror (camenduru/...) whose model.safetensors has
  the sha256 checked below, and RMBG-2.0 (CC BY-NC 4.0), which is never downloaded.

Run with MODELS_ROOT (default /models) and HF_TOKEN (optional: nothing here is gated). WHICH picks the
flow models: "multiview", "single" or "all" (default).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request

ROOT = os.environ.get("MODELS_ROOT", "/models")
PIXAL3D_DIR = os.path.join(ROOT, "pixal3d")
TRELLIS_DIR = os.path.join(ROOT, "TRELLIS.2-4B")
DINO_DIR = os.path.join(ROOT, "dinov3-vitl16")
BIREFNET_DIR = os.path.join(ROOT, "BiRefNet")

PIXAL3D = ("TencentARC/Pixal3D", "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f")
MOGE = ("Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968")
NAF_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
# camenduru/dinov3-vitl16-pretrain-lvd1689m's model.safetensors, the copy Pixal3D was trained with
DINOV3_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"

FLOW_MODELS = {
    "single": (
        "ckpts/ss_flow_img_dit_1_3B_64_bf16",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    ),
    "multiview": (
        "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv",
    ),
}
CONFIGS = {"single": "pipeline.json", "multiview": "pipeline_mv.json"}
# Pixal3D's decoders, with their sha256 at the pinned revision, and where the TRELLIS.2 worker keeps
# the same files (its download script puts TRELLIS-image-large's sparse-structure decoder in TRELLIS_DIR)
DECODERS = {
    "ckpts/ss_dec_conv3d_16l8_fp16": "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
    "ckpts/shape_dec_next_dc_f16c32_fp16": "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
    "ckpts/tex_dec_next_dc_f16c32_fp16": "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
}
NOTICES = ("LICENSE", "NOTICE", "README.md")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decoder_path(name: str, expected: str) -> str:
    """
    Where a decoder's checkpoint is used from (without extension): the TRELLIS.2 worker's identical
    copy if it is there, else Pixal3D's own, downloaded now.
    """
    shared = os.path.join(TRELLIS_DIR, name)
    if os.path.exists(f"{shared}.safetensors") and sha256(f"{shared}.safetensors") == expected:
        print(f"{name}: the TRELLIS.2 worker's copy (sha256 matches)")
        return shared
    print(f"{name}: no identical copy in {TRELLIS_DIR}, downloading Pixal3D's")
    from huggingface_hub import hf_hub_download

    for ext in ("json", "safetensors"):
        hf_hub_download(PIXAL3D[0], f"{name}.{ext}", revision=PIXAL3D[1], local_dir=PIXAL3D_DIR)
    local = os.path.join(PIXAL3D_DIR, name)
    if sha256(f"{local}.safetensors") != expected:
        raise SystemExit(f"{name}.safetensors does not have the pinned sha256")
    return local


def rewrite_config(config_file: str, decoders: dict[str, str]) -> None:
    """Points a pipeline config at the decoders in use, BiRefNet and the volume's DINOv3."""
    path = os.path.join(PIXAL3D_DIR, config_file)
    with open(path) as f:
        config = json.load(f)
    args = config["args"]
    for key, model in args["models"].items():
        if model in decoders:
            args["models"][key] = decoders[model]
    args["image_cond_model"]["args"]["model_name"] = DINO_DIR
    # Stock: BiRefNet's code with RMBG-2.0's weights (CC BY-NC 4.0). ZhengPeng7/BiRefNet is MIT
    args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": BIREFNET_DIR}}
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def download_naf() -> None:
    target = os.path.join(PIXAL3D_DIR, "naf", "naf_release.pth")
    if os.path.exists(target) and sha256(target) == NAF_SHA256:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    with urllib.request.urlopen(NAF_URL, timeout=120) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    if sha256(partial) != NAF_SHA256:
        os.remove(partial)
        raise SystemExit("naf_release.pth does not have the pinned sha256")
    os.replace(partial, target)


def main() -> None:
    from huggingface_hub import hf_hub_download

    which = os.environ.get("WHICH", "all")
    sets = list(FLOW_MODELS) if which == "all" else [which]
    if any(name not in FLOW_MODELS for name in sets):
        raise SystemExit("WHICH must be single, multiview or all")
    for name in (DINO_DIR, BIREFNET_DIR):
        if not os.path.isdir(name):
            raise SystemExit(f"{name} is missing: download the TRELLIS.2 worker's weights first")
    dino = sha256(os.path.join(DINO_DIR, "model.safetensors"))
    print(f"DINOv3: {'identical to' if dino == DINOV3_SHA256 else 'DIFFERENT from'} the copy Pixal3D names")
    if dino != DINOV3_SHA256:
        raise SystemExit("the volume's DINOv3 is not the one Pixal3D was trained with")

    token = os.environ.get("HF_TOKEN") or None
    for file in NOTICES:
        hf_hub_download(PIXAL3D[0], file, revision=PIXAL3D[1], local_dir=PIXAL3D_DIR, token=token)
    decoders = {name: decoder_path(name, digest) for name, digest in DECODERS.items()}
    for name in sets:
        hf_hub_download(PIXAL3D[0], CONFIGS[name], revision=PIXAL3D[1], local_dir=PIXAL3D_DIR, token=token)
        for model in FLOW_MODELS[name]:
            for ext in ("json", "safetensors"):
                hf_hub_download(PIXAL3D[0], f"{model}.{ext}", revision=PIXAL3D[1], local_dir=PIXAL3D_DIR, token=token)
            print(f"{model}: ready")
        rewrite_config(CONFIGS[name], decoders)

    moge_dir = os.path.join(PIXAL3D_DIR, "moge-2-vitl")
    hf_hub_download(MOGE[0], "model.pt", revision=MOGE[1], local_dir=moge_dir, token=token)
    download_naf()

    cache = os.path.join(PIXAL3D_DIR, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    print("Pixal3D weights ready in", PIXAL3D_DIR)


if __name__ == "__main__":
    sys.exit(main())
