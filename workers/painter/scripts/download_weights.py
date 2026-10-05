"""
Puts the view painter's models under MODELS_ROOT (the orainge-models volume), pinned, every weight file checked
against its size and sha256.

- Qwen/Qwen-Image-Edit-2511 (Apache-2.0): the whole diffusers repository but .gitattributes: the 20B MMDiT
  transformer (40.9 GB in bf16, five files), the Qwen2.5-VL-7B text encoder (16.6 GB, four files), the VAE
  (0.25 GB), the tokenizer, the processor and the configs. No code: diffusers and transformers have the models.
- lightx2v/Qwen-Image-Edit-2511-Lightning (Apache-2.0): only the 8-step Lightning LoRA, in fp32 (1.7 GB, the one
  painter_worker/qwen.py loads) and bf16 (0.85 GB), and the model card. Not the 4-step LoRAs or the 20 GB fp8
  models; the repository has no JSON but the fp8 split model's index.

Neither repository is gated; a token (HF_TOKEN) only raises rate limits. The licenses are the model cards' at
these revisions (license: apache-2.0). Revisions, sizes and sha256 values are pinned so an upstream change can't
silently change the painter.

    python scripts/download_weights.py [--which all|model|lora]   download, then check every file's sha256
    python scripts/download_weights.py --check [--sha256]          check what is there, without downloading
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("MODELS_ROOT", "/models")

MODELS = {
    "model": ("Qwen/Qwen-Image-Edit-2511", "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"),
    "lora": ("lightx2v/Qwen-Image-Edit-2511-Lightning", "d74eba145674fd7e31b949324e148e21e7118abd"),
}
# Folder names match painter_worker/qwen.py
FOLDERS = {"model": "Qwen-Image-Edit-2511", "lora": "Qwen-Image-Edit-2511-Lightning"}
LORA_FILES = [
    "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
    "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
]
PATTERNS = {
    # The weights, the configs and indexes, the tokenizer's and processor's files, the chat templates, the card
    "model": ["*.safetensors", "*.json", "*.txt", "*.jinja", "README.md"],
    # Exact names: "*8steps*" would also take the 20 GB fp8 model fused with the 8-step LoRA
    "lora": [*LORA_FILES, "README.md"],
}
# Every weight file's size in bytes and LFS sha256 at those revisions, from Hugging Face's API (the revision's
# file tree and each file's page agree)
WEIGHTS = {
    "model": {
        "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": (
            9_973_578_592,
            "2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
        ),
        "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": (
            9_987_326_072,
            "54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259",
        ),
        "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": (
            9_987_307_440,
            "c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16",
        ),
        "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": (
            9_930_685_712,
            "ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d",
        ),
        "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": (
            982_130_472,
            "2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940",
        ),
        "text_encoder/model-00001-of-00004.safetensors": (
            4_968_243_304,
            "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
        ),
        "text_encoder/model-00002-of-00004.safetensors": (
            4_991_495_816,
            "b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
        ),
        "text_encoder/model-00003-of-00004.safetensors": (
            4_932_751_040,
            "09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6",
        ),
        "text_encoder/model-00004-of-00004.safetensors": (
            1_691_924_384,
            "5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
        ),
        "vae/diffusion_pytorch_model.safetensors": (
            253_806_966,
            "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
        ),
    },
    "lora": {
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors": (
            1_698_951_104,
            "e2e24eda6295bf08222638b0b8278628461e86e8e1bc1b10411c7def13451262",
        ),
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors": (
            849_608_296,
            "a9e81a58a78f260f67b337a6f615e8fa4cd3bc79847c77b7d61a581b789b1ba8",
        ),
    },
}
# The other files the pipeline reads (and the model cards), all of the repository's at that revision
FILES = {
    "model": [
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "processor/added_tokens.json",
        "processor/chat_template.jinja",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/special_tokens_map.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
    ],
    "lora": ["README.md"],
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_files(folder: str) -> list:
    """Every .safetensors file under the folder, as paths relative to it (huggingface_hub's .cache aside)."""
    found = []
    for parent, folders, names in os.walk(folder):
        folders[:] = [name for name in folders if name != ".cache"]
        found += [os.path.join(parent, name) for name in names if name.endswith(".safetensors")]
    return sorted(os.path.relpath(path, folder).replace(os.sep, "/") for path in found)


def check(which: str, folder: str, hashes: bool = True) -> None:
    """
    Every pinned file is there, every weight file with its pinned size (and sha256, when `hashes`), and there is
    no other weight file.
    """
    missing = [name for name in [*FILES[which], *WEIGHTS[which]] if not os.path.isfile(os.path.join(folder, name))]
    if missing:
        raise SystemExit(f"{which}: {folder} is missing {', '.join(missing)}")
    expected = WEIGHTS[which]
    present = weight_files(folder)
    if present != sorted(expected):
        raise SystemExit(f"{which}: the weight files are {present}, not the pinned {sorted(expected)}")
    wrong = sorted(name for name, (size, _) in expected.items() if os.path.getsize(os.path.join(folder, name)) != size)
    if wrong:
        raise SystemExit(f"{which}: {', '.join(wrong)} do not have the pinned size")
    if not hashes:
        return
    # hashlib lets other threads run while it hashes, so the files are checked side by side
    names = sorted(expected)
    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as pool:
        found = dict(zip(names, pool.map(lambda name: sha256(os.path.join(folder, name)), names)))
    wrong = [name for name in names if found[name] != expected[name][1]]
    if wrong:
        raise SystemExit(f"{which}: {', '.join(wrong)} do not have the pinned sha256")


def download(which: str, root: str) -> None:
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[which]
    folder = os.path.join(root, FOLDERS[which])
    token = os.environ.get("HF_TOKEN") or None
    snapshot_download(repo, revision=revision, local_dir=folder, allow_patterns=PATTERNS[which], token=token)
    check(which, folder)
    # huggingface_hub leaves download bookkeeping in .cache; nothing reads it at run time
    cache = os.path.join(folder, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    print(f"{which}: {repo} at {revision} ready in {folder}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download or check the view painter's pinned weights.")
    parser.add_argument("--which", choices=["all", *MODELS], default="all")
    parser.add_argument("--check", action="store_true", help="check the files already there, without downloading")
    parser.add_argument("--sha256", action="store_true", help="with --check: also hash every weight file (minutes)")
    options = parser.parse_args(argv)
    if options.sha256 and not options.check:
        parser.error("--sha256 goes with --check (a download always checks the sha256)")
    for which in MODELS if options.which == "all" else [options.which]:
        if options.check:
            folder = os.path.join(ROOT, FOLDERS[which])
            check(which, folder, hashes=options.sha256)
            print(f"{which}: {folder} has every pinned file" + (", every sha256 matching" if options.sha256 else ""))
        else:
            download(which, ROOT)


if __name__ == "__main__":
    main()
