"""
Puts the judge's models under MODELS_ROOT (the orainge-models volume), pinned, every weight file checked
against its sha256.

- Qwen/Qwen3-VL-8B-Instruct (Apache-2.0): dense, 8.8B parameters, 17.5 GB in bf16 (four files)
- Qwen/Qwen3-VL-30B-A3B-Instruct (Apache-2.0): a mixture of experts, 31B parameters with 3B active per
  token, 62 GB in bf16 (thirteen files)

Both repositories hold the weights, the configs, the tokenizer and the processor's settings, and no
code: transformers has the models. Neither is gated; a token only raises rate limits. The licenses are
the model cards' at these revisions (license: apache-2.0). WHICH picks "8b", "30b" or "all" (default).

Revisions and sha256 values are pinned so an upstream change can't silently change the judge.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("MODELS_ROOT", "/models")
WHICH = os.environ.get("WHICH", "all")

MODELS = {
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "30b": ("Qwen/Qwen3-VL-30B-A3B-Instruct", "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"),
}
# Folder names match judge_worker/model.py
FOLDERS = {"8b": "Qwen3-VL-8B-Instruct", "30b": "Qwen3-VL-30B-A3B-Instruct"}
# Everything but .gitattributes: the weights, model.safetensors.index.json, the configs, the tokenizer
# (tokenizer.json, vocab.json, merges.txt), the processor's settings and the model card
PATTERNS = ["*.safetensors", "*.json", "*.txt", "README.md"]
# The weight files' LFS sha256 at those revisions, from Hugging Face's API (the revision's file list and
# the commit's tree agree)
WEIGHTS = {
    "8b": {
        "model-00001-of-00004.safetensors": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
        "model-00002-of-00004.safetensors": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
        "model-00003-of-00004.safetensors": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
        "model-00004-of-00004.safetensors": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    },
    "30b": {
        "model-00001-of-00013.safetensors": "6b205ea6332a532e5b97fd6281ac700f125a1a5f9a3457a22b797edfd35a5f67",
        "model-00002-of-00013.safetensors": "a739a546e00643e0555bbb6bcd57fa5ba60fb2de24177c8819d2f899f02c2223",
        "model-00003-of-00013.safetensors": "c57ec0f3fd06cae62980851f2661bd2b5e9dcdff37b8f9ece0cf44dea645efc3",
        "model-00004-of-00013.safetensors": "46b07bc82396976dc62f86bccb51c3c585a616b36b8ad8eef700bde75fedf88b",
        "model-00005-of-00013.safetensors": "3fa2fdf4b835d885efcaf0e4f6c7c00905cbaad7e293af988a0cf5408c3078c8",
        "model-00006-of-00013.safetensors": "89a762b82ff9db9351cd38794ecd1ee170bbe87d7fa56110ca837c088158ccfb",
        "model-00007-of-00013.safetensors": "634e437c0a038d8f447fceb44fb4f5230600af6a71d0fbe112711d9e84d6f07a",
        "model-00008-of-00013.safetensors": "4c6f316b27536bd914a6ea09c03ec1b80a9a8d55e3478b1bd5a307f8de8d7fbd",
        "model-00009-of-00013.safetensors": "e2e2a338c724ad75aea2b64211472c1037945fab398a879116cbbbc3112391da",
        "model-00010-of-00013.safetensors": "adca9184b757627840f7fce81e8e4e1203c7551fa2c1bdb469503dc7dfab0583",
        "model-00011-of-00013.safetensors": "71b3d1ff9ddc9358de332aa52b92fc001565edfcfd61f9e3679f96f85c7cbc22",
        "model-00012-of-00013.safetensors": "f4b204bedb467a2cea642dc38d30c2cdd29e48fbd2d7d5ee5cc14ba4b786ef6f",
        "model-00013-of-00013.safetensors": "e167e2806d63c404660b6d609a7eaef2e82bca478688ba6f83155128d8187539",
    },
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check(which: str, folder: str) -> None:
    """Every pinned weight file is there with its sha256, and no other weight file is."""
    expected = WEIGHTS[which]
    present = sorted(name for name in os.listdir(folder) if name.endswith(".safetensors"))
    if present != sorted(expected):
        raise SystemExit(f"{which}: the weight files are {present}, not the pinned {sorted(expected)}")
    # hashlib lets other threads run while it hashes, so the files are checked side by side
    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as pool:
        found = dict(zip(expected, pool.map(lambda name: sha256(os.path.join(folder, name)), expected)))
    wrong = sorted(name for name, digest in found.items() if digest != expected[name])
    if wrong:
        raise SystemExit(f"{which}: {', '.join(wrong)} do not have the pinned sha256")


def main() -> None:
    from huggingface_hub import snapshot_download

    if WHICH != "all" and WHICH not in MODELS:
        raise SystemExit(f"WHICH must be all, {', '.join(MODELS)}, not {WHICH!r}")
    token = os.environ.get("HF_TOKEN") or None
    for which in MODELS if WHICH == "all" else [WHICH]:
        repo, revision = MODELS[which]
        folder = os.path.join(ROOT, FOLDERS[which])
        snapshot_download(repo, revision=revision, local_dir=folder, allow_patterns=PATTERNS, token=token)
        check(which, folder)
        # huggingface_hub leaves download bookkeeping in .cache; nothing reads it at runtime
        cache = os.path.join(folder, ".cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
        print(f"{which}: {repo} at {revision} ready in {folder}")


if __name__ == "__main__":
    main()
