"""
Phase 8: the weights the view painter needs, into the orainge-models volume, from a staging app (nothing
is deployed; production's orainge-ai doesn't read these folders).

- Qwen/Qwen-Image-Edit-2511 (Apache-2.0): the whole diffusers repo (transformer, Qwen2.5-VL text encoder,
  VAE, processor, scheduler), at the revision the Hub has now; the run prints it, to be pinned.
- lightx2v/Qwen-Image-Edit-2511-Lightning (Apache-2.0): only its 8-step LoRA files (the repo also holds
  FP8 builds of the whole model, which we don't need yet).

    ORAINGE_APP_NAME=orainge-p8-download modal run ops/p8_download.py
"""

from __future__ import annotations

import json
import os

import modal

APP_NAME = os.environ.get("ORAINGE_APP_NAME", "orainge-p8-download")
if APP_NAME == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

app = modal.App(APP_NAME)
models = modal.Volume.from_name("orainge-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# folder in the volume: (repo, which files: None for all, "lora8" for the 8-step LoRA files)
REPOS = {
    "Qwen-Image-Edit-2511": ("Qwen/Qwen-Image-Edit-2511", None),
    "Qwen-Image-Edit-2511-Lightning": ("lightx2v/Qwen-Image-Edit-2511-Lightning", "lora8"),
}


def _wanted(name: str, size: int, kind) -> bool:
    lower = name.lower()
    if kind is None:
        return True
    if lower.endswith((".md", ".json", ".txt")) or "license" in lower:
        return True
    return lower.endswith(".safetensors") and "8step" in lower and size < 3_000_000_000


@app.function(
    image=image,
    volumes={"/models": models},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=2 * 60 * 60,
    cpu=8,
    memory=16384,
)
def download() -> dict:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    report = {}
    for folder, (repo, kind) in REPOS.items():
        info = api.model_info(repo, files_metadata=True)
        files = [(s.rfilename, int(s.size or 0)) for s in info.siblings]
        chosen = [name for name, size in files if _wanted(name, size, kind)]
        snapshot_download(repo, revision=info.sha, local_dir=f"/models/{folder}", allow_patterns=chosen)
        models.commit()
        card = info.card_data.to_dict() if info.card_data else {}
        report[folder] = {
            "repo": repo,
            "revision": info.sha,
            "license": card.get("license"),
            "gigabytes": round(sum(size for name, size in files if name in chosen) / 1e9, 2),
            "files": [f"{name} ({size / 1e9:.2f} GB)" for name, size in files],
            "downloaded": chosen if kind else "all",
        }
        print(f"[p8] {folder}: {repo}@{info.sha} license={card.get('license')} {report[folder]['gigabytes']} GB", flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(download.remote(), indent=1))
