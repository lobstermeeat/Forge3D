"""Bakes FLUX.1 [schnell] (Apache-2.0) into the image, pinned (run during `docker build`).

The repository asks you to accept its terms on Hugging Face first, so the build needs a token.
Only FLUX.1 [schnell] is used: FLUX.1 [dev] is licensed for non-commercial use only.
"""

import os

from huggingface_hub import snapshot_download

FLUX = ("black-forest-labs/FLUX.1-schnell", "741f7c3ce8b383c54771c7003378a50191e9efe9")
TARGET = os.environ.get("FLUX_MODEL_DIR", "/models/FLUX.1-schnell")


def main() -> None:
    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        raise SystemExit("HF_TOKEN is required: accept the FLUX.1 [schnell] terms on Hugging Face first")
    snapshot_download(
        FLUX[0],
        revision=FLUX[1],
        local_dir=TARGET,
        token=token,
        # diffusers layout only: skip the 24 GB single-file duplicate and sample images
        ignore_patterns=["flux1-schnell.safetensors", "*.png", "*.jpg"],
    )
    print("FLUX.1 [schnell] ready in", TARGET)


if __name__ == "__main__":
    main()
