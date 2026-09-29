"""
Text prompt -> a few reference images the user picks from before any 3D is generated.

Image-to-3D works best on one centred object on a plain background with soft light, so
the prompt is wrapped in a product-shot template. Picking an image costs seconds of GPU;
throwing away a 3D result costs a minute.
"""

from __future__ import annotations

import io
import random
import re
import time
import traceback
from typing import Any, Callable, Protocol

REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_PROMPT = 500
MAX_COUNT = 4

TEMPLATE = (
    "{prompt}. A single object, centered and fully in frame, on a plain light gray background, "
    "soft even studio lighting, three-quarter view from slightly above, sharp focus, "
    "no text, no other objects"
)


class InputError(ValueError):
    pass


_GPU_FAULTS = ("cuda", "out of memory", "outofmemory", "cublas", "cudnn", "device-side assert")


def needs_restart(err: BaseException) -> bool:
    """A GPU fault can leave CUDA unusable in this process; other failures (storage, network) can't."""
    text = f"{type(err).__name__}: {err}".lower()
    return any(marker in text for marker in _GPU_FAULTS)


class Generator(Protocol):
    def __call__(self, prompt: str, seed: int) -> Any: ...  # returns a PIL image


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> dict: ...


def build_prompt(prompt: str) -> str:
    return TEMPLATE.format(prompt=prompt.strip().rstrip("."))


def parse(payload: object, fallback_id: str) -> tuple[str, int, int, str]:
    if not isinstance(payload, dict):
        raise InputError("input must be an object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("prompt is required")
    if len(prompt) > MAX_PROMPT:
        raise InputError(f"prompt is longer than {MAX_PROMPT} characters")
    count = payload.get("count", MAX_COUNT)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_COUNT:
        raise InputError(f"count must be between 1 and {MAX_COUNT}")
    seed = payload.get("seed")
    if seed is None:
        seed = random.randrange(2**31 - MAX_COUNT)
    elif not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31 - MAX_COUNT:
        raise InputError("seed must be a non-negative 31-bit integer")
    request_id = payload.get("request_id", fallback_id)
    if not isinstance(request_id, str) or not REQUEST_ID.match(request_id):
        raise InputError("request_id must be 1-64 letters, digits, '-' or '_'")
    return prompt, count, seed, request_id


def handle_job(job: dict, generate: Generator, storage: Storage) -> dict:
    """Input: ``{"prompt", "count"?: 1-4, "seed"?, "request_id"?}``. Image i uses seed + i."""
    try:
        prompt, count, seed, request_id = parse(job.get("input"), str(job.get("id", "job")))
    except InputError as err:
        return {"error": f"invalid input: {err}"}

    started = time.perf_counter()
    full_prompt = build_prompt(prompt)
    images = []
    try:
        for i in range(count):
            image = generate(full_prompt, seed + i)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            # The seed keeps each image at its own URL, so caches never serve a stale one
            key = f"ai/{request_id}/reference-{seed + i}.png"
            stored = storage.put(key, buf.getvalue(), "image/png")
            images.append({**stored, "seed": seed + i})
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        result = {"error": f"generation failed: {type(err).__name__}: {err}"}
        if needs_restart(err):
            result["refresh_worker"] = True
        return result

    return {
        "request_id": request_id,
        "prompt": full_prompt,
        "images": images,
        "seconds": round(time.perf_counter() - started, 3),
    }


def make_flux_generator(model_dir: str, cpu_offload: bool) -> Callable[[str, int], Any]:
    """Loads FLUX.1 [schnell] (Apache-2.0) once; 4 steps, no guidance, 1024x1024."""
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()  # fits 24 GB cards, slower
    else:
        pipe.to("cuda")

    def generate(prompt: str, seed: int):
        generator = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=256,
            height=1024,
            width=1024,
            generator=generator,
        ).images[0]

    return generate
