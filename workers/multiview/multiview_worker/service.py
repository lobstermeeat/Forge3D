"""
One picture of an object -> six views around it, for the 3D step to see the sides the picture hides.

Job handling, kept free of GPU code so it can be tested anywhere; generator.py holds the model.
"""

from __future__ import annotations

import io
import time
import traceback
from typing import Any, Optional, Protocol

from .cameras import AZIMUTHS, ELEVATION, camera_info
from .inputs import Fetch, InputError, parse_job


class Generator(Protocol):
    """Picture, seed and caption -> one RGBA image per azimuth in AZIMUTHS, in that order."""

    def __call__(self, image: Any, seed: int, prompt: str) -> list: ...


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> dict: ...


_GPU_FAULTS = ("cuda", "out of memory", "outofmemory", "cublas", "cudnn", "device-side assert")


def needs_restart(err: BaseException) -> bool:
    """A GPU fault can leave CUDA unusable in this process; other failures (storage, network) can't."""
    text = f"{type(err).__name__}: {err}".lower()
    return any(marker in text for marker in _GPU_FAULTS)


def view_key(request_id: str, seed: int, azimuth: int) -> str:
    # The seed keeps each result at its own URL, so caches never serve a stale view
    return f"ai/{request_id}/view-{seed}-{azimuth}.png"


def handle_job(job: dict, generate: Generator, storage: Storage, fetch: Optional[Fetch] = None) -> dict:
    """
    Input: ``{"image_url" | "image_base64", "seed"?, "prompt"?, "request_id"?}``. ``prompt`` is an
    optional short description of the object ("a retro arcade machine"); without it the model gets
    MV-Adapter's default caption.

    Output: ``{"request_id", "seed", "views": [{"azimuth", "elevation", "key", "url" | "base64"}, ...],
    "camera", "seconds", "timings"}``: six RGBA PNG cutouts at the azimuths in cameras.AZIMUTHS,
    where 0 is the picture's own view, and the orthographic camera they share (cameras.py).
    """
    try:
        spec = parse_job(job.get("input"), fallback_id=str(job.get("id", "job")), fetch=fetch)
    except InputError as err:
        return {"error": f"invalid input: {err}"}

    started = time.perf_counter()
    try:
        images = generate(spec.image, spec.seed, spec.prompt)
        timings = dict(getattr(generate, "last_timings", None) or {})
        if len(images) != len(AZIMUTHS):
            raise RuntimeError(f"expected {len(AZIMUTHS)} views, got {len(images)}")
        uploaded = time.perf_counter()
        views = []
        for azimuth, image in zip(AZIMUTHS, images):
            buffer = io.BytesIO()
            image.convert("RGBA").save(buffer, format="PNG")
            stored = storage.put(view_key(spec.request_id, spec.seed, azimuth), buffer.getvalue(), "image/png")
            views.append({"azimuth": azimuth, "elevation": ELEVATION, **stored})
        timings["upload_s"] = round(time.perf_counter() - uploaded, 3)
    except InputError as err:
        # e.g. no object found in the picture
        return {"error": f"invalid input: {err}"}
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        traceback.print_exc()
        result = {"error": f"generation failed: {type(err).__name__}: {err}"}
        if needs_restart(err):
            result["refresh_worker"] = True
        return result

    return {
        "request_id": spec.request_id,
        "seed": spec.seed,
        "views": views,
        "camera": camera_info(),
        "seconds": round(time.perf_counter() - started, 3),
        "timings": timings,
    }
