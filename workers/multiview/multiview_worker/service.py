"""
One picture of an object -> six views around it, for the 3D step to see the sides the picture hides.
And for experiments, a picture and a mesh's geometry -> six views of that mesh (handle_geometry_job).

Job handling, kept free of GPU code so it can be tested anywhere; generator.py and geometry.py hold
the models.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np
from PIL import Image

from .cameras import AZIMUTHS, ELEVATION, IG2MV_VIEWS, camera_info
from .generator import GUIDANCE, REFERENCE_SCALE, STEPS
from .geometry import CONTROL_SCALE, unpack_control
from .inputs import DEFAULT_PROMPT, MAX_IMAGE_BYTES, MAX_PROMPT, Fetch, InputError, decode_image, parse_job


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
    where 0 looks level at the front of the object in the picture, and the orthographic camera
    they share (cameras.py).
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


# Image+geometry views (geometry.py), for experiments: the GeometryViews class in modal_app.py


class Drawer(Protocol):
    """geometry.GeometryViewGenerator: draw_views and the last call's timings."""

    last_timings: dict

    def draw_views(self, reference: Any, control: Any, prompt: str, seed: int, **settings: Any) -> list: ...


@dataclass(frozen=True)
class GeometryJob:
    reference: Image.Image
    control: np.ndarray
    prompt: str
    seed: int
    steps: int
    guidance: float
    reference_scale: float
    control_scale: float


GEOMETRY_FIELDS = (
    "image_base64",
    "control_pngs",
    "prompt",
    "seed",
    "steps",
    "guidance",
    "reference_scale",
    "control_scale",
    "id",  # what run_job adds
)
MAX_STEPS = 100
# The ranges a number setting may take
GEOMETRY_RANGES = {"guidance": (0.0, 20.0), "reference_scale": (0.0, 3.0), "control_scale": (0.0, 3.0)}


def _number(payload: dict, name: str, default: float) -> float:
    value = payload.get(name, default)
    low, high = GEOMETRY_RANGES[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name} must be a number")
    if not low <= value <= high:
        raise InputError(f"{name} must be between {low:g} and {high:g}")
    return float(value)


def parse_geometry_job(payload: object) -> GeometryJob:
    """A geometry job (handle_geometry_job) -> validated draw_views arguments."""
    if not isinstance(payload, dict):
        raise InputError("input must be an object")
    unknown = sorted(str(key) for key in payload if key not in GEOMETRY_FIELDS)
    if unknown:
        raise InputError(f"unknown fields: {', '.join(unknown)}")

    encoded = payload.get("image_base64")
    if not isinstance(encoded, str):
        raise InputError("image_base64 must be a base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as err:
        raise InputError("image_base64 is not valid base64") from err
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("image is larger than 20 MB")
    reference = decode_image(data)
    control = unpack_control(payload.get("control_pngs"))

    prompt = payload.get("prompt")
    if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
        prompt = DEFAULT_PROMPT
    elif not isinstance(prompt, str):
        raise InputError("prompt must be a string")
    elif len(prompt) > MAX_PROMPT:
        raise InputError(f"prompt is longer than {MAX_PROMPT} characters")

    seed = payload.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise InputError("seed must be an integer between 0 and 2^31 - 1")
    steps = payload.get("steps", STEPS)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_STEPS:
        raise InputError(f"steps must be an integer between 1 and {MAX_STEPS}")

    return GeometryJob(
        reference=reference,
        control=control,
        prompt=prompt.strip(),
        seed=seed,
        steps=steps,
        guidance=_number(payload, "guidance", GUIDANCE),
        reference_scale=_number(payload, "reference_scale", REFERENCE_SCALE),
        control_scale=_number(payload, "control_scale", CONTROL_SCALE),
    )


def handle_geometry_job(job: dict, drawer: Drawer) -> dict:
    """
    Input: ``{"image_base64", "control_pngs", "prompt"?, "seed"?, "steps"?, "guidance"?,
    "reference_scale"?, "control_scale"?}``: the picture; the mesh's maps as geometry.pack_control
    makes them (12 base64 PNGs: six position maps, then six normal maps, views in IG2MV_VIEWS order);
    the caption ("high quality" by default) and seed (0); 30 steps, guidance 3 and both scales 1 unless
    given.

    Output: ``{"views": [six base64 RGB PNGs, IG2MV_VIEWS order], "seconds", "timings"}``, or
    ``{"error": "invalid input: …" | "generation failed: …"}``.
    """
    try:
        spec = parse_geometry_job(job)
    except InputError as err:
        return {"error": f"invalid input: {err}"}

    started = time.perf_counter()
    try:
        images = drawer.draw_views(
            spec.reference,
            spec.control,
            spec.prompt,
            spec.seed,
            steps=spec.steps,
            guidance=spec.guidance,
            reference_scale=spec.reference_scale,
            control_scale=spec.control_scale,
        )
        if len(images) != len(IG2MV_VIEWS):
            raise RuntimeError(f"expected {len(IG2MV_VIEWS)} views, got {len(images)}")
        views = []
        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            views.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
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
        "views": views,
        "seconds": round(time.perf_counter() - started, 3),
        "timings": dict(getattr(drawer, "last_timings", None) or {}),
    }
