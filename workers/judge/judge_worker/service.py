"""
A judge job: base64 images in, the judge's verdicts and pick out. Kept free of GPU code, so it can be
tested anywhere; model.py holds the model.
"""

from __future__ import annotations

import base64
import binascii
import io
import random
import traceback
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

from . import prompt
from .judge import MAX_NEW_TOKENS, Chat, judge

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_PROMPT = 500
MAX_REPLY_TOKENS = 4096
FIELDS = ("picture_png", "candidates_png", "prompt", "order", "seed", "max_new_tokens", "id")  # "id": run_job's
_GPU_FAULTS = ("cuda", "out of memory", "outofmemory", "cublas", "cudnn", "device-side assert")


class InputError(ValueError):
    """The request is malformed. The message is returned to the caller."""


@dataclass(frozen=True)
class JudgeJob:
    picture: Image.Image
    candidates: list
    prompt: str
    order: Optional[list]
    max_new_tokens: int


def needs_restart(err: BaseException) -> bool:
    """A GPU fault can leave CUDA unusable in this process; other failures can't."""
    text = f"{type(err).__name__}: {err}".lower()
    return any(marker in text for marker in _GPU_FAULTS)


def _image(encoded: Any, name: str) -> Image.Image:
    if not isinstance(encoded, str):
        raise InputError(f"{name} must be a base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as err:
        raise InputError(f"{name} is not valid base64") from err
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError(f"{name} is larger than 20 MB")
    try:
        image = Image.open(io.BytesIO(data))
        if max(image.size) > MAX_IMAGE_SIDE:
            raise InputError(f"{name} is larger than {MAX_IMAGE_SIDE}px on a side")
        image.load()
    except InputError:
        raise
    except Exception as err:  # PIL raises many types for bad data
        raise InputError(f"{name} could not be decoded") from err
    return image


def shuffled(count: int, seed: int) -> list:
    """The candidates in a seeded random order (the same for the same seed and count)."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


def parse_job(payload: object) -> JudgeJob:
    """A judge job (handle_job) -> validated arguments for judge()."""
    if not isinstance(payload, dict):
        raise InputError("input must be an object")
    unknown = sorted(str(key) for key in payload if key not in FIELDS)
    if unknown:
        raise InputError(f"unknown fields: {', '.join(unknown)}")
    picture = _image(payload.get("picture_png"), "picture_png")
    encoded = payload.get("candidates_png")
    if not isinstance(encoded, list) or not 1 <= len(encoded) <= len(prompt.LETTERS):
        raise InputError(f"candidates_png must be a list of 1 to {len(prompt.LETTERS)} base64 images")
    candidates = [_image(item, f"candidates_png[{number}]") for number, item in enumerate(encoded)]

    text = payload.get("prompt")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        raise InputError("prompt must be a string")
    elif len(text) > MAX_PROMPT:
        raise InputError(f"prompt is longer than {MAX_PROMPT} characters")

    order, seed = payload.get("order"), payload.get("seed")
    if order is not None and seed is not None:
        raise InputError("give order or seed, not both")
    if order is not None:
        if (
            not isinstance(order, list)
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in order)
            or sorted(order) != list(range(len(candidates)))
        ):
            raise InputError(f"order must list each candidate's index (0 to {len(candidates) - 1}) once")
    elif seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
            raise InputError("seed must be an integer between 0 and 2^31 - 1")
        order = shuffled(len(candidates), seed)

    tokens = payload.get("max_new_tokens", MAX_NEW_TOKENS)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= MAX_REPLY_TOKENS:
        raise InputError(f"max_new_tokens must be an integer between 1 and {MAX_REPLY_TOKENS}")
    return JudgeJob(picture=picture, candidates=candidates, prompt=text.strip(), order=order, max_new_tokens=tokens)


def handle_job(job: dict, model: Chat, name: Optional[str] = None) -> dict:
    """
    Input: ``{"picture_png", "candidates_png": [...], "prompt"?, "order"? | "seed"?, "max_new_tokens"?}``:
    the picture the shape was made from and each candidate's turntable grid, as base64 images (PNG or
    JPEG; candidate 0 is the generation's own texture); what the user typed; the order to show the
    candidates in (a list of their indices), or a seed to shuffle them with (as given if neither).

    Output: judge()'s result (``verdicts``, ``best``, ``why``, ``raw``, ``seconds``, ``letters``,
    ``problems``, and ``parse_error``, ``notes`` or ``tokens`` when there are some), with ``"model"``
    (``name``), or ``{"error": "invalid input: …" | "judge failed: …"}``.
    """
    try:
        spec = parse_job(job)
    except InputError as err:
        return {"error": f"invalid input: {err}"}
    try:
        result = judge(
            spec.picture,
            spec.candidates,
            spec.prompt,
            model=model,
            order=spec.order,
            max_new_tokens=spec.max_new_tokens,
        )
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        traceback.print_exc()
        result = {"error": f"judge failed: {type(err).__name__}: {err}"}
        if needs_restart(err):
            result["refresh_worker"] = True
        return result
    if name:
        result["model"] = name
    return result
