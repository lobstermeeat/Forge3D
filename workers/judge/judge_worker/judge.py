"""
The judge: which of a few textures of one shape a creator would rather use, from one chat with a
vision-language model (model.py's Qwen3-VL, or anything called the same way).

The model sees the picture the shape was made from and each candidate's turntable grid (judgeviews.py
in the TRELLIS.2 worker) under a letter, and answers with a verdict per candidate (the reviewers'
"publish", "edits" or "reject") and a pick. The letters stay in here: callers give candidates in their
own order and get verdicts and the pick back by index. ``order`` shows them in another order, which is
how an experiment checks for a bias towards a position.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Protocol, Sequence

from PIL import Image

from . import parse, prompt

# Image sizes the model sees (Qwen3-VL: one token per 32 x 32 pixels): the picture at most 768 on its
# longest side (576 tokens), a grid at most 1152 x 768 (judgeviews' 3 x 2 views of 384; 864 tokens).
# Larger images are shrunk to fit, smaller ones kept
PICTURE_SIDE = 768
GRID_SIDE = 1152
GRID_PIXELS = 1152 * 768
# The answer is a short JSON object: about 60 tokens a candidate and 60 more
MAX_NEW_TOKENS = 1024
# Where a picture or grid has transparency, it goes on this
UNDERLAY = (255, 255, 255)


class Chat(Protocol):
    """A loaded model: the chat (transformers' message format, images as PIL images) in, the reply's text out."""

    def __call__(self, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str: ...


def fit(image: Image.Image, side: int, pixels: Optional[int] = None) -> Image.Image:
    """An RGB copy (transparency on white) no larger than ``side`` on its longest side and ``pixels`` in area."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, UNDERLAY + (255,)), rgba)
    image = image.convert("RGB")
    scale = min(1.0, side / max(image.size))
    if pixels is not None:
        scale = min(scale, (pixels / (image.width * image.height)) ** 0.5)
    if scale < 1.0:
        # Rounded down (less a hair, for 2000 * 0.384 coming out as 767.99…), so neither limit is passed
        size = tuple(max(1, math.floor(length * scale + 1e-6)) for length in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def judge(
    picture: Image.Image,
    candidates: Sequence[Image.Image],
    prompt_text: Optional[str],
    *,
    model: Optional[Chat] = None,
    order: Optional[Sequence[int]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """
    One chat call: the picture and every candidate's grid, a verdict for each and the best one.

    ``candidates`` are grids of the same shape with different textures (candidate 0 is the generation's own
    texture); ``prompt_text`` is what the user typed (None or "" for a photo run). ``model`` defaults to
    the one model.load() loaded in this process. ``order`` lists the candidates in the order they are
    shown (lettered K, L, M, ... in that order); by default as given. Decoding is greedy (model.py).

    Returns ``{"verdicts": ["publish" | "edits" | "reject" | None, ...], "best": index, "why", "raw": the
    reply, "seconds", "letters": the letter each candidate was shown under, "problems": what the model
    saw on each (or None)}``, plus ``"notes"`` (what in the reply was off but read anyway) and ``"tokens"``
    when there are some. A reply with no valid pick gives ``best`` 0, the generation's own texture, and
    ``"parse_error"`` saying why. Raises ValueError on bad arguments; the model's own errors go through.
    """
    started = time.perf_counter()
    count = len(candidates)
    if not 1 <= count <= len(prompt.LETTERS):
        raise ValueError(f"the judge compares 1 to {len(prompt.LETTERS)} candidates, not {count}")
    shown = list(range(count)) if order is None else [int(number) for number in order]
    if sorted(shown) != list(range(count)):
        raise ValueError(f"order must list each of the {count} candidates once, not {list(order or [])}")
    if model is None:
        from .model import loaded

        model = loaded()
    letters = prompt.LETTERS[:count]
    picture = fit(picture, PICTURE_SIDE)
    grids = [fit(candidates[number], GRID_SIDE, GRID_PIXELS) for number in shown]
    raw = model(prompt.messages(picture, grids, prompt_text, letters), max_new_tokens=max_new_tokens)
    raw = raw if isinstance(raw, str) else str(raw)
    reply = parse.parse_reply(raw, letters)

    where = {number: place for place, number in enumerate(shown)}  # candidate -> shown position
    result: dict[str, Any] = {
        "verdicts": [reply.verdicts.get(where[number]) for number in range(count)],
        "best": shown[reply.best] if reply.best is not None else 0,
        "why": reply.why,
        "raw": raw,
        "seconds": round(time.perf_counter() - started, 2),
        "letters": [letters[where[number]] for number in range(count)],
        "problems": [reply.problems.get(where[number]) for number in range(count)],
    }
    if reply.best is None:
        result["parse_error"] = reply.error or "no valid pick in the reply"
    if reply.notes:
        result["notes"] = list(reply.notes)
    tokens = getattr(model, "last_tokens", None)
    if isinstance(tokens, dict) and tokens:
        result["tokens"] = dict(tokens)
    return result
