"""
Text prompt -> a few reference images the user picks from before any 3D is generated.

Image-to-3D works best on one centred object on a plain background with soft light, so
the prompt is wrapped in a product-shot template. Picking an image costs seconds of GPU;
throwing away a 3D result costs a minute. Each image is scored on how well it frames its
object, so the best start can be picked or recommended without looking.
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

# The angle comes first and as what the picture shows. The old template ("{prompt}. A single object, centered
# ... three-quarter view from slightly above ...") got a three-quarter view in 19-25% of FLUX.1 [schnell]'s
# pictures, which mostly drew products straight from the front or the side, as catalogues show them, so
# image-to-3D had to make up the sides and back. This one got 69-75% on two sets of seeds (Phase 8, 28 prompts:
# Phase 2's twenty and eight real products, 4 pictures each, labelled blind). Round things (a cup, a bottle, a
# balloon) still come at eye level, and a phone shows its back.
TEMPLATE = (
    "A studio product photo of {prompt}: an angled perspective shot from the front left corner, slightly above, "
    "showing its front and its side. One whole object, fully in frame, on a plain light gray background, soft "
    "even lighting, sharp focus, no text, no other objects"
)


# Scoring. Pictures are measured at this size (longest side): plenty for framing, and quick on a CPU
SCORE_SIZE = 512
# The background is flooded in from the picture's edges. The flood starts from edge pixels within EDGE_MATCH
# of the edges' usual colour and crosses to neighbours within STEP (differences summed over R, G and B; STEP
# suits SCORE_SIZE): smooth light and soft shadows change that little from pixel to pixel, an object's
# outline doesn't. It never enters a pixel whose tint is TINT or more from the edges': a coloured object.
EDGE_MATCH = 36.0
STEP = 8.0
TINT = 12.0
# Object pieces closer than this share of the picture count as one: a faint part between them (a white
# collar, a glass neck) is lost to the flood more often than two objects stand that close
MERGE = 0.012
# Smaller pieces are noise; pieces over PIECE of the largest one's area count as separate objects
SPECK = 0.001
PIECE = 0.05
# The view. Seen straight on (from the front, the back, or square to a side), an object's edges mirror each
# other about a vertical line through it, and its horizontal edges stay horizontal; turned towards a
# three-quarter view, neither holds. MIRROR is the share of the object's edge energy that matches its mirror
# image about the best such line within AXIS_RANGE of the object's width from its middle: up to MIRROR[0] the
# view counts as turned, from MIRROR[1] as straight on. Phase 8 measured it on 896 labelled FLUX pictures:
# three-quarter views had a median of 0.08, straight front views 0.31.
MIRROR = (0.10, 0.30)
AXIS_RANGE = 0.2
# A picture seen straight on loses up to this share of its score: image-to-3D has to make up the sides it hides,
# so a turned picture that frames its object nearly as well is suggested first
STRAIGHT_PENALTY = 0.3


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


def score_picture(image: Any) -> dict:
    """
    How good a start for 3D a picture is, judged by its framing: ``{"score": 0-1, "issues": [...]}``.

    Image-to-3D crops to the object and makes up what it can't see, so the best start shows one
    whole object, large and near the centre. The background is flooded in from the edges (see
    STEP), and whatever it can't reach is the object. Then:

    - size: the object's longest side as a share of the picture's. 70-90% scores 1, falling to 0
      at 10% ("small in the frame" below 35%) and to 0.6 when it spans the whole picture;
    - touching the picture's edge along 1% of a side or more ("cut off at the edge") halves it;
    - more than one sizeable piece ("several separate objects") takes off 30%;
    - an object away from the centre loses up to 30% ("off-centre" past a fifth of the picture);
    - an object seen straight on loses up to 30% ("seen straight on" past halfway; see MIRROR), so a
      three-quarter view that frames its object about as well is suggested first.

    The view check can't tell a round object (a bottle, a donut) seen from above from one seen straight
    on: both mirror themselves. As all four pictures of such an object lose about the same, framing
    still decides between them. Nor does it see see-through parts, or whether the picture shows what
    was asked for.
    """
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")  # a copy, so the caller's image is left alone
    rgb.thumbnail((SCORE_SIZE, SCORE_SIZE), Image.Resampling.BOX)
    pixels = np.asarray(rgb, dtype=np.float32)
    found = _object_mask(pixels)
    h, w = found.shape

    # Specks go by their own size; the rest are grouped into objects by MERGE
    ids = _label(found)
    is_piece = np.bincount(ids[found], minlength=h * w) >= max(1.0, SPECK * h * w)  # by label
    kept = np.append(is_piece, False)[ids]  # the background's label, h * w, is no piece
    if not kept.any():
        return {"score": 0.0, "issues": ["no clear object"]}
    several = False
    if np.count_nonzero(is_piece) > 1:
        groups = _label(_grow(kept, max(1, round(MERGE * max(h, w)))))
        sizes = np.bincount(groups[kept])
        several = np.count_nonzero(sizes >= PIECE * sizes.max()) > 1
    rows, cols = np.flatnonzero(kept.any(axis=1)), np.flatnonzero(kept.any(axis=0))
    top, bottom = rows[0] / h, (rows[-1] + 1) / h
    left, right = cols[0] / w, (cols[-1] + 1) / w
    touching = max(kept[0].mean(), kept[-1].mean(), kept[:, 0].mean(), kept[:, -1].mean()) >= 0.01
    size = float(max(right - left, bottom - top))
    off = float(np.hypot((left + right) / 2 - 0.5, (top + bottom) / 2 - 0.5))

    issues = []
    score = min(1.0, max(0.0, (size - 0.1) / 0.6))
    if size < 0.35:
        issues.append("small in the frame")
    if size > 0.9:
        score *= 1 - 4 * (size - 0.9)  # this close to the edges, a part may be cut off unseen
        if not touching:
            issues.append("fills the frame edge to edge")
    if touching:
        score *= 0.5
        issues.append("cut off at the edge")
    if several:
        score *= 0.7
        issues.append("several separate objects")
    score *= 1 - min(0.3, max(0.0, off - 0.1))
    if off > 0.2:
        issues.append("off-centre")
    straight = straight_on(pixels, kept)
    score *= 1 - STRAIGHT_PENALTY * straight
    if straight >= 0.5:
        issues.append("seen straight on")
    return {"score": round(score, 3), "issues": issues}


def straight_on(pixels: Any, kept: Any) -> float:
    """
    How squarely the object in `kept` (a mask over `pixels`, RGB) faces the camera: 0 turned, 1 straight
    on. It is the share of the object's edge energy (luminance gradients, its outline included) that
    matches its mirror image about the best vertical line within AXIS_RANGE of the object's middle,
    scaled from MIRROR[0] to MIRROR[1]. A mirrored edge keeps its vertical gradient and flips its
    horizontal one, so for every line x = k / 2 the match is the sum of gy(x) gy(k - x) - gx(x) gx(k - x)
    over the object's rows, which one FFT along the rows gives for all lines at once.
    """
    import numpy as np

    if not kept.any():
        return 0.0
    luminance = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
    p = np.pad(luminance, 1, mode="edge")  # Sobel gradients
    gx = 2 * (p[1:-1, 2:] - p[1:-1, :-2]) + p[:-2, 2:] - p[:-2, :-2] + p[2:, 2:] - p[2:, :-2]
    gy = 2 * (p[2:, 1:-1] - p[:-2, 1:-1]) + p[2:, :-2] - p[:-2, :-2] + p[2:, 2:] - p[:-2, 2:]
    near = _grow(kept, 2)  # the outline's edges too
    gx = np.where(near, gx, 0.0).astype(np.float64)
    gy = np.where(near, gy, 0.0).astype(np.float64)
    energy = float((gx * gx + gy * gy).sum())
    if energy <= 0.0:
        return 0.0
    match = _mirror_sums(gy) - _mirror_sums(gx)
    cols = np.flatnonzero(kept.any(axis=0))
    middle = int(cols[0] + cols[-1])  # twice the middle's x
    reach = max(1, round(AXIS_RANGE * (cols[-1] - cols[0] + 1)))
    best = float(match[max(0, middle - reach) : middle + reach + 1].max()) / energy
    return min(1.0, max(0.0, (best - MIRROR[0]) / (MIRROR[1] - MIRROR[0])))


def _mirror_sums(values: Any) -> Any:
    """For every k: the sum over rows and x of values[y, x] * values[y, k - x] (a self-convolution along rows)."""
    import numpy as np

    width = values.shape[1]
    spectrum = np.fft.rfft(values, n=2 * width, axis=1)  # padded, so the convolution doesn't wrap around
    return np.fft.irfft(spectrum * spectrum, n=2 * width, axis=1).sum(axis=0)


def _object_mask(pixels: Any) -> Any:
    """True where the flood from the picture's edges can't reach: the object and anything on it."""
    import numpy as np

    h, w = pixels.shape[:2]
    edge = np.zeros((h, w), bool)
    edge[[0, -1]] = True
    edge[:, [0, -1]] = True
    tint = pixels - pixels.mean(axis=2, keepdims=True)
    tint -= np.median(tint[edge], axis=0)
    plain = np.einsum("ijk,ijk->ij", tint, tint) < TINT**2
    seeds = np.zeros((h, w), np.uint8)
    rim = pixels[edge]
    seeds[edge] = plain[edge] & (np.abs(rim - np.median(rim, axis=0)).sum(axis=1) < EDGE_MATCH)
    right = (np.abs(pixels[:, 1:] - pixels[:, :-1]).sum(axis=2) < STEP) & plain[:, 1:] & plain[:, :-1]
    down = (np.abs(pixels[1:] - pixels[:-1]).sum(axis=2) < STEP) & plain[1:] & plain[:-1]
    return _spread(seeds, right, down, np.maximum) == 0


def _label(mask: Any) -> Any:
    """Each connected part of `mask` numbered by its first pixel; everything else is h * w."""
    import numpy as np

    h, w = mask.shape
    ids = np.where(mask, np.arange(h * w, dtype=np.int32).reshape(h, w), np.int32(h * w))
    return _spread(ids, mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1], np.minimum)


def _spread(values: Any, right: Any, down: Any, reduce: Any) -> Any:
    """
    Repeats `reduce` (np.maximum or np.minimum) over every run of pixels joined along a row
    (`right`: x to x+1 allowed) or a column (`down`: y to y+1) until nothing changes. Whole runs
    fill at once, so a flood takes a few passes rather than one per pixel.
    """
    import numpy as np

    h, w = values.shape
    rows = np.concatenate([np.ones((h, 1), bool), ~right], axis=1).ravel()  # True where a row run starts
    cols = np.concatenate([np.ones((w, 1), bool), ~down.T], axis=1).ravel()
    row_starts, row_run = np.flatnonzero(rows), np.cumsum(rows) - 1
    col_starts, col_run = np.flatnonzero(cols), np.cumsum(cols) - 1
    while True:
        before = values
        values = reduce.reduceat(values.ravel(), row_starts)[row_run].reshape(h, w)
        values = reduce.reduceat(values.T.ravel(), col_starts)[col_run].reshape(w, h).T
        if (values == before).all():
            return values


def _grow(mask: Any, steps: int) -> Any:
    """`mask` grown by `steps` pixels up, down, left and right."""
    for _ in range(steps):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def handle_job(job: dict, generate: Generator, storage: Storage) -> dict:
    """
    Input: ``{"prompt", "count"?: 1-4, "seed"?, "request_id"?}``. Image i uses seed + i, and comes
    back with its framing ``score`` and ``issues`` (see score_picture).
    """
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
            try:
                framing = score_picture(image)
            except Exception:  # noqa: BLE001 - the score only ranks the pictures; a bug in it mustn't lose them
                traceback.print_exc()
                framing = {}
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            # The seed keeps each image at its own URL, so caches never serve a stale one
            key = f"ai/{request_id}/reference-{seed + i}.png"
            stored = storage.put(key, buf.getvalue(), "image/png")
            images.append({**stored, "seed": seed + i, **framing})
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
