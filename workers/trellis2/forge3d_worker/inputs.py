"""Validates a job's input and loads its pictures."""

from __future__ import annotations

import base64
import dataclasses
import io
import ipaddress
import math
import os
import random
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image, ImageOps

from .settings import FALLBACK_PIPELINE, MAX_TEXTURES, MODES, PRESETS, TEXTURE_COUNT, Mode

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
# Extra views of the object (the multiview worker draws 6)
MAX_VIEWS = 8
# A view's weight against the others (see settings.MultiView)
MAX_VIEW_WEIGHT = 100.0
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InputError(ValueError):
    """The request is malformed. The message is returned to the caller."""


@dataclass(frozen=True)
class View:
    """Another picture of the object: azimuth 0 is the main picture's side, elevation 0 level with it."""

    image: Image.Image
    azimuth: float  # degrees, 0 to 360
    elevation: float  # degrees, -90 to 90
    weight: float = 1.0


@dataclass(frozen=True)
class Job:
    mode: Mode
    seed: int
    image: Image.Image
    request_id: str
    # Extra views (the other sides); empty for a single-picture job
    views: tuple[View, ...] = ()
    # How many textures a "textures" job makes (1 to MAX_TEXTURES); 0 for the other modes
    count: int = 0
    # The pipeline a "textures" job makes the final's shape with at once: the one the final fell back to
    # ("512"). None makes it as the final job did (the final preset's own), and for the other modes
    pipeline: Optional[str] = None

    @property
    def output_key(self) -> str:
        # The seed keeps each result at its own URL, so caches never serve a stale model
        return f"ai/{self.request_id}/{self.mode}-{self.seed}.glb"

    def texture_key(self, number: int) -> str:
        """Where a "textures" job stores texture ``number`` (1 to count): beside the final's model."""
        # The final's key, with "-texture-<number>" before ".glb"
        final = dataclasses.replace(self, mode="final").output_key
        return f"{final.removesuffix('.glb')}-texture-{number}.glb"


Fetch = Callable[[str], bytes]


def _allowed_hosts() -> set[str]:
    raw = os.environ.get("ALLOWED_IMAGE_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _check_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        port = parsed.port
    except ValueError as err:
        raise InputError("image_url is not a valid URL") from err
    if parsed.scheme != "https" or not parsed.hostname:
        raise InputError("image_url must be an https URL")
    host = parsed.hostname.lower()
    # Only fetch from known hosts (e.g. the R2 bucket). An open fetcher could be pointed at
    # internal addresses, including via DNS that changes between this check and the fetch.
    allowed = _allowed_hosts()
    if not allowed:
        raise InputError("image_url is disabled: set ALLOWED_IMAGE_HOSTS or send image_base64")
    if host not in allowed:
        raise InputError(f"image_url host {host} is not in ALLOWED_IMAGE_HOSTS")
    # Defence in depth: refuse private, loopback and link-local targets
    try:
        infos = socket.getaddrinfo(host, port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise InputError(f"image_url host {host} does not resolve") from err
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise InputError("image_url must point to a public address")


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect could point at an address _check_url would refuse."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise InputError("image_url must not redirect")


_OPENER = urllib.request.build_opener(_NoRedirects)


def fetch_url(url: str) -> bytes:
    _check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "forge3d-trellis2-worker"})
    try:
        with _OPENER.open(request, timeout=30) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except InputError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as err:
        print(f"[forge3d] fetching {url} failed: {err}")
        raise InputError("image_url could not be fetched") from err
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("image is larger than 20 MB")
    return data


def _decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as err:  # PIL raises many types for bad data
        raise InputError("image could not be decoded") from err
    # The header gives the size, so check it before decoding the pixels
    if max(image.size) > MAX_IMAGE_SIDE:
        raise InputError(f"image is larger than {MAX_IMAGE_SIDE}px on a side")
    try:
        image.load()
        # Phone photos store their rotation in EXIF; apply it or the model comes out sideways
        image = ImageOps.exif_transpose(image)
    except Exception as err:
        raise InputError("image could not be decoded") from err
    # Keep transparency: TRELLIS.2 skips background removal when alpha is present
    return image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")


def _read_image(source: dict, fetch: Optional[Fetch]) -> Image.Image:
    """The image of an object with exactly one of ``image_url`` or ``image_base64``."""
    url = source.get("image_url")
    encoded = source.get("image_base64")
    if (url is None) == (encoded is None):
        raise InputError("provide exactly one of image_url or image_base64")
    if url is not None:
        if not isinstance(url, str):
            raise InputError("image_url must be a string")
        data = (fetch or fetch_url)(url)
    else:
        if not isinstance(encoded, str):
            raise InputError("image_base64 must be a string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as err:
            raise InputError("image_base64 is not valid base64") from err
        if len(data) > MAX_IMAGE_BYTES:
            raise InputError("image is larger than 20 MB")
    return _decode_image(data)


def _number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _parse_view(entry: object, fetch: Optional[Fetch]) -> View:
    if not isinstance(entry, dict):
        raise InputError("must be an object")
    azimuth = _number(entry.get("azimuth"))
    if azimuth is None:
        raise InputError("azimuth must be a number of degrees")
    elevation = _number(entry.get("elevation", 0))
    if elevation is None or not -90 <= elevation <= 90:
        raise InputError("elevation must be a number of degrees between -90 and 90")
    weight = _number(entry.get("weight", 1.0))
    if weight is None or not 0 < weight <= MAX_VIEW_WEIGHT:
        raise InputError(f"weight must be a number above 0 and at most {MAX_VIEW_WEIGHT:g}")
    return View(image=_read_image(entry, fetch), azimuth=azimuth % 360, elevation=elevation, weight=weight)


def _parse_views(views: object, fetch: Optional[Fetch]) -> tuple[View, ...]:
    if views is None:
        return ()
    if not isinstance(views, list):
        raise InputError("views must be a list")
    if len(views) > MAX_VIEWS:
        raise InputError(f"views can have at most {MAX_VIEWS} images")
    parsed = []
    for number, entry in enumerate(views):
        try:
            parsed.append(_parse_view(entry, fetch))
        except InputError as err:
            raise InputError(f"views[{number}]: {err}") from err
    return tuple(parsed)


def _texture_count(count: object) -> int:
    if count is None:
        return TEXTURE_COUNT
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_TEXTURES:
        raise InputError(f"count must be an integer from 1 to {MAX_TEXTURES}")
    return count


def _texture_pipeline(pipeline: object) -> Optional[str]:
    """
    A textures job's pipeline, which is the final's: the final preset's own (the same as none: the shape is
    made as the final job made it), or the one a final falls back to (FALLBACK_PIPELINE), for one that did.
    """
    final = PRESETS["final"].pipeline_type
    fallback = FALLBACK_PIPELINE[final]
    if pipeline is None or (isinstance(pipeline, str) and pipeline == final):
        return None
    if not isinstance(pipeline, str) or pipeline != fallback:
        raise InputError(f"pipeline must be the final's: {final!r} or {fallback!r}")
    return pipeline


def parse_job(payload: object, fallback_id: str, fetch: Optional[Fetch] = None) -> Job:
    """Turn a RunPod ``input`` payload into a validated ``Job``."""
    if not isinstance(payload, dict):
        raise InputError("input must be an object")

    mode = payload.get("mode", "final")
    if not isinstance(mode, str) or mode not in MODES:
        raise InputError(f"mode must be one of {sorted(MODES)}")

    seed = payload.get("seed")
    if seed is None and mode == "textures":
        # Another seed would make another shape: texture options are for the final's
        raise InputError("seed is required for textures: send the final's seed")
    if seed is None:
        seed = random.randrange(2**31)
    elif not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise InputError("seed must be an integer between 0 and 2^31 - 1")

    # Only a textures job reads these; the other modes ignore them, as they ignore any field they don't know
    textures = mode == "textures"
    count = _texture_count(payload.get("count")) if textures else 0
    pipeline = _texture_pipeline(payload.get("pipeline")) if textures else None

    request_id = payload.get("request_id", fallback_id)
    if not isinstance(request_id, str) or not REQUEST_ID.match(request_id):
        raise InputError("request_id must be 1-64 letters, digits, '-' or '_'")

    image = _read_image(payload, fetch)
    views = _parse_views(payload.get("views"), fetch)
    return Job(
        mode=mode, seed=seed, image=image, request_id=request_id, views=views, count=count, pipeline=pipeline
    )
