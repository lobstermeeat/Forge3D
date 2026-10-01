"""Validates a multiview job's input and loads its picture (the same rules as the TRELLIS.2 worker's)."""

from __future__ import annotations

import base64
import io
import ipaddress
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

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_PROMPT = 500
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# MV-Adapter's default caption (its inference script and demo); a short description of the object
# can be sent instead as "prompt"
DEFAULT_PROMPT = "high quality"


class InputError(ValueError):
    """The request is malformed. The message is returned to the caller."""


@dataclass(frozen=True)
class Job:
    image: Image.Image
    seed: int
    request_id: str
    prompt: str


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
    request = urllib.request.Request(url, headers={"User-Agent": "orainge-multiview-worker"})
    try:
        with _OPENER.open(request, timeout=30) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except InputError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as err:
        print(f"[orainge] fetching {url} failed: {err}")
        raise InputError("image_url could not be fetched") from err
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("image is larger than 20 MB")
    return data


def decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as err:  # PIL raises many types for bad data
        raise InputError("image could not be decoded") from err
    # The header gives the size, so check it before decoding the pixels
    if max(image.size) > MAX_IMAGE_SIDE:
        raise InputError(f"image is larger than {MAX_IMAGE_SIDE}px on a side")
    try:
        image.load()
        # Phone photos store their rotation in EXIF; apply it or the views come out sideways
        image = ImageOps.exif_transpose(image)
    except Exception as err:
        raise InputError("image could not be decoded") from err
    # Keep transparency: a picture with its own cutout skips background removal
    return image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")


def parse_job(payload: object, fallback_id: str, fetch: Optional[Fetch] = None) -> Job:
    """Turn a job's ``input`` into a validated ``Job``."""
    if not isinstance(payload, dict):
        raise InputError("input must be an object")

    seed = payload.get("seed")
    if seed is None:
        seed = random.randrange(2**31)
    elif not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise InputError("seed must be an integer between 0 and 2^31 - 1")

    request_id = payload.get("request_id", fallback_id)
    if not isinstance(request_id, str) or not REQUEST_ID.match(request_id):
        raise InputError("request_id must be 1-64 letters, digits, '-' or '_'")

    prompt = payload.get("prompt")
    if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
        prompt = DEFAULT_PROMPT
    elif not isinstance(prompt, str):
        raise InputError("prompt must be a string")
    elif len(prompt) > MAX_PROMPT:
        raise InputError(f"prompt is longer than {MAX_PROMPT} characters")

    url = payload.get("image_url")
    encoded = payload.get("image_base64")
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

    return Job(image=decode_image(data), seed=seed, request_id=request_id, prompt=prompt.strip())
