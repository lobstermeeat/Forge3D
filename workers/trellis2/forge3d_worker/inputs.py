"""Validates a job's input and loads its reference image."""

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

from PIL import Image

from .settings import PRESETS, Mode

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InputError(ValueError):
    """The request is malformed. The message is returned to the caller."""


@dataclass(frozen=True)
class Job:
    mode: Mode
    seed: int
    image: Image.Image
    request_id: str

    @property
    def output_key(self) -> str:
        return f"ai/{self.request_id}/{self.mode}.glb"


Fetch = Callable[[str], bytes]


def _allowed_hosts() -> set[str]:
    raw = os.environ.get("ALLOWED_IMAGE_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise InputError("image_url must be an https URL")
    host = parsed.hostname.lower()
    allowed = _allowed_hosts()
    if allowed and host not in allowed:
        raise InputError(f"image_url host {host} is not in ALLOWED_IMAGE_HOSTS")
    # Refuse private, loopback and link-local targets (e.g. cloud metadata endpoints)
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
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
    except (urllib.error.URLError, OSError) as err:
        raise InputError(f"image_url could not be fetched: {err}") from err
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("image is larger than 20 MB")
    return data


def _decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as err:  # PIL raises many types for bad data
        raise InputError("image could not be decoded") from err
    if max(image.size) > MAX_IMAGE_SIDE:
        raise InputError(f"image is larger than {MAX_IMAGE_SIDE}px on a side")
    # Keep transparency: TRELLIS.2 skips background removal when alpha is present
    return image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")


def parse_job(payload: object, fallback_id: str, fetch: Optional[Fetch] = None) -> Job:
    """Turn a RunPod ``input`` payload into a validated ``Job``."""
    if not isinstance(payload, dict):
        raise InputError("input must be an object")

    mode = payload.get("mode", "final")
    if not isinstance(mode, str) or mode not in PRESETS:
        raise InputError(f"mode must be one of {sorted(PRESETS)}")

    seed = payload.get("seed")
    if seed is None:
        seed = random.randrange(2**31)
    elif not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise InputError("seed must be an integer between 0 and 2^31 - 1")

    request_id = payload.get("request_id", fallback_id)
    if not isinstance(request_id, str) or not REQUEST_ID.match(request_id):
        raise InputError("request_id must be 1-64 letters, digits, '-' or '_'")

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

    return Job(mode=mode, seed=seed, image=_decode_image(data), request_id=request_id)
