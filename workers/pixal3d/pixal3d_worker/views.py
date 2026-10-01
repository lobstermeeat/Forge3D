"""
Multi-view input: the views a job brings, checked, framed and turned into Pixal3D's view bundle.

A job's optional ``"views"`` are pictures of the same object from around it, as the multiview worker
makes them: ``[{"image_url" | "image_base64", "azimuth", "elevation"}]``, square RGBA cutouts from
orthographic cameras that share one scale (``"camera"``, optional, says how wide their frame is; by
default MV-Adapter's, +-0.55). Azimuth 0 at elevation 0 is the picture's own view, and positive
azimuth goes towards the picture's right (see multiview_worker/cameras.py).

Pixal3D needs a main view first, the front, and every view's camera. The main view is either the
multiview worker's redrawn 0-degree view or the picture itself, placed where that view has the object
(``MAIN_*``): same centre, same longer side, so the two describe the same camera.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from forge3d_worker.inputs import MAX_IMAGE_BYTES, Fetch, InputError, _decode_image, fetch_url

from . import cameras

MAX_VIEWS = 8
# Which picture is the main view: the multiview worker's redrawn front, or the user's own picture
MAIN_VIEW = "view"
MAIN_PICTURE = "picture"
MAIN_CHOICES = (MAIN_VIEW, MAIN_PICTURE)
# A pixel belongs to the object above this alpha (MV-Adapter frames on alpha > 0; a soft matte's
# faint halo shouldn't move the frame, so a little more is asked)
OBJECT_ALPHA = 0.5
# MV-Adapter's framing of the picture when no 0-degree view says otherwise
DEFAULT_FILL = 0.9


@dataclass(frozen=True)
class View:
    image: Image.Image
    azimuth: float  # degrees, [0, 360)
    elevation: float  # degrees

    @property
    def is_front(self) -> bool:
        return self.azimuth == 0.0 and self.elevation == 0.0


@dataclass(frozen=True)
class ViewCamera:
    """The views' shared orthographic camera: their frame spans +-half_extent world units."""

    half_extent: float = cameras.MV_ADAPTER_HALF_WIDTH


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name} must be a number")
    return float(value)


def parse_camera(raw: object) -> ViewCamera:
    """The job's optional ``"camera"`` (the multiview worker's own, or just its ``half_extent``)."""
    if raw is None:
        return ViewCamera()
    if not isinstance(raw, dict):
        raise InputError("camera must be an object")
    kind = raw.get("type", "orthographic")
    if kind != "orthographic":
        raise InputError("camera type must be orthographic")
    half_extent = _number(raw.get("half_extent", cameras.MV_ADAPTER_HALF_WIDTH), "camera half_extent")
    if not 0.1 <= half_extent <= 10:
        raise InputError("camera half_extent must be between 0.1 and 10")
    return ViewCamera(half_extent=half_extent)


def parse_views(raw: object, fetch: Optional[Fetch] = None) -> list[View]:
    """Checks and decodes a job's ``"views"``. Raises InputError with a message for the caller."""
    if not isinstance(raw, list) or not raw:
        raise InputError("views must be a non-empty list")
    if len(raw) > MAX_VIEWS:
        raise InputError(f"at most {MAX_VIEWS} views")
    views: list[View] = []
    seen: set[tuple[float, float]] = set()
    for number, item in enumerate(raw, 1):
        label = f"views[{number - 1}]"
        if not isinstance(item, dict):
            raise InputError(f"{label} must be an object")
        azimuth = _number(item.get("azimuth"), f"{label}.azimuth") % 360.0
        elevation = _number(item.get("elevation", 0), f"{label}.elevation")
        if abs(elevation) > cameras.MAX_ELEVATION_DEG:
            raise InputError(f"{label}.elevation must be within +-{cameras.MAX_ELEVATION_DEG:g} degrees")
        key = (round(azimuth, 3), round(elevation, 3))
        if key in seen:
            raise InputError(f"{label} repeats azimuth {azimuth:g}, elevation {elevation:g}")
        seen.add(key)
        url, encoded = item.get("image_url"), item.get("image_base64")
        if (url is None) == (encoded is None):
            raise InputError(f"{label} needs exactly one of image_url or image_base64")
        if url is not None:
            if not isinstance(url, str):
                raise InputError(f"{label}.image_url must be a string")
            data = (fetch or fetch_url)(url)
        else:
            if not isinstance(encoded, str):
                raise InputError(f"{label}.image_base64 must be a string")
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as err:
                raise InputError(f"{label}.image_base64 is not valid base64") from err
            if len(data) > MAX_IMAGE_BYTES:
                raise InputError(f"{label} is larger than 20 MB")
        image = _decode_image(data)
        if image.width != image.height:
            raise InputError(f"{label} must be square (the views' cameras frame a square)")
        views.append(View(image=image, azimuth=azimuth, elevation=elevation))
    return views


def object_box(alpha: np.ndarray, threshold: float = OBJECT_ALPHA) -> Optional[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1), exclusive ends, of the pixels with alpha (0..1) above the threshold."""
    ys, xs = np.nonzero(np.asarray(alpha) > threshold)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def alpha_of(image: Image.Image) -> np.ndarray:
    """An RGBA picture's alpha as 0..1 floats."""
    return np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0


def has_cutout(image: Image.Image) -> bool:
    """Whether the picture brings its own matte (alpha that isn't all opaque), as upstream tests it."""
    return image.mode == "RGBA" and bool(np.any(np.asarray(image.getchannel("A")) < 255))


def place_like(
    cutout: Image.Image, size: int, reference: Optional[Image.Image] = None, fill: float = DEFAULT_FILL
) -> Image.Image:
    """
    The object of an RGBA cutout moved onto a transparent ``size`` x ``size`` canvas: centred on the
    reference view's object with the same longer side, or (no reference) centred with its longer side
    ``fill`` of the canvas, as MV-Adapter frames the picture it redraws.
    """
    box = object_box(alpha_of(cutout))
    if box is None:
        raise InputError("no object found in the image: use one object on a plain background")
    target = object_box(alpha_of(reference)) if reference is not None else None
    if reference is not None and reference.size != (size, size):
        raise ValueError("the reference view must be the canvas size")
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if target is not None:
        tx0, ty0, tx1, ty1 = target
        side = max(tx1 - tx0, ty1 - ty0)
        centre = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)
    else:
        side = fill * size
        centre = (size / 2, size / 2)
    scale = side / max(width, height)
    resized = cutout.crop(box).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    left = round(centre[0] - resized.width / 2)
    top = round(centre[1] - resized.height / 2)
    canvas.alpha_composite(resized, (max(left, 0), max(top, 0)), (max(-left, 0), max(-top, 0)))
    return canvas


def order(views: Sequence[View], azimuths: Optional[Sequence[float]] = None) -> list[View]:
    """
    The views Pixal3D gets, the front first: the ones at the given azimuths (all of them when None),
    then by azimuth. A missing front stays missing (the picture becomes the main view then).
    """
    chosen = list(views)
    if azimuths is not None:
        wanted = {float(a) % 360.0 for a in azimuths}
        chosen = [view for view in chosen if view.azimuth in wanted]
    return sorted(chosen, key=lambda view: (not view.is_front, view.azimuth, view.elevation))


def touches_edge(image: Image.Image, margin: int = 1) -> bool:
    """Whether the object reaches the frame's edge (cut off: that view's camera can't see all of it)."""
    box = object_box(alpha_of(image))
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return x0 < margin or y0 < margin or x1 > image.width - margin or y1 > image.height - margin


def cond_tensor(image: Image.Image, size: int):
    """
    One view as Pixal3D's condition: LANCZOS-resized to ``size``, colour premultiplied by alpha so the
    background is black (``inference_mv.to_cond_tensor``, how training read its views).
    """
    import torch

    image = image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    alpha = torch.from_numpy(np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0)
    rgb = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return rgb * alpha.unsqueeze(0)


def bundle(
    images: Sequence[Image.Image],
    angles: Sequence[tuple[float, float]],
    camera: ViewCamera,
    sizes: Sequence[int] = (512, 1024),
    fov_deg: float = cameras.NEAR_ORTHO_FOV_DEG,
) -> dict:
    """
    Pixal3D's multi-view bundle (``Pixal3DMVImageTo3DPipeline.run_mv``): the views as premultiplied
    tensors at each stage's size, with their cameras. ``angles`` are (azimuth, elevation) per image,
    the first the front.
    """
    import torch

    if len(images) != len(angles):
        raise ValueError("one (azimuth, elevation) per image")
    rig = cameras.rig(angles, half_width=camera.half_extent, fov_deg=fov_deg)
    return {
        "images": {
            size: torch.stack([cond_tensor(image, size) for image in images])[None] for size in sorted(set(sizes))
        },
        "camera_angle_x": torch.tensor(rig["camera_angle_x"], dtype=torch.float32),
        "camera_distance": torch.tensor(rig["camera_distance"], dtype=torch.float32),
        "transform_matrix": torch.tensor(rig["transform_matrix"], dtype=torch.float32),
        "mesh_scale": 1.0,
        "view_names": [f"azim{a:03.0f}_elev{e:+03.0f}" for a, e in angles],
    }
