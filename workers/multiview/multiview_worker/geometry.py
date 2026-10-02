"""
MV-Adapter's image+geometry-to-multiview SDXL model ("ig2mv"): six views of a GIVEN mesh, drawn from
a picture of the object, so they can be baked onto that mesh's texture. For experiments: nothing in
production calls it.

It does what MV-Adapter's scripts/inference_ig2mv_sdxl.py (Apache-2.0) does, except that the caller
renders the mesh (upstream renders with nvdiffrast, which is research-only and never installed here)
and every model comes from a local folder, as in generator.py:

1. The picture is cut out with BiRefNet (a picture with its own transparency keeps it) and framed by
   MV-Adapter's preprocess_image: its longer side at 90% of 768 px, centred on mid-gray. This is the
   image-to-multiview worker's reference, and upstream's two scripts prepare it the same way.
2. The mesh comes as its position and normal maps from the six cameras in cameras.py (IG2MV_*):
   front, right, back, left, top, bottom, 768 x 768 each. The control array stacks them per view as
   upstream's script does: channels 0-2 are the position + 0.5, channels 3-5 the unit normal / 2 + 0.5,
   both clamped to [0, 1] and both 0.5 where no surface is (encode_control). Positions and normals are
   in MV-Adapter's world (+Z up, front -Y), the mesh scaled as upstream's load_mesh(rescale=True)
   scales it (mesh_to_world); they are not relative to the camera. MV-Adapter's training data uses the
   same encoding: world positions + 0.5, world normals (its use_camera_space_normal is off), mid-gray
   (0.5) where nothing is.
3. SDXL with the ig2mv adapter, whose attention runs along rows and columns across the views
   (DecoupledMVRowColSelfAttnProcessor2_0), draws the six views with the image-to-multiview worker's
   settings: 30 steps (upstream's script uses 50; its training config evaluates at 30), guidance 3,
   the same negative prompt, the ShiftSNR noise schedule, the fp16-fix VAE, and control and reference
   scales 1. The views come back as drawn: RGB on mid-gray, lighting included. The caller has the masks.

The caller renders in another container, so the control array travels as 12 8-bit PNGs
(pack_control, unpack_control): the six position maps, then the six normal maps, in view order. 8 bits
keep positions within 0.002 world units (1.4 px) and normals within 0.004.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import time
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .cameras import FILL, IG2MV_VIEWS, IMAGE_SIZE
from .generator import (
    ADAPTER_DIR,
    BIREFNET_DIR,
    GUIDANCE,
    NEGATIVE_PROMPT,
    REFERENCE_SCALE,
    STEPS,
    BiRefNetRemover,
    Remover,
    cut_out,
    load_pipeline,
    prepare_reference,
)
from .inputs import DEFAULT_PROMPT, InputError

IG2MV_FILE = "mvadapter_ig2mv_sdxl.safetensors"  # in ADAPTER_DIR, next to the image-to-multiview adapter
CONTROL_SCALE = 1.0
CONTROL_SHAPE = (len(IG2MV_VIEWS), 6, IMAGE_SIZE, IMAGE_SIZE)  # views, channels, height, width
CONTROL_PNGS = 2 * len(IG2MV_VIEWS)
MAX_CONTROL_PNG_BYTES = 4 * 1024 * 1024  # a 768 x 768 RGB PNG is at most about 1.8 MB
# Values this far outside [0, 1] mean maps that were not encoded (raw normals are in [-1, 1])
TOLERANCE = 1e-3


def check_control(control: Any) -> np.ndarray:
    """The control array as float32, after checking its shape and range."""
    array = np.asarray(control, dtype=np.float32)
    if array.shape != CONTROL_SHAPE:
        raise InputError(
            f"control must have shape {CONTROL_SHAPE} (views, channels, height, width), not {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InputError("control has values that are not finite")
    low, high = float(array.min()), float(array.max())
    if low < -TOLERANCE or high > 1 + TOLERANCE:
        raise InputError(
            "control values must be in [0, 1] (position + 0.5, normal / 2 + 0.5), "
            f"not [{low:.3f}, {high:.3f}]"
        )
    return np.clip(array, 0.0, 1.0)


def gltf_to_world(points: Any) -> np.ndarray:
    """Points or normals (n, 3) from glTF's frame (+Y up, front +Z) to MV-Adapter's (+Z up, front -Y)."""
    x, y, z = np.asarray(points, dtype=np.float64).T
    return np.stack([x, -z, y], axis=1)


def mesh_to_world(vertices: Any) -> np.ndarray:
    """
    A glTF mesh's vertices (n, 3) where upstream's load_mesh(path, rescale=True) puts them: scaled so
    that the largest |coordinate| is 0.5, without moving the origin, then turned to +Z up
    (gltf_to_world). The front camera then sees the glTF's front (+Z), where TRELLIS.2 puts it.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a non-empty (n, 3) array of finite numbers")
    extent = float(np.abs(vertices).max())
    if extent == 0:
        raise ValueError("every vertex is at the origin")
    return gltf_to_world(vertices * (0.5 / extent))


def encode_control(positions: Any, normals: Any, masks: Any) -> np.ndarray:
    """
    The control array from the six views' maps: `positions` and `normals` (6, H, W, 3) in MV-Adapter's
    world (mesh_to_world, gltf_to_world), `masks` (6, H, W) true where the mesh is. As upstream: the
    normals made unit length, position + 0.5 and normal / 2 + 0.5 clamped to [0, 1], and both 0 before
    encoding (so 0.5) where there is no surface. Returns (6, 6, H, W) float32.
    """
    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    views = len(IG2MV_VIEWS)
    if positions.ndim != 4 or positions.shape[0] != views or positions.shape[3] != 3:
        raise ValueError(f"positions must be ({views}, H, W, 3), not {positions.shape}")
    if normals.shape != positions.shape or masks.shape != positions.shape[:3]:
        raise ValueError(
            f"normals {normals.shape} and masks {masks.shape} must match positions {positions.shape}"
        )
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(length, 1e-12)
    inside = masks[..., None]
    position = np.clip(np.where(inside, positions, 0.0) + 0.5, 0.0, 1.0)
    normal = np.clip(np.where(inside, normals, 0.0) / 2 + 0.5, 0.0, 1.0)
    control = np.concatenate([position, normal], axis=-1).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(control, dtype=np.float32)


def pack_control(control: Any) -> list[str]:
    """
    The control array as 12 base64 PNGs (8-bit RGB), as a job's "control_pngs": the six position maps
    (channels 0-2), then the six normal maps (channels 3-5), each in IG2MV_VIEWS order.
    """
    array = check_control(control)
    pngs = []
    for first in (0, 3):
        for view in array:
            rgb = np.round(view[first : first + 3].transpose(1, 2, 0) * 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(rgb, "RGB").save(buffer, format="PNG")
            pngs.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return pngs


def unpack_control(pngs: Sequence[Union[str, bytes]]) -> np.ndarray:
    """pack_control's PNGs (base64 text, or the files' bytes) back to the (6, 6, 768, 768) float32 array."""
    if not isinstance(pngs, (list, tuple)) or len(pngs) != CONTROL_PNGS:
        raise InputError(
            f"control_pngs must be a list of {CONTROL_PNGS} PNGs: six position maps, then six normal maps"
        )
    control = np.empty(CONTROL_SHAPE, dtype=np.float32)
    for index, png in enumerate(pngs):
        if isinstance(png, str):
            try:
                data = base64.b64decode(png, validate=True)
            except (binascii.Error, ValueError) as err:
                raise InputError(f"control PNG {index} is not valid base64") from err
        elif isinstance(png, bytes):
            data = png
        else:
            raise InputError(f"control PNG {index} must be a base64 string")
        if len(data) > MAX_CONTROL_PNG_BYTES:
            raise InputError(f"control PNG {index} is larger than {MAX_CONTROL_PNG_BYTES // (1024 * 1024)} MB")
        try:
            image = Image.open(io.BytesIO(data))
        except Exception as err:  # PIL raises many types for bad data
            raise InputError(f"control PNG {index} could not be decoded") from err
        if image.format != "PNG" or image.mode != "RGB":
            raise InputError(f"control PNG {index} must be an 8-bit RGB PNG, not {image.format} {image.mode}")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            width, height = image.size
            raise InputError(f"control PNG {index} must be {IMAGE_SIZE} x {IMAGE_SIZE}, not {width} x {height}")
        try:
            pixels = np.asarray(image, dtype=np.float32)
        except Exception as err:
            raise InputError(f"control PNG {index} could not be decoded") from err
        view, first = index % len(IG2MV_VIEWS), 3 * (index // len(IG2MV_VIEWS))
        control[view, first : first + 3] = pixels.transpose(2, 0, 1) / 255.0
    return control


class GeometryViewGenerator:
    """
    Loads SDXL with the ig2mv adapter once (about 10 GB of weights); draw_views turns a picture and a
    mesh's control maps into six views of that mesh.
    """

    def __init__(self, models_root: str, device: str = "cuda", remover: Optional[Remover] = None) -> None:
        adapter = os.path.join(models_root, ADAPTER_DIR, IG2MV_FILE)
        if not os.path.isfile(adapter):
            # A volume filled before this adapter joined the multiview download has the rest without it
            raise FileNotFoundError(f"{adapter} is missing: download the multiview weights again")
        from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0

        self.pipe = load_pipeline(
            models_root, IG2MV_FILE, len(IG2MV_VIEWS), device, DecoupledMVRowColSelfAttnProcessor2_0
        )
        self.device = device
        self.remover = remover or BiRefNetRemover(os.path.join(models_root, BIREFNET_DIR), device)
        # The last call's prepared reference and timings, for experiments and debugging
        self.last_reference: Optional[Image.Image] = None
        self.last_timings: dict[str, float] = {}

    def draw_views(
        self,
        reference: Image.Image,
        control: Any,
        prompt: str = DEFAULT_PROMPT,
        seed: int = 0,
        *,
        steps: int = STEPS,
        guidance: float = GUIDANCE,
        reference_scale: float = REFERENCE_SCALE,
        control_scale: float = CONTROL_SCALE,
        fill: float = FILL,
    ) -> list[Image.Image]:
        """
        Six 768 x 768 RGB views of the mesh, in IG2MV_VIEWS order. `reference` is the picture (RGB, or
        RGBA with its own cutout); `control` the (6, 6, 768, 768) array of floats in [0, 1] that
        encode_control makes (position RGB, then normal RGB, per view).
        """
        import torch

        control = check_control(control)
        started = time.perf_counter()
        prepared = prepare_reference(cut_out(reference, self.remover), fill=fill)
        prepared_at = time.perf_counter()
        views = self.pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=len(IG2MV_VIEWS),
            control_image=torch.from_numpy(control).to(self.device),
            control_conditioning_scale=control_scale,
            reference_image=prepared,
            reference_conditioning_scale=reference_scale,
            negative_prompt=NEGATIVE_PROMPT,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images
        if len(views) != len(IG2MV_VIEWS):
            raise RuntimeError(f"expected {len(IG2MV_VIEWS)} views, got {len(views)}")
        self.last_reference = prepared
        self.last_timings = {
            "cutout_s": round(prepared_at - started, 3),
            "views_s": round(time.perf_counter() - prepared_at, 3),
        }
        return [view.convert("RGB") for view in views]
