"""
The multi-view painter (Phase 8): photo-real views of the model, painted by an image-editing model over
renders of its own texture, baked back into the texture.

TRELLIS.2 gets the shape right but its texture is soft and blotchy, copies the picture's reflections and
shadows, and makes up the sides the picture doesn't show. An image-editing model can turn a render of the
model into a clean product photo of the pictured object without moving anything: the render fixes the
outline and where each part is, the picture says what the object looks like. This module is everything
around that model, on the mesh as ``to_glb`` makes it (after ``unpremultiply``, before the picture's
projection, which production then runs on top as always):

1. The picture's camera, from the projection run on a stand-in for the mesh (``projection.project_picture``
   never touches the mesh itself here). The picture painted on that stand-in is what the renders show, so
   every view starts from the picture where it reaches; the projection's weights say where that is.
2. Cameras round the object (``ring``): ``around`` of them at ``elevation``, the first at the picture's
   azimuth, then a top and a bottom view. Each is a perspective camera framed tight on the object
   (``frame``): about a megapixel, sides a multiple of 16, the shape the editing model takes.
3. View by view, nearest the picture first (``paint_views``): the current texture is rendered (``render``:
   base colour lit from the camera, on a plain light grey); the editing model paints it, with the picture
   and the painted view nearest to it as references; the painted object's outline is checked against the
   render's (``object_mask``, then ``align``: the best small shift and scale, and the silhouette IoU after
   it) and so are its edges (``novelty``: what it drew that the render doesn't have, such as a second front
   on a plain back); a view under ``min_iou`` or over ``max_novelty`` is painted again with another seed,
   then left out (and a bottom view whose render is dark isn't painted at all). Its brightness and
   saturation are brought to what the picture and the views before it already say where they overlap
   (``tone``); and it is baked (``view_samples``: depth-tested visibility, a power of the cosine, fades at
   silhouettes and depth edges), so the next render shows it.
4. Every view's own colours brought to agree by a tone solved for all of them together (``joint_tone``:
   brightness and saturation only, so a grey stays grey; held to the picture's own colour where the
   picture saw the surface well), then blended into the original texture with each texel taken mostly
   from its best view (``select_weights``) (``compose``: the change carried into the gutters); also
   robustly (``robust_colour``: only the views near the weighted median luminance count at a texel, so a
   highlight or a ghost one view drew is left out). The caller exports from there as production does:
   the picture's projection, smoothed normals, glass, gltfpack.

The editing model is the caller's: ``paint(render, picture, neighbour, seed, view) -> image``, where
``view`` says which side it shows (``describe``) and the picture's main colours (``main_colours``). The module runs
on the CPU as well as the GPU (the tests use a fake painter).
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import projection, uv_raster
from .mvtexture import _fade, _into_gutters
from .projection import rasterize_depth, rasterize_faces

# --- Cameras ------------------------------------------------------------------------------------------
# Perspective strength (projection's: 1 / the camera's distance in bounding radii): 0.3 is about 35
# degrees across the bounding sphere, a product photographer's longish lens
PERSPECTIVE = 0.3
ELEVATION = 15.0  # degrees above the horizon, for the cameras round the object
AROUND = 8
TOP_ELEVATION = 89.0  # the top and bottom views (exactly 90 works too; this keeps the azimuth meaningful)
# Space round the object in a view, each side, as a share of its larger extent there
MARGIN = 0.08
# A view's area in pixels at most, its sides multiples of MULTIPLE (FLUX.2's latent patches are 16 pixels;
# its references are resized to a megapixel at most), and its aspect within MAX_ASPECT either way
PIXELS = 1024 * 1024
MULTIPLE = 16
MAX_ASPECT = 2.5
# The plain background of renders and references (sRGB)
BACKGROUND = (0.92, 0.92, 0.92)

# --- Renders ------------------------------------------------------------------------------------------
# Light from the camera: ambient + (1 - ambient) * cos(normal, view). Shape shows; colours stay readable
AMBIENT = 0.55
SUPERSAMPLE = 2  # renders are drawn this many times larger each way and averaged down (smooth outlines)
# The picture as a reference: its object cut out on the background, cropped with this margin, at most
# this many pixels
REFERENCE_MARGIN = 0.1
REFERENCE_PIXELS = 768 * 768

# --- Checks -------------------------------------------------------------------------------------------
# A painted view is used when its object's outline, after the best small shift and scale, overlaps the
# render's by at least MIN_IOU (intersection over union). Shifts up to MAX_SHIFT of the view's longer side
# and scales up to exp(MAX_LOG_SCALE) either way are tried
MIN_IOU = 0.9
MAX_SHIFT = 0.04
MAX_LOG_SCALE = 0.08
ALIGN_SIZE = 256  # the outlines are compared at this size (longer side)
# The painted object: everything not reached from the image's border through background-coloured pixels
# (within BACKGROUND_TOLERANCE per sRGB channel of the border's median), nor, outside the render's object,
# through shadow-coloured ones (grey: channels within SHADOW_CHROMA of each other; at least SHADOW_LUMA of
# the background's brightness), so a soft shadow on the floor isn't counted as the object
BACKGROUND_TOLERANCE = 0.07
SHADOW_CHROMA = 0.05
SHADOW_LUMA = 0.45

# What the painted view adds: the share of its edge energy (the Sobel magnitude of log luminance above EDGE,
# at NOVELTY_SIZE pixels on the longer side) that lies more than EDGE_REACH pixels from any edge of the
# render. A view over MAX_NOVELTY drew something the render doesn't have (run 2: the arcade machine's plain
# back painted as a second front with a screen and a door, 0.43; the top of a car painted on its underside,
# 0.68; good views 0.00-0.08)
NOVELTY_SIZE = 256
EDGE = 0.5
EDGE_REACH = 2
MAX_NOVELTY = 0.2
# Robust blend: where views disagree, only those within ROBUST_TOLERANCE (log luminance, about 30%) of the
# weighted median count (a highlight or a ghost in one view doesn't go in)
ROBUST_TOLERANCE = 0.25

# --- Bake ---------------------------------------------------------------------------------------------
# As mvtexture.bake_views: a view's weight at a texel is cos ** COS_POWER (fading to nothing between
# MIN_COS and MIN_COS + 0.1), times fades over FEATHER pixels from its usable pixels' edge and from depth
# edges (EDGE_JUMP pixels' worth of depth); a texel is visible when within DEPTH_BIAS pixels' worth of
# depth of the nearest surface (more on slopes). A higher power than mvtexture's 3: each texel mostly
# takes one view, so a detail two views put a pixel apart isn't drawn twice
COS_POWER = 4.0
MIN_COS = 0.25
FEATHER = 6.0
EDGE_JUMP = 8.0
DEPTH_BIAS = 2.0
# Views whose weights add up to this replace a texel's colour in full; below it the old colour shows through
FULL_WEIGHT = 0.25
# Renders keep the picture's paint where the projection used the picture this much (its detail weight):
# fully above the upper end, not at all below the lower
PROTECT = (0.3, 0.7)
# Colour match: a view's colour is scaled per channel (linear light) by the weighted median ratio of what
# is already established (the picture's paint, earlier views) to it, where both see the surface and the
# established colour is CONFIDENT; at most MAX_GAIN either way, from at least GAIN_TEXELS texels. Run 2: the
# editing model painted the BMW a brighter blue than the picture's, by more than 1.4 in red and green
MAX_GAIN = 2.0
GAIN_TEXELS = 500
CONFIDENT = 0.5
GAIN_SAMPLES = 400_000
# Joint colour match (run 3: the BMW's left side came out a lighter blue than its back, a seam where they met).
# After every view is in, per-view per-channel gains are solved together, so that the views agree with each other
# where two of them see the same texel and with the picture's paint where it is CONFIDENT: weighted least squares
# on log colour, with JOINT_ROBUST more rounds that all but ignore texels disagreeing by much more than
# JOINT_SCALE (log units; Cauchy weights: a decal one view drew and another didn't), each gain pulled towards 1
# by JOINT_PRIOR of its weight
JOINT_SCALE = 0.15
JOINT_ROBUST = 4
JOINT_PRIOR = 0.01
# Tone instead of per-channel gains (run 5: anchored to the picture's paint, which in the projection's "detail
# only" mode is TRELLIS.2's own pale colour with the picture's detail on it, the per-channel gains turned the
# BMW's grey wheels bronze while pulling its blue towards that paint). A view's colour c changes only in
# brightness and saturation, c -> g (Y + s (c - Y)) with Y its luminance, so a grey stays grey whatever the
# paint colours do; and the anchor is the picture's own colour where the picture saw the surface well, not the
# paint. Saturation counts only where both colours have a relative chroma (|c - Y| / Y) of at least MIN_CHROMA
# and are brighter than CHROMA_LUMA, and changes at most MAX_SATURATION either way
MIN_CHROMA = 0.08
CHROMA_LUMA = 0.02
MAX_SATURATION = 1.5
# The final blend takes each texel mostly from its best view: the views' weights sharpened by a softmax over their
# logs at SELECT (weights to the power 1 / SELECT, normalised). Run 5 in the lab: blended by cos^4 alone, the
# BMW's wheel spokes and grille slats came out doubled where two views drew them a pixel or two apart; at 0.1
# they keep the sharpness of one view, and the joint tone keeps the views' colours together where they meet
SELECT = 0.1
# A bottom view whose render is this dark (median sRGB luminance over the object) isn't painted: the editing model
# turned the dark underside of two cars into a second top, roof and windows (run 4 and 5), and a dark underside
# has nothing to gain from it
DARK_BOTTOM = 0.12


def _device(device: Optional[Any]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# --- Cameras ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """
    A perspective camera on the normalised model (bounding sphere of radius 1 round its box's centre), in
    projection's terms: ``azimuth`` turns about +Y (0 looks at the model's +Z side, 90 at +X),
    ``elevation`` is above the horizon, no roll. Its image shows the window of half sizes ``half`` round
    ``centre`` in the picture plane through the origin, ``size`` (width, height) pixels.
    """

    name: str
    azimuth: float
    elevation: float
    perspective: float
    centre: tuple[float, float]
    half: tuple[float, float]
    size: tuple[int, int]

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def params(self, device: Any = None) -> torch.Tensor:
        """projection's (1, 7) view parameters."""
        return torch.tensor(
            [[self.azimuth, self.elevation, 0.0, self.perspective, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )

    def direction(self) -> np.ndarray:
        """Unit vector from the model's centre towards the camera."""
        return view_direction(self.azimuth, self.elevation)

    def pixel(self) -> float:
        """Normalised model units per pixel in the window's plane."""
        return 2 * self.half[0] / self.width

    def project(self, points: torch.Tensor, scale: int = 1):
        """
        Points (N, 3) as the camera sees them: pixel coordinates (N, 2) (x right, y down; pixel (r, c)
        centred at (c + 0.5, r + 0.5)) in an image ``scale`` times the camera's size, depth (N,; larger is
        farther) and the perspective factor (N,) that rasterize_depth interpolates with.
        """
        x, y, depth, s = projection.project(points, self.params(points.device))
        col = ((x[0] - self.centre[0]) / self.half[0] + 1) * (self.width / 2) * scale
        row = ((self.centre[1] - y[0]) / self.half[1] + 1) * (self.height / 2) * scale
        return torch.stack([col, row], -1), depth[0], s[0]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "azimuth": round(self.azimuth % 360, 1),
            "elevation": round(self.elevation, 1),
            "size": list(self.size),
        }


def view_direction(azimuth: float, elevation: float) -> np.ndarray:
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])


def frame(
    verts: torch.Tensor,
    name: str,
    azimuth: float,
    elevation: float,
    *,
    perspective: float = PERSPECTIVE,
    margin: float = MARGIN,
    pixels: int = PIXELS,
    multiple: int = MULTIPLE,
    max_aspect: float = MAX_ASPECT,
) -> Camera:
    """
    A camera looking from ``azimuth`` and ``elevation`` with the normalised mesh ``verts`` framed tight:
    ``margin`` of its larger extent on each side, the window widened to an aspect within ``max_aspect``,
    sides multiples of ``multiple`` and at most ``pixels`` in all, square pixels.
    """
    params = torch.tensor([[azimuth, elevation, 0.0, perspective, 0.0, 0.0, 0.0]], dtype=torch.float32, device=verts.device)
    x, y, _, _ = projection.project(verts, params)
    xlo, xhi, ylo, yhi = (float(v) for v in (x.min(), x.max(), y.min(), y.max()))
    extent = max(xhi - xlo, yhi - ylo, 1e-6)
    hx = (xhi - xlo) / 2 + margin * extent
    hy = (yhi - ylo) / 2 + margin * extent
    if hx / hy > max_aspect:
        hy = hx / max_aspect
    elif hy / hx > max_aspect:
        hx = hy / max_aspect
    aspect = hx / hy
    width = max(multiple, int(math.sqrt(pixels * aspect) // multiple) * multiple)
    height = max(multiple, int(math.sqrt(pixels / aspect) // multiple) * multiple)
    # Square pixels: the window widened to the image's aspect
    if width / height > hx / hy:
        hx = hy * width / height
    else:
        hy = hx * height / width
    return Camera(
        name=name,
        azimuth=float(azimuth) % 360,
        elevation=float(elevation),
        perspective=float(perspective),
        centre=((xlo + xhi) / 2, (ylo + yhi) / 2),
        half=(hx, hy),
        size=(width, height),
    )


def ring(
    verts: torch.Tensor,
    azimuth: float = 0.0,
    *,
    elevation: float = ELEVATION,
    around: int = AROUND,
    top: bool = True,
    bottom: bool = True,
    **framing: Any,
) -> list[Camera]:
    """
    ``around`` cameras evenly round the object at ``elevation``, the first at ``azimuth``, then a top and
    a bottom view (facing the same way as the first), each framed by ``frame``. Named by their azimuth
    from the first ("a000", "a045", ...), "top" and "bottom".
    """
    cameras = []
    for k in range(around):
        offset = k * 360.0 / around
        cameras.append(frame(verts, f"a{round(offset):03d}", azimuth + offset, elevation, **framing))
    if top:
        cameras.append(frame(verts, "top", azimuth, TOP_ELEVATION, **framing))
    if bottom:
        cameras.append(frame(verts, "bottom", azimuth, -TOP_ELEVATION, **framing))
    return cameras


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions."""
    cos = float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def by_angle(cameras: Sequence[Camera], azimuth: float, elevation: float) -> list[Camera]:
    """The cameras nearest the direction (azimuth, elevation) first; ties keep their order."""
    target = view_direction(azimuth, elevation)
    return sorted(cameras, key=lambda camera: round(angle_between(camera.direction(), target), 6))


# --- The mesh -----------------------------------------------------------------------------------------


@dataclass
class Geometry:
    """The mesh normalised as projection normalises it, on one device."""

    verts: torch.Tensor  # (V, 3): bounding sphere of radius 1 round the box's centre
    faces: torch.Tensor  # (F, 3) long
    uv: torch.Tensor  # (V, 2), trimesh's convention (v up)
    normals: torch.Tensor  # (V, 3) unit, welded across UV seams, facing out
    face_normals: torch.Tensor  # (F, 3) unit, facing out
    flipped: bool = False  # the triangles wind inwards, so the normals were turned round


def geometry(mesh: Any, device: Optional[Any] = None) -> Geometry:
    """``mesh`` (to_glb's trimesh: vertices, faces, visual.uv) normalised, with smooth and face normals."""
    device = _device(device)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3 or not np.isfinite(verts).all():
        raise ValueError("the mesh has no usable vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("the mesh's faces don't match its vertices")
    if uv is None or np.asarray(uv).shape != (len(verts), 2) or not np.isfinite(np.asarray(uv, dtype=np.float64)).all():
        raise ValueError("the mesh has no UVs matching its vertices")
    centre = (verts.min(0) + verts.max(0)) / 2
    radius = float(np.linalg.norm(verts - centre, axis=1).max())
    if radius <= 0:
        raise ValueError("the mesh is a point")
    v = torch.tensor((verts - centre) / radius, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device)
    a, b, c = (v[f[:, i]] for i in range(3))
    face_normals = torch.cross(b - a, c - a, dim=-1)
    face_normals = face_normals / face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    geom = Geometry(
        verts=v,
        faces=f,
        uv=torch.tensor(np.asarray(uv, dtype=np.float64), dtype=torch.float32, device=device),
        normals=projection._welded_normals(v, f),
        face_normals=face_normals,
    )
    _orient(geom)
    return geom


def _orient(geom: Geometry, size: int = 96) -> None:
    """
    Turns the normals round when the surface six cameras see (front, back, sides, top, bottom) mostly
    faces away from them: a mesh whose triangles wind inwards.
    """
    facing = []
    for azimuth, elevation in ((0, 0), (90, 0), (180, 0), (270, 0), (0, 89), (0, -89)):
        params = torch.tensor([[azimuth, elevation, 0.0, 0.0, 0, 0, 0]], dtype=torch.float32, device=geom.verts.device)
        x, y, depth, s = projection.project(geom.verts, params)
        xy = torch.stack([(x[0] + 1.05) / 2.1 * size, (1.05 - y[0]) / 2.1 * size], -1)
        zbuf = rasterize_depth(xy, depth[0], s[0], geom.faces, size, size)
        face, _ = rasterize_faces(xy, depth[0], s[0], geom.faces, zbuf)
        _, _, back = projection.view_axes(params)
        facing.append(geom.face_normals[face[face >= 0]] @ back[0])
    facing = torch.cat(facing)
    if facing.numel() and float(facing.median()) < 0:
        geom.normals = -geom.normals
        geom.face_normals = -geom.face_normals
        geom.flipped = True


@dataclass
class Texels:
    """The texels the mesh covers, with the surface point and normals each one paints."""

    flat: torch.Tensor  # (N,) long: index into the texture, row-major, row 0 at the top (v = 1)
    covered: torch.Tensor  # (H, W) bool
    points: torch.Tensor  # (N, 3) normalised
    normals: torch.Tensor  # (N, 3) unit, smooth
    face_normals: torch.Tensor  # (N, 3) unit: the texel's triangle's
    size: tuple[int, int]  # (H, W)


def texels(geom: Geometry, size: tuple[int, int]) -> Texels:
    """Every covered texel of an (H, W) texture: the mesh rasterised in UV space, as mvtexture does."""
    tex_h, tex_w = size
    uv_clip = torch.cat([geom.uv * 2 - 1, torch.zeros_like(geom.uv[:, :1]), torch.ones_like(geom.uv[:, :1])], -1)
    rast, _ = uv_raster.rasterize(None, uv_clip[None], geom.faces.int(), resolution=[tex_h, tex_w])
    rast = rast.flip(1)  # uv_raster works bottom-up, the image top-down
    covered = rast[0, ..., 3] > 0
    flat = torch.nonzero(covered.flatten()).flatten()
    if flat.numel() == 0:
        raise ValueError("the texture has no covered texels")
    attrs = torch.cat([geom.verts, geom.normals], -1)
    texel = uv_raster.interpolate(attrs[None], rast, geom.faces.int())[0][0].view(-1, 6)[flat]
    face = rast[0, ..., 3].flatten()[flat].round().long() - 1
    del rast
    normals = texel[:, 3:] / texel[:, 3:].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return Texels(
        flat=flat, covered=covered, points=texel[:, :3], normals=normals, face_normals=geom.face_normals[face], size=(tex_h, tex_w)
    )


def texture_of(mesh: Any, device: Optional[Any] = None) -> tuple[torch.Tensor, Optional[np.ndarray]]:
    """The base colour texture as (H, W, 3) sRGB in [0, 1] (row 0 at v = 1), and its alpha (as it was)."""
    material = getattr(getattr(mesh, "visual", None), "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        raise ValueError("the mesh has no base colour texture")
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    alpha = np.asarray(image.getchannel("A")).copy() if image.mode in ("RGBA", "LA") else None
    rgb = torch.tensor(np.asarray(image.convert("RGB"), dtype=np.float32) / 255, device=_device(device))
    return rgb, alpha


def as_image(rgb: torch.Tensor, alpha: Optional[np.ndarray] = None) -> Image.Image:
    """(H, W, 3) sRGB in [0, 1] as a PIL image, with ``alpha`` put back when there was one."""
    image = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8), "RGB")
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))
    return image


def as_tensor(image: Image.Image, device: Optional[Any] = None) -> torch.Tensor:
    """A PIL image as (H, W, 3) sRGB in [0, 1]."""
    return torch.tensor(np.asarray(image.convert("RGB"), dtype=np.float32) / 255, device=_device(device))


# --- Renders ------------------------------------------------------------------------------------------


@dataclass
class Render:
    image: torch.Tensor  # (H, W, 3) sRGB in [0, 1]
    mask: torch.Tensor  # (H, W) bool: the object covers at least half the pixel
    coverage: torch.Tensor  # (H, W) float: how much of the pixel it covers


def render(
    geom: Geometry,
    texture: torch.Tensor,
    camera: Camera,
    *,
    ambient: float = AMBIENT,
    supersample: int = SUPERSAMPLE,
    background: Sequence[float] = BACKGROUND,
) -> Render:
    """
    ``texture`` (H, W, 3 sRGB) on the mesh as ``camera`` sees it, lit from the camera (``ambient`` +
    the rest times the cosine between the smooth normal and the view), on ``background``. Drawn
    ``supersample`` times larger each way and averaged down in linear light.
    """
    device = geom.verts.device
    scale = max(1, int(supersample))
    height, width = camera.height * scale, camera.width * scale
    xy, depth, s = camera.project(geom.verts, scale)
    zbuf = rasterize_depth(xy, depth, s, geom.faces, height, width)
    face, bary = rasterize_faces(xy, depth, s, geom.faces, zbuf)
    del zbuf
    hit = face >= 0
    corners = geom.faces[face.clamp_min(0)]  # (h, w, 3)
    uv = (bary[..., None] * geom.uv[corners]).sum(-2)
    grid = torch.stack([uv[..., 0] * 2 - 1, (1 - uv[..., 1]) * 2 - 1], -1)[None]
    albedo = F.grid_sample(texture.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    albedo = albedo[0].permute(1, 2, 0)
    del uv, grid
    normal = (bary[..., None] * geom.normals[corners]).sum(-2)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    point = (bary[..., None] * geom.verts[corners]).sum(-2)
    del bary, corners
    towards = projection.view_dirs(point.view(-1, 3), camera.params(device)).view(height, width, 3)
    light = ambient + (1 - ambient) * (normal * towards).sum(-1).clamp(0, 1)
    del normal, point, towards
    sky = projection._srgb_to_linear(torch.tensor(background, dtype=torch.float32, device=device))
    linear = torch.where(hit[..., None], projection._srgb_to_linear(albedo) * light[..., None], sky)
    del albedo, light
    linear = F.avg_pool2d(linear.permute(2, 0, 1)[None], scale)[0].permute(1, 2, 0)
    coverage = F.avg_pool2d(hit.float()[None, None], scale)[0, 0]
    return Render(image=projection._linear_to_srgb(linear), mask=coverage >= 0.5, coverage=coverage)


def picture_reference(
    cutout: Image.Image,
    *,
    background: Sequence[float] = BACKGROUND,
    margin: float = REFERENCE_MARGIN,
    pixels: int = REFERENCE_PIXELS,
    multiple: int = MULTIPLE,
) -> Image.Image:
    """
    The picture as the editing model's reference: its object (the cutout's alpha) on ``background``,
    cropped to its bounding box with ``margin`` of its larger side round it, at most ``pixels`` in all,
    sides multiples of ``multiple``.
    """
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.float32) / 255
    alpha = rgba[..., 3:]
    rgb = rgba[..., :3] * alpha + np.asarray(background, dtype=np.float32) * (1 - alpha)
    rows = np.nonzero((alpha[..., 0] > 0.5).any(1))[0]
    cols = np.nonzero((alpha[..., 0] > 0.5).any(0))[0]
    if rows.size == 0:
        raise ValueError("no object in the cutout")
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    pad = int(round(margin * max(bottom - top, right - left)))
    canvas = np.empty((bottom - top + 2 * pad, right - left + 2 * pad, 3), np.float32)
    canvas[:] = np.asarray(background, dtype=np.float32)
    y0, x0 = top - pad, left - pad
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(rgb.shape[0], bottom + pad), min(rgb.shape[1], right + pad)
    canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    image = Image.fromarray((canvas * 255 + 0.5).astype(np.uint8), "RGB")
    h, w = canvas.shape[:2]
    scale = min(1.0, math.sqrt(pixels / (h * w)))
    width = max(multiple, int(w * scale) // multiple * multiple)
    height = max(multiple, int(h * scale) // multiple * multiple)
    return image.resize((width, height), Image.Resampling.LANCZOS)


# --- Checks -------------------------------------------------------------------------------------------


def _flood(passable: torch.Tensor, seeds: torch.Tensor, check_every: int = 16) -> torch.Tensor:
    """The pixels (H, W bool) reached from ``seeds`` through ``passable`` ones, 8-connected."""
    reach = (seeds & passable).float()[None, None]
    allowed = passable.float()[None, None]
    limit = 4 * (passable.shape[0] + passable.shape[1])
    for step in range(limit):
        grown = F.max_pool2d(reach, 3, 1, 1) * allowed
        if step % check_every == check_every - 1 and torch.equal(grown, reach):
            break
        reach = grown
    return reach[0, 0] > 0


def object_mask(
    image: torch.Tensor,
    expected: Optional[torch.Tensor] = None,
    *,
    tolerance: float = BACKGROUND_TOLERANCE,
    border: int = 4,
) -> torch.Tensor:
    """
    Where a painted view (H, W, 3 sRGB) shows its object: every pixel not reached from the image's border
    through background-coloured pixels (within ``tolerance`` of the border's median colour). With
    ``expected`` (the render's mask), grey pixels darker than the background but not black (a soft shadow)
    are background too outside it. The background's colour inside the object (a white car's paint against
    white) is the object's when walled off by its outline.
    """
    h, w = image.shape[:2]
    edge = torch.zeros((h, w), dtype=torch.bool, device=image.device)
    edge[:border] = True
    edge[-border:] = True
    edge[:, :border] = True
    edge[:, -border:] = True
    colour = image[edge].median(0).values
    passable = (image - colour).abs().amax(-1) <= tolerance
    if expected is not None:
        luma = projection._luma(image)
        chroma = image.amax(-1) - image.amin(-1)
        shadow = (chroma <= SHADOW_CHROMA) & (luma >= SHADOW_LUMA * projection._luma(colour)) & (luma <= float(colour.max()) + tolerance)
        outside = ~(projection._dilate(expected.float()[None, None], 1)[0, 0] > 0)
        passable = passable | (shadow & outside)
    return ~_flood(passable, edge)


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Intersection over union of two masks (bool, or float coverages: min over max)."""
    a, b = a.float(), b.float()
    union = float(torch.maximum(a, b).sum())
    return float(torch.minimum(a, b).sum()) / union if union > 0 else 0.0


@dataclass(frozen=True)
class Alignment:
    """
    Where a painted view's object sits against the render's: render pixel p shows what the painted image
    has at ``centre + (p - centre) * exp(log_scale) + shift`` (pixels).
    """

    shift: tuple[float, float]
    log_scale: float
    centre: tuple[float, float]
    iou_before: float
    iou: float

    def as_dict(self) -> dict:
        return {
            "iou_before": round(self.iou_before, 4),
            "iou": round(self.iou, 4),
            "shift": [round(self.shift[0], 2), round(self.shift[1], 2)],
            "scale": round(math.exp(self.log_scale), 4),
        }


def _sample_grid(height: int, width: int, shift, log_scale, centre, device) -> torch.Tensor:
    """grid_sample's grid (1, H, W, 2) for an Alignment's mapping, in an H x W image."""
    cols = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    rows = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    k = math.exp(log_scale)
    sx = centre[0] + (cols - centre[0]) * k + shift[0]
    sy = centre[1] + (rows - centre[1]) * k + shift[1]
    gx = sx / width * 2 - 1
    gy = sy / height * 2 - 1
    return torch.stack(torch.broadcast_tensors(gx[None, :], gy[:, None]), -1)[None]


def align(
    painted: torch.Tensor,
    rendered: torch.Tensor,
    *,
    max_shift: float = MAX_SHIFT,
    max_log_scale: float = MAX_LOG_SCALE,
    size: int = ALIGN_SIZE,
    rounds: int = 24,
) -> Alignment:
    """
    The small shift and scale (about the render's object's centre) that best lays the painted object's
    mask on the render's (both (H, W) bool), by pattern search on the masks at ``size`` pixels: shifts up
    to ``max_shift`` of the longer side, scales up to exp(``max_log_scale``) either way.
    """
    h, w = rendered.shape
    device = rendered.device
    factor = max(1, math.ceil(max(h, w) / size))
    small_painted = F.avg_pool2d(painted.float()[None, None], factor, ceil_mode=True)
    small_rendered = F.avg_pool2d(rendered.float()[None, None], factor, ceil_mode=True)[0, 0]
    hs, ws = small_rendered.shape
    total = float(small_rendered.sum())
    if total <= 0:
        return Alignment((0.0, 0.0), 0.0, (w / 2, h / 2), 0.0, 0.0)
    rows = torch.arange(hs, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(ws, device=device, dtype=torch.float32) + 0.5
    centre_small = (float((small_rendered.sum(0) * cols).sum()) / total, float((small_rendered.sum(1) * rows).sum()) / total)
    limit = (max_shift * max(hs, ws), max_shift * max(hs, ws), max_log_scale)

    def score(trial: tuple[float, float, float]) -> float:
        grid = _sample_grid(hs, ws, trial[:2], trial[2], centre_small, device)
        warped = F.grid_sample(small_painted, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        return iou(warped, small_rendered)

    current = (0.0, 0.0, 0.0)
    best = before = score(current)
    steps = [1.0, 1.0, 0.01]
    for _ in range(rounds):
        improved = False
        for dim in range(3):
            for sign in (-1.0, 1.0):
                trial = list(current)
                trial[dim] = max(-limit[dim], min(limit[dim], trial[dim] + sign * steps[dim]))
                trial = tuple(trial)
                value = score(trial)
                if value > best + 1e-6:
                    current, best, improved = trial, value, True
        if not improved:
            steps = [step / 2 for step in steps]
            if steps[0] < 0.05:
                break
    shift = (current[0] * factor, current[1] * factor)
    centre = (centre_small[0] * factor, centre_small[1] * factor)
    # The IoU at full size, which the gate reads
    full = warp(painted.float()[..., None], Alignment(shift, current[2], centre, 0.0, 0.0))[..., 0] > 0.5
    return Alignment(shift=shift, log_scale=current[2], centre=centre, iou_before=iou(painted, rendered), iou=iou(full, rendered))


def warp(image: torch.Tensor, alignment: Alignment) -> torch.Tensor:
    """A painted view (H, W, C) resampled so that its object lies on the render's (bilinear)."""
    h, w = image.shape[:2]
    grid = _sample_grid(h, w, alignment.shift, alignment.log_scale, alignment.centre, image.device)
    out = F.grid_sample(image.permute(2, 0, 1)[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0)


def _log_luma(image: torch.Tensor) -> torch.Tensor:
    return torch.log(projection._luma(projection._srgb_to_linear(image.clamp(0, 1))) + 0.02)


def _sobel(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device)[None, None]
    gx = F.conv2d(x[None, None], kx, padding=1)[0, 0]
    gy = F.conv2d(x[None, None], kx.transpose(2, 3), padding=1)[0, 0]
    return (gx.square() + gy.square()).sqrt()


def novelty(rendered: torch.Tensor, painted: torch.Tensor, mask: torch.Tensor, *, size: int = NOVELTY_SIZE) -> float:
    """
    How much of a painted view's structure (H, W, 3, laid on the render) the render (H, W, 3) doesn't have,
    inside the render's ``mask``: the share of the painted view's strong edge energy (EDGE) more than
    EDGE_REACH pixels from the render's edges, at ``size`` pixels on the longer side. Edges are taken on
    log luminance, so a change of colour or exposure isn't structure. 0 when the painted view has almost no
    edges.
    """
    h, w = mask.shape
    factor = max(1, math.ceil(max(h, w) / size))

    def pooled(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x[None, None], factor, ceil_mode=True)[0, 0]

    inside = projection._erode((pooled(mask.float()) > 0.99).float()[None, None], 1)[0, 0] > 0
    edges_r = _sobel(pooled(_log_luma(rendered)))
    edges_p = _sobel(pooled(_log_luma(painted)))
    near = projection._dilate((edges_r > EDGE).float()[None, None], EDGE_REACH)[0, 0] > 0
    strong = (edges_p > EDGE) & inside
    total = float((edges_p * strong).sum())
    floor = 0.01 * EDGE * float(inside.sum())  # a few edges on a plain surface aren't a new structure
    return float((edges_p * (strong & ~near)).sum()) / max(total, floor, 1e-9)


def describe(camera: Camera, azimuth: float, elevation: float) -> str:
    """
    Where ``camera`` sees the object from, for the editing model, taking the picture's camera (``azimuth``,
    ``elevation``) as its front: "from the front", "from the side", "from directly behind, showing its back",
    "from directly above", ...
    """
    if camera.elevation >= 60:
        return "from directly above"
    if camera.elevation <= -60:
        return "from directly below, showing its underside"
    offset = abs((camera.azimuth - azimuth + 180) % 360 - 180)
    if offset <= 22.5:
        return "from the front"
    if offset <= 67.5:
        return "from the front, turned to one side"
    if offset <= 112.5:
        return "from the side"
    if offset <= 157.5:
        return "from behind, turned to one side"
    return "from directly behind, showing its back"


# Plain colour names (sRGB) for telling the editing model the picture's main colours
COLOUR_NAMES = {
    "black": (20, 20, 22),
    "charcoal grey": (55, 60, 66),
    "dark grey": (95, 95, 95),
    "grey": (135, 135, 135),
    "silver": (190, 192, 196),
    "white": (245, 245, 245),
    "dark red": (120, 20, 25),
    "red": (200, 35, 35),
    "coral red": (235, 95, 85),
    "pink": (240, 150, 180),
    "orange": (240, 130, 30),
    "brown": (115, 70, 40),
    "tan": (195, 155, 110),
    "beige": (225, 205, 170),
    "gold": (205, 165, 60),
    "yellow": (240, 210, 40),
    "olive green": (110, 120, 45),
    "green": (40, 150, 60),
    "dark green": (25, 80, 40),
    "teal": (0, 125, 125),
    "turquoise": (60, 200, 200),
    "light blue": (155, 195, 230),
    "steel blue": (75, 125, 180),
    "blue": (30, 80, 200),
    "navy blue": (20, 30, 90),
    "purple": (110, 50, 150),
    "lavender": (180, 160, 220),
    "magenta": (200, 40, 160),
}


def main_colours(cutout: Image.Image, *, count: int = 3, min_share: float = 0.1) -> list:
    """
    The picture's main colours by name (COLOUR_NAMES, nearest in CIELAB), most of the object first: those
    covering at least ``min_share`` of the object (the cutout's alpha), at most ``count``.
    """
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.float32) / 255
    inside = rgba[..., 3] > 0.5
    if not inside.any():
        return []
    pixels = torch.tensor(rgba[..., :3][inside])
    if pixels.shape[0] > 200_000:
        pixels = pixels[torch.linspace(0, pixels.shape[0] - 1, 200_000).long()]
    names = list(COLOUR_NAMES)
    swatches = torch.tensor([COLOUR_NAMES[n] for n in names], dtype=torch.float32) / 255
    lab = projection._lab(projection._srgb_to_linear(pixels))
    lab_swatches = projection._lab(projection._srgb_to_linear(swatches))
    nearest = torch.cdist(lab, lab_swatches).argmin(1)
    shares = torch.bincount(nearest, minlength=len(names)).float() / nearest.numel()
    order = torch.argsort(shares, descending=True).tolist()
    return [names[i] for i in order if float(shares[i]) >= min_share][:count]


# --- Bake ---------------------------------------------------------------------------------------------


@dataclass
class Samples:
    """What one view says about each covered texel."""

    weight: torch.Tensor  # (N,)
    colour: torch.Tensor  # (N, 3) linear light


def view_samples(
    geom: Geometry,
    tex: Texels,
    camera: Camera,
    image: torch.Tensor,
    usable: torch.Tensor,
    *,
    cos_power: float = COS_POWER,
    min_cos: float = MIN_COS,
    feather: float = FEATHER,
    edge_jump: float = EDGE_JUMP,
    depth_bias: float = DEPTH_BIAS,
) -> Samples:
    """
    A view's colour (``image``, (H, W, 3) sRGB, the camera's size) and weight at every covered texel: the
    texel must be visible (depth-tested against the view's own depth map) and land on a ``usable`` pixel
    (H, W bool); its weight is cos ** ``cos_power`` of the angle between its normal and the view, fading
    out between ``min_cos`` and ``min_cos`` + 0.1, and near the usable pixels' edge and depth edges
    (``feather`` pixels; ``edge_jump`` pixels' worth of depth make an edge), as mvtexture.bake_views does.
    """
    h, w = image.shape[:2]
    if (w, h) != camera.size or tuple(usable.shape) != (h, w):
        raise ValueError(f"the view is {w} x {h} and its mask {tuple(usable.shape)}, the camera {camera.size}")
    device = geom.verts.device
    pixel = camera.pixel()
    vxy, vdepth, vs = camera.project(geom.verts)
    zbuf = rasterize_depth(vxy, vdepth, vs, geom.faces, h, w)
    usable = usable & torch.isfinite(zbuf)
    fade = _fade(zbuf, usable, feather, edge_jump * pixel)

    xy, depth, _ = camera.project(tex.points)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    at = xy[:, 1].long().clamp(0, h - 1) * w + xy[:, 0].long().clamp(0, w - 1)
    towards = projection.view_dirs(tex.points, camera.params(device))
    cos = (tex.normals * towards).sum(-1)
    cos_face = (tex.face_normals * towards).sum(-1)
    slope = (1 - cos_face.square()).clamp_min(0).sqrt() / cos_face.abs().clamp_min(0.05)
    slack = pixel * (depth_bias + 0.75 * slope.clamp(max=20.0))
    visible = inside & (depth <= zbuf.flatten()[at] + slack)
    del towards, slope, slack, at, zbuf

    grid = torch.stack([xy[:, 0] / w * 2 - 1, xy[:, 1] / h * 2 - 1], -1)[None, None]
    usable_f = usable.float()[None]
    linear = projection._srgb_to_linear(image.clamp(0, 1)).permute(2, 0, 1) * usable_f
    sampled = F.grid_sample(
        torch.cat([linear, usable_f, fade[None]])[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, :, 0].T  # (N, 5)
    colour = (sampled[:, :3] / sampled[:, 3:4].clamp_min(1e-6)).clamp(0, 1)
    weight = (
        visible.float()
        * cos.clamp_min(0) ** cos_power
        * projection._smoothstep(min_cos, min_cos + 0.1, cos)
        * projection._smoothstep(0.0, 0.1, cos_face)
        * sampled[:, 4]
        * (sampled[:, 3] > 1e-3)
    )
    return Samples(weight=weight, colour=colour)


def gains(
    colour: torch.Tensor,
    established: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_gain: float = MAX_GAIN,
    min_texels: int = GAIN_TEXELS,
) -> Optional[torch.Tensor]:
    """
    Per-channel gains (3,) that bring a view's colours (N, 3, linear) to the established ones: the
    weighted median of their ratios over the texels with ``weight`` > 0 (the view's weight times how sure
    the established colour is), neither near black, at most ``max_gain`` either way. None from fewer than
    ``min_texels`` texels.
    """
    use = (weight > 0) & (projection._luma(colour) > 0.004) & (projection._luma(established) > 0.004)
    if int(use.sum()) < min_texels:
        return None
    ratio = torch.log((established[use] + 1e-4) / (colour[use] + 1e-4))
    w = weight[use]
    if ratio.shape[0] > GAIN_SAMPLES:
        keep = torch.linspace(0, ratio.shape[0] - 1, GAIN_SAMPLES, device=ratio.device).long()
        ratio, w = ratio[keep], w[keep]
    median = torch.stack([projection._wquantile(ratio[:, c], w, 0.5) for c in range(3)])
    return torch.exp(median).clamp(1 / max_gain, max_gain)


def _joint_solve(
    log: torch.Tensor,
    usable: torch.Tensor,
    weight: torch.Tensor,
    log_anchor: torch.Tensor,
    anchored: torch.Tensor,
    limits: Sequence[float],
    *,
    scale: float,
    rounds: int,
    prior: float,
) -> torch.Tensor:
    """
    Per-view offsets (K, C) for K views' values ``log`` (K, N, C, log units), solved together: per channel they
    minimise

        sum over view pairs i, j and texels t of  min(w_i, w_j) u_i u_j (l_i + g_i - l_j - g_j)^2
      + sum over views i and texels t of          anchored_i (l_i + g_i - l_anchor)^2
      + prior * (each view's total weight) * g_i^2

    where ``weight`` (K, N) is each view's weight w, ``usable`` (K, N, C) u says where a view's value counts, and
    ``anchored`` (K, N, C) is how much each view's value is held to ``log_anchor`` (N, C). After the first solve,
    ``rounds`` more re-weight each term by Cauchy's weight on its residual (``scale`` log units), so a detail one
    view drew and another didn't hardly pulls the result. Offsets stay within ``limits`` (C, log units) either
    way; a view that overlaps nothing keeps 0.
    """
    count, _, channels = log.shape
    device = log.device
    usable = usable.float()
    limit = torch.tensor(list(limits), device=device, dtype=torch.float32)
    totals = weight.sum(1).double()  # (K,)
    gains = torch.zeros((count, channels), device=device)
    for round_number in range(max(0, rounds) + 1):
        matrix = torch.zeros((channels, count, count), device=device, dtype=torch.float64)
        rhs = torch.zeros((channels, count), device=device, dtype=torch.float64)
        for i in range(count):
            # Pairs: view i against every view j (both orders appear, so each pair counts once per order)
            both = torch.minimum(weight[i][None], weight)[..., None] * usable[i][None] * usable  # (K, N, C)
            both[i] = 0
            difference = log[i][None] - log  # (K, N, C): l_i - l_j
            if round_number:
                residual = difference + (gains[i][None, None] - gains[:, None])  # with the current offsets
                both = both * _cauchy_weight(residual, scale)
            pair = both.sum(1).double()  # (K, C): S_ij per channel
            matrix[:, i, i] += pair.sum(0)
            matrix[:, i, :] -= pair.T
            rhs[:, i] -= (both * difference).sum(1).sum(0).double()
            del both, difference
            # The anchor
            to_anchor = log[i] - log_anchor  # (N, C)
            a = anchored[i]
            if round_number:
                a = a * _cauchy_weight(to_anchor + gains[i][None], scale)
            matrix[:, i, i] += a.sum(0).double()
            rhs[:, i] -= (a * to_anchor).sum(0).double()
        # Pull towards no change, in proportion to each view's weight (and a little for views overlapping nothing)
        ridge = prior * totals + 1e-6
        matrix += torch.diag_embed(ridge.expand(channels, -1))
        solved = torch.linalg.solve(matrix, rhs[..., None])[..., 0].T.float()  # (K, C)
        gains = torch.maximum(torch.minimum(solved, limit), -limit)
    return gains


def joint_gains(
    samples: Sequence[Samples],
    anchor: torch.Tensor,
    anchor_weight: torch.Tensor,
    *,
    max_gain: float = MAX_GAIN,
    scale: float = JOINT_SCALE,
    rounds: int = JOINT_ROBUST,
    prior: float = JOINT_PRIOR,
) -> torch.Tensor:
    """
    Per-view, per-channel gains (K, 3) for K views' ``samples`` (colour in linear light), solved together on log
    colour (``_joint_solve``): the views agree where they overlap, without a seam where one view's exposure
    differed, and as a whole match the ``anchor`` colour (N, 3, linear) where ``anchor_weight`` (N,) says it is
    sure. Texels near black in either colour don't count. Gains stay within ``max_gain`` either way; a view that
    overlaps nothing keeps 1.
    """
    count = len(samples)
    device = anchor.device
    if count == 0:
        return torch.ones((0, 3), device=device)
    eps = 1e-4
    weight = torch.stack([s.weight for s in samples])  # (K, N)
    colour = torch.stack([s.colour for s in samples])  # (K, N, 3)
    lit = (projection._luma(colour) > 0.004).float()[..., None].expand(-1, -1, 3)  # (K, N, 3)
    anchored = (anchor_weight * (projection._luma(anchor) > 0.004).float())[None, :, None] * weight[..., None] * lit
    found = _joint_solve(
        torch.log(colour + eps), lit, weight, torch.log(anchor + eps), anchored, [math.log(max_gain)] * 3,
        scale=scale, rounds=rounds, prior=prior,
    )
    return torch.exp(found)


def tone_features(colour: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Colours (..., 3, linear) as (..., 2): log luminance and log relative chroma (|c - Y| / Y), with where each
    counts (..., 2, bool): luminance off black; chroma at least MIN_CHROMA on a colour brighter than CHROMA_LUMA.
    """
    luma = projection._luma(colour)
    chroma = (colour - luma[..., None]).norm(dim=-1) / luma.clamp_min(eps)
    features = torch.stack([torch.log(luma + eps), torch.log(chroma + eps)], -1)
    usable = torch.stack([luma > 0.004, (chroma >= MIN_CHROMA) & (luma > CHROMA_LUMA)], -1)
    return features, usable


def apply_tone(colour: torch.Tensor, tone: torch.Tensor) -> torch.Tensor:
    """``colour`` (N, 3, linear) with a tone (2,: gain g, saturation s) applied: g (Y + s (c - Y)), clamped."""
    luma = projection._luma(colour)[..., None]
    return (tone[0] * (luma + tone[1] * (colour - luma))).clamp(0, 1)


def joint_tone(
    samples: Sequence[Samples],
    anchor: torch.Tensor,
    anchor_weight: torch.Tensor,
    *,
    max_gain: float = MAX_GAIN,
    max_saturation: float = MAX_SATURATION,
    scale: float = JOINT_SCALE,
    rounds: int = JOINT_ROBUST,
    prior: float = JOINT_PRIOR,
) -> torch.Tensor:
    """
    Per-view tones (K, 2: brightness gain, saturation scale; apply_tone) for K views' ``samples``, solved together
    as joint_gains solves gains, on tone_features instead of log colour: the views agree in brightness and
    saturation where they overlap, and match the ``anchor`` colour (N, 3, linear: the picture's own) where
    ``anchor_weight`` (N,) says it is sure. Hues are left as each view painted them, so a grey stays grey.
    """
    count = len(samples)
    device = anchor.device
    if count == 0:
        return torch.ones((0, 2), device=device)
    weight = torch.stack([s.weight for s in samples])  # (K, N)
    features, usable = zip(*(tone_features(s.colour) for s in samples))
    features, usable = torch.stack(features), torch.stack(usable).float()  # (K, N, 2)
    anchor_features, anchor_usable = tone_features(anchor)
    anchored = anchor_weight[None, :, None] * weight[..., None] * usable * anchor_usable.float()[None]
    found = _joint_solve(
        features, usable, weight, anchor_features, anchored, [math.log(max_gain), math.log(max_saturation)],
        scale=scale, rounds=rounds, prior=prior,
    )
    return torch.exp(found)


def tone(
    colour: torch.Tensor,
    established: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_gain: float = MAX_GAIN,
    max_saturation: float = MAX_SATURATION,
    min_texels: int = GAIN_TEXELS,
) -> Optional[torch.Tensor]:
    """
    The tone (2,: gain, saturation; apply_tone) that brings a view's colours (N, 3, linear) to the established
    ones, as ``gains`` does per channel: weighted medians of the log ratios of luminance and of relative chroma
    over the texels with ``weight`` > 0 (saturation over those with chroma in both; 1 without enough of them).
    None from fewer than ``min_texels`` texels.
    """
    mine, mine_ok = tone_features(colour)
    theirs, theirs_ok = tone_features(established)
    found = []
    for channel, limit in ((0, max_gain), (1, max_saturation)):
        use = (weight > 0) & mine_ok[:, channel] & theirs_ok[:, channel]
        if int(use.sum()) < min_texels:
            if channel == 0:
                return None
            found.append(torch.ones((), device=colour.device))
            continue
        ratio = (theirs[:, channel] - mine[:, channel])[use]
        w = weight[use]
        if ratio.shape[0] > GAIN_SAMPLES:
            keep = torch.linspace(0, ratio.shape[0] - 1, GAIN_SAMPLES, device=ratio.device).long()
            ratio, w = ratio[keep], w[keep]
        found.append(torch.exp(projection._wquantile(ratio, w, 0.5)).clamp(1 / limit, limit))
    return torch.stack(found)


def _cauchy_weight(residual: torch.Tensor, scale: float) -> torch.Tensor:
    """Cauchy's IRLS weight, 1 / (1 + (r / scale)^2): near 1 within ``scale``, near 0 for gross misfits."""
    return 1.0 / (1.0 + (residual / scale).square())


class Blend:
    """The views' colours at the covered texels, summed by weight in linear light."""

    def __init__(self, count: int, device: Any) -> None:
        self.mixed = torch.zeros((count, 3), device=device)
        self.total = torch.zeros(count, device=device)

    def add(self, samples: Samples) -> None:
        self.mixed += samples.weight[:, None] * samples.colour
        self.total += samples.weight

    def colour(self) -> torch.Tensor:
        return self.mixed / self.total.clamp_min(1e-9)[:, None]

    def amount(self, full_weight: float = FULL_WEIGHT) -> torch.Tensor:
        """How much of the views' colour each texel takes: in full where their weights add up to ``full_weight``."""
        if full_weight <= 0:
            return (self.total > 0).float()
        return projection._smoothstep(0.0, full_weight, self.total)


def robust_colour(samples: Sequence[Samples], tolerance: float = ROBUST_TOLERANCE, select: float = 0.0) -> torch.Tensor:
    """
    The views' colour at each texel (N, 3, linear), blended by weight over the views whose luminance lies
    within ``tolerance`` (log) of the weighted median of all of them: a highlight or a ghost that only one
    view drew doesn't go in. Where one view has all the weight, it is that view's colour. With ``select``, the
    views kept are blended by their weights sharpened (select_weights), so each texel takes mostly the best of them.
    """
    weight = torch.stack([s.weight for s in samples])  # (K, N)
    colour = torch.stack([s.colour for s in samples])  # (K, N, 3)
    luma = torch.log(projection._luma(colour) + 1e-3)
    order = torch.argsort(luma, dim=0)
    sorted_luma = torch.gather(luma, 0, order)
    cumulative = torch.cumsum(torch.gather(weight, 0, order), 0)
    half = 0.5 * cumulative[-1:]
    index = (cumulative < half).sum(0, keepdim=True).clamp(max=len(samples) - 1)
    median = torch.gather(sorted_luma, 0, index)  # (1, N)
    keep = weight * ((luma - median).abs() <= tolerance).float()
    if select > 0:
        keep = select_weights(keep, select)
        weight = select_weights(weight, select)
    total = keep.sum(0)
    robust = (keep[..., None] * colour).sum(0) / total.clamp_min(1e-9)[:, None]
    plain = (weight[..., None] * colour).sum(0) / weight.sum(0).clamp_min(1e-9)[:, None]
    return torch.where((total > 0)[:, None], robust, plain)


def select_weights(weights: torch.Tensor, temperature: float = SELECT) -> torch.Tensor:
    """
    Views' weights (K, N) sharpened towards each texel's best view: a softmax over their logs at ``temperature``
    (so in proportion to weight ** (1 / temperature)), nothing where a view had no weight. A temperature of 0 or
    less leaves them as they are.
    """
    if temperature <= 0:
        return weights
    logs = torch.log(weights.clamp_min(1e-12)) / temperature
    return torch.softmax(logs, 0) * (weights > 0).float()


def compose(texture: torch.Tensor, tex: Texels, colour: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """
    ``texture`` (H, W, 3 sRGB) with the covered texels moved ``amount`` (N,) of the way to ``colour`` (N, 3,
    linear) in linear light, the change carried on into the gutters (so filtering at chart edges doesn't
    bring the old colours back).
    """
    tex_h, tex_w = tex.size
    old = projection._srgb_to_linear(texture.reshape(-1, 3)[tex.flat])
    new = old + amount[:, None] * (colour - old)
    out = texture.clone().reshape(-1, 3)
    touched = amount > 0
    out[tex.flat[touched]] = projection._linear_to_srgb(new[touched])
    out = out.view(tex_h, tex_w, 3)
    filled = _into_gutters((out - texture).permute(2, 0, 1), tex.covered)
    return torch.where(tex.covered[..., None], out, (texture + filled.permute(1, 2, 0)).clamp(0, 1))


# --- The picture --------------------------------------------------------------------------------------


@dataclass
class Picture:
    """The picture's projection onto a stand-in of the mesh: the texture it made, and where it used the picture."""

    texture: torch.Tensor  # (H, W, 3) sRGB: the picture painted on the side it shows (the original if not applied)
    weight: torch.Tensor  # (N,) per covered texel: the projection's detail weight (0 where not applied)
    azimuth: float  # the picture's camera (0 and ELEVATION when the projection didn't find it)
    elevation: float
    report: dict
    # (N, 3) linear: the picture's own colour at each covered texel, before the projection matched its exposure to
    # the texture's (None where not applied); the joint colour match's anchor, where ``weight`` says it saw well
    colour: Optional[torch.Tensor] = None


def project_picture(mesh: Any, cutout: Any, tex: Texels, texture: torch.Tensor, device: Any) -> Picture:
    """
    The production projection run on a stand-in for ``mesh`` (its vertices, faces, UVs and material
    textures; the mesh itself is untouched), to find the picture's camera and paint the picture where it
    reaches, with the weight it gave each texel.
    """
    material = getattr(mesh.visual, "material", None)
    stand_in = SimpleNamespace(
        vertices=mesh.vertices,
        faces=mesh.faces,
        visual=SimpleNamespace(
            uv=mesh.visual.uv,
            material=SimpleNamespace(
                baseColorTexture=material.baseColorTexture,
                metallicRoughnessTexture=getattr(material, "metallicRoughnessTexture", None),
                roughnessFactor=getattr(material, "roughnessFactor", None),
                metallicFactor=getattr(material, "metallicFactor", None),
            ),
        ),
    )
    debug: dict = {}
    _, report = projection.project_picture(stand_in, cutout, device=device, debug=debug)
    weight = torch.zeros(tex.flat.numel(), device=device)
    colour = None
    azimuth, elevation = 0.0, ELEVATION
    painted = texture
    if report.get("applied"):
        painted = as_tensor(stand_in.visual.material.baseColorTexture, device)
        if tuple(painted.shape[:2]) != tex.size:
            raise ValueError("the projection changed the texture's size")
        at = debug["flat"].to(device)
        full = torch.zeros(tex.size[0] * tex.size[1], device=device)
        full[at] = debug["weight"].to(device).float()
        weight = full[tex.flat]
        if debug.get("picture_linear") is not None:
            full_colour = torch.zeros((tex.size[0] * tex.size[1], 3), device=device)
            full_colour[at] = debug["picture_linear"].to(device).float()
            colour = full_colour[tex.flat]
        params = debug["params"][0]
        azimuth, elevation = float(params[0]) % 360, float(params[1])
    elif debug.get("params") is not None:
        # Found but not applied (too little fits, or an ambiguous camera): its direction still orders the views
        params = debug["params"][0]
        azimuth, elevation = float(params[0]) % 360, float(params[1])
    summary = projection.summary(report)
    return Picture(texture=painted, weight=weight, azimuth=azimuth, elevation=elevation, report=summary, colour=colour)


# --- The painter --------------------------------------------------------------------------------------

# paint(render, picture, neighbour, seed, view) -> painted view: the editing model. ``render`` is the view to
# repaint (its size is the camera's), ``picture`` the picture as a reference, ``neighbour`` the painted view
# nearest to this one (None for the first), ``view`` what the view is: {"name", "side" (describe's words),
# "colours" (main_colours of the picture)}
Painter = Callable[[Image.Image, Image.Image, Optional[Image.Image], int, dict], Image.Image]


@dataclass
class View:
    """One camera's turn: its render, what the editing model painted, and whether it went in."""

    camera: Camera
    render: Image.Image
    painted: Optional[Image.Image] = None  # the last attempt, as drawn
    aligned: Optional[Image.Image] = None  # the accepted attempt, laid on the render
    accepted: bool = False
    attempts: list = field(default_factory=list)
    gains: Optional[list] = None
    joint_gains: Optional[list] = None  # the joint colour match's, when it ran (these replace ``gains``)
    texels: int = 0  # covered texels it set more than half of, when it went in
    seconds: dict = field(default_factory=dict)
    skipped: Optional[str] = None  # why it wasn't painted at all

    def as_dict(self) -> dict:
        return {
            **self.camera.as_dict(),
            "accepted": self.accepted,
            "attempts": self.attempts,
            "gains": self.gains,
            "joint_gains": self.joint_gains,
            "texels": self.texels,
            "seconds": self.seconds,
            **({"skipped": self.skipped} if self.skipped else {}),
        }


@dataclass
class Result:
    texture: Image.Image  # the new base colour texture (before the picture's projection, which goes on top)
    views: list  # View, in the order they were painted
    report: dict
    robust: Optional[Image.Image] = None  # the same views blended robustly (robust_colour)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def paint_views(
    mesh: Any,
    cutout: Any,
    paint: Painter,
    *,
    cameras: Optional[Sequence[Camera]] = None,
    seed: int = 0,
    attempts: int = 2,
    min_iou: float = MIN_IOU,
    max_novelty: float = MAX_NOVELTY,
    full_weight: float = FULL_WEIGHT,
    match_colour: bool = True,
    joint: bool = True,
    colour_model: str = "tone",
    anchor: str = "picture",
    select: float = SELECT,
    device: Optional[Any] = None,
    log: Callable[[str], None] = print,
    **camera_options: Any,
) -> Result:
    """
    Paints ``mesh`` (to_glb's trimesh, unpremultiplied, before the picture's projection) from views round
    it with ``paint``, and returns its new base colour texture with what happened. ``cutout`` is the
    picture with its background removed (RGBA, full frame; the projection paints from it and the editing
    model sees it). ``cameras`` defaults to ``ring`` round the picture's camera (``camera_options`` go to
    ``ring``: elevation, around, top, bottom, perspective, margin, pixels); they are painted nearest the
    picture first. Each view gets ``attempts`` tries (seeds ``seed + 100 * view + attempt``) to reach
    ``min_iou`` with no more than ``max_novelty``. ``match_colour`` brings each view to the colours already
    established, for the renders of the views after it; with ``joint`` the final blend instead takes every
    view's own colours brought to agree by a match solved for all views together: they agree where they overlap,
    and with the ``anchor``. ``colour_model`` is what a match changes: "tone" brightness and saturation only
    (tone, joint_tone: a grey stays grey), "gains" each channel (gains, joint_gains). ``anchor`` is "picture"
    (the picture's own colour where it saw the surface well), "paint" (the picture's paint as the projection
    left it, where it used the picture: run 5's) or "none" (the views only agree with each other). The final
    blend sharpens the views' weights by ``select`` (select_weights), so each texel takes mostly its best view.
    A bottom view whose render is dark (DARK_BOTTOM) isn't painted. The mesh isn't changed. The result has the
    views blended by weight (``texture``) and robustly (``robust``).
    """
    if colour_model not in ("tone", "gains"):
        raise ValueError(f"unknown colour model {colour_model!r}")
    if anchor not in ("picture", "paint", "none"):
        raise ValueError(f"unknown anchor {anchor!r}")
    device = _device(device)
    started = time.perf_counter()
    timings: dict = {}
    last = [started]

    def clock(name: str) -> None:
        _sync(device)
        now = time.perf_counter()
        timings[name] = round(timings.get(name, 0.0) + now - last[0], 3)
        last[0] = now

    with torch.no_grad():
        geom = geometry(mesh, device)
        texture, alpha = texture_of(mesh, device)
        tex = texels(geom, tuple(texture.shape[:2]))
        clock("setup_s")
        picture = project_picture(mesh, cutout, tex, texture, device)
        clock("projection_s")
        log(f"[paint] picture: {picture.report}")
        reference = picture_reference(cutout)
        colours = main_colours(cutout)
        if cameras is None:
            cameras = ring(geom.verts, picture.azimuth, **camera_options)
        ordered = by_angle(cameras, picture.azimuth, picture.elevation)

        protect = projection._smoothstep(PROTECT[0], PROTECT[1], picture.weight)
        painted_picture = picture.texture
        current = painted_picture
        paint_colour = projection._srgb_to_linear(painted_picture.reshape(-1, 3)[tex.flat])
        established = paint_colour
        confidence = protect.clone()
        # What the views' colours are held to: the picture's own colour where it saw the surface well, or its paint
        if anchor == "picture" and picture.colour is not None:
            held, held_weight = picture.colour, picture.weight
        elif anchor == "paint":
            held, held_weight = paint_colour, protect
        else:
            held, held_weight = paint_colour, torch.zeros_like(protect)
        blend = Blend(tex.flat.numel(), device)
        views: list[View] = []
        weights: list = []  # (view, its weight at each texel), for the shares at the end
        kept: list = []  # every accepted view's samples, for the robust blend
        raw: list = []  # the same before the sequential colour match, for the joint one
        for number, camera in enumerate(ordered):
            view_started = time.perf_counter()
            shot = render(geom, current, camera)
            view = View(camera=camera, render=as_image(shot.image))
            view.seconds["render_s"] = round(time.perf_counter() - view_started, 3)
            accepted = [v for v in views if v.accepted]
            neighbour = None
            if accepted:
                nearest = min(accepted, key=lambda v: angle_between(v.camera.direction(), camera.direction()))
                neighbour = nearest.aligned
            about = {"name": camera.name, "side": describe(camera, picture.azimuth, picture.elevation), "colours": colours}
            if camera.name == "bottom" and shot.mask.any():
                dark = float(projection._luma(shot.image[shot.mask]).median())
                if dark < DARK_BOTTOM:
                    view.skipped = f"a dark underside (median luminance {dark:.2f})"
                    log(f"[paint] {camera.name}: not painted, {view.skipped}")
                    views.append(view)
                    clock("views_s")
                    continue
            chosen = None
            for attempt in range(max(1, attempts)):
                attempt_seed = seed + 100 * number + attempt
                clock_paint = time.perf_counter()
                painted = paint(view.render, reference, neighbour, attempt_seed, about)
                paint_s = round(time.perf_counter() - clock_paint, 3)
                if painted.size != camera.size:
                    painted = painted.convert("RGB").resize(camera.size, Image.Resampling.LANCZOS)
                view.painted = painted.convert("RGB")
                image = as_tensor(view.painted, device)
                mask = object_mask(image, shot.mask)
                fit = align(mask, shot.mask)
                new = novelty(shot.image, warp(image, fit), shot.mask) if fit.iou >= min_iou else None
                entry = {"seed": attempt_seed, "paint_s": paint_s, **fit.as_dict()}
                if new is not None:
                    entry["novelty"] = round(new, 3)
                view.attempts.append(entry)
                log(f"[paint] {camera.name}: attempt {attempt + 1}, {entry}")
                if fit.iou >= min_iou and new is not None and new <= max_novelty:
                    chosen = (image, mask, fit)
                    break
            if chosen is None:
                log(f"[paint] {camera.name}: left out (IoU under {min_iou} or novelty over {max_novelty})")
                views.append(view)
                clock("views_s")
                continue
            bake_started = time.perf_counter()
            image, mask, fit = chosen
            aligned = warp(image, fit).clamp(0, 1)
            aligned_mask = warp(mask.float()[..., None], fit)[..., 0] > 0.5
            view.aligned = as_image(aligned)
            view.accepted = True
            samples = view_samples(geom, tex, camera, aligned, shot.mask & aligned_mask)
            raw.append((view, Samples(weight=samples.weight, colour=samples.colour.clone())))
            if match_colour:
                # Matched to what's established, with the anchor's colour where the picture is what's established
                target = established
                if anchor == "picture" and picture.colour is not None:
                    target = established + protect[:, None] * (picture.colour - established)
                sure = samples.weight * (confidence >= CONFIDENT).float() * confidence
                if colour_model == "tone":
                    found = tone(samples.colour, target, sure)
                    if found is not None:
                        samples.colour = apply_tone(samples.colour, found)
                else:
                    found = gains(samples.colour, target, sure)
                    if found is not None:
                        samples.colour = (samples.colour * found).clamp(0, 1)
                if found is not None:
                    view.gains = [round(float(g), 3) for g in found]
            blend.add(samples)
            weights.append((view, samples.weight))
            kept.append(samples)
            # The next render shows the views over the picture's paint (kept where the picture was used)
            amount = blend.amount(full_weight) * (1 - protect)
            current = compose(painted_picture, tex, blend.colour(), amount)
            established = projection._srgb_to_linear(current.reshape(-1, 3)[tex.flat])
            confidence = torch.maximum(protect, blend.amount(full_weight))
            view.seconds["bake_s"] = round(time.perf_counter() - bake_started, 3)
            views.append(view)
            clock("views_s")

        # The views over the original texture: the picture's projection goes on top afterwards
        joint_report = None
        if joint and match_colour and raw:
            solve = joint_tone if colour_model == "tone" else joint_gains
            found = solve([s for _, s in raw], held, held_weight)
            blend = Blend(tex.flat.numel(), device)
            kept = []
            for (view, samples), gain in zip(raw, found):
                if colour_model == "tone":
                    colour = apply_tone(samples.colour, gain)
                else:
                    colour = (samples.colour * gain).clamp(0, 1)
                matched = Samples(weight=samples.weight, colour=colour)
                blend.add(matched)
                kept.append(matched)
                view.joint_gains = [round(float(g), 3) for g in gain]
            joint_report = {
                "model": colour_model,
                "anchor": anchor if float(held_weight.sum()) > 0 else "none",
                "views": {view.camera.name: view.joint_gains for view, _ in raw},
            }
            clock("joint_s")
        del raw
        amount = blend.amount(full_weight)
        selected = kept
        if kept and select > 0:
            # Each texel mostly from its best view
            sharp = select_weights(torch.stack([s.weight for s in kept]), select)
            selected = [Samples(weight=w, colour=s.colour) for w, s in zip(sharp, kept)]
            del sharp
            blend = Blend(tex.flat.numel(), device)
            for samples in selected:
                blend.add(samples)
        final = compose(texture, tex, blend.colour(), amount)
        # Robustly: the median over the views' own weights decides which views a texel may take, then the best of those
        robust = compose(texture, tex, robust_colour(kept, select=select), amount) if kept else final
        # Each view's share: the texels it set more than half of (kept follows the accepted views' order)
        for (view, _), samples in zip(weights, selected):
            view.texels = int((amount * samples.weight / blend.total.clamp_min(1e-9) > 0.5).sum())
        del kept, selected, weights
        clock("compose_s")

    report = {
        "picture": picture.report,
        "picture_camera": {"azimuth": round(picture.azimuth, 1), "elevation": round(picture.elevation, 1)},
        "colours": colours,
        "views": [view.as_dict() for view in views],
        "accepted": sum(view.accepted for view in views),
        "texels": int(tex.flat.numel()),
        "changed": int((amount > 0.5).sum()),
        "changed_share": round(float((amount > 0.5).float().mean()), 4),
        "joint_gains": joint_report,
        "select": select,
        "flipped": geom.flipped,
        "timings": {**timings, "total_s": round(time.perf_counter() - started, 3)},
    }
    return Result(texture=as_image(final, alpha), views=views, report=report, robust=as_image(robust, alpha))


# --- Review sheets ------------------------------------------------------------------------------------


def sheet(views: Sequence[View], height: int = 256) -> Image.Image:
    """Each view on a row: its render, what was painted, and the painted view laid on the render (if used)."""
    rows = []
    for view in views:
        row = []
        for image in (view.render, view.painted, view.aligned):
            if image is None:
                image = Image.new("RGB", view.render.size, (40, 40, 40))
            scale = height / image.height
            row.append(image.convert("RGB").resize((max(1, round(image.width * scale)), height), Image.Resampling.LANCZOS))
        rows.append(row)
    width = max(sum(image.width for image in row) for row in rows) if rows else 1
    out = Image.new("RGB", (width, height * max(1, len(rows))), (24, 24, 28))
    for r, row in enumerate(rows):
        x = 0
        for image in row:
            out.paste(image, (x, r * height))
            x += image.width
    return out


# --- Moving a mesh between containers -----------------------------------------------------------------


def pack_mesh(mesh: Any) -> bytes:
    """
    to_glb's textured trimesh as bytes (numpy's npz: arrays as they are, textures as PNG), so another
    container can rebuild it exactly (``unpack_mesh``): vertices, faces, UVs and the PBR material's
    textures, factors, alpha mode and sidedness. Not its vertex normals: nothing after to_glb reads them
    (the export's shading normals are made afresh).
    """
    material = mesh.visual.material

    def png(image: Any) -> np.ndarray:
        if image is None:
            return np.zeros(0, np.uint8)
        image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return np.frombuffer(buffer.getvalue(), np.uint8)

    def factor(value: Any) -> np.ndarray:
        return np.asarray([] if value is None else value, dtype=np.float64).reshape(-1)

    arrays = {
        "vertices": np.asarray(mesh.vertices),
        "faces": np.asarray(mesh.faces),
        "uv": np.asarray(mesh.visual.uv),
        "base_color": png(getattr(material, "baseColorTexture", None)),
        "metallic_roughness": png(getattr(material, "metallicRoughnessTexture", None)),
        "base_color_factor": factor(getattr(material, "baseColorFactor", None)),
        "metallic_factor": factor(getattr(material, "metallicFactor", None)),
        "roughness_factor": factor(getattr(material, "roughnessFactor", None)),
        "alpha_mode": np.asarray(str(getattr(material, "alphaMode", None) or "")),
        "double_sided": np.asarray(bool(getattr(material, "doubleSided", False))),
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def unpack_mesh(data: bytes) -> Any:
    """The trimesh ``pack_mesh`` packed: the same vertices, faces and UVs in the same order, the same material."""
    import trimesh

    arrays = np.load(io.BytesIO(data), allow_pickle=False)

    def image(name: str) -> Optional[Image.Image]:
        raw = arrays[name]
        if raw.size == 0:
            return None
        loaded = Image.open(io.BytesIO(raw.tobytes()))
        loaded.load()
        return loaded

    def factor(name: str) -> Any:
        value = arrays[name]
        if value.size == 0:
            return None
        return float(value[0]) if value.size == 1 else value

    base_color_factor = factor("base_color_factor")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image("base_color"),
        baseColorFactor=None if base_color_factor is None else np.asarray(base_color_factor).astype(np.uint8),
        metallicRoughnessTexture=image("metallic_roughness"),
        metallicFactor=factor("metallic_factor"),
        roughnessFactor=factor("roughness_factor"),
        alphaMode=str(arrays["alpha_mode"]) or None,
        doubleSided=bool(arrays["double_sided"]),
    )
    return trimesh.Trimesh(
        vertices=arrays["vertices"],
        faces=arrays["faces"],
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=arrays["uv"], material=material),
    )
