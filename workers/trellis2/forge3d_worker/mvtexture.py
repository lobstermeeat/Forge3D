"""
Six views of a model, drawn from MV-Adapter's image+geometry cameras, baked into its base colour texture.

TRELLIS.2 makes a good shape but invents the colour of the sides the picture doesn't show (a smeared
back, a ghost of the front). MV-Adapter's image+geometry model (ig2mv, on SDXL) draws six views of a
given shape from a picture, steered by the shape's position and normal maps. This module is the
geometry side of that, on the final mesh as ``to_glb`` makes it:

1. ``frame_for``: where the mesh sits in MV-Adapter's world. Upstream's ``load_mesh(rescale=True)``
   turns a glTF mesh (+Y up, its front +Z) to +Z up and scales it so that its largest coordinate is
   0.5, without moving its centre. Optionally the mesh is first turned about its up axis, so that the
   side the picture shows (the projection's pose search reports its azimuth) faces the front camera.
2. ``control_maps``: the position and normal maps upstream's ``scripts/inference_ig2mv_sdxl.py``
   renders for its control image (world space; positions + 0.5 and normals / 2 + 0.5, both 0.5 where
   the mesh isn't), with each view's mask and depth. ``as_control`` stacks them the way the pipeline
   takes them.
3. ``bake_views``: the six views drawn from those maps go into the texture. Every texel's surface
   point and normal come from rasterising the mesh in UV space. A texel takes colour from a view
   where it is visible there (a depth test against the view's own depth map), weighted by how
   squarely the view sees it (a power of the cosine), fading out near the view's silhouette and depth
   edges. Views are blended by weight in linear light, after an optional colour match to the old
   texture (per-channel gains, from where both agree in hue). Where the views' weights add up to little
   the old colour stays, fading in. A view's background never gets in: only pixels the mesh covers are
   used, less any of the background's colour that reaches in from outside. The change is carried on
   into the texture's gutters, so filtering doesn't bring old colours back at chart edges.

The cameras (``scripts/inference_ig2mv_sdxl.py``, upstream commit 4277e00): orthographic, framing
[-0.55, 0.55] on each axis of the image plane in 768 x 768 pixels, distance 1.8, elevations 0, 0, 0,
0, 89.99 and -89.99 degrees, azimuths 0, 90, 180, 270, 180 and 180 less 90. A camera at azimuth a and
elevation e sits at 1.8 (cos e cos a, cos e sin a, sin e) and looks at the origin; its image right is
look x +Z and its image up is right x look. Image row 0 is the top (upstream's projection matrix
flips y and nvdiffrast stores rows bottom-up). In glTF terms the six views see the mesh's front (+Z),
the side on the front view's right (+X), the back (-Z), the side on its left (-X), the top and the
bottom. The top view has the front at the top of the image, the bottom view the back; both have the
mesh's -X on the right. ``multiview_worker/cameras.py`` spells out the same conventions for the
image-only model's cameras.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import projection, uv_raster
from .projection import rasterize_depth, rasterize_faces

# --- MV-Adapter's ig2mv cameras -----------------------------------------------------------------------
SIZE = 768
HALF_EXTENT = 0.55  # the orthographic frame spans [-0.55, 0.55] world units each way
DISTANCE = 1.8
ELEVATIONS = (0.0, 0.0, 0.0, 0.0, 89.99, -89.99)
AZIMUTHS = tuple(a - 90.0 for a in (0.0, 90.0, 180.0, 270.0, 180.0, 180.0))
NAMES = ("front", "right", "back", "left", "top", "bottom")
# load_mesh(rescale=True): the mesh's largest absolute coordinate becomes this
EXTENT = 0.5
# The control image where the mesh isn't: upstream's positions are 0 there and normal_background is 0,
# so pos + 0.5 and normal / 2 + 0.5 are both 0.5
BACKGROUND = 0.5
# glTF axes (+Y up, front +Z) to MV-Adapter's world (+Z up, front -Y): load_mesh's mesh2std for its
# defaults, shape_init_mesh_up "+y" and shape_init_mesh_front "+x"
GLTF_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# --- Baking defaults (each is a keyword argument of bake_views) ---------------------------------------
# Weight of a view at a texel: cos ** COS_POWER, cos between the texel's normal and the direction to
# the camera, fading to nothing between MIN_COS and MIN_COS + 0.1 (upstream's own texture projection
# uses a power of 3 and leaves out cosines under 0.2)
COS_POWER = 3.0
MIN_COS = 0.2
# Fade-outs, in pixels of the view: near its silhouette and near depth edges (neighbouring pixels whose
# depths differ by more than EDGE_JUMP pixels' worth)
FEATHER = 4.0
EDGE_JUMP = 8.0
# A texel is visible in a view when it is at most this many pixels' worth of depth behind the nearest
# surface there (more on slopes, where depth changes within a pixel)
DEPTH_BIAS = 2.0
# Views whose weights add up to this replace a texel's colour in full; below it, the old colour shows
# through, fully where they add up to nothing
FULL_WEIGHT = 0.25
# A view's background: pixels within BACKGROUND_TOLERANCE (sRGB, per channel) of its colour outside the
# mesh, reached from outside through such pixels at most BACKGROUND_BAND pixels into the mesh
BACKGROUND_BAND = 8
BACKGROUND_TOLERANCE = 0.06
# Colour match: the views' colour is scaled per channel (linear light) by the median ratio of the old
# texture's colour to theirs, over the texels where both are colourful (chroma at least 0.2 of the
# brightness: greys and blacks agree with any grey) and show about the same hue (within MATCH_HUE
# degrees), by at most MAX_GAIN either way. "together" (the default) pools all six views for one set of
# gains; "each" matches every view on its own, which lets a view that replaces a wrong texture be pulled
# towards it (a clean back towards the red ghost of the front on the arcade machine's old back). With
# such texels under MATCH_AGREEMENT of the weight, or fewer than MATCH_TEXELS, the views stay as drawn
MATCH_MODES = ("together", "each")
MAX_GAIN = 2.0
MATCH_HUE = 20.0
MATCH_AGREEMENT = 0.05
MATCH_TEXELS = 200
MATCH_SAMPLES = 400_000  # the median is taken over at most this many (evenly spaced) texels
# The change runs on into the gutters: ring by ring from each chart's edge this many texels out (so a
# chart's edge colour continues, for bilinear filtering and the first mip levels), coarse to fine beyond
GUTTER_RINGS = 8


def _device(device: Optional[Any]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# --- Cameras ------------------------------------------------------------------------------------------


def camera_axes() -> np.ndarray:
    """
    Each camera's right, up and back (towards the camera) unit vectors in MV-Adapter's world, (6, 3, 3)
    with one row each: the columns of upstream's ``get_c2w``.
    """
    azimuth = np.radians(AZIMUTHS)
    elevation = np.radians(ELEVATIONS)
    position = DISTANCE * np.stack(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], -1
    )
    look = -position / np.linalg.norm(position, axis=-1, keepdims=True)
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(right, look)
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    return np.stack([right, up, -look], 1)


def _cameras(device) -> torch.Tensor:
    return torch.tensor(camera_axes(), dtype=torch.float32, device=device)


def _to_view(points: torch.Tensor, axes: torch.Tensor, height: int, width: int):
    """
    World points (N, 3) as one camera sees them: pixel coordinates (N, 2), x right and y down, pixel
    (r, c) centred at (c + 0.5, r + 0.5); and depth (N,), the distance from the camera along its view.
    """
    right, up, back = axes
    x = (points @ right / HALF_EXTENT + 1) * (width / 2)
    y = (1 - points @ up / HALF_EXTENT) * (height / 2)
    return torch.stack([x, y], -1), DISTANCE - points @ back


# --- Frame --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """Where a mesh sits in MV-Adapter's world: ``scale * rotation @ p`` for its points."""

    rotation: np.ndarray  # (3, 3): the mesh's axes to the world's (normals turn by this alone)
    scale: float
    azimuth: float  # degrees: the mesh was turned by minus this about its up axis

    def points(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points, dtype=np.float64) @ self.rotation.T)

    def directions(self, directions: np.ndarray) -> np.ndarray:
        return np.asarray(directions, dtype=np.float64) @ self.rotation.T

    def as_dict(self) -> dict:
        return {"azimuth": round(self.azimuth, 2), "scale": round(self.scale, 5)}


def frame_for(mesh: Any, azimuth_deg: float = 0.0) -> Frame:
    """
    MV-Adapter's world for ``mesh`` (to_glb's trimesh: glTF axes, +Y up, its front +Z). The mesh is
    turned about +Y by ``-azimuth_deg`` first, so that a camera at that azimuth (projection's: 0 on +Z,
    90 on +X) becomes the front camera. Then load_mesh(rescale=True)'s conversion: +Y up becomes +Z,
    +Z becomes -Y (the front camera's side), and the mesh is scaled so its largest absolute coordinate
    is 0.5. Its centre doesn't move.
    """
    turn = math.radians(-float(azimuth_deg))
    c, s = math.cos(turn), math.sin(turn)
    about_up = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    rotation = GLTF_TO_WORLD @ about_up
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("the mesh has no usable vertices")
    largest = float(np.abs(vertices @ rotation.T).max())
    if largest <= 0:
        raise ValueError("the mesh is a point")
    return Frame(rotation=rotation, scale=EXTENT / largest, azimuth=float(azimuth_deg) % 360)


# --- The mesh in the world ----------------------------------------------------------------------------


@dataclass
class _Mesh:
    world: torch.Tensor  # (V, 3)
    faces: torch.Tensor  # (F, 3) long
    normals: torch.Tensor  # (V, 3) unit, shared by the copies of a vertex at UV seams
    face_normals: torch.Tensor  # (F, 3) unit
    flipped: bool = False  # the normals were turned round: the mesh's triangles wind inwards


def _geometry(mesh: Any, frame: Frame, device) -> _Mesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("the mesh has no triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("the mesh's faces don't match its vertices")
    world = torch.tensor(frame.points(vertices), dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device)
    a, b, c = (world[f[:, i]] for i in range(3))
    face_normals = torch.cross(b - a, c - a, dim=-1)
    face_normals = face_normals / face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    geometry = _Mesh(world=world, faces=f, normals=projection._welded_normals(world, f), face_normals=face_normals)
    _orient(geometry, _cameras(device))
    return geometry


def _orient(geometry: _Mesh, cameras: torch.Tensor, size: int = 128) -> None:
    """
    Turns the normals round when the surface the cameras see mostly faces away from them (triangles
    wound inwards): the median over every covered pixel of the six views, at a low resolution.
    """
    facing = []
    for axes in cameras:
        xy, depth = _to_view(geometry.world, axes, size, size)
        ones = torch.ones_like(depth)
        zbuf = rasterize_depth(xy, depth, ones, geometry.faces, size, size)
        face, _ = rasterize_faces(xy, depth, ones, geometry.faces, zbuf)
        seen = face[face >= 0]
        facing.append(geometry.face_normals[seen] @ axes[2])
    facing = torch.cat(facing)
    if facing.numel() and float(facing.median()) < 0:
        geometry.normals = -geometry.normals
        geometry.face_normals = -geometry.face_normals
        geometry.flipped = True


def _texture(mesh: Any):
    """The base colour texture as (H, W, 3) sRGB in [0, 1] (row 0 at v = 1), its alpha (kept as it was) and the UVs."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is None or uv is None:
        raise ValueError("the mesh has no base colour texture or UVs")
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
        raise ValueError("the mesh's UVs don't match its vertices")
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    alpha = np.asarray(image.getchannel("A")).copy() if image.mode in ("RGBA", "LA") else None
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return rgb, alpha, uv


# --- Control maps -------------------------------------------------------------------------------------


@dataclass
class ControlMaps:
    """What upstream renders for ig2mv's control image, per view in upstream's camera order."""

    position: np.ndarray  # (6, H, W, 3) float32: world position + 0.5, clamped to [0, 1]; 0.5 off the mesh
    normal: np.ndarray  # (6, H, W, 3) float32: world normal / 2 + 0.5; 0.5 off the mesh
    mask: np.ndarray  # (6, H, W) bool: where the mesh is
    depth: np.ndarray  # (6, H, W) float32: distance from the camera along its view; +inf off the mesh
    frame: Frame
    flipped: bool = False  # the mesh's normals were turned round (its triangles wind inwards)


def control_maps(mesh: Any, frame: Frame, size: int = SIZE, device: Optional[Any] = None) -> ControlMaps:
    """
    The mesh's position and normal maps from the six cameras, as upstream builds its control image:
    world positions (pos + 0.5) and smooth world normals (normal / 2 + 0.5), both clamped to [0, 1] and
    0.5 where the mesh isn't; with each view's mask and depth. Like nvdiffrast without antialiasing, a
    pixel shows the triangle nearest the camera at its centre.
    """
    device = _device(device)
    with torch.no_grad():
        geometry = _geometry(mesh, frame, device)
        positions, normals, masks, depths = [], [], [], []
        for axes in _cameras(device):
            xy, depth = _to_view(geometry.world, axes, size, size)
            ones = torch.ones_like(depth)
            zbuf = rasterize_depth(xy, depth, ones, geometry.faces, size, size)
            face, bary = rasterize_faces(xy, depth, ones, geometry.faces, zbuf)
            hit = face >= 0
            corners = geometry.faces[face.clamp_min(0)]  # (H, W, 3)
            position = (bary[..., None] * geometry.world[corners]).sum(-2)
            normal = (bary[..., None] * geometry.normals[corners]).sum(-2)
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            positions.append(torch.where(hit[..., None], position, torch.zeros_like(position)))
            normals.append(torch.where(hit[..., None], normal, torch.zeros_like(normal)))
            masks.append(hit)
            depths.append(zbuf)
        position = (torch.stack(positions) + 0.5).clamp(0, 1)
        normal = (torch.stack(normals) / 2 + 0.5).clamp(0, 1)
    return ControlMaps(
        position=position.cpu().numpy().astype(np.float32),
        normal=normal.cpu().numpy().astype(np.float32),
        mask=torch.stack(masks).cpu().numpy(),
        depth=torch.stack(depths).cpu().numpy().astype(np.float32),
        frame=frame,
        flipped=geometry.flipped,
    )


def as_control(maps: ControlMaps) -> np.ndarray:
    """The ig2mv pipeline's ``control_image``: (6, 6, H, W) float32, position channels then normal."""
    return np.ascontiguousarray(np.concatenate([maps.position, maps.normal], -1).transpose(0, 3, 1, 2), dtype=np.float32)


def control_images(maps: ControlMaps) -> tuple[list, list]:
    """The position and normal maps as images, as upstream saves them (``*_pos.png``, ``*_nor.png``)."""

    def images(values: np.ndarray) -> list:
        return [Image.fromarray((v * 255).astype(np.uint8)) for v in values]  # upstream truncates

    return images(maps.position), images(maps.normal)


# --- Colour renders -----------------------------------------------------------------------------------


def render_views(
    mesh: Any, frame: Frame, size: int = SIZE, background: float = BACKGROUND, device: Optional[Any] = None
) -> list:
    """
    The mesh's own base colour from the six cameras on the views' grey: what ig2mv would draw if it
    copied the texture (no light). For checking a bake end to end; RGB images, upstream's camera order.
    """
    device = _device(device)
    rgb, _, uv = _texture(mesh)
    with torch.no_grad():
        geometry = _geometry(mesh, frame, device)
        texture = torch.tensor(rgb, device=device).permute(2, 0, 1)[None]
        uv = torch.tensor(uv, dtype=torch.float32, device=device)
        views = []
        for axes in _cameras(device):
            xy, depth = _to_view(geometry.world, axes, size, size)
            ones = torch.ones_like(depth)
            zbuf = rasterize_depth(xy, depth, ones, geometry.faces, size, size)
            face, bary = rasterize_faces(xy, depth, ones, geometry.faces, zbuf)
            hit = face >= 0
            at = (bary[..., None] * uv[geometry.faces[face.clamp_min(0)]]).sum(-2)  # (H, W, 2), v up
            grid = torch.stack([at[..., 0] * 2 - 1, (1 - at[..., 1]) * 2 - 1], -1)[None]
            colour = F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            colour = torch.where(hit[None], colour, torch.full_like(colour, background)).permute(1, 2, 0)
            views.append(Image.fromarray((colour.cpu().numpy() * 255 + 0.5).astype(np.uint8), "RGB"))
    return views


# --- Baking -------------------------------------------------------------------------------------------


def _background(rgb: torch.Tensor, cover: torch.Tensor, band: int, tolerance: float) -> torch.Tensor:
    """
    Where a view's background shows inside the mesh's silhouette ((h, w) bool), when the view drew the
    object a little smaller than the mesh: pixels of the background's colour (the median outside the
    mesh, clear of its outline) that connect to the outside through such pixels, at most ``band`` pixels
    in. Only so far, so a grey part of the object itself isn't taken for background.
    """
    if band <= 0:
        return torch.zeros_like(cover)
    clear = ~(projection._dilate(cover.float()[None, None], 2)[0, 0] > 0)
    colour = rgb[clear].median(0).values if int(clear.sum()) >= 16 else torch.full((3,), BACKGROUND, device=rgb.device)
    passable = ((rgb - colour).abs().amax(-1) <= tolerance) | ~cover
    reach = ~cover
    for _ in range(band):
        grown = projection._dilate(reach.float()[None, None], 1)[0, 0] > 0
        grown &= passable
        if torch.equal(grown, reach):
            break
        reach = grown
    return reach & cover


def _fade(zbuf: torch.Tensor, usable: torch.Tensor, feather: float, jump: float) -> torch.Tensor:
    """
    A view's (h, w) fade: 0 at the edge of the usable pixels (the mesh's silhouette, less background)
    and at depth edges, rising to 1 at ``feather`` pixels away from both.
    """
    width = int(round(feather))
    if width <= 0:
        return usable.float()
    inside = projection._ramp_inside(usable.float()[None, None], width)[0, 0]
    z = torch.where(torch.isfinite(zbuf), zbuf, torch.full_like(zbuf, 1e3))
    dx = (z[:, 1:] - z[:, :-1]).abs() > jump
    dy = (z[1:] - z[:-1]).abs() > jump
    edges = torch.zeros_like(zbuf, dtype=torch.bool)
    edges[:, 1:] |= dx
    edges[:, :-1] |= dx
    edges[1:] |= dy
    edges[:-1] |= dy
    away = projection._ramp_away(edges.float()[None, None], width)[0, 0]
    return inside * away


def _match_samples(old: torch.Tensor, new: torch.Tensor, weight: torch.Tensor):
    """
    What a colour match may use of one view: the log ratios (M, 3) of the old texture's colours to the
    view's (linear, (N, 3)) and the view's weights (M,), at the texels it sees where both colours are
    colourful and about the same hue. Texels whose colours disagree (a back painted with a ghost of the
    front) don't pull the match; nor do greys, whose hue agrees with any grey.
    """
    luma_old, luma_new = projection._luma(old), projection._luma(new)
    chroma_old = (old - luma_old[:, None]).norm(dim=-1)
    chroma_new = (new - luma_new[:, None]).norm(dim=-1)
    colourful = (chroma_old > 0.2 * luma_old.clamp_min(0.02)) & (chroma_new > 0.2 * luma_new.clamp_min(0.02))
    cos = (old * new).sum(-1) / (old.norm(dim=-1) * new.norm(dim=-1)).clamp_min(1e-9)
    use = (weight > 0) & colourful & (cos >= math.cos(math.radians(MATCH_HUE))) & (luma_old > 0.003) & (luma_new > 0.003)
    return torch.log((old[use] + 1e-4) / (new[use] + 1e-4)), weight[use]


def _gains(ratios: list, weights: list, total: float, max_gain: float) -> Optional[torch.Tensor]:
    """
    Per-channel gains (3,) from match samples (``_match_samples``, one or more views): the weighted
    median ratio, clamped to ``max_gain`` either way. None when they carry under MATCH_AGREEMENT of the
    views' ``total`` weight or number under MATCH_TEXELS.
    """
    if not ratios:
        return None
    ratio, weight = torch.cat(ratios), torch.cat(weights)
    if ratio.shape[0] < MATCH_TEXELS or float(weight.sum()) < MATCH_AGREEMENT * total:
        return None
    if ratio.shape[0] > MATCH_SAMPLES:
        keep = torch.linspace(0, ratio.shape[0] - 1, MATCH_SAMPLES, device=ratio.device).long()
        ratio, weight = ratio[keep], weight[keep]
    median = torch.stack([projection._wquantile(ratio[:, c], weight, 0.5) for c in range(3)])
    return torch.exp(median).clamp(1 / max_gain, max_gain)


def _into_gutters(change: torch.Tensor, covered: torch.Tensor, rings: int = GUTTER_RINGS) -> torch.Tensor:
    """
    A (C, H, W) change of the covered texels carried on into the gutters: ring by ring, each gutter texel
    next to filled ones takes their mean, so every chart's edge continues outwards; texels further out
    are filled coarse to fine (push-pull), as the projection does.
    """
    known = covered.float()[None, None]
    values = (change * known[0])[None]
    for _ in range(rings):
        count = F.avg_pool2d(known, 3, 1, 1)
        grow = (count > 0) & (known == 0)
        if not bool(grow.any()):
            break
        total = F.avg_pool2d(values * known, 3, 1, 1)
        values = torch.where(grow, total / count.clamp_min(1e-9), values)
        known = torch.where(grow, torch.ones_like(known), known)
    return projection._push_pull(values[0], known[0, 0])


def _view_image(view: Any, device) -> torch.Tensor:
    """A view as (H, W, 3) sRGB in [0, 1]: a PIL image, or an (H, W, 3+) array (uint8, or floats in [0, 1])."""
    if isinstance(view, Image.Image):
        array = np.asarray(view.convert("RGB"), dtype=np.float32) / 255
    else:
        array = np.asarray(view)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"a view must be an image or an (H, W, 3) array, not {array.shape}")
        array = array[..., :3].astype(np.float32) / (255 if array.dtype == np.uint8 else 1)
    return torch.tensor(array, device=device).clamp(0, 1)


def bake_views(
    mesh: Any,
    frame: Frame,
    views: Sequence[Any],
    *,
    cos_power: float = COS_POWER,
    min_cos: float = MIN_COS,
    feather: float = FEATHER,
    edge_jump: float = EDGE_JUMP,
    depth_bias: float = DEPTH_BIAS,
    full_weight: float = FULL_WEIGHT,
    front_weight: float = 1.0,
    view_weights: Optional[Sequence[float]] = None,
    match_colour: Union[bool, str] = True,
    max_gain: float = MAX_GAIN,
    background_band: int = BACKGROUND_BAND,
    background_tolerance: float = BACKGROUND_TOLERANCE,
    device: Optional[Any] = None,
    debug: Optional[dict] = None,
) -> dict:
    """
    Bakes six views (images, upstream's camera order; ig2mv draws them 768 x 768) of ``mesh`` placed
    by ``frame`` into its base colour texture, in place: ``mesh.visual.material.baseColorTexture``
    becomes a new image (its alpha kept). Texels no view sees well keep their colour.

    Knobs: ``cos_power`` and ``min_cos`` (how squarely a view must see a texel), ``feather`` (pixels of
    fade near silhouettes and depth edges), ``edge_jump`` (pixels' worth of depth that make an edge),
    ``depth_bias`` (pixels' worth of depth a visible texel may lie behind the nearest surface),
    ``full_weight`` (the sum of weights that replaces a texel's colour in full), ``front_weight`` (times
    the front view's weight; the picture's own projection covers the front after this),
    ``view_weights`` (six more factors), ``match_colour`` (scale the views' colour to the old texture's
    first, by at most ``max_gain`` either way: True or "together" for one set of gains for all six
    views, "each" for a set per view, False for none), ``background_band`` and
    ``background_tolerance`` (the views' background inside the silhouette; a band of 0 turns that off).
    ``debug``, a dict, collects
    per texel (the covered ones, ``flat`` indices into the texture) its world point, each view's weight
    and the amount of the views' colour it took.

    Returns a summary: per view the texels it set (more than half of their colour) and its share of the
    covered texels, its colour gains and background pixels; in all, the texels covered and changed (more
    than half), and seconds.
    """
    started = time.perf_counter()
    if len(views) != len(NAMES):
        raise ValueError(f"expected {len(NAMES)} views (upstream's camera order), got {len(views)}")
    factors = [float(front_weight)] + [1.0] * (len(NAMES) - 1)
    if view_weights is not None:
        if len(view_weights) != len(NAMES):
            raise ValueError(f"expected {len(NAMES)} view weights, got {len(view_weights)}")
        factors = [a * float(b) for a, b in zip(factors, view_weights)]
    match = MATCH_MODES[0] if match_colour is True else (match_colour or None)
    if match is not None and match not in MATCH_MODES:
        raise ValueError(f"match_colour is True, False or one of {MATCH_MODES}, not {match_colour!r}")
    device = _device(device)
    rgb, alpha, uv = _texture(mesh)
    with torch.no_grad():
        geometry = _geometry(mesh, frame, device)
        texture = torch.tensor(rgb, device=device)
        tex_h, tex_w = texture.shape[:2]

        # Every texel's surface point and normal (uv_raster works bottom-up, the image top-down)
        uv_t = torch.tensor(uv, dtype=torch.float32, device=device)
        uv_clip = torch.cat([uv_t * 2 - 1, torch.zeros_like(uv_t[:, :1]), torch.ones_like(uv_t[:, :1])], -1)
        rast, _ = uv_raster.rasterize(None, uv_clip[None], geometry.faces.int(), resolution=[tex_h, tex_w])
        rast = rast.flip(1)
        covered = rast[0, ..., 3] > 0
        flat = torch.nonzero(covered.flatten()).flatten()
        if flat.numel() == 0:
            raise ValueError("the texture has no covered texels")
        attrs = torch.cat([geometry.world, geometry.normals], -1)
        texel = uv_raster.interpolate(attrs[None], rast, geometry.faces.int())[0][0].view(-1, 6)[flat]
        face = rast[0, ..., 3].flatten()[flat].round().long() - 1
        del rast
        points, normals = texel[:, :3], texel[:, 3:]
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        face_normals = geometry.face_normals[face]
        del texel, face
        old = projection._srgb_to_linear(texture.view(-1, 3)[flat])

        total = torch.zeros(flat.numel(), device=device)
        mixed = torch.zeros((flat.numel(), 3), device=device)
        weights, reports = [], []
        pooled_ratios, pooled_weights, pooled_total = [], [], 0.0  # the match samples of all views
        for number, axes in enumerate(_cameras(device)):
            image = _view_image(views[number], device)
            h, w = image.shape[:2]
            pixel = 2 * HALF_EXTENT / max(h, w)  # world units per pixel

            # The view's depth, and which of its pixels may be used
            vxy, vdepth = _to_view(geometry.world, axes, h, w)
            zbuf = rasterize_depth(vxy, vdepth, torch.ones_like(vdepth), geometry.faces, h, w)
            cover = torch.isfinite(zbuf)
            background = _background(image, cover, int(background_band), background_tolerance)
            usable = cover & ~background
            fade = _fade(zbuf, usable, feather, edge_jump * pixel)

            # The texels it sees
            xy, depth = _to_view(points, axes, h, w)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            at = xy[:, 1].long().clamp(0, h - 1) * w + xy[:, 0].long().clamp(0, w - 1)
            cos = normals @ axes[2]
            cos_face = face_normals @ axes[2]
            slope = (1 - cos_face.square()).clamp_min(0).sqrt() / cos_face.abs().clamp_min(0.05)
            slack = pixel * (depth_bias + 0.75 * slope.clamp(max=20.0))
            visible = inside & (depth <= zbuf.flatten()[at] + slack)

            grid = torch.stack([xy[:, 0] / w * 2 - 1, xy[:, 1] / h * 2 - 1], -1)[None, None]
            usable_f = usable.float()[None]
            linear = projection._srgb_to_linear(image).permute(2, 0, 1) * usable_f
            sampled = F.grid_sample(
                torch.cat([linear, usable_f, fade[None]])[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0, :, 0].T  # (N, 5)
            colour = sampled[:, :3] / sampled[:, 3:4].clamp_min(1e-6)
            weight = (
                visible.float()
                * cos.clamp_min(0) ** cos_power
                * projection._smoothstep(min_cos, min_cos + 0.1, cos)
                * projection._smoothstep(0.0, 0.1, cos_face)
                * sampled[:, 4]
                * (sampled[:, 3] > 1e-3)
            )
            gains = None
            if match is not None:
                ratio, sample_weight = _match_samples(old, colour, weight)
                if match == "each":
                    gains = _gains([ratio], [sample_weight], float(weight.sum()), max_gain)
                    if gains is not None:
                        colour = (colour * gains).clamp(0, 1)
                else:
                    pooled_ratios.append(ratio)
                    pooled_weights.append(sample_weight)
                    pooled_total += float(weight.sum())
            weight = weight * factors[number]
            total += weight
            mixed += weight[:, None] * colour
            weights.append(weight)
            reports.append(
                {
                    "name": NAMES[number],
                    "gain": [round(float(g), 3) for g in gains] if gains is not None else None,
                    "background": int(background.sum()),
                }
            )
            del zbuf, cover, background, usable, fade, xy, depth, at, cos, cos_face, slope, slack, visible, grid
            del linear, sampled, colour

        # Views blended by weight; the old colour where they add up to little
        amount = projection._smoothstep(0.0, full_weight, total) if full_weight > 0 else (total > 0).float()
        blend = mixed / total.clamp_min(1e-9)[:, None]
        if match == "together":
            gains = _gains(pooled_ratios, pooled_weights, pooled_total, max_gain)
            if gains is not None:
                blend = (blend * gains).clamp(0, 1)  # the same gains for every view: the same as scaling each
            for report in reports:
                report["gain"] = [round(float(g), 3) for g in gains] if gains is not None else None
            del pooled_ratios, pooled_weights
        new = old + amount[:, None] * (blend - old)
        out = texture.clone().view(-1, 3)
        touched = amount > 0
        out[flat[touched]] = projection._linear_to_srgb(new[touched])
        out = out.view(tex_h, tex_w, 3)
        # The gutters were inpainted from the old colours: carry the change into them
        filled = _into_gutters((out - texture).permute(2, 0, 1), covered)
        out = torch.where(covered[..., None], out, (texture + filled.permute(1, 2, 0)).clamp(0, 1))

        share = [amount * weight / total.clamp_min(1e-9) for weight in weights]
        for report, part in zip(reports, share):
            report["texels"] = int((part > 0.5).sum())
            report["share"] = round(float(part.sum()) / flat.numel(), 4)
        if debug is not None:
            debug.update(
                flat=flat, covered=covered, points=points, amount=amount, weights=torch.stack(weights), total=total,
                texture_size=(tex_h, tex_w), flipped=geometry.flipped,
            )

    image = Image.fromarray((out.cpu().numpy() * 255 + 0.5).astype(np.uint8), "RGB")
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))
    mesh.visual.material.baseColorTexture = image
    return {
        "views": reports,
        "texels": int(flat.numel()),
        "changed": int((amount > 0.5).sum()),
        "texture": [tex_w, tex_h],
        "frame": frame.as_dict(),
        "flipped": geometry.flipped,
        "seconds": round(time.perf_counter() - started, 3),
    }
