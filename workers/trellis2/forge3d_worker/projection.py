"""
Paints the input picture onto the side of a TRELLIS.2 model that the picture shows.

TRELLIS.2 gets a picture's shape right but bakes colour from a coarse voxel field, so painted detail (a
lion on a shield, graffiti, a guitar's pickups) comes out smeared. The picture has that detail. This is
single-view texture back-projection, as dedicated texturing pipelines do it:

1. Camera. TRELLIS.2's output frame relative to the picture is unknown, so views are searched: azimuth
   all round, elevation -20..80 degrees, roll near 0 and three perspective strengths. Each is scored by
   the IoU of the model's silhouette with the picture's, both cropped to their bounding square the way
   ``preprocess_image`` crops, from point splats at low resolution. The best few distinct views are
   refined by pattern search, now also moving and scaling the silhouette in the picture (a shadow left
   in the cutout shifts its bounding square). Symmetric shapes (a cup looks the same from every side)
   are told apart by colour: the model's own texture, which TRELLIS.2 made roughly like the picture on
   the side it saw, is rendered from each view and correlated with the picture. The winner's placement
   is polished against the exact silhouette at the picture's resolution.
2. Confidence. Nothing changes (the mesh comes back as it was, with the reason) when the silhouettes
   don't agree well enough, when a different view fits as well and colour can't tell them apart, or when
   too little of the model would change.
3. Projection. Each texel's 3D point and normal come from rasterising the mesh in UV space (uv_raster).
   A z-buffer from the camera keeps what the picture actually shows. The picture's lighting comes off
   first: where it and the texture show the same paint, their brightness ratio is fitted as a function
   of the surface normal, which matches the exposure and removes the picture's shading. The picture then
   replaces the texture where the surface faces the camera squarely, fading out at grazing angles, near
   the picture's outline, near depth edges (where a slightly wrong camera would put one surface's colour
   on another), near misfits (where the model's silhouette isn't the picture's) and at specular
   highlights. Colour and detail fade differently: the picture's detail only where it was seen well, its
   broad colour (the difference to the texture, smoothed in 3D per paint of the texture) further round
   the sides, so a texture that got a colour wrong doesn't end in a visible seam. The texture's gutters
   are refilled so filtering doesn't bring the old colours back at chart edges.

``project_picture`` never raises: on bad input or low confidence it returns the mesh unchanged and says
why in its report.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import uv_raster

# --- Camera search ------------------------------------------------------------------------------------
# Silhouettes are compared at these sizes (pixels on a side of the picture's bounding square)
SEARCH_SIZE = 64
REFINE_SIZE = 128
COLOUR_SIZE = 64
# The exact silhouettes are compared at most this size (pixels on the picture's longest side)
POLISH_SIZE = 512
# Surface samples splatted per view
SEARCH_POINTS = 30_000
REFINE_POINTS = 100_000
# The grid. Azimuth turns about the up axis (+Y in a GLB; 0 puts the camera on +Z, like the gallery's
# turntable); elevation is above the horizon; roll turns the picture plane. TRELLIS.2 keeps objects
# upright as pictured, so roll stays near 0 (refinement goes up to MAX_ROLL)
AZIMUTHS = tuple(range(0, 360, 10))
ELEVATIONS = tuple(range(-20, 81, 10))
ROLLS = (-8.0, 0.0, 8.0)
# Perspective strength: the reciprocal of the camera's distance in bounding radii (0 is orthographic,
# 0.25 about 29 degrees of field of view around the bounding sphere, 0.5 about 60)
PERSPECTIVES = (0.0, 0.25, 0.5)
MAX_ROLL = 25.0
MAX_PERSPECTIVE = 0.7
# Distinct views refined, and how far apart (rotation angle) two views must be to count as different
CANDIDATES = 6
DISTINCT_DEG = 25.0
REFINE_ROUNDS = 12
# Pattern-search steps: azimuth, elevation, roll (degrees), perspective, log scale, shift x, shift y
# (shares of the picture's bounding square)
REFINE_STEPS = (5.0, 5.0, 4.0, 0.1, 0.03, 0.015, 0.015)
# Colour breaks ties: score = IoU + COLOUR_WEIGHT * colour correlation + FRONT_PRIOR * cos(azimuth).
# The last is a nudge towards TRELLIS.2's canonical front (+Z): on the Phase 4 finals every object with
# a clear front was pictured from within 65 degrees of it, so it settles front-or-back ties (a shield)
COLOUR_WEIGHT = 0.1
FRONT_PRIOR = 0.01

# --- Gate ---------------------------------------------------------------------------------------------
# Silhouette IoU at the picture's resolution: the model must be the pictured shape, or the picture's
# detail lands beside its place. On the Phase 4 finals the models that came out better or the same
# scored 0.93-0.99 (the lowest a guitar at 0.934); two gaming chairs at 0.91, whose bases and backs
# TRELLIS.2 made thinner than pictured, came out worse, and a ramen bowl at 0.93 no better; a warped
# guitar body, whose silhouette no view matches, 0.65-0.68
MIN_IOU = 0.93
# Another view at least DISTINCT_DEG away scoring within AMBIGUOUS of the best is a coin toss, unless it
# sees the same shape (a cup from any side, a donut, a balloon): relief (depth less its best plane) that
# differs by less than SAME_SHAPE of its size. On the Phase 4 finals symmetric shapes measured 0.0-0.23,
# a shield's two near-identical faces 0.19-0.36, other objects seen from far round 0.7-1.1
AMBIGUOUS = 0.01
SAME_SHAPE = 0.4
# Fewer changed texels than this share of the covered ones isn't worth it (or the camera is wrong)
MIN_COVERAGE = 0.02

# --- Projection ---------------------------------------------------------------------------------------
# Detail weight: smoothstep of the cosine between normal and view direction
FACING = (0.15, 0.55)
# Texels this squarely seen say what colour each paint is in the picture
FACING_DATA = (0.3, 0.7)
# Fade-outs near the picture's outline and near depth edges, as a share of the object's size in the
# picture
OUTLINE_FADE = 0.015
EDGE_FADE = 0.015
# ... and near where the model's silhouette and the picture's disagree by more than a sliver: there the
# model isn't the pictured shape, and the picture's detail nearby would land beside its place
MISFIT_SLIVER = 0.004
MISFIT_FADE = 0.04
# Holes in the depth map narrower than about twice this share of the object's size are closed before
# looking for depth edges
PINHOLE = 0.004
# Lighting: the picture is lit, the texture is (roughly) unlit colour. Where the picture will be used
# and the two agree in hue (the same paint), the log of their brightness ratio (linear light) is fitted
# as a function of the surface normal, a + b . n (a robust fit, shrunk towards no shading), and the
# picture is divided by it: that brings its exposure to the texture's and takes out its shading (a
# cabinet's side in shadow, a low-poly fox's facets), which would otherwise be painted on. No
# per-channel gains: a red cabinet would tint a white screen. On the Phase 4 finals the texture came out
# 0.24-1.3x as bright as the picture where they agree; 4-5x for a chrome camera and an armoured knight,
# whose few non-metal parts are in shadow. Outside GAIN_RANGE the comparison isn't trusted and the
# picture is used as it is
GAIN_RANGE = (0.2, 3.0)
SHADING_SHRINK = 0.05
LIGHT_BINS = 6  # per axis of the normal, for the fit
# Texels whose colours point within HUE_AGREEMENT degrees of each other (linear RGB) are the same paint;
# with fewer than MIN_AGREEMENT of the non-metal ones the texture's colour is wrong (Phase 4: 0-15% for
# the "cola" milk tea and an all-black chair, 44-99% for the rest)
HUE_AGREEMENT = 20.0
MIN_AGREEMENT = 0.3
# Specular highlights in the picture (a white addition: picture minus texture about equal in R, G and
# B) are left out where the texture says the surface is glossy or metallic. On matte surfaces white is
# paint: TRELLIS.2's roughness is 0.8-1.0 on cloth, wood and fur, 0.15-0.3 on glass, gloss and visors
GLOSSY = (0.3, 0.5)  # roughness: fully glossy below, matte above
METALLIC = (0.5, 0.9)
# The picture works at this size at most (BiRefNet's cutout is no larger)
MAX_PICTURE = 1024

# --- Paints: what changes all round -------------------------------------------------------------------
# The texture's colours are clustered into paints: k-means over the colours it uses, each about equally
# (so a dragon's few grey claws aren't lumped in with its salmon belly), by hue more than lightness (so
# shading baked into the texture doesn't split a paint); centres closer than PAINT_MERGE are one paint.
# Membership is soft: paints within about 0.2 of each other blend
PAINTS = 8
PAINT_SPREAD = 0.1
PAINT_MERGE = 0.06
# A paint's colour in the picture is read where the picture saw it squarely, outside the picture's
# decals (below: a design on a paint isn't the paint) and where it is the texture's paint all around
# (PURITY: its share of the texels within PURITY_RADIUS of the object's size). A texture's sprinkles,
# scrawls and specks rarely line up with the picture's, which would read them as the paint around them
PURITY_RADIUS = 0.02
PURITY = (0.5, 0.8)
# Colours are compared as CIELAB a*b* at one brightness: a paint's hue and saturation, whatever the light
# on it. A paint takes the picture's colour all round (the texture's own variation kept) when
# - enough of it was seen: RECOLOUR_SEEN of its texels and RECOLOUR_COUNT texels;
# - the picture shows it as one colour: RECOLOUR_ONE of what was seen lies within RECOLOUR_TOLERANCE
#   (delta E) of the median (a potion's chrome in the texture, orange liquid in the picture's body and
#   clear glass in its neck, is not one colour);
# - that colour differs: RECOLOUR_CHANGE (delta E; on the Phase 4 finals a fox's red against the
#   picture's orange 15-37, a cola cup's black against milk tea 37-44, a cabin's logs 9-10, a car's
#   blue 5-8);
# - and isn't only darker, as in shadow: RECOLOUR_DARKER (brightness ratio; a knight's cape lining in
#   the cape's shadow, 0.18), unless the paint is near black (RECOLOUR_DARK, linear brightness).
# Each is a ramp and the paint changes by their product, in full (both at their medians) where all hold
RECOLOUR_SEEN = (0.003, 0.01)
RECOLOUR_COUNT = (300.0, 1000.0)
RECOLOUR_TOLERANCE = 20.0
RECOLOUR_BRIGHTNESS = 2.5
RECOLOUR_ONE = (0.7, 0.85)
RECOLOUR_CHANGE = (10.0, 16.0)
RECOLOUR_DARKER = (0.35, 0.5)
RECOLOUR_DARK = 0.02
# The change is a gain per channel on (texture + RECOLOUR_EPS): shades of the paint stay shades of it
RECOLOUR_EPS = 0.02

# --- Detail: what the picture adds where it sees it ---------------------------------------------------
# The picture's difference from the (recoloured) texture is split, in the picture, into what changes
# within DETAIL_SIGMA of the object's size (detail: lines, lettering, texture; kept) and the rest (light,
# glow, a paint's colour on one side only; left out). The smoothing stays within regions of one of the
# picture's colours (DETAIL_CLASSES k-means clusters in CIELAB, soft over CLASS_SOFTNESS delta E), so a
# region's edge leaves no halo
DETAIL_SIGMA = 0.02
DETAIL_CLASSES = 16
CLASS_SOFTNESS = 6.0
# Decals: regions of one colour that lie inside what the picture shows (DECAL_MARGIN pixels clear of its
# outline, depth edges and misfits), stand out from what is round them (DECAL_CONTRAST, delta E, with an
# edge at least DECAL_SHARPNESS of that), are seen well (DECAL_SEEN: their mean detail weight) and cover
# at most DECAL_AREA of the object: a lion on a shield, letters, a camera's lens. They are painted on
# whole, colour and all, except where the difference is only light (LIGHT_ONLY delta E in a*b* and a
# brightness ratio within LIGHT_RATIO either way: a face in a helmet's shadow), or where decals that
# change the colour (DECAL_CHANGE, delta E) would cover over DECAL_TAKEOVER of a paint (a car's windows,
# white in the texture, show the seats through the glass in the picture: no design on the paint)
DECAL_MARGIN = 2
DECAL_CONTRAST = 12.0
DECAL_SHARPNESS = 0.3
DECAL_SEEN = (0.25, 0.5)
DECAL_AREA = 0.1
LIGHT_RATIO = 3.0
LIGHT_FIT = 0.25
DECAL_GARBLED = 2.0
FACING_DECAL = (0.05, 0.3)


@dataclass(frozen=True)
class View:
    azimuth: float  # degrees
    elevation: float  # degrees
    roll: float  # degrees
    perspective: float  # 1 / camera distance in bounding radii (0 = orthographic)
    log_scale: float = 0.0  # the silhouette's size in the picture, relative to bounding squares matching
    shift_x: float = 0.0  # its offset, as shares of the picture's bounding square
    shift_y: float = 0.0

    @property
    def fov(self) -> float:
        """Field of view (degrees) at which the bounding sphere just fills the frame."""
        return math.degrees(2 * math.asin(min(1.0, self.perspective)))

    def as_dict(self) -> dict:
        return {
            "azimuth": round(self.azimuth % 360, 1),
            "elevation": round(self.elevation, 1),
            "roll": round(self.roll, 1),
            "fov": round(self.fov, 1),
            "scale": round(math.exp(self.log_scale), 3),
            "shift": [round(self.shift_x, 4), round(self.shift_y, 4)],
        }


def _view(p: torch.Tensor) -> View:
    return View(*(float(x) for x in p.tolist()))


# --- Camera -------------------------------------------------------------------------------------------


def view_axes(params: torch.Tensor):
    """Camera right, up and back (towards the camera) unit vectors for (B, >=3) view parameters."""
    az, el, roll = (torch.deg2rad(params[:, i]) for i in range(3))
    ca, sa, ce, se = torch.cos(az), torch.sin(az), torch.cos(el), torch.sin(el)
    zero = torch.zeros_like(az)
    back = torch.stack([sa * ce, se, ca * ce], -1)
    right0 = torch.stack([ca, zero, -sa], -1)
    up0 = torch.stack([-se * sa, ce, -se * ca], -1)
    cr, sr = torch.cos(roll)[:, None], torch.sin(roll)[:, None]
    return cr * right0 + sr * up0, cr * up0 - sr * right0, back


def project(points: torch.Tensor, params: torch.Tensor):
    """
    Points (N, 3), in the model's normalised frame (bounding sphere of radius 1 at the origin), seen from
    each of B views: image x and y (B, N) (x right, y up, in the picture plane through the origin), depth
    (B, N; larger is farther) and the perspective factor s (B, N), which is affine in screen space.
    """
    right, up, back = view_axes(params)
    xc, yc, zc = points @ right.T, points @ up.T, points @ back.T  # (N, B)
    s = 1.0 / (1.0 - params[:, 3] * zc)
    return (xc * s).T, (yc * s).T, (-zc).T, s.T


def view_dirs(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Unit vectors from each point towards the camera of one view, (N, 3)."""
    _, _, back = view_axes(params[:1])
    d = back[0][None] - params[0, 3] * points  # k * (camera - point)
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _boxes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bounding squares (B, 3): centre x, centre y, side, of each view's projected points."""
    xlo, xhi, ylo, yhi = x.amin(1), x.amax(1), y.amin(1), y.amax(1)
    side = torch.maximum(xhi - xlo, yhi - ylo).clamp_min(1e-6)
    return torch.stack([(xlo + xhi) / 2, (ylo + yhi) / 2, side], -1)


def _place(x: torch.Tensor, y: torch.Tensor, params: torch.Tensor, box: Optional[torch.Tensor] = None):
    """
    Projected points to the picture's bounding square, (u, v) in [0, 1] (u right, v down): bounding
    squares matched, then the view's own scale and shift. ``box`` defaults to the points' own.
    """
    box = _boxes(x, y) if box is None else box
    k = torch.exp(params[:, 4:5]) / box[:, 2:]
    u = (x - box[:, :1]) * k + 0.5 + params[:, 5:6]
    v = (box[:, 1:2] - y) * k + 0.5 + params[:, 6:7]
    return u, v


# --- Small image helpers ------------------------------------------------------------------------------


def _erode(mask: torch.Tensor, steps: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, 2 * steps + 1, 1, steps) if steps > 0 else mask


def _dilate(mask: torch.Tensor, steps: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * steps + 1, 1, steps) if steps > 0 else mask


def _ramp_inside(mask: torch.Tensor, width: int) -> torch.Tensor:
    """0 on and outside a binary mask's edge, rising linearly to 1 at `width` pixels inside. (B,1,H,W)."""
    total = torch.zeros_like(mask)
    current = mask
    for _ in range(width):
        current = _erode(current, 1)
        total += current
    return total / max(1, width)


def _ramp_away(edges: torch.Tensor, width: int) -> torch.Tensor:
    """0 on edge pixels, rising linearly to 1 at `width` pixels away. (B,1,H,W)."""
    total = torch.zeros_like(edges)
    current = edges
    for _ in range(width):
        current = _dilate(current, 1)
        total += 1 - current
    return total / max(1, width)


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def _luma(c: torch.Tensor) -> torch.Tensor:
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


def _smoothstep(lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# --- The z-buffer rasterizer --------------------------------------------------------------------------

# float32 edge functions on ~1000 px coordinates are off by ~1e-5 of a small triangle's area; a tighter
# tolerance leaves cracks along shared edges, which would read as depth edges
EDGE_EPS = 1e-4


def _fragments(
    xy: torch.Tensor, depth: torch.Tensor, s: torch.Tensor, faces: torch.Tensor, height: int, width: int, max_candidates: int
):
    """
    The pixels each triangle covers, in batches: (pixel index, depth, face, barycentrics (perspective
    correct, (B, 3))). Triangles are enumerated over their pixel bounding boxes, at most
    ``max_candidates`` pixels at a time.
    """
    device = xy.device
    faces = faces.long()
    if faces.shape[0] == 0 or height <= 0 or width <= 0:
        return
    px, py = xy[:, 0] - 0.5, xy[:, 1] - 0.5  # integers are pixel centres
    x0, x1, x2 = px[faces[:, 0]], px[faces[:, 1]], px[faces[:, 2]]
    y0, y1, y2 = py[faces[:, 0]], py[faces[:, 1]], py[faces[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    xmin = torch.ceil(torch.minimum(torch.minimum(x0, x1), x2)).clamp(min=0).long()
    xmax = torch.floor(torch.maximum(torch.maximum(x0, x1), x2)).clamp(max=width - 1).long()
    ymin = torch.ceil(torch.minimum(torch.minimum(y0, y1), y2)).clamp(min=0).long()
    ymax = torch.floor(torch.maximum(torch.maximum(y0, y1), y2)).clamp(max=height - 1).long()
    nx = (xmax - xmin + 1).clamp(min=0)
    ny = (ymax - ymin + 1).clamp(min=0)
    counts = torch.where(area.abs() > 1e-12, nx * ny, torch.zeros_like(nx))
    ends = torch.cumsum(counts, 0)
    total_all = int(ends[-1])
    first, num = 0, faces.shape[0]
    while first < num and total_all > 0:
        base = int(ends[first - 1]) if first > 0 else 0
        if base >= total_all:
            break
        last = int(torch.searchsorted(ends, torch.tensor(base + max_candidates, device=device), right=True))
        last = max(last, first + 1)
        ids = torch.arange(first, last, device=device)
        cnt = counts[first:last]
        total = int(cnt.sum())
        first = last
        if total == 0:
            continue
        local = torch.repeat_interleave(torch.arange(ids.numel(), device=device), cnt)
        start = torch.cumsum(cnt, 0) - cnt
        k = torch.arange(total, device=device) - start[local]
        t = ids[local]
        wbox = nx[t]
        xs = (xmin[t] + k % wbox).float()
        ys = (ymin[t] + torch.div(k, wbox, rounding_mode="floor")).float()
        f = faces[t]
        ax, ay, bx, by, cx, cy = px[f[:, 0]], py[f[:, 0]], px[f[:, 1]], py[f[:, 1]], px[f[:, 2]], py[f[:, 2]]
        a = area[t]
        w0 = ((cx - bx) * (ys - by) - (cy - by) * (xs - bx)) / a
        w1 = ((ax - cx) * (ys - cy) - (ay - cy) * (xs - cx)) / a
        w2 = 1 - w0 - w1
        inside = (w0 >= -EDGE_EPS) & (w1 >= -EDGE_EPS) & (w2 >= -EDGE_EPS)
        if not bool(inside.any()):
            continue
        f, t = f[inside], t[inside]
        weighted = torch.stack([w0[inside] * s[f[:, 0]], w1[inside] * s[f[:, 1]], w2[inside] * s[f[:, 2]]], -1)
        q = weighted.sum(-1)
        d = (weighted * depth[f]).sum(-1) / q
        pix = ys[inside].long() * width + xs[inside].long()
        yield pix, d, t, weighted / q[:, None]


def rasterize_depth(
    xy: torch.Tensor,
    depth: torch.Tensor,
    s: torch.Tensor,
    faces: torch.Tensor,
    height: int,
    width: int,
    max_candidates: int = 1 << 22,
) -> torch.Tensor:
    """
    Nearest depth at each pixel centre of an (height, width) image, +inf where nothing is.

    ``xy`` (V, 2) are pixel coordinates (x to the right, y down; pixel (r, c) has its centre at
    (c + 0.5, r + 0.5)), ``depth`` (V,) the depth to compare and ``s`` (V,) the perspective factor, which
    is affine in screen space: depth is interpolated perspective-correctly as sum(b s depth) / sum(b s).
    Triangles are enumerated over their pixel bounding boxes in batches of at most ``max_candidates``.
    """
    zbuf = torch.full((max(0, height) * max(0, width),), float("inf"), device=xy.device)
    for pix, d, _, _ in _fragments(xy, depth, s, faces, height, width, max_candidates):
        zbuf.scatter_reduce_(0, pix, d, reduce="amin", include_self=True)
    return zbuf.view(height, width)


def rasterize_faces(
    xy: torch.Tensor,
    depth: torch.Tensor,
    s: torch.Tensor,
    faces: torch.Tensor,
    zbuf: torch.Tensor,
    max_candidates: int = 1 << 22,
):
    """
    The triangle ``rasterize_depth`` found nearest at each pixel ((H, W), -1 where none) and the pixel
    centre's perspective-correct barycentrics in it ((H, W, 3)).
    """
    height, width = zbuf.shape
    nearest = zbuf.flatten()
    face = torch.full((height * width,), -1, dtype=torch.long, device=xy.device)
    bary = torch.zeros((height * width, 3), device=xy.device)
    for pix, d, t, b in _fragments(xy, depth, s, faces, height, width, max_candidates):
        win = d <= nearest[pix]
        face[pix[win]] = t[win]
        bary[pix[win]] = b[win]
    return face.view(height, width), bary.view(height, width, 3)


# --- Inputs -------------------------------------------------------------------------------------------


class _Skip(Exception):
    """Raised inside project_picture to return the mesh unchanged with a reason."""


@dataclass
class _Model:
    verts: torch.Tensor  # (V, 3) normalised: bounding sphere radius 1 at the origin
    faces: torch.Tensor  # (F, 3) long
    uv: torch.Tensor  # (V, 2) trimesh convention (v up)
    normals: torch.Tensor  # (V, 3) unit, continuous across UV seams
    texture: torch.Tensor  # (H, W, 3) sRGB in [0, 1], row 0 at the top (v = 1)
    texture_alpha: Optional[np.ndarray]  # the texture's alpha channel, kept as it was
    roughness_metallic: torch.Tensor  # (H', W', 2), the same atlas (1 x 1 without a texture)


@dataclass
class _Picture:
    rgb: torch.Tensor  # (h, w, 3) sRGB [0, 1]
    mask: torch.Tensor  # (h, w) float {0, 1}: alpha > 0.5
    box: tuple[float, float, float]  # centre x, centre y, side (pixels), of the mask's bounding square

    def to_pixels(self, u: torch.Tensor, v: torch.Tensor):
        cx, cy, side = self.box
        return cx + (u - 0.5) * side, cy + (v - 0.5) * side


def _welded_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, shared by vertices at the same position (UV seams split them)."""
    key = torch.round(verts * 1e5).long()
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)
    acc = torch.zeros((int(inverse.max()) + 1, 3), dtype=verts.dtype, device=verts.device)
    for i in range(3):
        acc.index_add_(0, inverse[faces[:, i]], fn)
    n = acc[inverse]
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _load_model(mesh: Any, device) -> _Model:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is None or uv is None:
        raise _Skip("the mesh has no base colour texture or UVs")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3 or faces.ndim != 2 or len(faces) == 0:
        raise _Skip("the mesh is empty")
    if uv.shape != (len(verts), 2) or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(verts):
        raise _Skip("the mesh's UVs or faces don't match its vertices")
    if not (np.isfinite(verts).all() and np.isfinite(uv).all()):
        raise _Skip("the mesh has non-finite vertices or UVs")
    centre = (verts.min(0) + verts.max(0)) / 2
    radius = float(np.linalg.norm(verts - centre, axis=1).max())
    if radius <= 0:
        raise _Skip("the mesh is a point")
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    if min(image.size) < 2:
        raise _Skip("the base colour texture is empty")
    alpha = np.asarray(image.getchannel("A")).copy() if image.mode in ("RGBA", "LA") else None
    v = torch.tensor((verts - centre) / radius, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device)
    return _Model(
        verts=v,
        faces=f,
        uv=torch.tensor(uv, dtype=torch.float32, device=device),
        normals=_welded_normals(v, f),
        texture=torch.tensor(np.asarray(image.convert("RGB")), device=device).float() / 255,
        texture_alpha=alpha,
        roughness_metallic=_roughness_metallic(material, device),
    )


def _roughness_metallic(material: Any, device) -> torch.Tensor:
    """glTF packing: roughness in green, metallic in blue, each times its factor."""
    image = getattr(material, "metallicRoughnessTexture", None)
    rough = getattr(material, "roughnessFactor", None)
    metal = getattr(material, "metallicFactor", None)
    # glTF defaults both factors to 1; without a texture, an unset metallic means "not metal" here
    rough = 1.0 if rough is None else float(rough)
    metal = (1.0 if image is not None else 0.0) if metal is None else float(metal)
    if image is None:
        return torch.tensor([[[rough, metal]]], device=device)
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    gb = torch.tensor(np.asarray(image.convert("RGB")), device=device)[..., 1:3].float() / 255
    return gb * torch.tensor([rough, metal], device=device)


def _load_picture(picture: Any, device) -> _Picture:
    if not isinstance(picture, Image.Image):
        raise _Skip("no picture")
    if picture.mode not in ("RGBA", "LA", "PA") and "transparency" not in picture.info:
        raise _Skip("the picture has no alpha (background removal didn't run)")
    image = picture.convert("RGBA")
    if max(image.size) > MAX_PICTURE:
        scale = MAX_PICTURE / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    rgba = torch.tensor(np.asarray(image), device=device).float() / 255
    mask = (rgba[..., 3] > 0.5).float()
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    if rows.numel() == 0 or float(mask.mean()) < 0.002:
        raise _Skip("no object in the picture")
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    side = max(bottom - top, right - left)
    if side < 32:
        raise _Skip("the object is too small in the picture")
    return _Picture(rgb=rgba[..., :3], mask=mask, box=((left + right) / 2, (top + bottom) / 2, float(side)))


def _square_crop(picture: _Picture, channels: torch.Tensor, size: int) -> torch.Tensor:
    """(C, h, w) picture channels inside the mask's bounding square, area-resampled to (C, size, size)."""
    cx, cy, side = picture.box
    h, w = channels.shape[1:]
    side_i = int(round(side))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = torch.zeros((channels.shape[0], side_i, side_i), device=channels.device)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side_i), min(h, y0 + side_i)
    canvas[:, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = channels[:, sy0:sy1, sx0:sx1]
    return F.interpolate(canvas[None], size=(size, size), mode="area")[0]


# --- Camera search ------------------------------------------------------------------------------------


@dataclass
class _Samples:
    points: torch.Tensor  # (N, 3)
    normals: torch.Tensor  # (N, 3)
    colours: torch.Tensor  # (N, 3) sRGB


def _surface_samples(model: _Model, count: int, seed: int = 0) -> _Samples:
    """Area-weighted samples on the surface, with the vertices themselves (the silhouette's extremes)."""
    v, f = model.verts, model.faces
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = torch.cross(b - a, c - a, dim=-1)
    area = cross.norm(dim=-1)
    fn = cross / area[:, None].clamp_min(1e-12)
    gen = torch.Generator().manual_seed(seed)
    keep = torch.randperm(v.shape[0], generator=gen)[:count].to(v.device)
    pts, uvs, nrm = [v[keep]], [model.uv[keep]], [model.normals[keep]]
    extra = count - keep.numel()
    if extra > 0 and float(area.sum()) > 0:
        pick = torch.multinomial(area.cpu().double(), extra, replacement=True, generator=gen).to(v.device)
        r1 = torch.rand(extra, generator=gen).to(v.device).sqrt()
        r2 = torch.rand(extra, generator=gen).to(v.device)
        w = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], -1)[..., None]
        fp = f[pick]
        pts.append((w * v[fp]).sum(1))
        uvs.append((w * model.uv[fp]).sum(1))
        nrm.append(fn[pick])
    uvs = torch.cat(uvs)
    return _Samples(points=torch.cat(pts), normals=torch.cat(nrm), colours=_sample_texture(_small_texture(model), uvs))


def _small_texture(model: _Model, size: int = 256) -> torch.Tensor:
    tex = model.texture.permute(2, 0, 1)[None]
    if max(tex.shape[-2:]) > size:
        tex = F.interpolate(tex, size=(size, size), mode="area")
    return tex


def _sample_texture(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear samples (N, 3) of a (1, 3, H, W) texture at trimesh UVs (v up)."""
    grid = torch.stack([uv[:, 0] * 2 - 1, (1 - uv[:, 1]) * 2 - 1], -1)[None, None]
    return F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def _silhouettes(points: torch.Tensor, params: torch.Tensor, size: int, batch: int = 64) -> torch.Tensor:
    """Point-splat silhouettes (B, size, size) in the picture's bounding square."""
    out = []
    for i in range(0, params.shape[0], batch):
        p = params[i : i + batch]
        x, y, _, _ = project(points, p)
        u, v = _place(x, y, p)
        b = p.shape[0]
        col, row = (u * size).floor(), (v * size).floor()
        ok = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        flat = torch.arange(b, device=points.device)[:, None] * size * size + row.long() * size + col.long()
        occ = torch.zeros(b * size * size, device=points.device)
        occ[flat[ok]] = 1.0
        occ = occ.view(b, 1, size, size)
        # Close pinholes between splats; real holes (a mug's handle) are wider than this
        occ = _erode(_dilate(occ, 1), 1)
        out.append(occ[:, 0])
    return torch.cat(out)


def _iou(silhouettes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = torch.minimum(silhouettes, target).sum(dim=(-2, -1))
    union = torch.maximum(silhouettes, target).sum(dim=(-2, -1)).clamp_min(1e-6)
    return inter / union


def _colour_renders(samples: _Samples, params: torch.Tensor, size: int):
    """The model's own colours from each view (B, 3, size, size), and where they were seen (B, size, size)."""
    images, seen = [], []
    for i in range(params.shape[0]):
        p = params[i : i + 1]
        x, y, depth, _ = project(samples.points, p)
        u, v = _place(x, y, p)
        facing = (samples.normals * view_dirs(samples.points, p)).sum(-1) > 0.05
        col, row = (u[0] * size).floor(), (v[0] * size).floor()
        ok = facing & (col >= 0) & (col < size) & (row >= 0) & (row < size)
        flat = (row.clamp(0, size - 1) * size + col.clamp(0, size - 1)).long()
        z = torch.where(ok, depth[0], torch.full_like(depth[0], float("inf")))
        zbuf = torch.full((size * size,), float("inf"), device=p.device)
        zbuf.scatter_reduce_(0, flat, z, reduce="amin", include_self=True)
        # Within ~2 pixels' depth of the nearest splat counts as the same surface
        visible = ok & (z <= zbuf[flat] + 4.0 / size)
        acc = torch.zeros((size * size, 3), device=p.device).index_add_(0, flat[visible], samples.colours[visible])
        cnt = torch.zeros(size * size, device=p.device).index_add_(0, flat[visible], torch.ones_like(z[visible]))
        images.append((acc / cnt.clamp_min(1)[:, None]).T.reshape(3, size, size))
        seen.append((cnt > 0).float().view(size, size))
    return torch.stack(images), torch.stack(seen)


def _colour_scores(renders: torch.Tensor, seen: torch.Tensor, picture_rgb: torch.Tensor, picture_mask: torch.Tensor):
    """Zero-mean normalised cross-correlation of colours where both are known, per view (B,)."""
    valid = seen * (picture_mask[None] > 0.5).float()  # (B, s, s)
    n = valid.sum(dim=(1, 2)).clamp_min(1)
    pic = picture_rgb[None].expand_as(renders)
    mr = (renders * valid[:, None]).sum(dim=(2, 3)) / n[:, None]
    mp = (pic * valid[:, None]).sum(dim=(2, 3)) / n[:, None]
    dr = (renders - mr[..., None, None]) * valid[:, None]
    dp = (pic - mp[..., None, None]) * valid[:, None]
    num = (dr * dp).sum(dim=(1, 2, 3))
    den = (dr.square().sum(dim=(1, 2, 3)) * dp.square().sum(dim=(1, 2, 3))).sqrt().clamp_min(1e-9)
    return num / den


def _rotation(params: torch.Tensor) -> torch.Tensor:
    right, up, back = view_axes(params)
    return torch.stack([right, up, back], -1)  # (B, 3, 3), columns


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle (degrees) between the camera frames of view a (1, P) and views b (B, P)."""
    rel = _rotation(a).transpose(1, 2) @ _rotation(b)
    tr = rel.diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def _distinct(params: torch.Tensor, scores: torch.Tensor, count: int, apart: float) -> list[int]:
    """The best-scoring views, each at least `apart` degrees from those before it (greedy)."""
    left = scores.clone()
    chosen: list[int] = []
    for _ in range(count):
        i = int(torch.argmax(left))
        if not torch.isfinite(left[i]):
            break
        chosen.append(i)
        left[_angle_between(params[i : i + 1], params) <= apart] = -float("inf")
    return chosen


def _clamp_params(p: torch.Tensor) -> torch.Tensor:
    p = p.clone()
    p[..., 1] = p[..., 1].clamp(-60, 85)
    p[..., 2] = p[..., 2].clamp(-MAX_ROLL, MAX_ROLL)
    p[..., 3] = p[..., 3].clamp(0, MAX_PERSPECTIVE)
    p[..., 4] = p[..., 4].clamp(-0.4, 0.4)
    p[..., 5:7] = p[..., 5:7].clamp(-0.25, 0.25)
    return p


def _refine(points, start: torch.Tensor, target: torch.Tensor, size: int, rounds: int):
    """Pattern search on all view parameters from each start view (C, 7)."""
    current = start.clone()
    best = _iou(_silhouettes(points, current, size), target)
    dims = current.shape[1]
    steps = torch.tensor(REFINE_STEPS, device=start.device).repeat(current.shape[0], 1)
    for _ in range(rounds):
        trials = []
        for d in range(dims):
            for sign in (-1.0, 1.0):
                t = current.clone()
                t[:, d] += sign * steps[:, d]
                trials.append(_clamp_params(t))
        trial = torch.stack(trials, 1)  # (C, 2 * dims, 7)
        c, n = trial.shape[:2]
        scores = _iou(_silhouettes(points, trial.view(-1, dims), size), target).view(c, n)
        top, which = scores.max(dim=1)
        better = top > best + 1e-4
        current = torch.where(better[:, None], trial[torch.arange(c), which], current)
        best = torch.where(better, top, best)
        steps = torch.where(better[:, None], steps, steps / 2)
    return current, best


@dataclass
class _Pose:
    params: torch.Tensor  # (1, 7)
    iou: float  # silhouette IoU at REFINE_SIZE (point splats)
    colour: float
    score: float
    runner_up: Optional[dict]  # the best view at least DISTINCT_DEG away, for the report
    rivals: torch.Tensor  # (R, 7): such views scoring within AMBIGUOUS of the best
    candidates: list


def _estimate_pose(model: _Model, picture: _Picture, clock) -> _Pose:
    device = model.verts.device
    mask_ch = picture.mask[None]
    target_search = _square_crop(picture, mask_ch, SEARCH_SIZE)[0]
    target_refine = _square_crop(picture, mask_ch, REFINE_SIZE)[0]
    rgb_premul = picture.rgb.permute(2, 0, 1) * picture.mask[None]
    colour_mask = _square_crop(picture, mask_ch, COLOUR_SIZE)[0]
    colour_target = _square_crop(picture, rgb_premul, COLOUR_SIZE) / colour_mask.clamp_min(1e-6)[None]

    coarse = _surface_samples(model, SEARCH_POINTS, seed=1)
    grid = torch.tensor(
        [[a, e, r, k, 0.0, 0.0, 0.0] for a in AZIMUTHS for e in ELEVATIONS for r in ROLLS for k in PERSPECTIVES],
        dtype=torch.float32,
        device=device,
    )
    grid_iou = _iou(_silhouettes(coarse.points, grid, SEARCH_SIZE), target_search)
    clock("search_s")

    starts = _distinct(grid, grid_iou, CANDIDATES, DISTINCT_DEG)
    fine = _surface_samples(model, REFINE_POINTS, seed=2)
    refined, refined_iou = _refine(fine.points, grid[starts], target_refine, REFINE_SIZE, REFINE_ROUNDS)
    clock("refine_s")

    renders, seen = _colour_renders(fine, refined, COLOUR_SIZE)
    colour = _colour_scores(renders, seen, colour_target, colour_mask)
    score = refined_iou + COLOUR_WEIGHT * colour + FRONT_PRIOR * torch.cos(torch.deg2rad(refined[:, 0]))
    order = torch.argsort(score, descending=True).tolist()
    best = order[0]
    runner_up, rivals = None, []
    for i in order[1:]:
        angle = float(_angle_between(refined[best : best + 1], refined[i : i + 1])[0])
        if angle <= DISTINCT_DEG:
            continue
        if runner_up is None:
            runner_up = {
                "view": _view(refined[i]).as_dict(),
                "iou": round(float(refined_iou[i]), 4),
                "colour": round(float(colour[i]), 4),
                "score_gap": round(float(score[best] - score[i]), 4),
                "angle": round(angle, 1),
            }
        if float(score[i]) >= float(score[best]) - AMBIGUOUS:
            rivals.append(i)
    candidates = [
        {
            "view": _view(refined[i]).as_dict(),
            "iou": round(float(refined_iou[i]), 4),
            "colour": round(float(colour[i]), 4),
        }
        for i in order
    ]
    clock("colour_s")
    return _Pose(
        params=refined[best : best + 1],
        iou=float(refined_iou[best]),
        colour=float(colour[best]),
        score=float(score[best]),
        runner_up=runner_up,
        rivals=refined[rivals],
        candidates=candidates,
    )


# --- Exact placement ----------------------------------------------------------------------------------


class _Mapping:
    """Model points to picture pixels for one view: the picture's bounding square and the view's placement."""

    def __init__(self, model: _Model, picture: _Picture, params: torch.Tensor) -> None:
        self.picture = picture
        self.params = params
        x, y, _, _ = project(model.verts, params)
        self.box = _boxes(x, y)
        # Picture pixels per normalised model unit
        self.pixels_per_unit = float(picture.box[2] * torch.exp(params[0, 4]) / self.box[0, 2])

    def __call__(self, points: torch.Tensor):
        x, y, depth, s = project(points, self.params)
        u, v = _place(x, y, self.params, self.box)
        col, row = self.picture.to_pixels(u[0], v[0])
        return torch.stack([col, row], -1), depth[0], s[0]


def _render_depth(model: _Model, picture: _Picture, params: torch.Tensor):
    mapping = _Mapping(model, picture, params)
    xy, depth, s = mapping(model.verts)
    h, w = picture.mask.shape
    return mapping, rasterize_depth(xy, depth, s, model.faces, h, w)


def _shape_difference(za: torch.Tensor, zb: torch.Tensor) -> float:
    """
    How differently two views see the model (0: the same shape), from their depth maps in the picture's
    frame. Each map's best-fitting plane comes off first: turning a view a little mostly tilts that
    plane, which doesn't change where the picture lands; relief and curvature do.
    """
    both = torch.isfinite(za) & torch.isfinite(zb)
    if int(both.sum()) < 100:
        return float("inf")
    rows, cols = torch.nonzero(both, as_tuple=True)
    size = float(max(za.shape))
    basis = torch.stack([cols.float() / size, rows.float() / size, torch.ones_like(cols, dtype=torch.float32)], -1)

    def relief(z: torch.Tensor) -> torch.Tensor:
        values = z[both][:, None]
        plane = torch.linalg.lstsq(basis.cpu(), values.cpu()).solution.to(z.device)
        return (values - basis @ plane)[:, 0]

    ra, rb = relief(za), relief(zb)
    # Normalised by the larger relief, but never below 2% of the bounding radius (a flat plate's
    # relief is noise)
    scale = max(float(ra.abs().mean()), float(rb.abs().mean()), 0.02)
    return float((ra - rb).abs().mean()) / scale


def _polish(model: _Model, picture: _Picture, params: torch.Tensor, rounds: int = 10) -> torch.Tensor:
    """
    Fine-tune the view's scale and shift against the exact silhouette. Moving and scaling a rendered
    silhouette is the same as moving and scaling the view's placement, so it is rendered once and warped.
    """
    _, zbuf = _render_depth(model, picture, params)
    h, w = zbuf.shape
    shrink = max(1, math.ceil(max(h, w) / POLISH_SIZE))
    model_mask = F.avg_pool2d(torch.isfinite(zbuf).float()[None, None], shrink)[0, 0]
    target = F.avg_pool2d(picture.mask[None, None], shrink)[0, 0]
    hs, ws = model_mask.shape
    cx, cy, side = picture.box
    base = params[0].clone()
    # A view's placement scales about its shifted centre (pixels)
    c0x, c0y = cx + float(base[5]) * side, cy + float(base[6]) * side
    cols = (torch.arange(ws, device=zbuf.device).float() + 0.5) * shrink
    rows = (torch.arange(hs, device=zbuf.device).float() + 0.5) * shrink

    def warped(trial: torch.Tensor) -> torch.Tensor:  # (T, 7) -> (T, hs, ws)
        r = torch.exp(trial[:, 4] - base[4])
        c1x = cx + trial[:, 5] * side
        c1y = cy + trial[:, 6] * side
        sx = c0x + (cols[None] - c1x[:, None]) / r[:, None]  # (T, ws) source column
        sy = c0y + (rows[None] - c1y[:, None]) / r[:, None]
        gx = (sx / (ws * shrink)) * 2 - 1
        gy = (sy / (hs * shrink)) * 2 - 1
        grid = torch.stack(torch.broadcast_tensors(gx[:, None, :], gy[:, :, None]), -1)
        return F.grid_sample(model_mask[None, None].expand(trial.shape[0], 1, hs, ws), grid, align_corners=False)[:, 0]

    current = base.clone()
    best = float(_iou(warped(current[None])[0], target))
    steps = torch.tensor([0.0, 0, 0, 0, 0.01, 0.005, 0.005], device=zbuf.device)
    for _ in range(rounds):
        trials = []
        for d in (4, 5, 6):
            for sign in (-1.0, 1.0):
                t = current.clone()
                t[d] += sign * steps[d]
                trials.append(t)
        trial = torch.stack(trials)
        scores = _iou(warped(trial), target[None])
        top = int(torch.argmax(scores))
        if float(scores[top]) > best + 1e-5:
            current, best = trial[top], float(scores[top])
        else:
            steps = steps / 2
    return current[None]


# --- Projection ---------------------------------------------------------------------------------------


def _push_pull(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Fill unknown texels (known == 0) of (C, H, W) values from known ones, coarse to fine."""
    levels = []
    v, k = values * known[None], known  # weighted sums and weights, averaged level by level
    while True:
        levels.append((v, k))
        if min(k.shape) <= 1:
            break
        if k.shape[-1] % 2 or k.shape[-2] % 2:
            v = F.pad(v, (0, k.shape[-1] % 2, 0, k.shape[-2] % 2))
            k = F.pad(k, (0, k.shape[-1] % 2, 0, k.shape[-2] % 2))
        v, k = F.avg_pool2d(v[None], 2)[0], F.avg_pool2d(k[None, None], 2)[0, 0]
    filled = levels[-1][0] / levels[-1][1].clamp_min(1e-12)[None]
    for v, k in reversed(levels[:-1]):
        up = F.interpolate(filled[None], scale_factor=2, mode="bilinear", align_corners=False)[0]
        up = up[:, : k.shape[-2], : k.shape[-1]]
        filled = torch.where(k[None] > 0, v / k.clamp_min(1e-12)[None], up)
    return filled


def _gaussian(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, C, H, W) images, zeros beyond their edges."""
    if sigma <= 0:
        return images
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    b, c, h, w = images.shape
    x = images.reshape(1, b * c, h, w)
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).expand(b * c, 1, 1, -1).contiguous(), padding=(0, radius), groups=b * c)
    x = F.conv2d(x, kernel.view(1, 1, -1, 1).expand(b * c, 1, -1, 1).contiguous(), padding=(radius, 0), groups=b * c)
    return x.view(b, c, h, w)


def _smooth(values: torch.Tensor, weight: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Weighted Gaussian average: blur(values * weight) / blur(weight), for (B, C, H, W) values and (B, 1,
    H, W) weights. Worked out on a grid coarse enough for sigma to span two of its cells, then
    interpolated back: the averages are smooth.
    """
    h, w = values.shape[-2:]
    step = max(1, int(sigma // 2))
    num, den = values * weight, weight
    if step > 1:
        num = F.avg_pool2d(num, step, ceil_mode=True)
        den = F.avg_pool2d(den, step, ceil_mode=True)
    num, den = _gaussian(num, sigma / step), _gaussian(den, sigma / step)
    if step > 1:
        num = F.interpolate(num, size=(h, w), mode="bilinear", align_corners=False)
        den = F.interpolate(den, size=(h, w), mode="bilinear", align_corners=False)
    return num / den.clamp_min(1e-6)


_XYZ = ((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505))
_WHITE = (0.9505, 1.0, 1.089)


def _lab(linear: torch.Tensor) -> torch.Tensor:
    """CIELAB (..., 3) of linear sRGB colours (..., 3), D65 white."""
    matrix = torch.tensor(_XYZ, device=linear.device, dtype=linear.dtype)
    t = (linear @ matrix.T) / torch.tensor(_WHITE, device=linear.device, dtype=linear.dtype)
    f = torch.where(t > 0.008856, t.clamp_min(1e-12) ** (1 / 3), 7.787 * t + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _hue(linear: torch.Tensor, brightness: float = 0.2) -> torch.Tensor:
    """
    CIELAB a*b* (..., 2) of linear colours brought to one brightness: their hue and saturation, whatever
    the light on them. Near-black reads as neutral.
    """
    c = linear.clamp_min(0) + 0.004
    return _lab(c * (brightness / _luma(c))[..., None])[..., 1:]


def _ramp(edges: tuple[float, float], x: float) -> float:
    return float(_smoothstep(edges[0], edges[1], torch.tensor(float(x))))


def _wquantile(x: torch.Tensor, w: torch.Tensor, q: float) -> torch.Tensor:
    """The weighted q-quantile of x (N,), weights w (N,)."""
    order = torch.argsort(x)
    total = torch.cumsum(w[order], 0)
    i = int(torch.searchsorted(total, (q * total[-1]).reshape(1))[0].clamp(max=x.numel() - 1))
    return x[order[i]]


def _hex(linear: torch.Tensor) -> str:
    return "#" + "".join(f"{round(float(c) * 255):02x}" for c in _linear_to_srgb(linear))


# --- Paints -------------------------------------------------------------------------------------------


def _paint_features(srgb: torch.Tensor) -> torch.Tensor:
    """
    Where a colour (N, 3, sRGB) sits among paints: its red-green and yellow-blue opponents, and its
    lightness at a third of the weight, so the same paint lit or shaded differently in the texture (a
    bowl's white going grey underneath) stays one paint while differently coloured ones don't.
    """
    r, g, b = srgb.unbind(-1)
    return torch.stack([0.3 * (r + g + b) / 3, r - g, (r + g) / 2 - b], -1)


def _paints(colours: torch.Tensor, count: int, levels: int = 16, rounds: int = 10) -> torch.Tensor:
    """
    The texture's main paints: k-means centres (at most count, 3), in _paint_features, of ``colours``
    (N, 3, sRGB). Over the colours used rather than the texels: colours are binned (``levels`` per
    channel) and each occupied bin weighs the square root of its texel count, so a small part's colour
    gets a centre of its own. Centres closer than PAINT_MERGE are merged.
    """
    key = (colours.clamp(0, 1) * (levels - 1)).round().long()
    key = (key[:, 0] * levels + key[:, 1]) * levels + key[:, 2]
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    x = torch.zeros((counts.numel(), 3), device=colours.device).index_add_(0, inverse, colours)
    x = _paint_features(x / counts[:, None])
    weight = counts.float().sqrt()
    gen = torch.Generator().manual_seed(0)
    centres = x[int(torch.argmax(weight)) :][:1]
    for _ in range(count - 1):  # k-means++ seeding: each next centre far from those so far
        d = torch.cdist(x, centres).amin(1).square() * weight
        if float(d.sum()) <= 0:
            break
        pick = int(torch.multinomial((d / d.sum()).cpu().double(), 1, generator=gen))
        centres = torch.cat([centres, x[pick : pick + 1]])
    for _ in range(rounds):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x * weight[:, None])
        total = torch.zeros(centres.shape[0], device=x.device).index_add_(0, assign, weight)
        centres = torch.where(total[:, None] > 0, sums / total.clamp_min(1e-9)[:, None], centres)
    assign = torch.cdist(x, centres).argmin(1)
    total = torch.zeros(centres.shape[0], device=x.device).index_add_(0, assign, weight)
    centres, total = centres[total > 0], total[total > 0]
    while centres.shape[0] > 1:
        d = torch.cdist(centres, centres)
        d.fill_diagonal_(float("inf"))
        a, b = divmod(int(torch.argmin(d)), centres.shape[0])
        if float(d[a, b]) >= PAINT_MERGE:
            break
        merged = (centres[a] * total[a] + centres[b] * total[b]) / (total[a] + total[b])
        keep = [i for i in range(centres.shape[0]) if i not in (a, b)]
        centres = torch.cat([centres[keep], merged[None]])
        total = torch.cat([total[keep], (total[a] + total[b])[None]])
    return centres


def _membership(colours: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Soft membership (N, K) of each colour (sRGB) in each paint (_paints' centres), summing to 1."""
    distance = torch.cdist(_paint_features(colours), centres)
    return torch.softmax(-distance.square() / (2 * PAINT_SPREAD**2), dim=1)


@dataclass
class _Recolour:
    gains: torch.Tensor  # (K, 3) per paint: picture over texture, per channel, of (colour + RECOLOUR_EPS)
    amounts: torch.Tensor  # (K,) how far each paint goes there, 0-1
    trusted: torch.Tensor  # (K,) how far the picture's detail and decals on each paint are trusted, 0-1
    paints: list  # per paint, for the report


def _recolour_plan(old: torch.Tensor, new: torch.Tensor, membership: torch.Tensor, weight: torch.Tensor) -> _Recolour:
    """
    Whether each paint takes the picture's colour, from its texels' texture and picture colours (linear,
    (N, 3)), soft membership (N, K) and how far each texel counts (N,): seen squarely, the paint's own
    all around, outside decals, not metal.
    """
    count = membership.shape[1]
    owner = membership.argmax(1)
    hue_new = _hue(new)
    stats = []
    for k in range(count):
        mine = owner == k
        texels = int(mine.sum())
        w = weight * mine
        use = w > 1e-3
        entry: dict = {"texels": texels, "seen": float(w.sum())}
        if texels and int(use.sum()) >= 20:
            w, o, n, h = w[use], old[use], new[use], hue_new[use]
            t_med = torch.stack([_wquantile(o[:, c], w, 0.5) for c in range(3)])
            s_med = torch.stack([_wquantile(n[:, c], w, 0.5) for c in range(3)])
            centre = torch.stack([_wquantile(h[:, c], w, 0.5) for c in range(2)])
            bright = torch.log(_luma(n).clamp_min(1e-4))
            alike = (h - centre).norm(dim=-1) <= RECOLOUR_TOLERANCE
            alike &= (bright - _wquantile(bright, w, 0.5)).abs() <= math.log(RECOLOUR_BRIGHTNESS)
            entry.update(
                t_med=t_med,
                s_med=s_med,
                one=float((w * alike).sum() / w.sum()),
                change=float((_hue(s_med) - _hue(t_med)).norm()),
                ratio=float(_luma(s_med) / _luma(t_med).clamp_min(1e-4)),
            )
        stats.append(entry)
    gains = torch.ones((count, 3), device=old.device)
    amounts = torch.zeros(count, device=old.device)
    trusted = torch.ones(count, device=old.device)
    paints = []
    for k, e in enumerate(stats):
        if not e["texels"]:
            paints.append({"texture": None, "share": 0.0, "amount": 0.0, "decision": "unused"})
            continue
        mine = owner == k
        report = {
            "texture": _hex(old[mine].median(0).values),
            "share": round(e["texels"] / old.shape[0], 3),
            "seen": round(e["seen"] / e["texels"], 3),
        }
        if "ratio" not in e:
            paints.append({**report, "amount": 0.0, "decision": "kept: not seen"})
            continue
        # Near black can't be darker; a paint much darker in the picture is in shade there, or isn't
        # what the picture shows (a car's white windows: the seats behind the glass). Its colour stays,
        # and, as far as that is clear, its detail and decals aren't trusted
        shade = 1.0 if float(_luma(e["t_med"])) < RECOLOUR_DARK else _ramp(RECOLOUR_DARKER, e["ratio"])
        seen = _ramp(RECOLOUR_SEEN, e["seen"] / e["texels"]) * _ramp(RECOLOUR_COUNT, e["seen"])
        one = _ramp(RECOLOUR_ONE, e["one"])
        factors = {
            "kept: too little seen": seen,
            "detail only: the picture shows it in several colours": one,
            "kept: the same colour": _ramp(RECOLOUR_CHANGE, e["change"]),
            "kept: only darker in the picture (shade)": shade,
        }
        amount = math.prod(factors.values())
        gains[k] = (e["s_med"] + RECOLOUR_EPS) / (e["t_med"] + RECOLOUR_EPS)
        amounts[k] = amount
        trusted[k] = 1 - (1 - shade) * seen * one
        paints.append(
            {
                **report,
                "picture": _hex(e["s_med"]),
                "one_colour": round(e["one"], 3),
                "change": round(e["change"], 1),
                "brightness": round(e["ratio"], 3),
                "amount": round(amount, 3),
                "decision": "recoloured" if amount >= 0.5 else min(factors, key=factors.get),
            }
        )
    return _Recolour(gains=gains, amounts=amounts, trusted=trusted, paints=paints)


def _recoloured(old: torch.Tensor, membership: torch.Tensor, plan: _Recolour) -> torch.Tensor:
    """Texture colours (linear, (N, 3)) with each paint taken as far as planned to the picture's."""
    shift = (membership * plan.amounts[None])[..., None] * (plan.gains[None] - 1)  # (N, K, 3)
    return (old + (old + RECOLOUR_EPS) * shift.sum(1)).clamp(0, 1)


# --- The picture side ---------------------------------------------------------------------------------


def _components(labels: torch.Tensor) -> torch.Tensor:
    """
    4-connected components of equal labels in an (H, W) label image (negative: no label). Returns ids
    0..n-1 per pixel (-1 where no label): label propagation with pointer jumping.
    """
    h, w = labels.shape
    n = h * w
    valid = labels >= 0
    if not bool(valid.any()):
        return torch.full_like(labels, -1)

    def runs(grid: torch.Tensor) -> torch.Tensor:  # run of equal labels along each row: an id per pixel
        start = torch.ones_like(grid, dtype=torch.bool)
        start[:, 1:] = grid[:, 1:] != grid[:, :-1]
        return torch.cumsum(start.flatten(), 0) - 1

    across = runs(labels)  # row-major pixel order
    down = runs(labels.T.contiguous()).view(w, h).T.flatten()  # the same along columns
    index = torch.arange(n, device=labels.device)
    comp = torch.where(valid.flatten(), index, torch.full_like(index, n))
    while True:
        previous = comp
        # Each run takes its least label, along rows, then columns; then each pixel follows its label (a
        # pixel of its component) to that pixel's label
        for run in (across, down):
            least = torch.full((int(run[-1]) + 1,), n, device=labels.device, dtype=comp.dtype)
            comp = least.scatter_reduce(0, run, comp, reduce="amin")[run]
        for _ in range(2):
            comp = torch.minimum(comp, torch.where(comp < n, comp[comp.clamp(max=n - 1)], comp))
        if torch.equal(comp, previous):
            break
    ids = torch.full_like(labels, -1)
    ids[valid] = torch.unique(comp.view(h, w)[valid], return_inverse=True)[1]
    return ids


def _pairs(x: torch.Tensor, dy: int, dx: int):
    """Two aligned views of (..., H, W) x: x[p] and x[p + (dy, dx)], over the p where both exist."""
    h, w = x.shape[-2:]
    a = x[..., max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
    b = x[..., max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
    return a, b


class _Canvas:
    """
    The part of the picture the model covers (cropped to it), and where texels land on it: each visible
    texel at the pixel its point projects to.
    """

    def __init__(self, picture: _Picture, zbuf: torch.Tensor, xy: torch.Tensor, visible: torch.Tensor) -> None:
        inside = (picture.mask > 0.5) & torch.isfinite(zbuf)
        rows = torch.nonzero(inside.any(1)).flatten()
        cols = torch.nonzero(inside.any(0)).flatten()
        if rows.numel() == 0:
            raise _Skip("the model and the picture don't overlap")
        self.top, self.left = int(rows[0]), int(cols[0])
        self.height, self.width = int(rows[-1]) + 1 - self.top, int(cols[-1]) + 1 - self.left
        self.window = (slice(self.top, self.top + self.height), slice(self.left, self.left + self.width))
        self.inside = inside[self.window]
        col = xy[:, 0].floor().long() - self.left
        row = xy[:, 1].floor().long() - self.top
        landed = visible & (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        self.pixel = (row * self.width + col).clamp(0, self.height * self.width - 1)
        self.landed = landed & self.inside.flatten()[self.pixel]
        u = (xy[:, 0] - self.left) / self.width * 2 - 1
        v = (xy[:, 1] - self.top) / self.height * 2 - 1
        self.grid = torch.stack([u, v], -1)[None, None]

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        """(C, H, W) of the whole picture to (C, h, w)."""
        return image[(slice(None),) + self.window]

    def counts(self, values: torch.Tensor):
        """Sums of (N, C) texel values per pixel (C, h, w), and how many texels landed there (h, w)."""
        n = self.height * self.width
        idx = self.pixel[self.landed]
        sums = torch.zeros((values.shape[1], n), device=values.device).index_add_(1, idx, values[self.landed].T)
        counts = torch.zeros(n, device=values.device).index_add_(0, idx, torch.ones_like(idx, dtype=values.dtype))
        return sums.view(-1, self.height, self.width), counts.view(self.height, self.width)

    def splat(self, values: torch.Tensor) -> torch.Tensor:
        """The mean of (N, C) texel values at each pixel (C, h, w), filled in where none landed."""
        sums, counts = self.counts(values)
        return _push_pull(sums / counts.clamp_min(1), (counts > 0).float())

    def sample(self, image: torch.Tensor) -> torch.Tensor:
        """(C, h, w) at each texel, (N, C), bilinear; 0 for texels that don't land on the picture."""
        out = F.grid_sample(image[None], self.grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out[0, :, 0].T * self.landed[:, None]


def _purity(canvas: _Canvas, owner: torch.Tensor, paints: int, radius: float) -> torch.Tensor:
    """Per texel (N,): the share of the texels landing within about ``radius`` pixels that are its paint."""
    sums, counts = canvas.counts(F.one_hot(owner, paints).float())
    shares = _smooth((sums / counts.clamp_min(1))[None], counts[None, None], radius)[0]
    return canvas.sample(shares).gather(1, owner[:, None])[:, 0]


def _colour_classes(lab: torch.Tensor, inside: torch.Tensor, count: int):
    """
    The picture's colours (CIELAB, (3, h, w)) clustered over the ``inside`` pixels (k-means). Returns soft
    memberships (K, h, w), 0 outside, and hard labels (h, w), -1 outside.
    """
    device = lab.device
    pixels = lab.permute(1, 2, 0)[inside]
    gen = torch.Generator().manual_seed(0)
    sample = pixels[torch.randperm(pixels.shape[0], generator=gen)[:20000].to(device)]
    centres = sample[:1]
    for _ in range(count - 1):
        d = torch.cdist(sample, centres).amin(1).square()
        if float(d.sum()) <= 0:
            break
        pick = int(torch.multinomial((d / d.sum()).cpu().double(), 1, generator=gen))
        centres = torch.cat([centres, sample[pick : pick + 1]])
    for _ in range(10):
        assign = torch.cdist(sample, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, sample)
        total = torch.zeros(centres.shape[0], device=device).index_add_(0, assign, torch.ones_like(assign, dtype=lab.dtype))
        centres = torch.where(total[:, None] > 0, sums / total.clamp_min(1)[:, None], centres)
    k = centres.shape[0]
    distance = torch.cdist(pixels, centres)
    membership = torch.zeros((k,) + inside.shape, device=device)
    membership[:, inside] = torch.softmax(-distance.square() / (2 * CLASS_SOFTNESS**2), 1).T
    labels = torch.full(inside.shape, -1, dtype=torch.long, device=device)
    labels[inside] = distance.argmin(1)
    # A pixel none of whose four neighbours shares its cluster joins the majority round it (a lone
    # pixel of noise; corners where two regions of one colour meet diagonally stay apart)
    votes = torch.zeros((k,) + inside.shape, device=device)
    votes[:, inside] = F.one_hot(labels[inside], k).T.float()
    votes = F.avg_pool2d(votes[None], 3, 1, 1)[0]
    padded = F.pad(labels[None, None].float(), (1, 1, 1, 1), value=-2)[0, 0]
    alone = inside.clone()
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        alone &= padded[dy : dy + labels.shape[0], dx : dx + labels.shape[1]] != labels
    labels = torch.where(alone, votes.argmax(0), labels)
    return membership, labels


@dataclass
class _Regions:
    """The picture's regions of one colour, and which of them may be decals."""

    membership: torch.Tensor  # (K, h, w) soft colour classes
    ids: torch.Tensor  # (h, w) region of each pixel, -1 outside
    candidate: torch.Tensor  # (R,) bool: may be a decal
    area: torch.Tensor  # (R,) pixels
    seen: torch.Tensor  # (R,) mean detail weight
    core: torch.Tensor  # (h, w) bool: pixels whose four neighbours are in their region


def _regions(lab: torch.Tensor, inside: torch.Tensor, blocked: torch.Tensor, seen: torch.Tensor, highlight: torch.Tensor) -> _Regions:
    """
    Regions of one colour class in the picture (CIELAB (3, h, w) over ``inside``), and which may be
    decals: clear of ``blocked`` pixels (the outline, depth edges and misfits, widened), standing out from
    what is round them with a sharp edge, seen well (``seen``: detail weight per pixel), not highlights and
    not too big.
    """
    membership, labels = _colour_classes(lab, inside, DETAIL_CLASSES)
    ids = _components(labels)
    flat = ids.flatten()
    has = flat >= 0
    count = int(flat.max()) + 1 if bool(has.any()) else 0
    device = lab.device

    def total(values: torch.Tensor) -> torch.Tensor:  # (h, w) or (C, h, w) -> sums per region (R, C)
        v = values.reshape(-1, inside.numel()).T
        return torch.zeros((count, v.shape[1]), device=device).index_add_(0, flat[has], v[has].float())

    area = total(torch.ones_like(seen))[:, 0]
    mean_lab = total(lab) / area[:, None]
    touches = total(blocked.float())[:, 0] > 0
    mean_seen = total(seen)[:, 0] / area
    mean_highlight = total(highlight)[:, 0] / area
    ring = torch.zeros((count, 3), device=device)
    ring_n = torch.zeros(count, device=device)
    step = torch.zeros(count, device=device)
    step_n = torch.zeros(count, device=device)
    # Core pixels: all four neighbours in the same region (a region's mean colour is read there, clear of
    # its anti-aliased edge)
    core = torch.zeros_like(inside)
    middle = ids[1:-1, 1:-1]
    core[1:-1, 1:-1] = (
        (middle >= 0)
        & (ids[:-2, 1:-1] == middle)
        & (ids[2:, 1:-1] == middle)
        & (ids[1:-1, :-2] == middle)
        & (ids[1:-1, 2:] == middle)
    )
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        for distance in (1, 3):
            a, b = _pairs(ids, dy * distance, dx * distance)
            la, lb = _pairs(lab, dy * distance, dx * distance)
            differ = (a >= 0) & (b >= 0) & (a != b)
            if distance == 1:
                d = (la - lb).norm(dim=0)[differ]
                step.index_add_(0, a[differ], d)
                step_n.index_add_(0, a[differ], torch.ones_like(d))
            else:
                ring.index_add_(0, a[differ], lb.permute(1, 2, 0)[differ])
                ring_n.index_add_(0, a[differ], torch.ones_like(a[differ], dtype=lab.dtype))
    contrast = (mean_lab - ring / ring_n.clamp_min(1)[:, None]).norm(dim=-1)
    sharp = step / step_n.clamp_min(1)
    candidate = (
        ~touches
        & (ring_n > 0)
        & (area >= 3)
        & (area <= DECAL_AREA * float(inside.sum()))
        & (contrast >= DECAL_CONTRAST)
        & (sharp >= DECAL_SHARPNESS * contrast)
        & (mean_seen >= DECAL_SEEN[0])
        & (mean_highlight < 0.5)
    )
    return _Regions(membership=membership, ids=ids, candidate=candidate, area=area, seen=mean_seen, core=core)


def _detail(residual: torch.Tensor, membership: torch.Tensor, valid: torch.Tensor, sigma: float):
    """
    The detail in a (3, h, w) residual, what is left after smoothing it over ``sigma`` pixels within each
    colour class (soft ``membership``, (K, h, w)), and the smoothed rest: both on ``valid`` pixels (0
    elsewhere).
    """
    weight = membership * valid
    low = torch.zeros_like(residual)
    for start in range(0, weight.shape[0], 4):  # a few classes at a time: (B, 3, h, w) at once
        part = weight[start : start + 4]
        means = _smooth(residual[None].expand(part.shape[0], -1, -1, -1), part[:, None], sigma)
        low += (part[:, None] * means).sum(0)
    low = low / weight.sum(0, keepdim=True).clamp_min(1e-6)
    return (residual - low) * valid, low * valid


def _only_light(picture: torch.Tensor, texture: torch.Tensor, glossy: torch.Tensor) -> torch.Tensor:
    """
    Whether each picture colour (linear, (R, 3)) is the texture's (R, 3) under different light: the
    texture's times a brightness within LIGHT_RATIO either way, plus white on a glossy surface (glossy
    (R,), 0-1: a reflection), to within LIGHT_FIT of their difference.
    """
    k = torch.exp(torch.linspace(-math.log(LIGHT_RATIO), math.log(LIGHT_RATIO), 25, device=picture.device))
    lit = k[None, :, None] * texture[:, None, :]  # (R, 25, 3)
    white = (picture[:, None, :] - lit).mean(-1, keepdim=True).clamp_min(0) * (glossy > 0.5)[:, None, None]
    misfit = (picture[:, None, :] - lit - white).norm(dim=-1).amin(1)
    return misfit <= LIGHT_FIT * (picture - texture).norm(dim=-1)


def _decal_fills(
    regions: _Regions,
    residual: torch.Tensor,
    low: torch.Tensor,
    picture: torch.Tensor,
    base: torch.Tensor,
    glossy: torch.Tensor,
    canvas: _Canvas,
    trust: torch.Tensor,
    weight: torch.Tensor,
):
    """
    What the decals add on top of the detail, as an image (3, h, w): inside each, the part of the
    picture's difference from the recoloured texture that the detail left out (``low``), so the decal
    goes on whole, times how well it was seen and how far its paint is trusted. And a report.
    ``residual`` is that difference; ``picture``, ``base`` and ``glossy`` are the picture (light taken
    out), the recoloured texture and the texture's glossiness in the picture ((3, h, w), (h, w)); per
    texel, ``trust`` is how far its paint is trusted and ``weight`` its detail weight.
    """
    h, w = regions.ids.shape
    flat = regions.ids.flatten()
    has = flat >= 0
    count = regions.area.numel()
    report = {"count": 0, "area": 0.0, "only_light": 0, "untrusted": 0}
    if count == 0 or not bool(regions.candidate.any()):
        return torch.zeros_like(residual), report
    device = residual.device

    def total(values: torch.Tensor, where: torch.Tensor) -> torch.Tensor:  # (C, h, w) -> (R, C)
        v = values.reshape(values.shape[0], -1).T
        return torch.zeros((count, v.shape[1]), device=device).index_add_(0, flat[where], v[where])

    ones = torch.ones((1, h, w), device=device)
    core = regions.core.flatten() & has
    cored = total(ones, core)[:, 0] > 0
    # A region's colours are read on its core pixels, or all of them if it's too thin to have any
    use = has & torch.where(cored[flat.clamp_min(0)], core, torch.ones_like(core))
    n = total(ones, use)[:, 0].clamp_min(1)[:, None]
    seen = total(picture, use) / n
    texture = total(base, use) / n
    shine = total(glossy[None], use)[:, 0] / n[:, 0]
    # How much each varies inside: the texture's own garbled try at a design (a skateboard's smeared
    # letters) varies where the picture's clean one doesn't
    spread_seen = (total(picture.square(), use) / n - seen.square()).clamp_min(0).sum(-1).sqrt()
    spread_texture = (total(base.square(), use) / n - texture.square()).clamp_min(0).sum(-1).sqrt()
    garbled = (spread_texture > DECAL_GARBLED * spread_seen) & (spread_texture > 0.03)
    light = regions.candidate & _only_light(seen, texture, shine) & ~garbled
    decal = regions.candidate & ~light
    # How far the paints of the texels landing in each region are trusted
    at = torch.where(canvas.landed, flat[canvas.pixel], torch.full((trust.shape[0],), -1, dtype=torch.long, device=device))
    landed = at >= 0
    mass = torch.zeros(count, device=device).index_add_(0, at[landed], weight[landed])
    trusted = torch.zeros(count, device=device).index_add_(0, at[landed], (weight * trust)[landed])
    trusted = torch.where(mass > 0, trusted / mass.clamp_min(1e-9), torch.ones_like(mass))
    strength = _smoothstep(*DECAL_SEEN, regions.seen) * trusted * decal
    image = low * (strength[flat.clamp_min(0)] * has).view(1, h, w)
    applied = strength > 0.05
    report.update(
        count=int(applied.sum()),
        area=round(float(regions.area[applied].sum()) / max(1.0, float(has.sum())), 4),
        only_light=int(light.sum()),
        untrusted=int((decal & (trusted < 0.5)).sum()),
    )
    return image, report


def _lighting(old: torch.Tensor, new: torch.Tensor, normals: torch.Tensor, metallic: torch.Tensor):
    """
    How the picture's brightness relates to the texture's, from texels it saw well: colours (linear,
    (N, 3)), surface normals and metalness. Returns ``(coefficients, note)``: the picture times
    exp(-(a + b . n)) matches the texture (None when it can't be told). Measured where the two agree in
    hue (the same paint, lit), preferring coloured paint: whites and greys agree in hue whatever they
    show (a texture's opaque white car windows against the interior a picture sees through them). Metal
    is left out: its base colour isn't what a picture shows of it. Medians per direction and a robust
    fit (Huber weights), so a white sole or a highlight can't drag it.
    """
    nonmetal = metallic < 0.8
    if int(nonmetal.sum()) < 500:
        return None, "kept: too little seen well to compare"
    luma_old, luma_new = _luma(old), _luma(new)
    hue = (old * new).sum(-1) / (old.norm(dim=-1) * new.norm(dim=-1)).clamp_min(1e-9)
    agree = (hue > math.cos(math.radians(HUE_AGREEMENT))) & (luma_old > 0.003) & (luma_new > 0.003) & nonmetal
    share = float(agree.sum()) / float(nonmetal.sum())
    if share < MIN_AGREEMENT or int(agree.sum()) < 200:
        return None, f"kept: the texture's colours differ from the picture's ({share:.0%} agree)"
    chroma_old = (old - luma_old[:, None]).norm(dim=-1)
    chroma_new = (new - luma_new[:, None]).norm(dim=-1)
    colourful = agree & (chroma_old > 0.2 * luma_old.clamp_min(0.02)) & (chroma_new > 0.2 * luma_new.clamp_min(0.02))
    basis = colourful if int(colourful.sum()) >= max(200, 0.3 * int(agree.sum())) else agree
    ratio = (torch.log(luma_new[basis]) - torch.log(luma_old[basis])).double()
    normals = normals[basis].double()
    # Each direction the surface faces counts about the same, whatever its area (a cabinet's side, seen
    # at an angle, against its front): the ratio's median per bin of normals, weighted by the square
    # root of the bin's count
    key = ((normals + 1) / 2 * (LIGHT_BINS - 1)).round().long()
    key = (key[:, 0] * LIGHT_BINS + key[:, 1]) * LIGHT_BINS + key[:, 2]
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    order = torch.argsort(ratio)
    order = order[torch.argsort(inverse[order], stable=True)]
    starts = torch.cumsum(counts, 0) - counts
    y = ratio[order[starts + counts // 2]]
    mean_normal = torch.zeros((counts.numel(), 3), dtype=torch.float64, device=y.device).index_add_(0, inverse, normals)
    x = torch.cat([torch.ones_like(y[:, None]), mean_normal / counts[:, None]], -1)
    bin_weight = counts.double().sqrt() * (counts >= 20)
    coef = torch.zeros(4, dtype=torch.float64, device=y.device)
    coef[0] = torch.median(ratio)
    # Shrink the shading terms: a narrow range of normals (a skateboard's deck) can't tell shading apart
    shrink = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64, device=y.device) * SHADING_SHRINK
    for _ in range(8):
        residual = y - x @ coef
        w = (bin_weight * 0.2 / residual.abs().clamp_min(0.2))[:, None]  # Huber, at 0.2 in log brightness
        a = (x * w).T @ x + torch.diag(shrink) * float(w.sum())
        coef = torch.linalg.solve(a, (x * w).T @ y)
    full = torch.cat([torch.ones_like(ratio[:, None]), normals], -1)
    gain = math.exp(-float(torch.median(full @ coef)))
    if not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
        return None, f"kept: the texture is {gain:.3g}x as bright as the picture"
    return coef.float(), "matched"




def _bake(model: _Model, picture: _Picture, mapping: _Mapping, zbuf: torch.Tensor, report: dict, clock, debug):
    tex_h, tex_w = model.texture.shape[:2]
    h, w = zbuf.shape
    params = mapping.params

    # Every texel's surface point and normal (uv_raster works bottom-up, the image top-down)
    uv_clip = torch.cat([model.uv * 2 - 1, torch.zeros_like(model.uv[:, :1]), torch.ones_like(model.uv[:, :1])], -1)
    rast, _ = uv_raster.rasterize(None, uv_clip[None], model.faces.int(), resolution=[tex_h, tex_w])
    rast = rast.flip(1)
    covered = rast[0, ..., 3] > 0
    flat = torch.nonzero(covered.flatten()).flatten()
    if flat.numel() == 0:
        raise _Skip("the texture has no covered texels")
    attrs = torch.cat([model.verts, model.normals], -1)
    texel = uv_raster.interpolate(attrs[None], rast, model.faces.int())[0][0].view(-1, 6)[flat]
    face = rast[0, ..., 3].flatten()[flat].round().long() - 1
    del rast
    points, normals = texel[:, :3], texel[:, 3:]
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    a, b, c = (model.verts[model.faces[:, i]] for i in range(3))
    all_face_normals = torch.cross(b - a, c - a, dim=-1)
    all_face_normals = all_face_normals / all_face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    face_normals = all_face_normals[face]
    clock("texels_s")

    # What the camera sees of each texel
    xy, depth, _ = mapping(points)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    pix = xy[:, 1].long().clamp(0, h - 1) * w + xy[:, 0].long().clamp(0, w - 1)
    world_px = 1.0 / mapping.pixels_per_unit  # one picture pixel in normalised model units
    visible = inside & (depth <= zbuf.flatten()[pix] + max(0.01, 3 * world_px))
    towards = view_dirs(points, params)
    cos = (normals * towards).sum(-1)
    # The texel's own triangle must face the camera too: smoothed normals bend round hard edges, and
    # would let a side the picture can't see take colour from the one next to it
    cos_face = (face_normals * towards).sum(-1)
    # Normals pointing inwards (a flipped mesh) would hide the whole visible side
    seen_cos = cos[visible]
    if seen_cos.numel() and float(seen_cos.median()) < 0:
        cos, cos_face, face_normals, all_face_normals = -cos, -cos_face, -face_normals, -all_face_normals

    # Fade out near the picture's outline, near depth edges and near misfits
    side_px = picture.box[2]
    outline = _ramp_inside(picture.mask[None, None], max(1, round(OUTLINE_FADE * side_px)))[0, 0]
    misfit = (torch.isfinite(zbuf).float() != picture.mask).float()[None, None]
    sliver = max(1, round(MISFIT_SLIVER * side_px))
    misfit = _dilate(_erode(misfit, sliver), sliver)
    misfit_fade = _ramp_away(misfit, max(1, round(MISFIT_FADE * side_px)))[0, 0]
    zf = torch.where(torch.isfinite(zbuf), zbuf, torch.full_like(zbuf, 1e3))[None, None]
    # Pinholes in a thin surface (TRELLIS.2 makes some plates as two thin shells) show what's behind;
    # closing them first keeps each from fading out a disc of texels around it
    pin = max(1, round(PINHOLE * side_px))
    zf = -_erode(_dilate(-zf, pin), pin)[0, 0]
    jump = max(0.03, 8 * world_px)
    edges = torch.zeros_like(zbuf, dtype=torch.bool)
    dx = (zf[:, 1:] - zf[:, :-1]).abs() > jump
    dy = (zf[1:] - zf[:-1]).abs() > jump
    edges[:, 1:] |= dx
    edges[:, :-1] |= dx
    edges[1:] |= dy
    edges[:-1] |= dy
    edge_fade = _ramp_away(edges.float()[None, None], max(1, round(EDGE_FADE * side_px)))[0, 0]
    grid = torch.stack([xy[:, 0] / w * 2 - 1, xy[:, 1] / h * 2 - 1], -1)[None, None]

    def sample(image: torch.Tensor) -> torch.Tensor:  # (C, h, w) -> (N, C)
        return F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T

    fades = sample(torch.stack([outline, edge_fade, misfit_fade])).prod(-1)
    seen = visible.float() * fades * _smoothstep(0.0, 0.15, cos_face)
    old_lin = _srgb_to_linear(model.texture.view(-1, 3)[flat])
    new_lin = _srgb_to_linear(sample(picture.rgb.permute(2, 0, 1)))
    # The picture's own colour at each texel with its shading taken out but its exposure kept (debug: what the
    # painter holds its views to); the picture as it is where the lighting can't be fitted
    picture_texels = new_lin

    # The texture's roughness and metallic at each texel (the same atlas, at its own resolution)
    rows = torch.div(flat, tex_w, rounding_mode="floor").float()
    cols = (flat % tex_w).float()
    at = torch.stack([(cols + 0.5) / tex_w * 2 - 1, (rows + 0.5) / tex_h * 2 - 1], -1)[None, None]
    rm = model.roughness_metallic.permute(2, 0, 1)[None]
    rm = F.grid_sample(rm, at, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T

    # Lighting: the picture's exposure brought to the texture's, and its shading taken out (measured
    # wherever the picture will be used, so a side seen at an angle is in the fit)
    strong = seen * _smoothstep(*FACING, cos) > 0.5
    coef, note = _lighting(old_lin[strong], new_lin[strong], face_normals[strong], rm[strong, 1])
    exposure = {"gain": 1.0, "shading": 1.0, "note": note}
    if coef is not None:
        factor = torch.exp(-(face_normals @ coef[1:] + coef[0])).clamp(*GAIN_RANGE)
        picture_texels = (new_lin * (factor / factor[strong].median())[:, None]).clamp(0, 1)
        new_lin = (new_lin * factor[:, None]).clamp(0, 1)
        seen_factor = factor[strong]
        exposure["gain"] = round(float(seen_factor.median()), 3)
        # How much the shading correction varies over the side seen well (1: none)
        exposure["shading"] = round(float(torch.quantile(seen_factor, 0.95) / torch.quantile(seen_factor, 0.05)), 3)

    # Specular highlights: a white addition (about equal in R, G and B) on a glossy or metallic surface
    glossy = torch.maximum(1 - _smoothstep(*GLOSSY, rm[:, 0]), _smoothstep(*METALLIC, rm[:, 1]))
    added = new_lin - old_lin
    least, most = added.amin(-1), added.amax(-1)
    highlight = (
        glossy
        * _smoothstep(0.03, 0.12, least)
        * _smoothstep(0.3, 0.6, least / most.clamp_min(1e-4))
        * _smoothstep(0.3, 0.6, _luma(new_lin))
    )
    seen = seen * (1 - highlight)
    detail_weight = seen * _smoothstep(*FACING, cos)
    clock("weights_s")

    changed = detail_weight > 0.02
    coverage = float(changed.float().mean())
    report["coverage"] = round(coverage, 4)
    report["exposure"] = exposure
    visible_count = visible.float().sum().clamp_min(1)
    report["highlights"] = round(float((highlight * visible.float() > 0.5).float().sum() / visible_count), 4)
    if debug is not None:
        debug.update(
            weight=detail_weight, flat=flat, xy=xy, zbuf=zbuf, outline=outline, edge_fade=edge_fade,
            misfit_fade=misfit_fade, cos=cos, visible=visible, highlight=highlight, texture_size=(tex_h, tex_w),
            picture_linear=picture_texels,
        )
    if coverage < MIN_COVERAGE:
        raise _Skip(f"too little of the model faces the camera ({coverage:.1%} of texels)")

    # The picture's regions of one colour, and which may be decals (designs painted on a paint)
    canvas = _Canvas(picture, zbuf, xy, visible)
    picture_lin = canvas.crop(_srgb_to_linear(picture.rgb).permute(2, 0, 1))
    seen_px = canvas.splat(detail_weight[:, None])[0]
    highlight_px = canvas.splat(highlight[:, None])[0]
    clear = canvas.inside & ~canvas.crop(edges[None])[0] & ~(canvas.crop(misfit[0])[0] > 0)
    blocked = _dilate((~clear).float()[None, None], DECAL_MARGIN)[0, 0] > 0
    lab = _lab(picture_lin.permute(1, 2, 0)).permute(2, 0, 1)
    regions = _regions(lab, canvas.inside, blocked, seen_px, highlight_px)
    del lab
    clock("regions_s")

    # Paints: each takes the picture's colour all round, or keeps its own
    old_srgb = model.texture.view(-1, 3)[flat]
    membership = _membership(old_srgb, _paints(old_srgb, PAINTS))
    del old_srgb
    owner = membership.argmax(1)
    purity = _purity(canvas, owner, membership.shape[1], PURITY_RADIUS * side_px)
    region = torch.where(canvas.landed, regions.ids.flatten()[canvas.pixel], torch.full_like(owner, -1))
    in_decal = (region >= 0) & regions.candidate[region.clamp_min(0)] if regions.area.numel() else region >= 0
    metal = rm[:, 1] >= 0.8
    data = seen * _smoothstep(*FACING_DATA, cos) * _smoothstep(*PURITY, purity) * ~in_decal * ~metal
    plan = _recolour_plan(old_lin, new_lin, membership, data)
    base = _recoloured(old_lin, membership, plan)
    report["paints"] = plan.paints
    clock("paints_s")

    # The recoloured texture as the camera sees it: each pixel's triangle, its UV there, the texture there
    # (its gutters carried along, for bilinear lookups at chart edges)
    old_tex = _srgb_to_linear(model.texture).permute(2, 0, 1)
    change = torch.zeros_like(old_tex).view(3, -1)
    change[:, flat] = (base - old_lin).T
    base_tex = (old_tex + _push_pull(change.view(3, tex_h, tex_w), covered.float())).clamp(0, 1)
    del old_tex, change
    vxy, vdepth, vs = mapping(model.verts)
    face_px, bary_px = rasterize_faces(vxy, vdepth, vs, model.faces, zbuf)
    face_px, bary_px = face_px[canvas.window], bary_px[canvas.window]
    uv_px = (bary_px[..., None] * model.uv[model.faces[face_px.clamp_min(0)]]).sum(-2)
    lookup = torch.stack([uv_px[..., 0] * 2 - 1, (1 - uv_px[..., 1]) * 2 - 1], -1)[None]
    base_px = F.grid_sample(base_tex[None], lookup, mode="bilinear", padding_mode="border", align_corners=False)[0]
    del base_tex, lookup, uv_px
    # ... and the picture with its light taken out, as at the texels
    factor_px = torch.ones_like(base_px[0])
    if coef is not None:
        factor_px = torch.exp(-(all_face_normals[face_px.clamp_min(0)] @ coef[1:] + coef[0])).clamp(*GAIN_RANGE)

    # Detail: the picture's difference from the recoloured texture, less what changes slowly, plus the
    # decals whole
    picture_px = (picture_lin * factor_px).clamp(0, 1)
    residual = picture_px - base_px
    valid = (canvas.inside & (highlight_px < 0.5)).float()
    high, low = _detail(residual, regions.membership, valid, max(1.0, DETAIL_SIGMA * side_px))
    trust = membership @ plan.trusted
    glossy_px = canvas.splat(glossy[:, None])[0]
    fill, decals = _decal_fills(regions, residual, low, picture_px, base_px, glossy_px, canvas, trust, detail_weight)
    del low
    report["decals"] = decals
    recoloured = any(p["amount"] >= 0.5 for p in plan.paints)
    report["mode"] = "recolour and detail" if recoloured else "detail only"
    # Detail where the picture saw the surface well; decals on whole wherever they show, but not where it
    # turns away
    decal_weight = seen * _smoothstep(*FACING_DECAL, cos)
    blended = base + (detail_weight * trust)[:, None] * canvas.sample(high) + decal_weight[:, None] * canvas.sample(fill)
    if debug is not None:
        debug.update(
            canvas=canvas, regions=regions, residual=residual, high=high, fill=fill, base=base, purity=purity,
            membership=membership, data=data, in_decal=in_decal,
        )

    texture = model.texture.clone().view(-1, 3)
    texture[flat] = _linear_to_srgb(blended)
    texture = texture.view(tex_h, tex_w, 3)
    # The gutters were inpainted from the old colours: carry the change into them
    delta_tex = (texture - model.texture).permute(2, 0, 1)
    filled = _push_pull(delta_tex, covered.float())
    texture = torch.where(covered[..., None], texture, (model.texture + filled.permute(1, 2, 0)).clamp(0, 1))
    clock("blend_s")
    return texture


# --- Entry point --------------------------------------------------------------------------------------


def summary(report: dict) -> dict:
    """A report's essentials, for a job's result and the worker's log."""
    keys = ("applied", "reason", "iou", "colour", "coverage", "pose", "mode", "gpu_fault")
    out = {key: report[key] for key in keys if report.get(key) is not None}
    recoloured = [
        {"from": p["texture"], "to": p["picture"], "share": p["share"]}
        for p in report.get("paints") or []
        if p.get("decision") == "recoloured"
    ]
    if recoloured:
        out["recoloured"] = recoloured
    decals = (report.get("decals") or {}).get("count")
    if decals:
        out["decals"] = decals
    seconds = report.get("timings", {}).get("total_s")
    if seconds is not None:
        out["seconds"] = seconds
    return out


def project_picture(
    mesh: Any, picture: Any, device: Optional[Any] = None, debug: Optional[dict] = None
) -> tuple[Any, dict]:
    """
    Blend ``picture`` (the background-removed input, RGBA, full frame) into the base colour texture of
    ``mesh`` (``o_voxel.postprocess.to_glb``'s trimesh) on the side it shows.

    Returns ``(mesh, report)``. The report says whether it was applied and why not, the camera found
    (``pose``), how well the silhouettes agree (``iou``) and the colours (``colour``), the runner-up
    camera and any rivals, the lighting taken out (``exposure``), how much of the texture changed
    (``coverage``, ``broad_change``) and timings. Never raises; when it doesn't apply, the mesh is
    returned untouched. ``debug``, a dict, collects intermediate tensors.
    """
    report: dict = {"applied": False, "reason": "", "pose": None, "iou": None, "colour": None, "timings": {}}
    started = time.perf_counter()
    last = [started]

    def clock(name: str) -> None:
        now = time.perf_counter()
        report["timings"][name] = round(now - last[0], 3)
        last[0] = now

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        try:
            texture, alpha = _run(mesh, picture, device, report, clock, debug)
        except torch.OutOfMemoryError:
            if device.type == "cpu":
                raise
            torch.cuda.empty_cache()
            report["retried_on_cpu"] = True
            texture, alpha = _run(mesh, picture, torch.device("cpu"), report, clock, debug)
        image = Image.fromarray((texture.cpu().numpy() * 255 + 0.5).astype(np.uint8), "RGB")
        if alpha is not None:
            image.putalpha(Image.fromarray(alpha))
        mesh.visual.material.baseColorTexture = image
        report["applied"] = True
        report["reason"] = "applied"
    except _Skip as skip:
        report["reason"] = str(skip)
    except Exception as err:  # noqa: BLE001 - a failed projection must never fail the export
        report["reason"] = f"error: {type(err).__name__}: {err}"
        # The job can still finish without it, but CUDA may be unusable afterwards: restart the worker
        text = report["reason"].lower()
        if any(marker in text for marker in ("cuda", "cublas", "cudnn", "device-side assert")):
            report["gpu_fault"] = True
    report["timings"]["total_s"] = round(time.perf_counter() - started, 3)
    return mesh, report


def _run(mesh, picture, device, report, clock, debug):
    with torch.no_grad():
        model = _load_model(mesh, device)
        pic = _load_picture(picture, device)
        clock("load_s")
        pose = _estimate_pose(model, pic, clock)
        params = _polish(model, pic, pose.params)
        mapping, zbuf = _render_depth(model, pic, params)
        iou = float(_iou(torch.isfinite(zbuf).float(), pic.mask))
        clock("polish_s")
        report["pose"] = _view(params[0]).as_dict()
        report["iou"] = round(iou, 4)
        report["colour"] = round(pose.colour, 4)
        report["runner_up"] = pose.runner_up
        if debug is not None:
            debug.update(candidates=pose.candidates, params=params, zbuf=zbuf)
        if iou < MIN_IOU:
            raise _Skip(f"the silhouettes don't match well enough (IoU {iou:.3f} < {MIN_IOU})")
        # A rival camera that fits as well is harmless only if it sees the same shape
        report["rivals"] = []
        for rival in pose.rivals:
            angle = float(_angle_between(params, rival[None])[0])
            difference = _shape_difference(zbuf, _render_depth(model, pic, rival[None])[1])
            report["rivals"].append({"angle": round(angle, 1), "shape_difference": round(difference, 3)})
            if difference > SAME_SHAPE:
                raise _Skip(
                    f"ambiguous camera: a view {angle:.0f} degrees away fits as well but sees a different shape"
                )
        texture = _bake(model, pic, mapping, zbuf, report, clock, debug)
        return texture, model.texture_alpha
