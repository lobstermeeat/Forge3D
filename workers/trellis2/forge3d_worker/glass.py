"""
Glass in TRELLIS.2's models: see-through where TRELLIS.2 built something behind it, glossy glass elsewhere.

TRELLIS.2 predicts an opacity with the base colour, and to_glb bakes it into the base colour texture's alpha
channel, with the material OPAQUE ("alpha channel is preserved ... not active", says its README). The export
then divides the colour by that alpha (``pipeline.unpremultiply``, Phase 4's fix for spurious alpha on a
dragon's skin) and drops the alpha. Glass has a genuinely low alpha, so it came out opaque, up to four times
brighter, and as rough and as metallic as TRELLIS.2 made it: the founder's blue BMW got pale mint windows.
What Phase 8 measured on to_glb's raw output (the BMW's final and three more textures of it, and Phase 2's
helmet, cartoon car, bubble tea, potion and two dragons), over the texels a triangle covers:

- Glass is low alpha on glossy texels, in regions tens of voxels across: the BMW's windows and lamps (alpha
  below 0.5 on 4.0 % of its texels; roughness 0.48-0.55, or 0.12-0.31 in another texture), the cartoon car's
  windows (alpha 0.2, roughness 0), the bubble tea's cup (alpha under 0.1 on 45 % of its texels).
- Spurious alpha is matte: Phase 2's dragon had alpha below 0.75 on 90 % of its skin at roughness 0.93-0.98,
  a later dragon on 2.8 % at 0.97. Its few glossy low-alpha texels are specks, thin edges and the eyes.
- Glass colour is straight colour, not premultiplied: light grey (187) at alpha 0.21 on the cartoon car, dark
  teal (12, 18, 17) at 0.43 on the BMW, black on the cup. Dividing it by alpha is what made windows pale.
- TRELLIS.2 builds what is behind glass. Rendered as the gallery's turntable does (single-sided faces,
  8 views), 98 % of the BMW's window pixels have an opaque surface behind them (its dark cabin, 30-330
  voxels in; 2 % show the background, through both side windows), 94 % of the cartoon car's, and 89 % of
  the cup's (the tea, 18-26 voxels in; the rest is the empty top of the cup). Windows and cups are thin
  slabs, their outer and inner surfaces both glass, about 2.3 voxels apart. Lamp lenses are glass only on
  the body's outer surface, with the body's shell and its inside behind them.

So glass is split off into a second primitive with its own material, alphaMode BLEND: a dark tint (one
luminance, GLASS_LUMA, in TRELLIS.2's hue where it has one), glossy (GLASS_ROUGHNESS), not metal, and an
opacity from TRELLIS.2's own alpha (OPACITY). The body stays OPAQUE: a single BLEND material would make the
whole model transparent-sorted (three.js draws it without depth writes, so a car's far side shows through its
near side). Only glass with something behind it is made see-through. Other glass (headlights, a smoked tail
light) stays in the body, glossy and not metal, in the colour it had: there the picture's projection often
painted what the photo shows (a headlight's lamps), which one dark tint would wipe out.

Which faces are glass:

1. Texels: weight 1 at alpha at or below ALPHA[0], 0 at or above ALPHA[1], times 1 at roughness at or below
   ROUGHNESS[0], 0 at or above ROUGHNESS[1] (what keeps the dragons' skin out).
2. Faces: the mean weight over FACE_SAMPLES^2 points of the face's UV triangle, glass at FACE or more.
3. Regions: glass faces joined across shared edges (vertices welded by position, so UV seams don't split a
   region). A region of less than MIN_AREA square voxels, or narrower than MIN_WIDTH voxels (twice its area
   over its perimeter: a strip along an edge is narrow however long), is left alone: specks and thin edges.
4. Colour: a region of warm, saturated straight colour (red its largest channel, SATURATED or more: tail
   lights, amber lamps, a dragon's eyes) is left alone. Its colour is the point of it, the division leaves it
   right, and "over" blending can't filter what is behind it red: see-through, a tail light over a grey trunk
   came out grey. Neutral and cool glass (clear, smoked, teal, blue, green) goes on.
5. See-through: RAYS rays go inwards (along -normal) from the region's faces. Past the glass's own slab (back
   faces within SLAB voxels), a ray that meets a front face (one turned towards it: the cabin, the tea) has
   something to show, as does one that passes through more glass and out (through both side windows of a
   car). A ray that meets a back face first is inside a closed solid (a window on a car without a cabin),
   and see-through glass would show the background through the model. A region is see-through when at
   least SEE_THROUGH of its rays have something to show past a slab whose back is glass too (a window, a
   cup: glass painted on an opaque shell, like a smoked tail light on the body, would show the inside of
   the body), or when it is the inner surface of such a region's slab (its rays cross the outer surface, a
   back face within SLAB, and leave).

The glass primitive keeps the input's vertex normals (call this after the shading normals) and drops its UVs:
its material has no texture. In the body's textures, the texels under see-through glass (and the glass-like
gutters beside them) take the glass's colour, so filtering and mip levels at the edges of the body's charts
don't bring the pale colour back; every glass region's texels become glossy and not metal, by their weight.

``split_glass`` never raises: on bad input, or when there is no glass, it returns the mesh it was given,
untouched, and says why. numpy only: 1.2-1.4 s for the BMW's 94,000-100,000-face finals on one CPU core, half
of it the rays (faces binned in a grid), 0.1 s for a texture without transparent texels.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from .normals import _components

# --- Which texels and faces are glass -----------------------------------------------------------------
# Glass weight from alpha (1 at or below the first, 0 at or above the second) and from roughness
ALPHA = (0.6, 0.85)
ROUGHNESS = (0.7, 0.9)
# A face is glass when the mean weight over its texels is at least this
FACE = 0.5
# Points sampled per face: the centres of an n-fold subdivision of its UV triangle (n^2 of them)
FACE_SAMPLES = 4

# --- Which regions count (connected glass faces), in voxels of the mesh's grid ------------------------
# The BMW's smallest lamps measured 2100 square voxels and 15 voxels across, its windows 4800-60000 and
# 27-71; the specks and edge strips, under 200 and mostly under 4
MIN_AREA = 150.0
MIN_WIDTH = 4.0
# Warm coloured glass is left alone: a region whose straight colour (linear) has red as its largest channel,
# at least COLOUR_FLOOR, and a saturation ((largest - smallest) / largest) of SATURATED or more. The BMW's
# tail lights measured 0.79 and 0.99, a dragon's eyes 0.75; its windows 0.14-0.38 (teal, so not warm anyway)
SATURATED = 0.5
COLOUR_FLOOR = 0.01

# --- See-through --------------------------------------------------------------------------------------
RAYS = 16
# How far behind a face its own slab's back may be (TRELLIS.2's glass: about 2.3 voxels thick)
SLAB = 6.0
SEE_THROUGH = 0.6

# --- What glass becomes -------------------------------------------------------------------------------
# Linear luminance of glass's base colour (sRGB about 48 of 255). Its hue is TRELLIS.2's where that colour is
# bright enough to have one (TINT_LUMA, linear), else neutral, and no channel goes over MAX_TINT times the
# luminance (noise stays grey)
GLASS_LUMA = 0.03
TINT_LUMA = (0.004, 0.03)
MAX_TINT = 3.0
GLASS_ROUGHNESS = 0.05
# The see-through glass's opacity: OPACITY[0] + OPACITY[1] * TRELLIS.2's mean alpha over it, kept within
# OPACITY[2]..OPACITY[3]: the cup (alpha 0.02) 0.32, the cartoon car's windows 0.43, the BMW's 0.47-0.61 over
# its four textures
OPACITY = (0.3, 0.6, 0.3, 0.7)
# Texels that no glass face covers but that are glass-like (alpha and roughness) take the region of a glass
# texel up to this many texels away: the texture's gutters, for filtering and the first mip levels
GUTTER = 4

LUMA = np.array([0.2126, 0.7152, 0.0722])
GLASS = "glass"
# How many regions the report describes, largest first
_REPORTED = 12
# Texels tested at once by the UV rasteriser
_TEXELS = 1 << 21
# The material fields copied to the body's new material (all of trimesh's PBRMaterial's)
_FIELDS = (
    "name",
    "emissiveFactor",
    "emissiveTexture",
    "baseColorFactor",
    "metallicFactor",
    "roughnessFactor",
    "normalTexture",
    "occlusionTexture",
    "baseColorTexture",
    "metallicRoughnessTexture",
    "doubleSided",
    "alphaMode",
    "alphaCutoff",
)


def split_glass(mesh: Any, original: Any, voxel_size: Optional[float]) -> tuple[Any, dict]:
    """
    ``mesh`` (to_glb's textured trimesh, after unpremultiply, the projection and the shading normals) with its
    glass made glass: a ``trimesh.Scene`` of the body (OPAQUE, as before) and the see-through glass (BLEND),
    or, when no glass is see-through, a new trimesh mesh whose glass is glossy and not metal. ``original``
    is the base colour texture as to_glb made it, with TRELLIS.2's alpha (``unpremultiply`` drops it);
    ``voxel_size`` the mesh's grid spacing (TRELLIS.2's ``mesh.voxel_size``), which sizes regions.

    Returns ``(model, report)``; ``model`` is ``mesh`` itself, untouched, when nothing is glass or anything
    fails (the report's ``reason`` says which). The input is never modified.
    """
    started = time.perf_counter()
    report: dict = {"applied": False, "reason": ""}
    try:
        model, reason = _split(mesh, original, voxel_size, report)
        report["reason"] = reason
        report["applied"] = model is not mesh
    except Exception as err:  # noqa: BLE001 - glass is cosmetic: the model goes out as it was
        model = mesh
        report = {"applied": False, "reason": f"error: {type(err).__name__}: {' '.join(str(err).split())}"}
    report["seconds"] = round(time.perf_counter() - started, 3)
    return model, report


def face_count(model: Any) -> int:
    """Triangles in what ``split_glass`` returned: a trimesh mesh's, or every geometry's in a scene."""
    geometry = getattr(model, "geometry", None)
    if isinstance(geometry, dict):
        return int(sum(len(getattr(g, "faces", ())) for g in geometry.values()))
    return int(len(model.faces))


def summary(report: dict) -> dict:
    """The report without its per-region list, for a job's result."""
    return {key: value for key, value in report.items() if key != "regions"}


# --- The steps ----------------------------------------------------------------------------------------


def _split(mesh: Any, original: Any, voxel_size: Optional[float], report: dict) -> tuple[Any, str]:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    uv = getattr(visual, "uv", None)
    colour = getattr(material, "baseColorTexture", None)
    if material is None or uv is None or colour is None:
        return mesh, "the mesh has no base colour texture or UVs"
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or uv.shape != (len(vertices), 2):
        return mesh, "the mesh's faces or UVs don't fit its vertices"
    if faces.min() < 0 or faces.max() >= len(vertices):
        return mesh, "the mesh's faces or UVs don't fit its vertices"
    if not (np.isfinite(vertices).all() and np.isfinite(uv).all()):
        return mesh, "the mesh has non-finite vertices or UVs"
    voxel = float(voxel_size) if voxel_size is not None else 0.0
    if not (math.isfinite(voxel) and voxel > 0):
        return mesh, "no voxel size"
    srgb, alpha = _original(original)
    if alpha is None:
        return mesh, "the original texture has no alpha"
    if not (alpha < ALPHA[1]).any():
        return mesh, "no transparent texels"
    colour = colour if isinstance(colour, Image.Image) else Image.fromarray(np.asarray(colour))
    size = (colour.height, colour.width)
    alpha = _resized(alpha, size)
    if srgb.shape[:2] != size:
        srgb = np.stack([_resized(srgb[..., k], size) for k in range(3)], -1)
    rough_metal, rough = _roughness_metallic(material, size)
    weight = _texel_weight(alpha, rough)
    if not (weight > 0).any():
        return mesh, "no glass: the transparent texels are matte"

    # Faces: their mean weight, and for glass faces TRELLIS.2's alpha and straight colour (linear)
    glass = _sampled(uv, faces, weight).mean(1) >= FACE
    if not glass.any():
        return mesh, "no glass faces"
    ids = np.flatnonzero(glass)
    face_alpha = np.zeros(len(faces))
    face_alpha[ids] = _sampled(uv, faces[ids], alpha).mean(1)
    face_colour = np.zeros((len(faces), 3))
    face_colour[ids] = _srgb_to_linear(_sampled(uv, faces[ids], srgb)).mean(1)

    # Regions
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.sqrt(np.einsum("ij,ij->i", cross, cross))
    area = double_area / 2 / voxel**2
    welded = _welded(vertices, faces)
    region, regions = _regions(welded, glass)
    region_area = np.bincount(region[glass], area[glass], regions)
    perimeter = _perimeters(vertices, faces, welded, glass, region, regions) / voxel
    width = 2 * region_area / np.maximum(perimeter, 1e-12)
    big = (region_area >= MIN_AREA) & (width >= MIN_WIDTH)
    tint = np.stack([np.bincount(region[glass], area[glass] * face_colour[glass, k], regions) for k in range(3)], -1)
    tint /= np.maximum(region_area, 1e-12)[:, None]
    top = tint.max(1)
    saturation = (top - tint.min(1)) / np.maximum(top, 1e-12)
    coloured = big & (top >= COLOUR_FLOOR) & (saturation >= SATURATED) & (np.argmax(tint, 1) == 0)
    used = big & ~coloured
    total = float(area.sum()) or 1.0
    report["coloured"] = _tally(region, glass, area, coloured, total)
    if not used.any():
        return mesh, "no glass region is large enough" if not coloured.any() else "only coloured glass"

    # See-through: regions with something behind them, and the inner surfaces of their slabs
    unit = cross / np.maximum(double_area, 1e-300)[:, None]
    face_region = np.where(glass & used[np.maximum(region, 0)], region, -1)
    see, rays = _see_through(vertices, faces, unit, area, face_region, used, voxel, report)
    opaque = used & ~see
    report["see_through"] = _tally(region, glass, area, see, total)
    report["opaque"] = _tally(region, glass, area, opaque, total)
    verdicts = (see, opaque, coloured)
    report["regions"] = _described(region, glass, area, face_alpha, vertices, faces, tint, total, verdicts, rays)

    # The body's textures. Every used region's texels (and the glass-like gutters beside them) become glossy
    # and not metal, by their weight; under see-through faces, which no longer sample them, entirely, and
    # there (and in their gutters) they take the glass's colour too
    covered = _rasterise(uv, faces, face_region, size)
    labels = _grow(covered, weight > 0, GUTTER)
    texel_weight = np.where(labels >= 0, weight, 0.0)
    if see.any():
        texel_weight = np.where(np.isin(covered, np.flatnonzero(see)), 1.0, texel_weight)
        under_glass = np.isin(labels, np.flatnonzero(see))
        new_colour = _recoloured(colour, labels, np.where(under_glass, texel_weight, 0.0), _glass_colour(tint))
    else:
        new_colour = colour
    new_rough_metal = _glossy(rough_metal, texel_weight, material) if rough_metal is not None else None
    body_material = _copy_material(material, new_colour, new_rough_metal)

    if not see.any():
        if new_rough_metal is None:
            return mesh, "nothing see-through, and no metallic-roughness texture to make glossy"
        body = _part(mesh, vertices, faces, np.ones(len(faces), dtype=bool), uv, body_material)
        return body, "glossy glass (nothing see-through)"

    import trimesh

    in_glass = (face_region >= 0) & see[np.maximum(face_region, 0)]
    weights = area * in_glass
    mean_alpha = float((face_alpha * weights).sum() / max(weights.sum(), 1e-12))
    opacity = float(np.clip(OPACITY[0] + OPACITY[1] * mean_alpha, OPACITY[2], OPACITY[3]))
    mean_tint = (tint[see] * region_area[see, None]).sum(0) / max(float(region_area[see].sum()), 1e-12)
    rgb = _glass_colour(mean_tint[None])[0]
    glass_material = trimesh.visual.material.PBRMaterial(
        name=GLASS,
        baseColorFactor=[*(float(c) for c in rgb), opacity],
        metallicFactor=0.0,
        roughnessFactor=GLASS_ROUGHNESS,
        alphaMode="BLEND",
        doubleSided=bool(getattr(material, "doubleSided", False)),
    )
    report["opacity"] = round(opacity, 3)
    report["tint"] = _hex(rgb)
    scene = trimesh.Scene()
    body = _part(mesh, vertices, faces, ~in_glass, uv, body_material)
    scene.add_geometry(body, node_name="model", geom_name="model")
    scene.add_geometry(_part(mesh, vertices, faces, in_glass, None, glass_material), node_name=GLASS, geom_name=GLASS)
    return scene, "see-through glass"


def _tally(region: np.ndarray, glass: np.ndarray, area: np.ndarray, chosen: np.ndarray, total: float) -> dict:
    faces = glass & chosen[np.maximum(region, 0)]
    return {
        "regions": int(chosen.sum()),
        "faces": int(faces.sum()),
        "area_share": round(float(area[faces].sum() / total), 4),
    }


def _described(region, glass, area, face_alpha, vertices, faces, tint, total, verdicts, rays) -> list:
    """The _REPORTED largest regions: verdict, share of the surface, TRELLIS.2's alpha and colour, rays, centre."""
    see, opaque, coloured = verdicts
    count = len(see)
    region_area = np.bincount(region[glass], area[glass], count)
    alpha = np.bincount(region[glass], area[glass] * face_alpha[glass], count) / np.maximum(region_area, 1e-12)
    centre = np.stack(
        [np.bincount(region[glass], area[glass] * vertices[faces[glass]][:, :, k].mean(1), count) for k in range(3)], -1
    ) / np.maximum(region_area, 1e-12)[:, None]
    out = []
    for r in np.argsort(-region_area, kind="stable")[:_REPORTED]:
        verdict = "see-through" if see[r] else "opaque" if opaque[r] else "coloured" if coloured[r] else "small"
        entry = {
            "verdict": verdict,
            "area_share": round(float(region_area[r] / total), 5),
            "alpha": round(float(alpha[r]), 3),
            "colour": _hex(tint[r]),
            "centre": [round(float(c), 3) for c in centre[r]],
        }
        if r in rays:
            entry["rays"] = rays[r]
        out.append(entry)
    return out


# --- Texels -------------------------------------------------------------------------------------------


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _texel_weight(alpha: np.ndarray, rough: np.ndarray) -> np.ndarray:
    """How glass-like each texel is on its own (0-1): low alpha on a glossy surface."""
    return (1 - _smoothstep(*ALPHA, alpha)) * (1 - _smoothstep(*ROUGHNESS, rough))


def _original(original: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    to_glb's base colour as (straight colour (H, W, 3), sRGB; alpha (H, W)), 0-1; (None, None) without an
    alpha. A PIL image (RGBA, LA, or with transparency), or an array: (H, W, 4), (H, W, 2), or (H, W), the
    alpha channel on its own (then there is no colour to take a hue from: glass is neutral). uint8 or 0-1.
    """
    if original is None:
        return None, None
    if isinstance(original, Image.Image):
        if original.mode not in ("RGBA", "LA", "PA") and "transparency" not in original.info:
            return None, None
        rgba = np.asarray(original.convert("RGBA"), dtype=np.float32) / 255
    else:
        array = np.asarray(original)
        rgba = array.astype(np.float32) / (255 if array.dtype == np.uint8 else 1)
        if rgba.ndim == 2:
            rgba = np.concatenate([np.zeros((*rgba.shape, 3), np.float32), rgba[..., None]], -1)
        elif rgba.ndim == 3 and rgba.shape[2] == 2:
            rgba = np.concatenate([np.repeat(rgba[..., :1], 3, -1), rgba[..., 1:]], -1)
        elif rgba.ndim != 3 or rgba.shape[2] != 4:
            return None, None
    if min(rgba.shape[:2]) < 1:
        return None, None
    return rgba[..., :3], rgba[..., 3].astype(np.float64)


def _resized(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(H, W) float values at another (height, width), bilinear."""
    if values.shape == size:
        return values
    image = Image.fromarray(values.astype(np.float32), "F").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float64)


def _factor(material: Any, name: str) -> float:
    """A glTF factor (1 when unset)."""
    value = getattr(material, name, None)
    return 1.0 if value is None else float(value)


def _roughness_metallic(material: Any, size: tuple[int, int]) -> tuple[Optional[np.ndarray], np.ndarray]:
    """The metallic-roughness texture (RGB, 0-1, its own size; None without one) and roughness at ``size``."""
    image = getattr(material, "metallicRoughnessTexture", None)
    if image is None:
        return None, np.full(size, _factor(material, "roughnessFactor"))
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    return rgb, _resized(rgb[..., 1], size) * _factor(material, "roughnessFactor")


def _face_samples(n: int) -> np.ndarray:
    """Barycentric weights (n^2, 3) of the centres of the n^2 triangles of an n-fold subdivision."""
    points = []
    for i in range(n):
        for j in range(n - i):
            points.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            if i + j <= n - 2:
                points.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
    a = np.array(points)
    return np.stack([a[:, 0], a[:, 1], 1 - a[:, 0] - a[:, 1]], -1)


def _texel_index(uv: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the texel under each UV (trimesh's: v up, so row 0 is v = 1)."""
    height, width = size
    col = np.clip(np.floor(uv[..., 0] * width), 0, width - 1).astype(np.int64)
    row = np.clip(np.floor((1 - uv[..., 1]) * height), 0, height - 1).astype(np.int64)
    return row, col


def _sampled(uv: np.ndarray, faces: np.ndarray, values: np.ndarray, n: int = FACE_SAMPLES) -> np.ndarray:
    """A map's values (H, W) or (H, W, C) at each face's n^2 sample points: (F, n^2) or (F, n^2, C)."""
    points = np.einsum("sk,fkd->fsd", _face_samples(n), uv[faces])
    row, col = _texel_index(points, values.shape[:2])
    return values[row, col]


# --- Regions ------------------------------------------------------------------------------------------


def _welded(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces over vertices welded by position (+ 0.0 makes -0.0 equal 0.0): UV seams split nothing."""
    _, weld = np.unique(vertices + 0.0, axis=0, return_inverse=True)
    return weld.reshape(-1)[faces]


def _edges(welded: np.ndarray, chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The chosen faces' edges: (sorted key, owning face, slot k for the edge from corner k to k + 1)."""
    ids = np.flatnonzero(chosen)
    count = int(welded.max()) + 1
    keys, owners, slots = [], [], []
    for k in range(3):
        a, b = welded[ids, k], welded[ids, (k + 1) % 3]
        keys.append(np.minimum(a, b) * count + np.maximum(a, b))
        owners.append(ids)
        slots.append(np.full(len(ids), k))
    key, owner, slot = np.concatenate(keys), np.concatenate(owners), np.concatenate(slots)
    order = np.argsort(key, kind="stable")
    return key[order], owner[order], slot[order]


def _regions(welded: np.ndarray, glass: np.ndarray) -> tuple[np.ndarray, int]:
    """Glass faces joined across shared edges: each face's region (-1 for other faces) and the count."""
    key, owner, _ = _edges(welded, glass)
    same = key[1:] == key[:-1]
    compact = np.full(len(welded), -1)
    ids = np.flatnonzero(glass)
    compact[ids] = np.arange(len(ids))
    label = _components(len(ids), compact[owner[1:][same]], compact[owner[:-1][same]])
    _, dense = np.unique(label, return_inverse=True)
    region = np.full(len(welded), -1)
    region[ids] = dense.reshape(-1)
    return region, int(dense.max()) + 1 if len(ids) else 0


def _perimeters(vertices, faces, welded, glass: np.ndarray, region: np.ndarray, count: int) -> np.ndarray:
    """Each region's perimeter: its edges that no other glass face shares (a mesh border or other faces)."""
    key, owner, slot = _edges(welded, glass)
    starts = np.r_[True, key[1:] != key[:-1]]
    runs = np.cumsum(starts) - 1
    once = np.bincount(runs)[runs] == 1
    a = vertices[faces[owner, slot]]
    b = vertices[faces[owner, (slot + 1) % 3]]
    length = np.linalg.norm(b - a, axis=1)
    return np.bincount(region[owner[once]], length[once], count)


# --- See-through --------------------------------------------------------------------------------------


def _see_through(
    vertices: np.ndarray,
    faces: np.ndarray,
    unit: np.ndarray,
    area: np.ndarray,
    face_region: np.ndarray,
    used: np.ndarray,
    voxel: float,
    report: dict,
) -> tuple[np.ndarray, dict]:
    """
    Which regions are see-through (bool per region; see the module's docstring, step 5), and how each tested
    region's rays ended. Each ray, from a face inwards, ends as one of

    - "cavity": past the glass's own slab, the first surface that isn't glass faces it (a cabin, a drink);
    - "through": it passes through more glass and leaves the model (both side windows of a car, an empty
      bottle) without meeting anything else;
    - "solid": the first surface that isn't glass turns its back to it: the ray is inside a closed solid;
    - "exit": it crosses the back of its own slab and leaves: it started on the inner surface of glass,
      looking out, which says nothing about what is behind the glass seen from outside;
    - "open": nothing at all behind the face.

    A region is see-through when "cavity" and "through" make up SEE_THROUGH of its rays other than "exit"
    ones, and as many of those crossed a slab whose back is glass too: TRELLIS.2 builds windows and cups as
    slabs of glass, while glass painted on an opaque shell (the BMW's smoked tail lights, on its body) has
    the body's inner surface behind it, and see-through it shows the inside of the body. A region whose rays
    nearly all "exit" is the inner surface of a slab, see-through with the region whose back its rays crossed.
    """
    regions = len(used)
    origins, directions, owners = [], [], []
    rng = np.random.default_rng(0)
    for r in np.flatnonzero(used):
        ids = np.flatnonzero(face_region == r)
        pick = rng.choice(ids, size=min(RAYS, len(ids)), replace=False, p=area[ids] / area[ids].sum())
        centre = vertices[faces[pick]].mean(axis=1)
        # Just behind the face, so that the face itself is not met
        origins.append(centre - unit[pick] * (1e-3 * voxel))
        directions.append(-unit[pick])
        owners.append(np.full(len(pick), r))
    origin, direction, owner = np.concatenate(origins), np.concatenate(directions), np.concatenate(owners)
    hits = _ray_hits(origin, direction, vertices, faces)
    counts = {name: np.zeros(regions) for name in ("cavity", "through", "solid", "exit", "open")}
    # Rays that "showed" something past a slab whose back is glass
    glazed = np.zeros(regions)
    partners: dict[int, list] = {}
    slab = SLAB * voxel
    for i, (t, hit) in enumerate(hits):
        own = int(owner[i])
        partner, slab_seen, slab_glass, crossed, verdict = -1, False, False, False, None
        for distance, face in zip(t, hit):
            region = int(face_region[face])
            front = float(unit[face] @ direction[i]) < 0
            if distance <= slab and not front:  # the back of the glass's own slab
                slab_seen = True
                slab_glass |= region >= 0
                if partner < 0 and region >= 0 and region != own:
                    partner = region
                continue
            if region >= 0:  # more glass (another region, or the far side of this one): look through it
                crossed = True
                continue
            verdict = "cavity" if front else "solid"
            break
        if verdict is None:
            verdict = "through" if crossed else "exit" if slab_seen else "open"
        counts[verdict][own] += 1
        glazed[own] += slab_glass and verdict in ("cavity", "through")
        if verdict == "exit" and partner >= 0:
            partners.setdefault(own, []).append(partner)
    shown = counts["cavity"] + counts["through"]
    judged = shown + counts["solid"] + counts["open"]
    rays = judged + counts["exit"]
    # Judged by what a viewer outside would see, when enough of the region's rays tell
    telling = judged >= np.maximum(2, 0.25 * rays)
    see = used & telling & (glazed >= SEE_THROUGH * np.maximum(judged, 1))
    # The inner surface of a see-through slab: its rays cross the outer surface and leave
    for region, found in partners.items():
        if see[region] or telling[region]:
            continue
        values, times = np.unique(found, return_counts=True)
        best = int(values[np.argmax(times)])
        if see[best] and times.max() >= SEE_THROUGH * counts["exit"][region]:
            see[region] = True
    counts["glazed"] = glazed
    report["rays"] = {name: int(value.sum()) for name, value in counts.items()}
    names = tuple(counts)
    per_region = {int(r): {n: int(counts[n][r]) for n in names if counts[n][r]} for r in np.flatnonzero(used)}
    return see, per_region


def _ray_hits(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> list:
    """
    Per ray, the distances (sorted) and faces of every triangle it crosses ahead of its origin. Exact: the
    faces are binned in a grid (each in every cell its bounding box overlaps), each ray walks the cells it
    passes through (Amanatides and Woo), and only the faces met there are tested (Moller-Trumbore).
    """
    grid = _Grid(vertices, faces)
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    out = []
    for o, d in zip(origin, direction):
        ids = grid.faces_along(o, d)
        if len(ids) == 0:
            out.append((np.zeros(0), np.zeros(0, dtype=np.int64)))
            continue
        a, b, c = v0[ids], e1[ids], e2[ids]
        p = np.cross(d, c)
        det = np.einsum("ij,ij->i", b, p)
        ok = np.abs(det) > 1e-18
        inv = 1.0 / np.where(ok, det, 1.0)
        s = o - a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, b)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", c, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        order = np.argsort(t[hit], kind="stable")
        out.append((t[hit][order], ids[hit][order]))
    return out


class _Grid:
    """Faces binned in a uniform grid over the mesh's bounding box: which faces a ray can meet."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, per_cell: float = 4.0) -> None:
        corners = vertices[faces]
        self.lo = vertices.min(axis=0)
        extent = np.maximum(vertices.max(axis=0) - self.lo, 1e-12)
        cells = (len(faces) / per_cell) ** (1 / 3)
        self.count = np.clip(np.round(cells * extent / extent.max()), 1, 256).astype(np.int64)
        self.size = extent / self.count
        first = np.clip(np.floor((corners.min(axis=1) - self.lo) / self.size), 0, self.count - 1).astype(np.int64)
        last = np.clip(np.floor((corners.max(axis=1) - self.lo) / self.size), 0, self.count - 1).astype(np.int64)
        span = last - first + 1
        total = span.prod(axis=1)
        face = np.repeat(np.arange(len(faces)), total)
        k = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
        sx, sy = span[face, 0], span[face, 1]
        cell = self._index(first[face, 0] + k % sx, first[face, 1] + (k // sx) % sy, first[face, 2] + k // (sx * sy))
        order = np.argsort(cell, kind="stable")
        self.faces = face[order]
        self.starts = np.searchsorted(cell[order], np.arange(int(self.count.prod()) + 1))

    def _index(self, x: Any, y: Any, z: Any) -> Any:
        return (x * self.count[1] + y) * self.count[2] + z

    def faces_along(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """The faces in the cells the ray from ``origin`` along ``direction`` passes through."""
        lo, hi = self.lo, self.lo + self.size * self.count
        enter, leave = 0.0, math.inf
        for k in range(3):  # the ray's span inside the grid's box
            if direction[k] == 0:
                if not lo[k] <= origin[k] <= hi[k]:
                    return np.zeros(0, dtype=np.int64)
                continue
            a, b = (lo[k] - origin[k]) / direction[k], (hi[k] - origin[k]) / direction[k]
            enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
        if enter > leave:
            return np.zeros(0, dtype=np.int64)
        point = origin + direction * enter
        cell = [int(min(max(math.floor((point[k] - lo[k]) / self.size[k]), 0), self.count[k] - 1)) for k in range(3)]
        step = [1 if direction[k] > 0 else -1 for k in range(3)]
        next_t, delta = [], []
        for k in range(3):
            if direction[k] == 0:
                next_t.append(math.inf)
                delta.append(math.inf)
                continue
            edge = lo[k] + (cell[k] + (step[k] > 0)) * self.size[k]
            next_t.append(enter + (edge - point[k]) / direction[k])
            delta.append(self.size[k] / abs(direction[k]))
        found = []
        while True:
            index = int(self._index(*cell))
            begin, end = self.starts[index], self.starts[index + 1]
            if end > begin:
                found.append(self.faces[begin:end])
            axis = 0 if next_t[0] <= min(next_t[1], next_t[2]) else 1 if next_t[1] <= next_t[2] else 2
            if next_t[axis] > leave:
                break
            cell[axis] += step[axis]
            if not 0 <= cell[axis] < self.count[axis]:
                break
            next_t[axis] += delta[axis]
        return np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)


# --- Textures -----------------------------------------------------------------------------------------


def _rasterise(uv: np.ndarray, faces: np.ndarray, face_region: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The region of each texel whose centre a glass face covers in UV space ((H, W), -1 elsewhere)."""
    height, width = size
    labels = np.full(height * width, -1, dtype=np.int64)
    ids = np.flatnonzero(face_region >= 0)
    if len(ids) == 0:
        return labels.reshape(size)
    # Texel space with integer texel centres: x right, y down (row 0 is v = 1)
    x = uv[:, 0] * width - 0.5
    y = (1 - uv[:, 1]) * height - 0.5
    tri = faces[ids]
    x0, x1, x2 = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    y0, y1, y2 = y[tri[:, 0]], y[tri[:, 1]], y[tri[:, 2]]
    det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    xmin = np.clip(np.ceil(np.minimum(np.minimum(x0, x1), x2)), 0, width - 1).astype(np.int64)
    xmax = np.clip(np.floor(np.maximum(np.maximum(x0, x1), x2)), 0, width - 1).astype(np.int64)
    ymin = np.clip(np.ceil(np.minimum(np.minimum(y0, y1), y2)), 0, height - 1).astype(np.int64)
    ymax = np.clip(np.floor(np.maximum(np.maximum(y0, y1), y2)), 0, height - 1).astype(np.int64)
    nx, ny = np.maximum(xmax - xmin + 1, 0), np.maximum(ymax - ymin + 1, 0)
    counts = np.where(np.abs(det) > 1e-12, nx * ny, 0)
    ends = np.cumsum(counts)
    first = 0
    while first < len(ids):
        base = int(ends[first - 1]) if first else 0
        last = max(first + 1, int(np.searchsorted(ends, base + _TEXELS, side="right")))
        cnt = counts[first:last]
        total = int(cnt.sum())
        if total:
            local = np.repeat(np.arange(first, last), cnt)
            k = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            px = (xmin[local] + k % np.maximum(nx[local], 1)).astype(np.float64)
            py = (ymin[local] + k // np.maximum(nx[local], 1)).astype(np.float64)
            w0 = ((x1[local] - px) * (y2[local] - py) - (y1[local] - py) * (x2[local] - px)) / det[local]
            w1 = ((x2[local] - px) * (y0[local] - py) - (y2[local] - py) * (x0[local] - px)) / det[local]
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (1 - w0 - w1 >= -1e-6)
            texel = py[inside].astype(np.int64) * width + px[inside].astype(np.int64)
            labels[texel] = face_region[ids[local[inside]]]
        first = last
    return labels.reshape(size)


def _grow(labels: np.ndarray, allowed: np.ndarray, steps: int) -> np.ndarray:
    """Labels spread into unlabelled ``allowed`` texels, a texel (of 8 neighbours) per step."""
    height, width = labels.shape
    out = labels.reshape(-1).copy()
    free = np.flatnonzero(allowed.reshape(-1) & (out < 0))
    row, col = free // width, free % width
    for _ in range(steps):
        taken = np.full(len(free), -1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                r, c = row + dy, col + dx
                inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
                neighbour = np.where(inside, out[np.where(inside, r * width + c, 0)], -1)
                take = (taken < 0) & (neighbour >= 0)
                taken[take] = neighbour[take]
        new = taken >= 0
        if not new.any():
            break
        out[free[new]] = taken[new]
        free, row, col = free[~new], row[~new], col[~new]
    return out.reshape(height, width)


def _glass_colour(linear: np.ndarray) -> np.ndarray:
    """Glass's base colour (linear, (n, 3)) for TRELLIS.2's straight colour: GLASS_LUMA, in its hue if it has one."""
    luma = linear @ LUMA
    hue = linear / np.maximum(luma, 1e-6)[:, None]
    trust = _smoothstep(*TINT_LUMA, luma)[:, None]
    tint = np.clip(1 + trust * (hue - 1), 0.0, MAX_TINT)
    return GLASS_LUMA * tint / np.maximum(tint @ LUMA, 1e-6)[:, None]


def _recoloured(colour: Image.Image, labels: np.ndarray, weight: np.ndarray, glass_rgb: np.ndarray) -> Image.Image:
    """The base colour with each labelled texel blended (linear light, by ``weight``) towards its region's glass."""
    rgb = np.asarray(colour.convert("RGB"), dtype=np.float64) / 255
    where = (labels >= 0) & (weight > 0)
    w = weight[where][:, None]
    linear = _srgb_to_linear(rgb[where]) * (1 - w) + glass_rgb[labels[where]] * w
    rgb[where] = _linear_to_srgb(linear)
    out = Image.fromarray(np.round(rgb * 255).astype(np.uint8), "RGB")
    if colour.mode in ("RGBA", "LA"):  # an alpha channel (none after unpremultiply) is kept as it was
        out.putalpha(colour.getchannel("A"))
    return out


def _glossy(rough_metal: np.ndarray, weight: np.ndarray, material: Any) -> Image.Image:
    """The metallic-roughness texture glossy (GLASS_ROUGHNESS) and not metal where ``weight`` says glass."""
    w = _resized(weight, rough_metal.shape[:2])
    where = w > 0
    out = rough_metal.copy()
    glossy = GLASS_ROUGHNESS / max(_factor(material, "roughnessFactor"), 1e-6)
    k = w[where]
    rough, metal = out[..., 1][where], out[..., 2][where]
    out[..., 1][where] = rough * (1 - k) + np.minimum(rough, glossy) * k
    out[..., 2][where] = metal * (1 - k)
    return Image.fromarray(np.round(out * 255).astype(np.uint8), "RGB")


def _copy_material(material: Any, colour: Image.Image, rough_metal: Optional[Image.Image]) -> Any:
    """A new PBRMaterial like ``material`` (which stays as it was) with these textures."""
    import trimesh

    fields = {name: getattr(material, name, None) for name in _FIELDS}
    fields["baseColorTexture"] = colour
    if rough_metal is not None:
        fields["metallicRoughnessTexture"] = rough_metal
    return trimesh.visual.material.PBRMaterial(**{k: v for k, v in fields.items() if v is not None})


def _part(mesh: Any, vertices: np.ndarray, faces: np.ndarray, chosen: np.ndarray, uv: Any, material: Any) -> Any:
    """The chosen faces as a mesh of their own, with the input's vertex normals (and UVs, if given)."""
    import trimesh

    used, inverse = np.unique(faces[chosen], return_inverse=True)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return trimesh.Trimesh(
        vertices=vertices[used],
        faces=inverse.reshape(-1, 3),
        vertex_normals=normals[used],
        visual=trimesh.visual.TextureVisuals(uv=None if uv is None else uv[used], material=material),
        process=False,
    )


def _hex(linear: np.ndarray) -> str:
    srgb = np.round(_linear_to_srgb(np.asarray(linear, dtype=np.float64)) * 255).astype(int)
    return "#" + "".join(f"{int(c):02x}" for c in srgb)
