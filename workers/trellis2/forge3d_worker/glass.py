"""
TRELLIS.2's base colour made opaque for export: spurious alpha divided out, glass made dark, glossy glass.

TRELLIS.2 bakes an alpha channel with the base colour, and the GLB is opaque, so the alpha has to be
folded into the colour. It means two different things (Phase 8 measured the texture pass's raw RGBA on the
founder's BMW, three more textures of it, and Phase 2's helmet, cartoon car, bubble tea, potion and dragon):

- Spurious alpha. The 1024 texture pass sometimes returns an opaque surface's colour multiplied by an alpha
  below 1, and dividing by that alpha gives the paint back (``unpremultiply``). Phase 2's dragon had it over
  91% of its skin (alpha 0.3-0.7: dark blotches until Phase 4 divided it out), a later dragon along the
  edges of its thin parts (0.75-0.98). It sat on matte texels: none of Phase 2's dragon's had a roughness
  below 0.6, 93% of the later one's 0.85 or more, and the 5% below 0.6 were specks and thin edges.
- Glass: windows, lamps, a cup. Regions of low alpha (0 to 0.6) on glossy texels: roughness 0.0-0.55 in
  five of the six samples with glass, 0.6-0.85 in one texture of the BMW. Its colour comes either way:
  light (straight colour, linear 0.55 at alpha 0.2 on the cartoon car's windows) or dark (premultiplied,
  most of the BMW's), so dividing by alpha made the pale, opaque windows the founder saw.

Glass is exported opaque, not see-through. TRELLIS.2 rarely builds what is behind glass: rays through the
measured windows and the cup found nothing at all behind the glass's own thin slab 43-50% of the time
(with back faces culled, a see-through window would show the background through the car), and the rest hit
seats, the far door or the tea at depths that vary from texel to texel. So glass becomes what a window over a
dark cabin looks like: a dark tint (GLASS_LUMA, its hue from TRELLIS.2's colour), glossy (GLASS_ROUGHNESS),
not metal, so the Studio's environment reflects in it. One luminance for all of it, because TRELLIS.2's
glass colour is noise at that brightness, and the picture's projection, which recolours a paint the picture
shows in another colour all round, would multiply that noise into blotches (the bubble tea's cup).

Each texel's glass weight is the product of
- its alpha: 1 at or below ALPHA[0], 0 at or above ALPHA[1];
- its gloss: 1 at or below ROUGHNESS[0], 0 at or above ROUGHNESS[1] (what keeps the dragons' skin out);
- the region it lies in: the area-weighted share of the first two products over the surface in a box of
  REGION_CELLS^3 cells of CELL_VOXELS voxels (24 voxels across), taken at its largest among the boxes round
  the texel's own cell, from REGION[0] (none) to REGION[1] (all). Windows, lamps and cups are tens of voxels
  across, and their borders count as their middles do; spurious alpha on a glossy part runs along an edge a
  few voxels wide, which no box finds mostly glass, and is left to the division.
Colour, roughness and metallic each blend between the division's result and glass by that weight, so a
window's edge fades rather than steps. Texels with no weight come out exactly as ``unpremultiply`` makes them.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image

# The 1024 texture pass sometimes returns colour already multiplied by a spurious alpha, on up to
# half a model's texels. The GLB is opaque, so those texels showed up dark (a blotchy dragon's skin).
# Dividing by alpha is capped at 1/ALPHA_FLOOR so near-transparent texels' noise isn't blown up.
ALPHA_FLOOR = 0.25

# Glass weight from alpha (1 at or below the first, 0 at or above the second) and from roughness
ALPHA = (0.6, 0.85)
ROUGHNESS = (0.7, 0.9)
# Regions: the glass share is averaged over boxes of REGION_CELLS^3 cells of CELL_VOXELS voxels around each
# face (24 voxels across), and counts from REGION[0] (nothing) to REGION[1] (fully)
CELL_VOXELS = 8.0
REGION_CELLS = 3
REGION = (0.25, 0.5)
# What glass becomes: linear luminance of its base colour (sRGB about 48 of 255), roughness, no metal.
# Its hue is TRELLIS.2's where that colour is bright enough to have one (TINT_LUMA, linear), else neutral
GLASS_LUMA = 0.03
GLASS_ROUGHNESS = 0.08
TINT_LUMA = (0.004, 0.03)
# A tint's channels are kept within this factor of its luminance (a red lamp stays red, noise stays grey)
MAX_TINT = 3.0
# Gutter texels (no triangle covers them) take the region of the chart up to this many texels away: enough
# for filtering and the first mip levels; further out they are only divided
GUTTER = 8
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def unpremultiply(texture: Image.Image, floor: float = ALPHA_FLOOR) -> Image.Image:
    """Opaque RGB: colour divided by its alpha in linear light. Fully opaque texels stay as they are."""
    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    linear = srgb_to_linear(rgb)
    linear = np.clip(linear / np.maximum(alpha, floor), 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    out = np.where(alpha < 1.0, srgb, rgb)
    return Image.fromarray(np.round(out * 255).astype(np.uint8), "RGB")


def texel_weight(alpha: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """How glass-like each texel is on its own (0-1): low alpha on a glossy surface."""
    return (1 - smoothstep(*ALPHA, alpha)) * (1 - smoothstep(*ROUGHNESS, roughness))


def glass_colour(linear: np.ndarray) -> np.ndarray:
    """Glass's base colour (linear, (..., 3)) for TRELLIS.2's colour there: GLASS_LUMA, in its hue when it has one."""
    luma = linear @ LUMA
    hue = linear / np.maximum(luma, 1e-6)[..., None]
    trust = smoothstep(*TINT_LUMA, luma)[..., None]
    tint = np.clip(1 + trust * (hue - 1), 0.0, MAX_TINT)
    return GLASS_LUMA * tint / np.maximum(tint @ LUMA, 1e-6)[..., None]


def faces_at_texels(uv: np.ndarray, faces: np.ndarray, size: tuple[int, int], device: Any = None) -> np.ndarray:
    """
    The face covering each texel of a (height, width) texture, -1 where none (trimesh UVs, v up), as to_glb
    rasterizes them. On the GPU when there is one, and on the CPU if the GPU runs out of memory.
    """
    import torch

    from . import uv_raster

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def rasterize(on: Any) -> np.ndarray:
        uv_t = torch.tensor(np.asarray(uv, dtype=np.float32), device=on)
        clip = torch.cat([uv_t * 2 - 1, torch.zeros_like(uv_t[:, :1]), torch.ones_like(uv_t[:, :1])], -1)[None]
        tri = torch.tensor(np.asarray(faces, dtype=np.int64), device=on)
        rast, _ = uv_raster.rasterize(None, clip, tri, resolution=[size[0], size[1]])
        # uv_raster's rows run bottom-up (v = 0 first), the image's top-down
        return (rast[0, ..., 3].flip(0).round().long() - 1).cpu().numpy()

    try:
        return rasterize(device)
    except torch.OutOfMemoryError:
        if device.type == "cpu":
            raise
        torch.cuda.empty_cache()
        return rasterize(torch.device("cpu"))


def _neighbours(keys: np.ndarray, span: int):
    """For each offset of the REGION_CELLS^3 box: which occupied cells have that neighbour, and its index."""
    cells = np.stack([keys // (span * span), (keys // span) % span, keys % span], axis=1)
    reach = REGION_CELLS // 2
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                other = cells + np.array([dx, dy, dz])
                key = (other[:, 0] * span + other[:, 1]) * span + other[:, 2]
                at = np.clip(np.searchsorted(keys, key), 0, len(keys) - 1)
                hit = (keys[at] == key) & (other >= 0).all(axis=1)
                yield hit, at[hit]


def region_share(vertices: np.ndarray, faces: np.ndarray, weight: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Per face: how much of its region is glass. Faces are binned into cells of CELL_VOXELS voxels; each cell's
    share is the area-weighted mean of ``weight`` (per face; NaN for faces with no texels, left out) over the
    REGION_CELLS^3 cells around it, and a face gets the largest share among the cells around its own. So a
    window's border, where the box around it is half body, counts as much as its middle, while a speck or a
    thin edge, which no box finds mostly glass, stays low.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    corners = v[f]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    known = np.isfinite(weight)
    area = np.where(known, area, 0.0)
    weight = np.where(known, weight, 0.0)
    cell = np.floor((corners.mean(axis=1) - v.min(axis=0)) / (CELL_VOXELS * float(voxel_size))).astype(np.int64)
    span = int(cell.max()) + REGION_CELLS + 1
    keys, inverse = np.unique(((cell[:, 0] * span) + cell[:, 1]) * span + cell[:, 2], return_inverse=True)
    inverse = inverse.reshape(-1)
    glass = np.bincount(inverse, area * weight, len(keys))
    total = np.bincount(inverse, area, len(keys))
    near_glass, near_total = np.zeros(len(keys)), np.zeros(len(keys))
    for hit, at in _neighbours(keys, span):
        near_glass[hit] += glass[at]
        near_total[hit] += total[at]
    share = near_glass / np.maximum(near_total, 1e-30)
    largest = share.copy()
    for hit, at in _neighbours(keys, span):
        largest[hit] = np.maximum(largest[hit], share[at])
    return largest[inverse]


def fill(values: np.ndarray, known: np.ndarray, needed: Optional[np.ndarray] = None, steps: int = GUTTER) -> np.ndarray:
    """
    ``values`` (H, W) where ``known``; elsewhere grown out from the known texels a texel at a time, each new
    texel the mean of its known neighbours (of 8), for at most ``steps`` texels or until every ``needed``
    texel has a value. Texels left unknown are 0.
    """
    value = np.where(known, values, 0.0).astype(np.float64)
    have = known.copy()
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for _ in range(steps):
        if needed is not None and not (needed & ~have).any():
            break
        total, count = np.zeros_like(value), np.zeros_like(value)
        padded_value, padded_have = np.pad(value * have, 1), np.pad(have.astype(np.float64), 1)
        h, w = value.shape
        for dy, dx in shifts:
            total += padded_value[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            count += padded_have[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        new = ~have & (count > 0)
        if not new.any():
            break
        value[new] = total[new] / count[new]
        have |= new
    return np.where(have, value, 0.0)


def weights(
    mesh: Any, alpha: np.ndarray, roughness: np.ndarray, voxel_size: float, device: Any = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Each texel's glass weight (H, W), 0-1: texel_weight, times how much of its region is glass. And which
    texels a triangle covers (None when no texel could be glass: nothing was rasterized).
    """
    own = texel_weight(alpha, roughness)
    if not (own > 0).any():
        return own, None
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face = faces_at_texels(mesh.visual.uv, faces, alpha.shape, device)
    covered = face >= 0
    count = np.bincount(face[covered], None, len(faces))
    mean = np.bincount(face[covered], own[covered], len(faces)) / np.maximum(count, 1)
    mean = np.where(count > 0, mean, np.nan)
    region = smoothstep(*REGION, region_share(mesh.vertices, faces, mean, voxel_size))
    at_texel = np.zeros(alpha.shape)
    at_texel[covered] = region[face[covered]]
    # Gutters (texels no triangle covers, which to_glb inpaints from its charts) take their charts' region, so
    # filtering at a chart's edge agrees with it
    return own * fill(at_texel, covered, needed=own > 0), covered


def resized(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(H, W) float values at another (height, width), bilinear."""
    if values.shape == size:
        return values
    image = Image.fromarray(values.astype(np.float32), "F").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float64)


def roughness_factor(material: Any) -> float:
    """glTF's roughnessFactor (1 when unset), which scales the texture's green channel."""
    factor = getattr(material, "roughnessFactor", None)
    return 1.0 if factor is None else float(factor)


def roughness_metallic(material: Any, size: tuple[int, int]) -> tuple[Optional[np.ndarray], np.ndarray]:
    """The material's metallic-roughness texture (RGB, 0-1, or None), and its roughness at ``size``."""
    image = getattr(material, "metallicRoughnessTexture", None)
    if image is None:
        return None, np.full(size, roughness_factor(material))
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    return rgb, resized(rgb[..., 1], size) * roughness_factor(material)


def export_textures(mesh: Any, voxel_size: float, device: Any = None) -> dict:
    """
    ``mesh`` (to_glb's textured trimesh) with its base colour made opaque RGB in place: divided by alpha
    (``unpremultiply``), except that glass becomes dark glass, and the metallic-roughness texture glossy and
    not metal there. Returns ``{"glass": share of the covered texels that are glass}``. Without an alpha
    below ALPHA[1] anywhere, this is ``unpremultiply`` alone.
    """
    material = mesh.visual.material
    texture = material.baseColorTexture
    texture = texture if isinstance(texture, Image.Image) else Image.fromarray(np.asarray(texture))
    divided = unpremultiply(texture)
    if texture.mode not in ("RGBA", "LA"):
        material.baseColorTexture = divided
        return {"glass": 0.0}
    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255
    alpha = rgba[..., 3]
    size = alpha.shape
    if not (alpha < ALPHA[1]).any():
        material.baseColorTexture = divided
        return {"glass": 0.0}
    mr, rough = roughness_metallic(material, size)
    weight, covered = weights(mesh, alpha, rough, voxel_size, device)
    if covered is None or not (weight > 0).any():
        material.baseColorTexture = divided
        return {"glass": 0.0}
    # Colour: the division's result and glass, blended in linear light; texels without weight untouched
    out = np.asarray(divided, dtype=np.float32) / 255
    where = weight > 0
    w = weight[where][:, None]
    linear = srgb_to_linear(out[where]) * (1 - w) + glass_colour(srgb_to_linear(rgba[..., :3][where])) * w
    out[where] = linear_to_srgb(linear)
    colour = Image.fromarray(np.round(out * 255).astype(np.uint8), "RGB")
    rough_metal = None
    if mr is not None:
        # Glossy and not metal where it is glass (glTF: roughness in green, metallic in blue)
        at = resized(weight, mr.shape[:2])
        glossy = GLASS_ROUGHNESS / max(roughness_factor(material), 1e-6)
        mr = mr.copy()
        mr[..., 1] = mr[..., 1] * (1 - at) + np.minimum(mr[..., 1], glossy) * at
        mr[..., 2] = mr[..., 2] * (1 - at)
        rough_metal = Image.fromarray(np.round(mr * 255).astype(np.uint8), "RGB")
    # The material changes only now, with everything worked out
    material.baseColorTexture = colour
    if rough_metal is not None:
        material.metallicRoughnessTexture = rough_metal
    # The share of the surface that is glass: the weight over the texels a triangle covers
    return {"glass": round(float(weight[covered].mean()), 4) if covered.any() else 0.0}
