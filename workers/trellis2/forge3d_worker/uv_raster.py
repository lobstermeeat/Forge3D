"""
A permissively licensed stand-in for the two nvdiffrast calls TRELLIS.2 needs.

``o_voxel.postprocess.to_glb`` bakes textures by rasterizing the mesh in UV space
(``dr.rasterize``) and interpolating the 3D position of every texel
(``dr.interpolate``). nvdiffrast's license allows research and evaluation use
only, so this module implements exactly that subset in plain PyTorch and is
registered as ``nvdiffrast.torch`` before ``o_voxel`` is imported (``install()``).
Anything else from nvdiffrast raises instead of silently doing the wrong thing.

Conventions follow nvdiffrast, so ``to_glb`` works unchanged:

- input positions are clip space ``(x, y, z, w)``; pixel ``(row i, col j)`` samples
  NDC ``((j + 0.5) / W * 2 - 1, (i + 0.5) / H * 2 - 1)``
- memory order is bottom-up: row 0 is ``y = -1``, as in OpenGL
- output channels are ``(u, v, z/w, triangle_id + 1)``; ``u`` weights vertex 0,
  ``v`` vertex 1 and ``1 - u - v`` vertex 2; empty pixels are all zeros
- where triangles share a pixel (edges), the highest triangle id wins

Only affine input (``w == 1``) is supported, which is what a UV-space bake uses.
"""

from __future__ import annotations

import sys
import types
from typing import Optional, Sequence, Tuple

import torch

# Texel candidates tested per batch; bounds peak memory (~180 bytes each, ~380 MB per batch).
MAX_CANDIDATES = 1 << 21
# Barycentric tolerance: pixel centres exactly on an edge count as inside.
EDGE_EPS = 1e-6


class RasterizeCudaContext:
    """nvdiffrast needs a GL/CUDA context object; this rasterizer doesn't."""

    def __init__(self, device: Optional[torch.device] = None, **_: object) -> None:
        self.device = device


RasterizeGLContext = RasterizeCudaContext


def _pixel_coords(pos: torch.Tensor, height: int, width: int):
    if pos.dim() != 3 or pos.shape[0] != 1 or pos.shape[2] != 4:
        raise NotImplementedError("uv_raster supports a single (1, V, 4) position tensor only")
    p = pos[0].float()
    w = p[:, 3]
    if not torch.allclose(w, torch.ones_like(w)):
        raise NotImplementedError("uv_raster supports affine (w == 1) input only, as used for UV baking")
    # Pixel space where integer coordinates are pixel centres
    px = (p[:, 0] + 1) * 0.5 * width - 0.5
    py = (p[:, 1] + 1) * 0.5 * height - 0.5
    return px, py, p[:, 2]


def _barycentric(px, py, tri, x, y):
    """Weights of vertices 0, 1, 2 of each triangle ``tri`` at points ``(x, y)``."""
    ax, ay = px[tri[:, 0]], py[tri[:, 0]]
    bx, by = px[tri[:, 1]], py[tri[:, 1]]
    cx, cy = px[tri[:, 2]], py[tri[:, 2]]
    # Edge functions relative to a triangle vertex keep float32 precise on large textures
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    w0 = ((cx - bx) * (y - by) - (cy - by) * (x - bx)) / area
    w1 = ((ax - cx) * (y - cy) - (ay - cy) * (x - cx)) / area
    return w0, w1, 1 - w0 - w1


def rasterize(
    glctx: object,
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution: Sequence[int],
    ranges: Optional[torch.Tensor] = None,
    grad_db: bool = True,
) -> Tuple[torch.Tensor, None]:
    """Rasterize triangles; returns ``(rast[1, H, W, 4], None)`` like ``nvdiffrast.torch.rasterize``."""
    if ranges is not None:
        raise NotImplementedError("uv_raster does not support range mode")
    height, width = int(resolution[0]), int(resolution[1])
    device = pos.device
    out = torch.zeros((1, height, width, 4), dtype=torch.float32, device=device)
    tri = tri.long()
    if tri.shape[0] == 0:
        return out, None

    px, py, pz = _pixel_coords(pos, height, width)
    x0, x1, x2 = px[tri[:, 0]], px[tri[:, 1]], px[tri[:, 2]]
    y0, y1, y2 = py[tri[:, 0]], py[tri[:, 1]], py[tri[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    # Integer pixel-centre bounding boxes, clipped to the image
    xmin = torch.ceil(torch.minimum(torch.minimum(x0, x1), x2)).clamp(min=0).long()
    xmax = torch.floor(torch.maximum(torch.maximum(x0, x1), x2)).clamp(max=width - 1).long()
    ymin = torch.ceil(torch.minimum(torch.minimum(y0, y1), y2)).clamp(min=0).long()
    ymax = torch.floor(torch.maximum(torch.maximum(y0, y1), y2)).clamp(max=height - 1).long()
    nx = (xmax - xmin + 1).clamp(min=0)
    ny = (ymax - ymin + 1).clamp(min=0)
    counts = torch.where(area.abs() > 1e-12, nx * ny, torch.zeros_like(nx))

    # Pass 1: which triangle owns each pixel (highest id among those covering it)
    owner = torch.full((height * width,), -1, dtype=torch.long, device=device)
    ends = torch.cumsum(counts, 0)
    first = 0
    num_tris = tri.shape[0]
    while first < num_tris:
        # Grow the batch until it holds MAX_CANDIDATES texels (always at least one triangle)
        base = int(ends[first - 1]) if first > 0 else 0
        last = int(torch.searchsorted(ends, torch.tensor(base + MAX_CANDIDATES, device=device), right=True))
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
        w_box = nx[t]
        xs = xmin[t] + k % w_box
        ys = ymin[t] + torch.div(k, w_box, rounding_mode="floor")
        w0, w1, w2 = _barycentric(px, py, tri[t], xs.float(), ys.float())
        inside = (w0 >= -EDGE_EPS) & (w1 >= -EDGE_EPS) & (w2 >= -EDGE_EPS)
        owner.scatter_reduce_(0, (ys * width + xs)[inside], t[inside], reduce="amax", include_self=True)

    # Pass 2: barycentrics and depth of the owning triangle at each covered pixel centre
    pix = torch.nonzero(owner >= 0).squeeze(1)
    if pix.numel() == 0:
        return out, None
    t = owner[pix]
    xs = (pix % width).float()
    ys = torch.div(pix, width, rounding_mode="floor").float()
    w0, w1, w2 = _barycentric(px, py, tri[t], xs, ys)
    z = w0 * pz[tri[t, 0]] + w1 * pz[tri[t, 1]] + w2 * pz[tri[t, 2]]
    flat = out.view(height * width, 4)
    flat[pix] = torch.stack([w0, w1, z, (t + 1).float()], dim=1)
    return out, None


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    tri: torch.Tensor,
    rast_db: Optional[torch.Tensor] = None,
    diff_attrs: object = None,
) -> Tuple[torch.Tensor, None]:
    """Interpolate vertex attributes over a raster; returns ``(out[1, H, W, C], None)``."""
    if attr.dim() == 3:
        if attr.shape[0] != 1:
            raise NotImplementedError("uv_raster supports a single attribute minibatch only")
        attr = attr[0]
    tri = tri.long()
    height, width = rast.shape[1], rast.shape[2]
    flat = rast.reshape(-1, 4)
    ids = flat[:, 3].round().long() - 1
    valid = ids >= 0
    out = torch.zeros((height * width, attr.shape[1]), dtype=attr.dtype, device=attr.device)
    if bool(valid.any()):
        t = tri[ids[valid]]
        u = flat[valid, 0:1].to(attr.dtype)
        v = flat[valid, 1:2].to(attr.dtype)
        out[valid] = u * attr[t[:, 0]] + v * attr[t[:, 1]] + (1 - u - v) * attr[t[:, 2]]
    return out.view(1, height, width, attr.shape[1]), None


class UnsupportedNvdiffrastCall(AttributeError, NotImplementedError):
    """Raised for any nvdiffrast API this stand-in doesn't implement."""


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedNvdiffrastCall(
        f"nvdiffrast.torch.{name} is not available: FORGE 3D replaces nvdiffrast "
        "(research-only license) with forge3d_worker.uv_raster, which only covers UV baking"
    )


def install() -> None:
    """Register this module as ``nvdiffrast`` / ``nvdiffrast.torch``. Call before importing o_voxel."""
    this = sys.modules[__name__]
    package = types.ModuleType("nvdiffrast")
    package.torch = this  # type: ignore[attr-defined]
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["nvdiffrast"] = package
    sys.modules["nvdiffrast.torch"] = this
