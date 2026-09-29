"""Checks the nvdiffrast stand-in against a brute-force reference and analytic results (CPU only)."""

import importlib

import numpy as np
import pytest
import torch

from forge3d_worker import uv_raster


def uv_to_clip(uvs: torch.Tensor) -> torch.Tensor:
    """What o_voxel.postprocess.to_glb feeds the rasterizer: UVs as clip-space x/y, z = 0, w = 1."""
    return torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1)[None]


def reference_rasterize(uvs: np.ndarray, faces: np.ndarray, height: int, width: int) -> np.ndarray:
    """Slow float64 rasterizer with the same conventions (bottom-up rows, highest id wins)."""
    out = np.zeros((height, width, 4))
    px = uvs[:, 0] * width - 0.5
    py = uvs[:, 1] * height - 0.5
    for i in range(height):
        for j in range(width):
            for f in range(len(faces) - 1, -1, -1):
                a, b, c = faces[f]
                area = (px[b] - px[a]) * (py[c] - py[a]) - (py[b] - py[a]) * (px[c] - px[a])
                if abs(area) < 1e-12:
                    continue
                w0 = ((px[c] - px[b]) * (i - py[b]) - (py[c] - py[b]) * (j - px[b])) / area
                w1 = ((px[a] - px[c]) * (i - py[c]) - (py[a] - py[c]) * (j - px[c])) / area
                w2 = 1 - w0 - w1
                if min(w0, w1, w2) >= -1e-6:
                    out[i, j] = (w0, w1, 0.0, f + 1)
                    break
    return out


def grid_mesh(n: int):
    """A UV-mapped n x n grid covering [0, 1]^2, with 3D position = an affine function of UV."""
    lin = torch.linspace(0, 1, n + 1)
    v, u = torch.meshgrid(lin, lin, indexing="ij")
    uvs = torch.stack([u.flatten(), v.flatten()], dim=1)
    faces = []
    for r in range(n):
        for c in range(n):
            a = r * (n + 1) + c
            faces += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    faces = torch.tensor(faces, dtype=torch.int32)
    positions = torch.stack([2 * uvs[:, 0] - 1, 0.5 * uvs[:, 1] + 0.25 * uvs[:, 0], 3 * uvs[:, 1]], dim=1)
    return uvs, faces, positions


def affine(uvs: torch.Tensor) -> torch.Tensor:
    return torch.stack([2 * uvs[..., 0] - 1, 0.5 * uvs[..., 1] + 0.25 * uvs[..., 0], 3 * uvs[..., 1]], dim=-1)


def test_matches_brute_force_reference():
    gen = torch.Generator().manual_seed(7)
    uvs = torch.rand((60, 2), generator=gen)
    faces = torch.randint(0, 60, (25, 3), generator=gen, dtype=torch.int32)
    height, width = 24, 32

    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[height, width])
    ref = reference_rasterize(uvs.double().numpy(), faces.numpy(), height, width)

    got = rast[0].numpy()
    ids_equal = got[..., 3] == ref[..., 3]
    # Only pixel centres sitting within rounding of an edge may differ between float32 and float64
    assert ids_equal.mean() > 0.995
    both = ids_equal & (ref[..., 3] > 0)
    assert both.sum() > 50
    np.testing.assert_allclose(got[both][:, :2], ref[both][:, :2], atol=1e-4)


def test_full_coverage_and_exact_interpolation_on_a_uv_grid():
    uvs, faces, positions = grid_mesh(6)
    size = 64
    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[size, size])
    assert bool((rast[0, ..., 3] > 0).all()), "every texel centre lies inside the grid"

    pos, _ = uv_raster.interpolate(positions[None], rast, faces)
    centres = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    tv, tu = torch.meshgrid(centres, centres, indexing="ij")  # row i -> v (bottom-up)
    expected = affine(torch.stack([tu, tv], dim=-1))
    torch.testing.assert_close(pos[0], expected, atol=1e-5, rtol=0)


def test_rows_are_bottom_up_like_nvdiffrast():
    # One triangle in the lower half of UV space (v < 0.5) must land in the first rows
    uvs = torch.tensor([[0.1, 0.05], [0.9, 0.05], [0.5, 0.45]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[32, 32])
    covered_rows = torch.nonzero(rast[0, ..., 3] > 0)[:, 0]
    assert covered_rows.min() >= 1 and covered_rows.max() < 16


def test_barycentric_weights_follow_vertex_order():
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    size = 16
    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[size, size])
    # Pixel (row 0, col 12) sits near vertex 1 at uv (1, 0): its weight v must dominate
    u, v = rast[0, 0, 12, 0].item(), rast[0, 0, 12, 1].item()
    assert v > 0.7 and u < 0.25
    attr = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out, _ = uv_raster.interpolate(attr[None], rast, faces)
    torch.testing.assert_close(out[0, 0, 12], torch.tensor([u, v, 1 - u - v]), atol=1e-6, rtol=0)


def test_chunked_bake_like_to_glb_matches_single_call(monkeypatch):
    """Replays the texture-bake loop from o_voxel.postprocess.to_glb (chunks with id offsets)."""
    uvs, faces, positions = grid_mesh(8)
    size = 48
    uvs_rast = uv_to_clip(uvs)

    single, _ = uv_raster.rasterize(None, uvs_rast, faces, resolution=[size, size])

    chunk = 7  # to_glb uses 100000; a small chunk exercises the same logic
    rast = torch.zeros((1, size, size, 4))
    for i in range(0, faces.shape[0], chunk):
        rast_chunk, _ = uv_raster.rasterize(None, uvs_rast, faces[i : i + chunk], resolution=[size, size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)
    mask = rast[0, ..., 3] > 0
    pos = uv_raster.interpolate(positions[None], rast, faces)[0][0]

    assert bool(mask.all())
    torch.testing.assert_close(pos, uv_raster.interpolate(positions[None], single, faces)[0][0], atol=1e-5, rtol=0)


def test_small_batches_give_identical_results(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    uvs = torch.rand((200, 2), generator=gen)
    faces = torch.randint(0, 200, (120, 3), generator=gen, dtype=torch.int32)
    full, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[40, 40])
    monkeypatch.setattr(uv_raster, "MAX_CANDIDATES", 50)
    batched, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[40, 40])
    torch.testing.assert_close(batched, full)


def test_empty_and_degenerate_input():
    uvs = torch.tensor([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]])  # collinear
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces, resolution=[8, 8])
    assert float(rast.abs().sum()) == 0.0
    rast, _ = uv_raster.rasterize(None, uv_to_clip(uvs), faces[:0], resolution=[8, 8])
    assert rast.shape == (1, 8, 8, 4) and float(rast.abs().sum()) == 0.0


def test_rejects_perspective_input():
    pos = torch.tensor([[[0.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 2.0]]])
    with pytest.raises(NotImplementedError):
        uv_raster.rasterize(None, pos, torch.tensor([[0, 1, 2]]), resolution=[4, 4])


def test_install_replaces_nvdiffrast_and_blocks_other_calls():
    uv_raster.install()
    dr = importlib.import_module("nvdiffrast.torch")
    assert dr.rasterize is uv_raster.rasterize
    assert isinstance(dr.RasterizeCudaContext(), uv_raster.RasterizeCudaContext)
    assert not hasattr(dr, "texture")
    with pytest.raises(NotImplementedError):
        dr.antialias  # noqa: B018
