"""
One-off check that forge3d_worker.uv_raster matches nvdiffrast on a real GPU.

Evaluation only: run it on a dev machine that has nvdiffrast installed (its license allows
research and evaluation use). It is not part of the worker image.

    python scripts/compare_nvdiffrast.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nvdiffrast.torch as dr  # noqa: E402  (the real one: don't call uv_raster.install() here)

from forge3d_worker import uv_raster  # noqa: E402


def random_uv_mesh(num_faces: int, generator: torch.Generator):
    uvs = torch.rand((num_faces * 3, 2), generator=generator, device="cuda")
    # Small triangles, like a UV atlas
    centres = uvs[0::3].repeat_interleave(3, dim=0)
    uvs = (centres + (uvs - centres) * 0.05).clamp(0, 1)
    faces = torch.arange(num_faces * 3, device="cuda", dtype=torch.int32).view(-1, 3)
    return uvs, faces


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    ctx = dr.RasterizeCudaContext()
    worst = 0.0
    for size, faces_count in [(512, 2_000), (2048, 50_000), (4096, 200_000)]:
        uvs, faces = random_uv_mesh(faces_count, generator)
        pos = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1)[None]
        ref, _ = dr.rasterize(ctx, pos.contiguous(), faces, resolution=[size, size])
        got, _ = uv_raster.rasterize(None, pos, faces, resolution=[size, size])

        ref_cov, got_cov = ref[..., 3] > 0, got[..., 3] > 0
        coverage_diff = (ref_cov != got_cov).float().mean().item()
        same = ref_cov & got_cov & (ref[..., 3] == got[..., 3])
        bary_err = (ref[..., :2] - got[..., :2]).abs()[same].max().item() if same.any() else 0.0

        attr = torch.rand((uvs.shape[0], 3), generator=generator, device="cuda")
        ref_pos, _ = dr.interpolate(attr[None], ref, faces)
        got_pos, _ = uv_raster.interpolate(attr[None], got, faces)
        pos_err = (ref_pos - got_pos).abs()[same].max().item() if same.any() else 0.0

        worst = max(worst, coverage_diff)
        print(
            f"{size}px, {faces_count} faces: coverage differs on {coverage_diff:.5%} of texels, "
            f"max barycentric error {bary_err:.2e}, max interpolation error {pos_err:.2e}"
        )
    # Only texel centres lying exactly on an edge may be assigned differently
    if worst > 1e-3:
        raise SystemExit("uv_raster disagrees with nvdiffrast; do not ship")
    print("uv_raster matches nvdiffrast")


if __name__ == "__main__":
    main()
