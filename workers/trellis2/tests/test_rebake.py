"""
The cached texture layout (rebake.py): TRELLIS.2's own to_glb runs on the CPU, with stand-ins for its CUDA
pieces (CuMesh's hole filling, remesh, UV unwrap and BVH; FlexGEMM's grid_sample_3d; OpenCV's inpaint when
it isn't installed) and this worker's nvdiffrast stand-in, as production installs it. The capture must see
what to_glb bakes with and change nothing; a rebake must give what to_glb gives for a new texture of the
same shape, bit for bit; anything else must fall back to to_glb in full. Needs TRELLIS2_SRC (TRELLIS.2's
source) for to_glb; the tests that don't run it run without.
"""

import base64
import dataclasses
import importlib.util
import io
import itertools
import math
import os
import pathlib
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from forge3d_worker import rebake, service, uv_raster
from forge3d_worker.pipeline import SHAPE, Trellis2Runtime
from forge3d_worker.settings import PRESETS

# Production's final, at a texture size the CPU stand-ins bake in a blink
FINAL = dataclasses.replace(PRESETS["final"], texture_size=64)
LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
RESOLUTION = 16  # the voxel grid's: voxel_size 1 / 16 over the [-0.5, 0.5] cube
INFLATE = 1.02  # the stand-in remesh scales the mesh by this, so the BVH projection has work to do


def to_glb_source():
    source = os.environ.get("TRELLIS2_SRC")
    path = pathlib.Path(source or "/nonexistent") / "o-voxel" / "o_voxel" / "postprocess.py"
    return path if path.is_file() else None


# --- Stand-ins for to_glb's CUDA pieces ---------------------------------------------------------------------


def barycentric(points, a, b, c):
    v0, v1, v2 = b - a, c - a, points - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    denominator = d00 * d11 - d01 * d01
    wb = (d11 * d20 - d01 * d21) / denominator
    wc = (d00 * d21 - d01 * d20) / denominator
    return torch.stack([1 - wb - wc, wb, wc], -1)


def closest_on_triangles(points, a, b, c):
    """Each point's closest point on each triangle, (P, F, 3), and its barycentric weights, (P, F, 3)."""
    p = points[:, None]
    normal = torch.cross(b - a, c - a, dim=-1)
    foot = p - ((p - a) * normal).sum(-1, keepdim=True) / (normal * normal).sum(-1, keepdim=True) * normal
    weights = barycentric(foot, a, b, c)
    distance = torch.where((weights >= 0).all(-1), (foot - p).norm(dim=-1), torch.full(weights.shape[:-1], math.inf))
    best, best_weights = foot, weights
    for (u, i), (v, j) in ((a, 0), (b, 1)), ((b, 1), (c, 2)), ((c, 2), (a, 0)):
        edge = v - u
        s = (((p - u) * edge).sum(-1, keepdim=True) / (edge * edge).sum(-1, keepdim=True)).clamp(0, 1)
        point = u + s * edge
        on_edge = torch.zeros_like(weights)
        on_edge[..., i], on_edge[..., j] = 1 - s[..., 0], s[..., 0]
        nearer = (point - p).norm(dim=-1) < distance
        distance = torch.where(nearer, (point - p).norm(dim=-1), distance)
        best = torch.where(nearer[..., None], point, best)
        best_weights = torch.where(nearer[..., None], on_edge, best_weights)
    return best, best_weights


# What the stand-ins were asked to do, in order
CALLS: list = []


class FakeCuMesh:
    """CuMesh as to_glb drives it, on a clean, closed mesh: every repair is a no-op; one UV chart per face."""

    def __init__(self) -> None:
        CALLS.append("CuMesh")

    def init(self, vertices, faces) -> None:
        self.vertices, self.faces = vertices.float(), faces.int()

    def read(self):
        return self.vertices, self.faces

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def fill_holes(self, max_hole_perimeter) -> None:
        CALLS.append("fill_holes")

    def simplify(self, target, verbose=False) -> None:
        CALLS.append("simplify")

    def remove_duplicate_faces(self) -> None:
        CALLS.append("remove_duplicate_faces")

    def repair_non_manifold_edges(self) -> None:
        CALLS.append("repair_non_manifold_edges")

    def remove_small_connected_components(self, threshold) -> None:
        CALLS.append("remove_small_connected_components")

    def unify_face_orientations(self) -> None:
        CALLS.append("unify_face_orientations")

    def uv_unwrap(self, compute_charts_kwargs=None, return_vmaps=False, verbose=False):
        """
        Each face its own chart: a corner of its own cell in a square atlas (vertices split per face). The
        corners keep every texel centre off the charts' edges, where float rounding would decide coverage.
        """
        CALLS.append("uv_unwrap")
        faces = self.faces.long()
        count = len(faces)
        cells = math.ceil(math.sqrt(count))
        index = torch.arange(count)
        origin = torch.stack([index % cells, index // cells], -1).float()
        corners = torch.tensor([[0.1, 0.1], [0.85, 0.1], [0.1, 0.85]])
        uvs = ((origin[:, None] + corners[None]) / cells).reshape(-1, 2)
        vmaps = faces.reshape(-1)
        return self.vertices[vmaps], torch.arange(3 * count, dtype=torch.int32).reshape(count, 3), uvs, vmaps

    def compute_vertex_normals(self) -> None:
        v, f = self.vertices, self.faces.long()
        normal = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
        summed = torch.zeros_like(v).index_add_(0, f.reshape(-1), normal.repeat_interleave(3, 0))
        self.normals = summed / summed.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def read_vertex_normals(self):
        return self.normals


class FakeBVH:
    """cuBVH's unsigned_distance, by brute force: the nearest point of every triangle."""

    def __init__(self, vertices, faces) -> None:
        CALLS.append("cuBVH")
        self.vertices, self.faces = vertices.float(), faces.long()

    def unsigned_distance(self, points, return_uvw=False):
        CALLS.append("unsigned_distance")
        a, b, c = (self.vertices[self.faces[:, k]] for k in range(3))
        nearest, weights = closest_on_triangles(points.float(), a, b, c)
        distance = (nearest - points[:, None]).norm(dim=-1)
        face = distance.argmin(dim=1)
        rows = torch.arange(len(points))
        return distance[rows, face], face.int(), weights[rows, face]


def remesh_narrow_band_dc(vertices, faces, center, scale, resolution, band, project_back, verbose, bvh):
    """The same surface, a little bigger: to_glb's BVH projection then pulls every texel back onto it."""
    CALLS.append("remesh")
    return (vertices - center) * INFLATE + center, faces


def grid_sample_3d(feats, coords, shape, grid, mode="trilinear"):
    """FlexGEMM's sampler as a reference: trilinear between voxel centres (i + 0.5), zero outside the voxels."""
    CALLS.append(("grid_sample_3d", feats, grid))
    batch, channels, *size = shape
    dense = torch.zeros(batch, *size, channels, dtype=feats.dtype)
    at = coords.long()
    dense[at[:, 0], at[:, 1], at[:, 2], at[:, 3]] = feats
    q = grid[0] - 0.5
    low = q.floor().long()
    t = q - low
    limit = torch.tensor(size)
    out = torch.zeros(len(q), channels, dtype=feats.dtype)
    for corner in itertools.product((0, 1), repeat=3):
        offset = torch.tensor(corner)
        index = low + offset
        weight = torch.where(offset.bool(), t, 1 - t).prod(-1)
        inside = ((index >= 0) & (index < limit)).all(-1)
        index = index.clamp(min=0).minimum(limit - 1)
        out += (weight * inside)[:, None] * dense[0, index[:, 0], index[:, 1], index[:, 2]]
    return out[None]  # (1, N, C), as upstream's MeshWithVoxel.query_attrs takes it


class FakeCv2(types.ModuleType):
    """OpenCV's inpaint where OpenCV isn't installed: holes take the mean of the rest, plus the radius."""

    INPAINT_TELEA = 1

    @staticmethod
    def inpaint(src, mask, radius, flags):
        out = np.array(src, copy=True)
        if out.ndim == 3 and out.shape[2] == 1:
            out = out[..., 0]  # as OpenCV returns a one-channel image
        hole = mask.astype(bool)
        if hole.any() and (~hole).any():
            out[hole] = np.clip(out[~hole].astype(np.float64).mean(axis=0) + radius, 0, 255).astype(np.uint8)
        return out


class CpuTorch:
    """torch, with to_glb's device='cuda' allocations made on the CPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return torch.zeros(*args, **kwargs)


def module(name, **attributes):
    made = types.ModuleType(name)
    made.__dict__.update(attributes)
    return made


@pytest.fixture
def postprocess(monkeypatch):
    """TRELLIS.2's o_voxel/postprocess.py, loaded with the stand-ins, its tensors on the CPU."""
    path = to_glb_source()
    if path is None:
        pytest.skip("TRELLIS2_SRC: TRELLIS.2's source for to_glb")
    nvdiffrast = module("nvdiffrast", torch=uv_raster)
    nvdiffrast.__path__ = []
    remeshing = module("cumesh.remeshing", remesh_narrow_band_dc=remesh_narrow_band_dc)
    stand_ins = {
        "cumesh": module("cumesh", CuMesh=FakeCuMesh, cuBVH=FakeBVH, remeshing=remeshing),
        "flex_gemm": module("flex_gemm", __path__=[]),
        "flex_gemm.ops": module("flex_gemm.ops", __path__=[]),
        "flex_gemm.ops.grid_sample": module("flex_gemm.ops.grid_sample", grid_sample_3d=grid_sample_3d),
        "nvdiffrast": nvdiffrast,
        "nvdiffrast.torch": uv_raster,  # what uv_raster.install() registers in production
    }
    if importlib.util.find_spec("cv2") is None:
        stand_ins["cv2"] = FakeCv2("cv2")
    for name, stand_in in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, stand_in)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    spec = importlib.util.spec_from_file_location("o_voxel_postprocess_on_the_cpu", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.torch = CpuTorch()
    CALLS.clear()
    return loaded


# --- A shape and its textures ----------------------------------------------------------------------------------


def box_mesh(tint, half=0.3, coords_of=None):
    """
    Upstream's MeshWithVoxel for a closed box, as decode_latent makes it: the shape (vertices, faces, the
    voxels' coords: every voxel of the 16³ grid, a view into the decoder's (L, 4) coords) is the same
    whatever the texture; the attribute volume (base colour, metallic, roughness, alpha) depends on ``tint``.
    """
    corners = torch.tensor(list(itertools.product((-half, half), repeat=3)), dtype=torch.float32)
    faces = torch.tensor(  # wound outwards
        [[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7], [0, 4, 1], [1, 4, 5],
         [2, 3, 6], [3, 7, 6], [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7]],
        dtype=torch.int32,
    )
    grid = torch.tensor(list(itertools.product(range(RESOLUTION), repeat=3)), dtype=torch.int32)
    decoded = torch.cat([torch.zeros_like(grid[:, :1]), grid], dim=1) if coords_of is None else coords_of
    centre = (decoded[:, 1:].float() + 0.5) / RESOLUTION - 0.5
    tint = torch.as_tensor(tint, dtype=torch.float32)
    attrs = torch.cat(
        [
            0.5 + 0.45 * torch.sin(5 * centre + 6 * tint),
            0.2 + 0.5 * tint[:1].expand(len(centre), 1),
            0.5 + 0.4 * centre[:, :1],
            torch.full((len(centre), 1), 0.8),  # the 1024 pass's spurious alpha: unpremultiply has work too
        ],
        dim=1,
    )
    return types.SimpleNamespace(
        vertices=corners,
        faces=faces,
        coords=decoded[:, 1:],
        attrs=attrs,
        layout=dict(LAYOUT),
        voxel_size=1 / RESOLUTION,
    )


def options_for(mesh, preset=FINAL) -> dict:
    """to_glb's options besides the mesh, as Trellis2Runtime._textured passes them."""
    return {
        "attr_layout": mesh.layout,
        "voxel_size": mesh.voxel_size,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": preset.max_faces,
        "texture_size": preset.texture_size,
        "remesh": preset.remesh,
        "remesh_band": 1,
        "remesh_project": 0,
    }


def full(postprocess, mesh, preset=FINAL):
    """to_glb in full, as Trellis2Runtime._textured calls it."""
    options = options_for(mesh, preset)
    tensors = {"vertices": mesh.vertices, "faces": mesh.faces, "attr_volume": mesh.attrs, "coords": mesh.coords}
    return postprocess.to_glb(**tensors, **options)


def assert_same_glb(one, two):
    """Same geometry (values and dtypes), same material, same GLB bytes."""
    for name in ("vertices", "faces", "vertex_normals"):
        a, b = np.asarray(getattr(one, name)), np.asarray(getattr(two, name))
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert np.asarray(one.visual.uv).dtype == np.asarray(two.visual.uv).dtype
    assert np.array_equal(one.visual.uv, two.visual.uv)
    m1, m2 = one.visual.material, two.visual.material
    for name in ("baseColorTexture", "metallicRoughnessTexture"):
        i1, i2 = getattr(m1, name), getattr(m2, name)
        assert i1.mode == i2.mode and np.array_equal(np.asarray(i1), np.asarray(i2)), name
    for name in ("baseColorFactor", "metallicFactor", "roughnessFactor", "alphaMode", "doubleSided"):
        assert np.array_equal(getattr(m1, name), getattr(m2, name)), name
    assert one.export(file_type="glb") == two.export(file_type="glb")


def watched(postprocess, mesh, preset=FINAL):
    with rebake.capturing(postprocess) as watch:
        glb = full(postprocess, mesh, preset)
    return glb, watch


# --- The capture -----------------------------------------------------------------------------------------------


def test_the_capture_sees_what_to_glb_bakes_with_and_changes_nothing(postprocess):
    mesh = box_mesh([0.1, 0.4, 0.7])
    plain = full(postprocess, mesh)
    glb, watch = watched(postprocess, mesh)

    assert_same_glb(glb, plain)  # to_glb's result, bit for bit
    # Put back afterwards: production's helpers again
    assert postprocess.dr is uv_raster and postprocess.grid_sample_3d is grid_sample_3d
    (seen,) = watch.interpolated
    (sampled,) = watch.sampled
    vertices, faces, mask = seen
    assert (vertices, faces) == (len(glb.vertices), len(glb.faces)) == (36, 12)
    # The mask: the texels the unwrapped mesh's UV triangles cover, as to_glb rasterizes them
    unwrapped = FakeCuMesh()
    unwrapped.init(mesh.vertices, mesh.faces)
    _, chart_faces, uv, _ = unwrapped.uv_unwrap()
    clip = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], -1)[None]
    raster, _ = uv_raster.rasterize(None, clip, chart_faces, [64, 64])
    assert torch.equal(mask, raster[0, ..., 3] > 0) and 500 < int(mask.sum()) < 64 * 64
    # ...which are the ones the returned mesh's UVs cover (to_glb flips v on the way out)
    returned = np.asarray(glb.visual.uv)
    np.testing.assert_allclose(returned[:, 0], uv[:, 0].numpy(), atol=1e-6)
    np.testing.assert_allclose(1 - returned[:, 1], uv[:, 1].numpy(), atol=1e-6)
    # The grid: one sample per covered texel, in voxel units, after the BVH projection: on the box's own
    # surface (half 0.3), not on the remeshed one (INFLATE times bigger), where the texels were rasterized
    grid = sampled["grid"]
    assert grid.shape == (1, int(mask.sum()), 3) and sampled["mode"] == "trilinear"
    points = grid[0] / RESOLUTION - 0.5
    torch.testing.assert_close(points.abs().max(dim=1).values, torch.full((len(points),), 0.3), atol=1e-5, rtol=0)
    assert tuple(sampled["shape"]) == (1, 6, RESOLUTION, RESOLUTION, RESOLUTION)


@pytest.mark.parametrize("remesh", [True, False])
def test_a_rebake_is_what_to_glb_makes_for_a_new_texture_of_the_same_shape(postprocess, remesh):
    preset = dataclasses.replace(FINAL, remesh=remesh)
    first, second = box_mesh([0.1, 0.4, 0.7]), box_mesh([0.9, 0.2, 0.5])
    shape = object()
    glb, watch = watched(postprocess, first, preset)
    layout = rebake.keep(watch, glb, first, shape, options_for(first, preset))
    assert rebake.unfit(layout, second, shape, options_for(second, preset)) is None

    expected = full(postprocess, second, preset)
    CALLS.clear()
    rebaked = rebake.rebake(layout, second, postprocess)

    assert_same_glb(rebaked, expected)
    assert rebaked.visual.material.doubleSided is (not remesh)  # to_glb's rule
    # A new texture, not the first one again
    new, old = (np.asarray(made.visual.material.baseColorTexture) for made in (rebaked, glb))
    assert not np.array_equal(new, old)
    # Only the new volume sampled, at the kept texels' positions: none of the geometry work
    (call,) = CALLS
    assert call[0] == "grid_sample_3d" and call[1] is second.attrs and torch.equal(call[2], layout.grid)


def test_the_layout_is_kept_on_the_cpu_and_its_own(postprocess):
    mesh = box_mesh([0.1, 0.4, 0.7])
    glb, watch = watched(postprocess, mesh)
    layout = rebake.keep(watch, glb, mesh, "shape", options_for(mesh))
    assert all(t.device.type == "cpu" for t in (layout.coords, layout.mask, layout.grid))
    # Copies: not views into the decoder's coords, the watched grid or to_glb's mesh
    storage = layout.coords.untyped_storage().data_ptr()
    assert layout.coords.is_contiguous() and storage != mesh.coords.untyped_storage().data_ptr()
    assert layout.grid.data_ptr() != watch.sampled[0]["grid"].data_ptr()
    assert not np.shares_memory(layout.vertices, np.asarray(glb.vertices))
    assert layout.channels == 6 and layout.remesh is True and layout.mode == "trilinear"
    # A rebake's mesh has its own copies too: what the export does to it never reaches the layout
    rebaked = rebake.rebake(layout, mesh, postprocess)
    pairs = ((layout.vertices, rebaked.vertices), (layout.uv, rebaked.visual.uv), (layout.faces, rebaked.faces))
    for kept, used in pairs:
        assert not np.shares_memory(kept, np.asarray(used))


def watch_of(interpolated=1, sampled=1, texels=10, size=64):
    watch = rebake.Watch()
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask.view(-1)[:texels] = True
    watch.interpolated = [(36, 12, mask)] * interpolated
    grid = torch.zeros(1, texels, 3)
    watch.sampled = [{"shape": torch.Size([1, 6, 16, 16, 16]), "grid": grid, "mode": "trilinear"}] * sampled
    return watch


@pytest.mark.parametrize(
    "watch, message",
    [
        (watch_of(interpolated=0), "not seen once: 0 interpolations and 1 samplings"),
        (watch_of(sampled=2), "not seen once: 1 interpolations and 2 samplings"),
        (watch_of(size=32), "the raster is not the texture's 64 x 64 texels"),
    ],
)
def test_nothing_is_kept_from_a_bake_not_seen_exactly_once(watch, message):
    mesh = box_mesh([0.1, 0.4, 0.7])
    glb = types.SimpleNamespace(vertices=np.zeros((36, 3)), faces=np.zeros((12, 3)), vertex_normals=np.zeros((36, 3)))
    with pytest.raises(ValueError, match=message):
        rebake.keep(watch, glb, mesh, "shape", options_for(mesh))


def test_a_raster_the_watch_could_not_read_keeps_nothing():
    watch = watch_of()
    watch.interpolated = [IndexError("too many indices")]
    mesh = box_mesh([0.1, 0.4, 0.7])
    with pytest.raises(ValueError, match="the raster could not be read: IndexError: too many indices"):
        rebake.keep(watch, None, mesh, "shape", options_for(mesh))


def test_a_mesh_that_is_not_the_one_baked_keeps_nothing(postprocess):
    mesh = box_mesh([0.1, 0.4, 0.7])
    glb, watch = watched(postprocess, mesh)
    other = full(postprocess, mesh)
    other.update_faces(np.arange(6))  # what a wrapper changing to_glb's mesh afterwards would hand back
    other.remove_unreferenced_vertices()
    with pytest.raises(ValueError, match="to_glb returned another mesh than the one it baked"):
        rebake.keep(watch, other, mesh, "shape", options_for(mesh))


def test_a_wrapped_postprocess_is_not_watched_and_is_left_as_it_was(postprocess):
    """Like Pixal3D's view-aligned wrapper: its to_glb calls the module's, which the watch can't reach."""

    class Wrapper:
        def __init__(self, inner):
            self._inner = inner

        def to_glb(self, **kwargs):
            return self._inner.to_glb(**kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    wrapper = Wrapper(postprocess)
    mesh = box_mesh([0.1, 0.4, 0.7])
    glb, watch = watched(wrapper, mesh)
    assert watch.interpolated == [] and watch.sampled == []
    assert "dr" not in vars(wrapper) and "grid_sample_3d" not in vars(wrapper)
    assert postprocess.dr is uv_raster and postprocess.grid_sample_3d is grid_sample_3d
    with pytest.raises(ValueError, match="not seen once"):
        rebake.keep(watch, glb, mesh, "shape", options_for(mesh))


def test_a_layout_serves_its_own_shape_with_the_same_options_and_voxels_only():
    mesh = box_mesh([0.1, 0.4, 0.7])
    shape = object()
    size = FINAL.texture_size
    layout = rebake.TextureLayout(
        shape=shape,
        options=rebake.options_key(options_for(mesh)),
        coords=mesh.coords.clone(),
        channels=6,
        mask=torch.zeros(size, size, dtype=torch.bool),
        grid=torch.zeros(1, 0, 3),
        volume_shape=torch.Size([1, 6, 16, 16, 16]),
        mode="trilinear",
        remesh=True,
        vertices=np.zeros((3, 3)),
        faces=np.zeros((1, 3), dtype=np.int64),
        uv=np.zeros((3, 2)),
        normals=np.zeros((3, 3)),
    )
    assert rebake.unfit(layout, box_mesh([0.9, 0.2, 0.5]), shape, options_for(mesh)) is None
    assert rebake.unfit(layout, mesh, object(), options_for(mesh)) == "the layout is another shape's"
    other_options = "the layout was made with other export options"
    bigger, fewer_faces = dataclasses.replace(FINAL, texture_size=128), dataclasses.replace(FINAL, max_faces=5000)
    for preset in (bigger, fewer_faces, PRESETS["preview"]):
        assert rebake.unfit(layout, mesh, shape, options_for(mesh, preset)) == other_options
    for change in ({"voxel_size": 1 / 32}, {"layout": {**LAYOUT, "alpha": slice(5, 7)}}):
        other = types.SimpleNamespace(**{**vars(mesh), **change})
        assert rebake.unfit(layout, other, shape, options_for(other)) == other_options
    moved = box_mesh([0.1, 0.4, 0.7])
    moved.coords = moved.coords.clone()
    moved.coords[7, 2] += 1
    assert rebake.unfit(layout, moved, shape, options_for(moved)) == "the voxels differ"
    fewer = box_mesh([0.1, 0.4, 0.7])
    fewer.coords, fewer.attrs = fewer.coords[:-1], fewer.attrs[:-1]
    assert rebake.unfit(layout, fewer, shape, options_for(fewer)) == "the voxels differ"
    wider = box_mesh([0.1, 0.4, 0.7])
    wider.attrs = torch.cat([wider.attrs, wider.attrs[:, :1]], 1)
    assert rebake.unfit(layout, wider, shape, options_for(wider)) == "the attribute volume has other channels"


def test_options_keys_compare_values():
    mesh = box_mesh([0.1, 0.4, 0.7])
    key = rebake.options_key(options_for(mesh))
    # The same values make the same key, in whatever order and whatever objects hold them
    reordered = dict(reversed(list(LAYOUT.items())))
    assert key == rebake.options_key({**options_for(mesh), "attr_layout": reordered})
    assert key == rebake.options_key(dict(reversed(list(options_for(mesh).items()))))
    assert key == rebake.options_key({**options_for(mesh), "aabb": ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))})
    assert key != rebake.options_key({**options_for(mesh), "remesh": False})
    assert key != rebake.options_key({**options_for(mesh), "attr_layout": {**LAYOUT, "alpha": slice(5, 6, 2)}})
    tensor = rebake.options_key({**options_for(mesh), "voxel_size": torch.tensor([0.0625] * 3)})
    assert tensor == rebake.options_key({**options_for(mesh), "voxel_size": np.array([0.0625] * 3)}) != key
    assert hash(key) == hash(rebake.options_key(options_for(box_mesh([0.9, 0.2, 0.5]))))


# --- The runtime: texture options' exports -----------------------------------------------------------------------


class ShapePipeline:
    """
    Upstream's pipeline as generate() and retexture() drive it, on the CPU: every mesh is the same box (the
    shape), and its attribute volume is tinted by the texture latent, three numbers drawn from torch's
    generator (the texture flow's noise). ``coords_of`` makes the decoder return other voxels.
    """

    tex_slat_sampler_params = {"steps": 12}

    def __init__(self) -> None:
        self.models = {"tex_slat_flow_model_512": object(), "tex_slat_flow_model_1024": object()}
        self.image_cond_model = None
        self.rembg_model = None
        self.low_vram = False
        self.device = torch.device("cpu")
        self.coords_of = None
        self.fail = False

    def cuda(self) -> None:
        pass

    def preprocess_image(self, image):
        return image

    def get_cond(self, images, resolution, include_neg_cond=True):
        return {"cond": torch.zeros(1)}

    def sample_tex_slat(self, cond, flow_model, shape_slat, sampler_params={}):  # noqa: B006 - upstream's
        return torch.rand(3)

    def run(self, image, seed, pipeline_type, preprocess_image, return_latent):
        if self.fail:
            raise ValueError("Invalid pipeline type: 2048")
        torch.manual_seed(seed)
        tex_slat = self.sample_tex_slat(None, None, None)
        shape_slat = torch.zeros(4)
        return self.decode_latent(shape_slat, tex_slat, 1024), (shape_slat, tex_slat, 1024)

    def decode_latent(self, shape_slat, tex_slat, resolution):
        return [box_mesh(tex_slat, coords_of=self.coords_of)]


def runtime_around(postprocess) -> Trellis2Runtime:
    runtime = Trellis2Runtime.__new__(Trellis2Runtime)
    runtime.pipeline = ShapePipeline()
    runtime.low_vram_configured = False
    runtime._o_voxel = types.SimpleNamespace(postprocess=postprocess)
    return runtime


def picture() -> Image.Image:
    return Image.new("RGB", (32, 32), (200, 80, 40))  # no alpha and no remover: no cutout, so no projection


def test_texture_one_exports_in_full_and_keeps_the_layout_the_next_ones_rebake(postprocess, capsys):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    paths, glbs = [], []
    for k in (1, 2, 3):
        mesh = runtime.retexture(seed=7 + 1000 * k)
        assert getattr(mesh, SHAPE) is runtime.last_latent
        glbs.append(runtime.export(mesh, FINAL))
        paths.append(runtime.last_export)
        # The same mesh exported the full way, as before: the same GLB
        runtime.rebake_textures = False
        assert runtime.export(mesh, FINAL) == glbs[-1]
        assert runtime.last_export["path"] == "to_glb" and set(runtime.last_export) == {"path", "seconds"}
        runtime.rebake_textures = True

    assert [p["path"] for p in paths] == ["to_glb", "rebake", "rebake"]
    assert paths[0]["captured"] is True and set(paths[1]) == set(paths[2]) == {"path", "seconds"}
    assert all(isinstance(p["seconds"], float) and p["seconds"] >= 0 for p in paths)
    assert len({glb for glb, _ in glbs}) == 3  # three textures
    assert {triangles for _, triangles in glbs} == {12}
    # UV unwrapping and the BVH ran for the first texture and the full-way exports alone
    assert CALLS.count("uv_unwrap") == 1 + 3 and CALLS.count("unsigned_distance") == 1 + 3
    log = capsys.readouterr().out
    assert '[forge3d] export: {"path": "to_glb", "seconds": ' in log
    assert log.count('[forge3d] export: {"path": "rebake"') == 2


def test_previews_and_finals_export_as_before_unwatched(postprocess, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError("a final's to_glb was watched")

    monkeypatch.setattr(rebake, "capturing", unexpected)
    runtime = runtime_around(postprocess)
    for preset in (FINAL, dataclasses.replace(PRESETS["preview"], texture_size=64)):
        mesh = runtime.generate(picture(), preset, seed=7)
        assert not hasattr(mesh, SHAPE)
        runtime.export(mesh, preset)
        assert set(runtime.last_export) == {"path", "seconds"} and runtime.last_export["path"] == "to_glb"
        assert runtime.texture_layout is None
    assert "[forge3d] export:" not in capsys.readouterr().out


def test_a_new_generation_drops_the_layout_even_when_it_fails(postprocess):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    runtime.export(runtime.retexture(seed=1007), FINAL)
    assert runtime.texture_layout is not None
    runtime.generate(picture(), FINAL, seed=7)  # a new job, even with the same picture and seed
    assert runtime.texture_layout is None
    runtime.export(runtime.retexture(seed=1007), FINAL)
    assert runtime.last_export["path"] == "to_glb" and runtime.texture_layout is not None
    runtime.pipeline.fail = True
    with pytest.raises(ValueError):
        runtime.generate(picture(), FINAL, seed=8)
    assert runtime.texture_layout is None


def test_another_shape_never_reuses_the_layout(postprocess):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    earlier = runtime.last_latent
    runtime.generate(picture(), FINAL, seed=8)
    runtime.export(runtime.retexture(seed=1008), FINAL)
    kept = runtime.texture_layout
    assert kept.shape is runtime.last_latent

    runtime.export(runtime.retexture(seed=1007, latent=earlier), FINAL)

    assert runtime.last_export["path"] == "to_glb" and runtime.last_export["captured"] is True
    assert runtime.last_export["fallback"] == "the layout is another shape's"
    assert runtime.texture_layout is not kept and runtime.texture_layout.shape is earlier


def test_other_voxels_for_the_same_shape_export_in_full_and_keep_their_own_layout(postprocess):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    runtime.export(runtime.retexture(seed=1007), FINAL)
    # The decoder gives the same latent other voxels (one moved): the kept positions are not this volume's
    moved = torch.cat([torch.zeros(RESOLUTION**3, 1, dtype=torch.int32), runtime.texture_layout.coords.clone()], 1)
    moved[100, 3] = (moved[100, 3] + 1) % RESOLUTION
    runtime.pipeline.coords_of = moved
    mesh = runtime.retexture(seed=2007)
    data = runtime.export(mesh, FINAL)

    assert runtime.last_export["path"] == "to_glb" and runtime.last_export["fallback"] == "the voxels differ"
    assert runtime.last_export["captured"] is True and torch.equal(runtime.texture_layout.coords, moved[:, 1:])
    runtime.rebake_textures = False
    assert runtime.export(mesh, FINAL) == data
    runtime.rebake_textures = True
    runtime.export(runtime.retexture(seed=3007), FINAL)
    assert runtime.last_export["path"] == "rebake"


def test_a_failing_rebake_falls_back_to_to_glb_and_keeps_a_new_layout(postprocess, monkeypatch, capsys):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    runtime.export(runtime.retexture(seed=1007), FINAL)
    old = runtime.texture_layout
    mesh = runtime.retexture(seed=2007)
    runtime.rebake_textures = False
    expected = runtime.export(mesh, FINAL)
    runtime.rebake_textures = True

    def broken(layout, mesh, postprocess):
        raise RuntimeError("CUDA error: an illegal memory access\nwas encountered")

    monkeypatch.setattr(rebake, "rebake", broken)
    assert runtime.export(mesh, FINAL) == expected
    assert runtime.last_export["path"] == "to_glb" and runtime.last_export["captured"] is True
    reason = "the rebake failed: RuntimeError: CUDA error: an illegal memory access was encountered"
    assert runtime.last_export["fallback"] == reason
    assert runtime.texture_layout is not None and runtime.texture_layout is not old
    assert '"fallback": "the rebake failed: ' in capsys.readouterr().out


def test_a_layout_that_cannot_be_kept_fails_nothing(postprocess, monkeypatch):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)

    def refuse(watch, glb, mesh, shape, options):
        raise ValueError("to_glb's bake was not seen once:\n2 interpolations")

    monkeypatch.setattr(rebake, "keep", refuse)
    for seed in (1007, 2007):
        data, triangles = runtime.export(runtime.retexture(seed=seed), FINAL)
        assert data[:4] == b"glTF" and triangles == 12
        assert runtime.last_export["path"] == "to_glb" and "fallback" not in runtime.last_export
        assert runtime.last_export["capture_error"] == "ValueError: to_glb's bake was not seen once: 2 interpolations"
        assert runtime.texture_layout is None


def test_with_rebaking_off_every_export_runs_to_glb_and_the_layout_stays_as_it_was(postprocess, monkeypatch):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    runtime.export(runtime.retexture(seed=1007), FINAL)
    kept = runtime.texture_layout
    runtime.rebake_textures = False
    monkeypatch.setattr(rebake, "capturing", None)  # never entered
    runtime.export(runtime.retexture(seed=2007), FINAL)
    assert runtime.last_export["path"] == "to_glb" and runtime.texture_layout is kept


def test_what_an_export_does_to_its_mesh_never_reaches_the_layout(postprocess):
    runtime = runtime_around(postprocess)
    runtime.generate(picture(), FINAL, seed=7)
    runtime.export(runtime.retexture(seed=1007), FINAL)
    layout = runtime.texture_layout
    kept = {name: getattr(layout, name).copy() for name in ("vertices", "faces", "uv", "normals")}

    def vandal(glb, mesh):  # an experiment's hook, writing into to_glb's mesh in place
        glb.vertices[:] += 1.0
        glb.visual.uv[:] = 0.5
        return {}

    runtime.before_projection = vandal
    runtime.export(runtime.retexture(seed=2007), FINAL)
    assert runtime.last_export["path"] == "rebake"
    assert all(np.array_equal(getattr(layout, name), value) for name, value in kept.items())


# --- The textures job ------------------------------------------------------------------------------------------


class Storage:
    def __init__(self) -> None:
        self.saved = {}

    def put(self, key, data, content_type):
        self.saved[key] = data
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def png() -> str:
    buffer = io.BytesIO()
    picture().save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_a_textures_job_says_how_each_texture_was_exported(postprocess, monkeypatch):
    monkeypatch.setitem(service.PRESETS, "final", FINAL)
    runtime = runtime_around(postprocess)
    job = {"id": "t", "input": {"image_base64": png(), "mode": "textures", "seed": 7, "count": 3}}
    out = service.handle_job(job, runtime, Storage(), lambda raw, limit: raw)

    assert "error" not in out and len(out["textures"]) == 3
    exports = [texture["export"] for texture in out["textures"]]
    assert [e["path"] for e in exports] == ["to_glb", "rebake", "rebake"] and exports[0]["captured"] is True
    assert CALLS.count("uv_unwrap") == 1
    # The final of the same picture and seed: exported as ever, and its result says nothing of it
    final = {"id": "f", "input": {"image_base64": png(), "mode": "final", "seed": 7}}
    out = service.handle_job(final, runtime, Storage(), lambda raw, limit: raw)
    assert "error" not in out and "export" not in out and runtime.texture_layout is None


def test_keep_layout_marks_the_last_generations_mesh_with_its_shape():
    runtime = runtime_around(None)  # to_glb isn't needed
    assert runtime.keep_layout(types.SimpleNamespace()) is False  # nothing generated yet: no shape
    mesh = runtime.generate(picture(), FINAL, seed=7)
    assert not hasattr(mesh, SHAPE)  # a final's mesh is never marked by itself
    assert runtime.keep_layout(mesh) is True and getattr(mesh, SHAPE) is runtime.last_latent
    # The shape's retextures carry the same latent, so a layout kept from this mesh serves them
    assert getattr(runtime.retexture(seed=1007), SHAPE) is getattr(mesh, SHAPE)
    assert runtime.keep_layout(("a mesh", "that takes no attributes")) is False


def test_a_judged_textures_job_exports_the_own_texture_in_full_once_and_rebakes_every_new_one(postprocess, monkeypatch):
    from forge3d_worker import judgeviews

    monkeypatch.setitem(service.PRESETS, "final", FINAL)
    runtime = runtime_around(postprocess)
    drawn, requests = [], []

    def draw(raw):  # the real grid, small, of each GLB as the runtime exported it
        drawn.append(raw)
        return judgeviews.from_glb(raw, size=48, supersample=1)

    def judge(request):
        requests.append(request)
        return {"verdicts": ["edits"] * 4, "best": 2, "why": "M has the cleanest back", "seconds": 1.0, "model": "8b"}

    job = {"image_base64": png(), "mode": "textures", "seed": 7, "count": 3, "judge": True, "prompt": "a box"}
    storage = Storage()
    out = service.handle_job({"id": "t", "input": job}, runtime, storage, lambda raw, limit: raw, judge=judge, render=draw)

    assert "error" not in out and "judge_error" not in out and out["judge"]["pick"] == 2
    own = out["own_texture"]["export"]
    assert own["path"] == "to_glb" and own["captured"] is True
    assert [texture["export"]["path"] for texture in out["textures"]] == ["rebake"] * 3
    assert CALLS.count("uv_unwrap") == 1  # to_glb's geometry work ran once, for the final's own texture
    # The judge saw each new texture as it was stored, and the final's own texture as the final job exports it
    assert drawn[1:] == [storage.saved[f"ai/t/final-7-texture-{k}.glb"] for k in (1, 2, 3)]
    final = Storage()
    service.handle_job({"id": "f", "input": {"image_base64": png(), "mode": "final", "seed": 7}}, runtime, final, lambda raw, limit: raw)
    assert drawn[0] == final.saved["ai/f/final-7.glb"]
    grids = [Image.open(io.BytesIO(base64.b64decode(grid))) for grid in requests[0]["candidates_png"]]
    assert [grid.size for grid in grids] == [(144, 96)] * 4
