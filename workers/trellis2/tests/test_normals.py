"""Shading normals on synthetic meshes shaped like the remesher's output, and their way into the GLB."""

import json
import math
import shutil
import struct
import subprocess
import time
import types

import numpy as np
import pytest
from PIL import Image

from forge3d_worker import normals, pipeline
from forge3d_worker.compress import GLTFPACK, gltfpack_args
from forge3d_worker.normals import shading_normals
from forge3d_worker.settings import PRESETS

VOXEL = 1 / 1024  # the final's grid over the unit box


def angles(a, b):
    """Degrees between matching rows of two arrays of directions."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def face_normals(vertices, faces):
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    return cross / np.linalg.norm(cross, axis=1, keepdims=True)


def outward(vertices, faces):
    """Faces wound so their normals point away from the origin."""
    centroids = vertices[faces].mean(axis=1)
    flip = (face_normals(vertices, faces) * centroids).sum(axis=1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def faceted_band(segments=32, rings=16, split=6, radius=0.35, first_ring=3, last_ring=13):
    """
    The middle of a sphere made of flat facets, `segments` around and `rings` from pole to pole, each
    facet cut into split x split quads. 32 x 16 is what the generator made of a potion's bulb: facets
    some 70 voxels wide, 11.25 degrees apart, as wide as the remesher's triangles allow (split 6 gives
    triangles about 12 voxels across, as in the potion's final).
    Returns vertices, faces and each vertex's row (0 and the last are the band's open edges).
    """
    theta = 2 * np.pi * np.arange(segments) / segments
    phi = -np.pi / 2 + np.pi * np.arange(rings + 1) / rings

    def corner(i, j):
        i = i % segments
        x, z = np.cos(phi[j]) * np.cos(theta[i]), np.cos(phi[j]) * np.sin(theta[i])
        return radius * np.stack([x, np.sin(phi[j]), z], -1)

    a = np.arange(segments * split)
    b = np.arange((last_ring - first_ring) * split + 1)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    i, s = aa // split, (aa % split) / split
    j = np.minimum(first_ring + bb // split, last_ring - 1)
    t = (bb - (j - first_ring) * split) / split
    s, t = s[..., None], t[..., None]
    points = (
        (1 - s) * (1 - t) * corner(i, j) + s * (1 - t) * corner(i + 1, j)
        + s * t * corner(i + 1, j + 1) + (1 - s) * t * corner(i, j + 1)
    )
    columns, rows = len(a), len(b)
    index = np.arange(columns * rows).reshape(columns, rows)
    right = np.roll(index, -1, axis=0)
    quads = np.stack([index[:, :-1], right[:, :-1], right[:, 1:], index[:, 1:]], -1).reshape(-1, 4)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    vertices = points.reshape(-1, 3)
    return vertices, outward(vertices, faces), np.broadcast_to(b, (columns, rows)).reshape(-1)


def plain_vertex_normals(vertices, faces):
    """What the remesher shipped: area-weighted averages of the faces around each vertex."""
    return shading_normals(vertices, faces, VOXEL, smooth_voxels=0).normals


@pytest.mark.parametrize(
    "segments, rings, split, first_ring, last_ring, before, after",
    [
        # The potion: facets 70 voxels wide, 11.25 degrees apart, triangles 12 voxels across
        (32, 16, 6, 3, 13, 4.5, 0.5),
        # Terraces at the voxel scale: 17 voxels wide, 2.8 degrees apart, triangles 4 voxels across
        (128, 64, 4, 24, 40, 0.8, 0.1),
    ],
)
def test_a_terraced_sphere_gets_smooth_normals(segments, rings, split, first_ring, last_ring, before, after):
    vertices, faces, _ = faceted_band(segments, rings, split, first_ring=first_ring, last_ring=last_ring)
    smooth = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)  # the sphere the facets approximate
    # Away from the band's open edges, where averaging can only reach one way (64 voxels, twice the reach)
    latitude = np.arcsin(smooth[:, 1])
    lowest, highest = -np.pi / 2 + np.pi * np.array([first_ring, last_ring]) / rings
    inside = 0.35 * np.minimum(latitude - lowest, highest - latitude) > 64 * VOXEL

    result = shading_normals(vertices, faces, VOXEL)

    assert np.array_equal(result.source, np.arange(len(vertices)))  # nothing split: no hard edges here
    assert angles(plain_vertex_normals(vertices, faces)[inside], smooth[inside]).max() > before
    assert angles(result.normals[inside], smooth[inside]).max() < after


def tessellated_cube(n=8, half=0.25, charts=True):
    """A cube whose faces are n x n grids: six UV charts (split vertices along the edges), or one."""
    u, v = np.meshgrid(np.linspace(-half, half, n + 1), np.linspace(-half, half, n + 1), indexing="ij")
    vertices, faces = [], []
    for axis in range(3):
        for sign in (-1, 1):
            p = np.zeros(((n + 1) ** 2, 3))
            p[:, axis] = sign * half
            p[:, (axis + 1) % 3], p[:, (axis + 2) % 3] = u.reshape(-1), v.reshape(-1)
            k = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) + sum(len(x) for x in vertices)
            quads = np.stack([k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]], -1).reshape(-1, 4)
            vertices.append(p)
            faces.append(np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    if not charts:
        vertices, weld = np.unique(vertices, axis=0, return_inverse=True)
        faces = weld.reshape(-1)[faces]
    return vertices, outward(vertices, faces)


@pytest.mark.parametrize("charts", [True, False])
def test_a_cube_keeps_its_90_degree_edges(charts):
    vertices, faces = tessellated_cube(charts=charts)
    result = shading_normals(vertices, faces, VOXEL)

    # Every corner of every face carries exactly that face's normal: flat sides, crisp edges
    corner_normals = result.normals[result.faces]
    expected = np.repeat(face_normals(vertices, faces)[:, None], 3, axis=1)
    np.testing.assert_allclose(corner_normals, expected, atol=1e-6)
    # Copies only where an edge needs them: one per face meeting at an edge or a corner
    assert len(result.source) == 6 * 9 * 9
    np.testing.assert_array_equal(vertices[result.source][result.faces], vertices[faces])  # same triangles


def low_poly_icosphere(subdivisions=1, split=8, radius=0.4):
    """A low-poly sphere (80 flat faces after 1 subdivision), each cut into split² coplanar triangles."""
    g = (1 + 5**0.5) / 2
    corners = np.array(
        [[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g],
         [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1]], dtype=float)
    tris = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
         [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    corners /= np.linalg.norm(corners, axis=1, keepdims=True)
    for _ in range(subdivisions):
        mid = [(corners[a] + corners[b]) / 2 for a, b, c in tris for a, b in ((a, b), (b, c), (c, a))]
        mid = np.array(mid) / np.linalg.norm(mid, axis=1, keepdims=True)
        first = len(corners)
        corners = np.concatenate([corners, mid])
        new = []
        for f, (a, b, c) in enumerate(tris):
            ab, bc, ca = first + 3 * f, first + 3 * f + 1, first + 3 * f + 2
            new += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        tris = np.array(new)
    corners *= radius
    # Cut each flat face into split² triangles without leaving its plane; weld the shared edges
    points, faces, facet = [], [], []
    for f, (a, b, c) in enumerate(tris):
        grid = {}
        for i in range(split + 1):
            for j in range(split + 1 - i):
                grid[i, j] = len(points)
                towards_b = (corners[b] - corners[a]) * i / split
                towards_c = (corners[c] - corners[a]) * j / split
                points.append(corners[a] + towards_b + towards_c)
        for i in range(split):
            for j in range(split - i):
                faces.append([grid[i, j], grid[i + 1, j], grid[i, j + 1]])
                facet.append(f)
                if i + j < split - 1:
                    faces.append([grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
                    facet.append(f)
    points = np.round(np.array(points), 12)
    vertices, weld = np.unique(points, axis=0, return_inverse=True)
    faces = weld.reshape(-1)[np.array(faces)]
    return vertices, outward(vertices, faces), np.array(facet)


def test_a_low_poly_icosphere_keeps_its_facets():
    vertices, faces, facet = low_poly_icosphere()
    flat = face_normals(vertices, faces)
    # The facets meet at about 21 degrees: twice the facet angle, where the filter stops averaging
    result = shading_normals(vertices, faces, VOXEL)

    # A vertex used by one facet only is inside it: its normal is still the facet's own
    facets_of_vertex = {}
    for f, tri in enumerate(faces):
        for v in tri:
            facets_of_vertex.setdefault(v, set()).add(facet[f])
    inner = np.array([v for v, fs in facets_of_vertex.items() if len(fs) == 1])
    owner = np.array([next(iter(facets_of_vertex[v])) for v in inner])
    facet_normal = np.array([flat[facet == f][0] for f in range(facet.max() + 1)])
    assert len(inner) > 0.6 * len(vertices)
    assert angles(result.normals[inner], facet_normal[owner]).max() < 0.25
    # On the facets' edges, normals blend the facets that meet there and nothing else
    edge = np.setdiff1d(np.arange(len(vertices)), inner)
    meeting = [angles(result.normals[v][None], facet_normal[sorted(facets_of_vertex[v])]).max() for v in edge]
    assert max(meeting) < 25


def test_vertices_split_at_uv_seams_get_identical_normals():
    vertices, faces, _ = faceted_band(split=4)
    whole = shading_normals(vertices, faces, VOXEL)
    # Cut the band open along one meridian, as a UV layout does: faces on one side use copies
    seam = np.flatnonzero(np.isclose(vertices[:, 2], 0) & (vertices[:, 0] > 0))
    copy_of = dict(zip(seam, len(vertices) + np.arange(len(seam))))
    split_vertices = np.concatenate([vertices, vertices[seam]])
    side = vertices[faces].mean(axis=1)[:, 2] < 0
    split_faces = faces.copy()
    for f in np.flatnonzero(side):
        split_faces[f] = [copy_of.get(v, v) for v in split_faces[f]]
    assert len(np.unique(split_faces)) == len(vertices) + len(seam)

    result = shading_normals(split_vertices, split_faces, VOXEL)

    copies = len(vertices) + np.arange(len(seam))
    np.testing.assert_array_equal(result.normals[seam], result.normals[copies])
    np.testing.assert_allclose(result.normals[: len(vertices)], whole.normals, atol=1e-12)


def test_two_sheets_back_to_back_each_get_their_own_smooth_side():
    # A curved sheet and its reverse on the same vertices: how the remesher draws a thin part (a fox's ear)
    vertices, faces, _ = faceted_band(segments=64, rings=32, split=2, first_ring=12, last_ring=20)
    front = faces[vertices[faces].mean(axis=1)[:, 0] > 0.2]  # a patch, not the whole band
    used = np.unique(front)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    vertices, front = vertices[used], remap[front]
    back = front[:, [0, 2, 1]]
    alone = shading_normals(vertices, front, VOXEL)

    together = shading_normals(vertices, np.concatenate([front, back]), VOXEL)

    # Each side smooth, as if the other weren't there: averaging them together would cancel them out
    corners = together.normals[together.faces]
    np.testing.assert_allclose(corners[: len(front)], alone.normals[alone.faces], atol=1e-6)
    np.testing.assert_allclose(corners[len(front) :], -alone.normals[alone.faces][:, [0, 2, 1]], atol=1e-6)
    assert len(together.source) == 2 * len(vertices)  # each vertex once per side


def test_empty_and_degenerate_input():
    empty = shading_normals(np.zeros((3, 3)), np.zeros((0, 3), dtype=int), VOXEL)
    assert empty.faces.shape == (0, 3) and np.allclose(np.linalg.norm(empty.normals, axis=1), 1)
    # A triangle with no area next to a proper one: no NaN, unit normals everywhere
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float) * 0.1
    result = shading_normals(vertices, np.array([[0, 1, 2], [0, 1, 3]]), VOXEL)
    assert np.isfinite(result.normals).all()
    assert np.allclose(np.linalg.norm(result.normals, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(result.normals[2], [0, 0, 1], atol=1e-6)


def test_a_final_sized_mesh_takes_about_a_second():
    vertices, faces, _ = faceted_band(segments=128, rings=96, split=3, first_ring=6, last_ring=90)
    assert len(faces) > 90_000
    start = time.perf_counter()
    shading_normals(vertices, faces, VOXEL)
    # Typically about a second on one CPU core; the bound only catches something gone quadratic
    assert time.perf_counter() - start < 10


def read_glb(data: bytes):
    """The JSON and binary chunks of a GLB."""
    length = struct.unpack("<I", data[12:16])[0]
    meta = json.loads(data[20 : 20 + length])
    binary = data[20 + length + 8 :]
    return meta, binary


def accessor(meta, binary, index):
    a = meta["accessors"][index]
    view = meta["bufferViews"][a["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
    offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(binary, dtype, a["count"] * width, offset).reshape(-1, width)


def textured_cube():
    """What o_voxel's to_glb returns: a textured trimesh mesh (here a cube with one UV chart per face)."""
    trimesh = pytest.importorskip("trimesh")
    vertices, faces = tessellated_cube(n=4)
    uv = np.stack([vertices[:, 0] + 0.5, vertices[:, 1] + 0.5], 1)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGBA", (16, 16), (200, 80, 40, 255)),
        metallicRoughnessTexture=Image.new("RGB", (16, 16), (0, 128, 0)),
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    remeshers = np.zeros_like(vertices) + [0, 1, 0]
    return trimesh.Trimesh(vertices, faces, vertex_normals=remeshers, visual=visual, process=False)


def test_export_writes_the_shading_normals_into_the_glb():
    cube = textured_cube()
    calls = []

    def to_glb(**options):
        calls.append(options)
        return cube

    runtime = pipeline.Trellis2Runtime.__new__(pipeline.Trellis2Runtime)
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=to_glb))
    mesh = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None)
    mesh.voxel_size = 1 / 512
    glb, triangles = runtime._export(mesh, PRESETS["final"])

    assert triangles == len(cube.faces) and calls[0]["voxel_size"] == 1 / 512
    meta, binary = read_glb(glb)
    primitive = meta["meshes"][0]["primitives"][0]
    positions = accessor(meta, binary, primitive["attributes"]["POSITION"]).astype(float)
    written = accessor(meta, binary, primitive["attributes"]["NORMAL"]).astype(float)
    indices = accessor(meta, binary, primitive["indices"]).reshape(-1, 3)
    # Positions untouched (the same triangles), and every corner carries its own side's normal
    np.testing.assert_allclose(positions[indices], np.asarray(cube.vertices)[cube.faces], atol=1e-6)
    np.testing.assert_allclose(written[indices], np.repeat(cube.face_normals[:, None], 3, axis=1), atol=1e-6)
    assert "TEXCOORD_0" in primitive["attributes"]
    assert "baseColorTexture" in meta["materials"][0]["pbrMetallicRoughness"]


def test_export_keeps_the_remeshers_normals_if_shading_fails(monkeypatch, capsys):
    cube = textured_cube()

    def broken(mesh, voxel_size):
        raise ValueError("no luck")

    monkeypatch.setattr(normals, "with_shading_normals", broken)
    assert pipeline.shade(cube, 1 / 1024) is cube
    expected = "[forge3d] shading normals failed, keeping the remesher's: ValueError: no luck\n"
    assert capsys.readouterr().out == expected


@pytest.mark.skipif(shutil.which(GLTFPACK) is None, reason="gltfpack not installed")
def test_gltfpack_keeps_the_supplied_normals(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    # Normals that are not the geometry's (tilted 20 degrees): recomputing them would show
    vertices, faces, _ = faceted_band(split=2)
    tilt = math.radians(20)
    radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    supplied = np.stack([radial[:, 0] * math.cos(tilt) - radial[:, 1] * math.sin(tilt),
                         radial[:, 0] * math.sin(tilt) + radial[:, 1] * math.cos(tilt), radial[:, 2]], 1)
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=supplied, process=False)
    source = tmp_path / "in.glb"
    source.write_bytes(mesh.export(file_type="glb"))

    # Unquantised, gltfpack passes them through as they are
    plain = tmp_path / "plain.glb"
    subprocess.run([GLTFPACK, "-i", str(source), "-o", str(plain), "-noq"], check=True, capture_output=True)
    meta, binary = read_glb(plain.read_bytes())
    primitive = meta["meshes"][0]["primitives"][0]
    out_positions = accessor(meta, binary, primitive["attributes"]["POSITION"]).astype(float)
    out_normals = accessor(meta, binary, primitive["attributes"]["NORMAL"]).astype(float)
    match = {tuple(np.round(p, 5)): n for p, n in zip(vertices, supplied)}
    expected = np.array([match[tuple(np.round(p, 5))] for p in out_positions])
    assert angles(out_normals, expected).max() < 0.01

    # With our flags they stay too, as 8-bit octahedral vectors (meshopt's OCTAHEDRAL filter)
    packed = tmp_path / "packed.glb"
    command = [GLTFPACK, "-i", str(source), "-o", str(packed), *gltfpack_args(256)]
    subprocess.run(command, check=True, capture_output=True)
    meta, _ = read_glb(packed.read_bytes())
    primitive = meta["meshes"][0]["primitives"][0]
    normal = meta["accessors"][primitive["attributes"]["NORMAL"]]
    view = meta["bufferViews"][normal["bufferView"]]
    assert (normal["componentType"], normal["normalized"], normal["count"]) == (5120, True, len(vertices))
    assert view["extensions"]["EXT_meshopt_compression"]["filter"] == "OCTAHEDRAL"
