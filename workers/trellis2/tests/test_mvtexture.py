"""
Views baked into the texture (mvtexture.py) on synthetic boxes, CPU only: MV-Adapter's cameras, the frame
from glTF axes to MV-Adapter's world, the control maps, and bakes of views painted per side.
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
import trimesh.visual
from PIL import Image

from forge3d_worker import mvtexture as M
from forge3d_worker import projection as P

CELL = 64  # each side gets a CELL x CELL square of the atlas, its UVs inset by GUTTER texels
GUTTER = 6
INNER = CELL - 2 * GUTTER
# A box's sides as (axis, sign), glTF axes: +Y up, +Z the front
SIDES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
FRONT = SIDES.index((2, +1))
# The camera that sees each side
SEEN_BY = {(0, +1): "right", (0, -1): "left", (1, +1): "top", (1, -1): "bottom", (2, +1): "front", (2, -1): "back"}
COLOURS = {
    (0, +1): (200, 40, 40),
    (0, -1): (40, 170, 60),
    (1, +1): (50, 70, 210),
    (1, -1): (220, 200, 40),
    (2, +1): (240, 120, 200),
    (2, -1): (40, 190, 200),
}
HALF = (0.5, 0.35, 0.25)  # largest 0.5: the frame's scale is 1
OLD = (90, 60, 30)  # an old texture's colour, unlike any side's
# Texels this far in from a side's edge are clear of the bake's fade-out at the silhouette (4 pixels of
# 128-pixel views: under 4 texels) and of the normals' bend round the box's edges
INSET = 5
UPSTREAM_CAMERA = Path(__file__).resolve().parents[2] / "multiview" / "mvadapter" / "utils" / "mesh_utils" / "camera.py"


def side_frame(axis: int, sign: int):
    """Two in-plane axes for a side, ordered so that (a, b, outward normal) is right-handed."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    return (a, b) if sign > 0 else (b, a)


def boxes(specs, n: int = 6, inward: bool = False) -> trimesh.Trimesh:
    """
    One mesh of boxes, each spec (centre, half sizes, {side: colour}). Each side is an n x n grid with its
    own vertices (split at UV seams, like to_glb's) and its own cell of the atlas, side after side, box
    after box. Like to_glb's inpainting, a side's colour runs on into the gutter round its square.
    ``inward`` winds every triangle the other way.
    """
    count = len(specs) * len(SIDES)
    grid = math.ceil(math.sqrt(count))
    size = grid * CELL
    texture = np.full((size, size, 3), 128, np.uint8)
    verts, uvs, faces = [], [], []
    steps = np.linspace(-1, 1, n + 1)
    k = 0
    for centre, half, colours in specs:
        for axis, sign in SIDES:
            a, b = side_frame(axis, sign)
            c0, r0 = (k % grid) * CELL + GUTTER, (k // grid) * CELL + GUTTER
            texture[r0 - GUTTER : r0 - GUTTER + CELL, c0 - GUTTER : c0 - GUTTER + CELL] = colours[(axis, sign)]
            base = len(verts)
            for sb in steps:
                for sa in steps:
                    p = np.array(centre, dtype=np.float64)
                    p[axis] += sign * half[axis]
                    p[a] += sa * half[a]
                    p[b] += sb * half[b]
                    verts.append(p)
                    # trimesh UVs: v up, image row 0 at v = 1; sb = +1 is the cell's top row
                    uvs.append(((c0 + (sa + 1) / 2 * INNER) / size, 1 - (r0 + (1 - sb) / 2 * INNER) / size))
            for j in range(n):
                for i in range(n):
                    q = base + j * (n + 1) + i
                    for tri in ([q, q + 1, q + n + 2], [q, q + n + 2, q + n + 1]):
                        faces.append(tri[::-1] if inward else tri)
            k += 1
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture), metallicFactor=0.0)
    return trimesh.Trimesh(
        vertices=np.array(verts),
        faces=np.array(faces),
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs), material=material),
    )


def box(colours=None, **kwargs) -> trimesh.Trimesh:
    colours = colours or {side: OLD for side in SIDES}
    return boxes([((0.0, 0.0, 0.0), HALF, colours)], **kwargs)


def uniform(colour):
    return {side: colour for side in SIDES}


def cell(texture: np.ndarray, k: int, inset: int = 0, grid: int = 3) -> np.ndarray:
    """
    The texels of side k's UV square (its covered texels), less ``inset`` round the edge. A box's atlas is
    3 x 3 cells; three boxes' 5 x 5.
    """
    c0, r0 = (k % grid) * CELL + GUTTER, (k // grid) * CELL + GUTTER
    return texture[r0 + inset : r0 + INNER - inset, c0 + inset : c0 + INNER - inset].astype(int)


def texture_of(mesh) -> np.ndarray:
    return np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))


def object_mask(view: Image.Image) -> np.ndarray:
    """Where a view isn't its grey background."""
    return np.abs(np.asarray(view, dtype=int) - 128).max(-1) > 4


# --- Cameras ------------------------------------------------------------------------------------------


def upstream_cameras():
    if not UPSTREAM_CAMERA.exists():
        pytest.skip("the multiview worker's vendored MV-Adapter camera code isn't here")
    spec = importlib.util.spec_from_file_location("mvadapter_camera", UPSTREAM_CAMERA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # As upstream's scripts/inference_ig2mv_sdxl.py calls it
    return module.get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )


def test_the_cameras_are_upstreams():
    cameras = upstream_cameras()
    axes = M.camera_axes()
    np.testing.assert_allclose(axes.transpose(0, 2, 1), cameras.c2w[:, :3, :3].numpy(), atol=1e-5)
    # Pixels as nvdiffrast rasterises upstream's clip coordinates: its rows run bottom-up in clip space and
    # upstream's projection flips y, so row 0 is the top of the image; depth is distance along the view
    points = torch.rand(64, 3, generator=torch.Generator().manual_seed(0)) - 0.5
    homogeneous = torch.cat([points, torch.ones(64, 1)], -1)
    for i, view_axes in enumerate(torch.tensor(axes, dtype=torch.float32)):
        clip = homogeneous @ cameras.mvp_mtx[i].T
        ndc = clip[:, :2] / clip[:, 3:]
        xy, depth = M._to_view(points, view_axes, 768, 768)
        torch.testing.assert_close(xy, (ndc + 1) / 2 * 768, atol=2e-3, rtol=0)
        torch.testing.assert_close(depth, -(homogeneous @ cameras.w2c[i].T)[:, 2], atol=1e-5, rtol=0)


def test_each_camera_sees_its_side_upright():
    mesh = box(COLOURS)
    # The front's cell: its top half (the cell's top rows: the side's top, +Y) white, its bottom black
    texture = texture_of(mesh).copy()
    c0, r0 = (FRONT % 3) * CELL + GUTTER, (FRONT // 3) * CELL + GUTTER
    texture[r0 - GUTTER : r0 + INNER // 2, c0 - GUTTER : c0 + INNER + GUTTER] = 255
    texture[r0 + INNER // 2 : r0 + INNER + GUTTER, c0 - GUTTER : c0 + INNER + GUTTER] = 0
    mesh.visual.material.baseColorTexture = Image.fromarray(texture)
    frame = M.frame_for(mesh)
    views = [np.asarray(v) for v in M.render_views(mesh, frame, size=128, device="cpu")]
    for side, name in SEEN_BY.items():
        if name != "front":
            assert tuple(views[M.NAMES.index(name)][64, 64]) == COLOURS[side], name
    front = views[M.NAMES.index("front")]
    assert tuple(front[50, 64]) == (255, 255, 255) and tuple(front[78, 64]) == (0, 0, 0)  # up is up
    assert tuple(front[2, 2]) == (128, 128, 128)  # the views' grey round the object

    # Which way each image runs, in glTF terms: from the centre pixel, the one to its left and the one above
    maps = M.control_maps(mesh, frame, size=128, device="cpu")
    world = (maps.position - 0.5) / frame.scale
    gltf = np.stack([world[..., 0], world[..., 2], -world[..., 1]], -1)  # x, y (up), z (front)
    expected = {  # (axis, sign) of the step left, of the step up
        "front": ((0, -1), (1, +1)),
        "right": ((2, +1), (1, +1)),  # the front faces image-left
        "back": ((0, +1), (1, +1)),
        "left": ((2, -1), (1, +1)),
        "top": ((0, +1), (2, +1)),  # the front at the top, the mesh's -X on the right
        "bottom": ((0, +1), (2, -1)),  # the back at the top
    }
    for i, name in enumerate(M.NAMES):
        centre = gltf[i, 64, 64]
        for (axis, sign), step in zip(expected[name], (gltf[i, 64, 54] - centre, gltf[i, 54, 64] - centre)):
            assert sign * step[axis] > 0.05, name
            assert np.abs(np.delete(step, axis)).max() < 1e-4, name  # and only along that axis


# --- Frame --------------------------------------------------------------------------------------------


def test_the_frame_turns_gltf_axes_into_mvadapters_world():
    mesh = SimpleNamespace(vertices=np.array([[0.4, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.2], [0.1, -0.2, -0.3]]))
    frame = M.frame_for(mesh)
    world = frame.points(mesh.vertices)
    assert frame.scale == pytest.approx(0.5) and np.abs(world).max() == pytest.approx(0.5)
    np.testing.assert_allclose(world[0], [0.2, 0.0, 0.0], atol=1e-12)  # +X stays +X
    np.testing.assert_allclose(world[1], [0.0, 0.0, 0.5], atol=1e-12)  # up (+Y) becomes +Z
    np.testing.assert_allclose(world[2], [0.0, -0.1, 0.0], atol=1e-12)  # the front (+Z) faces the front camera
    np.testing.assert_allclose(M.camera_axes()[0, 2], [0.0, -1.0, 0.0], atol=1e-12)  # which sits on -Y
    # Scaled about the origin, not re-centred
    shifted = SimpleNamespace(vertices=mesh.vertices + [0.0, 0.0, 1.0])
    moved = M.frame_for(shifted).points(shifted.vertices)
    assert np.abs(moved).max() == pytest.approx(0.5) and moved.mean(0)[1] < -0.2


@pytest.mark.parametrize("azimuth", [0.0, 30.0, 90.0, 200.0, 338.8])
def test_the_frame_turns_the_pictures_camera_to_the_front(azimuth):
    mesh = SimpleNamespace(vertices=np.random.default_rng(1).uniform(-0.4, 0.4, (100, 3)))
    frame = M.frame_for(mesh, azimuth)
    # The projection's camera at that azimuth (level) looks along the front camera's view
    params = torch.tensor([[azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    towards_camera = P.view_axes(params)[2][0].double().numpy()
    np.testing.assert_allclose(frame.directions(towards_camera), M.camera_axes()[0, 2], atol=1e-6)
    np.testing.assert_allclose(frame.directions([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)  # up stays up
    assert np.abs(frame.points(mesh.vertices)).max() == pytest.approx(0.5)
    assert frame.azimuth == pytest.approx(azimuth % 360)


def test_the_frame_needs_vertices():
    with pytest.raises(ValueError):
        M.frame_for(SimpleNamespace(vertices=np.zeros((0, 3))))
    with pytest.raises(ValueError):
        M.frame_for(SimpleNamespace(vertices=np.zeros((4, 3))))


# --- Control maps -------------------------------------------------------------------------------------


def test_the_control_maps_of_a_box():
    mesh = box(COLOURS)
    frame = M.frame_for(mesh)
    assert frame.scale == pytest.approx(1.0)
    maps = M.control_maps(mesh, frame, size=128, device="cpu")
    assert maps.position.shape == maps.normal.shape == (6, 128, 128, 3) and maps.position.dtype == np.float32
    assert maps.mask.shape == maps.depth.shape == (6, 128, 128) and not maps.flipped

    # The front view's centre pixel: the front side (glTF z = 0.25, world y = -0.25), square to the camera
    x, z = (64.5 / 64 - 1) * 0.55, (1 - 64.5 / 64) * 0.55  # its centre on the image plane
    np.testing.assert_allclose(maps.position[0, 64, 64], [x + 0.5, 0.25, z + 0.5], atol=1e-5)
    np.testing.assert_allclose(maps.normal[0, 64, 64], [0.5, 0.0, 0.5], atol=1e-5)  # world -Y
    assert maps.depth[0, 64, 64] == pytest.approx(1.55)  # the camera is 1.8 from the origin
    # The right side (world +X, 0.5 from the centre) and the top (world +Z, 0.35)
    np.testing.assert_allclose(maps.position[1, 64, 64, 0], 1.0)
    np.testing.assert_allclose(maps.normal[1, 64, 64], [1.0, 0.5, 0.5], atol=1e-5)
    np.testing.assert_allclose(maps.normal[4, 64, 64], [0.5, 0.5, 1.0], atol=1e-5)
    assert maps.depth[4, 64, 64] == pytest.approx(1.45)
    # Off the mesh: 0.5 in both maps, as upstream's control image
    assert not maps.mask[0, 2, 2] and np.isinf(maps.depth[0, 2, 2])
    assert (maps.position[~maps.mask] == 0.5).all() and (maps.normal[~maps.mask] == 0.5).all()
    # The front covers x in [-0.5, 0.5] and up in [-0.35, 0.35] of the [-0.55, 0.55] frame: pixel centres
    # 6..121 across and 23..104 down
    rows, cols = np.nonzero(maps.mask[0])
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (23, 104, 6, 121)
    assert maps.mask[0].sum() == 82 * 116

    control = M.as_control(maps)
    assert control.shape == (6, 6, 128, 128) and control.dtype == np.float32
    np.testing.assert_array_equal(control[:, :3], maps.position.transpose(0, 3, 1, 2))
    np.testing.assert_array_equal(control[:, 3:], maps.normal.transpose(0, 3, 1, 2))
    positions, normals = M.control_images(maps)
    assert len(positions) == len(normals) == 6 and positions[0].size == (128, 128)
    assert positions[0].getpixel((2, 2)) == (127, 127, 127)  # upstream truncates 0.5 * 255


def test_an_inside_out_mesh_still_gets_outward_normals():
    mesh = box(COLOURS, inward=True)
    maps = M.control_maps(mesh, M.frame_for(mesh), size=64, device="cpu")
    assert maps.flipped
    np.testing.assert_allclose(maps.normal[0, 32, 32], [0.5, 0.0, 0.5], atol=1e-5)


# --- Baking -------------------------------------------------------------------------------------------


def test_the_bake_paints_each_side_with_its_view():
    mesh = box()
    frame = M.frame_for(mesh)
    views = M.render_views(box(COLOURS), frame, size=128, device="cpu")
    summary = M.bake_views(mesh, frame, views, match_colour=False, device="cpu")
    texture = texture_of(mesh)
    for k, side in enumerate(SIDES):
        assert np.abs(cell(texture, k, inset=INSET) - COLOURS[side]).max() <= 2, side
    assert summary["texels"] == 6 * INNER * INNER
    assert summary["changed"] >= 0.8 * summary["texels"]  # less a texel or two round each side (fade-out)
    for view in summary["views"]:
        assert view["texels"] > 0.12 * summary["texels"] and view["gain"] is None and view["background"] == 0
    assert summary["seconds"] > 0 and summary["frame"] == {"azimuth": 0.0, "scale": 1.0}


def test_up_in_a_view_is_up_on_the_mesh():
    # The front view's top half white and bottom half black must land on the front side's top and bottom
    target = box(COLOURS)
    frame = M.frame_for(target)
    views = [np.asarray(v).copy() for v in M.render_views(target, frame, size=128, device="cpu")]
    front = views[M.NAMES.index("front")]
    inside = object_mask(Image.fromarray(front))
    front[:64][inside[:64]] = 255
    front[64:][inside[64:]] = 0
    mesh = box()
    M.bake_views(mesh, frame, [Image.fromarray(v) for v in views], match_colour=False, device="cpu")
    side = cell(texture_of(mesh), FRONT, inset=INSET)
    half = INNER // 2 - INSET
    assert (side[: half - 2] == 255).all() and (side[half + 2 :] < 5).all()


def test_hidden_surfaces_keep_their_colour():
    big = ((0.0, 0.0, 0.0), (0.4, 0.4, 0.2), COLOURS)
    inside = ((0.0, 0.0, 0.0), (0.1, 0.1, 0.05), uniform((10, 20, 30)))  # inside the big box
    behind = ((0.0, 0.0, -0.45), (0.15, 0.15, 0.1), uniform((30, 20, 10)))  # behind it, its front facing it
    mesh = boxes([big, inside, behind])
    painted = boxes([big, ((0.0, 0.0, 0.0), (0.1, 0.1, 0.05), uniform((250, 250, 250))),
                     ((0.0, 0.0, -0.45), (0.15, 0.15, 0.1), uniform((250, 250, 250)))])
    frame = M.frame_for(mesh)
    views = M.render_views(painted, frame, size=256, device="cpu")
    # The front view shows the big box where the box behind it is
    assert tuple(np.asarray(views[0])[128, 128]) == COLOURS[(2, +1)]
    old = texture_of(mesh).copy()
    M.bake_views(mesh, frame, views, match_colour=False, device="cpu")
    new = texture_of(mesh)
    for k in range(6, 12):  # the inner box: never seen
        np.testing.assert_array_equal(cell(new, k, grid=5), cell(old, k, grid=5))
    behind_front = 12 + FRONT  # the box behind: its front only faces the big box's back
    np.testing.assert_array_equal(cell(new, behind_front, grid=5), cell(old, behind_front, grid=5))
    behind_back = 12 + SIDES.index((2, -1))  # its back is in plain sight of the back view
    assert np.abs(cell(new, behind_back, inset=INSET, grid=5) - 250).max() <= 2


def shrunk(view: Image.Image, pixels: int) -> Image.Image:
    """The view with its object drawn ``pixels`` smaller all round: the views' grey takes the rim."""
    rgb = np.asarray(view).copy()
    mask = torch.tensor(object_mask(view), dtype=torch.float32)[None, None]
    core = (-torch.nn.functional.max_pool2d(-mask, 2 * pixels + 1, 1, pixels))[0, 0].numpy() > 0.5
    rgb[object_mask(view) & ~core] = 128
    return Image.fromarray(rgb)


def test_the_views_grey_background_is_never_baked():
    frame = M.frame_for(box())
    views = [shrunk(v, 5) for v in M.render_views(box(uniform((220, 30, 30))), frame, size=256, device="cpu")]
    old = uniform((30, 30, 220))
    mesh = box(old)
    summary = M.bake_views(mesh, frame, views, match_colour=False, device="cpu")
    texture = texture_of(mesh)
    sides = np.concatenate([cell(texture, k).reshape(-1, 3) for k in range(6)])
    # Red, blue and anything between have green at 30; the views' grey (128) would raise it
    assert sides[:, 1].max() <= 31
    assert (sides[:, 0] > 200).mean() > 0.6  # and the red got on
    assert all(view["background"] > 0 for view in summary["views"])
    # Without looking for the background, the rim's grey gets baked
    leaky = box(old)
    M.bake_views(leaky, frame, views, match_colour=False, background_band=0, device="cpu")
    sides = np.concatenate([cell(texture_of(leaky), k).reshape(-1, 3) for k in range(6)])
    assert sides[:, 1].max() > 45


def test_weights_fade_out_near_the_silhouette():
    frame = M.frame_for(box())
    views = M.render_views(box(uniform((220, 30, 30))), frame, size=128, device="cpu")

    def front_amounts(feather: float):
        debug = {}
        # A fine grid: the normals bend round the box's edges only within a few pixels of them
        M.bake_views(box(n=32), frame, views, feather=feather, match_colour=False, device="cpu", debug=debug)
        points, amount = debug["points"], debug["amount"]
        on_front = (points[:, 1] + 0.25).abs() < 1e-4  # world y = -0.25: the front side
        # Pixels from the edge of the front's silhouette (x in [-0.5, 0.5], up in [-0.35, 0.35])
        edge = torch.minimum(0.5 - points[:, 0].abs(), 0.35 - points[:, 2].abs()) * (128 / 1.1)
        return edge[on_front], amount[on_front]

    edge, amount = front_amounts(10.0)
    assert amount[edge < 1.5].max() < 0.5  # at the silhouette: mostly the old colour
    assert amount[edge > 11].min() > 0.999  # past the feather: the view's colour
    middle = (edge > 3) & (edge < 8)
    assert 0.3 < float(amount[middle].mean()) < 1.0
    # Only the feather does that: the side views see the front edge-on and add nothing
    _, amount = front_amounts(0.0)
    assert amount.min() > 0.999


def test_depth_edges_fade_too():
    zbuf = torch.full((32, 32), float("inf"))
    zbuf[4:28, 4:28] = 1.0
    zbuf[4:28, 16:28] = 1.5  # a step down
    fade = M._fade(zbuf, torch.isfinite(zbuf), feather=4, jump=0.1)
    assert fade[16, 4] == 0 and fade[16, 15] == 0 and fade[16, 16] == 0  # the outline and both sides of the step
    assert fade[16, 10] == 1 and fade[16, 22] == 1  # 4 pixels clear of both
    assert 0 < fade[16, 13] < 1
    assert torch.equal(M._fade(zbuf, torch.isfinite(zbuf), feather=0, jump=0.1), torch.isfinite(zbuf).float())


def darker(view: Image.Image, gain: float) -> Image.Image:
    """The object in the view in different light: its linear colour times ``gain``; the grey stays."""
    rgb = torch.tensor(np.asarray(view), dtype=torch.float32) / 255
    lit = P._linear_to_srgb(P._srgb_to_linear(rgb) * gain)
    mask = torch.tensor(object_mask(view))[..., None]
    return Image.fromarray((torch.where(mask, lit, rgb).numpy() * 255 + 0.5).astype(np.uint8))


def test_colour_match_brings_each_view_to_the_old_texture():
    frame = M.frame_for(box())
    views = [darker(v, 0.6) for v in M.render_views(box(COLOURS), frame, size=128, device="cpu")]
    mesh = box(COLOURS)  # the old texture had the colours right; the views are darker
    summary = M.bake_views(mesh, frame, views, device="cpu")
    for k, side in enumerate(SIDES):
        assert np.abs(cell(texture_of(mesh), k, inset=INSET) - COLOURS[side]).max() <= 3, side
    for view in summary["views"]:
        np.testing.assert_allclose(view["gain"], [1 / 0.6] * 3, rtol=0.03)
    # Off, the views' darker colours go on as drawn
    plain = box(COLOURS)
    M.bake_views(plain, frame, views, match_colour=False, device="cpu")
    side = SIDES.index((0, +1))
    dark = np.asarray(darker(Image.new("RGB", (1, 1), COLOURS[(0, +1)]), 0.6)).reshape(3)
    assert np.abs(cell(texture_of(plain), side, inset=INSET) - dark).max() <= 3
    # Views whose colours agree nowhere with the old texture's (light grey here) aren't matched
    other = box(uniform((200, 200, 200)))
    summary = M.bake_views(other, frame, views, device="cpu")
    assert all(view["gain"] is None for view in summary["views"])


def test_the_front_view_can_be_left_to_the_pictures_projection():
    frame = M.frame_for(box())
    views = M.render_views(box(COLOURS), frame, size=128, device="cpu")
    mesh = box()
    summary = M.bake_views(mesh, frame, views, front_weight=0.0, match_colour=False, device="cpu")
    texture = texture_of(mesh)
    assert (cell(texture, FRONT) == OLD).all()
    assert np.abs(cell(texture, SIDES.index((0, +1)), inset=INSET) - COLOURS[(0, +1)]).max() <= 2
    assert summary["views"][0]["texels"] == 0
    # Per-view weights work the same way
    mesh = box()
    M.bake_views(mesh, frame, views, view_weights=[1, 1, 0, 1, 1, 1], match_colour=False, device="cpu")
    assert (cell(texture_of(mesh), SIDES.index((2, -1))) == OLD).all()


def test_the_change_runs_on_into_the_gutters():
    frame = M.frame_for(box())
    mesh = box(n=24)
    # Large views and no fade-out at the silhouette: a side's edge texels change as much as its middle
    views = M.render_views(box(COLOURS), frame, size=512, device="cpu")
    M.bake_views(mesh, frame, views, feather=0, match_colour=False, device="cpu")
    texture = texture_of(mesh).astype(int)
    k = SIDES.index((0, +1))
    c0, r0 = (k % 3) * CELL + GUTTER, (k // 3) * CELL + GUTTER
    # The two texels round the square (what bilinear filtering and the first mip level read; they were the
    # old colour), away from its corners, where the normals' bend leaves a little of the old colour
    ring = np.zeros(texture.shape[:2], bool)
    ring[r0 - 2 : r0 + INNER + 2, c0 + 3 : c0 + INNER - 3] = True  # above and below the square
    ring[r0 + 3 : r0 + INNER - 3, c0 - 2 : c0 + INNER + 2] = True  # left and right of it
    ring[r0 : r0 + INNER, c0 : c0 + INNER] = False
    assert np.abs(texture[ring] - COLOURS[(0, +1)]).max() <= 4


def test_a_texture_baked_from_its_own_renders_barely_changes():
    # Each side a smooth gradient, so resampling barely shows: what is left is the bake's own error
    mesh = box(COLOURS)
    texture = texture_of(mesh).astype(np.float64)
    rows, cols = np.mgrid[0 : texture.shape[0], 0 : texture.shape[1]]
    texture = np.clip(texture * (0.6 + 0.4 * rows / texture.shape[0])[..., None] + 40 * np.sin(cols / 20)[..., None], 0, 255)
    mesh.visual.material.baseColorTexture = Image.fromarray(texture.astype(np.uint8))
    frame = M.frame_for(mesh, 25.0)  # turned, so no view is square to a side
    before = texture_of(mesh).astype(int)
    views = M.render_views(mesh, frame, size=256, device="cpu")
    debug = {}
    summary = M.bake_views(mesh, frame, views, device="cpu", debug=debug)
    after = texture_of(mesh).astype(int)
    baked = np.zeros(before.shape[:2], bool).reshape(-1)
    baked[debug["flat"][debug["amount"] > 0.5].numpy()] = True
    baked = baked.reshape(before.shape[:2])
    assert baked.sum() > 0.9 * summary["texels"]
    error = np.abs(after - before)[baked]
    assert error.mean() < 1.0 and np.percentile(error, 99) <= 4
    for view in summary["views"]:
        np.testing.assert_allclose(view["gain"], [1, 1, 1], atol=0.02)


def test_the_textures_alpha_is_kept():
    mesh = box()
    rgba = Image.fromarray(texture_of(mesh)).convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 77))
    mesh.visual.material.baseColorTexture = rgba
    frame = M.frame_for(mesh)
    M.bake_views(mesh, frame, M.render_views(box(COLOURS), frame, size=64, device="cpu"), device="cpu")
    image = mesh.visual.material.baseColorTexture
    assert image.mode == "RGBA" and (np.asarray(image)[..., 3] == 77).all()


def test_bad_input_is_refused():
    mesh = box()
    frame = M.frame_for(mesh)
    views = M.render_views(mesh, frame, size=32, device="cpu")
    with pytest.raises(ValueError, match="expected 6 views"):
        M.bake_views(mesh, frame, views[:5], device="cpu")
    with pytest.raises(ValueError, match="view weights"):
        M.bake_views(mesh, frame, views, view_weights=[1, 1], device="cpu")
    bare = trimesh.creation.box()
    with pytest.raises(ValueError, match="texture"):
        M.bake_views(bare, M.frame_for(bare), views, device="cpu")
