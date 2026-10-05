"""
The multi-view painter (paint.py) on synthetic meshes, CPU only: cameras and framing, renders, the painted
object's outline and its alignment, colour gains, baking views into the texture, the whole loop with a fake
editing model, and moving a mesh between containers.
"""

import io
import math

import numpy as np
import pytest
import torch
import trimesh
import trimesh.visual
from PIL import Image

from forge3d_worker import paint as P
from forge3d_worker import projection

CELL = 48  # each side's square in the atlas, its UVs inset by GUTTER texels
GUTTER = 6
INNER = CELL - 2 * GUTTER
SIDES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]  # (axis, sign), glTF axes: +Y up, +Z front
OLD = (90, 60, 30)
HALF = (0.5, 0.35, 0.25)


def side_frame(axis: int, sign: int):
    a, b = (axis + 1) % 3, (axis + 2) % 3
    return (a, b) if sign > 0 else (b, a)


def box(colours=None, half=HALF, n: int = 4, inward: bool = False) -> trimesh.Trimesh:
    """A box, each side an n x n grid with its own vertices and its own cell of a 3 x 3 atlas (gutters filled)."""
    colours = colours or {side: OLD for side in SIDES}
    size = 3 * CELL
    texture = np.full((size, size, 3), 128, np.uint8)
    verts, uvs, faces = [], [], []
    steps = np.linspace(-1, 1, n + 1)
    for k, (axis, sign) in enumerate(SIDES):
        a, b = side_frame(axis, sign)
        c0, r0 = (k % 3) * CELL + GUTTER, (k // 3) * CELL + GUTTER
        texture[r0 - GUTTER : r0 - GUTTER + CELL, c0 - GUTTER : c0 - GUTTER + CELL] = colours[(axis, sign)]
        base = len(verts)
        for sb in steps:
            for sa in steps:
                p = np.zeros(3)
                p[axis] = sign * half[axis]
                p[a] = sa * half[a]
                p[b] = sb * half[b]
                verts.append(p)
                uvs.append(((c0 + (sa + 1) / 2 * INNER) / size, 1 - (r0 + (1 - sb) / 2 * INNER) / size))
        for j in range(n):
            for i in range(n):
                q = base + j * (n + 1) + i
                for tri in ([q, q + 1, q + n + 2], [q, q + n + 2, q + n + 1]):
                    faces.append(tri[::-1] if inward else tri)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture), metallicFactor=0.0)
    return trimesh.Trimesh(
        vertices=np.array(verts),
        faces=np.array(faces),
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs), material=material),
    )


def cell(texture: np.ndarray, k: int, inset: int = 0) -> np.ndarray:
    c0, r0 = (k % 3) * CELL + GUTTER, (k // 3) * CELL + GUTTER
    return texture[r0 + inset : r0 + INNER - inset, c0 + inset : c0 + INNER - inset].astype(int)


def small(**overrides):
    """Camera options for quick CPU runs."""
    return {"pixels": 96 * 96, **overrides}


# --- Cameras ------------------------------------------------------------------------------------------


def test_frame_fits_the_object_in_a_window_of_square_pixels():
    geom = P.geometry(box(), "cpu")
    for azimuth, elevation in ((0, 0), (90, 15), (200, 40), (0, 89), (30, -89)):
        camera = P.frame(geom.verts, "v", azimuth, elevation, pixels=200 * 120)
        width, height = camera.size
        assert width % 16 == 0 and height % 16 == 0 and width * height <= 200 * 120
        # Square pixels: the window has the image's aspect
        assert camera.half[0] / camera.half[1] == pytest.approx(width / height, rel=1e-6)
        xy, depth, s = camera.project(geom.verts)
        margin = P.MARGIN * 0.5 * min(width, height)  # at least half the margin on the tighter side
        assert float(xy[:, 0].min()) > margin * 0.5 and float(xy[:, 0].max()) < width - margin * 0.5
        assert float(xy[:, 1].min()) > margin * 0.5 and float(xy[:, 1].max()) < height - margin * 0.5
        assert bool((s > 0).all())  # everything in front of the camera


def test_frame_keeps_its_aspect_within_bounds():
    # A long thin rod seen side on
    rod = box(half=(1.0, 0.02, 0.02))
    camera = P.frame(P.geometry(rod, "cpu").verts, "side", 0, 0, max_aspect=2.5)
    assert camera.width / camera.height <= 2.5 + 0.2


def test_the_camera_looks_where_projection_says():
    geom = P.geometry(box(), "cpu")
    camera = P.frame(geom.verts, "v", 0, 0, perspective=0.0)
    # Azimuth 0 looks at the +Z side: a point on it is nearer than one on -Z, and +X is to the right
    points = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    xy, depth, _ = camera.project(points)
    assert float(depth[0]) < float(depth[1])
    assert float(xy[2, 0]) > float(xy[0, 0])  # +X right
    assert float(xy[3, 1]) < float(xy[0, 1])  # +Y up (rows run down)
    np.testing.assert_allclose(camera.direction(), [0, 0, 1], atol=1e-9)
    # Twice the size: twice the pixel coordinates
    xy2, _, _ = camera.project(points, scale=2)
    torch.testing.assert_close(xy2, xy * 2)


def test_ring_starts_at_the_picture_and_orders_by_angle():
    geom = P.geometry(box(), "cpu")
    cameras = P.ring(geom.verts, 30.0, around=8, **small())
    assert [c.name for c in cameras] == ["a000", "a045", "a090", "a135", "a180", "a225", "a270", "a315", "top", "bottom"]
    assert cameras[0].azimuth == pytest.approx(30.0) and cameras[2].azimuth == pytest.approx(120.0)
    ordered = P.by_angle(cameras, 30.0, P.ELEVATION)
    assert ordered[0].name == "a000"
    assert {ordered[1].name, ordered[2].name} == {"a045", "a315"}
    assert ordered[-1].name in ("a180", "bottom")
    assert P.angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(90.0)


# --- The mesh and renders -----------------------------------------------------------------------------


def test_geometry_normalises_and_orients():
    geom = P.geometry(box(), "cpu")
    assert float(geom.verts.norm(dim=-1).max()) == pytest.approx(1.0, abs=1e-5)
    assert not geom.flipped
    inward = P.geometry(box(inward=True), "cpu")
    assert inward.flipped
    # Turned round, the normals face out again: the +Z side's normals point along +Z (bent at its edges)
    on_front = inward.verts[:, 2] > inward.verts[:, 2].max() - 1e-6
    assert float(inward.normals[on_front][:, 2].mean()) > 0.7


def test_geometry_refuses_meshes_without_uvs():
    mesh = trimesh.Trimesh(vertices=np.eye(3), faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError):
        P.geometry(mesh, "cpu")


def test_texels_cover_each_side_with_its_point_and_normal():
    geom = P.geometry(box(), "cpu")
    tex = P.texels(geom, (3 * CELL, 3 * CELL))
    assert int(tex.covered.sum()) == tex.flat.numel() >= 6 * (INNER - 1) ** 2
    # The front's texels sit on the front and face +Z
    rows, cols = tex.flat // (3 * CELL), tex.flat % (3 * CELL)
    k = SIDES.index((2, +1))
    c0, r0 = (k % 3) * CELL + GUTTER, (k // 3) * CELL + GUTTER
    inside = (rows >= r0 + 2) & (rows < r0 + INNER - 2) & (cols >= c0 + 2) & (cols < c0 + INNER - 2)
    front_z = float(geom.verts[:, 2].max())
    assert torch.allclose(tex.points[inside][:, 2], torch.full_like(tex.points[inside][:, 2], front_z), atol=1e-4)
    assert float(tex.face_normals[inside][:, 2].min()) > 0.999
    # The smooth normals bend round the box's edges, within a grid cell of them
    quad = INNER // 4
    middle = (rows >= r0 + quad) & (rows < r0 + INNER - quad) & (cols >= c0 + quad) & (cols < c0 + INNER - quad)
    assert float(tex.normals[middle][:, 2].min()) > 0.999 and float(tex.normals[inside][:, 2].min()) > 0.7


def test_render_shows_the_texture_lit_from_the_camera():
    colours = {side: OLD for side in SIDES}
    colours[(2, +1)] = (200, 40, 40)
    geom = P.geometry(box(colours), "cpu")
    texture, _ = P.texture_of(box(colours), "cpu")
    camera = P.frame(geom.verts, "front", 0, 0, perspective=0.0, pixels=96 * 64)
    shot = P.render(geom, texture, camera, ambient=0.0, supersample=2)
    h, w = shot.mask.shape
    centre = shot.image[h // 2, w // 2]
    # Seen square on with all the light from the camera: the texture's colour itself
    torch.testing.assert_close(centre * 255, torch.tensor([200.0, 40.0, 40.0]), atol=2.0, rtol=0)
    corner = shot.image[1, 1]
    torch.testing.assert_close(corner, torch.tensor(P.BACKGROUND), atol=1e-3, rtol=0)
    assert bool(shot.mask[h // 2, w // 2]) and not bool(shot.mask[1, 1])
    # Some ambient: still the texture's colour, square on
    lit = P.render(geom, texture, camera, ambient=0.55, supersample=1)
    torch.testing.assert_close(lit.image[h // 2, w // 2] * 255, torch.tensor([200.0, 40.0, 40.0]), atol=2.0, rtol=0)


def test_picture_reference_crops_the_object_on_the_background():
    picture = np.zeros((300, 400, 4), np.uint8)
    picture[..., :3] = (10, 200, 10)
    picture[100:200, 150:250, 3] = 255  # the object: a 100 x 100 square
    image = P.picture_reference(Image.fromarray(picture, "RGBA"), margin=0.1, pixels=64 * 64)
    assert image.width % 16 == 0 and image.height % 16 == 0 and image.width * image.height <= 64 * 64
    array = np.asarray(image).astype(int)
    assert tuple(array[image.height // 2, image.width // 2]) == (10, 200, 10)
    grey = round(P.BACKGROUND[0] * 255)
    assert np.abs(array[0, 0] - grey).max() <= 2  # the margin is background


# --- Checks -------------------------------------------------------------------------------------------


def scene(h=120, w=160, centre=(80, 60), radius=30, colour=(0.2, 0.3, 0.8), background=0.92):
    """A disc on a plain background, with its mask."""
    yy, xx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    disc = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius**2
    image = torch.full((h, w, 3), background)
    image[disc] = torch.tensor(colour)
    return image, disc


def test_object_mask_finds_the_object_but_not_its_shadow():
    image, disc = scene()
    # A grey shadow on the floor below the disc, and a patch of background colour inside the disc
    yy, xx = torch.meshgrid(torch.arange(120) + 0.5, torch.arange(160) + 0.5, indexing="ij")
    shadow = ((xx - 80) / 45) ** 2 + ((yy - 100) / 10) ** 2 <= 1
    image[shadow & ~disc] = torch.tensor([0.6, 0.6, 0.6])
    hole = (xx - 80) ** 2 + (yy - 60) ** 2 <= 8**2
    image[hole] = torch.tensor([0.92, 0.92, 0.92])
    mask = P.object_mask(image, disc)
    assert P.iou(mask, disc) > 0.99  # the hole is walled in, the shadow is floor
    # Without the render's mask a shadow counts as the object
    assert P.iou(P.object_mask(image), disc) < 0.9


def test_align_finds_a_small_shift_and_scale():
    _, rendered = scene(centre=(80, 60), radius=30)
    _, painted = scene(centre=(83, 58), radius=32)  # drawn 3 px right, 2 up, a little larger
    fit = P.align(painted, rendered, max_shift=0.1)
    assert fit.iou_before < 0.85 and fit.iou > 0.97
    assert fit.shift[0] == pytest.approx(3.0, abs=0.75) and fit.shift[1] == pytest.approx(-2.0, abs=0.75)
    assert math.exp(fit.log_scale) == pytest.approx(32 / 30, abs=0.02)
    # Warped by it, the painted disc lies on the rendered one
    warped = P.warp(painted.float()[..., None], fit)[..., 0] > 0.5
    assert P.iou(warped, rendered) > 0.97


def test_align_stays_within_its_bounds():
    _, rendered = scene(centre=(50, 60))
    _, painted = scene(centre=(110, 60))  # far off: no small shift fixes it
    fit = P.align(painted, rendered, max_shift=0.04)
    assert abs(fit.shift[0]) <= 0.04 * 160 + 1e-6
    assert fit.iou < 0.5


def test_warp_with_no_change_is_the_identity():
    image, _ = scene()
    same = P.warp(image, P.Alignment((0.0, 0.0), 0.0, (80.0, 60.0), 1.0, 1.0))
    torch.testing.assert_close(same, image, atol=1e-5, rtol=0)


def test_gains_bring_a_view_to_the_established_colours():
    generator = torch.Generator().manual_seed(0)
    established = torch.rand(2000, 3, generator=generator) * 0.8 + 0.1
    view = established / torch.tensor([1.2, 0.9, 1.1])
    found = P.gains(view, established, torch.ones(2000))
    torch.testing.assert_close(found, torch.tensor([1.2, 0.9, 1.1]), atol=1e-3, rtol=0)
    # At most MAX_GAIN either way, and nothing from too few texels
    assert float(P.gains(view / 3, established, torch.ones(2000)).max()) == pytest.approx(P.MAX_GAIN)
    assert P.gains(view, established, torch.cat([torch.ones(100), torch.zeros(1900)])) is None


def test_novelty_counts_only_what_the_render_does_not_have():
    image, disc = scene(h=128, w=128, centre=(64, 64), radius=40)
    darker = image.clone()
    darker[disc] = darker[disc] * 0.6  # another colour, the same structure
    assert P.novelty(image, darker, disc) < 0.05
    # The same disc with a bar across it: a new edge
    barred = image.clone()
    barred[56:72, 30:98] = torch.tensor([0.05, 0.05, 0.05])
    assert P.novelty(image, barred, disc) > 0.5
    # A bar the render has too isn't new
    assert P.novelty(barred, barred.clone(), disc) < 0.05


def test_describe_names_the_side_from_the_pictures_camera():
    geom = P.geometry(box(), "cpu")
    names = {}
    for camera in P.ring(geom.verts, 30.0, around=8, **small()):
        names[camera.name] = P.describe(camera, 30.0, 10.0)
    assert names["a000"] == "from the front"
    assert names["a045"] == names["a315"] == "from the front, turned to one side"
    assert names["a090"] == names["a270"] == "from the side"
    assert names["a135"] == names["a225"] == "from behind, turned to one side"
    assert names["a180"] == "from directly behind, showing its back"
    assert names["top"] == "from directly above" and names["bottom"] == "from directly below, showing its underside"


def test_main_colours_names_what_covers_the_object():
    cutout = np.zeros((100, 100, 4), np.uint8)
    cutout[..., :3] = (255, 255, 255)  # the background, outside the alpha
    cutout[10:90, 10:90, 3] = 255
    cutout[10:70, 10:90, :3] = (72, 122, 178)  # 75 %: steel blue
    cutout[70:90, 10:90, :3] = (18, 18, 20)  # 25 %: black
    assert P.main_colours(Image.fromarray(cutout, "RGBA")) == ["steel blue", "black"]
    cutout[..., 3] = 0
    assert P.main_colours(Image.fromarray(cutout, "RGBA")) == []


def test_robust_colour_leaves_out_what_one_view_alone_drew():
    n = 4
    base = torch.tensor([0.2, 0.3, 0.5])
    views = [P.Samples(weight=torch.ones(n), colour=base.expand(n, 3).clone()) for _ in range(3)]
    views[1].colour[0] = torch.tensor([0.9, 0.9, 0.9])  # a highlight in one view at texel 0
    plain = sum(v.weight[:, None] * v.colour for v in views) / 3
    robust = P.robust_colour(views)
    torch.testing.assert_close(robust[0], base)
    assert float(plain[0, 0]) > 0.4
    # One view alone at a texel: its colour
    lonely = [P.Samples(weight=torch.tensor([1.0, 0.0]), colour=torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.9, 0.9]]))]
    torch.testing.assert_close(P.robust_colour(lonely)[0], torch.tensor([0.1, 0.2, 0.3]))


# --- Bake ---------------------------------------------------------------------------------------------


def painted_like(geom, camera, colour, size=None):
    """A painted view of ``camera``: the object in one flat colour on the background (as an editing model would paint it)."""
    texture = torch.full((3 * CELL, 3 * CELL, 3), 0.5)
    shot = P.render(geom, texture, camera, supersample=1)
    image = torch.where(shot.mask[..., None], torch.tensor(colour), torch.tensor(P.BACKGROUND))
    return image, shot.mask


def test_a_view_bakes_into_the_sides_it_sees():
    mesh = box()
    geom = P.geometry(mesh, "cpu")
    texture, _ = P.texture_of(mesh, "cpu")
    tex = P.texels(geom, tuple(texture.shape[:2]))
    camera = P.frame(geom.verts, "front", 0, 0, pixels=128 * 96)
    image, mask = painted_like(geom, camera, (0.1, 0.7, 0.2))
    samples = P.view_samples(geom, tex, camera, image, mask)
    blend = P.Blend(tex.flat.numel(), "cpu")
    blend.add(samples)
    out = P.compose(texture, tex, blend.colour(), blend.amount())
    out = (out.numpy() * 255 + 0.5).astype(np.uint8)
    front, back = SIDES.index((2, +1)), SIDES.index((2, -1))
    assert np.abs(cell(out, front, inset=INNER // 4) - (26, 179, 51)).max() <= 3  # the view's colour where seen square on
    assert np.abs(cell(out, back, inset=0) - OLD).max() <= 1  # the back isn't seen: untouched
    # Towards the front's edges (its silhouette in this view) the old colour shows through more
    edge = cell(out, front)[INNER // 2, 0]
    assert np.abs(edge - (26, 179, 51)).max() > 3


def test_compose_carries_the_change_into_the_gutters():
    mesh = box()
    geom = P.geometry(mesh, "cpu")
    texture, _ = P.texture_of(mesh, "cpu")
    tex = P.texels(geom, tuple(texture.shape[:2]))
    k = SIDES.index((2, +1))
    c0, r0 = (k % 3) * CELL + GUTTER, (k // 3) * CELL + GUTTER
    rows, cols = tex.flat // (3 * CELL), tex.flat % (3 * CELL)
    on_front = (rows >= r0 - 1) & (rows <= r0 + INNER) & (cols >= c0 - 1) & (cols <= c0 + INNER)
    target = projection._srgb_to_linear(torch.tensor([0.1, 0.7, 0.2]))
    out = P.compose(texture, tex, target.expand(tex.flat.numel(), 3), on_front.float())
    out = (out.numpy() * 255 + 0.5).astype(np.uint8)
    assert np.abs(cell(out, k) - (26, 179, 51)).max() <= 1
    # Its gutter carries the change on, so filtering doesn't bring the old colour back at the chart's edge
    assert np.abs(out[r0 + INNER // 2, c0 - 3].astype(int) - (26, 179, 51)).max() <= 2
    # Other charts keep theirs
    assert np.abs(cell(out, SIDES.index((2, -1))) - OLD).max() <= 1


def test_view_samples_refuses_a_view_of_the_wrong_size():
    geom = P.geometry(box(), "cpu")
    tex = P.texels(geom, (3 * CELL, 3 * CELL))
    camera = P.frame(geom.verts, "front", 0, 0, pixels=128 * 96)
    with pytest.raises(ValueError):
        P.view_samples(geom, tex, camera, torch.zeros(10, 10, 3), torch.zeros(10, 10, dtype=torch.bool))


def test_two_views_blend_by_how_squarely_they_see():
    geom = P.geometry(box(), "cpu")
    tex = P.texels(geom, (3 * CELL, 3 * CELL))
    front = P.frame(geom.verts, "front", 0, 0, pixels=128 * 96)
    slant = P.frame(geom.verts, "slant", 60, 0, pixels=128 * 96)
    blend = P.Blend(tex.flat.numel(), "cpu")
    for camera, colour in ((front, (1.0, 0.0, 0.0)), (slant, (0.0, 0.0, 1.0))):
        image, mask = painted_like(geom, camera, colour)
        blend.add(P.view_samples(geom, tex, camera, image, mask))
    colour = blend.colour()
    on_front = (tex.normals[:, 2] > 0.99) & (tex.points[:, 0].abs() < 0.3) & (tex.points[:, 1].abs() < 0.3)
    on_side = (tex.normals[:, 0] > 0.99) & (tex.points[:, 2].abs() < 0.2) & (tex.points[:, 1].abs() < 0.3)
    # The front mostly from the front view, the +X side only from the slanted one
    assert float(colour[on_front][:, 0].mean()) > 0.8
    assert float(colour[on_side][:, 2].mean()) > 0.99


# --- The whole loop -----------------------------------------------------------------------------------


def no_picture():
    """A cutout whose object matches nothing: the projection doesn't apply, so the views start from azimuth 0."""
    cutout = np.zeros((64, 64, 4), np.uint8)
    cutout[..., :3] = 200
    cutout[20:44, 30:34, 3] = 255  # a thin bar
    return Image.fromarray(cutout, "RGBA")


def flat_painter(colour=(40, 160, 220), shift=(0, 0)):
    """An editing model that paints the render's object one colour, shifted by ``shift`` pixels."""
    calls = []

    def paint(render, picture, neighbour, seed, view):
        calls.append(
            {"size": render.size, "neighbour": neighbour is not None, "seed": seed, "picture": picture.size, "view": view}
        )
        array = np.asarray(render).astype(int)
        background = np.round(np.asarray(P.BACKGROUND) * 255).astype(int)
        mask = np.abs(array - background).max(-1) > 6
        out = np.empty_like(array)
        out[:] = background
        out[mask] = colour
        out = np.roll(out, shift=(shift[1], shift[0]), axis=(0, 1))
        return Image.fromarray(out.astype(np.uint8), "RGB")

    return paint, calls


def test_paint_views_repaints_what_the_views_see():
    mesh = box()
    before = np.asarray(mesh.visual.material.baseColorTexture).copy()
    paint, calls = flat_painter(shift=(2, 1))
    result = P.paint_views(mesh, no_picture(), paint, device="cpu", around=4, top=True, bottom=True, log=lambda _: None, **small())
    report = result.report
    assert report["picture"]["applied"] is False
    assert report["accepted"] == 6 and len(result.views) == 6
    assert all(view.accepted and view.attempts[0]["iou"] > 0.95 for view in result.views)
    # The first view painted has no neighbour; every later one gets the nearest painted view
    assert [call["neighbour"] for call in calls] == [False] + [True] * 5
    # Each is told which side it shows (the picture's camera, at azimuth 0 here, is the front)
    sides = {call["view"]["name"]: call["view"]["side"] for call in calls}
    assert sides == {
        "a000": "from the front",
        "a090": "from the side",
        "a270": "from the side",
        "a180": "from directly behind, showing its back",
        "top": "from directly above",
        "bottom": "from directly below, showing its underside",
    }
    # With one colour everywhere the robust blend is the same
    robust = np.asarray(result.robust.convert("RGB"))
    for k in range(6):
        assert np.abs(cell(robust, k, inset=INNER // 4) - (40, 160, 220)).max() <= 4, k
    # Every side of the box is seen square on by one view: all of it takes the painted colour
    out = np.asarray(result.texture.convert("RGB"))
    for k in range(6):
        assert np.abs(cell(out, k, inset=INNER // 4) - (40, 160, 220)).max() <= 4, k
    # The mesh itself isn't changed
    np.testing.assert_array_equal(np.asarray(mesh.visual.material.baseColorTexture), before)
    # Most texels take it in full; nearer the box's edges (each side's silhouette in its own view, and where its
    # smooth normals bend) the old colour shows through a little
    assert report["changed_share"] > 0.6
    assert sum(view.texels for view in result.views) <= report["texels"]


def test_paint_views_leaves_out_a_view_that_moved_and_retries_it():
    mesh = box()
    seeds = []

    def paint(render, picture, neighbour, seed, view):
        seeds.append(seed)
        array = np.asarray(render)
        return Image.fromarray(np.roll(array, shift=40, axis=1))  # far off every time

    result = P.paint_views(mesh, no_picture(), paint, device="cpu", around=2, top=False, bottom=False, attempts=2, log=lambda _: None, **small())
    assert result.report["accepted"] == 0
    assert all(not view.accepted and len(view.attempts) == 2 for view in result.views)
    assert seeds == [0, 1, 100, 101]
    # Nothing went in: the texture is the old one
    np.testing.assert_array_equal(np.asarray(result.texture.convert("RGB")), np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")))


def test_paint_views_leaves_out_a_view_that_draws_what_the_render_does_not_have():
    mesh = box()

    def paint(render, picture, neighbour, seed, view):
        array = np.asarray(render).astype(int)
        background = np.round(np.asarray(P.BACKGROUND) * 255).astype(int)
        mask = np.abs(array - background).max(-1) > 6
        out = np.empty_like(array)
        out[:] = background
        out[mask] = (200, 60, 50)
        if view["name"] == "a180":  # the back: stripes the render doesn't have
            stripes = np.zeros(mask.shape, bool)
            stripes[:, ::6] = True
            out[mask & stripes] = (20, 20, 20)
        return Image.fromarray(out.astype(np.uint8), "RGB")

    result = P.paint_views(mesh, no_picture(), paint, device="cpu", around=4, top=False, bottom=False, log=lambda _: None, **small())
    by_name = {view.camera.name: view for view in result.views}
    assert not by_name["a180"].accepted and by_name["a180"].attempts[0]["novelty"] > P.MAX_NOVELTY
    assert all(by_name[name].accepted and by_name[name].attempts[0]["novelty"] < 0.05 for name in ("a000", "a090", "a270"))


def test_paint_views_scales_later_views_to_the_first():
    mesh = box()
    colours = iter([(40, 160, 220)] + [(20, 80, 110)] * 10)  # later views darker
    calls = []

    def paint(render, picture, neighbour, seed, view):
        calls.append(seed)
        colour = next(colours)
        painter, _ = flat_painter(colour)
        return painter(render, picture, neighbour, seed, view)

    result = P.paint_views(mesh, no_picture(), paint, device="cpu", around=8, top=False, bottom=False, log=lambda _: None, **small())
    gains = {view.camera.name: view.gains for view in result.views}
    assert gains["a000"] is None  # nothing established before the first view
    # The two views either side of it see the first view's side at 45 degrees: brought up towards it
    assert min(gains["a045"]) > 1.2 and min(gains["a315"]) > 1.2
    # A view that shares nothing sure with the views before it (a box's next side, seen square on) keeps its colour
    assert gains["a090"] is None or max(abs(g - 1) for g in gains["a090"]) < 0.05


# --- Moving a mesh ------------------------------------------------------------------------------------


def test_pack_and_unpack_give_the_same_mesh():
    mesh = box()
    rough = np.zeros((8, 8, 3), np.uint8)
    rough[..., 1] = 200
    mesh.visual.material.metallicRoughnessTexture = Image.fromarray(rough)
    mesh.visual.material.roughnessFactor = 1.0
    mesh.visual.material.alphaMode = "OPAQUE"
    again = P.unpack_mesh(P.pack_mesh(mesh))
    np.testing.assert_array_equal(again.vertices, mesh.vertices)
    np.testing.assert_array_equal(again.faces, mesh.faces)
    np.testing.assert_array_equal(again.visual.uv, mesh.visual.uv)
    material = again.visual.material
    np.testing.assert_array_equal(np.asarray(material.baseColorTexture), np.asarray(mesh.visual.material.baseColorTexture))
    np.testing.assert_array_equal(np.asarray(material.metallicRoughnessTexture), rough)
    assert material.roughnessFactor == 1.0 and material.metallicFactor == 0.0
    assert material.alphaMode == "OPAQUE"
    # And it exports as a GLB, as production's export does
    glb = again.export(file_type="glb")
    assert isinstance(glb, bytes) and glb[:4] == b"glTF"


def test_sheet_puts_each_view_on_a_row():
    render = Image.new("RGB", (64, 32), (200, 200, 200))
    views = [P.View(camera=None, render=render, painted=render, aligned=render), P.View(camera=None, render=render)]
    image = P.sheet(views, height=40)
    assert image.size == (3 * 80, 80)
