"""
The judge's turntable grid (judgeviews.py) on synthetic textured meshes, CPU only: which side each view
sees, the texture where the UVs put it, the background where nothing is, the framing, the light.
"""

import math

import numpy as np
import pytest
import trimesh
import trimesh.visual
from PIL import Image

from forge3d_worker import judgeviews as J
from forge3d_worker import projection as P

UNLIT = dict(ambient=1.0, key=0.0, rim=0.0)  # the texture's own colours
BACKGROUND = round(J.BACKGROUND * 255)
# A box's sides as (axis, sign), glTF axes: +Y up, +Z the front
SIDES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
COLOURS = {
    (0, +1): (200, 40, 40),  # +X: the front view's right
    (0, -1): (40, 170, 60),  # -X
    (1, +1): (50, 70, 210),  # top
    (1, -1): (220, 200, 40),  # bottom
    (2, +1): (240, 120, 200),  # front
    (2, -1): (40, 190, 200),  # back
}
CELL = 32


def textured(vertices, faces, uv, texture) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(np.asarray(texture, np.uint8)))
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, float), faces=np.asarray(faces), process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.asarray(uv, float), material=material),
    )


def box(colours=COLOURS, half=(0.5, 0.35, 0.25), n=2, inward=False) -> trimesh.Trimesh:
    """A box, each side an n x n grid with its own cell of a 3 x 3 atlas in that side's colour (gutters too)."""
    texture = np.full((3 * CELL, 3 * CELL, 3), 128, np.uint8)
    verts, uvs, faces = [], [], []
    steps = np.linspace(-1, 1, n + 1)
    for k, (axis, sign) in enumerate(SIDES):
        a, b = ((axis + 1) % 3, (axis + 2) % 3) if sign > 0 else ((axis + 2) % 3, (axis + 1) % 3)
        c0, r0 = (k % 3) * CELL, (k // 3) * CELL
        texture[r0 : r0 + CELL, c0 : c0 + CELL] = colours[(axis, sign)]
        base = len(verts)
        for sb in steps:
            for sa in steps:
                p = np.zeros(3)
                p[axis], p[a], p[b] = sign * half[axis], sa * half[a], sb * half[b]
                verts.append(p)
                # Inset 4 texels: bilinear filtering at a side's edge stays in its colour
                uvs.append(((c0 + 4 + (sa + 1) / 2 * (CELL - 8)) / (3 * CELL), 1 - (r0 + 4 + (1 - sb) / 2 * (CELL - 8)) / (3 * CELL)))
        for j in range(n):
            for i in range(n):
                q = base + j * (n + 1) + i
                for tri in ([q, q + 1, q + n + 2], [q, q + n + 2, q + n + 1]):
                    faces.append(tri[::-1] if inward else tri)
    return textured(verts, faces, uvs, texture)


def quad(texture) -> trimesh.Trimesh:
    """A unit square facing +Z (the front), u along +X and v along +Y."""
    vertices = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    return textured(vertices, [[0, 1, 2], [0, 2, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]], texture)


def cells(image: Image.Image, size: int, count: int, columns: int = 3) -> list:
    array = np.asarray(image)
    return [array[(i // columns) * size : (i // columns + 1) * size, (i % columns) * size : (i % columns + 1) * size] for i in range(count)]


def object_mask(cell: np.ndarray) -> np.ndarray:
    return np.abs(cell.astype(int) - BACKGROUND).max(-1) > 2


def test_the_front_shows_the_front_and_the_views_go_round():
    image = J.turntable(box(), size=64, views=4, elevation=0.0, supersample=1, device="cpu", **UNLIT)
    assert image.size == (3 * 64, 2 * 64) and image.mode == "RGB"
    front, right, back, left = cells(image, 64, 4)
    # Azimuth 0 sees +Z, 90 sees +X, 180 sees -Z, 270 sees -X
    assert tuple(front[32, 32]) == COLOURS[(2, +1)]
    assert tuple(right[32, 32]) == COLOURS[(0, +1)]
    assert tuple(back[32, 32]) == COLOURS[(2, -1)]
    assert tuple(left[32, 32]) == COLOURS[(0, -1)]
    assert J.view_azimuths(4) == [0.0, 90.0, 180.0, 270.0]


def test_six_views_from_above_put_the_front_top_left_and_the_back_bottom_left():
    image = J.turntable(box(), size=64, device="cpu", **UNLIT)
    assert image.size == (192, 128)
    views = cells(image, 64, 6)
    assert tuple(views[0][36, 32]) == COLOURS[(2, +1)] and tuple(views[3][36, 32]) == COLOURS[(2, -1)]
    # 20 degrees above: the top shows above each view's middle, the bottom nowhere
    for view in views:
        colours = {tuple(pixel) for pixel in view.reshape(-1, 3)}
        assert COLOURS[(1, +1)] in colours and COLOURS[(1, -1)] not in colours
    rows = np.nonzero((views[0] == COLOURS[(1, +1)]).all(-1))[0]
    assert rows.max() < 32
    # Between front and right at 60 degrees: the front on the image's left, +X on its right
    assert tuple(views[1][36, 20]) == COLOURS[(2, +1)] and tuple(views[1][36, 44]) == COLOURS[(0, +1)]


def test_the_texture_lands_where_the_uvs_say():
    texture = np.zeros((64, 64, 3), np.uint8)
    texture[:32, :32] = (255, 0, 0)  # row 0 is the top of the texture: v = 1, the quad's top (+Y)
    texture[:32, 32:] = (0, 255, 0)
    texture[32:, :32] = (0, 0, 255)
    texture[32:, 32:] = (255, 255, 0)
    image = J.turntable(quad(texture), size=80, views=1, elevation=0.0, columns=1, supersample=1, device="cpu", **UNLIT)
    view = np.asarray(image)
    assert image.size == (80, 80)
    # Seen from the front, +X is on the image's right and +Y up
    assert tuple(view[25, 25]) == (255, 0, 0) and tuple(view[25, 55]) == (0, 255, 0)
    assert tuple(view[55, 25]) == (0, 0, 255) and tuple(view[55, 55]) == (255, 255, 0)
    # The quad fills 92% of the frame's height and is centred
    rows, columns = np.nonzero(object_mask(view))
    assert abs((rows.min() + rows.max()) / 2 - 39.5) <= 1 and abs((columns.min() + columns.max()) / 2 - 39.5) <= 1
    assert 0.88 * 80 <= rows.max() - rows.min() + 1 <= 0.94 * 80


def test_the_background_is_where_nothing_is():
    image = J.turntable(box(), size=96, views=4, device="cpu")
    views = cells(image, 96, 6)
    for view in views[:4]:
        # The model stays 4% of the frame (3.8 pixels here) clear of each edge
        assert (view[:3] == BACKGROUND).all() and (view[-3:] == BACKGROUND).all()
        assert (view[:, :3] == BACKGROUND).all() and (view[:, -3:] == BACKGROUND).all()
        assert 0.15 < object_mask(view).mean() < 0.8
    assert (views[4] == BACKGROUND).all() and (views[5] == BACKGROUND).all()  # no fifth or sixth view
    black = np.asarray(J.turntable(box(), size=48, views=4, background=0.0, device="cpu"))
    assert (black[:3, :3] == 0).all()


def test_the_model_keeps_its_size_and_fits_every_view():
    long = box(half=(1.0, 0.1, 0.1))
    image = J.turntable(long, size=96, views=6, elevation=20.0, supersample=1, device="cpu")
    spans = []
    for view in cells(image, 96, 6):
        rows, columns = np.nonzero(object_mask(view))
        # Inside the frame from every side, the long axis nearly across it when seen side on
        assert rows.min() >= 3 and columns.min() >= 3 and rows.max() <= 92 and columns.max() <= 92
        spans.append(columns.max() - columns.min() + 1)
    # One distance for all: views that see the box the same way round (0 and 180, 60 and 240) match
    assert abs(spans[0] - spans[3]) <= 1 and abs(spans[1] - spans[4]) <= 1 and abs(spans[2] - spans[5]) <= 1
    assert spans[0] > 70  # side on, the long box spans most of the frame


def test_plain_light_and_a_faint_rim():
    grey = {side: (128, 128, 128) for side in SIDES}
    lit = np.asarray(J.turntable(box(grey), size=64, views=4, elevation=0.0, supersample=1, device="cpu"))
    # Facing the camera: base colour x (ambient + key x cos(angle to the key light)), no rim
    key = np.array(J.KEY_DIRECTION) / np.linalg.norm(J.KEY_DIRECTION)
    expected = P._linear_to_srgb(P._srgb_to_linear(P.torch.tensor(128 / 255)) * (J.AMBIENT + J.KEY * key[2]))
    assert abs(int(lit[32, 32, 0]) - float(expected) * 255) <= 1.5
    # Eight views from the side: at 45 degrees the face on the image's left (towards the key light) is the
    # brighter of the two seen
    eight = J.turntable(box(grey, n=8), size=96, views=8, elevation=0.0, columns=4, supersample=1, device="cpu")
    view = cells(eight, 96, 8, columns=4)[1].astype(int)
    assert view[48, 36, 0] > view[48, 78, 0] + 15  # the front (+Z) on the left, +X on the right
    # The rim brightens where the surface turns away from the camera (round the outline, where the normals
    # bend between faces), and nothing that faces the camera
    no_rim = np.asarray(J.turntable(box(grey), size=64, views=4, elevation=0.0, supersample=1, device="cpu", rim=0.0)).astype(int)
    front, plain = lit[:64, :64].astype(int), no_rim[:64, :64]
    inside = object_mask(lit[:64, :64])
    outline = inside & ~(np.pad(inside, 1)[2:, 1:-1] & np.pad(inside, 1)[:-2, 1:-1] & np.pad(inside, 1)[1:-1, 2:] & np.pad(inside, 1)[1:-1, :-2])
    assert (front[outline] >= plain[outline]).all() and (front[outline] - plain[outline]).mean() > 2
    assert np.abs(front[32, 32] - plain[32, 32]).max() == 0


def test_a_box_wound_inwards_looks_the_same():
    outward = np.asarray(J.turntable(box(), size=48, device="cpu")).astype(int)
    inward = np.asarray(J.turntable(box(inward=True), size=48, device="cpu")).astype(int)
    assert np.abs(outward - inward).max() <= 1


def test_supersampling_smooths_the_outline():
    plain = cells(J.turntable(box(), size=48, views=6, supersample=1, device="cpu", **UNLIT), 48, 6)[1]
    smooth = cells(J.turntable(box(), size=48, views=6, supersample=4, device="cpu", **UNLIT), 48, 6)[1]

    def blended(view):
        known = [np.array(colour) for colour in COLOURS.values()] + [np.full(3, BACKGROUND)]
        distance = np.min([np.abs(view.astype(int) - colour).max(-1) for colour in known], axis=0)
        return int((distance > 3).sum())

    assert blended(plain) == 0 and blended(smooth) > 20


def test_the_mesh_is_left_as_it_was():
    mesh = box()
    vertices, texture = mesh.vertices.copy(), np.asarray(mesh.visual.material.baseColorTexture).copy()
    J.turntable(mesh, size=32, device="cpu")
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(np.asarray(mesh.visual.material.baseColorTexture), texture)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="texture"):
        J.turntable(trimesh.creation.box(), size=16, device="cpu")
    empty = box()
    empty.faces = np.zeros((0, 3), int)
    with pytest.raises(ValueError, match="no triangles"):
        J.turntable(empty, size=16, device="cpu")
    with pytest.raises(ValueError, match="at least 1"):
        J.turntable(box(), size=0, device="cpu")
    point = box()
    point.vertices = np.zeros_like(point.vertices)
    with pytest.raises(ValueError, match="a point"):
        J.turntable(point, size=16, device="cpu")


def test_the_cameras_are_the_projections():
    # Azimuth 0 looks from +Z, 90 from +X; up stays up at any azimuth
    for azimuth, back in ((0.0, [0, 0, 1]), (90.0, [1, 0, 0]), (180.0, [0, 0, -1])):
        right, up, towards = J._axes(azimuth, 0.0)
        np.testing.assert_allclose(towards, back, atol=1e-9)
        np.testing.assert_allclose(up, [0, 1, 0], atol=1e-9)
    right, up, towards = J._axes(0.0, 20.0)
    np.testing.assert_allclose(towards, [0, math.sin(math.radians(20)), math.cos(math.radians(20))], atol=1e-9)
