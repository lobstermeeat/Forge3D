"""
Picture projection on synthetic data (CPU only): a box with a different colour on each side and a
cylinder, and pictures of them ray-cast from known views.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
import trimesh.visual
from PIL import Image

from forge3d_worker import projection as P

# The box's half sizes and each side's colour in the model's own texture. The front's is the light colour
# of the checker the pictures paint on it: TRELLIS.2 got the paint, and blurred the pattern away
HALF = np.array([0.5, 0.35, 0.25])
SIDES = [  # (axis, sign, colour)
    (0, +1, (200, 40, 40)),
    (0, -1, (40, 170, 60)),
    (1, +1, (50, 70, 210)),
    (1, -1, (220, 200, 40)),
    (2, +1, (240, 230, 240)),
    (2, -1, (40, 190, 200)),
]
TEXTURE = 256
CELL = TEXTURE // 4  # each side gets a CELL x CELL square of the atlas, its UVs inset by GUTTER texels
GUTTER = 6
INNER = CELL - 2 * GUTTER
VIEW = P.View(azimuth=30.0, elevation=20.0, roll=0.0, perspective=0.25)


@pytest.fixture(autouse=True)
def small_search(monkeypatch):
    """Fewer samples than production: simple shapes need few, and it keeps the tests quick on a CPU."""
    monkeypatch.setattr(P, "SEARCH_POINTS", 4000)
    monkeypatch.setattr(P, "REFINE_POINTS", 20000)


def side_frame(axis: int, sign: int):
    """Two in-plane axes for a side, ordered so that (a, b, outward normal) is right-handed."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    return (a, b) if sign > 0 else (b, a)


def cell_origin(i: int):
    return (i % 4) * CELL + GUTTER, (i // 4) * CELL + GUTTER  # texel column, row of the UV square


def textured(verts, faces, uvs, texture) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture), metallicFactor=0.0)
    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces),
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs), material=material),
    )


def make_box(n: int = 8) -> trimesh.Trimesh:
    """
    Each side an n x n grid of quads with its own vertices (split at the UV seams like to_glb's) and its
    own square of an atlas texture. Fine enough that smoothed normals bend only near the edges. Like
    to_glb's inpainting, each side's colour runs on into the gutter around its square.
    """
    verts, uvs, faces = [], [], []
    texture = np.full((TEXTURE, TEXTURE, 3), 128, np.uint8)
    steps = np.linspace(-1, 1, n + 1)
    for i, (axis, sign, colour) in enumerate(SIDES):
        a, b = side_frame(axis, sign)
        c0, r0 = cell_origin(i)
        texture[r0 - GUTTER : r0 - GUTTER + CELL, c0 - GUTTER : c0 - GUTTER + CELL] = colour
        base = len(verts)
        for sb in steps:
            for sa in steps:
                p = np.zeros(3)
                p[axis], p[a], p[b] = sign * HALF[axis], sa * HALF[a], sb * HALF[b]
                verts.append(p)
                # trimesh UVs: v up, image row 0 at v = 1
                uvs.append(((c0 + (sa + 1) / 2 * INNER) / TEXTURE, 1 - (r0 + (1 - sb) / 2 * INNER) / TEXTURE))
        for j in range(n):
            for k in range(n):
                q = base + j * (n + 1) + k
                faces += [[q, q + 1, q + n + 2], [q, q + n + 2, q + n + 1]]
    return textured(verts, faces, uvs, texture)


def checker(points: np.ndarray) -> np.ndarray:
    """The front (+Z) side's painted detail in the picture: a 4 x 3 checker of light and dark."""
    i = np.floor((points[:, 0] + HALF[0]) / (2 * HALF[0]) * 4).astype(int)
    j = np.floor((points[:, 1] + HALF[1]) / (2 * HALF[1]) * 3).astype(int)
    return np.where(((i + j) % 2 == 0)[:, None], [[240, 230, 240]], [[90, 20, 90]]).astype(np.float64)


def camera_rays(view: P.View, corners: np.ndarray, size: int, fill: float):
    """
    Rays through each pixel of a size x size picture in which the points `corners` (normalised: the
    model's bounding sphere has radius 1) fill `fill` of the frame, centred, the way projection's camera
    model places a model (its bounding square onto the picture's).
    """
    params = torch.tensor([[view.azimuth, view.elevation, view.roll, view.perspective, 0, 0, 0]], dtype=torch.float32)
    x, y, _, _ = P.project(torch.tensor(corners, dtype=torch.float32), params)
    cx, cy, side = (float(t) for t in P._boxes(x, y)[0])
    right, up, back = (t[0].numpy().astype(np.float64) for t in P.view_axes(params))
    offset = (1 - fill) / 2 * size
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    xi = cx + ((cols - offset) / (fill * size) - 0.5) * side
    yi = cy - ((rows - offset) / (fill * size) - 0.5) * side
    target = xi[..., None] * right + yi[..., None] * up  # on the picture plane through the origin
    origin = np.broadcast_to(back / view.perspective, target.shape) if view.perspective > 0 else target + 10 * back
    direction = target - origin
    return origin, direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def rgba(rgb: np.ndarray, hit: np.ndarray) -> Image.Image:
    alpha = np.where(hit, 255, 0)[..., None]
    return Image.fromarray(np.concatenate([rgb, alpha], -1).astype(np.uint8), "RGBA")


def to_linear(srgb: np.ndarray) -> np.ndarray:
    c = srgb / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.clip(linear, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def render_box(
    view: P.View, size: int = 256, fill: float = 0.7, pattern: bool = True, light: np.ndarray | None = None
) -> Image.Image:
    """
    Ray-cast the box from `view`; with `pattern`, its front shows the checker. With `light` (a direction),
    each side is shaded: ambient plus diffuse light, in linear light.
    """
    radius = float(np.linalg.norm(HALF))
    half = HALF / radius
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    origin, direction = camera_rays(view, corners, size, fill)
    with np.errstate(divide="ignore", invalid="ignore"):  # slab test
        t1 = (-half - origin) / direction
        t2 = (half - origin) / direction
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (near <= far) & (far > 0)
    point = origin + near[..., None] * direction
    axis = np.argmax(np.abs(point) / half, axis=-1)
    rgb = np.zeros((size, size, 3))
    for ax, sign, colour in SIDES:
        on = hit & (axis == ax) & (np.sign(point[..., ax]) == sign)
        rgb[on] = colour
        if pattern and ax == 2 and sign > 0:
            rgb[on] = checker(point[on] * radius)
        if light is not None:
            normal = np.zeros(3)
            normal[ax] = sign
            shade = 0.35 + 0.65 * max(0.0, float(normal @ light) / float(np.linalg.norm(light)))
            rgb[on] = to_srgb(to_linear(rgb[on]) * shade)
    return rgba(rgb, hit)


def texel_points(side: int, inset: float = 0.0, step: int = 4):
    """3D points and texel (row, col) of a side's texels, keeping clear of its outer `inset` share."""
    axis, sign, _ = SIDES[side]
    a, b = side_frame(axis, sign)
    c0, r0 = cell_origin(side)
    points, texels = [], []
    for dr in range(1, INNER - 1, step):
        for dc in range(1, INNER - 1, step):
            sa = (dc + 0.5) / INNER * 2 - 1
            sb = 1 - (dr + 0.5) / INNER * 2
            if max(abs(sa), abs(sb)) > 1 - inset:
                continue
            p = np.zeros(3)
            p[axis], p[a], p[b] = sign * HALF[axis], sa * HALF[a], sb * HALF[b]
            points.append(p)
            texels.append((r0 + dr, c0 + dc))
    return np.array(points), np.array(texels)


def exposed(colours: np.ndarray, exposure: dict) -> np.ndarray:
    """sRGB colours put through the projection's exposure match, as reported (one face: one gain)."""
    return to_srgb(to_linear(colours) * exposure["gain"])


def angle_to(report_pose: dict, view: P.View) -> float:
    a = torch.tensor([[report_pose["azimuth"], report_pose["elevation"], report_pose["roll"], 0, 0, 0, 0]])
    b = torch.tensor([[view.azimuth, view.elevation, view.roll, 0, 0, 0, 0]])
    return float(P._angle_between(a.float(), b.float())[0])


def checker_square(points: np.ndarray):
    """Which of the checker's squares (column i, row j from the bottom) each front point is in."""
    i = np.floor((points[:, 0] + HALF[0]) / (2 * HALF[0]) * 4).astype(int)
    j = np.floor((points[:, 1] + HALF[1]) / (2 * HALF[1]) * 3).astype(int)
    return i, j


def test_finds_the_camera_and_paints_the_side_it_shows():
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).copy()
    mesh, report = P.project_picture(mesh, render_box(VIEW), device="cpu")

    assert report["applied"], report["reason"]
    assert angle_to(report["pose"], VIEW) < 4.0, report["pose"]
    assert report["iou"] > 0.97
    # Where the picture shows the texture's paint (the light squares) it is as bright: nothing to match
    assert report["exposure"]["note"] == "matched"
    assert report["exposure"]["gain"] == pytest.approx(1.0, abs=0.05)
    # Half the front is dark squares in the picture: not one colour, so no paint changes all round
    assert report["mode"] == "detail only"
    after = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    assert after.shape == before.shape

    # The front (+Z) faces the camera. The checker's dark squares that lie inside what the picture shows
    # are designs on the paint: painted on whole. Those on the picture's outline (the front's left and
    # bottom edges, from this view) might go on round the side the picture doesn't show: left as they were
    points, texels = texel_points(side=4, inset=0.05, step=2)
    i, j = checker_square(points)
    fx = (points[:, 0] + HALF[0]) / (2 * HALF[0]) * 4
    fy = (points[:, 1] + HALF[1]) / (2 * HALF[1]) * 3
    clear = (np.abs(fx - np.round(fx)) > 0.15) & (np.abs(fy - np.round(fy)) > 0.15)  # off the squares' edges
    expected = exposed(checker(points), report["exposure"])
    got = after[texels[:, 0], texels[:, 1]]
    error = np.abs(got - expected).mean(axis=1)
    dark = (i + j) % 2 == 1
    inner = dark & (i > 0) & (j > 0) & clear
    assert inner.sum() > 50 and np.median(error[inner]) < 10, np.median(error[inner])
    assert (error[inner] < 30).mean() > 0.95
    outer = dark & ((i == 0) | (j == 0)) & clear
    unchanged = np.abs(got - before[texels[:, 0], texels[:, 1]]).max(axis=1)
    assert outer.sum() > 50 and np.median(unchanged[outer]) < 12, np.median(unchanged[outer])
    light = ~dark & clear
    assert np.median(unchanged[light]) < 6, np.median(unchanged[light])

    # The back (-Z), bottom (-Y) and left (-X) never face this camera: untouched
    for side in (1, 3, 5):
        _, tx = texel_points(side=side)
        change = np.abs(after[tx[:, 0], tx[:, 1]] - before[tx[:, 0], tx[:, 1]]).max(axis=1)
        assert change.max() <= 3, (side, change.max())


def test_the_pictures_shading_is_taken_out():
    # The picture shows the texture's own colours, lit from the upper left: the right side in shadow,
    # the top bright. Painting that on would bake the light in; the texture should come back as it was
    light = np.array([-0.5, 0.8, 0.4])
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float64)
    mesh, report = P.project_picture(mesh, render_box(VIEW, pattern=False, light=light), device="cpu")
    assert report["applied"], report["reason"]
    assert report["exposure"]["note"] == "matched"
    assert report["exposure"]["shading"] > 1.5  # the seen sides were lit very differently
    after = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    for side in (0, 2, 4):  # right, top and front: all seen
        _, tx = texel_points(side=side, inset=0.2)
        change = np.abs(after[tx[:, 0], tx[:, 1]] - before[tx[:, 0], tx[:, 1]]).max(axis=1)
        assert np.median(change) < 12, (side, np.median(change))


def test_colour_tells_a_symmetric_shape_from_its_mirror_view():
    # A box looks the same from opposite azimuths; only its colours say which way the picture faced
    view = P.View(azimuth=200.0, elevation=25.0, roll=0.0, perspective=0.0)
    mesh, report = P.project_picture(make_box(), render_box(view, pattern=False), device="cpu")
    assert report["applied"], report["reason"]
    assert angle_to(report["pose"], view) < 4.0, report["pose"]
    assert report["colour"] > 0.9
    # The view from the other side fits the silhouette as well, but not the colours
    assert report["runner_up"]["angle"] > 150
    assert report["runner_up"]["iou"] > 0.95 and report["runner_up"]["colour"] < report["colour"] - 0.3


def plus_sign(size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[:size, :size]
    bar = lambda a, b: (np.abs(a - size / 2) < size * 0.08) & (np.abs(b - size / 2) < size * 0.35)  # noqa: E731
    shape = bar(xx, yy) | bar(yy, xx)
    return rgba(np.full((size, size, 3), (200, 40, 40)), shape)


def test_a_picture_of_something_else_changes_nothing():
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).copy()
    out, report = P.project_picture(mesh, plus_sign(), device="cpu")  # no view of a box has that outline

    assert out is mesh and not report["applied"]
    assert "silhouettes don't match" in report["reason"]
    assert report["iou"] < P.MIN_IOU
    np.testing.assert_array_equal(np.asarray(mesh.visual.material.baseColorTexture), before)


@pytest.mark.parametrize(
    "picture, reason",
    [
        (None, "no picture"),
        (Image.new("RGB", (64, 64), (200, 30, 30)), "no alpha"),
        (Image.new("RGBA", (64, 64), (0, 0, 0, 0)), "no object"),
        (Image.new("RGBA", (16, 16), (200, 30, 30, 255)), "too small"),
    ],
)
def test_bad_pictures_are_skipped_with_a_reason(picture, reason):
    mesh = make_box()
    out, report = P.project_picture(mesh, picture, device="cpu")
    assert out is mesh and not report["applied"] and reason in report["reason"]


def test_bad_meshes_are_skipped_with_a_reason():
    picture = render_box(VIEW, size=96)
    _, report = P.project_picture(trimesh.creation.box(), picture, device="cpu")
    assert not report["applied"] and "no base colour texture" in report["reason"]
    _, report = P.project_picture(object(), picture, device="cpu")
    assert not report["applied"] and "no base colour texture" in report["reason"]
    box = make_box()
    material = SimpleNamespace(baseColorTexture=box.visual.material.baseColorTexture)

    def fake(vertices, uv):
        return SimpleNamespace(vertices=vertices, faces=box.faces, visual=SimpleNamespace(uv=uv, material=material))

    _, report = P.project_picture(fake(box.vertices, box.visual.uv[:5]), picture, device="cpu")
    assert not report["applied"] and "don't match" in report["reason"]
    _, report = P.project_picture(fake(box.vertices * np.nan, box.visual.uv), picture, device="cpu")
    assert not report["applied"] and "non-finite" in report["reason"]


def test_an_internal_failure_is_reported_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(P, "_estimate_pose", boom)
    mesh = make_box()
    out, report = P.project_picture(mesh, render_box(VIEW, size=96), device="cpu")
    assert out is mesh and report["reason"] == "error: RuntimeError: kaboom"
    assert set(P.summary(report)) == {"applied", "reason", "seconds"}

    def fault(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(P, "_estimate_pose", fault)
    _, report = P.project_picture(mesh, render_box(VIEW, size=96), device="cpu")
    assert P.summary(report)["gpu_fault"] is True  # the job finishes; the worker restarts after it


def test_a_misfit_fades_the_picture_out_near_it_but_slivers_dont(monkeypatch):
    picture = render_box(VIEW)
    debug: dict = {}
    _, report = P.project_picture(make_box(), picture, device="cpu", debug=debug)
    assert report["applied"], report["reason"]
    assert float(debug["misfit_fade"].min()) > 0.99  # outlines a pixel apart are no misfit

    # A lump on the picture's outline that the model doesn't have (as if TRELLIS.2 had left off a handle)
    rgba_ = np.asarray(picture).copy()
    size = rgba_.shape[0]
    mid = size // 2
    right = int(np.nonzero(rgba_[mid, :, 3])[0].max())
    lump = (slice(mid - 8, mid + 8), slice(right - 2, right + 12))
    rgba_[lump] = (90, 20, 90, 255)
    monkeypatch.setattr(P, "AMBIGUOUS", -1.0)  # the lump makes another view fit nearly as well; not tested here
    debug = {}
    _, report = P.project_picture(make_box(), Image.fromarray(rgba_), device="cpu", debug=debug)
    assert report["applied"], report["reason"]
    assert angle_to(report["pose"], VIEW) < 6.0, report["pose"]
    fade = debug["misfit_fade"]
    assert float(fade[mid, right + 6]) < 0.05
    assert float(fade[mid, right - 2]) < 0.5  # the model's own surface right next to it, too
    # Well away from it nothing changes
    mask = np.asarray(Image.fromarray(rgba_))[..., 3] > 0
    rows, cols = np.nonzero(mask)
    far = np.hypot(rows - mid, cols - (right + 6)) > (P.MISFIT_FADE + 0.04) * size + 10
    assert float(fade[torch.tensor(rows[far]), torch.tensor(cols[far])].min()) > 0.99


def test_the_texture_keeps_its_alpha_and_mode():
    mesh = make_box()
    texture = mesh.visual.material.baseColorTexture.convert("RGBA")
    alpha = np.asarray(texture)[..., 3].copy()
    alpha[:10] = 77
    texture.putalpha(Image.fromarray(alpha))
    mesh.visual.material.baseColorTexture = texture
    mesh, report = P.project_picture(mesh, render_box(VIEW), device="cpu")
    assert report["applied"]
    out = mesh.visual.material.baseColorTexture
    assert out.mode == "RGBA"
    np.testing.assert_array_equal(np.asarray(out)[..., 3], alpha)


def test_the_gutters_follow_the_new_colours():
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float64)
    mesh, report = P.project_picture(mesh, render_box(VIEW), device="cpu")
    assert report["applied"]
    after = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    # Just outside the front's square, the gutter now continues its edge texels (checked half way
    # along each of the checker's rows, clear of its own edges)
    c0, r0 = cell_origin(4)
    for k in range(3):
        row = round(r0 + (k + 0.5) * INNER / 3)
        edge, gutter = after[row, c0 + INNER - 1], after[row, c0 + INNER + 1]
        assert np.abs(gutter - edge).max() < 35, (row, edge, gutter)
        if k == 0:  # the checker's dark square at the top of that edge, painted on: the gutter followed it
            assert np.abs(gutter - before[row, c0 + INNER + 1]).max() > 40, (row, gutter)
    # Next to the back's square, which didn't change, the gutter is as it was
    c0, r0 = cell_origin(5)
    np.testing.assert_allclose(after[r0 + INNER // 2, c0 + INNER + 1], before[r0 + INNER // 2, c0 + INNER + 1], atol=3)


# --- A cylinder: the colour change fades round its sides --------------------------------------------

RADIUS, HEIGHT = 0.4, 1.0
ORANGE, BLACK = (225, 150, 70), (14, 12, 12)


def make_cylinder(segments: int = 72, rings: int = 12, colour=BLACK, lower=None) -> trimesh.Trimesh:
    """
    The side as one UV chart (u round, v up) over the top half of the atlas; the caps below it. Its
    texture is near-black by default: the model's texture got the colour wrong. With `lower`, the side's
    lower half (and the bottom cap) has that colour instead.
    """
    verts, uvs, faces = [], [], []
    texture = np.full((TEXTURE, TEXTURE, 3), colour, np.uint8)
    if lower is not None:
        texture[round((1 - 0.75) * TEXTURE) :] = lower  # rows below the side's middle, and the caps
        texture[round((1 - 0.48) * TEXTURE) :, : TEXTURE // 2] = colour  # the top cap stays upper
    for j in range(rings + 1):
        for i in range(segments + 1):  # the seam column is duplicated, as UV unwrapping does
            phi = 2 * math.pi * i / segments
            verts.append((RADIUS * math.sin(phi), HEIGHT * (j / rings - 0.5), RADIUS * math.cos(phi)))
            uvs.append((0.02 + 0.96 * i / segments, 0.52 + 0.46 * j / rings))
    for j in range(rings):
        for i in range(segments):
            q = j * (segments + 1) + i
            faces += [[q, q + 1, q + segments + 2], [q, q + segments + 2, q + segments + 1]]
    for cap, (y, sign) in enumerate(((HEIGHT / 2, 1), (-HEIGHT / 2, -1))):
        centre = len(verts)
        cu = 0.25 + 0.5 * cap
        verts.append((0.0, y, 0.0))
        uvs.append((cu, 0.25))
        for i in range(segments):
            phi = 2 * math.pi * i / segments
            verts.append((RADIUS * math.sin(phi), y, RADIUS * math.cos(phi)))
            uvs.append((cu + 0.2 * math.sin(phi), 0.25 + 0.2 * math.cos(phi)))
        for i in range(segments):
            a, b = centre + 1 + i, centre + 1 + (i + 1) % segments
            faces.append([centre, a, b] if sign > 0 else [centre, b, a])
    return textured(verts, faces, uvs, texture)


def render_cylinder(
    view: P.View, size: int = 256, fill: float = 0.7, stripes: bool = False, upper=ORANGE, lower=None
) -> Image.Image:
    """
    Ray-cast the cylinder, `upper` (orange) all over, or `lower` below its middle; with `stripes`, black
    and white bands up its picture instead.
    """
    radius = math.hypot(RADIUS, HEIGHT / 2)
    r, h = RADIUS / radius, HEIGHT / 2 / radius
    ring = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    corners = np.array([(r * np.sin(a), y, r * np.cos(a)) for a in ring for y in (-h, h)])
    origin, direction = camera_rays(view, corners, size, fill)
    ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    a = dx**2 + dz**2
    b = 2 * (ox * dx + oz * dz)
    c = ox**2 + oz**2 - r**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_side = (-b - np.sqrt(disc)) / (2 * a)
        side_hit = (disc >= 0) & (np.abs(oy + t_side * dy) <= h)
        t_caps = [(y - oy) / dy for y in (h, -h)]
        cap_hit = [(np.hypot(ox + t * dx, oz + t * dz) <= r) & (t > 0) for t in t_caps]
    hit = side_hit | cap_hit[0] | cap_hit[1]
    colour = np.broadcast_to(np.array(upper, np.float64), (size, size, 3))
    if lower is not None:
        with np.errstate(invalid="ignore"):
            y = np.where(side_hit, oy + t_side * dy, np.where(cap_hit[0], h, -h))
        colour = np.where((y < 0)[..., None], np.array(lower, np.float64), colour)
    if stripes:
        band = (np.arange(size) // (size // 24)) % 2 == 0
        colour = np.where(band[None, :, None], (240, 240, 240), BLACK) * np.ones((size, 1, 1))
    return rgba(np.where(hit[..., None], colour, (0, 0, 0)), hit)


def side_profile(texture: np.ndarray, segments: int = 72, height: float = 0.5) -> np.ndarray:
    """The side's colour `height` of the way up, at each of its segments (texel columns of its chart)."""
    row = round((1 - (0.52 + 0.46 * height)) * TEXTURE)
    cols = [round((0.02 + 0.96 * (i + 0.5) / segments) * TEXTURE) for i in range(segments)]
    return texture[row, cols].astype(np.float64)


def round_the_side(texture: np.ndarray, azimuth: float, height: float = 0.5):
    """The side's colour `height` of the way up (segments, 3), and each segment's degrees from the camera."""
    profile = side_profile(texture, height=height)
    phi = (np.arange(72) + 0.5) * 5.0
    return profile, np.abs((phi - azimuth + 180) % 360 - 180)


def test_a_paint_the_picture_shows_in_another_colour_changes_all_round():
    # The model's texture is near-black (the "cola" milk tea); the picture shows the cup orange. The cup
    # looks the same from every side, so any camera round it will do
    view = P.View(azimuth=0.0, elevation=15.0, roll=0.0, perspective=0.25)
    mesh, report = P.project_picture(make_cylinder(), render_cylinder(view), device="cpu")
    assert report["applied"], report["reason"]
    assert report["exposure"]["note"].startswith("kept")  # far too dark to be the same colour, dimmer
    assert all(r["shape_difference"] < P.SAME_SHAPE for r in report["rivals"])
    assert abs(report["pose"]["elevation"] - view.elevation) < 5
    assert report["mode"] == "recolour and detail"
    assert [p["decision"] for p in report["paints"] if p["share"] > 0.5] == ["recoloured"]

    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    for height in (0.2, 0.5, 0.8):
        profile, away = round_the_side(texture, report["pose"]["azimuth"], height)
        # Orange all round, the far side included, with no step where the picture stopped seeing it
        assert np.abs(profile - np.array(ORANGE)).max() < 12, (height, np.abs(profile - np.array(ORANGE)).max())
        assert np.abs(np.diff(profile[np.argsort(away)], axis=0)).max() < 6


def test_a_paint_the_picture_shows_in_several_colours_keeps_its_own():
    # The model's orange is right; the picture shows black and white stripes up the side it sees. That is
    # no one colour for the paint: it keeps its own all round, and the stripes, which run on past the
    # picture's outline, aren't painted on either
    view = P.View(azimuth=0.0, elevation=15.0, roll=0.0, perspective=0.25)
    picture = render_cylinder(view, stripes=True)
    mesh, report = P.project_picture(make_cylinder(colour=ORANGE), picture, device="cpu")
    assert report["applied"], report["reason"]
    assert report["mode"] == "detail only"
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    profile, away = round_the_side(texture, report["pose"]["azimuth"])
    assert np.abs(profile[away > 70] - np.array(ORANGE)).max() < 8
    assert np.median(np.abs(profile[away < 40] - np.array(ORANGE)).max(axis=1)) < 12


def test_each_paint_changes_on_its_own(monkeypatch):
    # The model's top half is orange and its bottom half blue. The picture shows the top green and the
    # bottom blue, as the model has it: the top turns green all round, the blue stays blue
    blue, green = (40, 90, 200), (60, 170, 70)
    view = P.View(azimuth=0.0, elevation=15.0, roll=0.0, perspective=0.25)

    def sides() -> tuple[np.ndarray, np.ndarray, dict]:
        model = make_cylinder(colour=ORANGE, lower=blue)
        mesh, report = P.project_picture(model, render_cylinder(view, upper=green, lower=blue), device="cpu")
        assert report["applied"], report["reason"]
        texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
        top, _ = round_the_side(texture, report["pose"]["azimuth"], height=0.85)
        bottom, _ = round_the_side(texture, report["pose"]["azimuth"], height=0.3)
        return top, bottom, report

    top, bottom, report = sides()
    assert np.abs(top - np.array(green)).max() < 15, np.abs(top - np.array(green)).max()
    assert np.abs(bottom - np.array(blue)).max() < 8
    # One paint for the whole model would show two colours in the picture: then nothing changes
    monkeypatch.setattr(P, "PAINTS", 1)
    top, bottom, report = sides()
    assert report["mode"] == "detail only"
    assert np.abs(top - np.array(ORANGE)).max() < 12 and np.abs(bottom - np.array(blue)).max() < 12


def glow(picture: Image.Image, centre: tuple[float, float], radius: float, colour, strength: float) -> Image.Image:
    """Add a soft round glow of light (linear) to a picture: the burner under a balloon."""
    rgba_ = np.asarray(picture).astype(np.float64)
    size = rgba_.shape[0]
    yy, xx = np.mgrid[:size, :size] / size
    g = np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * radius**2))[..., None]
    lit = to_linear(rgba_[..., :3]) + strength * g * (np.array(colour) / 255)
    rgba_[..., :3] = to_srgb(lit)
    return Image.fromarray(rgba_.astype(np.uint8), "RGBA")


def test_glow_and_shade_are_left_out():
    # The picture shows the model's own colours, with a yellow glow on the front and a broad shadow over
    # the right side: light, not paint. Nothing should change
    picture = render_box(VIEW, pattern=False)
    picture = glow(picture, (0.42, 0.62), 0.12, (255, 190, 60), 0.6)
    shade = np.asarray(picture).astype(np.float64)
    size = shade.shape[0]
    xx = np.arange(size)[None, :, None] / size
    shade[..., :3] = to_srgb(to_linear(shade[..., :3]) * (1 - 0.5 * np.clip((xx - 0.55) / 0.2, 0, 1)))
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float64)
    mesh, report = P.project_picture(mesh, Image.fromarray(shade.astype(np.uint8), "RGBA"), device="cpu")
    assert report["applied"], report["reason"]
    after = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    for side in (0, 2, 4):
        _, tx = texel_points(side=side, inset=0.1)
        change = np.abs(after[tx[:, 0], tx[:, 1]] - before[tx[:, 0], tx[:, 1]]).max(axis=1)
        assert np.percentile(change, 95) < 12, (side, np.percentile(change, 95))


def test_a_design_is_painted_on_whole_but_a_shadow_is_not():
    # Two small squares inside the front in the picture: one green (a design), one the front's own
    # colour, half as bright (a shadow cast on it). Only the design goes on
    picture = np.asarray(render_box(VIEW, pattern=False)).copy()
    front = np.all(np.abs(picture[..., :3].astype(int) - np.array(SIDES[4][2])) < 3, axis=-1)
    rows, cols = np.nonzero(front)
    r, c, half = int(rows.mean()), int(cols.mean()), 9
    design = (slice(r - half, r + half), slice(c - 3 * half, c - half))
    shadow = (slice(r - half, r + half), slice(c + half, c + 3 * half))
    assert front[design].all() and front[shadow].all()
    picture[design + (slice(0, 3),)] = (40, 160, 70)
    picture[shadow + (slice(0, 3),)] = to_srgb(to_linear(np.array(SIDES[4][2], np.float64)) * 0.5)
    mesh = make_box()
    before = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float64)
    debug: dict = {}
    mesh, report = P.project_picture(mesh, Image.fromarray(picture), device="cpu", debug=debug)
    assert report["applied"], report["reason"]
    assert report["decals"]["count"] >= 1 and report["decals"]["only_light"] >= 1, report["decals"]
    after = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    # The texels whose points land well inside each square
    xy, flat, seen = debug["xy"].numpy(), debug["flat"].numpy(), debug["visible"].numpy()
    for (rs, cs), green in ((design, True), (shadow, False)):
        hit = seen & (xy[:, 1] >= rs.start + 3) & (xy[:, 1] < rs.stop - 3)
        hit &= (xy[:, 0] >= cs.start + 3) & (xy[:, 0] < cs.stop - 3)
        assert hit.sum() > 20
        texel_rows, texel_cols = np.divmod(flat[hit], TEXTURE)
        got = after[texel_rows, texel_cols]
        if green:
            assert np.median(np.abs(got - np.array([40, 160, 70])).max(axis=1)) < 20
        else:
            assert np.median(np.abs(got - before[texel_rows, texel_cols]).max(axis=1)) < 10


def test_rasterizer_depth_is_perspective_correct_and_nearest_wins():
    # A tilted quad seen in perspective: every pixel's depth must lie on the quad's plane
    points = torch.tensor([[-0.5, -0.5, 0.3], [0.5, -0.5, -0.3], [0.5, 0.5, -0.3], [-0.5, 0.5, 0.3]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    params = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]])
    x, y, depth, s = P.project(points, params)
    size = 64
    xy = torch.stack([(x[0] + 1) / 2 * size, (1 - y[0]) / 2 * size], -1)
    zbuf = P.rasterize_depth(xy, depth[0], s[0], faces, size, size)
    inside = torch.isfinite(zbuf)
    assert int(inside.sum()) > 1000
    # Unproject each covered pixel with its depth: the camera at z = 2 looks down -z
    rows, cols = torch.nonzero(inside, as_tuple=True)
    xi = (cols.float() + 0.5) / size * 2 - 1
    zc = -zbuf[inside]  # depth is -z
    px = xi * (1 - 0.5 * zc)  # x_image = x / (1 - k z)
    torch.testing.assert_close(zc, -0.6 * px, atol=2e-3, rtol=0)  # the quad's plane: z = -0.6 x

    # A nearer square over the middle hides the first there
    near = torch.tensor([[-0.2, -0.2, 0.6], [0.2, -0.2, 0.6], [0.2, 0.2, 0.6], [-0.2, 0.2, 0.6]])
    x, y, depth, s = P.project(torch.cat([points, near]), params)
    xy = torch.stack([(x[0] + 1) / 2 * size, (1 - y[0]) / 2 * size], -1)
    zbuf2 = P.rasterize_depth(xy, depth[0], s[0], torch.cat([faces, faces + 4]), size, size)
    assert float(zbuf2[size // 2, size // 2]) == pytest.approx(-0.6, abs=1e-4)
    # ... and small batches give the same picture
    small = P.rasterize_depth(xy, depth[0], s[0], torch.cat([faces, faces + 4]), size, size, max_candidates=64)
    torch.testing.assert_close(small, zbuf2)


def test_push_pull_keeps_known_values_and_fills_the_rest_smoothly():
    values = torch.zeros((1, 32, 32))
    known = torch.zeros((32, 32))
    values[0, 8:16, 8:16] = 1.0
    known[8:16, 8:16] = 1
    known[20:30, 20:30] = 1  # known zeros
    filled = P._push_pull(values, known)
    torch.testing.assert_close(filled[0][known > 0], values[0][known > 0])
    assert float(filled[0, 12, 18]) > float(filled[0, 12, 28]) > -1e-6  # decays away from the ones
    assert bool(torch.isfinite(filled).all())


def test_views_report_their_field_of_view():
    view = P.View(azimuth=370.0, elevation=12.0, roll=-3.0, perspective=math.sin(math.radians(15)))
    d = view.as_dict()
    assert d["azimuth"] == 10.0 and d["fov"] == pytest.approx(30.0, abs=0.1) and d["scale"] == 1.0


def test_paints_follow_colour_more_than_shade_and_small_parts_get_their_own():
    white, grey, blue, salmon = [0.95, 0.95, 0.95], [0.6, 0.6, 0.62], [0.05, 0.47, 0.61], [0.81, 0.43, 0.33]
    # A bowl's white and its shaded grey underside, a car's blue, and a small salmon part
    colours = torch.tensor([white] * 3000 + [grey] * 3000 + [blue] * 3000 + [salmon] * 30)
    centres = P._paints(colours, 4)
    m = P._membership(torch.tensor([white, grey, blue, salmon]), centres)
    shared = m @ m.T  # how much two colours share their paints
    assert shared[0, 1] > 0.4  # white and grey: largely the same paint
    assert shared[0, 2] < 0.01 and shared[1, 2] < 0.01  # not the blue
    assert float(m[3].max()) > 0.9  # 1% of the texels, but a paint of its own


def test_connected_regions_match_a_flood_fill():
    gen = torch.Generator().manual_seed(3)
    labels = torch.randint(0, 3, (40, 50), generator=gen)
    labels[torch.rand((40, 50), generator=gen) < 0.1] = -1
    ids = P._components(labels).numpy()
    grid = labels.numpy()
    expected = np.full(grid.shape, -1)
    count = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] < 0 or expected[r, c] >= 0:
                continue
            stack = [(r, c)]
            expected[r, c] = count
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y, x = a + da, b + db
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and expected[y, x] < 0 and grid[y, x] == grid[a, b]:
                        expected[y, x] = count
                        stack.append((y, x))
            count += 1
    assert ids.max() + 1 == count
    assert (ids < 0).sum() == (grid < 0).sum()
    # The same partition: each expected region is one id and each id one region
    pairs = set(zip(expected[grid >= 0].tolist(), ids[grid >= 0].tolist()))
    assert len(pairs) == count
