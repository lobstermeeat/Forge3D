"""
Glass told apart from spurious alpha (glass.py), on flat squares textured like the texture pass's raw RGBA:
glass (low alpha on a glossy surface, in a region) exported as dark, glossy glass; spurious alpha (on matte
surfaces, or in specks and thin edges) divided out exactly as before. Then the runtime's export around it.
"""

import types

import numpy as np
import pytest
import trimesh
from PIL import Image

from forge3d_worker import glass, pipeline, service
from forge3d_worker.pipeline import ALPHA_FLOOR, Trellis2Runtime, unpremultiply
from forge3d_worker.settings import PRESETS

VOXEL = 1 / 1024  # a final's voxel
SIZE = 64  # texels on a side of the test textures


def srgb(linear):
    """8-bit sRGB of a linear value."""
    return int(round(float(glass.linear_to_srgb(np.float64(linear))) * 255))


def square(voxels=64, uv_extent=1.0, cells=32):
    """
    A flat square ``voxels`` voxels across (z = 0), its UVs spanning [0, uv_extent]^2 (trimesh: v up): with
    the defaults, one texel of a SIZE texture is one voxel. Rows of the image run top-down (v = 1 first).
    """
    side = voxels * VOXEL
    xs = np.linspace(0.0, side, cells + 1)
    x, y = np.meshgrid(xs, xs)
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    uv = vertices[:, :2] / side * uv_extent
    n = cells + 1
    faces = []
    for i in range(cells):
        for j in range(cells):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.array(faces), uv


def textured(rgba, roughness, metallic=0, **shape):
    """The square with to_glb's material: RGBA base colour, metallic-roughness (G roughness, B metallic)."""
    vertices, faces, uv = square(**shape)
    mr = np.zeros(rgba.shape[:2] + (3,), np.uint8)
    mr[..., 1], mr[..., 2] = roughness, metallic
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.asarray(rgba, np.uint8), "RGBA"),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(mr, "RGB"),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def body(colour=(40, 90, 170), roughness=128):
    """An opaque, fairly glossy body (a car's paint): RGBA and roughness images."""
    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = colour, 255
    return rgba, np.full((SIZE, SIZE), roughness, np.uint8)


WINDOW = (slice(20, 44), slice(20, 44))  # 24 x 24 texels: 24 voxels across, in the middle


def window(rgba, rough, colour, alpha, roughness=10, where=WINDOW):
    rgba, rough = rgba.copy(), rough.copy()
    rgba[where + (slice(0, 3),)] = colour
    rgba[where + (3,)] = alpha
    rough[where] = roughness
    return rgba, rough


def exported(mesh):
    """What export_textures makes of the mesh: (report, RGB array, metallic-roughness array)."""
    report = glass.export_textures(mesh, VOXEL, device="cpu")
    material = mesh.visual.material
    return report, np.asarray(material.baseColorTexture), np.asarray(material.metallicRoughnessTexture)


def luma(rgb8):
    return glass.srgb_to_linear(np.asarray(rgb8, np.float64) / 255) @ glass.LUMA


# --- What is glass, and what it becomes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "colour, alpha",
    [
        ((196, 196, 196), 51),  # straight colour: linear 0.55 at alpha 0.2 (the cartoon car's windows)
        ((24, 26, 22), 100),  # premultiplied: dark at alpha 0.4 (the BMW's), which division made pale too
    ],
)
def test_glass_on_a_glossy_surface_becomes_dark_glossy_glass(colour, alpha):
    rgba, rough = window(*body(), colour, alpha)
    original = Image.fromarray(rgba, "RGBA")
    report, rgb, mr = exported(textured(rgba, rough, metallic=120))

    inside = rgb[WINDOW].reshape(-1, 3)
    assert np.allclose(luma(inside), glass.GLASS_LUMA, rtol=0.05)  # one dark luminance, not the division's
    assert (mr[WINDOW][..., 1] <= round(glass.GLASS_ROUGHNESS * 255) + 1).all()  # glossy
    assert (mr[WINDOW][..., 2] == 0).all()  # not metal
    # The body is exactly what the division makes of it, and keeps its roughness and metal
    divided = np.asarray(unpremultiply(original))
    outside = np.ones((SIZE, SIZE), bool)
    outside[16:48, 16:48] = False
    assert np.array_equal(rgb[outside], divided[outside])
    assert (mr[outside][:, 1] == 128).all() and (mr[outside][:, 2] == 120).all()
    assert report == {"glass": pytest.approx(24 * 24 / SIZE**2, rel=0.15)}


def test_what_dividing_by_alpha_did_to_glass():
    """The founder's bug, for the record: light glass at alpha 0.2 divided by its alpha comes out white."""
    rgba, _ = window(*body(), (196, 196, 196), 51)
    divided = np.asarray(unpremultiply(Image.fromarray(rgba, "RGBA")))
    assert (divided[WINDOW] == 255).all()


def test_spurious_alpha_on_a_matte_surface_is_divided_out_as_before():
    # Phase 2's dragon: alpha 0.3-0.7 over almost all of it, its colour darkened by the same factor, matte
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.3, 0.7, (SIZE, SIZE))
    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    paint = np.array([0.05, 0.3, 0.04])  # linear green
    rgba[..., :3] = np.round(glass.linear_to_srgb(paint * alpha[..., None]) * 255)
    rgba[..., 3] = np.round(alpha * 255)
    rough = np.full((SIZE, SIZE), 245, np.uint8)
    original = Image.fromarray(rgba, "RGBA")
    report, rgb, mr = exported(textured(rgba, rough))
    assert report == {"glass": 0.0}
    assert np.array_equal(rgb, np.asarray(unpremultiply(original)))
    assert (mr[..., 1] == 245).all() and (mr[..., 2] == 0).all()
    # ... which gives the paint back: no blotches
    assert np.allclose(glass.srgb_to_linear(rgb / 255), paint, atol=0.01)


def test_a_thin_glossy_edge_of_low_alpha_is_left_to_the_division():
    # A later dragon: low alpha along the edges of thin parts, a few voxels wide; here 2 voxels, glossy
    edge = (slice(10, 54), slice(30, 32))
    rgba, rough = window(*body(), (30, 60, 110), 110, roughness=60, where=edge)
    original = Image.fromarray(rgba, "RGBA")
    report, rgb, _ = exported(textured(rgba, rough))
    assert report == {"glass": 0.0}
    assert np.array_equal(rgb, np.asarray(unpremultiply(original)))


def test_matte_glass_like_alpha_stays_divided_glossy_matte_in_between_is_partly_glass():
    weight = glass.texel_weight(np.array([0.2, 0.2, 0.2, 0.2, 0.7, 0.95]), np.array([0.0, 0.6, 0.8, 0.95, 0.0, 0.0]))
    assert weight[0] == weight[1] == 1.0  # glossy (TRELLIS.2's glass: 0.0-0.55)
    assert 0 < weight[2] < 1  # one BMW texture's windows sat at 0.6-0.85
    assert weight[3] == 0.0  # the dragons' skin: 0.85 and more
    assert 0 < weight[4] < 1 and weight[5] == 0.0  # alpha near 1 is the division's


def test_a_window_is_glass_right_to_its_border():
    """A window's edge texels, whose surroundings are half body, are as much glass as its middle (no rim)."""
    rgba, rough = window(*body(), (196, 196, 196), 51)
    _, rgb, _ = exported(textured(rgba, rough))
    rows, cols = WINDOW
    top, bottom, left, right = rows.start, rows.stop - 1, cols.start, cols.stop - 1
    border = np.concatenate([rgb[top, cols], rgb[bottom, cols], rgb[rows, left], rgb[rows, right]])
    assert np.allclose(luma(border), glass.GLASS_LUMA, rtol=0.05)


def test_glass_keeps_its_hue_where_it_has_one():
    rgba, rough = window(*body(), (200, 30, 20), 25)  # a red lamp cover
    _, rgb, _ = exported(textured(rgba, rough))
    lamp = rgb[WINDOW].reshape(-1, 3).astype(int)
    assert (lamp[:, 0] > 2 * lamp[:, 1]).all() and (lamp[:, 0] > 2 * lamp[:, 2]).all()
    assert np.allclose(luma(lamp), glass.GLASS_LUMA, rtol=0.1)
    # Glass too dark to have a hue (noise) is neutral
    rgba, rough = window(*body(), (3, 0, 9), 25)
    _, rgb, _ = exported(textured(rgba, rough))
    dark = rgb[WINDOW].reshape(-1, 3).astype(int)
    assert (dark.max(axis=1) - dark.min(axis=1) <= 2).all()


def test_without_low_alpha_nothing_is_rasterized(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("rasterized a texture with no glass")

    monkeypatch.setattr(glass, "faces_at_texels", unexpected)
    rgba, rough = body()
    rgba[5:9, 5:9, 3] = 240  # alpha 0.94: the division's
    original = Image.fromarray(rgba, "RGBA")
    report, rgb, mr = exported(textured(rgba, rough))
    assert report == {"glass": 0.0} and np.array_equal(rgb, np.asarray(unpremultiply(original)))
    # Low alpha but matte: no texel could be glass, nothing rasterized either
    rgba, rough = window(*body(), (196, 196, 196), 51, roughness=250)
    report, rgb, _ = exported(textured(rgba, rough))
    assert report == {"glass": 0.0}


def test_an_rgb_texture_is_left_as_it_is():
    rgba, rough = body()
    mesh = textured(rgba, rough)
    mesh.visual.material.baseColorTexture = Image.fromarray(rgba[..., :3], "RGB")
    assert glass.export_textures(mesh, VOXEL, device="cpu") == {"glass": 0.0}
    assert np.array_equal(np.asarray(mesh.visual.material.baseColorTexture), rgba[..., :3])


def test_a_metallic_roughness_texture_of_another_size_gets_glass_at_its_size():
    rgba, rough = window(*body(), (196, 196, 196), 51)
    mesh = textured(rgba, rough)
    small = np.asarray(mesh.visual.material.metallicRoughnessTexture)[::4, ::4]
    mesh.visual.material.metallicRoughnessTexture = Image.fromarray(small.copy(), "RGB")
    _, _, mr = exported(mesh)
    assert mr.shape == (SIZE // 4, SIZE // 4, 3)
    assert mr[8, 8, 1] <= round(glass.GLASS_ROUGHNESS * 255) + 1 and mr[1, 1, 1] == 128


def test_gutters_beside_glass_are_glass_too():
    """Texels no triangle covers take their chart's weight: filtering at a chart's edge brings no pale glass back."""
    # The square's UVs cover the texture's bottom-left quarter (rows 32-63, columns 0-31): 2 voxels a texel.
    # Its window runs to the chart's right edge, and to_glb's inpainting carried it into the gutter beside
    rgba, rough = body()
    rgba, rough = window(rgba, rough, (196, 196, 196), 51, where=(slice(40, 56), slice(16, 36)))
    report, rgb, mr = exported(textured(rgba, rough, uv_extent=0.5))
    gutter = rgb[42:54, 32:36].reshape(-1, 3)
    assert np.allclose(luma(gutter), glass.GLASS_LUMA, rtol=0.05)
    assert (mr[42:54, 32:36, 1] <= round(glass.GLASS_ROUGHNESS * 255) + 1).all()
    # The share counts the texels a triangle covers: 16 x 16 of the chart's 32 x 32
    assert report == {"glass": pytest.approx(0.25, rel=0.15)}


def test_fill_spreads_known_values_into_the_unknown():
    known = np.zeros((8, 8), bool)
    known[:4, :4] = True
    assert np.allclose(glass.fill(np.where(known, 1.0, 0.0), known), 1.0)
    values = np.zeros((8, 8))
    values[:4, 2:4] = 1.0
    filled = glass.fill(values, known)
    assert np.array_equal(filled[:4, :4], values[:4, :4])  # known values are kept
    assert filled[1, 5] > filled[1, 0] and 0.0 <= filled.min() and filled.max() <= 1.0


def test_faces_at_texels_follow_the_image_rows():
    # Two triangles: the top half of the UV square (v > 0.5) and the bottom half
    uv = np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1], [0, 0], [1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 1], [4, 1, 0]])
    face = glass.faces_at_texels(uv, faces, (8, 8), device="cpu")
    assert set(face[:4].ravel()) <= {0, 1} and set(face[4:].ravel()) <= {2, 3}  # row 0 is the image's top
    assert (face >= 0).all()


def test_region_share_is_low_for_a_speck_and_full_across_a_window():
    vertices, faces, _ = square()
    centroid = vertices[faces].mean(axis=1)[:, :2] / VOXEL  # in the square's plane, in voxels: 0-64
    speck = np.where(np.linalg.norm(centroid - 32, axis=1) < 2, 1.0, 0.0)
    assert speck.sum() > 0
    assert glass.region_share(vertices, faces, speck, VOXEL).max() < glass.REGION[0]
    pane = np.where((np.abs(centroid - 32) < 12).all(axis=1), 1.0, 0.0)
    share = glass.region_share(vertices, faces, pane, VOXEL)
    assert (share[pane == 1] >= glass.REGION[1]).all()


# --- The runtime ------------------------------------------------------------------------------------------------


def runtime_with(glb) -> Trellis2Runtime:
    runtime = Trellis2Runtime.__new__(Trellis2Runtime)
    runtime.pipeline = types.SimpleNamespace(low_vram=False)
    runtime.low_vram_configured = False
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: glb))
    return runtime


def generated():
    """A generated mesh as export() sees it (no picture: nothing is projected)."""
    return types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=VOXEL)


def test_export_makes_glass_and_says_so(capsys):
    rgba, rough = window(*body(), (196, 196, 196), 51)
    glb = textured(rgba, rough)
    runtime = runtime_with(glb)
    data, triangles = runtime.export(generated(), PRESETS["final"])
    assert data[:4] == b"glTF" and triangles == len(glb.faces)
    assert runtime.last_glass == {"glass": pytest.approx(24 * 24 / SIZE**2, rel=0.15)}
    assert '[forge3d] glass: {"glass": ' in capsys.readouterr().out
    exported_mesh = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb", force="mesh")
    colour = np.asarray(exported_mesh.visual.material.baseColorTexture.convert("RGB"))
    assert np.allclose(luma(colour[WINDOW].reshape(-1, 3)), glass.GLASS_LUMA, rtol=0.05)


def test_with_glass_off_the_colour_is_only_divided_as_before():
    rgba, rough = window(*body(), (196, 196, 196), 51)
    glb = textured(rgba, rough)
    runtime = runtime_with(glb)
    runtime.glass_textures = False
    runtime.export(generated(), PRESETS["final"])
    assert runtime.last_glass is None
    divided = np.asarray(unpremultiply(Image.fromarray(rgba, "RGBA")))
    assert np.array_equal(np.asarray(glb.visual.material.baseColorTexture), divided)
    assert (np.asarray(glb.visual.material.metallicRoughnessTexture)[..., 1] == rough).all()


def test_production_makes_glass():
    assert Trellis2Runtime.glass_textures is True


def test_when_telling_glass_apart_fails_the_colour_is_divided_and_the_export_goes_on(monkeypatch, capsys):
    def broken(mesh, voxel_size, device=None):
        raise RuntimeError("uv_raster\nfailed")

    monkeypatch.setattr(glass, "export_textures", broken)
    rgba, rough = window(*body(), (196, 196, 196), 51)
    glb = textured(rgba, rough)
    runtime = runtime_with(glb)
    data, _ = runtime.export(generated(), PRESETS["final"])
    assert data[:4] == b"glTF"
    assert runtime.last_glass == {"error": "RuntimeError: uv_raster failed"}
    divided = np.asarray(unpremultiply(Image.fromarray(rgba, "RGBA")))
    assert np.array_equal(np.asarray(glb.visual.material.baseColorTexture), divided)
    assert '[forge3d] glass: {"error": "RuntimeError: uv_raster failed"}' in capsys.readouterr().out


def test_previews_get_glass_too():
    rgba, rough = window(*body(), (196, 196, 196), 51)
    glb = textured(rgba, rough)
    runtime = runtime_with(glb)
    runtime.export(generated(), PRESETS["preview"])
    assert runtime.last_glass["glass"] > 0


def test_the_division_is_still_the_pipelines():
    # Imported from the pipeline as it always was (Phase 4's tests and experiments use it from there)
    assert pipeline.unpremultiply is unpremultiply is glass.unpremultiply and ALPHA_FLOOR == 0.25


def test_results_say_how_much_was_glass():
    class Runtime:
        last_projection = None
        last_cleanup = None
        last_glass = {"glass": 0.041}

    assert service._export_notes(Runtime()) == {"glass": {"glass": 0.041}}
    Runtime.last_glass = None
    assert service._export_notes(Runtime()) == {}
