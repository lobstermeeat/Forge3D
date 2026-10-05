"""split_glass on synthetic cars: a hollow box (a cabin) or a solid one, with a window of TRELLIS.2-like glass."""

import io
import json
import shutil
import struct

import numpy as np
import pytest
from PIL import Image

from forge3d_worker import compress, glass

trimesh = pytest.importorskip("trimesh")

VOXEL = 0.01  # the grid spacing the regions are measured in
CELLS = 6  # quads along each side of a box
CELL_PX = 12  # texture pixels per quad's UV cell
WALL = 3 * VOXEL  # the body's thickness, as thin as TRELLIS.2's glass slabs (about 2.3 voxels)
BODY = {"rgb": (40, 70, 160), "alpha": 1.0, "rough": 0.6, "metal": 0.2}
WINDOW = {"rgb": (14, 22, 21), "alpha": 0.3, "rough": 0.1, "metal": 0.5}


def box_quads(half, inward=False):
    """The six sides of a cube of half-size ``half``, each cut into CELLS x CELLS quads: (side, i, j, corners)."""
    quads = []
    for axis in range(3):
        for sign in (1, -1):
            e = np.eye(3)
            t1, t2 = (e[(axis + 1) % 3], e[(axis + 2) % 3]) if sign > 0 else (e[(axis + 2) % 3], e[(axis + 1) % 3])
            edges = -half + 2 * half * np.arange(CELLS + 1) / CELLS
            for i in range(CELLS):
                for j in range(CELLS):
                    u0, u1, v0, v1 = edges[i], edges[i + 1], edges[j], edges[j + 1]
                    corners = [sign * half * e[axis] + u * t1 + v * t2 for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
                    if inward:
                        corners = corners[::-1]
                    quads.append(((axis, sign), i, j, np.array(corners)))
    return quads


def build(quads, paint):
    """
    What to_glb returns, unpremultiplied: a textured mesh with a UV cell of its own per quad, and its original
    RGBA base colour. ``paint(quad)`` gives each quad's straight colour, alpha, roughness and metallic.
    """
    grid = int(np.ceil(np.sqrt(len(quads))))
    size = grid * CELL_PX
    # Texels no quad uses are the body's, as to_glb's inpainting fills its gutters from the charts beside them
    rgba = np.zeros((size, size, 4), np.uint8) + np.array([*BODY["rgb"], 255], np.uint8)
    mr = np.zeros((size, size, 3), np.uint8) + np.array([0, round(255 * BODY["rough"]), round(255 * BODY["metal"])], np.uint8)
    vertices, uvs, faces = [], [], []
    for n, quad in enumerate(quads):
        gx, gy = n % grid, n // grid
        x0, y0 = gx * CELL_PX, gy * CELL_PX  # the cell's top-left texel (row 0 is v = 1)
        p = paint(quad)
        rgba[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX] = [*p["rgb"], round(255 * p["alpha"])]
        mr[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX] = [0, round(255 * p["rough"]), round(255 * p["metal"])]
        # The quad's UVs: its cell, 2 texels in from the edges
        u0, u1 = (x0 + 2) / size, (x0 + CELL_PX - 2) / size
        v1, v0 = 1 - (y0 + 2) / size, 1 - (y0 + CELL_PX - 2) / size
        base = len(vertices)
        vertices.extend(quad[3])
        uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    original = Image.fromarray(rgba, "RGBA")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=original.convert("RGB"),  # unpremultiply drops the alpha
        baseColorFactor=np.array([255, 255, 255, 255], np.uint8),
        metallicRoughnessTexture=Image.fromarray(mr, "RGB"),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=False,
    )
    vertices, faces = np.array(vertices), np.array(faces)
    # Stand-ins for the shading normals: each quad has vertices of its own, which take its normal
    normals = np.zeros_like(vertices)
    for k in range(3):
        normals[faces[:, k]] = trimesh.Trimesh(vertices, faces, process=False).face_normals
    visual = trimesh.visual.TextureVisuals(uv=np.array(uvs), material=material)
    return trimesh.Trimesh(vertices, faces, vertex_normals=normals, visual=visual, process=False), original


def is_window(quad, sides=((0, 1),)):
    """The middle 2 x 2 quads of a side (+x by default)."""
    side, i, j, _ = quad
    return side in sides and i in (2, 3) and j in (2, 3)


def car(window=WINDOW, hollow=True, sides=((0, 1),)):
    """
    A box with a window (on +x unless ``sides`` says otherwise). Hollow: the body is a shell WALL thick around
    a cabin, the window a slab of glass through it (its outer and inner surfaces both glass), as TRELLIS.2
    builds a car. Solid: the outer surface alone, nothing behind the window.
    """
    quads = box_quads(1.0) + (box_quads(1.0 - WALL, inward=True) if hollow else [])
    return build(quads, lambda quad: window if is_window(quad, sides) else BODY)


def window_faces(mesh, sides=((0, 1),)):
    """The faces of the windows (both surfaces of each slab): the middle third of their sides, by face centre."""
    centres = mesh.vertices[mesh.faces].mean(axis=1)
    keep = np.zeros(len(centres), bool)
    for axis, sign in sides:
        others = [k for k in range(3) if k != axis]
        on_side = sign * centres[:, axis] > 1 - 2 * WALL
        keep |= on_side & np.all(np.abs(centres[:, others]) < 1 / 3, axis=1)
    return keep


def centres(mesh, faces=None):
    faces = mesh.faces if faces is None else faces
    return {tuple(np.round(c, 6)) for c in mesh.vertices[faces].mean(axis=1)}


def corners(mesh, faces=None):
    """Every face corner's position and vertex normal, sorted."""
    faces = mesh.faces if faces is None else faces
    index = np.asarray(faces).reshape(-1)
    pairs = np.round(np.concatenate([mesh.vertices[index], mesh.vertex_normals[index]], 1), 6) + 0.0
    return sorted(map(tuple, pairs))


def read_glb(data: bytes):
    """The JSON and binary chunks of a GLB."""
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20 : 20 + length]), data[20 + length + 8 :]


def test_a_window_over_a_cabin_becomes_see_through_glass():
    mesh, original = car()
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert isinstance(model, trimesh.Scene)
    assert report["applied"] and report["reason"] == "see-through glass"
    assert report["see_through"]["regions"] == 2  # the slab's outer and inner surfaces
    assert report["opaque"]["regions"] == 0
    # Rays from the outer surface found the cabin; the inner surface's crossed the slab and left
    assert report["rays"]["cavity"] > 0 and report["rays"]["exit"] > 0 and report["rays"]["solid"] == 0

    body, pane = model.geometry["model"], model.geometry[glass.GLASS]
    expected = window_faces(mesh)
    assert expected.sum() == 16  # 4 quads on each surface of the slab
    assert centres(pane) == centres(mesh, mesh.faces[expected])
    assert len(body.faces) + len(pane.faces) == len(mesh.faces) == glass.face_count(model)

    material = pane.visual.material
    assert material.alphaMode == "BLEND" and material.name == glass.GLASS
    assert material.baseColorTexture is None and material.metallicRoughnessTexture is None
    assert material.metallicFactor == 0.0 and material.roughnessFactor == pytest.approx(glass.GLASS_ROUGHNESS)
    assert material.doubleSided is False
    # From TRELLIS.2's alpha over the glass, as the texture stores it
    opacity = glass.OPACITY[0] + glass.OPACITY[1] * round(255 * WINDOW["alpha"]) / 255
    assert material.baseColorFactor[3] / 255 == pytest.approx(opacity, abs=1 / 255)
    assert report["opacity"] == pytest.approx(opacity, abs=1e-3)
    # A dark, neutral tint (glTF's factor is linear): TRELLIS.2's colour here is too dark to trust its hue
    assert (material.baseColorFactor[:3] / 255.0) @ glass.LUMA == pytest.approx(glass.GLASS_LUMA, abs=0.004)
    assert np.ptp(material.baseColorFactor[:3]) <= 1
    assert body.visual.material.alphaMode == "OPAQUE"
    assert body.visual.material.baseColorTexture is not None

    # Both parts keep the input's vertex normals (the shading normals), corner by corner; the glass needs no UVs
    assert pane.visual.uv is None and body.visual.uv is not None
    assert corners(pane) == corners(mesh, mesh.faces[expected])
    assert corners(body) == corners(mesh, mesh.faces[~expected])


def test_glass_on_both_sides_of_a_cabin_is_see_through_through_both():
    sides = ((0, 1), (0, -1))
    mesh, original = car(sides=sides)
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert isinstance(model, trimesh.Scene)
    assert report["see_through"]["regions"] == 4
    # Looking in squarely through one window, a ray meets the other window and leaves through it
    assert report["rays"]["through"] > 0 and report["rays"]["solid"] == 0
    assert centres(model.geometry[glass.GLASS]) == centres(mesh, mesh.faces[window_faces(mesh, sides)])


def test_a_window_on_a_solid_body_stays_opaque_but_glossy():
    mesh, original = car(hollow=False)
    model, report = glass.split_glass(mesh, original, VOXEL)
    # Nothing behind the window but the inside of a closed solid: see-through would show the background
    assert isinstance(model, trimesh.Trimesh) and model is not mesh
    assert report["applied"] and report["reason"] == "glossy glass (nothing see-through)"
    assert report["see_through"]["regions"] == 0 and report["opaque"]["regions"] == 1
    assert report["rays"]["solid"] > 0 and report["rays"]["cavity"] == 0
    assert len(model.faces) == len(mesh.faces)
    assert model.visual.material.alphaMode == "OPAQUE"

    rough_before = np.asarray(mesh.visual.material.metallicRoughnessTexture)
    rough_after = np.asarray(model.visual.material.metallicRoughnessTexture)
    window = np.asarray(original)[..., 3] < 255
    # The window's texels: glossy and not metal, in the colour they had (the picture may have painted them)
    assert model.visual.material.baseColorTexture is mesh.visual.material.baseColorTexture
    assert rough_after[window][:, 1].max() <= round(255 * glass.GLASS_ROUGHNESS) + 1
    assert rough_after[window][:, 2].max() == 0
    # The body's texels, untouched
    np.testing.assert_array_equal(rough_after[~window], rough_before[~window])


def test_the_texels_under_see_through_glass_take_its_colour():
    mesh, original = car()
    model, report = glass.split_glass(mesh, original, VOXEL)
    body = model.geometry["model"]
    before = np.asarray(mesh.visual.material.baseColorTexture, dtype=float)
    after = np.asarray(body.visual.material.baseColorTexture, dtype=float)
    window = np.asarray(original)[..., 3] < 255
    # Glass's one dark luminance, so the body's charts don't filter pale colour in at their edges
    srgb = after[window] / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4) @ glass.LUMA
    assert np.abs(linear - glass.GLASS_LUMA).max() < 0.004
    np.testing.assert_array_equal(after[~window], before[~window])


def test_an_opaque_model_is_left_alone():
    mesh, original = build(box_quads(1.0), lambda quad: BODY)
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert model is mesh and not report["applied"]
    assert report["reason"] == "no transparent texels"


def test_without_the_original_alpha_the_mesh_is_left_alone():
    mesh, original = car()
    for given in (original.convert("RGB"), None):
        model, report = glass.split_glass(mesh, given, VOXEL)
        assert model is mesh and not report["applied"]
        assert report["reason"] == "the original texture has no alpha"


def test_the_alpha_channel_alone_is_enough():
    mesh, original = car()
    model, report = glass.split_glass(mesh, np.asarray(original)[..., 3], VOXEL)
    assert isinstance(model, trimesh.Scene) and report["see_through"]["regions"] == 2


def test_matte_transparency_is_spurious_not_glass():
    # Phase 2's dragon: alpha 0.3-0.7 over its skin, at roughness 0.93-0.98
    mesh, original = car(window={**WINDOW, "rough": 0.95})
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert model is mesh and report["reason"] == "no glass: the transparent texels are matte"


def test_glass_too_small_to_be_a_window_is_left_alone():
    # The same window measured on a coarse grid: 3.3 voxels across, 11 square voxels
    mesh, original = car()
    model, report = glass.split_glass(mesh, original, 0.2)
    assert model is mesh and report["reason"] == "no glass region is large enough"


def test_coloured_glass_is_left_alone():
    # A tail light: saturated red at a low alpha, which the division leaves bright red
    mesh, original = car(window={**WINDOW, "rgb": (200, 10, 10)})
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert model is mesh and report["reason"] == "only coloured glass"
    assert report["coloured"]["regions"] == 2


def test_the_input_is_never_modified():
    mesh, original = car()
    material = mesh.visual.material
    colour, rough = material.baseColorTexture, material.metallicRoughnessTexture
    pixels, rough_pixels = colour.tobytes(), rough.tobytes()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    model, _ = glass.split_glass(mesh, original, VOXEL)
    assert isinstance(model, trimesh.Scene)
    assert mesh.visual.material is material and material.alphaMode == "OPAQUE"
    assert material.baseColorTexture is colour and colour.tobytes() == pixels
    assert material.metallicRoughnessTexture is rough and rough.tobytes() == rough_pixels
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_bad_input_never_raises(monkeypatch):
    mesh, original = car()
    thing = object()
    model, report = glass.split_glass(thing, original, VOXEL)
    assert model is thing and report["reason"] == "the mesh has no base colour texture or UVs"
    model, report = glass.split_glass(trimesh.creation.box(), original, VOXEL)
    assert not report["applied"]
    model, report = glass.split_glass(mesh, original, None)
    assert model is mesh and report["reason"] == "no voxel size"

    def broken(*args, **kwargs):
        raise RuntimeError("no luck")

    monkeypatch.setattr(glass, "_ray_hits", broken)
    model, report = glass.split_glass(mesh, original, VOXEL)
    assert model is mesh and not report["applied"] and report["reason"] == "error: RuntimeError: no luck"


def test_the_export_is_a_glb_with_an_opaque_body_and_blend_glass():
    mesh, original = car()
    model, report = glass.split_glass(mesh, original, VOXEL)
    meta, _ = read_glb(model.export(file_type="glb"))
    primitives = [p for m in meta["meshes"] for p in m["primitives"]]
    assert len(primitives) == 2
    materials = [meta["materials"][p["material"]] for p in primitives]
    by_mode = {m.get("alphaMode", "OPAQUE"): (m, p) for m, p in zip(materials, primitives)}
    assert set(by_mode) == {"OPAQUE", "BLEND"}
    body, body_primitive = by_mode["OPAQUE"]
    pane, pane_primitive = by_mode["BLEND"]
    assert "baseColorTexture" in body["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" in body["pbrMetallicRoughness"]
    assert pane["name"] == glass.GLASS and pane.get("doubleSided", False) is False
    assert set(pane["pbrMetallicRoughness"]) == {"baseColorFactor", "metallicFactor", "roughnessFactor"}
    assert pane["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(report["opacity"], abs=1 / 255)
    assert "TEXCOORD_0" in body_primitive["attributes"] and "TEXCOORD_0" not in pane_primitive["attributes"]
    triangles = [meta["accessors"][p["indices"]]["count"] // 3 for p in (body_primitive, pane_primitive)]
    assert triangles == [len(mesh.faces) - 16, 16]
    # One texture each for colour and metallic-roughness: the glass shares none
    assert len(meta["images"]) == 2


@pytest.mark.skipif(shutil.which(compress.GLTFPACK) is None, reason="gltfpack not installed")
def test_gltfpack_keeps_the_glass_material():
    mesh, original = car()
    model, report = glass.split_glass(mesh, original, VOXEL)
    meta, _ = read_glb(compress.pack_glb(model.export(file_type="glb"), 256))
    primitives = [p for m in meta["meshes"] for p in m["primitives"]]
    materials = [meta["materials"][p["material"]] for p in primitives]
    blend = [m for m in materials if m.get("alphaMode") == "BLEND"]
    opaque = [m for m in materials if m.get("alphaMode", "OPAQUE") == "OPAQUE"]
    assert len(blend) == 1 and len(opaque) == 1
    factor = blend[0]["pbrMetallicRoughness"]["baseColorFactor"]
    assert factor[3] == pytest.approx(report["opacity"], abs=1 / 255)
    assert blend[0]["pbrMetallicRoughness"]["metallicFactor"] == 0
    assert "baseColorTexture" not in blend[0]["pbrMetallicRoughness"]
    colour = meta["textures"][opaque[0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    assert "EXT_texture_webp" in colour["extensions"]
    assert len(meta["images"]) == 2  # still one WebP colour and one KTX2 metallic-roughness


@pytest.mark.skipif(shutil.which(compress.GLTFPACK) is None, reason="gltfpack not installed")
def test_gltfpack_keeps_a_base_colour_alpha_in_webp():
    # Not what the export ships (unpremultiply drops the alpha), but what gltfpack does with one: -tw color
    # writes WebP with its alpha (VP8X + ALPH), even for an OPAQUE material
    mesh, original = car()
    material = mesh.visual.material
    material.baseColorTexture = original
    meta, binary = read_glb(compress.pack_glb(mesh.export(file_type="glb"), 256))
    source = meta["textures"][meta["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    image = meta["images"][source["extensions"]["EXT_texture_webp"]["source"]]
    view = meta["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    webp = Image.open(io.BytesIO(binary[start : start + view["byteLength"]]))
    assert webp.mode == "RGBA"
    alpha = np.asarray(webp)[..., 3]
    expected = np.asarray(original)[..., 3]
    assert np.abs(alpha.astype(int) - expected).max() <= 2


def test_the_summary_leaves_out_the_regions():
    mesh, original = car()
    _, report = glass.split_glass(mesh, original, VOXEL)
    assert len(report["regions"]) == 2 and report["regions"][0]["verdict"] == "see-through"
    short = glass.summary(report)
    assert "regions" not in short and short["see_through"] == report["see_through"]
    assert json.loads(json.dumps(short)) == short


def test_the_export_makes_see_through_glass(monkeypatch):
    """Through Trellis2Runtime.export as the integration wires it: after the shading normals, before the GLB."""
    import types

    from forge3d_worker import pipeline
    from forge3d_worker.settings import PRESETS

    mesh, original = car()
    to_glb_mesh = trimesh.Trimesh(
        mesh.vertices,
        mesh.faces,
        visual=trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=original,  # to_glb's RGBA
                metallicRoughnessTexture=mesh.visual.material.metallicRoughnessTexture,
                alphaMode="OPAQUE",
            ),
        ),
        process=False,
    )
    monkeypatch.setattr(pipeline.projection, "project_picture", lambda glb, picture: (glb, {"applied": False}))
    runtime = pipeline.Trellis2Runtime.__new__(pipeline.Trellis2Runtime)
    runtime.pipeline = types.SimpleNamespace(low_vram=True)
    runtime._free_gpu_memory = lambda: None
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: to_glb_mesh))
    generated = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=VOXEL)
    data, triangles = runtime.export(generated, PRESETS["final"])
    meta, _ = read_glb(data)
    modes = sorted(meta["materials"][p["material"]].get("alphaMode", "OPAQUE") for m in meta["meshes"] for p in m["primitives"])
    assert modes == ["BLEND", "OPAQUE"] and triangles == len(mesh.faces)
    assert runtime.last_glass["reason"] == "see-through glass" and "regions" not in runtime.last_glass
