"""drop_floaters on synthetic meshes: what floats apart goes, what touches or is big stays, UVs survive."""

import numpy as np
import pytest
import trimesh
from PIL import Image

from forge3d_worker.cleanup import MAX_AREA_SHARE, MIN_GAP_SHARE, drop_floaters, find_pieces


def parts(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    """One mesh of separate parts (no shared vertices), as the remesher leaves pieces that touch."""
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(mesh.vertices)
    return trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)


def at(mesh: trimesh.Trimesh, *offset: float) -> trimesh.Trimesh:
    return mesh.copy().apply_translation(offset)


def donut() -> trimesh.Trimesh:
    """A ring around z, 0.84 across and 0.24 thick."""
    return trimesh.creation.torus(major_radius=0.3, minor_radius=0.12, major_sections=48, minor_sections=24)


def box(*extents: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def arrays(mesh: trimesh.Trimesh):
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def diagonal(*meshes: trimesh.Trimesh) -> float:
    return float(np.linalg.norm(np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0)))


def test_a_ball_floating_beside_a_donut_is_dropped():
    ring = donut()
    ball = at(trimesh.creation.icosphere(subdivisions=2, radius=0.05), 0.42 + 0.1 + 0.05, 0, 0)
    mesh = parts(ring, ball)
    report = drop_floaters(mesh)
    assert report["pieces"] == 2 and report["dropped"] == 1
    assert report["faces_dropped"] == len(ball.faces) and len(mesh.faces) == len(ring.faces)
    assert np.allclose(mesh.vertices, ring.vertices)  # the donut, untouched
    floater = report["floaters"][0]
    assert floater["area_share"] == pytest.approx(ball.area / (ball.area + ring.area), abs=1e-4)
    diagonal = np.linalg.norm([0.42 + 0.62, 0.84, 0.24])
    assert floater["gap_share"] == pytest.approx(0.1 / diagonal, abs=2e-3)


def sprinkles(count: int = 40, seed: int = 0) -> list:
    """Short rods lying on the donut's top, sunk a little into the frosting, all around the ring."""
    rng = np.random.default_rng(seed)
    rods = []
    for _ in range(count):
        angle = rng.uniform(0, 2 * np.pi)
        radius = 0.3 + rng.uniform(-0.06, 0.06)
        top = np.sqrt(max(0.12**2 - (radius - 0.3) ** 2, 0.0))  # the donut's surface height there
        rod = trimesh.creation.capsule(height=0.03, radius=0.006, count=[6, 6])
        rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))  # lying down
        rod.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(0, np.pi), [0, 0, 1]))
        rods.append(at(rod, radius * np.cos(angle), radius * np.sin(angle), top))
    return rods


def test_sprinkles_sitting_on_a_donut_stay():
    rods = sprinkles()
    mesh = parts(donut(), *rods)
    report = drop_floaters(mesh)
    assert report["pieces"] == 41 and report["dropped"] == 0 and report["floaters"] == []
    assert len(mesh.faces) == len(donut().faces) + sum(len(r.faces) for r in rods)


def test_pieces_reach_the_model_through_each_other():
    """A stack of sprinkles on the frosting: the top ones are beyond the gap from the donut, not from the one below."""
    ring = donut()
    # Cubes a little taller than their spacing, so each one sinks into the one below (no shared corners)
    stack = [at(box(0.015, 0.015, 0.016), 0.3, 0, 0.12 + 0.0074 + 0.015 * level) for level in range(6)]
    pieces = find_pieces(*arrays(parts(ring, *stack)))
    assert pieces.keep.all() and pieces.why == ("big",) + ("near",) * 6
    above = stack[-1].vertices[:, 2].min() - 0.12
    assert above > MIN_GAP_SHARE * diagonal(ring, *stack)


def test_a_cups_handle_stays():
    cup = trimesh.creation.cylinder(radius=0.25, height=0.6, sections=48)
    handle = trimesh.creation.torus(major_radius=0.1, minor_radius=0.02, major_sections=24, minor_sections=12)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # upright, beside the cup
    handle = at(handle, 0.25 + 0.06, 0, 0)  # its ring runs into the wall
    pieces = find_pieces(*arrays(parts(cup, handle)))
    assert pieces.area_share[1] < MAX_AREA_SHARE  # small: it stays because it touches
    assert pieces.keep.all() and pieces.why == ("big", "near")


def test_chopsticks_resting_in_a_bowl_stay():
    """
    One lies on the noodles (the bowl's flat top, a fan of large triangles that none of its vertices is
    near), one rests on the rim only (the rim's vertices are near its long side, not its ends).
    """
    bowl = trimesh.creation.cylinder(radius=0.3, height=0.25, sections=24)  # top at z = 0.125
    lying = at(box(0.3, 0.012, 0.012), 0, 0.12, 0.125 + 0.006)
    # Tilted 20 degrees, its inner end up over the bowl, its underside on the rim at (-0.3, 0)
    leaning = box(0.5, 0.012, 0.012)
    leaning.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-20), [0, 1, 0]))
    leaning = at(leaning, -0.3, 0, 0.125 + 0.006 / np.cos(np.radians(20)) + 0.0005)
    pieces = find_pieces(*arrays(parts(bowl, lying, leaning)))
    assert pieces.keep.all() and pieces.why == ("big", "near", "near")
    # Neither has a vertex within the gap of the bowl's: only the surfaces are that close
    for stick in (lying, leaning):
        nearest = np.linalg.norm(stick.vertices[:, None] - bowl.vertices[None], axis=2).min()
        assert nearest > MIN_GAP_SHARE * diagonal(bowl, lying, leaning)


def balloon(ropes: bool = True) -> list:
    """An envelope with a basket hanging 0.2 below it, on four ropes that overlap both (separate pieces)."""
    envelope = at(trimesh.creation.icosphere(subdivisions=3, radius=0.4), 0, 0, 0.2)
    basket = at(box(0.12, 0.12, 0.1), 0, 0, -0.45)
    lines = []
    if ropes:
        for x, y in ((-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)):
            top, bottom = np.array([x * 1.5, y * 1.5, -0.17]), np.array([x, y, -0.41])
            rope = trimesh.creation.cylinder(radius=0.003, segment=[top, bottom], sections=6)
            lines.append(rope)
    return [envelope, basket, *lines]


def test_a_balloons_basket_on_its_ropes_stays():
    pieces = find_pieces(*arrays(parts(*balloon())))
    assert pieces.area_share[1] < MAX_AREA_SHARE  # the basket is small: it stays through the ropes
    assert pieces.keep.all() and pieces.why == ("big",) + ("near",) * 5
    # Cut loose, about 0.2 below the envelope (an eighth of the diagonal), it would go
    envelope, basket = balloon(ropes=False)
    loose = find_pieces(*arrays(parts(envelope, basket)))
    assert list(loose.keep) == [True, False]
    assert loose.gap_share[1] == pytest.approx((envelope.vertices[:, 2].min() + 0.4) / diagonal(envelope, basket), abs=5e-3)


def test_a_part_big_enough_stays_however_far_it_hangs():
    envelope = at(trimesh.creation.icosphere(subdivisions=3, radius=0.4), 0, 0, 0.2)
    basket = at(box(0.3, 0.3, 0.2), 0, 0, -0.5)  # 0.36 of area against the envelope's 2: 15 %
    pieces = find_pieces(*arrays(parts(envelope, basket)))
    assert pieces.keep.all() and pieces.why == ("big", "big")


def dragon() -> tuple:
    """A body with a tail, and the tail's tip floating off its end: two back-to-back sheets, as the remesher makes thin parts."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
    tail = trimesh.creation.cylinder(radius=0.04, segment=[[0.25, 0, -0.1], [0.55, 0, -0.25]], sections=16)
    sheet = box(0.12, 0.002, 0.08)
    front, back = at(sheet, 0.65, 0.0012, -0.3), at(sheet, 0.65, -0.0012, -0.3)
    return body, tail, front, back


def test_a_tail_tip_hanging_in_the_air_is_dropped():
    body, tail, front, back = dragon()
    mesh = parts(body, tail, front, back)
    pieces = find_pieces(*arrays(mesh))
    # The two sheets touch each other and are judged together; together they are still small, and apart
    assert list(pieces.keep) == [True, True, False, False]
    assert pieces.why == ("big", "near", "floater", "floater")
    assert pieces.area_share[2] + pieces.area_share[3] < MAX_AREA_SHARE
    assert min(pieces.gap_share[2:]) > MIN_GAP_SHARE
    report = drop_floaters(mesh)
    assert report["dropped"] == 2 and len(mesh.faces) == len(body.faces) + len(tail.faces)


def cabin() -> tuple:
    house = at(box(0.6, 0.5, 0.45), 0, 0, 0.275)  # standing on the platform
    platform = box(0.8, 0.7, 0.1)  # z from -0.05 to 0.05
    steps = at(box(0.2, 0.1, 0.04), 0, -0.4, -0.02)  # against the platform's front
    specks = [at(box(0.01, 0.01, 0.004), x, y, -0.05 - gap) for x, y, gap in ((0.1, 0.1, 0.06), (-0.2, 0.05, 0.08), (0.25, -0.2, 0.07))]
    return house, platform, steps, specks


def test_specks_under_a_cabin_are_dropped_and_its_steps_stay():
    house, platform, steps, specks = cabin()
    mesh = parts(house, platform, steps, *specks)
    pieces = find_pieces(*arrays(mesh))
    assert list(pieces.keep) == [True, True, True, False, False, False]
    report = drop_floaters(mesh)
    assert report["dropped"] == 3 and report["faces_dropped"] == 36
    assert [f["faces"] for f in report["floaters"]] == [12, 12, 12]


def test_floaters_near_each_other_are_judged_together():
    """A clump of small pieces, apart from the model but big together, is a part: it stays."""
    ring = donut()
    clump = [at(trimesh.creation.icosphere(subdivisions=2, radius=0.06), 0.62, 0.0, z) for z in (-0.1, 0.0, 0.1, 0.2)]
    pieces = find_pieces(*arrays(parts(ring, *clump)))
    assert all(share < MAX_AREA_SHARE for share in pieces.area_share[1:])
    assert sum(pieces.area_share[1:]) >= MAX_AREA_SHARE
    assert pieces.keep.all() and pieces.why == ("big",) + ("group",) * 4


def test_many_separate_things_are_left_as_they_are():
    """Coins spread over a table, each small: the floaters would take most of the model, so none goes."""
    coins = [at(trimesh.creation.cylinder(radius=0.05, height=0.01, sections=24), x, y, 0) for x in np.linspace(-0.5, 0.5, 4) for y in np.linspace(-0.5, 0.5, 4)]
    mesh = parts(*coins)
    report = drop_floaters(mesh)
    assert report["pieces"] == 16 and report["dropped"] == 0 and len(mesh.faces) == 16 * len(coins[0].faces)
    pieces = find_pieces(*arrays(parts(*coins)))
    assert pieces.why.count("many") == 15


def test_a_mesh_in_one_piece_and_an_empty_one_are_left_alone():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    report = drop_floaters(sphere)
    assert report["pieces"] == 1 and report["dropped"] == 0 and len(sphere.faces) == 320
    empty = trimesh.Trimesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), process=False)
    assert drop_floaters(empty)["pieces"] == 0


def seamed(mesh: trimesh.Trimesh, seed: int = 0) -> trimesh.Trimesh:
    """``mesh`` with every face's vertices its own (as UV seams split them everywhere), random UVs per vertex."""
    vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces).reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    uv = np.random.default_rng(seed).uniform(0, 1, (len(vertices), 2))
    texture = Image.new("RGB", (16, 16), (90, 140, 200))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=0.8)
    return trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)


def test_uv_seams_never_split_a_piece_and_the_faces_that_stay_keep_their_uvs_and_texture():
    ring = donut()
    ball = at(trimesh.creation.icosphere(subdivisions=2, radius=0.05), 0.62, 0, 0)
    mesh = seamed(parts(ring, ball))
    material = mesh.visual.material
    corners = np.asarray(mesh.vertices)[mesh.faces], np.asarray(mesh.visual.uv)[mesh.faces]
    report = drop_floaters(mesh)

    # Every face its own three vertices, yet two pieces: the donut stays whole, the ball goes
    assert report["pieces"] == 2 and report["dropped"] == 1 and len(mesh.faces) == len(ring.faces)
    assert len(mesh.vertices) == 3 * len(ring.faces) and len(mesh.visual.uv) == len(mesh.vertices)
    # The donut's faces, in order, with the same corners and UVs as before; the same material and texture
    assert np.array_equal(np.asarray(mesh.vertices)[mesh.faces], corners[0][: len(ring.faces)])
    assert np.array_equal(np.asarray(mesh.visual.uv)[mesh.faces], corners[1][: len(ring.faces)])
    assert mesh.visual.material is material and material.baseColorTexture.size == (16, 16)


def test_the_textured_mesh_still_exports():
    mesh = seamed(parts(donut(), at(trimesh.creation.icosphere(subdivisions=1, radius=0.05), 0.62, 0, 0)))
    drop_floaters(mesh)
    loaded = trimesh.load(trimesh.util.wrap_as_stream(mesh.export(file_type="glb")), file_type="glb", force="mesh", process=False)
    assert len(loaded.faces) == len(donut().faces)


@pytest.mark.parametrize("area, gap, dropped", [(MAX_AREA_SHARE, MIN_GAP_SHARE, 1), (0.01, MIN_GAP_SHARE, 0), (MAX_AREA_SHARE, 0.1, 0)])
def test_the_thresholds_are_the_callers(area, gap, dropped):
    """The ball is 2.1 % of the area and 7.3 % of the diagonal away."""
    ring = donut()
    ball = at(trimesh.creation.icosphere(subdivisions=2, radius=0.05), 0.42 + 0.1 + 0.05, 0, 0)
    assert drop_floaters(parts(ring, ball), max_area_share=area, min_gap_share=gap)["dropped"] == dropped
