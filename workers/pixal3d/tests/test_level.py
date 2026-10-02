"""Levelling: the camera-aligned mesh turned level by the picture's elevation, and where that elevation comes from."""

import math

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

from forge3d_worker import projection as P
from pixal3d_worker import level


@pytest.mark.parametrize("pose", [(0.0, 0.0, 0.0), (0.0, 28.8, 0.0), (0.0, 45.0, -7.9), (35.0, 10.0, 4.0), (0.0, -20.0, 25.0)])
def test_camera_axes_are_the_projections(pose):
    azimuth, elevation, roll = pose
    params = torch.tensor([[azimuth, elevation, roll, 0.0, 0.0, 0.0, 0.0]])
    right, up, back = (t[0].numpy() for t in P.view_axes(params))
    np.testing.assert_allclose(level.camera_axes(elevation, roll, azimuth), np.stack([right, up, back], 1), atol=1e-6)


def test_level_matrix_puts_a_tilted_object_back_on_its_feet():
    # A level object: a tall thin post, +Y up, in the GLB frame
    post = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
    for elevation, roll in [(28.8, 0.0), (45.0, -7.9), (-15.0, 3.0)]:
        axes = level.camera_axes(elevation, roll)
        # Pixal3D put the camera's right, up and back on +X, +Y, +Z: the post as it builds it
        as_built = post @ axes  # each row p -> axes.T @ p
        levelled = (level.level_matrix(elevation, roll) @ np.c_[as_built, np.ones(4)].T).T[:, :3]
        np.testing.assert_allclose(levelled, post, atol=1e-9)
    # No tilt: the identity. A camera looking down leans the built object towards the viewer (+Z):
    # its top goes to +Z and levelling brings it back to +Y
    np.testing.assert_allclose(level.level_matrix(0.0), np.eye(4))
    leaning = np.array([0.0, math.cos(math.radians(30)), math.sin(math.radians(30)), 1.0])
    np.testing.assert_allclose(level.level_matrix(30.0) @ leaning, [0.0, 1.0, 0.0, 1.0], atol=1e-9)


def test_level_matrix_rejects_nonsense():
    with pytest.raises(ValueError):
        level.level_matrix(90.0)
    with pytest.raises(ValueError):
        level.level_matrix(10.0, roll_deg=40.0)
    with pytest.raises(ValueError):
        level.level_matrix(float("nan"))


def plate(tilt_deg: float) -> trimesh.Trimesh:
    """A flat square plate, 1 wide and 0.05 thick, leaning towards +Z by tilt_deg as Pixal3D would build it."""
    box = trimesh.creation.box(extents=(1.0, 0.05, 1.0))
    box.apply_translation([0.2, 0.0, -0.1])  # off-centre, as a voxel grid's object may be
    box.apply_transform(level.level_matrix(tilt_deg).T)  # the inverse: into the camera-aligned frame
    return box


def test_level_turns_and_recentres_the_mesh():
    glb = plate(29.0)
    before = glb.extents.copy()
    assert before[1] > 0.4  # tilted: tall in Y
    note = level.level(glb, 29.0)
    assert note["applied"] is True and note["elevation"] == 29.0 and note["roll"] == 0.0
    assert glb.extents[1] == pytest.approx(0.05, abs=1e-6) and glb.extents[0] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose((glb.bounds[0] + glb.bounds[1]) / 2, 0.0, atol=1e-9)
    assert "recentred" in note


def test_tiny_tilts_are_left_alone():
    glb = plate(0.0)
    vertices = glb.vertices.copy()
    note = level.level(glb, 0.4, 0.3)
    assert note["applied"] is False
    np.testing.assert_array_equal(glb.vertices, vertices)
    assert level.needs_levelling(1.0) and not level.needs_levelling(0.99, 0.99)


def test_tilt_reads_a_gated_report():
    assert level.tilt(None) == (0.0, 0.0)
    assert level.tilt({"applied": False, "pose": {"elevation": 30.0, "roll": 0.0}}) == (0.0, 0.0)
    assert level.tilt({"applied": True, "pose": {"elevation": 28.8, "roll": -2.0}}) == (28.8, -2.0)
    assert level.tilt({"applied": True, "pose": {"elevation": 89.0, "roll": -40.0}}) == (85.0, -25.0)


def frustum() -> trimesh.Trimesh:
    """A truncated pyramid, wide at the bottom: its silhouette tells above from below. Textured, with UVs."""
    bottom, top, height = 0.6, 0.3, 0.8
    vertices = np.array(
        [[-bottom, -height / 2, -bottom], [bottom, -height / 2, -bottom], [bottom, -height / 2, bottom], [-bottom, -height / 2, bottom]]
        + [[-top, height / 2, -top], [top, height / 2, -top], [top, height / 2, top], [-top, height / 2, top]]
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    uv = (vertices[:, [0, 2]] + bottom) / (2 * bottom)
    texture = Image.new("RGB", (16, 16), (200, 120, 60))
    visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture))
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def picture_of(mesh: trimesh.Trimesh, elevation: float, size: int = 160) -> Image.Image:
    """An RGBA cutout of the mesh's silhouette from the projection's camera at this elevation (orthographic)."""
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    centre = (verts.min(0).values + verts.max(0).values) / 2
    verts = (verts - centre) / (verts - centre).norm(dim=1).max()
    params = torch.tensor([[0.0, elevation, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x, y, depth, s = P.project(verts, params)
    xy = torch.stack([(x[0] + 1.2) / 2.4 * size, (1.2 - y[0]) / 2.4 * size], -1)
    zbuf = P.rasterize_depth(xy, depth[0], s[0], torch.tensor(mesh.faces, dtype=torch.long), size, size)
    mask = torch.isfinite(zbuf).numpy()
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[mask] = (200, 120, 60, 255)
    return Image.fromarray(rgba, "RGBA")


def test_estimate_pose_finds_the_elevation_and_gates():
    mesh = frustum()
    report = level.estimate_pose(mesh, picture_of(mesh, 40.0), device="cpu")
    assert report["applied"] is True, report
    assert report["pose"]["elevation"] == pytest.approx(40.0, abs=6.0)
    assert report["iou"] >= P.MIN_IOU
    elevation, roll = level.tilt(report)
    assert elevation == report["pose"]["elevation"] and abs(roll) <= 6.0
    # Not the pictured shape: the gate fails and nothing is levelled
    wrong = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    wrong.paste(Image.new("RGBA", (120, 20), (200, 120, 60, 255)), (20, 70))
    report = level.estimate_pose(mesh, wrong, device="cpu")
    assert report["applied"] is False and "IoU" in report["reason"] and report["pose"] is not None
    assert level.tilt(report) == (0.0, 0.0)
    # No alpha, or no texture: skipped with the projection's reasons
    assert "alpha" in level.estimate_pose(mesh, Image.new("RGB", (64, 64)), device="cpu")["reason"]
    bare = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    assert "texture" in level.estimate_pose(bare, picture_of(mesh, 40.0), device="cpu")["reason"]


def test_a_camera_well_below_the_horizon_is_not_trusted(monkeypatch):
    # The search placed at -20 (a frustum pictured from +20 scores the same from -20 at a uniform
    # texture): the report keeps the pose but the gate fails, so nothing is levelled by the wrong sign
    mesh = frustum()
    found = torch.tensor([[0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    monkeypatch.setattr(
        P, "_estimate_pose", lambda model, pic, clock: P._Pose(found, 0.98, 0.0, 0.98, None, found[:0], [])
    )
    monkeypatch.setattr(P, "_polish", lambda model, pic, params: params)
    report = level.estimate_pose(mesh, picture_of(mesh, -20.0), device="cpu")
    assert report["applied"] is False and "below the horizon" in report["reason"]
    assert report["pose"]["elevation"] == -20.0 and level.tilt(report) == (0.0, 0.0)


def test_a_rival_camera_that_agrees_on_the_tilt_is_harmless(monkeypatch):
    # The projection would give up (which side to paint?); the tilt only needs elevation and roll
    mesh = frustum()
    best = torch.tensor([[275.0, 8.8, 0.0, 0.3, 0.0, 0.0, 0.0]])
    rival_same_tilt = torch.tensor([[357.5, 5.0, 0.0, 0.7, 0.0, 0.0, 0.0]])
    rival_other_tilt = torch.tensor([[357.5, 30.0, 0.0, 0.7, 0.0, 0.0, 0.0]])
    monkeypatch.setattr(P, "_polish", lambda model, pic, params: params)
    monkeypatch.setattr(P, "_shape_difference", lambda za, zb: 1.0)  # a different shape, always
    monkeypatch.setattr(P, "_iou", lambda a, b: torch.tensor(0.96))

    def pose_with(rivals):
        return lambda model, pic, clock: P._Pose(best, 0.96, 0.6, 0.96, None, rivals, [])

    monkeypatch.setattr(P, "_estimate_pose", pose_with(rival_same_tilt))
    report = level.estimate_pose(mesh, picture_of(mesh, 9.0), device="cpu")
    assert report["applied"] is True and report["rivals"][0]["tilt_gap"] == 3.8
    assert level.tilt(report) == (8.8, 0.0)
    monkeypatch.setattr(P, "_estimate_pose", pose_with(rival_other_tilt))
    report = level.estimate_pose(mesh, picture_of(mesh, 9.0), device="cpu")
    assert report["applied"] is False and "ambiguous camera" in report["reason"] and "21 degrees of tilt" in report["reason"]
