"""Pixal3D's cameras: its own conventions, MV-Adapter's views in them, and the GLB frame."""

import json
import math
import os

import numpy as np
import pytest

from pixal3d_worker import cameras

# Pixal3D's canonical front view (ProjGrid.front_view_transform_matrix, distance 2)
PIXAL3D_FRONT = np.array([[1, 0, 0, 0], [0, 0, -1, -2.0], [0, 1, 0, 0], [0, 0, 0, 1]])
# Pixal3D's shipped four-view rig (assets/mv_images/example/transforms.json): 20 degrees, d = 3.1192
EXAMPLE_DISTANCE = 3.1192049980163574
EXAMPLE = {
    0: [[1, 0, 0, 0], [0, 0, -1, -EXAMPLE_DISTANCE], [0, 1, 0, 0], [0, 0, 0, 1]],
    90: [[0, 0, 1, EXAMPLE_DISTANCE], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    180: [[-1, 0, 0, 0], [0, 0, 1, EXAMPLE_DISTANCE], [0, 1, 0, 0], [0, 0, 0, 1]],
    270: [[0, 0, -1, -EXAMPLE_DISTANCE], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
}


def mv_adapter_camera(azimuth, elevation, distance):
    """multiview_worker/cameras.py's camera_to_world, as its docstring states it."""
    a, e = math.radians(azimuth), math.radians(elevation)
    position = distance * np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])
    look = -position / np.linalg.norm(position)
    right = np.cross(look, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, up, -look, position
    return matrix


def test_front_is_pixal3d_main_view_exactly():
    assert np.array_equal(cameras.front_camera(2.0), PIXAL3D_FRONT)


@pytest.mark.parametrize("azimuth", sorted(EXAMPLE))
def test_matches_pixal3d_example_rig(azimuth):
    np.testing.assert_allclose(cameras.orbit_camera(azimuth, 0, EXAMPLE_DISTANCE), EXAMPLE[azimuth], atol=1e-12)


@pytest.mark.parametrize("azimuth", [0, 45, 90, 180, 270, 315])
@pytest.mark.parametrize("elevation", [0, 20, -10])
def test_matches_the_multiview_workers_cameras(azimuth, elevation):
    np.testing.assert_allclose(
        cameras.orbit_camera(azimuth, elevation, 1.8), mv_adapter_camera(azimuth, elevation, 1.8), atol=1e-12
    )


def test_the_90_degree_view_sees_the_pictures_right_side():
    """MV-Adapter's 90 degree view shows the picture's right-hand side, its front facing image-left."""
    right_side = np.array([[0.5, 0, 0]])  # +X: on the right of the front view
    front_side = np.array([[0, -0.5, 0]])  # -Y: towards the front camera
    front = cameras.front_camera(10.0)
    assert cameras.project(right_side, front, math.radians(20), 512)[0, 0] > 256  # right in the picture
    side = cameras.orbit_camera(90, 0, 10.0)
    depth_right = (np.linalg.inv(side) @ np.append(right_side[0], 1))[2]
    depth_front = (np.linalg.inv(side) @ np.append(front_side[0], 1))[2]
    assert depth_right > depth_front  # the right side is nearer the camera (camera z points at it)
    assert cameras.project(front_side, side, math.radians(20), 512)[0, 0] < 256  # the front on the left


def test_frame_width_matches_upstream_example():
    """Pixal3D's example rig frames +-0.55 at the origin, MV-Adapter's orthographic frame."""
    assert cameras.distance_for_half_width(0.55, 20.0) == pytest.approx(EXAMPLE_DISTANCE, rel=1e-4)
    edge = cameras.project(np.array([[0.55, 0, 0]]), cameras.front_camera(EXAMPLE_DISTANCE), math.radians(20), 512)
    assert edge[0, 0] == pytest.approx(512, abs=0.05)


def test_single_view_distance_is_upstreams_distance_from_fov():
    """inference.py's distance_from_fov: grid point (-1, 0, 0) on the picture's left edge (pixel 0)."""
    for fov in (0.2, 0.5, 0.9):
        distance = cameras.single_view_distance(fov)
        point = cameras.grid_to_world(np.array([[-1.0, 0.0, 0.0]]))
        x = cameras.project(point, cameras.front_camera(distance), fov, 512)[0, 0]
        assert x == pytest.approx(0.0, abs=1e-6)


def test_near_orthographic_cameras_frame_like_the_orthographic_views():
    """Every point of the unit cube lands within a few pixels of where an orthographic view draws it."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(2000, 3))
    size, half = 768, cameras.MV_ADAPTER_HALF_WIDTH
    distance = cameras.distance_for_half_width(half, cameras.NEAR_ORTHO_FOV_DEG)
    for azimuth in (0, 45, 90, 180, 270, 315):
        c2w = cameras.orbit_camera(azimuth, 0, distance)
        got = cameras.project(points, c2w, math.radians(cameras.NEAR_ORTHO_FOV_DEG), size)
        local = (points - c2w[:3, 3]) @ c2w[:3, :3]  # right, up, back
        scale = size / (2 * half)
        orthographic = np.stack([size / 2 + local[:, 0] * scale, size / 2 - local[:, 1] * scale], 1)
        assert np.abs(got - orthographic).max() < 6.0  # pixels at 768; at most 0.4 % of the frame


def test_grid_and_glb_frames_agree():
    """A vertex of the exported GLB is where Pixal3D's projection put its voxel."""
    native = np.array([[0.3, -0.2, 0.45], [-0.5, 0.5, 0.0]])  # mesh coordinates, [-0.5, 0.5]^3
    to_glb = native[:, [0, 2, 1]] * [1, 1, -1]  # o_voxel's to_glb: (x, y, z) -> (x, z, -y)
    glb = np.c_[to_glb, np.ones(2)] @ cameras.GLB_FROM_TO_GLB.T
    np.testing.assert_allclose(glb[:, :3], native)  # back to the grid's own axes: y up
    np.testing.assert_allclose(cameras.glb_to_world(glb[:, :3]), cameras.grid_to_world(native * 2))


def test_the_main_camera_ends_up_on_plus_z_of_the_glb():
    """Production's GLBs face +Z (the turntable's azimuth 0): so must the picture's camera."""
    position = cameras.front_camera(3.0)[:3, 3]  # world (Blender): (0, -3, 0)
    world_to_glb = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # inverse of glb_to_world
    np.testing.assert_allclose(world_to_glb @ position, [0, 0, 3], atol=1e-12)
    np.testing.assert_allclose(cameras.glb_to_world(np.array([[0, 0, 3.0]]))[0], position, atol=1e-12)
    # glTF up (+Y) is the world's +Z
    np.testing.assert_allclose(cameras.glb_to_world(np.array([[0, 1.0, 0]]))[0], [0, 0, 1])


def test_rig_layout_and_checks():
    rig = cameras.rig([(0, 0), (90, 0), (180, 0), (270, 0)])
    assert rig["transform_matrix"].shape == (1, 4, 4, 4)
    assert rig["camera_angle_x"].shape == rig["camera_distance"].shape == (1, 4)
    distance = cameras.distance_for_half_width(0.55, cameras.NEAR_ORTHO_FOV_DEG)
    np.testing.assert_allclose(rig["camera_distance"], distance)
    np.testing.assert_allclose(np.linalg.norm(rig["transform_matrix"][0, :, :3, 3], axis=-1), distance)
    assert np.array_equal(rig["transform_matrix"][0, 0], cameras.front_camera(distance))
    with pytest.raises(ValueError):
        cameras.rig([(90, 0), (0, 0)])
    with pytest.raises(ValueError):
        cameras.orbit_camera(0, 85, 1.0)


def test_transforms_json_is_upstreams_format(tmp_path):
    rig = cameras.rig([(0, 0), (90, 0)], half_width=0.55, fov_deg=20)
    data = cameras.transforms_json(["view00_azim000.png", "view01_azim090.png"], rig)
    path = os.path.join(tmp_path, "transforms.json")
    with open(path, "w") as f:
        json.dump(data, f)
    frames = json.load(open(path))["frames"]
    assert frames[0]["file_path"] == "view00_azim000.png"
    np.testing.assert_allclose(frames[1]["transform_matrix"], EXAMPLE[90], atol=1e-3)
    assert frames[1]["camera_angle_x"] == pytest.approx(math.radians(20))
