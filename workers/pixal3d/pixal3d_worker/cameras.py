"""
Cameras for Pixal3D, in its own convention, and the frame its meshes come out in.

Pixal3D (``pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py``) projects every voxel into
the picture with a perspective camera and samples the picture's DINOv3 features there:

- The world is Blender's: Z up. A camera is a 4x4 camera-to-world matrix that looks down its own -Z
  axis with +Y up, and its horizontal field of view (``camera_angle_x``, radians). The picture is
  square, and its pixel centres sit at (i + 0.5) / size.
- The voxel grid's axes map to the world as X = x, Y = -z, Z = y (``ProjGrid``'s rotation), scaled by
  1 / (2 * mesh_scale): the grid spans the cube [-0.5, 0.5]^3 at mesh_scale 1.
- The main view (frame 0) is the front camera ``F``: at (0, -distance, 0), looking along +Y. Every
  other view is placed relative to it (calc_mat_i = F @ inv(C_0) @ C_i), so only relative poses matter.

The mesh comes out in the voxel grid's frame: y up, the main camera on +z, the picture's right on +x.
That is the frame production's GLBs use once glTF's axes are put back (see ``GLB_FROM_TO_GLB``).

Views from MV-Adapter are orthographic. Pixal3D's projection is perspective only, so an orthographic
view gets a perspective camera with a tiny field of view (``NEAR_ORTHO_FOV_DEG``) far enough away that
its frame at the object spans the same width: at 1 degree, the near and far sides of the unit cube are
magnified within 1.6 % of each other (a few pixels at 768).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# MV-Adapter's image-to-multiview cameras: orthographic, the frame spanning [-0.55, 0.55] around the
# object (scripts/inference_i2mv_sdxl.py: get_orthogonal_camera(left=-0.55, right=0.55, ...))
MV_ADAPTER_HALF_WIDTH = 0.55
# Field of view standing in for an orthographic camera
NEAR_ORTHO_FOV_DEG = 1.0
# Views from overhead or underneath have no defined "right" with Z up; MV-Adapter uses elevation 0
MAX_ELEVATION_DEG = 80.0

# to_glb (o_voxel.postprocess) writes a vertex (x, y, z) as (x, z, -y), turning TRELLIS.2's Z-up grid
# into glTF's Y-up. Pixal3D's grid is already y-up with the camera on +z, so this undoes that: back to
# (x, y, z), which in glTF is upright with the pictured side facing +Z, as production's GLBs are (the
# gallery's turntable and the projection's camera search both put azimuth 0 on +Z).
GLB_FROM_TO_GLB = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def orbit_camera(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """
    Camera-to-world matrix of a camera on a sphere around the origin, looking at it, upright.

    Azimuth 0 is the front (the main view, at -Y); azimuth 90 is to the main picture's right (+X), as
    MV-Adapter and Pixal3D's example rig count it; elevation is above the horizon. This is
    MV-Adapter's ``get_c2w`` with its azimuth offset (it is called with azimuth - 90) folded in.
    """
    if abs(elevation_deg) > MAX_ELEVATION_DEG:
        raise ValueError(f"elevation must be within +-{MAX_ELEVATION_DEG} degrees, got {elevation_deg}")
    if distance <= 0:
        raise ValueError(f"distance must be positive, got {distance}")
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    position = distance * np.array(
        [
            math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        ]
    )
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -forward, position
    # Exact zeros where the trigonometry leaves 1e-17s, so frame 0 matches F to the bit
    c2w[np.abs(c2w) < 1e-12] = 0.0
    return c2w


def front_camera(distance: float) -> np.ndarray:
    """Pixal3D's canonical main view F with the given distance."""
    return orbit_camera(0.0, 0.0, distance)


def distance_for_half_width(half_width: float, fov_deg: float) -> float:
    """How far a camera with this field of view must be for its frame to span +-half_width at the origin."""
    return half_width / math.tan(math.radians(fov_deg) / 2)


def single_view_distance(fov_rad: float) -> float:
    """
    Pixal3D's distance for one picture cropped by ``preprocess_image``: the frame spans [-0.5, 0.5] at
    the origin (``distance_from_fov`` in inference.py, which this equals in closed form).
    """
    return 0.5 / math.tan(fov_rad / 2)


def project(points: np.ndarray, c2w: np.ndarray, fov_rad: float, resolution: int) -> np.ndarray:
    """
    Pixel coordinates of world points, exactly as Pixal3D's ``project_points_to_image_batch`` computes
    them (x right, y down, in pixels of a ``resolution``-wide picture).
    """
    points = np.asarray(points, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    focal = 16.0 / math.tan(fov_rad / 2) * resolution / 32.0
    depth = -camera[:, 2]
    x = focal * camera[:, 0] / depth + resolution / 2
    y = -focal * camera[:, 1] / depth + resolution / 2
    return np.stack([x, y], axis=-1)


def grid_to_world(points: np.ndarray, mesh_scale: float = 1.0) -> np.ndarray:
    """Pixal3D's voxel-grid coordinates ([-1, 1]^3) to the world its cameras live in."""
    points = np.asarray(points, dtype=np.float64)
    rotated = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)
    return rotated / mesh_scale / 2


def glb_to_world(points: np.ndarray) -> np.ndarray:
    """A point of an exported GLB (Y up, front +Z) in Pixal3D's camera world (Z up, front -Y)."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)


def rig(
    views: Iterable[tuple[float, float]],
    half_width: float = MV_ADAPTER_HALF_WIDTH,
    fov_deg: float = NEAR_ORTHO_FOV_DEG,
) -> dict:
    """
    Pixal3D's multi-view camera inputs for orthographic views at (azimuth, elevation) pairs, each
    framed +-half_width: ``transform_matrix`` [1, V, 4, 4], ``camera_angle_x`` [1, V] and
    ``camera_distance`` [1, V] as numpy arrays (``inference_mv.load_views``' layout). The first view is
    the main one and must be the front (azimuth 0, elevation 0).
    """
    views = [(float(a) % 360.0, float(e)) for a, e in views]
    if not views:
        raise ValueError("no views")
    if views[0] != (0.0, 0.0):
        raise ValueError(f"the first view must be the front (azimuth 0, elevation 0), got {views[0]}")
    distance = distance_for_half_width(half_width, fov_deg)
    matrices = np.stack([orbit_camera(a, e, distance) for a, e in views])
    fov = math.radians(fov_deg)
    return {
        "transform_matrix": matrices[None],
        "camera_angle_x": np.full((1, len(views)), fov),
        "camera_distance": np.full((1, len(views)), distance),
    }


def transforms_json(names: Sequence[str], camera_rig: dict, mesh_scale: float = 1.0) -> dict:
    """The same rig as ``inference_mv.py``'s transforms.json, for running upstream's script on a view set."""
    matrices = camera_rig["transform_matrix"][0]
    fovs = camera_rig["camera_angle_x"][0]
    return {
        "mesh_scale": mesh_scale,
        "frames": [
            {
                "file_path": name,
                "name": name.rsplit(".", 1)[0],
                "camera_angle_x": float(fov),
                "transform_matrix": [[float(v) for v in row] for row in matrix],
            }
            for name, matrix, fov in zip(names, matrices, fovs)
        ],
    }
