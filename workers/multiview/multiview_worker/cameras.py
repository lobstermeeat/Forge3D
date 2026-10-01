"""
Where the six views are seen from: MV-Adapter's own cameras, spelled out for the 3D step.

MV-Adapter's image-to-multiview model draws one object from six fixed orthographic cameras
(`get_orthogonal_camera` as upstream's scripts/inference_i2mv_sdxl.py calls it, with the
azimuths shifted by -90°). Its world is right-handed with +Z up and the object at the origin, and
the picture is the view from -Y:

- **Azimuth 0 is the picture's own view**, redrawn level (elevation 0). **Positive azimuth moves
  the camera counter-clockwise seen from above, towards the picture's right-hand side**: the 90°
  view shows the side that is on the right of the picture (whose front then faces image-left),
  180° the back, 270° the side on the left of the picture. Equivalently, from view to view the
  object turns clockwise seen from above.
- The camera at azimuth a, elevation e is at d·(cos e·sin a, -cos e·cos a, sin e), d = 1.8, looking
  at the origin; image right is (cos a, sin a, 0) and image up is +Z at elevation 0 (no roll).
  `camera_to_world` gives the 4x4 matrix (OpenGL convention: columns right, up, back, position).
- Every view is orthographic and frames [-0.55, 0.55]² of its image plane in 768 x 768 pixels,
  centred on the origin: 698.2 pixels per world unit in every view (`project`).
- The picture's object is centred and scaled so that its longer side spans 90% of the frame in
  the 0° view (691 px, 0.99 world units); the other views share that scale.
"""

from __future__ import annotations

import math

import numpy as np

AZIMUTHS = (0, 45, 90, 180, 270, 315)
ELEVATION = 0
IMAGE_SIZE = 768
HALF_EXTENT = 0.55  # the orthographic frame spans [-0.55, 0.55] world units each way
DISTANCE = 1.8
FILL = 0.9  # the picture's object, longer side, as a share of the frame
# MV-Adapter's azimuth 0 looks from +X; its inference script subtracts 90° so that 0 is the picture's
AZIMUTH_OFFSET = -90


def camera_to_world(azimuth: float, elevation: float = ELEVATION, distance: float = DISTANCE) -> np.ndarray:
    """
    The view's 4x4 camera-to-world matrix: columns right, up, back (the camera looks along minus
    this one) and position. The same numbers as MV-Adapter's get_c2w for azimuth + AZIMUTH_OFFSET.
    """
    azimuth_r = math.radians(azimuth + AZIMUTH_OFFSET)
    elevation_r = math.radians(elevation)
    position = distance * np.array(
        [
            math.cos(elevation_r) * math.cos(azimuth_r),
            math.cos(elevation_r) * math.sin(azimuth_r),
            math.sin(elevation_r),
        ]
    )
    look = -position / np.linalg.norm(position)
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    up /= np.linalg.norm(up)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, up, -look, position
    return matrix


def project(points: np.ndarray, azimuth: float, elevation: float = ELEVATION) -> np.ndarray:
    """World points (n, 3) -> pixel coordinates (n, 2) in the view: x right, y down from the top-left."""
    matrix = camera_to_world(azimuth, elevation)
    local = (np.asarray(points, dtype=float) - matrix[:3, 3]) @ matrix[:3, :3]  # (right, up, back)
    scale = IMAGE_SIZE / (2 * HALF_EXTENT)
    return np.stack([IMAGE_SIZE / 2 + local[:, 0] * scale, IMAGE_SIZE / 2 - local[:, 1] * scale], axis=1)


def camera_info() -> dict:
    """The views' shared camera, as the job output and views.json describe it."""
    return {
        "type": "orthographic",
        "image_size": IMAGE_SIZE,
        "half_extent": HALF_EXTENT,
        "pixels_per_unit": round(IMAGE_SIZE / (2 * HALF_EXTENT), 3),
        "distance": DISTANCE,
        "elevation": ELEVATION,
        "up": [0, 0, 1],
        "picture_camera": [0, -1, 0],
        "fill": FILL,
        "azimuth": (
            "degrees; 0 is the picture's own view, redrawn level; positive azimuth moves the camera "
            "counter-clockwise seen from above (up = +Z), towards the picture's right: 90 shows the "
            "picture's right-hand side, 180 the back, 270 its left-hand side"
        ),
        "position": "distance * (cos(elevation) sin(azimuth), -cos(elevation) cos(azimuth), sin(elevation))",
        "pixel": (
            "x = size/2 * (1 + r / half_extent), y = size/2 * (1 - u / half_extent), where r and u are a "
            "point's offsets along the camera's right (cos azimuth, sin azimuth, 0) and up (+Z) axes"
        ),
        "framing": (
            "the picture's object is centred, its longer side scaled to fill of the frame in the 0 degree "
            "view; every view has the same scale"
        ),
    }
