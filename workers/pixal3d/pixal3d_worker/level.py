"""
Levelling: Pixal3D builds in the picture's camera frame, production's models stand level.

Pixal3D's grid is tied to its main camera, which it always puts at the canonical front pose F (every
view's pose is taken relative to the main view, ``compute_relative_calc_mat``: calc_mat_0 == F whatever
pose the main view claims). A picture taken from above therefore comes out as if the camera had been
level: the object leans towards the viewer by the picture's elevation (a donut pictured from 29 degrees
up stands 29 degrees off the table; a bowl of ramen pictured from 45 stands on edge).

The picture's elevation is known cheaply. TRELLIS.2 keeps objects upright as pictured, and
``forge3d_worker.projection`` finds the picture's camera against a TRELLIS.2 model by silhouette search:
production's preview ('512', the same seed as the final) is such a model. ``estimate_pose`` runs that
search and applies the projection's own gate (silhouette IoU, no rival camera seeing another shape);
``level_matrix`` is the rotation that puts the object back on its feet, and ``level`` applies it to
``to_glb``'s mesh before the projection paints the picture on (the projection finds the camera again, so
it sees the picture's real elevation then).

Frames: a GLB from ``_ViewAlignedOVoxel`` has +Y up and the picture's camera on +Z, level. The real
camera, in the frame of a level object (the projection's convention, ``projection.view_axes``: azimuth
about +Y, 0 on +Z; elevation above the horizon; roll about the view axis), has axes right, up and back.
Pixal3D put those axes on +X, +Y and +Z, so a vertex's level-world position is C @ p, where C's columns
are the camera's axes. Azimuth is left at 0: the pictured side keeps facing +Z.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

# Tilts below this are left alone (the search's own resolution is coarser than this)
MIN_TILT_DEG = 1.0
# The projection's search clamps elevation to [-60, 85] and roll to +-25; the same bounds here
MAX_ELEVATION_DEG = 85.0
MAX_ROLL_DEG = 25.0
# A camera found this far below the horizon isn't trusted: a silhouette seen from +e and from -e differ
# little for most objects (a plain frustum pictured from +20 was placed at -20 in the tests), pictures
# of objects are taken from above or level, and levelling by the wrong sign doubles the tilt instead of
# removing it. Phase 5's twenty pictures were placed between -5 and +48 degrees
MIN_ELEVATION_DEG = -10.0


def camera_axes(elevation_deg: float, roll_deg: float = 0.0, azimuth_deg: float = 0.0) -> np.ndarray:
    """
    Right, up and back unit vectors (columns of a 3x3 matrix) of the projection's camera at this
    azimuth, elevation and roll, in the GLB frame (+Y up, azimuth 0 on +Z): ``projection.view_axes`` in
    numpy, one view at a time.
    """
    az, el, roll = (math.radians(v) for v in (azimuth_deg, elevation_deg, roll_deg))
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    back = np.array([sa * ce, se, ca * ce])
    right0 = np.array([ca, 0.0, -sa])
    up0 = np.array([-se * sa, ce, -se * ca])
    cr, sr = math.cos(roll), math.sin(roll)
    right = cr * right0 + sr * up0
    up = cr * up0 - sr * right0
    return np.stack([right, up, back], axis=1)


def level_matrix(elevation_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """
    The 4x4 transform that levels a camera-aligned GLB whose picture was taken from ``elevation_deg``
    above the horizon with the camera rolled by ``roll_deg``: a rotation about the camera's right axis
    (+X) by the elevation, and about its view axis (+Z) by the roll, tipping the object's top away from
    the viewer by the elevation. The pictured side still faces +Z.
    """
    if not (math.isfinite(elevation_deg) and math.isfinite(roll_deg)):
        raise ValueError("elevation and roll must be finite")
    if abs(elevation_deg) > MAX_ELEVATION_DEG:
        raise ValueError(f"elevation must be within +-{MAX_ELEVATION_DEG:g} degrees, got {elevation_deg}")
    if abs(roll_deg) > MAX_ROLL_DEG:
        raise ValueError(f"roll must be within +-{MAX_ROLL_DEG:g} degrees, got {roll_deg}")
    matrix = np.eye(4)
    matrix[:3, :3] = camera_axes(elevation_deg, roll_deg)
    matrix[np.abs(matrix) < 1e-12] = 0.0
    return matrix


def needs_levelling(elevation_deg: float, roll_deg: float = 0.0) -> bool:
    return abs(elevation_deg) >= MIN_TILT_DEG or abs(roll_deg) >= MIN_TILT_DEG


def level(glb: Any, elevation_deg: float, roll_deg: float = 0.0) -> dict:
    """
    Levels ``to_glb``'s trimesh in place (vertices and normals turned, then re-centred on its bounding
    box, where the voxel grid had it) and returns a note for the result: whether it was turned and by
    how much. Tilts under MIN_TILT_DEG are left alone.
    """
    note = {"applied": False, "elevation": round(float(elevation_deg), 1), "roll": round(float(roll_deg), 1)}
    if not needs_levelling(elevation_deg, roll_deg):
        return note
    glb.apply_transform(level_matrix(elevation_deg, roll_deg))
    bounds = np.asarray(glb.bounds, dtype=np.float64)
    centre = (bounds[0] + bounds[1]) / 2
    shift = np.eye(4)
    shift[:3, 3] = -centre
    glb.apply_transform(shift)
    note.update(applied=True, recentred=[round(float(v), 4) for v in -centre])
    return note


def estimate_pose(glb: Any, picture: Any, device: Optional[Any] = None) -> dict:
    """
    The picture's camera against a level model (TRELLIS.2's preview), by the projection's own search,
    polished against the exact silhouette and gated the way the projection gates itself: the model must
    be the pictured shape (silhouette IoU at least ``projection.MIN_IOU``), and no camera far from the
    winner may fit as well while seeing a different shape. Never raises.

    Returns ``{"applied", "reason", "pose", "iou", "colour", "runner_up", "seconds"}``: ``pose`` is the
    projection's view dict (azimuth, elevation, roll, fov, scale, shift), ``applied`` whether it passed
    the gate. ``tilt`` reads the elevation and roll to level by (0, 0 when it didn't).
    """
    import torch

    from forge3d_worker import projection as P

    report: dict = {"applied": False, "reason": "", "pose": None, "iou": None, "colour": None}
    started = time.perf_counter()
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            model = P._load_model(glb, torch.device(device))
            pic = P._load_picture(picture, torch.device(device))
            pose = P._estimate_pose(model, pic, lambda name: None)
            params = P._polish(model, pic, pose.params)
            _, zbuf = P._render_depth(model, pic, params)
            iou = float(P._iou(torch.isfinite(zbuf).float(), pic.mask))
            report.update(
                pose=P._view(params[0]).as_dict(),
                iou=round(iou, 4),
                colour=round(pose.colour, 4),
                runner_up=pose.runner_up,
            )
            if iou < P.MIN_IOU:
                raise P._Skip(f"the silhouettes don't match well enough (IoU {iou:.3f} < {P.MIN_IOU})")
            for rival in pose.rivals:
                angle = float(P._angle_between(params, rival[None])[0])
                difference = P._shape_difference(zbuf, P._render_depth(model, pic, rival[None])[1])
                if difference > P.SAME_SHAPE:
                    raise P._Skip(
                        f"ambiguous camera: a view {angle:.0f} degrees away fits as well but sees a different shape"
                    )
            elevation = float(report["pose"]["elevation"])
            if elevation < MIN_ELEVATION_DEG:
                raise P._Skip(f"the camera was placed {-elevation:.0f} degrees below the horizon, which isn't trusted")
            report["applied"] = True
            report["reason"] = "found"
    except P._Skip as skip:
        report["reason"] = str(skip)
    except Exception as err:  # noqa: BLE001 - a failed estimate means no levelling, never a failed job
        report["reason"] = f"error: {type(err).__name__}: {err}"
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


def tilt(report: Optional[dict]) -> tuple[float, float]:
    """(elevation, roll) to level by, from an estimate_pose report: (0, 0) unless it passed the gate."""
    if not report or not report.get("applied") or not report.get("pose"):
        return 0.0, 0.0
    pose = report["pose"]
    elevation = float(pose.get("elevation", 0.0))
    roll = float(pose.get("roll", 0.0))
    elevation = max(-MAX_ELEVATION_DEG, min(MAX_ELEVATION_DEG, elevation))
    roll = max(-MAX_ROLL_DEG, min(MAX_ROLL_DEG, roll))
    return elevation, roll
