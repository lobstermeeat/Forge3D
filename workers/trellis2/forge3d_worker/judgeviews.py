"""
Turntable renders of a textured model for the judge (workers/judge): what a reviewer looks at, in one image.

The judge, a vision-language model, compares textures of one shape the way Phase 7's reviewers did: six
views around the model in a 3x2 grid, the camera 20 degrees above, on a dark background. ``turntable``
draws that grid from to_glb's mesh (glTF axes: +Y up, the front +Z) with its base colour texture and UVs,
as ``mvtexture.render_views`` reads them, using the projection's torch rasteriser (never nvdiffrast), so
it runs on the GPU in the TRELLIS.2 container and on the CPU in tests.

- **Cameras**: view i sits at azimuth 360 i / views degrees (0 on +Z, the front; 90 on +X, as the
  projection's ``view_axes`` and the gallery's turntable turn), ``elevation`` degrees above, looking at
  the centre of the model's bounding box. A perspective camera with a 30 degree field of view, as the
  gallery's render.html, at one distance for every view: the closest at which the bounding box's corners
  stay inside 92% of the frame from every angle, so the model keeps its size as it turns.
- **Light**: plain and the same in every view, so the texture shows as it is: base colour times
  (ambient + a key light from the camera's upper left), plus a faint rim light at grazing angles so a
  dark model's outline still reads on the dark background. Smooth normals, lit from either side (a
  model whose triangles wind inwards looks the same). No gloss or metal: the judge compares colours.
- **Grid**: views left to right, top to bottom, three to a row: the front at the top left and, with six
  views, the back at the bottom left. Each view is rendered ``supersample`` times larger and averaged
  down (in linear light), which smooths edges and the texture's fine detail.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import projection
from .mvtexture import _device, _texture
from .projection import rasterize_depth, rasterize_faces

SIZE = 384  # pixels on a side of each view
VIEWS = 6
ELEVATION = 20.0  # degrees above the horizon
BACKGROUND = 0.12  # sRGB grey, a little darker than the gallery's (0.14)
COLUMNS = 3
SUPERSAMPLE = 2
FOV = 30.0  # degrees
FILL = 0.92  # the bounding box stays inside this share of the frame
# Light, in linear RGB: colour = base colour * (AMBIENT + KEY * max(n . key, 0)) + RIM * (1 - n . view) ** RIM_POWER.
# A surface facing the camera gets about 0.9 of its colour, one facing away from the key light 0.45
AMBIENT = 0.45
KEY = 0.6
# The key light's direction in the camera's terms (right, up, towards the camera): from the upper left
KEY_DIRECTION = (-0.45, 0.55, 0.7)
RIM = 0.15
RIM_POWER = 3.0


def view_azimuths(views: int = VIEWS) -> list[float]:
    """The views' azimuths in degrees: 0 (the front), then evenly round, turning from +Z towards +X."""
    return [360.0 * i / views for i in range(views)]


def _axes(azimuth: float, elevation: float) -> np.ndarray:
    """A camera's right, up and back (towards the camera) unit vectors, (3, 3): the projection's convention."""
    params = torch.tensor([[float(azimuth), float(elevation), 0.0]], dtype=torch.float64)
    return torch.stack([axis[0] for axis in projection.view_axes(params)]).numpy()


def _distance(corners: np.ndarray, cameras: list, fov: float, fill: float) -> float:
    """The closest distance at which every corner stays inside ``fill`` of the frame from every camera."""
    reach = math.tan(math.radians(fov) / 2) * fill
    distance = 0.0
    for right, up, back in cameras:
        depth = corners @ back
        distance = max(
            distance,
            float((depth + np.abs(corners @ right) / reach).max()),
            float((depth + np.abs(corners @ up) / reach).max()),
        )
    return distance


def _view(
    verts: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    face_normals: torch.Tensor,
    uv: torch.Tensor,
    texture: torch.Tensor,
    axes: torch.Tensor,
    distance: float,
    size: int,
    fov: float,
    background: torch.Tensor,
    light: tuple[float, float, float, float],
) -> torch.Tensor:
    """One view, (size, size, 3) in linear light, from a camera at ``distance`` along ``axes``' back vector."""
    right, up, back = axes
    focal = 1.0 / math.tan(math.radians(fov) / 2)
    # Distance from the camera along its view: positive for every point inside the bounding box
    w = distance - verts @ back
    xy = torch.stack(
        [(focal * (verts @ right) / w + 1) * size / 2, (1 - focal * (verts @ up) / w) * size / 2], -1
    )
    s = 1.0 / w  # affine in screen space, so the rasteriser interpolates perspective-correctly
    zbuf = rasterize_depth(xy, w, s, faces, size, size)
    face, bary = rasterize_faces(xy, w, s, faces, zbuf)
    hit = face >= 0
    corners = faces[face.clamp_min(0)]  # (H, W, 3)

    def interpolate(values: torch.Tensor) -> torch.Tensor:
        return (bary[..., None] * values[corners]).sum(-2)

    at = interpolate(uv)  # v up
    grid = torch.stack([at[..., 0] * 2 - 1, (1 - at[..., 1]) * 2 - 1], -1)[None]
    albedo = F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)

    point = interpolate(verts)
    to_camera = distance * back - point
    to_camera = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    normal = interpolate(normals)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # Lit from either side: a triangle the camera sees from behind (wound inwards) turns its normal round
    facing = torch.where((face_normals[face.clamp_min(0)] * to_camera).sum(-1) < 0, -1.0, 1.0)
    normal = normal * facing[..., None]

    ambient, key, rim, rim_power = light
    key_direction = KEY_DIRECTION[0] * right + KEY_DIRECTION[1] * up + KEY_DIRECTION[2] * back
    key_direction = key_direction / key_direction.norm()
    lit = ambient + key * (normal @ key_direction).clamp_min(0)
    grazing = (1 - (normal * to_camera).sum(-1).clamp(0, 1)) ** rim_power
    colour = (albedo * lit[..., None] + rim * grazing[..., None]).clamp(0, 1)
    return torch.where(hit[..., None], colour, background)


def turntable(
    mesh: Any,
    size: int = SIZE,
    views: int = VIEWS,
    elevation: float = ELEVATION,
    background: float = BACKGROUND,
    *,
    columns: int = COLUMNS,
    supersample: int = SUPERSAMPLE,
    fov: float = FOV,
    fill: float = FILL,
    ambient: float = AMBIENT,
    key: float = KEY,
    rim: float = RIM,
    rim_power: float = RIM_POWER,
    device: Optional[Any] = None,
) -> Image.Image:
    """
    ``views`` views of ``mesh`` (to_glb's trimesh: glTF axes, +Y up, the front +Z, a base colour texture
    with UVs) round it at ``elevation`` degrees, each ``size`` pixels square, in a grid ``columns`` wide
    (3 x 2 for six views): an RGB image, ``background`` (sRGB grey) where nothing is. The light's
    knobs (``ambient``, ``key``, ``rim``, ``rim_power``) are keyword arguments; ambient 1 and the rest 0
    show the texture's own colours. The mesh isn't changed.
    """
    if size < 1 or views < 1 or columns < 1 or supersample < 1:
        raise ValueError("size, views, columns and supersample must be at least 1")
    device = _device(device)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("the mesh has no usable vertices")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3 or len(faces_np) == 0:
        raise ValueError("the mesh has no triangles")
    if faces_np.min() < 0 or faces_np.max() >= len(vertices):
        raise ValueError("the mesh's faces don't match its vertices")
    rgb, _, uv_np = _texture(mesh)

    low, high = vertices.min(0), vertices.max(0)
    if float((high - low).max()) <= 0:
        raise ValueError("the mesh is a point")
    centre = (low + high) / 2
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]) - centre
    cameras = [_axes(azimuth, elevation) for azimuth in view_azimuths(views)]
    distance = _distance(corners, cameras, fov, fill)

    big = size * supersample
    with torch.no_grad():
        verts = torch.tensor(vertices - centre, dtype=torch.float32, device=device)
        faces = torch.tensor(faces_np, dtype=torch.long, device=device)
        a, b, c = (verts[faces[:, i]] for i in range(3))
        face_normals = torch.cross(b - a, c - a, dim=-1)
        face_normals = face_normals / face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        normals = projection._welded_normals(verts, faces)
        uv = torch.tensor(uv_np, dtype=torch.float32, device=device)
        # Filtered in linear light, as a GPU filters an sRGB texture
        texture = projection._srgb_to_linear(torch.tensor(rgb, device=device)).permute(2, 0, 1)[None]
        grey = projection._srgb_to_linear(torch.tensor(float(background), device=device))
        back = grey.expand(3).clone()
        light = (float(ambient), float(key), float(rim), float(rim_power))
        rows = math.ceil(views / columns)
        sheet = np.full((rows * size, columns * size, 3), round(float(background) * 255), np.uint8)
        for number, axes in enumerate(cameras):
            axes_t = torch.tensor(axes, dtype=torch.float32, device=device)
            linear = _view(verts, faces, normals, face_normals, uv, texture, axes_t, distance, big, fov, back, light)
            if supersample > 1:
                linear = F.avg_pool2d(linear.permute(2, 0, 1)[None], supersample)[0].permute(1, 2, 0)
            srgb = projection._linear_to_srgb(linear)
            tile = (srgb.cpu().numpy() * 255 + 0.5).astype(np.uint8)
            row, column = divmod(number, columns)
            sheet[row * size : (row + 1) * size, column * size : (column + 1) * size] = tile
    return Image.fromarray(sheet, "RGB")
