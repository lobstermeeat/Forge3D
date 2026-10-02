"""
Texture options re-sample only the texture: the texture layout the first export of a shape keeps.

A textures job exports several textures of one shape (service.make_textures). TRELLIS.2's ``to_glb``
(o_voxel/postprocess.py) does the same geometry work for every one of them: it fills holes, builds a BVH,
remeshes (``preset.remesh``) or simplifies, unwraps UVs, rasterizes the UV atlas and projects every texel
onto the decoded mesh. Only then does it sample the attribute volume (the texture) and build the material.
Between textures of one shape only the attribute volume (``mesh.attrs``) changes: ``mesh.coords`` and the
decoded geometry come from the same shape latent. So that work is done once:

- ``capturing`` watches one to_glb call. to_glb runs unchanged; the watch keeps what it hands two of its
  helpers: the raster it interpolates (which texels a triangle covers: the mask) and the ``grid`` it
  samples the volume at (each covered texel's position in voxel units, after the BVH projection), with
  the volume's shape.
- ``keep`` puts that together with the mesh to_glb returned (vertices, faces, UVs and normals) into a
  ``TextureLayout``, on the CPU.
- ``rebake`` makes what to_glb would make for another attribute volume of the same shape: it samples the
  new volume at the kept positions with to_glb's own ``grid_sample_3d`` call, then builds the material as
  to_glb does (the same clipping, inpainting and PBR fields) on a copy of the kept mesh.
- ``unfit`` says why a layout can't serve a mesh: another shape, other export options, other voxels.

Trellis2Runtime keeps one layout, for the last retextured shape it exported in full, and runs to_glb in
full whenever the layout doesn't fit or the rebake fails.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch
from PIL import Image

# grid_sample_3d's parameters in order (FlexGEMM's flex_gemm.ops.grid_sample, as o_voxel calls it)
SAMPLE_PARAMETERS = ("feats", "coords", "shape", "grid", "mode")


@dataclass(frozen=True)
class TextureLayout:
    """What to_glb worked out for one shape's texture bake, on the CPU (a few tens of MB at 2048²)."""

    shape: Any  # what the mesh was decoded from (the runtime's Latent): a layout serves that shape only
    options: tuple  # to_glb's options besides the mesh, as options_key gives them
    coords: torch.Tensor  # (L, 3): the voxels the attribute volume is on
    channels: int  # the attribute volume's channels
    mask: torch.Tensor  # (T, T) bool: the texels a triangle covers
    grid: torch.Tensor  # (1, N, 3) float: where those N texels sample the volume, in voxel units
    volume_shape: torch.Size  # grid_sample_3d's shape: (1, C, X, Y, Z)
    mode: Optional[str]  # grid_sample_3d's mode ("trilinear")
    remesh: bool  # to_glb's remesh option, which decides doubleSided
    vertices: np.ndarray  # the mesh to_glb returned (glTF axes), as it returned it
    faces: np.ndarray
    uv: np.ndarray
    normals: np.ndarray


def options_key(options: dict) -> tuple:
    """to_glb's options besides the mesh (texture size, faces, remesh, voxel size, attribute layout...) as a key."""
    return tuple(sorted((str(name), _plain(value)) for name, value in options.items()))


def _plain(value: Any) -> Any:
    """``value`` as plain, comparable Python values."""
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(key), _plain(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, torch.Tensor):
        return ("array", _plain(value.detach().cpu().tolist()))
    if isinstance(value, (np.ndarray, np.generic)):
        return ("array", _plain(value.tolist()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class _Overlay:
    """A module seen through, with some of its attributes replaced (here nvdiffrast's interpolate)."""

    def __init__(self, base: Any, **replaced: Any) -> None:
        self._base = base
        vars(self).update(replaced)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class Watch:
    """What one to_glb call handed its UV interpolation and its attribute sampling (see capturing)."""

    def __init__(self) -> None:
        self.interpolated: list = []  # per interpolate() call: (vertices, faces, mask), or the error reading them
        self.sampled: list = []  # per grid_sample_3d() call: its shape, grid and mode


@contextlib.contextmanager
def capturing(postprocess: Any) -> Iterator[Watch]:
    """
    While active, to_glb (``postprocess``: o_voxel.postprocess) runs as ever, with two of its helpers
    watched: nvdiffrast's ``interpolate`` (uv_raster's), whose raster says which texels a triangle covers,
    and ``grid_sample_3d``, whose ``grid`` says where they sample the attribute volume. Both are called
    exactly as to_glb calls them and return what they return. The watch keeps references to the grid and
    the volume's shape, and the mask as to_glb computes it (``rast[0, ..., 3] > 0``). Both are put back
    afterwards. Without them (not o_voxel's module) to_glb runs unwatched, and nothing can be kept.
    """
    watch = Watch()
    raster = getattr(postprocess, "dr", None)
    sample = getattr(postprocess, "grid_sample_3d", None)
    if raster is None or sample is None:
        yield watch
        return

    def interpolate(attr: Any, rast: Any, tri: Any, *args: Any, **kwargs: Any) -> Any:
        out = raster.interpolate(attr, rast, tri, *args, **kwargs)
        try:
            watch.interpolated.append((int(attr.shape[-2]), int(tri.shape[0]), rast[0, ..., 3] > 0))
        except Exception as err:  # noqa: BLE001 - the watch never gets in to_glb's way
            watch.interpolated.append(err)
        return out

    def grid_sample_3d(*args: Any, **kwargs: Any) -> Any:
        out = sample(*args, **kwargs)
        given = {**dict(zip(SAMPLE_PARAMETERS, args)), **kwargs}
        watch.sampled.append({name: given.get(name) for name in ("shape", "grid", "mode")})
        return out

    # Its own attributes (a module's globals) are put back; ones it only passed on are removed again
    own = {name: name in getattr(postprocess, "__dict__", {}) for name in ("dr", "grid_sample_3d")}
    postprocess.dr = _Overlay(raster, interpolate=interpolate)
    postprocess.grid_sample_3d = grid_sample_3d
    try:
        yield watch
    finally:
        for name, stock in (("dr", raster), ("grid_sample_3d", sample)):
            if own[name]:
                setattr(postprocess, name, stock)
            else:
                delattr(postprocess, name)


def keep(watch: Watch, glb: Any, mesh: Any, shape: Any, options: dict) -> TextureLayout:
    """
    The layout of the to_glb call ``watch`` saw, which returned ``glb`` for ``mesh`` (made from ``shape``,
    exported with ``options``: to_glb's arguments besides the mesh). Raises ValueError unless the watch saw
    exactly one bake, and it matches the mesh to_glb returned: then there is nothing to keep.
    """
    if len(watch.interpolated) != 1 or len(watch.sampled) != 1:
        counts = f"{len(watch.interpolated)} interpolations and {len(watch.sampled)} samplings"
        raise ValueError(f"to_glb's bake was not seen once: {counts}")
    seen = watch.interpolated[0]
    if isinstance(seen, BaseException):
        raise ValueError(f"the raster could not be read: {type(seen).__name__}: {seen}")
    vertex_count, face_count, mask = seen
    sampled = watch.sampled[0]
    grid, volume_shape = sampled["grid"], sampled["shape"]
    size = int(options["texture_size"])
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (size, size):
        raise ValueError(f"the raster is not the texture's {size} x {size} texels")
    texels = int(mask.sum())
    if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != (1, texels, 3):
        raise ValueError(f"the samples don't match the {texels} texels the triangles cover")
    channels = int(mesh.attrs.shape[1])
    if volume_shape is None or len(volume_shape) != 5 or volume_shape[0] != 1 or volume_shape[1] != channels:
        raise ValueError(f"the volume's shape {volume_shape} doesn't match its {channels} channels")
    vertices = np.array(glb.vertices, copy=True)
    faces = np.array(glb.faces, copy=True)
    uv = getattr(getattr(glb, "visual", None), "uv", None)
    normals = np.array(glb.vertex_normals, copy=True)
    if len(vertices) != vertex_count or len(faces) != face_count:
        raise ValueError("to_glb returned another mesh than the one it baked")
    if uv is None or np.shape(uv) != (len(vertices), 2) or normals.shape != vertices.shape:
        raise ValueError("to_glb's mesh has no UVs or normals to keep")
    return TextureLayout(
        shape=shape,
        options=options_key(options),
        coords=_copy(mesh.coords),
        channels=channels,
        mask=_copy(mask),
        grid=_copy(grid),
        volume_shape=torch.Size(volume_shape),
        mode=sampled["mode"],
        remesh=bool(options["remesh"]),
        vertices=vertices,
        faces=faces,
        uv=np.array(uv, copy=True),
        normals=normals,
    )


def _copy(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` copied to the CPU, on its own (not a view into a bigger tensor)."""
    return tensor.detach().to("cpu", copy=True).contiguous()


def unfit(layout: TextureLayout, mesh: Any, shape: Any, options: dict) -> Optional[str]:
    """Why ``layout`` can't serve ``mesh`` (made from ``shape``, exported with ``options``), or None when it can."""
    if layout.shape is not shape:
        return "the layout is another shape's"
    if layout.options != options_key(options):
        return "the layout was made with other export options"
    attrs, coords = mesh.attrs, mesh.coords
    if attrs.dim() != 2 or int(attrs.shape[1]) != layout.channels:
        return "the attribute volume has other channels"
    if tuple(coords.shape) != tuple(layout.coords.shape) or coords.dtype != layout.coords.dtype:
        return "the voxels differ"
    if not torch.equal(layout.coords.to(coords.device), coords):
        return "the voxels differ"
    if int(attrs.shape[0]) != int(coords.shape[0]):
        return "the attribute volume doesn't match its voxels"
    return None


def rebake(layout: TextureLayout, mesh: Any, postprocess: Any) -> Any:
    """
    What to_glb would return for ``mesh``, a mesh ``unfit`` passed for ``layout``: its attribute volume
    sampled at the kept texels' positions with to_glb's own grid_sample_3d call, and the material built as
    to_glb builds it, on a copy of the kept mesh. None of to_glb's geometry work runs.
    """
    import trimesh

    attr_volume, coords = mesh.attrs, mesh.coords
    device = attr_volume.device
    mode = {} if layout.mode is None else {"mode": layout.mode}
    with torch.no_grad():
        # As to_glb samples the volume, at the positions it found for the covered texels
        attrs = torch.zeros(*layout.mask.shape, attr_volume.shape[1], device=device)
        attrs[layout.mask.to(device)] = postprocess.grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=layout.volume_shape,
            grid=layout.grid.to(device),
            **mode,
        )
    material = pbr_material(postprocess.cv2, attrs, layout.mask.numpy(), mesh.layout, layout.remesh)
    return trimesh.Trimesh(
        vertices=layout.vertices.copy(),
        faces=layout.faces.copy(),
        vertex_normals=layout.normals.copy(),
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=layout.uv.copy(), material=material),
    )


def pbr_material(cv2: Any, attrs: torch.Tensor, mask: np.ndarray, attr_layout: dict, remesh: bool) -> Any:
    """
    to_glb's material from the sampled texels ``attrs`` (T, T, C) and the ``mask`` (T, T) of the ones a
    triangle covers. Copied from to_glb in TRELLIS.2's o-voxel/o_voxel/postprocess.py (MIT, Microsoft;
    commit 75fbf01), so that a rebaked texture is to_glb's bit for bit. ``cv2`` is the OpenCV to_glb uses.
    """
    import trimesh

    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout["base_color"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout["metallic"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout["roughness"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout["alpha"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = "OPAQUE"

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    return trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
