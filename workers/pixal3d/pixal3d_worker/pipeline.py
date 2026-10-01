"""
The GPU side for Pixal3D (TencentARC, MIT): a picture, or a picture and views around it, to a GLB.

Pixal3D is TRELLIS.2 with pixel-aligned conditioning: every voxel gets the DINOv3 features of the pixel
it projects to, so the picture's camera matters (MoGe-2 estimates it for a single picture; a view set
brings its own). Its multi-view weights fuse several views of the same object, which is how the side a
picture doesn't show can come from the multiview worker's drawings instead of being invented.

Everything after generation is production's (``Trellis2Runtime.export``: to_glb, unpremultiply, the
picture's projection, shading normals), so a Pixal3D GLB is packed and judged like a TRELLIS.2 one.
The decoders are TRELLIS.2's own files. Pixal3D's meshes come out in a camera-aligned frame, which
``_ViewAlignedOVoxel`` turns upright with the picture's side towards +Z, as production's GLBs are, and
levels by the picture's camera elevation (``level.py``): Pixal3D has no notion of gravity, so an object
pictured from above would otherwise lean towards the viewer by that angle. The elevation comes from
TRELLIS.2's '512' preview of the same picture and seed (the preview the user approves), against which
the projection's camera search places the picture.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import math
import os
import time
import traceback
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from forge3d_worker.inputs import InputError
from forge3d_worker.pipeline import CUTOUT, Trellis2Runtime, _configure_environment, _one_line, is_out_of_memory
from forge3d_worker.pipeline import MODEL_DIR as TRELLIS2_MODEL_DIR
from forge3d_worker.settings import PRESETS, Preset

from . import cameras, level, neighborhood
from .views import (
    MAIN_PICTURE,
    MAIN_VIEW,
    View,
    ViewCamera,
    bundle,
    fit_camera,
    has_cutout,
    order,
    place_like,
    touches_edge,
)

MODEL_DIR = os.environ.get("PIXAL3D_MODEL_DIR", "/models/pixal3d")
DINO_DIR = os.environ.get("PIXAL3D_DINO_DIR", "/models/dinov3-vitl16")
NAF_DIR = os.environ.get("PIXAL3D_NAF_DIR", "/opt/naf")
NAF_WEIGHTS = os.environ.get("PIXAL3D_NAF_WEIGHTS", f"{MODEL_DIR}/naf/naf_release.pth")
MOGE_WEIGHTS = os.environ.get("PIXAL3D_MOGE_WEIGHTS", f"{MODEL_DIR}/moge-2-vitl/model.pt")
CONFIG_SINGLE = "pipeline.json"
CONFIG_MULTIVIEW = "pipeline_mv.json"

# Each stage's image condition, as inference.py and inference_mv.py build them (DINOv3 ViT-L/16; NAF
# upsamples its patch features for the shape and texture stages)
STAGES = {
    "image_cond_model_ss": {"image_size": 512, "grid_resolution": 16},
    "image_cond_model_shape_512": {
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "image_cond_model_shape_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "image_cond_model_tex_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}
# Production's presets name TRELLIS.2 pipelines. Pixal3D only has cascades: its texture model works at
# 1024, so a preview is the same generation as the final (same seed, same mesh), exported lighter.
PIXAL3D_PIPELINE = {"512": "1024_cascade", "1024_cascade": "1024_cascade", "1536_cascade": "1536_cascade"}
# A single picture's field of view when MoGe-2 is off (PIXAL3D_FOV_DEG); upstream suggests 0.2 rad
DEFAULT_FOV_DEG = math.degrees(0.2)
# Where the picture's camera elevation comes from, to level the model (level.py)
LEVEL_PREVIEW = "preview"  # TRELLIS.2's '512' preview of the picture at the job's seed, the projection's search
LEVEL_GIVEN = "given"  # the caller's ``given_tilt`` (elevation, roll), e.g. a pose recorded earlier
LEVEL_NONE = "none"  # camera-aligned, as Pixal3D builds it
LEVEL_CHOICES = (LEVEL_PREVIEW, LEVEL_GIVEN, LEVEL_NONE)
# (elevation, roll) travels from generate() to export() on the mesh, like the cutout
TILT = "forge3d_tilt"
NO_TILT = (0.0, 0.0)


class WeightMismatchError(RuntimeError):
    """A checkpoint didn't match its model. Upstream loads with strict=False and would silently continue."""


def checkpoint_paths(model_dir: str, config_file: str) -> dict[str, str]:
    """Each model's checkpoint (without extension), as Pipeline.from_pretrained resolves it."""
    with open(os.path.join(model_dir, config_file)) as f:
        paths = json.load(f)["args"]["models"]
    return {name: path if os.path.isabs(path) else os.path.join(model_dir, path) for name, path in paths.items()}


def verify_weights(models: dict[str, Any], model_dir: str, config_file: str) -> list[str]:
    """Like forge3d_worker.pipeline.verify_weights, for a Pixal3D config whose paths may be absolute."""
    from safetensors import safe_open

    paths = checkpoint_paths(model_dir, config_file)
    problems, notes = [], []
    for name, model in models.items():
        with safe_open(f"{paths[name]}.safetensors", framework="pt") as ckpt:
            stored = set(ckpt.keys())
        missing = {n for n, _ in model.named_parameters()} - stored
        unexpected = stored - set(model.state_dict().keys())
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            problems.append(f"{name} is missing {len(missing)} parameters ({sample})")
        if unexpected:
            notes.append(f"{name} checkpoint has {len(unexpected)} unused tensors")
    if problems:
        raise WeightMismatchError("; ".join(problems))
    return notes


class _ViewAlignedOVoxel:
    """
    o_voxel as Trellis2Runtime._export uses it, with to_glb's result turned into production's frame and
    levelled by ``postprocess.tilt`` (elevation, roll), which export() sets per mesh.
    """

    def __init__(self, o_voxel: Any) -> None:
        self.postprocess = _ViewAlignedPostprocess(o_voxel.postprocess)


class _ViewAlignedPostprocess:
    def __init__(self, postprocess: Any) -> None:
        self._postprocess = postprocess
        self.tilt: tuple[float, float] = NO_TILT
        self.last_level: Optional[dict] = None

    def to_glb(self, *args: Any, **kwargs: Any) -> Any:
        glb = self._postprocess.to_glb(*args, **kwargs)
        glb.apply_transform(cameras.GLB_FROM_TO_GLB)
        # Before the projection paints the picture on: it finds the camera again, now at its real elevation
        self.last_level = level.level(glb, *self.tilt)
        return glb

    def __getattr__(self, name: str) -> Any:
        return getattr(self._postprocess, name)


class _SharedPretrained:
    """
    Stands in for DINOv3ViTModel while the four stage extractors are built: they all load the same
    frozen backbone, so they share one copy (one 1.2 GB load and one place on the GPU instead of four).
    """

    def __init__(self, cls: Any, path: str) -> None:
        self.cls, self.path, self.model = cls, path, None

    def from_pretrained(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            self.model = self.cls.from_pretrained(self.path, *args, **kwargs)
        return self.model


def load_naf(naf_dir: str = NAF_DIR, weights: str = NAF_WEIGHTS) -> Any:
    """
    NAF from its pinned clone and pinned weights. Upstream fetches both at run time
    (``torch.hub.load("valeoai/NAF", "naf", pretrained=True)``); this is the same entry point, offline.
    """
    import torch

    neighborhood.install()
    model = torch.hub.load(naf_dir, "naf", source="local", pretrained=False, trust_repo=True)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    model.requires_grad_(False)
    return model


def clear_cuda_error() -> Optional[str]:
    """
    Resets the CUDA runtime's sticky "last error", returning the message it held, or None.

    CuMesh checks its own cudaMalloc calls and raises a RuntimeError when one fails, but never calls
    cudaGetLastError, which is what resets the runtime's per-thread error flag. Torch's next kernel
    launch check then reports that stale flag as "CUDA error: out of memory" with the GPU all but
    empty, and a retry with the models off the GPU dies on its first tensor op (the 11 books' export
    in the first run; the same pattern in Phase 5's logs, where only the second retry got through,
    because the first had consumed the flag). A tiny launch here consumes it instead: the check
    raises, the flag is clear, and the error is swallowed. Nothing to do without CUDA.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").fill_(1)  # the launch check is what reads and resets the flag
        torch.cuda.synchronize()
        return None
    except RuntimeError as err:
        return _one_line(err)


PROJ_MODULE = "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj"


@contextlib.contextmanager
def _absolute_poses():
    """
    While this is active, Pixal3D's multi-view extractor projects through the views' cameras as given
    instead of snapping the main view to its canonical front F (``compute_relative_calc_mat``: calc_mat_i
    = F @ inv(C_0) @ C_i, which makes a lone view's pose irrelevant). The experiment's way to tell the
    model where the picture's camera really was.
    """
    import importlib

    module = importlib.import_module(PROJ_MODULE)
    original = module.compute_relative_calc_mat

    def as_given(transform_matrix, distance, front_view_transform_matrix):
        return transform_matrix

    module.compute_relative_calc_mat = as_given
    try:
        yield
    finally:
        module.compute_relative_calc_mat = original


def estimate_fov(moge: Any, picture: Image.Image) -> float:
    """
    Horizontal field of view (radians) of a picture cropped by preprocess_image, from MoGe-2, as
    inference.py's get_camera_params_wild_moge computes it.
    """
    import numpy as np
    import torch

    rgb = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
    device = next(moge.parameters()).device
    with torch.no_grad():
        output = moge.infer(torch.from_numpy(rgb).permute(2, 0, 1).to(device))
    fx = float(output["intrinsics"].squeeze()[0, 0]) * picture.width
    return 2 * math.atan(picture.width / (2 * fx))


class Pixal3DRuntime(Trellis2Runtime):
    """
    Holds the loaded Pixal3D pipeline between jobs. ``multiview`` picks the weights: Pixal3D's
    single-view set (pictures alone) or its multi-view set (``generate_views``; a picture alone then
    runs as a one-view set). ``main`` says which picture is the multi-view main view (views.MAIN_*),
    ``azimuths`` which of a job's views are used (None: all).

    ``level`` says how a single picture's model is levelled (LEVEL_*): by default the picture's camera
    is found against TRELLIS.2's '512' preview (loaded here too, kept off the GPU between uses) and the
    model is turned by its elevation at export; with LEVEL_GIVEN the caller sets ``given_tilt``. With
    ``posed`` (an experiment, multi-view weights only) the elevation goes into the picture's camera
    instead, as an absolute pose, and nothing is turned afterwards.
    """

    pipeline_used: Optional[str] = None
    views_used: int = 0
    level: str = LEVEL_PREVIEW
    posed: bool = False
    trellis2: Optional[Trellis2Runtime] = None  # loaded with LEVEL_PREVIEW; without it, nothing is levelled
    given_tilt: tuple[float, float] = NO_TILT
    last_camera: Optional[dict] = None
    last_pose: Optional[dict] = None  # level.estimate_pose's report for the last single picture
    last_level: Optional[dict] = None  # level.level's note for the last export
    last_preview: Optional[bytes] = None  # the TRELLIS.2 preview GLB the pose was found against (unpacked)

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        multiview: bool = True,
        main: str = MAIN_VIEW,
        azimuths: Optional[Sequence[float]] = None,
        fov_deg: Optional[float] = None,
        low_vram: Optional[bool] = None,
        level: str = LEVEL_PREVIEW,
        trellis2_model_dir: str = TRELLIS2_MODEL_DIR,
        posed: bool = False,
    ) -> None:
        if main not in (MAIN_VIEW, MAIN_PICTURE):
            raise ValueError(f"main must be {MAIN_VIEW!r} or {MAIN_PICTURE!r}")
        if level not in LEVEL_CHOICES:
            raise ValueError(f"level must be one of {LEVEL_CHOICES}")
        if posed and not multiview:
            raise ValueError("a posed picture needs the multi-view weights (the single-view camera is fixed)")
        _configure_environment()
        neighborhood.install()
        import o_voxel  # noqa: F401 - after the nvdiffrast stand-in is installed
        import torch
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline, Pixal3DMVImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as proj

        started = time.perf_counter()
        self.multiview, self.main, self.level, self.posed = multiview, main, level, posed
        self.azimuths = None if azimuths is None else tuple(float(a) % 360 for a in azimuths)
        config = CONFIG_MULTIVIEW if multiview else CONFIG_SINGLE
        pipeline_class = Pixal3DMVImageTo3DPipeline if multiview else Pixal3DImageTo3DPipeline
        pipeline = pipeline_class.from_pretrained(model_dir, config)
        for note in verify_weights(pipeline.models, model_dir, config):
            print(f"[pixal3d] {note}")

        extractor = proj.DinoV3ProjMultiViewFeatureExtractor if multiview else proj.DinoV3ProjFeatureExtractor
        fusion = {"multiview_fusion": "average"} if multiview else {}
        shared = _SharedPretrained(proj.DINOv3ViTModel, DINO_DIR)
        proj.DINOv3ViTModel = shared
        try:
            for name, args in STAGES.items():
                setattr(pipeline, name, extractor(model_name=DINO_DIR, **args, **fusion).eval())
        finally:
            proj.DINOv3ViTModel = shared.cls
        naf = load_naf()
        for name in STAGES:
            stage = getattr(pipeline, name)
            if stage.use_naf_upsample:
                stage.naf_model = naf

        self.moge = None
        self.fov = math.radians(fov_deg) if fov_deg else None
        if self.fov is None:
            from moge.model.v2 import MoGeModel

            self.moge = MoGeModel.from_pretrained(MOGE_WEIGHTS).eval()

        self._o_voxel = _ViewAlignedOVoxel(o_voxel)
        self.pipeline = pipeline
        if low_vram is None:
            low_vram = os.environ.get("PIXAL3D_LOW_VRAM", "0") == "1"
        pipeline._device = torch.device("cuda")
        if low_vram:
            pipeline.low_vram = True
        else:
            self._restore()
        print(f"[pixal3d] loaded {config} in {time.perf_counter() - started:.1f} s (low VRAM: {low_vram})")

        self.trellis2: Optional[Trellis2Runtime] = None
        if level == LEVEL_PREVIEW:
            started = time.perf_counter()
            self.trellis2 = Trellis2Runtime(trellis2_model_dir)
            # Off the GPU between uses, and in upstream's low-VRAM mode while it runs (each model on the
            # GPU only for its stage): Pixal3D's own models stay resident
            self.trellis2._offload()
            print(f"[pixal3d] loaded TRELLIS.2 for previews in {time.perf_counter() - started:.1f} s")

    # --- Device placement -------------------------------------------------------------------------

    def _modules(self) -> list[Any]:
        pipeline = self.pipeline
        modules = [*pipeline.models.values(), pipeline.rembg_model, self.moge]
        modules += [getattr(pipeline, name) for name in STAGES]
        return [module for module in modules if module is not None]

    def _offload(self) -> None:
        """Upstream's low-VRAM mode: each model on the GPU only while its stage runs."""
        for module in self._modules():
            module.cpu()
        self.pipeline.low_vram = True
        self._free_gpu_memory()

    def _restore(self) -> None:
        """Every model on the GPU, the way inference.py's standard mode puts them."""
        import torch

        self.pipeline.low_vram = False
        self.pipeline.to(torch.device("cuda"))  # flow models, decoders and the background remover
        for name in STAGES:
            getattr(self.pipeline, name).cuda()  # DINOv3 (shared) and NAF move with each stage
        if self.moge is not None:
            self.moge.cuda()

    # --- Generation -------------------------------------------------------------------------------

    def _attempt(self, work: Callable[[], Any]) -> Any:
        """Runs a generation; one that runs out of GPU memory gets one retry in low-VRAM mode."""
        try:
            return work()
        except Exception as err:
            if self.pipeline.low_vram or not is_out_of_memory(err):
                raise
            print(f"[pixal3d] out of GPU memory, retrying in low-VRAM mode: {_one_line(err)}")
        try:
            self._offload()
            return work()
        except BaseException as err:
            traceback.clear_frames(err.__traceback__)
            raise
        finally:
            self._restore()

    def _pixal3d_type(self, preset: Preset) -> str:
        pipeline_type = PIXAL3D_PIPELINE.get(preset.pipeline_type)
        if pipeline_type is None:
            raise ValueError(f"no Pixal3D pipeline for {preset.pipeline_type}")
        return pipeline_type

    def _prepare(self, image: Any) -> tuple[Image.Image, Optional[Image.Image]]:
        try:
            return self._preprocess(image)  # Trellis2Runtime's: upstream's crop, plus the full cutout
        except ValueError as err:
            raise InputError("no object found in the image: use one object on a plain background") from err

    def picture_fov(self, prepared: Image.Image) -> float:
        """The cropped picture's horizontal field of view, radians (MoGe-2, or the fixed one)."""
        if self.moge is None:
            return self.fov or math.radians(DEFAULT_FOV_DEG)
        if self.pipeline.low_vram:
            self.moge.cuda()
        try:
            return estimate_fov(self.moge, prepared)
        finally:
            if self.pipeline.low_vram:
                self.moge.cpu()

    def generate(self, image: Any, preset: Preset, seed: int) -> Any:
        """
        One picture to a mesh, its camera from MoGe-2 (or the fixed field of view), levelled at export by
        the picture's elevation (``level``: from TRELLIS.2's preview at this seed, given, or none).
        """
        pipeline_type = self._pixal3d_type(preset)
        self.pipeline_used, self.views_used = f"pixal3d-{pipeline_type}", 0
        self.last_pose, self.last_preview = None, None
        prepared, cutout = self._prepare(image)
        fov = self.picture_fov(prepared)
        self.last_camera = {"fov_deg": round(math.degrees(fov), 2)}
        print(f"[pixal3d] picture's field of view: {math.degrees(fov):.1f} degrees")
        tilt = self._tilt(image, seed)
        self.last_camera["tilt"] = {"elevation": round(tilt[0], 1), "roll": round(tilt[1], 1), "source": self.level}
        if self.multiview:
            # The multi-view weights with the picture as the only view: the single-view camera (its
            # frame spans [-0.5, 0.5] at the origin, cameras.single_view_distance)
            frame = ViewCamera(half_extent=0.5)
            views = bundle([prepared.convert("RGBA")], [(0.0, 0.0)], frame, fov_deg=math.degrees(fov))
            if self.posed and level.needs_levelling(tilt[0]):
                # The experiment: the picture's real camera, above the front by its elevation, as an
                # absolute pose (the model then has to build a level object seen from above)
                import torch

                distance = float(views["camera_distance"][0, 0])
                posed = cameras.orbit_camera(0.0, tilt[0], distance)
                views["transform_matrix"] = torch.tensor(posed, dtype=torch.float32)[None, None]
                self.last_camera["posed"] = {"elevation": round(tilt[0], 1), "distance": round(distance, 4)}
                print(f"[pixal3d] the picture's camera posed {tilt[0]:.1f} degrees above the front")
                return self._attempt(lambda: self._run(views, pipeline_type, seed, cutout, absolute=True))
            return self._attempt(lambda: self._run(views, pipeline_type, seed, cutout, tilt=tilt))
        camera = {"camera_angle_x": fov, "distance": cameras.single_view_distance(fov), "mesh_scale": 1.0}
        return self._attempt(lambda: self._run_single(prepared, camera, pipeline_type, seed, cutout, tilt))

    def _tilt(self, image: Any, seed: int) -> tuple[float, float]:
        """(elevation, roll) the single picture's model is levelled by, per ``level``."""
        if self.level == LEVEL_GIVEN:
            return (float(self.given_tilt[0]), float(self.given_tilt[1]))
        if self.level != LEVEL_PREVIEW or self.trellis2 is None:
            return NO_TILT
        started = time.perf_counter()
        try:
            report = self._preview_pose(image, seed)
        except Exception as err:  # noqa: BLE001 - a failed preview means no levelling, never a failed job
            report = {"applied": False, "reason": f"error: {type(err).__name__}: {_one_line(err)}", "pose": None}
            traceback.clear_frames(err.__traceback__)
        finally:
            self._free_gpu_memory()  # the preview's buffers, out of Pixal3D's way
        report["seconds"] = round(time.perf_counter() - started, 1)
        self.last_pose = report
        tilt = level.tilt(report)
        pose = report.get("pose") or {}
        print(
            f"[pixal3d] the picture's camera against TRELLIS.2's preview: "
            f"{'found' if report.get('applied') else 'not used'} (elevation {pose.get('elevation')}, "
            f"roll {pose.get('roll')}, IoU {report.get('iou')}, {report['seconds']} s): {report.get('reason')}"
        )
        return tilt

    def _preview_pose(self, image: Any, seed: int) -> dict:
        """
        TRELLIS.2's preview of the picture (production's preset, the same seed as the final), exported
        as production exports it, and the picture's camera found against it (level.estimate_pose).
        """
        import trimesh

        preset = PRESETS["preview"]
        mesh = self.trellis2.generate(image, preset, seed)
        cutout = getattr(mesh, CUTOUT, None)
        raw, _ = self.trellis2.export(mesh, preset)
        del mesh
        self.last_preview = raw
        glb = trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh")
        report = level.estimate_pose(glb, cutout)
        report["preview"] = {"pipeline": self.trellis2.pipeline_used or preset.pipeline_type, "bytes": len(raw)}
        return report

    def generate_views(
        self,
        image: Any,
        views: Sequence[View],
        preset: Preset,
        seed: int,
        camera: ViewCamera = ViewCamera(),
    ) -> Any:
        """The picture and views around it to a mesh (the multi-view weights)."""
        if not self.multiview:
            raise RuntimeError("this runtime has Pixal3D's single-view weights; views need the multi-view ones")
        pipeline_type = self._pixal3d_type(preset)
        prepared, cutout = self._prepare(image)
        chosen = order(views, self.azimuths)
        front = chosen[0] if chosen and chosen[0].is_front else None
        others = [view for view in chosen if not view.is_front]
        size = (front or chosen[0]).image.width if chosen else 768
        if front is not None and self.main == MAIN_VIEW:
            main = self._matted(front.image)
            used = 1 + len(others)
        else:  # the picture, framed where the redrawn front has the object
            picture = cutout if cutout is not None and has_cutout(cutout) else self._matted(prepared)
            reference = self._matted(front.image) if front is not None else None
            main = place_like(picture, size, reference)
            used = len(others)
        images = [main] + [self._matted(view.image) for view in others]
        angles = [(0.0, 0.0)] + [(view.azimuth, view.elevation) for view in others]
        for (azimuth, elevation), picture in zip(angles, images):
            if touches_edge(picture):
                print(f"[pixal3d] the view at azimuth {azimuth:g}, elevation {elevation:g} reaches its frame's edge")
        # The object must fill Pixal3D's cube as its training objects did: the views' world is
        # rescaled by what their silhouettes show (fit_camera)
        camera, fit = fit_camera(images, angles, camera)
        self.pipeline_used, self.views_used = f"pixal3d-mv-{pipeline_type}", used
        self.last_camera = {"views": [list(a) for a in angles], **fit}
        print(f"[pixal3d] {len(images)} views (main: {'view' if used == len(images) else 'picture'}): {angles}")
        print(f"[pixal3d] the views' frame: {json.dumps(fit)}")
        packed = bundle(images, angles, camera)
        return self._attempt(lambda: self._run(packed, pipeline_type, seed, cutout))

    def _matted(self, image: Image.Image) -> Image.Image:
        """A view as an RGBA cutout: its own alpha, or the background remover's (never cropped)."""
        if has_cutout(image):
            return image.convert("RGBA")
        pipeline = self.pipeline
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        try:
            return pipeline.rembg_model(image.convert("RGB")).convert("RGBA")
        finally:
            if pipeline.low_vram:
                pipeline.rembg_model.cpu()

    def _run(
        self,
        views: dict,
        pipeline_type: str,
        seed: int,
        cutout: Optional[Image.Image],
        tilt: tuple[float, float] = NO_TILT,
        absolute: bool = False,
    ) -> Any:
        if absolute:
            with _absolute_poses():
                mesh = self.pipeline.run_mv(views, seed=seed, pipeline_type=pipeline_type)[0]
        else:
            mesh = self.pipeline.run_mv(views, seed=seed, pipeline_type=pipeline_type)[0]
        return self._keep(mesh, cutout, tilt)

    def _run_single(
        self,
        prepared: Image.Image,
        camera: dict,
        pipeline_type: str,
        seed: int,
        cutout: Optional[Image.Image],
        tilt: tuple[float, float] = NO_TILT,
    ) -> Any:
        mesh = self.pipeline.run(
            prepared, camera_params=camera, seed=seed, pipeline_type=pipeline_type, preprocess_image=False
        )[0]
        return self._keep(mesh, cutout, tilt)

    @staticmethod
    def _keep(mesh: Any, cutout: Optional[Image.Image], tilt: tuple[float, float] = NO_TILT) -> Any:
        try:
            if cutout is not None:
                setattr(mesh, CUTOUT, cutout)  # what the final's projection paints from
            setattr(mesh, TILT, tuple(tilt))  # what export() levels by
        except (AttributeError, TypeError):
            pass
        return mesh

    # --- Export -----------------------------------------------------------------------------------

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]:
        """Production's export, with the mesh levelled by its tilt before the picture is painted on."""
        postprocess = self._o_voxel.postprocess
        postprocess.tilt = tuple(getattr(mesh, TILT, NO_TILT))
        postprocess.last_level = None
        self.last_level = None
        try:
            return super().export(mesh, preset)
        finally:
            self.last_level = postprocess.last_level
            postprocess.tilt = NO_TILT
            if self.last_level is not None:
                print(f"[pixal3d] levelled: {json.dumps(self.last_level)}")

    @staticmethod
    def _free_gpu_memory() -> None:
        """
        Production's (reference cycles collected, torch's cached blocks handed back to CUDA), plus the
        runtime's stale error flag cleared, so a retry after CuMesh ran out of memory can run at all.
        """
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        stale = clear_cuda_error()
        if stale:
            print(f"[pixal3d] cleared a stale CUDA error before going on: {stale}")


def runtime_from_env() -> Pixal3DRuntime:
    """The runtime a worker container builds, configured by PIXAL3D_* variables."""
    azimuths = os.environ.get("PIXAL3D_AZIMUTHS")
    fov = os.environ.get("PIXAL3D_FOV_DEG")
    return Pixal3DRuntime(
        multiview=os.environ.get("PIXAL3D_WEIGHTS", "multiview") == "multiview",
        main=os.environ.get("PIXAL3D_MAIN", MAIN_VIEW),
        azimuths=[float(a) for a in azimuths.split(",")] if azimuths else None,
        fov_deg=float(fov) if fov else None,
        level=os.environ.get("PIXAL3D_LEVEL", LEVEL_PREVIEW),
        trellis2_model_dir=os.environ.get("TRELLIS2_MODEL_DIR", TRELLIS2_MODEL_DIR),
    )
