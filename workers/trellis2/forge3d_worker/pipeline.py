"""The GPU side: loads stock TRELLIS.2 once and turns an image into a GLB."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import traceback
from typing import Any, Optional, Sequence

from PIL import Image

from . import multiview, normals, projection, uv_raster
from .inputs import InputError, View
from .settings import MULTIVIEW, MultiView, Preset

MODEL_DIR = os.environ.get("TRELLIS2_MODEL_DIR", "/models/TRELLIS.2-4B")
# The background-removed picture (RGBA, full frame) travels from generate() to export() on the mesh
CUTOUT = "forge3d_cutout"


class WeightMismatchError(RuntimeError):
    """A checkpoint didn't match its model. Upstream loads with strict=False and would silently continue."""


def _configure_environment() -> None:
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Weights are baked into the image; never reach out to Hugging Face at runtime
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # nvdiffrast is research-only; o_voxel's texture bake uses this stand-in instead
    uv_raster.install()


def verify_weights(models: dict[str, Any], model_dir: str) -> list[str]:
    """
    Fail loudly if a checkpoint lacks any of its model's parameters (upstream would leave
    them at random init). Extra tensors in a checkpoint are only reported.
    """
    from safetensors import safe_open

    with open(os.path.join(model_dir, "pipeline.json")) as f:
        paths = json.load(f)["args"]["models"]
    problems, notes = [], []
    for name, model in models.items():
        file = os.path.join(model_dir, f"{paths[name]}.safetensors")
        with safe_open(file, framework="pt") as ckpt:
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


# The 1024 texture pass sometimes returns colour already multiplied by a spurious alpha, on up to
# half a model's texels. The GLB is opaque, so those texels showed up dark (a blotchy dragon's skin).
# Dividing by alpha is capped at 1/ALPHA_FLOOR so near-transparent texels' noise isn't blown up.
ALPHA_FLOOR = 0.25


def unpremultiply(texture: Image.Image, floor: float = ALPHA_FLOOR) -> Image.Image:
    """Opaque RGB: colour divided by its alpha in linear light. Fully opaque texels stay as they are."""
    import numpy as np

    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear = np.clip(linear / np.maximum(alpha, floor), 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    out = np.where(alpha < 1.0, srgb, rgb)
    return Image.fromarray(np.round(out * 255).astype(np.uint8), "RGB")


def _one_line(err: BaseException) -> str:
    """An error's message on one line, whatever its own line breaks (CuMesh's span several)."""
    return " ".join(str(err).split())


class _KeepCutout:
    """
    Stands in for the background remover while preprocess_image runs and keeps what it returns: the
    whole picture with its alpha, before upstream crops it and premultiplies it onto black.
    """

    def __init__(self, remover: Any) -> None:
        self.remover = remover
        self.cutout: Optional[Image.Image] = None

    def __call__(self, image: Image.Image) -> Image.Image:
        output = self.remover(image)
        self.cutout = output.copy()
        return output

    def __getattr__(self, name: str) -> Any:  # to(), cpu() and the rest go to the real model
        return getattr(self.remover, name)


# How much of a frame edge a view's object may cover before the view is left out. Upstream crops a
# picture to its object's bounding box, so a view clipped by its frame (MV-Adapter's side views of a
# wide object run off both edges) reads as a whole object with its ends cut off, and TRELLIS.2 builds
# that: the car control came out crumpled with two such views. Real clipping covers a hundred pixels or
# more of a 768-pixel edge; a frosting tip brushing the frame covers a couple of dozen.
CLIPPED_EDGE = 0.04


def clipped_edges(cutout: Image.Image, fraction: float = CLIPPED_EDGE) -> tuple[str, ...]:
    """The frame edges ('left', 'right', 'top', 'bottom') the cutout's object runs off, by upstream's alpha cut."""
    import numpy as np

    alpha = np.asarray(cutout.convert("RGBA"))[..., 3] > 0.8 * 255  # upstream's bounding-box threshold
    sides = {"left": alpha[:, 0], "right": alpha[:, -1], "top": alpha[0], "bottom": alpha[-1]}
    return tuple(name for name, edge in sides.items() if int(edge.sum()) >= max(4, round(fraction * edge.size)))


def is_out_of_memory(err: BaseException) -> bool:
    """A failed GPU allocation, which a run with the weights off the GPU can get past."""
    import torch

    # torch.cuda.OutOfMemoryError is another name for this class
    if isinstance(err, torch.OutOfMemoryError):
        return True
    # CuMesh (hole filling, inside run) checks its own cudaMalloc calls and raises a plain RuntimeError
    return isinstance(err, RuntimeError) and "out of memory" in str(err)


# What a preset falls back to when it runs out of GPU memory even in low-VRAM mode. A final falls back to
# the preview's pipeline: with the same seed, '512' samples the same 32³ sparse structure and the same
# 512 shape latent as the cascade's first stage, so it rebuilds the shape the user approved, and it is
# known to fit, since the preview of this picture ran with it. The cascade can't be made cheaper instead:
# sample_shape_slat_cascade only lowers hr_resolution while it is above 1024, whatever max_num_tokens
# says, and a 768 or 896 cascade would mean re-implementing run() to drive the 1024 models at resolutions
# upstream never runs them at. Nor would it be sure to fit: what runs out is CuMesh's hole filling on the
# decoded mesh, after every model has left the GPU, and that mesh would still be 56 to 77 % of the size.
FALLBACK_PIPELINE = {"1024_cascade": "512"}


def clear_cuda_error() -> Optional[str]:
    """
    Resets the CUDA runtime's sticky "last error", returning the message it held, or None.

    CuMesh checks its own cudaMalloc calls and raises a RuntimeError when one fails, but never calls
    cudaGetLastError, which is what resets the runtime's per-thread error flag. Torch's next kernel
    launch check then reports that stale flag as "CUDA error: out of memory" with the GPU all but
    empty, so a retry or a fallback after it dies on its first tensor op (Phase 5's logs: only the
    second retry got through, because the first had consumed the flag). A tiny launch here consumes it
    instead: the check raises, the flag is clear, and the error is swallowed. Nothing to do without CUDA.
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


class Trellis2Runtime:
    """Holds the loaded pipeline between jobs (one per worker process)."""

    # The TRELLIS.2 pipeline that made the mesh the last generate() returned: the preset's, or its fallback
    pipeline_used: Optional[str] = None
    # How many of the job's extra views helped make that mesh (a view with no object in it is left out)
    views_used: int = 0
    # How extra views steer the flows; the same for previews and finals (experiments set their own)
    multiview: MultiView = MULTIVIEW
    # Asleep: every model off the GPU and upstream's low-VRAM mode for good, because another runtime has
    # the GPU (production's container runs Pixal3D's finals beside this; see sleep())
    asleep: bool = False
    # Deployed in low-VRAM mode (TRELLIS2_LOW_VRAM=1, an A10): the models live off the GPU from the start,
    # and a retry's restore leaves them there
    low_vram_configured: bool = False

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        _configure_environment()
        import o_voxel  # noqa: F401 - must import after the nvdiffrast stand-in is installed
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self._o_voxel = o_voxel
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
        for note in verify_weights(pipeline.models, model_dir):
            print(f"[forge3d] {note}")
        pipeline.low_vram = os.environ.get("TRELLIS2_LOW_VRAM", "1") == "1"
        pipeline.cuda()
        self.pipeline = pipeline
        self.low_vram_configured = pipeline.low_vram

    def generate(self, image: Any, preset: Preset, seed: int, views: Sequence[View] = ()) -> Any:
        """
        Image to mesh. ``views``, other pictures of the object (its sides and back), steer every flow
        along with the image (see multiview.py); each is cut out and cropped like the image, and
        views_used then says how many were used. The image alone is the cutout the projection paints.

        Running out of GPU memory gets one retry in low-VRAM mode, and a final that still runs out is
        made once more, still in low-VRAM mode, with the preview's pipeline (see FALLBACK_PIPELINE); a
        final that runs out while the models are already off the GPU (an A10, or asleep beside Pixal3D)
        goes to that pipeline at once. pipeline_used then says which pipeline made the mesh.
        """
        self.pipeline_used = preset.pipeline_type
        self.views_used = 0
        try:
            # Background removal and cropping; fails when nothing stands out from the background
            prepared, cutout = self._preprocess(image)
        except ValueError as err:
            raise InputError("no object found in the image: use one object on a plain background") from err
        extra = self._prepare_views(views)
        self.views_used = len(extra)
        if extra:
            weights = ", ".join(f"{weight:g}" for weight in self._weights(extra))
            print(f"[forge3d] {len(extra)} views besides the picture ({self.multiview.mode}; weights {weights})")
        fallback = FALLBACK_PIPELINE.get(preset.pipeline_type)
        try:
            return self._run(prepared, preset.pipeline_type, seed, cutout, extra)
        except Exception as err:
            if not is_out_of_memory(err):
                raise
            retry = not self.pipeline.low_vram  # with the models off the GPU there may be room
            if not retry and fallback is None:
                raise
            reason = _one_line(err)
            if retry:
                print(f"[forge3d] out of GPU memory in {preset.pipeline_type}, retrying in low-VRAM mode: {reason}")
            else:
                print(
                    f"[forge3d] out of GPU memory in {preset.pipeline_type} with the models already off the GPU, "
                    f"falling back to {fallback}: {reason}"
                )
                traceback.clear_frames(err.__traceback__)
        # Retried out here: inside the except block the traceback keeps the failed run's tensors,
        # and so their GPU memory, alive
        try:
            if retry:
                self._offload()
                try:
                    return self._run(prepared, preset.pipeline_type, seed, cutout, extra)
                except Exception as err:
                    if fallback is None or not is_out_of_memory(err):
                        raise
                    print(
                        f"[forge3d] out of GPU memory in {preset.pipeline_type} even in low-VRAM mode, "
                        f"falling back to {fallback}: {_one_line(err)}"
                    )
                    # The retry's traceback holds its tensors the same way: drop them before the cheaper run
                    traceback.clear_frames(err.__traceback__)
            self._free_gpu_memory()
            mesh = self._run(prepared, fallback, seed, cutout, extra)
            self.pipeline_used = fallback
            return mesh
        except BaseException as err:
            # This traceback holds the last run's tensors the same way; drop them so the weights fit back
            traceback.clear_frames(err.__traceback__)
            raise
        finally:
            self._restore()

    def _preprocess(self, image: Any) -> tuple[Image.Image, Optional[Image.Image]]:
        """Upstream preprocess_image, plus the full-frame cutout it crops (what projection paints from)."""
        remover = getattr(self.pipeline, "rembg_model", None)
        keeper = _KeepCutout(remover) if remover is not None else None
        if keeper is not None:
            self.pipeline.rembg_model = keeper
        try:
            prepared = self.pipeline.preprocess_image(image)
        finally:
            if keeper is not None:
                self.pipeline.rembg_model = remover
        cutout = keeper.cutout if keeper is not None else None
        if cutout is None and getattr(image, "mode", None) == "RGBA":
            cutout = image  # upstream skips background removal when the picture has its own alpha
        return prepared, cutout

    def _prepare_views(self, views: Sequence[View]) -> list[tuple[Image.Image, float]]:
        """
        Each view cut out and cropped as the picture is, with its weight. A view with no object in it, or
        whose object runs off the frame (see CLIPPED_EDGE), is left out.
        """
        prepared = []
        for number, view in enumerate(views):
            where = f"views[{number}] (azimuth {view.azimuth:g})"
            try:
                image, cutout = self._preprocess(view.image)
            except ValueError:
                print(f"[forge3d] {where} left out: no object found in it")
                continue
            edges = clipped_edges(cutout) if isinstance(cutout, Image.Image) else ()
            if edges:
                print(f"[forge3d] {where} left out: its object runs off the frame ({', '.join(edges)})")
                continue
            prepared.append((image, view.weight))
        return prepared

    def _weights(self, views: Sequence[tuple[Image.Image, float]]) -> list[float]:
        return [self.multiview.picture_weight, *(weight for _, weight in views)]

    def _run(
        self,
        prepared: Image.Image,
        pipeline_type: str,
        seed: int,
        cutout: Optional[Image.Image] = None,
        views: Sequence[tuple[Image.Image, float]] = (),
    ) -> Any:
        # Without views, upstream's single-picture run exactly
        conditions = (
            multiview.conditioned_on(
                self.pipeline, [prepared, *(image for image, _ in views)], self._weights(views), self.multiview.mode
            )
            if views
            else contextlib.nullcontext()
        )
        # Same seed (and views), same sparse structure: the final keeps the shape of the preview the user
        # approved. Its texture is sampled afresh at the higher resolution, so details can differ
        with conditions:
            mesh = self.pipeline.run(prepared, seed=seed, pipeline_type=pipeline_type, preprocess_image=False)[0]
        if cutout is not None:
            try:
                setattr(mesh, CUTOUT, cutout)
            except (AttributeError, TypeError):  # a mesh that takes no attributes is exported unprojected
                pass
        return mesh

    def _offload(self) -> None:
        """Switches to upstream's low-VRAM mode, which puts each model on the GPU only while it runs."""
        pipeline = self.pipeline
        # Everything upstream's to() moves. Not pipeline.cpu(): that would also make the CPU the
        # device the steps run on.
        for model in (*pipeline.models.values(), pipeline.image_cond_model, pipeline.rembg_model):
            if model is not None:
                model.cpu()
        pipeline.low_vram = True
        self._free_gpu_memory()

    @staticmethod
    def _free_gpu_memory() -> None:
        """
        Frees what a failed run left in reference cycles, then hands torch's cached blocks (the weights'
        old ones included, after an offload) back to CUDA: CuMesh allocates outside that cache. Then
        clears CUDA's stale error flag (clear_cuda_error), so the run after a CuMesh failure can start.
        """
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        stale = clear_cuda_error()
        if stale:
            print(f"[forge3d] cleared a stale CUDA error before going on: {stale}")

    def _restore(self) -> None:
        """
        Every model back where __init__ put them: on the GPU, unless the runtime is deployed in low-VRAM mode
        (TRELLIS2_LOW_VRAM=1) or asleep. Then everything goes back to the CPU instead: upstream's low-VRAM
        stages move their model to the GPU and back without a ``finally``, so a stage that failed leaves
        its model on the GPU.
        """
        if self.asleep or self.low_vram_configured:
            self._offload()
            return
        self.pipeline.low_vram = False
        self.pipeline.cuda()

    def sleep(self) -> None:
        """
        Every model off the GPU, for good, in upstream's low-VRAM mode (each model visits the GPU for its
        stage): another runtime has the GPU from now on. Production's container does this to TRELLIS.2 when
        Pixal3D takes its first final; previews then run this way, as the Pixal3D recipe's own '512'
        preview does. wake() undoes it.
        """
        self.asleep = True
        self._offload()

    def wake(self) -> None:
        """After sleep(), every model back where __init__ put them (on the GPU, unless deployed in low-VRAM mode)."""
        self.asleep = False
        self._restore()

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]:
        """
        Mesh to GLB. Running out of GPU memory (remeshing a complex final) gets one retry the same way.
        ``last_projection`` then summarises the picture's projection (None when the preset has it off).
        """
        self.last_projection: Optional[dict] = None
        try:
            return self._export(mesh, preset)
        except Exception as err:
            if self.pipeline.low_vram or not is_out_of_memory(err):
                raise
            reason = _one_line(err)
            print(f"[forge3d] out of GPU memory exporting, retrying with the models off the GPU: {reason}")
        try:
            self._offload()
            return self._export(mesh, preset)
        except BaseException as err:
            traceback.clear_frames(err.__traceback__)
            raise
        finally:
            self._restore()

    def _export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]:
        # CuMesh (to_glb's remesh) allocates outside torch's cache, so hand that cache back to CUDA first:
        # what the generation, or the last job's projection, left reserved would otherwise not be free
        self._free_gpu_memory()
        glb = self._o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=preset.max_faces,
            texture_size=preset.texture_size,
            remesh=preset.remesh,
            remesh_band=1,
            remesh_project=0,
        )
        material = glb.visual.material
        if getattr(material, "baseColorTexture", None) is not None:
            material.baseColorTexture = unpremultiply(material.baseColorTexture)
            if preset.project_picture:
                # Before the normals, which may split vertices: the projection works on to_glb's mesh
                self.last_projection = self._project(glb, getattr(mesh, CUTOUT, None))
        glb = shade(glb, mesh.voxel_size)
        return glb.export(file_type="glb"), int(len(glb.faces))

    def _project(self, glb: Any, cutout: Optional[Image.Image]) -> dict:
        """Paints the picture onto the side of the model it shows (never raises); returns a summary."""
        if cutout is None:
            report = {"applied": False, "reason": "no picture came with the mesh"}
        else:
            try:
                _, report = projection.project_picture(glb, cutout)
            finally:
                self._free_gpu_memory()  # its buffers stay out of the next job's CuMesh's way
        summary = projection.summary(report)
        print(f"[forge3d] projection: {json.dumps(summary)}")
        return summary


def shade(glb: Any, voxel_size: float) -> Any:
    """
    The exported mesh with smooth, feature-preserving shading normals (see normals.py). They only change
    how light falls on it, so if they fail the remesher's own normals are kept and the job goes on.
    """
    try:
        return normals.with_shading_normals(glb, float(voxel_size))
    except Exception as err:  # noqa: BLE001 - any failure here is cosmetic
        reason = f"{type(err).__name__}: {_one_line(err)}"
        print(f"[forge3d] shading normals failed, keeping the remesher's: {reason}")
        return glb
