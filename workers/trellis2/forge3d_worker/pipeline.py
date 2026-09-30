"""The GPU side: loads stock TRELLIS.2 once and turns an image into a GLB."""

from __future__ import annotations

import gc
import json
import os
import traceback
from typing import Any

from PIL import Image

from . import uv_raster
from .inputs import InputError
from .settings import Preset

MODEL_DIR = os.environ.get("TRELLIS2_MODEL_DIR", "/models/TRELLIS.2-4B")


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


def is_out_of_memory(err: BaseException) -> bool:
    """A failed GPU allocation, which a run with the weights off the GPU can get past."""
    import torch

    # torch.cuda.OutOfMemoryError is another name for this class
    if isinstance(err, torch.OutOfMemoryError):
        return True
    # CuMesh (hole filling, inside run) checks its own cudaMalloc calls and raises a plain RuntimeError
    return isinstance(err, RuntimeError) and "out of memory" in str(err)


class Trellis2Runtime:
    """Holds the loaded pipeline between jobs (one per worker process)."""

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

    def generate(self, image: Any, preset: Preset, seed: int) -> Any:
        """Image to mesh. Running out of GPU memory gets one retry in low-VRAM mode."""
        try:
            # Background removal and cropping; fails when nothing stands out from the background
            prepared = self.pipeline.preprocess_image(image)
        except ValueError as err:
            raise InputError("no object found in the image: use one object on a plain background") from err
        try:
            return self._run(prepared, preset, seed)
        except Exception as err:
            if self.pipeline.low_vram or not is_out_of_memory(err):
                raise
            reason = " ".join(str(err).split())  # one line, whatever the message
            print(f"[forge3d] out of GPU memory in {preset.pipeline_type}, retrying in low-VRAM mode: {reason}")
        # Retried out here: inside the except block the traceback keeps the failed run's tensors,
        # and so their GPU memory, alive
        try:
            self._offload()
            return self._run(prepared, preset, seed)
        except BaseException as err:
            # This traceback holds the retry's tensors the same way; drop them so the weights fit back
            traceback.clear_frames(err.__traceback__)
            raise
        finally:
            self._restore()

    def _run(self, prepared: Image.Image, preset: Preset, seed: int) -> Any:
        # Same seed, same sparse structure: the final keeps the shape of the preview the user approved.
        # Its texture is sampled afresh at the higher resolution, so details can differ
        return self.pipeline.run(
            prepared, seed=seed, pipeline_type=preset.pipeline_type, preprocess_image=False
        )[0]

    def _offload(self) -> None:
        """Switches to upstream's low-VRAM mode, which puts each model on the GPU only while it runs."""
        import torch

        pipeline = self.pipeline
        # Everything upstream's to() moves. Not pipeline.cpu(): that would also make the CPU the
        # device the steps run on.
        for model in (*pipeline.models.values(), pipeline.image_cond_model, pipeline.rembg_model):
            if model is not None:
                model.cpu()
        pipeline.low_vram = True
        # Free what the failed run left in reference cycles, then hand torch's cached blocks, the
        # weights' old ones included, back to CUDA: CuMesh allocates outside that cache
        gc.collect()
        torch.cuda.empty_cache()

    def _restore(self) -> None:
        """Every model back on the GPU, the way __init__ put them there."""
        self.pipeline.low_vram = False
        self.pipeline.cuda()

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]:
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
        return glb.export(file_type="glb"), int(len(glb.faces))
