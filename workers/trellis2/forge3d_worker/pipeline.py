"""The GPU side: loads stock TRELLIS.2 once and turns an image into a GLB."""

from __future__ import annotations

import json
import os
from typing import Any

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
        try:
            # Background removal and cropping; fails when nothing stands out from the background
            prepared = self.pipeline.preprocess_image(image)
        except ValueError as err:
            raise InputError("no object found in the image: use one object on a plain background") from err
        # Same seed, same sparse structure: the final pass refines the preview the user approved
        return self.pipeline.run(
            prepared, seed=seed, pipeline_type=preset.pipeline_type, preprocess_image=False
        )[0]

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
        return glb.export(file_type="glb"), int(len(glb.faces))
