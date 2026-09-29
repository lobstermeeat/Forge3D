"""Checkpoint verification (the strict=False trap) and gltfpack output."""

import json
import os
import shutil
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from forge3d_worker.compress import GLTFPACK, pack_glb
from forge3d_worker.pipeline import WeightMismatchError, verify_weights


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("scale", torch.ones(1))


def write_model_dir(tmp_path, tensors):
    (tmp_path / "ckpts").mkdir()
    save_file(tensors, str(tmp_path / "ckpts" / "tiny.safetensors"))
    (tmp_path / "pipeline.json").write_text(json.dumps({"args": {"models": {"tiny": "ckpts/tiny"}}}))
    return str(tmp_path)


def test_complete_checkpoint_passes(tmp_path):
    model = Tiny()
    model_dir = write_model_dir(tmp_path, {k: v.clone() for k, v in model.state_dict().items()})
    assert verify_weights({"tiny": model}, model_dir) == []


def test_missing_parameters_fail_instead_of_staying_random(tmp_path):
    model = Tiny()
    tensors = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("norm.")}
    model_dir = write_model_dir(tmp_path, tensors)
    with pytest.raises(WeightMismatchError, match="tiny is missing 2 parameters"):
        verify_weights({"tiny": model}, model_dir)


def test_extra_tensors_are_only_reported(tmp_path):
    model = Tiny()
    tensors = {k: v.clone() for k, v in model.state_dict().items()}
    tensors["ema.proj.weight"] = torch.zeros(4, 4)
    model_dir = write_model_dir(tmp_path, tensors)
    assert verify_weights({"tiny": model}, model_dir) == ["tiny checkpoint has 1 unused tensors"]


def glb_json(data: bytes) -> dict:
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20 : 20 + length])


@pytest.mark.skipif(shutil.which(GLTFPACK) is None, reason="gltfpack not installed")
def test_pack_glb_produces_meshopt_and_ktx2(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    sphere = trimesh.creation.uv_sphere(count=[32, 32])
    v = sphere.vertices
    uv = np.stack([0.5 + np.arctan2(v[:, 2], v[:, 0]) / (2 * np.pi), 0.5 + np.arcsin(np.clip(v[:, 1], -1, 1)) / np.pi], 1)
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (512, 512, 4), dtype=np.uint8))
    mr = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=base, metallicRoughnessTexture=mr)
    mesh = trimesh.Trimesh(v, sphere.faces, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    raw = mesh.export(file_type="glb")

    packed = pack_glb(raw, texture_limit=256)
    meta = glb_json(packed)
    assert {"EXT_meshopt_compression", "KHR_texture_basisu", "KHR_mesh_quantization"} <= set(meta["extensionsUsed"])
    assert all(image["mimeType"] == "image/ktx2" for image in meta["images"])
    assert len(packed) < len(raw)
