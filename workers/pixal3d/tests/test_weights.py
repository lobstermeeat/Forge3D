"""The weights script's pins, and the pipeline configs it rewrites."""

import importlib.util
import json
import os
import re
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "download_weights.py")


@pytest.fixture()
def weights(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("pixal3d_download_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_everything_is_pinned(weights):
    for repo, revision in (weights.PIXAL3D, weights.MOGE):
        assert re.fullmatch(r"[0-9a-f]{40}", revision), repo
    for digest in (weights.NAF_SHA256, weights.DINOV3_SHA256, *weights.DECODERS.values()):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert all(len(models) == 4 for models in weights.FLOW_MODELS.values())
    assert all(model.endswith("_mv") for model in weights.FLOW_MODELS["multiview"])
    assert "briaai/" not in open(SCRIPT).read()  # RMBG-2.0 (CC BY-NC 4.0) is never downloaded


def test_rewrite_config_points_at_shared_decoders_and_permissive_models(weights, tmp_path):
    os.makedirs(weights.PIXAL3D_DIR)
    stock = {
        "name": "Pixal3DMVImageTo3DPipeline",
        "args": {
            "models": {
                "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
                "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv",
                "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
            },
            "image_cond_model": {"name": "DinoV3FeatureExtractor", "args": {"model_name": "facebook/dinov3"}},
            "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}},
        },
    }
    with open(os.path.join(weights.PIXAL3D_DIR, "pipeline_mv.json"), "w") as f:
        json.dump(stock, f)
    shared = os.path.join(weights.TRELLIS_DIR, "ckpts", "ss_dec_conv3d_16l8_fp16")
    weights.rewrite_config("pipeline_mv.json", {"ckpts/ss_dec_conv3d_16l8_fp16": shared})
    args = json.load(open(os.path.join(weights.PIXAL3D_DIR, "pipeline_mv.json")))["args"]
    assert args["models"]["sparse_structure_decoder"] == shared
    assert args["models"]["sparse_structure_flow_model"] == "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv"
    assert args["rembg_model"] == {"name": "BiRefNet", "args": {"model_name": weights.BIREFNET_DIR}}
    assert args["image_cond_model"]["args"]["model_name"] == weights.DINO_DIR


def test_decoder_reuse_needs_an_identical_file(weights, tmp_path, monkeypatch):
    name = "ckpts/tex_dec_next_dc_f16c32_fp16"
    path = os.path.join(weights.TRELLIS_DIR, name)
    os.makedirs(os.path.dirname(path))
    with open(f"{path}.safetensors", "wb") as f:
        f.write(b"weights")
    digest = weights.sha256(f"{path}.safetensors")
    assert weights.decoder_path(name, digest) == path
    downloads = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: downloads.append(a[1]) or ""
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(FileNotFoundError):  # a different file: Pixal3D's own is fetched (here: faked)
        weights.decoder_path(name, "0" * 64)
    assert downloads == [f"{name}.json", f"{name}.safetensors"]
