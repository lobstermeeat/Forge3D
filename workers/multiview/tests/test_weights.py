"""The download script's pins, and that it fills the folders the generator reads."""

import fnmatch
import importlib.util
import pathlib
import re
import sys
import types

from multiview_worker import generator

WORKERS = pathlib.Path(__file__).parent.parent.parent
SCRIPT = WORKERS / "multiview" / "scripts" / "download_weights.py"


def load_script(monkeypatch, root):
    monkeypatch.setenv("MODELS_ROOT", str(root))
    spec = importlib.util.spec_from_file_location("multiview_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_revision_is_pinned_to_a_commit(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    for repo, revision in (script.MVADAPTER, script.SDXL, script.VAE, script.BIREFNET):
        assert re.fullmatch(r"[0-9a-f]{40}", revision), repo


def test_birefnet_is_the_trellis2_workers_revision(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    trellis2 = (WORKERS / "trellis2" / "scripts" / "download_weights.py").read_text()
    assert f'BIREFNET = ("{script.BIREFNET[0]}", "{script.BIREFNET[1]}")' in trellis2


def test_folders_match_the_generator(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    assert script.ADAPTER_DIR == str(tmp_path / generator.ADAPTER_DIR)
    assert script.SDXL_DIR == str(tmp_path / generator.SDXL_DIR)
    assert script.VAE_DIR == str(tmp_path / generator.VAE_DIR)
    assert script.BIREFNET_DIR == str(tmp_path / generator.BIREFNET_DIR)
    assert script.ADAPTER_FILE == generator.ADAPTER_FILE


SDXL_REPO = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "sd_xl_base_1.0.safetensors",
    "sd_xl_offset_example-lora_1.0.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder/model.onnx",
    "text_encoder_2/model.fp16.safetensors",
    "text_encoder_2/model.safetensors",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_flax_model.msgpack",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae_1_0/diffusion_pytorch_model.fp16.safetensors",
]


def test_only_sdxls_fp16_diffusers_files_are_fetched(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    fetched = [name for name in SDXL_REPO if any(fnmatch.fnmatch(name, pattern) for pattern in script.SDXL_FILES)]
    weights = [name for name in fetched if name.endswith((".safetensors", ".msgpack", ".onnx"))]
    assert all(".fp16." in name for name in weights)
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in fetched and "tokenizer_2/merges.txt" in fetched
    assert not any(name.startswith("vae_1_0/") for name in fetched)


def test_main_downloads_into_the_folders_and_reuses_birefnet(monkeypatch, tmp_path):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, filename, **kw: calls.append(("file", repo, filename, kw["revision"], kw["local_dir"]))
    hub.snapshot_download = lambda repo, **kw: calls.append(("snapshot", repo, kw["revision"], kw["local_dir"]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    script = load_script(monkeypatch, tmp_path)
    birefnet = tmp_path / "BiRefNet"
    birefnet.mkdir()
    for name in ("config.json", "birefnet.py", "model.safetensors"):
        (birefnet / name).write_text("x")
    (tmp_path / "mv-adapter" / ".cache").mkdir(parents=True)

    script.main()
    assert calls[0] == ("file", "huanngzh/mv-adapter", "mvadapter_i2mv_sdxl.safetensors", script.MVADAPTER[1], script.ADAPTER_DIR)
    assert [call[1] for call in calls] == ["huanngzh/mv-adapter", "stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"]
    assert not (tmp_path / "mv-adapter" / ".cache").exists()

    # Without the TRELLIS.2 worker's copy, BiRefNet is fetched at the same revision
    (birefnet / "model.safetensors").unlink()
    calls.clear()
    script.main()
    assert calls[-1] == ("snapshot", "ZhengPeng7/BiRefNet", script.BIREFNET[1], script.BIREFNET_DIR)
