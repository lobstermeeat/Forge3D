"""The download script's pins, its checks, and that it fills the folders painter_worker/qwen.py reads."""

import hashlib
import importlib.util
import pathlib
import re
import sys
import types

import pytest

from painter_worker import qwen

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "download_weights.py"


def load_script(monkeypatch, root):
    monkeypatch.setenv("MODELS_ROOT", str(root))
    spec = importlib.util.spec_from_file_location("painter_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_revision_and_weight_is_pinned(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    assert script.MODELS == {
        "model": ("Qwen/Qwen-Image-Edit-2511", "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"),
        "lora": ("lightx2v/Qwen-Image-Edit-2511-Lightning", "d74eba145674fd7e31b949324e148e21e7118abd"),
    }
    assert script.FOLDERS == {"model": qwen.MODEL_FOLDER, "lora": qwen.LORA_FOLDER}
    for which, weights in script.WEIGHTS.items():
        for name, (size, digest) in weights.items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest) and size > 0, name
    digests = [digest for weights in script.WEIGHTS.values() for _, digest in weights.values()]
    assert len(set(digests)) == len(digests)
    model = script.WEIGHTS["model"]
    assert sorted(model) == sorted(
        [f"transformer/diffusion_pytorch_model-{i:05d}-of-00005.safetensors" for i in range(1, 6)]
        + [f"text_encoder/model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
        + ["vae/diffusion_pytorch_model.safetensors"]
    )
    # The sizes qwen.estimate_memory() reports are these files'
    for component, size in qwen.WEIGHT_BYTES.items():
        assert sum(s for name, (s, _) in model.items() if name.startswith(component + "/")) == size
    # Only the 8-step LoRA, both precisions, the painter's file among them
    assert list(script.WEIGHTS["lora"]) == script.LORA_FILES == list(qwen.LORA_FILES)
    assert script.WEIGHTS["lora"][qwen.LORA_FILES[1]][0] == qwen.LORA_BYTES  # the bf16 file
    assert script.PATTERNS["lora"] == [*qwen.LORA_FILES, "README.md"]
    assert script.PATTERNS["model"] == ["*.safetensors", "*.json", "*.txt", "*.jinja", "README.md"]


def fake_hub(monkeypatch, calls, contents):
    """huggingface_hub's stand-in: snapshot_download writes the repository's files with ``contents``'s bytes."""

    def snapshot_download(repo, **options):
        calls.append((repo, options["revision"], options["local_dir"], options["allow_patterns"]))
        folder = pathlib.Path(options["local_dir"])
        (folder / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        for name, data in contents[repo].items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def pinned_contents(script):
    """Made-up bytes for every pinned file, and the script's WEIGHTS changed to their sizes and digests."""
    contents, weights = {}, {}
    for which, (repo, _) in script.MODELS.items():
        files = {name: f"{which}/{name}".encode() for name in script.WEIGHTS[which]}
        weights[which] = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()}
        contents[repo] = {**files, **{name: b"{}" for name in script.FILES[which]}}
    return contents, weights


def test_both_repositories_download_into_their_folders_and_are_checked(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    script.main([])
    assert calls == [
        ("Qwen/Qwen-Image-Edit-2511", script.MODELS["model"][1], str(tmp_path / qwen.MODEL_FOLDER), script.PATTERNS["model"]),
        (
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            script.MODELS["lora"][1],
            str(tmp_path / qwen.LORA_FOLDER),
            script.PATTERNS["lora"],
        ),
    ]
    assert not (tmp_path / qwen.MODEL_FOLDER / ".cache").exists()
    # What the painter reads is where it looks
    assert (tmp_path / qwen.MODEL_FOLDER / "model_index.json").is_file()
    assert qwen.lora_file(tmp_path / qwen.LORA_FOLDER) == str(tmp_path / qwen.LORA_FOLDER / qwen.LORA_FILE)


def test_which_picks_one(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    script.main(["--which", "lora"])
    assert [call[0] for call in calls] == ["lightx2v/Qwen-Image-Edit-2511-Lightning"]
    with pytest.raises(SystemExit):
        script.main(["--which", "vae"])


def test_check_looks_without_downloading(monkeypatch, tmp_path, capsys):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    with pytest.raises(SystemExit, match="model: .* is missing README.md, model_index.json"):
        script.main(["--check"])
    fake_hub(monkeypatch, calls, contents)
    script.main([])
    calls.clear()
    script.main(["--check"])
    script.main(["--check", "--sha256"])
    assert calls == [] and "every sha256 matching" in capsys.readouterr().out
    # A file of the wrong size fails the quick check; a same-sized but different one only the hashes
    weight = tmp_path / qwen.MODEL_FOLDER / "vae" / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(weight.read_bytes()[:-1])
    with pytest.raises(SystemExit, match="vae/diffusion_pytorch_model.safetensors do not have the pinned size"):
        script.main(["--check", "--which", "model"])
    weight.write_bytes(b"x" * weights["model"]["vae/diffusion_pytorch_model.safetensors"][0])
    script.main(["--check", "--which", "model"])
    with pytest.raises(SystemExit, match="do not have the pinned sha256"):
        script.main(["--check", "--sha256", "--which", "model"])
    with pytest.raises(SystemExit):
        script.main(["--sha256"])  # only with --check


def test_a_weight_without_its_pinned_sha256_stops_the_download(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    name = "transformer/diffusion_pytorch_model-00003-of-00005.safetensors"
    weights["model"][name] = (weights["model"][name][0], "0" * 64)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    with pytest.raises(SystemExit, match=f"model: {name} do not have the pinned sha256"):
        script.main(["--which", "model"])


def test_a_stray_weight_file_stops_the_download(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    contents["lightx2v/Qwen-Image-Edit-2511-Lightning"]["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"] = b"4"
    fake_hub(monkeypatch, calls, contents)
    with pytest.raises(SystemExit, match="lora: the weight files are"):
        script.main(["--which", "lora"])
