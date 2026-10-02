"""The download script's pins, its checks, and that it fills the folders model.py reads."""

import hashlib
import importlib.util
import pathlib
import re
import sys
import types

import pytest

from judge_worker import model

WORKERS = pathlib.Path(__file__).parent.parent.parent
SCRIPT = WORKERS / "judge" / "scripts" / "download_weights.py"


def load_script(monkeypatch, root, which="all"):
    monkeypatch.setenv("MODELS_ROOT", str(root))
    monkeypatch.setenv("WHICH", which)
    spec = importlib.util.spec_from_file_location("judge_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_revision_and_weight_is_pinned(monkeypatch, tmp_path):
    script = load_script(monkeypatch, tmp_path)
    assert list(script.MODELS) == list(model.MODELS) == ["8b", "30b"]
    for which, (repo, revision) in script.MODELS.items():
        assert repo == model.MODELS[which].repo
        assert script.FOLDERS[which] == model.MODELS[which].folder
        assert re.fullmatch(r"[0-9a-f]{40}", revision), repo
    assert script.MODELS["8b"][1] == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert script.MODELS["30b"][1] == "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    for which, count in (("8b", 4), ("30b", 13)):
        names = sorted(script.WEIGHTS[which])
        assert names == [f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)]
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in script.WEIGHTS[which].values())
    digests = [digest for weights in script.WEIGHTS.values() for digest in weights.values()]
    assert len(set(digests)) == len(digests)
    # Weights and everything transformers reads; not .gitattributes
    assert script.PATTERNS == ["*.safetensors", "*.json", "*.txt", "README.md"]


def fake_hub(monkeypatch, calls, contents):
    """huggingface_hub's stand-in: snapshot_download writes each pinned weight file with ``contents``'s bytes."""

    def snapshot_download(repo, **options):
        calls.append((repo, options["revision"], options["local_dir"], options["allow_patterns"]))
        folder = pathlib.Path(options["local_dir"])
        (folder / ".cache").mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}")
        for name, data in contents[repo].items():
            (folder / name).write_bytes(data)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def pinned_contents(script):
    """Made-up bytes for every pinned file, and the script's WEIGHTS changed to their digests."""
    contents, weights = {}, {}
    for which, (repo, _) in script.MODELS.items():
        contents[repo] = {name: f"{which}/{name}".encode() for name in script.WEIGHTS[which]}
        weights[which] = {name: hashlib.sha256(data).hexdigest() for name, data in contents[repo].items()}
    return contents, weights


def test_both_models_download_into_their_folders_and_are_checked(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path)
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    script.main()
    assert calls == [
        ("Qwen/Qwen3-VL-8B-Instruct", script.MODELS["8b"][1], str(tmp_path / "Qwen3-VL-8B-Instruct"), script.PATTERNS),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", script.MODELS["30b"][1], str(tmp_path / "Qwen3-VL-30B-A3B-Instruct"), script.PATTERNS),
    ]
    assert not (tmp_path / "Qwen3-VL-8B-Instruct" / ".cache").exists()


def test_which_picks_one(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path, which="30b")
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    script.main()
    assert [call[0] for call in calls] == ["Qwen/Qwen3-VL-30B-A3B-Instruct"]
    script = load_script(monkeypatch, tmp_path, which="7b")
    with pytest.raises(SystemExit, match="WHICH must be all, 8b, 30b"):
        script.main()


def test_a_weight_without_its_pinned_sha256_stops_the_download(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path, which="8b")
    contents, weights = pinned_contents(script)
    weights["8b"]["model-00003-of-00004.safetensors"] = "0" * 64
    monkeypatch.setattr(script, "WEIGHTS", weights)
    fake_hub(monkeypatch, calls, contents)
    with pytest.raises(SystemExit, match="8b: model-00003-of-00004.safetensors do not have the pinned sha256"):
        script.main()


def test_a_missing_or_extra_weight_file_stops_the_download(monkeypatch, tmp_path):
    calls = []
    script = load_script(monkeypatch, tmp_path, which="8b")
    contents, weights = pinned_contents(script)
    monkeypatch.setattr(script, "WEIGHTS", weights)
    repo = script.MODELS["8b"][0]
    contents[repo]["model-00005-of-00004.safetensors"] = b"stray"
    fake_hub(monkeypatch, calls, contents)
    with pytest.raises(SystemExit, match="the weight files are"):
        script.main()
