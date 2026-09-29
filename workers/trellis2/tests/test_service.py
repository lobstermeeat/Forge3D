"""Job handling with a fake GPU runtime: input validation, output shape, and failure reporting."""

import base64
import io
import socket

import pytest
from PIL import Image

from forge3d_worker import inputs
from forge3d_worker.inputs import InputError, parse_job
from forge3d_worker.service import handle_job
from forge3d_worker.settings import PRESETS


def png_bytes(size=(64, 48), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (200, 80, 40) if mode == "RGB" else (200, 80, 40, 128)).save(buf, "PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class FakeRuntime:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def generate(self, image, preset, seed):
        self.calls.append(("generate", preset.pipeline_type, seed, image.size))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return "mesh"

    def export(self, mesh, preset):
        self.calls.append(("export", preset.max_faces, preset.texture_size))
        return b"raw-glb" * 10, preset.max_faces


class FakeStorage:
    def __init__(self):
        self.saved = {}

    def put(self, key, data, content_type):
        self.saved[key] = (data, content_type)
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def pack(raw: bytes, texture_limit: int) -> bytes:
    return raw[:20]


def test_final_job_runs_every_stage_and_reports_credits():
    runtime, storage = FakeRuntime(), FakeStorage()
    job = {"id": "rp-1", "input": {"image_base64": b64(png_bytes()), "seed": 1234, "request_id": "gen_42"}}
    out = handle_job(job, runtime, storage, pack)

    assert "error" not in out
    assert out["mode"] == "final" and out["seed"] == 1234 and out["request_id"] == "gen_42"
    assert out["glb"]["url"] == "https://assets.example.com/ai/gen_42/final.glb"
    assert out["bytes"] == 20 and out["raw_bytes"] == 70 and out["triangles"] == 100_000
    assert set(out["timings"]) == {"generate_s", "export_s", "compress_s", "upload_s"}
    assert "Built with DINOv3" in out["credits"]
    assert runtime.calls[0] == ("generate", "1024_cascade", 1234, (64, 48))
    assert storage.saved["ai/gen_42/final.glb"][1] == "model/gltf-binary"


def test_preview_uses_the_cheap_preset_and_returns_its_seed():
    runtime = FakeRuntime()
    out = handle_job({"id": "rp-2", "input": {"image_base64": b64(png_bytes()), "mode": "preview"}}, runtime, FakeStorage(), pack)
    assert out["mode"] == "preview"
    assert isinstance(out["seed"], int)  # generated, so the final pass can reuse it
    assert runtime.calls[0][1] == PRESETS["preview"].pipeline_type == "512"
    assert out["glb"]["key"] == "ai/rp-2/preview.glb"


def test_invalid_input_is_reported_without_running_the_model():
    runtime = FakeRuntime()
    out = handle_job({"id": "x", "input": {"mode": "ultra", "image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert out["error"].startswith("invalid input: mode must be one of")
    assert runtime.calls == []


def test_gpu_failure_is_reported_and_restarts_the_worker():
    out = handle_job({"id": "x", "input": {"image_base64": b64(png_bytes())}}, FakeRuntime(fail=True), FakeStorage(), pack)
    assert out["error"] == "generation failed: RuntimeError: CUDA out of memory"
    assert out["refresh_worker"] is True


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not a dict", "input must be an object"),
        ({}, "provide exactly one of image_url or image_base64"),
        ({"image_url": "https://a/x.png", "image_base64": "AAAA"}, "provide exactly one"),
        ({"image_base64": "!!!"}, "not valid base64"),
        ({"image_base64": b64(b"not an image")}, "could not be decoded"),
        ({"image_base64": b64(png_bytes()), "seed": -1}, "seed must be an integer"),
        ({"image_base64": b64(png_bytes()), "seed": True}, "seed must be an integer"),
        ({"image_base64": b64(png_bytes()), "request_id": "../etc"}, "request_id must be"),
        ({"image_base64": b64(png_bytes()), "mode": ["final"]}, "mode must be one of"),
        ({"image_url": "http://assets.example.com/x.png"}, "must be an https URL"),
    ],
)
def test_rejects_bad_input(payload, message):
    with pytest.raises(InputError, match=message):
        parse_job(payload, fallback_id="job")


def test_keeps_alpha_so_background_removal_can_be_skipped():
    job = parse_job({"image_base64": b64(png_bytes(mode="RGBA"))}, fallback_id="job")
    assert job.image.mode == "RGBA"


def test_rejects_oversized_images():
    with pytest.raises(InputError, match="larger than 4096px"):
        parse_job({"image_base64": b64(png_bytes(size=(4097, 8)))}, fallback_id="job")


def test_url_fetch_refuses_private_addresses(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("169.254.169.254", 443))])
    with pytest.raises(InputError, match="public address"):
        inputs._check_url("https://metadata.internal/latest")


def test_url_fetch_respects_the_host_allowlist(monkeypatch):
    monkeypatch.setenv("ALLOWED_IMAGE_HOSTS", "assets.forge3d.app")
    with pytest.raises(InputError, match="not in ALLOWED_IMAGE_HOSTS"):
        inputs._check_url("https://evil.example.com/x.png")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("104.18.2.1", 443))])
    inputs._check_url("https://assets.forge3d.app/uploads/x.png")


def test_url_input_uses_the_fetcher():
    seen = []

    def fetch(url):
        seen.append(url)
        return png_bytes()

    job = parse_job({"image_url": "https://assets.forge3d.app/x.png"}, fallback_id="job", fetch=fetch)
    assert seen == ["https://assets.forge3d.app/x.png"] and job.image.size == (64, 48)


def test_fetch_errors_become_input_errors(monkeypatch):
    import urllib.error

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("104.18.2.1", 443))])

    class Broken:
        def open(self, *a, **k):
            raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(inputs, "_OPENER", Broken())
    with pytest.raises(InputError, match="could not be fetched"):
        inputs.fetch_url("https://assets.forge3d.app/x.png")
