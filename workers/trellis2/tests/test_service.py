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
    assert out["glb"]["url"] == "https://assets.example.com/ai/gen_42/final-1234.glb"
    assert out["bytes"] == 20 and out["raw_bytes"] == 70 and out["triangles"] == 100_000
    assert set(out["timings"]) == {"generate_s", "export_s", "compress_s", "upload_s"}
    assert "Built with DINOv3" in out["credits"]
    assert runtime.calls[0] == ("generate", "1024_cascade", 1234, (64, 48))
    assert storage.saved["ai/gen_42/final-1234.glb"][1] == "model/gltf-binary"


def test_the_result_names_the_pipeline_that_made_the_model():
    image = {"image_base64": b64(png_bytes()), "seed": 5}
    # A runtime that doesn't say (like an older one) gets its preset's pipeline reported
    out = handle_job({"id": "a", "input": image}, FakeRuntime(), FakeStorage(), pack)
    assert out["pipeline"] == "1024_cascade"
    out = handle_job({"id": "b", "input": {**image, "mode": "preview"}}, FakeRuntime(), FakeStorage(), pack)
    assert out["pipeline"] == "512"

    class FellBack(FakeRuntime):
        def generate(self, image, preset, seed):
            self.pipeline_used = "512"  # the cascade ran out of GPU memory even in low-VRAM mode
            return super().generate(image, preset, seed)

    runtime = FellBack()
    out = handle_job({"id": "c", "input": image}, runtime, FakeStorage(), pack)
    assert out["mode"] == "final" and out["pipeline"] == "512" and out["triangles"] == 100_000
    # Exported with the final's settings all the same
    assert runtime.calls[1] == ("export", 100_000, 2048)


def test_the_projection_summary_rides_along_when_the_runtime_has_one():
    runtime = FakeRuntime()
    out = handle_job({"id": "rp-3", "input": {"image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert "projection" not in out  # a runtime that doesn't project adds nothing
    runtime.last_projection = {"applied": True, "reason": "applied", "iou": 0.97, "seconds": 2.1}
    out = handle_job({"id": "rp-3", "input": {"image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert out["projection"] == runtime.last_projection
    # ... and it isn't a timing, so the GPU seconds (the timings' sum) stay right
    assert set(out["timings"]) == {"generate_s", "export_s", "compress_s", "upload_s"}
    assert "refresh_worker" not in out


def test_a_gpu_fault_while_projecting_keeps_the_model_but_restarts_the_worker():
    runtime = FakeRuntime()
    reason = "error: RuntimeError: CUDA error: an illegal memory access"
    runtime.last_projection = {"applied": False, "reason": reason, "gpu_fault": True}
    out = handle_job({"id": "rp-4", "input": {"image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert "error" not in out and out["glb"]["key"].endswith(".glb")
    assert out["refresh_worker"] is True


def test_preview_uses_the_cheap_preset_and_returns_its_seed():
    runtime = FakeRuntime()
    out = handle_job({"id": "rp-2", "input": {"image_base64": b64(png_bytes()), "mode": "preview"}}, runtime, FakeStorage(), pack)
    assert out["mode"] == "preview"
    assert isinstance(out["seed"], int)  # generated, so the final pass can reuse it
    assert runtime.calls[0][1] == PRESETS["preview"].pipeline_type == "512"
    assert out["glb"]["key"] == f"ai/rp-2/preview-{out['seed']}.glb"


def test_invalid_input_is_reported_without_running_the_model():
    runtime = FakeRuntime()
    out = handle_job({"id": "x", "input": {"mode": "ultra", "image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert out["error"].startswith("invalid input: mode must be one of")
    assert runtime.calls == []


def test_gpu_failure_is_reported_and_restarts_the_worker():
    out = handle_job({"id": "x", "input": {"image_base64": b64(png_bytes())}}, FakeRuntime(fail=True), FakeStorage(), pack)
    assert out["error"] == "generation failed: RuntimeError: CUDA out of memory"
    assert out["refresh_worker"] is True


def test_storage_failure_does_not_restart_the_worker():
    class BrokenStorage:
        def put(self, key, data, content_type):
            raise ConnectionError("R2 unreachable")

    out = handle_job({"id": "x", "input": {"image_base64": b64(png_bytes())}}, FakeRuntime(), BrokenStorage(), pack)
    assert out == {"error": "generation failed: ConnectionError: R2 unreachable"}


def test_no_object_in_the_image_is_an_input_error():
    class NothingFound(FakeRuntime):
        def generate(self, image, preset, seed):
            raise InputError("no object found in the image: use one object on a plain background")

    out = handle_job({"id": "x", "input": {"image_base64": b64(png_bytes())}}, NothingFound(), FakeStorage(), pack)
    assert out == {"error": "invalid input: no object found in the image: use one object on a plain background"}


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


def test_url_fetch_is_off_without_an_allowlist(monkeypatch):
    monkeypatch.delenv("ALLOWED_IMAGE_HOSTS", raising=False)
    with pytest.raises(InputError, match="image_url is disabled"):
        inputs._check_url("https://assets.forge3d.app/x.png")


def test_url_fetch_refuses_private_addresses(monkeypatch):
    monkeypatch.setenv("ALLOWED_IMAGE_HOSTS", "metadata.internal")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("169.254.169.254", 443))])
    with pytest.raises(InputError, match="public address"):
        inputs._check_url("https://metadata.internal/latest")


@pytest.mark.parametrize("url", ["https://[::1/x.png", "https://assets.forge3d.app:99999/x.png"])
def test_malformed_urls_are_input_errors(monkeypatch, url):
    monkeypatch.setenv("ALLOWED_IMAGE_HOSTS", "assets.forge3d.app")
    with pytest.raises(InputError, match="not a valid URL"):
        inputs._check_url(url)


def test_exif_rotation_is_applied():
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # rotated 90 degrees
    buf = io.BytesIO()
    image.save(buf, "JPEG", exif=exif)
    job = parse_job({"image_base64": b64(buf.getvalue())}, fallback_id="job")
    assert job.image.size == (32, 64)


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

    monkeypatch.setenv("ALLOWED_IMAGE_HOSTS", "assets.forge3d.app")
    monkeypatch.setattr(inputs, "_OPENER", Broken())
    with pytest.raises(InputError, match="^image_url could not be fetched$"):
        inputs.fetch_url("https://assets.forge3d.app/x.png")


# --- Extra views of the object ----------------------------------------------------------------------


def view(azimuth=90, **extra) -> dict:
    return {"image_base64": b64(png_bytes(mode="RGBA")), "azimuth": azimuth, "elevation": 0, **extra}


def test_views_are_parsed_with_their_camera_and_weight():
    job = parse_job(
        {"image_base64": b64(png_bytes()), "views": [view(90), view(-90, elevation=15.5, weight=0.5), view(450)]},
        fallback_id="job",
    )
    assert [(v.azimuth, v.elevation, v.weight) for v in job.views] == [(90, 0, 1.0), (270, 15.5, 0.5), (90, 0, 1.0)]
    assert all(v.image.mode == "RGBA" and v.image.size == (64, 48) for v in job.views)  # cutouts keep alpha
    assert job.image.mode == "RGB"


@pytest.mark.parametrize("views", [None, []])
def test_no_views_is_a_single_picture_job(views):
    assert parse_job({"image_base64": b64(png_bytes()), "views": views}, fallback_id="job").views == ()


def test_a_view_without_elevation_is_level():
    level = {"image_base64": b64(png_bytes()), "azimuth": 180}
    job = parse_job({"image_base64": b64(png_bytes()), "views": [level]}, "job")
    assert job.views[0].elevation == 0


@pytest.mark.parametrize(
    "views, message",
    [
        ("not a list", "^views must be a list$"),
        ([view()] * 9, "^views can have at most 8 images$"),
        (["x"], r"^views\[0\]: must be an object$"),
        ([view(), {"azimuth": 0}], r"^views\[1\]: provide exactly one of image_url or image_base64$"),
        ([{**view(), "image_url": "https://a/x.png"}], r"^views\[0\]: provide exactly one"),
        ([view(image_base64="!!!")], r"^views\[0\]: image_base64 is not valid base64$"),
        ([view(image_base64=b64(b"not an image"))], r"^views\[0\]: image could not be decoded$"),
        ([view(image_base64=b64(png_bytes(size=(4097, 8))))], r"^views\[0\]: image is larger than 4096px on a side$"),
        ([view(image_base64=7)], r"^views\[0\]: image_base64 must be a string$"),
        ([view(image_url="http://a.example.com/x.png", image_base64=None)], r"^views\[0\]: image_url must be an https"),
        ([view(azimuth=None)], r"^views\[0\]: azimuth must be a number of degrees$"),
        ([view(azimuth=True)], "azimuth must be a number"),
        ([view(azimuth="90")], "azimuth must be a number"),
        ([view(azimuth=float("inf"))], "azimuth must be a number"),
        ([view(elevation=91)], r"^views\[0\]: elevation must be a number of degrees between -90 and 90$"),
        ([view(elevation=float("nan"))], "elevation must be a number"),
        ([view(weight=0)], r"^views\[0\]: weight must be a number above 0 and at most 100$"),
        ([view(weight=101)], "weight must be a number above 0"),
        ([view(weight=False)], "weight must be a number above 0"),
    ],
)
def test_rejects_bad_views(views, message):
    with pytest.raises(InputError, match=message):
        parse_job({"image_base64": b64(png_bytes()), "views": views}, fallback_id="job")


def test_view_size_is_checked_like_the_picture(monkeypatch):
    monkeypatch.setattr(inputs, "MAX_IMAGE_BYTES", 1000)
    assert len(png_bytes()) < 1000
    with pytest.raises(InputError, match=r"^views\[0\]: image is larger than 20 MB$"):
        parse_job({"image_base64": b64(png_bytes()), "views": [view(image_base64=b64(b"x" * 1001))]}, fallback_id="job")


def test_view_urls_use_the_fetcher():
    seen = []

    def fetch(url):
        seen.append(url)
        return png_bytes(size=(32, 32))

    back = {"image_url": "https://assets.forge3d.app/v.png", "azimuth": 180}
    payload = {"image_base64": b64(png_bytes()), "views": [back]}
    job = parse_job(payload, fallback_id="job", fetch=fetch)
    assert seen == ["https://assets.forge3d.app/v.png"] and job.views[0].image.size == (32, 32)


class ViewRuntime(FakeRuntime):
    """A runtime that takes views and says how many it used."""

    def __init__(self, left_out: int = 0):
        super().__init__()
        self.left_out = left_out

    def generate(self, image, preset, seed, views=()):
        self.calls.append(("generate", preset.pipeline_type, seed, image.size, [(v.azimuth, v.weight) for v in views]))
        self.views_used = len(views) - self.left_out
        return "mesh"


@pytest.mark.parametrize("mode", ["preview", "final"])
def test_views_go_to_the_runtime_for_previews_and_finals(mode):
    runtime = ViewRuntime(left_out=1)
    payload = {"image_base64": b64(png_bytes()), "mode": mode, "seed": 3, "views": [view(90), view(180, weight=2)]}
    job = {"id": "v", "input": payload}
    out = handle_job(job, runtime, FakeStorage(), pack)
    assert "error" not in out and out["views_used"] == 1
    assert runtime.calls[0] == ("generate", PRESETS[mode].pipeline_type, 3, (64, 48), [(90, 1.0), (180, 2.0)])


def test_jobs_without_views_call_the_runtime_as_before_and_report_none():
    runtime = FakeRuntime()  # generate(image, preset, seed): no views parameter
    out = handle_job({"id": "s", "input": {"image_base64": b64(png_bytes())}}, runtime, FakeStorage(), pack)
    assert out["views_used"] == 0 and runtime.calls[0] == ("generate", "1024_cascade", out["seed"], (64, 48))


def test_a_runtime_that_does_not_count_views_is_taken_to_use_them_all():
    class Silent(FakeRuntime):
        def generate(self, image, preset, seed, views=()):
            return super().generate(image, preset, seed)

    payload = {"image_base64": b64(png_bytes()), "views": [view()] * 3}
    out = handle_job({"id": "s", "input": payload}, Silent(), FakeStorage(), pack)
    assert out["views_used"] == 3


def test_bad_views_are_reported_without_running_the_model():
    runtime = ViewRuntime()
    payload = {"image_base64": b64(png_bytes()), "views": [view(azimuth="left")]}
    out = handle_job({"id": "x", "input": payload}, runtime, FakeStorage(), pack)
    assert out == {"error": "invalid input: views[0]: azimuth must be a number of degrees"}
    assert runtime.calls == []
