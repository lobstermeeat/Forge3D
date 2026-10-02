"""Multiview job handling with a fake generator."""

import base64
import io

import pytest
from PIL import Image

from multiview_worker.cameras import AZIMUTHS
from multiview_worker.inputs import DEFAULT_PROMPT, InputError, parse_job
from multiview_worker.service import handle_job
from multiview_worker.storage import InlineStorage


class FakeStorage:
    def __init__(self):
        self.saved = {}

    def put(self, key, data, content_type):
        self.saved[key] = (data, content_type)
        return {"key": key, "url": f"https://assets.example.com/{key}"}


class FakeGenerator:
    """Six views, each a different shade, with a transparent border like a real cutout."""

    def __init__(self, count=len(AZIMUTHS)):
        self.calls = []
        self.count = count
        self.last_timings = {"cutout_s": 0.1, "views_s": 2.0, "view_cutouts_s": 0.3}

    def __call__(self, image, seed, prompt):
        self.calls.append((image, seed, prompt))
        views = []
        for index in range(self.count):
            view = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            view.paste((40 * index, 100, 200, 255), (4, 4, 12, 12))
            views.append(view)
        return views


def picture_b64(mode="RGB", size=(32, 24)):
    buffer = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 255)[: len(mode)]).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def job(**fields):
    return {"id": "fc-01K6TEST", "input": {"image_base64": picture_b64(), **fields}}


def test_six_rgba_views_at_the_protocol_azimuths():
    generate, storage = FakeGenerator(), FakeStorage()
    out = handle_job(job(seed=7, request_id="gen_42"), generate, storage)

    assert [view["azimuth"] for view in out["views"]] == [0, 45, 90, 180, 270, 315]
    assert all(view["elevation"] == 0 for view in out["views"])
    assert out["views"][2]["key"] == "ai/gen_42/view-7-90.png"
    assert out["views"][2]["url"] == "https://assets.example.com/ai/gen_42/view-7-90.png"
    data, content_type = storage.saved["ai/gen_42/view-7-180.png"]
    assert content_type == "image/png"
    view = Image.open(io.BytesIO(data))
    assert view.mode == "RGBA" and view.getpixel((0, 0))[3] == 0 and view.getpixel((8, 8)) == (120, 100, 200, 255)
    assert out["request_id"] == "gen_42" and out["seed"] == 7
    assert out["camera"]["type"] == "orthographic" and out["camera"]["image_size"] == 768
    assert out["seconds"] >= 0
    assert out["timings"]["views_s"] == 2.0 and "upload_s" in out["timings"]


def test_the_picture_reaches_the_generator_with_its_seed_and_caption():
    generate = FakeGenerator()
    handle_job(job(seed=3), generate, FakeStorage())
    image, seed, prompt = generate.calls[0]
    assert image.size == (32, 24) and image.mode == "RGB"
    assert seed == 3 and prompt == DEFAULT_PROMPT == "high quality"

    handle_job(job(seed=3, prompt="  a retro arcade machine "), generate, FakeStorage())
    assert generate.calls[1][2] == "a retro arcade machine"


def test_a_cutout_keeps_its_transparency():
    generate = FakeGenerator()
    payload = {"image_base64": picture_b64("RGBA"), "seed": 1}
    handle_job({"id": "j", "input": payload}, generate, FakeStorage())
    assert generate.calls[0][0].mode == "RGBA"


def test_defaults_to_a_random_seed_and_the_call_id():
    out = handle_job(job(), FakeGenerator(), FakeStorage())
    assert 0 <= out["seed"] < 2**31
    assert out["request_id"] == "fc-01K6TEST"
    assert out["views"][0]["key"] == f"ai/fc-01K6TEST/view-{out['seed']}-0.png"


def test_without_r2_the_views_come_back_inline():
    out = handle_job(job(seed=5), FakeGenerator(), InlineStorage())
    view = out["views"][1]
    assert view["url"] is None
    assert Image.open(io.BytesIO(base64.b64decode(view["base64"]))).mode == "RGBA"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not an object", "input must be an object"),
        ({}, "exactly one of image_url or image_base64"),
        ({"image_base64": "aGk=", "image_url": "https://a.example/x.png"}, "exactly one"),
        ({"image_base64": "%%%"}, "not valid base64"),
        ({"image_base64": base64.b64encode(b"not a picture").decode()}, "could not be decoded"),
        ({"image_base64": picture_b64(), "seed": -1}, "seed must be"),
        ({"image_base64": picture_b64(), "seed": True}, "seed must be"),
        ({"image_base64": picture_b64(), "request_id": "a/b"}, "request_id must be"),
        ({"image_base64": picture_b64(), "prompt": 5}, "prompt must be a string"),
        ({"image_base64": picture_b64(), "prompt": "x" * 501}, "longer than 500"),
        ({"image_url": "http://assets.example.com/x.png"}, "https"),
    ],
)
def test_rejects_bad_input(payload, message):
    with pytest.raises(InputError, match=message):
        parse_job(payload, "job")


def test_image_urls_need_an_allow_list(monkeypatch):
    monkeypatch.delenv("ALLOWED_IMAGE_HOSTS", raising=False)
    out = handle_job({"id": "j", "input": {"image_url": "https://assets.example.com/p.png"}}, FakeGenerator(), FakeStorage())
    assert out == {"error": "invalid input: image_url is disabled: set ALLOWED_IMAGE_HOSTS or send image_base64"}

    monkeypatch.setenv("ALLOWED_IMAGE_HOSTS", "assets.example.com")
    fetched = []

    def fetch(url):
        fetched.append(url)
        return base64.b64decode(picture_b64())

    out = handle_job({"id": "j", "input": {"image_url": "https://assets.example.com/p.png", "seed": 2}}, FakeGenerator(), FakeStorage(), fetch)
    assert fetched == ["https://assets.example.com/p.png"] and len(out["views"]) == 6


def test_no_object_in_the_picture_is_the_callers_problem():
    def empty(image, seed, prompt):
        raise InputError("no object found in the picture")

    out = handle_job(job(), empty, FakeStorage())
    assert out == {"error": "invalid input: no object found in the picture"}


def test_gpu_faults_restart_the_worker_but_other_failures_dont():
    def gpu_fault(image, seed, prompt):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    out = handle_job(job(), gpu_fault, FakeStorage())
    assert out["refresh_worker"] is True
    assert out["error"].startswith("generation failed: RuntimeError: CUDA out of memory")

    class BrokenStorage:
        def put(self, key, data, content_type):
            raise ConnectionError("R2 unreachable")

    out = handle_job(job(), FakeGenerator(), BrokenStorage())
    assert out == {"error": "generation failed: ConnectionError: R2 unreachable"}


def test_a_generator_that_returns_the_wrong_number_of_views_fails_the_job():
    out = handle_job(job(), FakeGenerator(count=4), FakeStorage())
    assert out == {"error": "generation failed: RuntimeError: expected 6 views, got 4"}
