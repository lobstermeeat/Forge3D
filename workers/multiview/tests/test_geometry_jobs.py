"""Image+geometry jobs (the GeometryViews class in modal_app.py) with a fake model."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from multiview_worker.geometry import CONTROL_SHAPE, pack_control
from multiview_worker.inputs import InputError
from multiview_worker.service import handle_geometry_job, parse_geometry_job

SIZE = CONTROL_SHAPE[-1]


def control():
    array = np.full(CONTROL_SHAPE, 0.5, dtype=np.float32)
    array[:, :, 200:500, 300:400] = np.linspace(0.1, 0.9, 6, dtype=np.float32)[None, :, None, None]
    return array


CONTROL_PNGS = pack_control(control())


def picture_b64(mode="RGB", size=(32, 24)):
    buffer = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 255)[: len(mode)]).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def job(**fields):
    return {"image_base64": picture_b64(), "control_pngs": CONTROL_PNGS, **fields}


class FakeModel:
    """GeometryViewGenerator's stand-in: six views, each a different shade."""

    def __init__(self, count=6, error=None):
        self.calls = []
        self.count = count
        self.error = error
        self.last_timings = {"cutout_s": 0.2, "views_s": 40.0}

    def draw_views(self, reference, control, prompt, seed, **settings):
        self.calls.append((reference, control, prompt, seed, settings))
        if self.error:
            raise self.error
        return [Image.new("RGB", (SIZE, SIZE), (30 * index, 60, 90)) for index in range(self.count)]


def test_six_views_come_back_as_base64_pngs_in_order():
    model = FakeModel()
    out = handle_geometry_job(job(seed=4), model)
    assert set(out) == {"views", "seconds", "timings"}
    assert len(out["views"]) == 6 and out["seconds"] >= 0
    for index, encoded in enumerate(out["views"]):
        view = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert (view.format, view.mode, view.size) == ("PNG", "RGB", (SIZE, SIZE))
        assert view.getpixel((0, 0)) == (30 * index, 60, 90)
    assert out["timings"] == {"cutout_s": 0.2, "views_s": 40.0}


def test_the_job_reaches_the_model_with_upstreams_defaults():
    model = FakeModel()
    handle_geometry_job(job(), model)
    reference, array, prompt, seed, settings = model.calls[0]
    assert reference.size == (32, 24) and reference.mode == "RGB"
    assert array.shape == CONTROL_SHAPE and np.abs(array - control()).max() <= 0.5 / 255 + 1e-6
    assert prompt == "high quality" and seed == 0
    assert settings == {"steps": 30, "guidance": 3.0, "reference_scale": 1.0, "control_scale": 1.0}


def test_settings_and_caption_are_passed_on():
    model = FakeModel()
    payload = job(prompt="  a vintage film camera ", seed=9, steps=50, guidance=5, reference_scale=0.7, control_scale=1.5)
    handle_geometry_job({**payload, "id": "fc-01K6TEST"}, model)  # run_job adds the call's id
    _, _, prompt, seed, settings = model.calls[0]
    assert prompt == "a vintage film camera" and seed == 9
    assert settings == {"steps": 50, "guidance": 5.0, "reference_scale": 0.7, "control_scale": 1.5}


def test_a_cutout_keeps_its_transparency():
    model = FakeModel()
    handle_geometry_job(job(image_base64=picture_b64("RGBA")), model)
    assert model.calls[0][0].mode == "RGBA"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not an object", "input must be an object"),
        ({"input": job()}, "unknown fields: input"),
        (job(guidence=4), "unknown fields: guidence"),
        ({"control_pngs": CONTROL_PNGS}, "image_base64 must be a base64 string"),
        (job(image_base64="%%%"), "image_base64 is not valid base64"),
        (job(image_base64=base64.b64encode(b"not a picture").decode()), "image could not be decoded"),
        ({"image_base64": picture_b64()}, "control_pngs must be a list of 12 PNGs"),
        (job(control_pngs=CONTROL_PNGS[:6]), "control_pngs must be a list of 12 PNGs"),
        (job(control_pngs=CONTROL_PNGS[:11] + ["%%%"]), "control PNG 11 is not valid base64"),
        (job(prompt=5), "prompt must be a string"),
        (job(prompt="x" * 501), "longer than 500"),
        (job(seed=-1), "seed must be"),
        (job(seed=True), "seed must be"),
        (job(seed=1.5), "seed must be"),
        (job(steps=0), "steps must be an integer between 1 and 100"),
        (job(steps=101), "steps must be"),
        (job(steps=30.0), "steps must be"),
        (job(guidance="3"), "guidance must be a number"),
        (job(guidance=float("nan")), "guidance must be a number"),
        (job(guidance=True), "guidance must be a number"),
        (job(guidance=21), "guidance must be between 0 and 20"),
        (job(reference_scale=-0.1), "reference_scale must be between 0 and 3"),
        (job(control_scale=3.5), "control_scale must be between 0 and 3"),
    ],
)
def test_rejects_bad_input(payload, message):
    with pytest.raises(InputError, match=message):
        parse_geometry_job(payload)
    model = FakeModel()
    out = handle_geometry_job(payload, model)
    assert out["error"].startswith("invalid input: ") and model.calls == []


def test_no_object_in_the_picture_is_the_callers_problem():
    out = handle_geometry_job(job(), FakeModel(error=InputError("no object found in the picture")))
    assert out == {"error": "invalid input: no object found in the picture"}


def test_gpu_faults_restart_the_worker_but_other_failures_dont():
    out = handle_geometry_job(job(), FakeModel(error=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
    assert out["refresh_worker"] is True
    assert out["error"].startswith("generation failed: RuntimeError: CUDA out of memory")
    out = handle_geometry_job(job(), FakeModel(error=ValueError("bad scheduler")))
    assert out == {"error": "generation failed: ValueError: bad scheduler"}


def test_a_model_that_returns_the_wrong_number_of_views_fails_the_job():
    out = handle_geometry_job(job(), FakeModel(count=4))
    assert out == {"error": "generation failed: RuntimeError: expected 6 views, got 4"}
