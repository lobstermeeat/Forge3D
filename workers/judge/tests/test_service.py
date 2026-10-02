"""A judge job: base64 images in, the verdicts and pick out, and every input checked."""

import base64
import io
import json

import pytest
from PIL import Image

from judge_worker import service

COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]


def encoded(image, kind="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=kind)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Model:
    """Picks the version shown in pure green, publishes it, asks for edits on the rest."""

    def __init__(self):
        self.chats = []

    def __call__(self, messages, max_new_tokens=1024):
        self.chats.append(max_new_tokens)
        content = messages[0]["content"]
        answer, letter = {}, None
        for text, image in zip(content[2::2], content[3::2]):
            letter = text["text"].strip().removeprefix("Version ").rstrip(":")
            pixel = image["image"].getpixel((0, 0))  # some candidates come as JPEG
            green = max(abs(a - b) for a, b in zip(pixel, COLOURS[1])) <= 8
            answer[letter] = {"problems": "none" if green else "a smear", "verdict": "publish" if green else "edits"}
            if green:
                answer["best"] = letter
        return json.dumps({**answer, "why": "the green one is clean"})


def job(count=4, **extra):
    return {
        "picture_png": encoded(Image.new("RGB", (64, 64), "white")),
        "candidates_png": [encoded(Image.new("RGB", (96, 64), COLOURS[i]), "JPEG" if i % 2 else "PNG") for i in range(count)],
        **extra,
    }


def test_a_job_is_judged_and_named():
    model = Model()
    result = service.handle_job(job(prompt=" a lamp ", id="fc-01K6"), model, name="8b")
    assert result["best"] == 1 and result["verdicts"] == ["edits", "publish", "edits", "edits"]
    assert result["letters"] == ["K", "L", "M", "N"] and result["model"] == "8b"
    assert result["why"] == "the green one is clean" and model.chats == [1024]


def test_a_seed_shuffles_the_order_the_same_way_every_time():
    first = service.handle_job(job(seed=7), Model())
    again = service.handle_job(job(seed=7), Model())
    assert first["letters"] == again["letters"] == ["KLMN"[i] for i in _positions(service.shuffled(4, 7))]
    assert first["best"] == 1 and first["verdicts"][1] == "publish"
    letters = {tuple(service.handle_job(job(seed=seed), Model())["letters"]) for seed in range(6)}
    assert len(letters) > 1  # different seeds, different orders


def _positions(order):
    """Each candidate's shown position, from the order they are shown in."""
    return [order.index(number) for number in range(len(order))]


def test_an_order_can_be_given():
    result = service.handle_job(job(order=[3, 2, 1, 0], max_new_tokens=300), Model())
    assert result["letters"] == ["N", "M", "L", "K"] and result["best"] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"picture_png": None}, "picture_png must be a base64 string"),
        ({"picture_png": "not base64!"}, "picture_png is not valid base64"),
        ({"picture_png": base64.b64encode(b"not an image").decode()}, "picture_png could not be decoded"),
        ({"candidates_png": []}, "candidates_png must be a list of 1 to 8 base64 images"),
        ({"candidates_png": "abc"}, "candidates_png must be a list"),
        ({"candidates_png": [encoded(Image.new("RGB", (8, 8)))] * 9}, "1 to 8"),
        ({"candidates_png": [encoded(Image.new("RGB", (8, 8))), 5]}, "candidates_png[1] must be a base64 string"),
        ({"prompt": 3}, "prompt must be a string"),
        ({"prompt": "x" * 501}, "prompt is longer than 500 characters"),
        ({"order": [0, 1, 2]}, "order must list each candidate's index (0 to 3) once"),
        ({"order": [0, 1, 2, 2]}, "order must list"),
        ({"order": [0, 1, 2, True]}, "order must list"),
        ({"seed": -1}, "seed must be an integer"),
        ({"seed": 1, "order": [0, 1, 2, 3]}, "give order or seed, not both"),
        ({"max_new_tokens": 0}, "max_new_tokens must be an integer between 1 and 4096"),
        ({"max_new_tokens": 5000}, "max_new_tokens"),
        ({"temperature": 0.7}, "unknown fields: temperature"),
    ],
)
def test_bad_input_is_refused(change, message):
    model = Model()
    payload = {**job(), **change}
    result = service.handle_job(payload, model)
    assert result["error"].startswith("invalid input: ") and message in result["error"]
    assert model.chats == []


def test_oversized_images_are_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(service, "MAX_IMAGE_SIDE", 32)
    assert service.handle_job(job(), Model()) == {"error": "invalid input: picture_png is larger than 32px on a side"}
    monkeypatch.setattr(service, "MAX_IMAGE_BYTES", 10)
    assert "larger than 20 MB" in service.handle_job(job(), Model())["error"]
    assert service.handle_job("nope", Model()) == {"error": "invalid input: input must be an object"}


def test_a_failing_model_is_reported_and_a_gpu_fault_restarts_the_container():
    def oom(messages, max_new_tokens=1024):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    result = service.handle_job(job(), oom)
    assert result == {"error": "judge failed: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", "refresh_worker": True}

    def broken(messages, max_new_tokens=1024):
        raise KeyError("pixel_values")

    assert service.handle_job(job(), broken) == {"error": "judge failed: KeyError: 'pixel_values'"}
