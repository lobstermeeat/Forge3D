"""Reference-image job handling with a fake image generator."""

import pytest
from PIL import Image

from reference_worker.service import InputError, build_prompt, handle_job, parse


class FakeStorage:
    def __init__(self):
        self.saved = {}

    def put(self, key, data, content_type):
        self.saved[key] = (data, content_type)
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def fake_generate(calls):
    def generate(prompt, seed):
        calls.append((prompt, seed))
        return Image.new("RGB", (8, 8), (seed % 255, 0, 0))

    return generate


def test_generates_consecutive_seeds_and_uploads_pngs():
    calls, storage = [], FakeStorage()
    out = handle_job({"id": "j", "input": {"prompt": "a brass pocket watch", "count": 3, "seed": 10, "request_id": "gen_7"}}, fake_generate(calls), storage)

    assert [c[1] for c in calls] == [10, 11, 12]
    assert calls[0][0].startswith("A studio product photo of a brass pocket watch: an angled perspective shot")
    assert [img["seed"] for img in out["images"]] == [10, 11, 12]
    assert out["images"][0]["url"] == "https://assets.example.com/ai/gen_7/reference-10.png"
    data, content_type = storage.saved["ai/gen_7/reference-12.png"]
    assert content_type == "image/png" and data[:8] == b"\x89PNG\r\n\x1a\n"


def test_defaults_to_four_images_with_a_random_seed():
    calls = []
    out = handle_job({"id": "j", "input": {"prompt": "a teapot"}}, fake_generate(calls), FakeStorage())
    assert len(out["images"]) == 4 and calls[1][1] == calls[0][1] + 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "prompt is required"),
        ({"prompt": "   "}, "prompt is required"),
        ({"prompt": "x" * 501}, "longer than 500"),
        ({"prompt": "a", "count": 0}, "count must be between 1 and 4"),
        ({"prompt": "a", "count": 5}, "count must be between 1 and 4"),
        ({"prompt": "a", "seed": -3}, "seed must be"),
        ({"prompt": "a", "request_id": "a/b"}, "request_id must be"),
    ],
)
def test_rejects_bad_input(payload, message):
    with pytest.raises(InputError, match=message):
        parse(payload, "job")


def test_gpu_faults_restart_the_worker_but_other_failures_dont():
    def gpu_fault(prompt, seed):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    out = handle_job({"id": "j", "input": {"prompt": "a lamp"}}, gpu_fault, FakeStorage())
    assert out["refresh_worker"] is True
    assert out["error"].startswith("generation failed: RuntimeError: CUDA error")

    class BrokenStorage:
        def put(self, key, data, content_type):
            raise ConnectionError("R2 unreachable")

    out = handle_job({"id": "j", "input": {"prompt": "a lamp"}}, fake_generate([]), BrokenStorage())
    assert out == {"error": "generation failed: ConnectionError: R2 unreachable"}


def test_template_takes_the_prompt_without_its_full_stop():
    assert build_prompt("a red kettle.").startswith("A studio product photo of a red kettle: an angled")
