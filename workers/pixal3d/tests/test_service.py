"""The Pixal3D worker's job contract: production's, plus views."""

import base64
import io

import pytest
from PIL import Image

from forge3d_worker.storage import InlineStorage
from pixal3d_worker import service
from pixal3d_worker.views import ViewCamera


def png(size=(32, 32), alpha=True) -> str:
    image = Image.new("RGBA" if alpha else "RGB", size, (0, 0, 0, 0) if alpha else (255, 255, 255))
    image.paste(Image.new(image.mode, (10, 12), (200, 0, 0, 255) if alpha else (200, 0, 0)), (11, 10))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeRuntime:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.last_projection = None

    def generate(self, image, preset, seed):
        self.calls.append(("generate", preset.pipeline_type, seed))
        self.pipeline_used, self.views_used = "pixal3d-1024_cascade", 0
        self.last_camera = {"fov_deg": 31.0, "tilt": {"elevation": 21.2, "roll": 0.0, "source": "preview"}}
        self.last_pose = {"applied": True, "reason": "found", "pose": {"elevation": 21.2}, "iou": 0.98}
        self.last_thin = {"extents": [0.15, 0.86, 1.0], "ratio": 0.15, "threshold": 0.2, "thin": True, "source": "preview", "decided": True}
        self.last_weights = "single"
        return "mesh"

    def generate_views(self, image, views, preset, seed, camera):
        if self.fail:
            raise self.fail
        self.calls.append(("views", [(v.azimuth, v.elevation) for v in views], camera, seed))
        self.pipeline_used, self.views_used = "pixal3d-mv-1024_cascade", len(views)
        self.last_camera = {"views": [[0, 0]], "half_extent": camera.half_extent}
        self.last_thin, self.last_weights = None, "multiview"
        return "mesh"

    def export(self, mesh, preset):
        if preset.project_picture:
            self.last_projection = {"applied": True, "reason": "applied"}
        if getattr(self, "last_pose", None):
            self.last_level = {"applied": True, "elevation": 21.2, "roll": 0.0}
        return b"glb", 1234


def job(**extra):
    return {"id": "job-1", "input": {"image_base64": png(alpha=False), "mode": "final", "seed": 5, **extra}}


def run(job_, runtime):
    return service.handle_job(job_, runtime, InlineStorage(), lambda raw, size: raw + b"!")


def test_views_reach_the_runtime_and_the_result():
    runtime = FakeRuntime()
    views = [{"image_base64": png(), "azimuth": a, "elevation": 0} for a in (0, 90, 180, 270)]
    result = run(job(views=views), runtime)
    assert "error" not in result, result
    kind, angles, camera, seed = runtime.calls[0]
    assert kind == "views" and angles == [(0, 0), (90, 0), (180, 0), (270, 0)] and seed == 5
    assert camera == ViewCamera()
    assert result["views_used"] == 4
    assert result["pipeline"] == "pixal3d-mv-1024_cascade"
    assert result["projection"] == {"applied": True, "reason": "applied"}
    assert result["camera"]["half_extent"] == 0.55
    assert result["weights"] == "multiview" and "thin" not in result
    assert result["credits"] == list(service.CREDITS)
    assert result["glb"]["key"] == "ai/job-1/final-5.glb" and result["bytes"] == 4


def test_the_views_camera_is_passed_on():
    runtime = FakeRuntime()
    views = [{"image_base64": png(), "azimuth": 0}]
    run(job(views=views, camera={"type": "orthographic", "half_extent": 0.6}), runtime)
    assert runtime.calls[0][2] == ViewCamera(half_extent=0.6)


def test_without_views_it_is_production():
    runtime = FakeRuntime()
    result = run(job(mode="preview"), runtime)
    assert runtime.calls == [("generate", "512", 5)]
    assert result["views_used"] == 0 and result["mode"] == "preview"
    assert result["camera"]["fov_deg"] == 31.0 and result["camera"]["tilt"]["elevation"] == 21.2
    assert result["pose"]["applied"] is True and result["level"] == {"applied": True, "elevation": 21.2, "roll": 0.0}
    assert result["weights"] == "single" and result["thin"]["ratio"] == 0.15 and result["thin"]["thin"] is True
    assert "projection" not in result


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"views": "all"}, "views must be a non-empty list"),
        ({"views": [{"image_base64": png((32, 16)), "azimuth": 0}]}, "square"),
        ({"views": [{"image_base64": png(), "azimuth": 0}], "camera": {"type": "perspective"}}, "orthographic"),
    ],
)
def test_bad_views_are_the_callers_error(extra, message):
    runtime = FakeRuntime()
    result = run(job(**extra), runtime)
    assert result["error"].startswith("invalid input:") and message in result["error"]
    assert runtime.calls == []


def test_generation_failures_are_reported_like_production():
    runtime = FakeRuntime(fail=RuntimeError("CUDA error: an illegal memory access was encountered"))
    result = run(job(views=[{"image_base64": png(), "azimuth": 0}]), runtime)
    assert result["error"].startswith("generation failed: RuntimeError")
    assert result["refresh_worker"] is True
