"""Pixal3DRuntime on a fake pipeline: which pictures and cameras Pixal3D gets, and the GLB's frame."""

import math
import types

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

from forge3d_worker.pipeline import CUTOUT
from forge3d_worker.settings import PRESETS
from pixal3d_worker import cameras, pipeline
from pixal3d_worker.pipeline import Pixal3DRuntime, _ViewAlignedOVoxel
from pixal3d_worker.views import MAIN_PICTURE, MAIN_VIEW, View, ViewCamera, cond_tensor, place_like


def cutout(size=96, box=(30, 20, 66, 80), colour=(30, 160, 60)) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (*colour, 255)), box[:2])
    return image


class FakeModule:
    def __init__(self):
        self.device = "cpu"
        self.use_naf_upsample = False

    def cpu(self):
        self.device = "cpu"
        return self

    def cuda(self):
        self.device = "cuda"
        return self

    def to(self, device):
        self.device = str(device)
        return self


class FakeRemover(FakeModule):
    """BiRefNet's wrapper: the picture with an alpha (here: everything not white is the object)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        rgb = np.asarray(image.convert("RGB"))
        alpha = np.where(rgb.min(axis=-1) < 250, 255, 0).astype(np.uint8)
        out = image.convert("RGB").copy()
        out.putalpha(Image.fromarray(alpha))
        return out


class FakePipeline:
    """The parts of Pixal3D's pipelines the runtime touches."""

    def __init__(self, failures=()):
        self.models = {"sparse_structure_flow_model": FakeModule(), "tex_slat_decoder": FakeModule()}
        self.rembg_model = FakeRemover()
        for name in pipeline.STAGES:
            setattr(self, name, FakeModule())
        self.low_vram = False
        self._device = torch.device("cuda")
        self.runs = []
        self.failures = list(failures)

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        if not self.low_vram:
            for model in self.models.values():
                model.to(device)

    def preprocess_image(self, image):
        """Upstream's: matte, crop around the object, premultiply onto black (RGB)."""
        output = self.rembg_model(image.convert("RGB"))
        box = output.getchannel("A").getbbox()
        return Image.alpha_composite(Image.new("RGBA", output.size, (0, 0, 0, 255)), output).crop(box).convert("RGB")

    def _run(self, kind, payload, kwargs):
        self.runs.append((kind, payload, kwargs, self.low_vram))
        if self.failures:
            raise self.failures.pop(0)
        return [types.SimpleNamespace(kind=kind)]

    def run_mv(self, views, **kwargs):
        return self._run("mv", views, kwargs)

    def run(self, image, camera_params=None, **kwargs):
        return self._run("single", (image, camera_params), kwargs)


def runtime(multiview=True, main=MAIN_VIEW, azimuths=None, failures=()) -> Pixal3DRuntime:
    rt = Pixal3DRuntime.__new__(Pixal3DRuntime)
    rt.pipeline = FakePipeline(failures)
    rt.multiview, rt.main, rt.azimuths = multiview, main, azimuths
    rt.moge, rt.fov = None, math.radians(30)
    return rt


def photo() -> Image.Image:
    """An RGB picture: a red block on white."""
    image = Image.new("RGB", (120, 100), (255, 255, 255))
    image.paste(Image.new("RGB", (40, 60), (220, 30, 30)), (50, 20))
    return image


def six_views(alpha=True):
    out = []
    for azimuth in (0, 45, 90, 180, 270, 315):
        image = cutout(colour=(azimuth % 256, 100, 100))
        if not alpha:
            flat = Image.new("RGB", image.size, (255, 255, 255))
            flat.paste(image, mask=image.getchannel("A"))
            image = flat
        out.append(View(image=image, azimuth=float(azimuth), elevation=0.0))
    return out


def test_views_with_the_redrawn_front_as_main():
    rt = runtime()
    given = six_views()
    mesh = rt.generate_views(photo(), list(reversed(given)), PRESETS["final"], seed=7)
    kind, packed, kwargs, _ = rt.pipeline.runs[0]
    assert kind == "mv" and kwargs == {"seed": 7, "pipeline_type": "1024_cascade"}
    assert packed["images"][512].shape[:2] == (1, 6)
    assert torch.equal(packed["images"][512][0, 0], cond_tensor(given[0].image, 512))
    positions = packed["transform_matrix"][0, :, :3, 3]  # the cameras: d (sin a, -cos a, 0)
    angles = torch.atan2(positions[:, 0], -positions[:, 1]).rad2deg() % 360
    assert [round(float(a)) for a in angles] == [0, 45, 90, 180, 270, 315]
    front = cameras.front_camera(float(packed["camera_distance"][0, 0]))
    assert torch.equal(packed["transform_matrix"][0, 0], torch.tensor(front, dtype=torch.float32))
    assert rt.views_used == 6 and rt.pipeline_used == "pixal3d-mv-1024_cascade"
    assert getattr(mesh, CUTOUT).mode == "RGBA"  # the picture's cutout, for the final's projection


def test_views_with_the_picture_as_main():
    rt = runtime(main=MAIN_PICTURE)
    given = six_views()
    rt.generate_views(photo(), given, PRESETS["preview"], seed=1)
    _, packed, kwargs, _ = rt.pipeline.runs[0]
    assert kwargs["pipeline_type"] == "1024_cascade"  # previews too: Pixal3D has no 512 pipeline
    assert packed["images"][512].shape[:2] == (1, 6)
    cut = rt.pipeline.rembg_model(photo())
    expected = cond_tensor(place_like(cut, 96, given[0].image), 512)
    assert torch.allclose(packed["images"][512][0, 0], expected, atol=1e-6)
    assert rt.views_used == 5  # the redrawn front is replaced by the picture


def test_view_subset_and_missing_front():
    rt = runtime(azimuths=[0, 90, 180, 270])
    rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    assert rt.pipeline.runs[-1][1]["images"][512].shape[1] == 4 and rt.views_used == 4
    rt = runtime(azimuths=[90, 180, 270])  # no front among them: the picture is the main view
    rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    assert rt.pipeline.runs[-1][1]["images"][512].shape[1] == 4 and rt.views_used == 3


def test_views_without_alpha_are_matted_uncropped():
    rt = runtime()
    rt.generate_views(photo(), six_views(alpha=False), PRESETS["final"], seed=1)
    packed = rt.pipeline.runs[0][1]
    assert rt.pipeline.rembg_model.calls == 1 + 6  # the picture, then each view
    # Same framing as the cutout views: the matte isn't cropped or rescaled
    assert torch.allclose(packed["images"][512][0, 0], cond_tensor(cutout(colour=(0, 100, 100)), 512), atol=1e-6)


def test_the_rig_frames_the_object_to_fill_the_cube():
    # The test views' object is 60 of 96 px tall, whatever frame the job's camera claims
    rt = runtime()
    rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1, camera=ViewCamera(half_extent=0.6))
    packed = rt.pipeline.runs[0][1]
    half = 0.99 / (2 * 60 / 96)
    expected = cameras.distance_for_half_width(half, cameras.NEAR_ORTHO_FOV_DEG)
    assert float(packed["camera_distance"][0, 0]) == pytest.approx(expected, rel=1e-6)
    assert rt.last_camera["half_extent"] == pytest.approx(half, abs=1e-4)
    assert rt.last_camera["given_half_extent"] == 0.6 and rt.last_camera["extent"] == pytest.approx(0.75)
    assert rt.last_camera["views"][0] == [0.0, 0.0]


def test_freeing_gpu_memory_clears_a_stale_cuda_error(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "clear_cuda_error", lambda: "CUDA error: out of memory")
    Pixal3DRuntime._free_gpu_memory()
    assert "cleared a stale CUDA error" in capsys.readouterr().out
    monkeypatch.undo()
    assert pipeline.clear_cuda_error() is None  # without CUDA (these tests) there is nothing to clear


def test_single_view_weights_refuse_views_and_run_the_picture():
    rt = runtime(multiview=False)
    with pytest.raises(RuntimeError):
        rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    rt.generate(photo(), PRESETS["final"], seed=3)
    kind, (image, camera), kwargs, _ = rt.pipeline.runs[0]
    assert kind == "single" and kwargs["preprocess_image"] is False and kwargs["seed"] == 3
    assert camera == {
        "camera_angle_x": pytest.approx(math.radians(30)),
        "distance": pytest.approx(0.5 / math.tan(math.radians(15))),
        "mesh_scale": 1.0,
    }
    assert rt.views_used == 0 and rt.pipeline_used == "pixal3d-1024_cascade"


def test_a_picture_alone_on_the_multiview_weights_is_a_one_view_set():
    rt = runtime()
    rt.generate(photo(), PRESETS["final"], seed=3)
    kind, packed, _, _ = rt.pipeline.runs[0]
    assert kind == "mv" and packed["images"][512].shape[1] == 1
    assert float(packed["camera_distance"][0, 0]) == pytest.approx(0.5 / math.tan(math.radians(15)), rel=1e-5)
    assert float(packed["camera_angle_x"][0, 0]) == pytest.approx(math.radians(30), rel=1e-6)


def test_out_of_memory_retries_in_low_vram_mode():
    rt = runtime(failures=[torch.OutOfMemoryError("CUDA out of memory")])
    rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    assert [low for *_, low in rt.pipeline.runs] == [False, True]
    assert rt.pipeline.low_vram is False  # back to normal for the next job
    assert all(getattr(rt.pipeline, name).device == "cuda" for name in pipeline.STAGES)


def test_other_failures_are_not_retried():
    rt = runtime(failures=[ValueError("boom")])
    with pytest.raises(ValueError):
        rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    assert len(rt.pipeline.runs) == 1


def test_to_glb_comes_out_upright_facing_plus_z():
    native = np.array([[0.1, 0.4, 0.3], [-0.2, -0.1, 0.0], [0.0, 0.0, -0.4]])

    def to_glb(**kwargs):  # o_voxel's to_glb ends by writing (x, y, z) as (x, z, -y)
        return trimesh.Trimesh(vertices=native[:, [0, 2, 1]] * [1, 1, -1], faces=[[0, 1, 2]], process=False)

    wrapped = _ViewAlignedOVoxel(types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=to_glb, other=1)))
    glb = wrapped.postprocess.to_glb(vertices=None)
    np.testing.assert_allclose(glb.vertices, native, atol=1e-12)
    assert wrapped.postprocess.other == 1
