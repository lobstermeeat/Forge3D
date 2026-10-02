"""Pixal3DRuntime on a fake pipeline: which pictures and cameras Pixal3D gets, and the GLB's frame."""

import io
import json
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
        self.devices_at_run = []
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
        self.devices_at_run.append({name: module.device for name, module in self.models.items()})
        if self.failures:
            raise self.failures.pop(0)
        return [types.SimpleNamespace(kind=kind)]

    def run_mv(self, views, **kwargs):
        return self._run("mv", views, kwargs)

    def run(self, image, camera_params=None, **kwargs):
        return self._run("single", (image, camera_params), kwargs)


def runtime(multiview=True, main=MAIN_VIEW, azimuths=None, failures=(), thin=False, single_failures=()) -> Pixal3DRuntime:
    rt = Pixal3DRuntime.__new__(Pixal3DRuntime)
    rt.pipeline = FakePipeline(failures)
    rt.multiview, rt.main, rt.azimuths = multiview, main, azimuths
    rt.moge, rt.fov = None, math.radians(30)
    rt.single, rt.swapped, rt.thin_ratio = None, (), None
    if thin:  # the single-view flow models beside the multi-view ones, as __init__ loads them
        rt.thin_ratio = pipeline.THIN_RATIO
        rt.single = FakePipeline(single_failures)
        rt.single.models["tex_slat_decoder"] = rt.pipeline.models["tex_slat_decoder"]  # the same file: shared
        rt.single.rembg_model = rt.pipeline.rembg_model
        rt.swapped = ("sparse_structure_flow_model",)
        for name in pipeline.STAGES:
            getattr(rt.single, name).cuda()
        rt.pipeline.to(torch.device("cuda"))
        rt.single.models["sparse_structure_flow_model"].cpu()  # off the GPU between thin objects
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


# --- Levelling ----------------------------------------------------------------------------------------


def tilted_plate(elevation: float) -> trimesh.Trimesh:
    """to_glb's result for a flat plate, as Pixal3D builds it when the picture looked down by elevation."""
    from pixal3d_worker import level

    box = trimesh.creation.box(extents=(1.0, 0.05, 1.0))
    box.apply_translation([0.1, 0.0, -0.1])
    box.apply_transform(level.level_matrix(elevation).T)  # into the camera-aligned frame
    texture = Image.new("RGB", (8, 8), (180, 90, 40))
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(box.vertices), 2)), material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture)
    )
    return box


class FakeOVoxel:
    """o_voxel.postprocess.to_glb returning a given mesh (already in the GLB's frame: the frame change is undone)."""

    def __init__(self, glb):
        self.glb = glb
        self.calls = 0

    def to_glb(self, **kwargs):
        self.calls += 1
        out = self.glb.copy()
        out.apply_transform(np.linalg.inv(cameras.GLB_FROM_TO_GLB))
        return out


def exportable(rt: Pixal3DRuntime, glb) -> types.SimpleNamespace:
    rt._o_voxel = _ViewAlignedOVoxel(types.SimpleNamespace(postprocess=FakeOVoxel(glb)))
    return types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=1 / 64)


def test_a_given_tilt_travels_on_the_mesh_and_levels_the_export():
    rt = runtime(multiview=False)
    rt.level, rt.given_tilt = pipeline.LEVEL_GIVEN, (28.8, 0.0)
    mesh = rt.generate(photo(), PRESETS["final"], seed=3)
    assert getattr(mesh, pipeline.TILT) == (28.8, 0.0)
    assert rt.last_camera["tilt"] == {"elevation": 28.8, "roll": 0.0, "source": "given"}
    assert rt.last_pose is None  # nothing was searched for
    plate = tilted_plate(28.8)
    assert plate.extents[1] > 0.4
    fake_mesh = exportable(rt, plate)
    setattr(fake_mesh, pipeline.TILT, (28.8, 0.0))
    raw, faces = rt.export(fake_mesh, PRESETS["preview"])
    assert raw[:4] == b"glTF" and faces == 12
    assert rt.last_level["applied"] is True and rt.last_level["elevation"] == 28.8
    levelled = trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh")
    assert levelled.extents[1] == pytest.approx(0.05, abs=1e-3) and levelled.extents[0] == pytest.approx(1.0, abs=1e-3)
    assert rt._o_voxel.postprocess.tilt == pipeline.NO_TILT  # reset for the next job
    # A mesh without a tilt (another runtime's, or LEVEL_NONE) is exported as it is
    raw, _ = rt.export(exportable(rt, plate), PRESETS["preview"])
    assert rt.last_level["applied"] is False
    assert trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh").extents[1] > 0.4


class FakeTrellis2:
    """Trellis2Runtime as the levelling uses it: a preview mesh, exported to a GLB."""

    def __init__(self, glb, fail=None):
        self.glb, self.fail, self.calls = glb, fail, []
        self.pipeline_used = "512"

    def generate(self, image, preset, seed):
        self.calls.append(("generate", preset.pipeline_type, seed))
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(**{CUTOUT: cutout()})

    def export(self, mesh, preset):
        self.calls.append(("export", preset.pipeline_type))
        return self.glb.export(file_type="glb"), 12


def test_the_tilt_comes_from_the_picture_against_the_trellis2_preview(monkeypatch):
    from pixal3d_worker import level

    rt = runtime(multiview=False)
    rt.level, rt.trellis2 = pipeline.LEVEL_PREVIEW, FakeTrellis2(tilted_plate(0.0))
    seen = {}

    def estimate_pose(glb, picture, device=None):
        seen["faces"], seen["picture"] = len(glb.faces), picture
        return {"applied": True, "reason": "found", "pose": {"elevation": 21.2, "roll": 0.5}, "iou": 0.98}

    monkeypatch.setattr(level, "estimate_pose", estimate_pose)
    mesh = rt.generate(photo(), PRESETS["final"], seed=11)
    assert rt.trellis2.calls == [("generate", "512", 11), ("export", "512")]  # the preview, at the final's seed
    assert seen["faces"] == 12 and seen["picture"].mode == "RGBA"
    assert getattr(mesh, pipeline.TILT) == (21.2, 0.5)
    assert rt.last_pose["applied"] is True and rt.last_pose["preview"]["pipeline"] == "512"
    assert rt.last_preview[:4] == b"glTF"
    assert rt.last_camera["tilt"] == {"elevation": 21.2, "roll": 0.5, "source": "preview"}


def test_a_failed_gate_or_preview_means_no_tilt(monkeypatch):
    from pixal3d_worker import level

    rt = runtime(multiview=False)
    rt.level, rt.trellis2 = pipeline.LEVEL_PREVIEW, FakeTrellis2(tilted_plate(0.0))
    report = {"applied": False, "reason": "the silhouettes don't match well enough (IoU 0.910 < 0.93)", "pose": {"elevation": 30.0, "roll": 0.0}, "iou": 0.91}
    monkeypatch.setattr(level, "estimate_pose", lambda glb, picture, device=None: dict(report))
    mesh = rt.generate(photo(), PRESETS["final"], seed=11)
    assert getattr(mesh, pipeline.TILT) == (0.0, 0.0)
    assert rt.last_pose["applied"] is False and "IoU" in rt.last_pose["reason"]
    # The preview itself failing is reported the same way, and the job goes on
    rt.trellis2 = FakeTrellis2(tilted_plate(0.0), fail=RuntimeError("CUDA error: out of memory"))
    mesh = rt.generate(photo(), PRESETS["final"], seed=11)
    assert getattr(mesh, pipeline.TILT) == (0.0, 0.0)
    assert rt.last_pose["reason"].startswith("error: RuntimeError") and rt.last_preview is None
    assert len(rt.pipeline.runs) == 2  # Pixal3D ran both times


def test_level_none_leaves_the_camera_frame():
    rt = runtime(multiview=False)
    rt.level = pipeline.LEVEL_NONE
    mesh = rt.generate(photo(), PRESETS["final"], seed=3)
    assert getattr(mesh, pipeline.TILT) == (0.0, 0.0) and rt.last_camera["tilt"]["source"] == "none"


def test_a_posed_picture_puts_the_elevation_in_the_camera(monkeypatch):
    import sys

    patched = []
    fake = types.ModuleType(pipeline.PROJ_MODULE)
    fake.compute_relative_calc_mat = lambda transform_matrix, distance, front: "relative"
    monkeypatch.setitem(sys.modules, pipeline.PROJ_MODULE, fake)
    rt = runtime()
    rt.level, rt.given_tilt, rt.posed = pipeline.LEVEL_GIVEN, (45.0, -7.9), True
    original = rt.pipeline.run_mv

    def run_mv(views, **kwargs):
        patched.append(fake.compute_relative_calc_mat(views["transform_matrix"], None, None))
        return original(views, **kwargs)

    rt.pipeline.run_mv = run_mv
    mesh = rt.generate(photo(), PRESETS["final"], seed=3)
    _, packed, _, _ = rt.pipeline.runs[0]
    camera = packed["transform_matrix"][0, 0].numpy()
    distance = float(packed["camera_distance"][0, 0])
    np.testing.assert_allclose(camera, cameras.orbit_camera(0.0, 45.0, distance), atol=1e-6)
    assert torch.equal(patched[0], packed["transform_matrix"])  # the poses were used as given
    assert fake.compute_relative_calc_mat(None, None, None) == "relative"  # and the patch is undone
    assert getattr(mesh, pipeline.TILT) == (0.0, 0.0)  # nothing to turn afterwards: the model was told
    assert rt.last_camera["posed"]["elevation"] == 45.0
    # No elevation to speak of: the ordinary one-view bundle, front camera
    rt.given_tilt = (0.3, 0.0)
    rt.generate(photo(), PRESETS["final"], seed=3)
    front = cameras.front_camera(distance)
    np.testing.assert_allclose(rt.pipeline.runs[1][1]["transform_matrix"][0, 0].numpy(), front, atol=1e-6)


# --- Thin objects: which weights ------------------------------------------------------------------------


def preview_runtime(glb, elevation=2.5, thin=True, multiview=True, **kwargs) -> Pixal3DRuntime:
    """A runtime whose TRELLIS.2 preview is ``glb`` and whose pose search finds ``elevation``."""
    from pixal3d_worker import level

    rt = runtime(multiview=multiview, thin=thin, **kwargs)
    rt.level, rt.trellis2 = pipeline.LEVEL_PREVIEW, FakeTrellis2(glb)
    level.estimate_pose = lambda glb, picture, device=None: {
        "applied": True, "reason": "found", "pose": {"elevation": elevation, "roll": 0.0}, "iou": 0.98
    }  # fmt: skip
    return rt


@pytest.fixture()
def real_estimate_pose():
    from pixal3d_worker import level

    original = level.estimate_pose
    yield
    level.estimate_pose = original


def test_a_thin_preview_picks_the_single_view_weights(real_estimate_pose):
    rt = preview_runtime(tilted_plate(0.0))  # extents (1.0, 0.05, 1.0): ratio 0.05
    mesh = rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.pipeline.runs == []  # the multi-view weights never ran
    kind, (image, camera), kwargs, low = rt.single.runs[0]
    assert kind == "single" and kwargs["seed"] == 3 and kwargs["preprocess_image"] is False and low is False
    assert camera["distance"] == pytest.approx(0.5 / math.tan(math.radians(15)))
    assert rt.last_weights == "single" and rt.pipeline_used == "pixal3d-1024_cascade"
    assert rt.last_thin["thin"] is True and rt.last_thin["ratio"] == pytest.approx(0.05)
    assert rt.last_thin["extents"] == pytest.approx([0.05, 1.0, 1.0]) and rt.last_thin["decided"] is True
    assert rt.last_thin["threshold"] == pipeline.THIN_RATIO and rt.last_thin["source"] == "preview"
    assert getattr(mesh, pipeline.TILT) == (2.5, 0.0)  # still levelled by the preview's pose
    # The swap: the single-view flow models on the GPU while they ran, the multi-view ones off it...
    assert rt.single.devices_at_run[0]["sparse_structure_flow_model"] == "cuda"
    assert rt.pipeline.models["sparse_structure_flow_model"].device == "cuda"  # ...and back afterwards
    assert rt.single.models["sparse_structure_flow_model"].device == "cpu"
    assert rt.single.models["tex_slat_decoder"] is rt.pipeline.models["tex_slat_decoder"]
    assert rt.pipeline.models["tex_slat_decoder"].device == "cuda"


def test_a_boxy_preview_keeps_the_multi_view_weights(real_estimate_pose):
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    rt = preview_runtime(cube)
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.single.runs == [] and rt.pipeline.runs[0][0] == "mv"
    assert rt.last_weights == "multiview"
    assert rt.last_thin["thin"] is False and rt.last_thin["ratio"] == pytest.approx(1.0)
    assert rt.single.models["sparse_structure_flow_model"].device == "cpu"  # never moved


def test_the_threshold_is_honoured(real_estimate_pose):
    pistol = trimesh.creation.box(extents=(1.0, 0.15, 0.86))  # the pistol's preview, about
    rt = preview_runtime(pistol)
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "single" and rt.last_thin["ratio"] == pytest.approx(0.15)
    rt = preview_runtime(pistol)
    rt.thin_ratio = 0.10  # stricter: the pistol stays with the multi-view weights
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "multiview" and rt.last_thin["threshold"] == 0.10 and rt.last_thin["thin"] is False
    rt = preview_runtime(pistol)
    rt.thin_ratio = 0.15  # at the threshold counts as thin
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "single"


def test_without_a_preview_or_the_rule_the_multi_view_weights_build(real_estimate_pose):
    rt = preview_runtime(tilted_plate(0.0))
    rt.level, rt.given_tilt = pipeline.LEVEL_GIVEN, (5.0, 0.0)  # no preview: nothing measured
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "multiview" and rt.last_thin is None and rt.pipeline.runs[0][0] == "mv"
    rt = preview_runtime(tilted_plate(0.0))
    rt.single, rt.swapped = None, ()  # the rule off (thin_ratio None): the preview is still measured
    rt.thin_ratio = None
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "multiview" and rt.pipeline.runs[0][0] == "mv"
    assert rt.last_thin["thin"] is True and rt.last_thin["decided"] is False
    # A runtime with the single-view weights alone reports them, and measures without deciding
    rt = preview_runtime(trimesh.creation.box(extents=(1.0, 1.0, 1.0)), thin=False, multiview=False)
    rt.thin_ratio = pipeline.THIN_RATIO
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert rt.last_weights == "single" and rt.pipeline.runs[0][0] == "single"
    assert rt.last_thin["thin"] is False and rt.last_thin["decided"] is False


def test_views_always_use_the_multi_view_weights():
    rt = runtime(thin=True)
    rt.generate_views(photo(), six_views(), PRESETS["final"], seed=1)
    assert rt.last_weights == "multiview" and rt.last_thin is None and rt.single.runs == []


def test_out_of_memory_on_the_single_view_weights_retries_and_swaps_back(real_estimate_pose):
    rt = preview_runtime(tilted_plate(0.0), single_failures=[torch.OutOfMemoryError("CUDA out of memory")])
    rt.generate(photo(), PRESETS["final"], seed=3)
    assert [low for *_, low in rt.single.runs] == [False, True]
    assert rt.single.devices_at_run[1]["sparse_structure_flow_model"] == "cpu"  # low-VRAM: off the GPU between stages
    assert rt.pipeline.low_vram is False and rt.single.low_vram is False  # both back to normal
    assert rt.pipeline.models["sparse_structure_flow_model"].device == "cuda"
    assert rt.single.models["sparse_structure_flow_model"].device == "cpu"
    assert all(getattr(rt.single, name).device == "cuda" for name in pipeline.STAGES)
    assert rt.last_weights == "single"


def test_own_checkpoints_are_the_flow_models(tmp_path):
    decoders = {"sparse_structure_decoder": "/models/TRELLIS.2-4B/ckpts/ss_dec", "shape_slat_decoder": "/models/TRELLIS.2-4B/ckpts/shape_dec"}
    for config, suffix in (("pipeline.json", ""), ("pipeline_mv.json", "_mv")):
        models = {**decoders, "sparse_structure_flow_model": f"ckpts/ss_flow{suffix}", "tex_slat_flow_model_1024": f"ckpts/tex_flow{suffix}"}
        (tmp_path / config).write_text(json.dumps({"args": {"models": models}}))
    own = pipeline.own_checkpoints(str(tmp_path), "pipeline.json", "pipeline_mv.json")
    assert own == {"sparse_structure_flow_model", "tex_slat_flow_model_1024"}


def test_the_thin_ratio_comes_from_the_environment():
    assert pipeline.thin_ratio_from_env(None) == pipeline.THIN_RATIO
    assert pipeline.thin_ratio_from_env("") == pipeline.THIN_RATIO
    assert pipeline.thin_ratio_from_env("none") is None and pipeline.thin_ratio_from_env("off") is None
    assert pipeline.thin_ratio_from_env("0.25") == 0.25
    with pytest.raises(ValueError):
        pipeline.thin_ratio_from_env("thin")


# --- The TRELLIS.2 that makes the recipe's previews -----------------------------------------------------


def test_the_recipe_uses_the_callers_trellis2_when_given(monkeypatch):
    loaded = []

    class OwnTrellis2:
        def __init__(self, model_dir):
            loaded.append(model_dir)
            self.asleep = False

        def sleep(self):
            self.asleep = True

    monkeypatch.setattr(pipeline, "Trellis2Runtime", OwnTrellis2)
    shared = object()
    choose = Pixal3DRuntime._trellis2_for_previews
    # The container's own TRELLIS.2 (production holds both runtimes): used as it is, nothing loaded
    assert choose(pipeline.LEVEL_PREVIEW, shared, "/models/TRELLIS.2-4B") is shared and loaded == []
    # Without one, a copy of its own, asleep: off the GPU between uses, and kept off it after a retry
    own = choose(pipeline.LEVEL_PREVIEW, None, "/models/TRELLIS.2-4B")
    assert isinstance(own, OwnTrellis2) and own.asleep and loaded == ["/models/TRELLIS.2-4B"]
    # No levelling against previews: no TRELLIS.2 at all, given or not
    assert choose(pipeline.LEVEL_NONE, shared, "/x") is None and choose(pipeline.LEVEL_GIVEN, None, "/x") is None
    assert loaded == ["/models/TRELLIS.2-4B"]
