"""Pixal3DRuntime on a fake pipeline: which pictures and cameras Pixal3D gets, and the GLB's frame."""

import io
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
