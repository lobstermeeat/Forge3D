"""Trellis2Runtime on a fake pipeline: the retry after running out of GPU memory."""

import dataclasses
import os
import sys
import types
import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from forge3d_worker.pipeline import Trellis2Runtime
from forge3d_worker.settings import FALLBACK_PIPELINE, PRESETS


def failure(kind: str, run: int) -> Exception:
    """What a failed run raises, marked with the run's number."""
    if kind == "oom":
        return torch.cuda.OutOfMemoryError(f"CUDA out of memory. Tried to allocate 742.00 MiB (run {run})")
    if kind == "cumesh-oom":  # CuMesh's own check of a failed cudaMalloc
        return RuntimeError(f"[CuMesh] CUDA error:\n    Error code: 2\n    Error text: out of memory\n(run {run})")
    if kind == "cuda-error":
        return RuntimeError(f"CUDA error: an illegal memory access was encountered (run {run})")
    return ValueError(f"Invalid pipeline type: 2048 (run {run})")


class FakeModel:
    """A model, or upstream's DINOv3 and BiRefNet wrappers, reduced to where its weights are."""

    def __init__(self) -> None:
        self.device = "cpu"  # from_pretrained loads onto the CPU

    def to(self, device) -> None:
        self.device = torch.device(device).type

    def cpu(self) -> None:
        self.to("cpu")


class Activations:
    """Stands in for the tensors a run holds: alive for as long as the run's frame is."""


class ShapeLatent:
    """
    Upstream's SparseTensor as the runtime handles a shape latent: to() makes a moved copy that shares the
    original's spatial cache (upstream's replace()), clear_spatial_cache() gives a tensor a new, empty one.
    """

    def __init__(self, cache=None, device="cuda") -> None:
        self.cache = {"neighbours at 1024": "big"} if cache is None else cache
        self.device = device

    def to(self, device) -> "ShapeLatent":
        return ShapeLatent(self.cache, torch.device(device).type)

    def clear_spatial_cache(self) -> None:
        self.cache = {}


def returned(meshes, options):
    """What upstream's run() returns: the meshes, and with return_latent=True the latents and resolution too."""
    if not options.get("return_latent"):
        return meshes
    return meshes, (ShapeLatent(), "texture latent", 512 if options.get("pipeline_type") == "512" else 1024)


class FakePipeline:
    """Upstream Trellis2ImageTo3DPipeline's device handling, around a scripted run()."""

    def __init__(self, outcomes=("mesh",), low_vram: bool = False) -> None:
        names = ("sparse_structure_flow_model", "shape_slat_flow_model_1024", "tex_slat_decoder")
        self.models = {name: FakeModel() for name in names}
        self.image_cond_model = FakeModel()
        self.rembg_model = FakeModel()
        self._device = "cpu"
        self.outcomes = list(outcomes)  # per run: "mesh", or the kind of failure()
        self.runs = []
        self.activations = []  # a weak reference to each run's
        self.live_at_cuda = []  # how many runs' activations were alive at each cuda()
        # What Trellis2Runtime.__init__ does
        self.low_vram = low_vram
        self.cuda()

    @property
    def device(self):
        return self._device

    def to(self, device) -> None:
        # Upstream's Trellis2ImageTo3DPipeline.to()
        self._device = device
        if not self.low_vram:
            for model in self.models.values():
                model.to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def cuda(self) -> None:
        self.live_at_cuda.append(self.live_runs())
        self.to(torch.device("cuda"))

    def live_runs(self) -> int:
        return sum(ref() is not None for ref in self.activations)

    def weights(self) -> set:
        """The devices holding model weights."""
        return {model.device for model in (*self.models.values(), self.image_cond_model, self.rembg_model)}

    def preprocess_image(self, image):
        return image  # upstream's background removal and crop, which these tests don't look at

    def run(self, image, **options):
        state = {"low_vram": self.low_vram, "weights": self.weights(), "device": torch.device(self.device).type}
        self.runs.append({**options, **state, "image": image, "live_runs": self.live_runs()})
        activations = Activations()
        self.activations.append(weakref.ref(activations))
        outcome = self.outcomes[len(self.runs) - 1]
        if outcome.startswith("stranded-"):
            # Upstream's low-VRAM stage moved its model to the GPU, then failed before moving it back
            self.models["shape_slat_flow_model_1024"].to("cuda")
            outcome = outcome[len("stranded-"):]
        if outcome != "mesh":
            raise failure(outcome, len(self.runs))
        return returned([outcome], options)


def runtime_around(pipeline: FakePipeline) -> Trellis2Runtime:
    """A Trellis2Runtime holding a fake pipeline; __init__ would load the real one onto a GPU."""
    runtime = Trellis2Runtime.__new__(Trellis2Runtime)
    runtime.pipeline = pipeline
    runtime.low_vram_configured = pipeline.low_vram  # as __init__ records TRELLIS2_LOW_VRAM
    return runtime


def cutout() -> Image.Image:
    """A red box, already cut out."""
    pixels = np.zeros((64, 64, 4), np.uint8)
    pixels[16:48, 16:48] = (200, 30, 30, 255)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("kind", ["oom", "cumesh-oom"])
def test_out_of_memory_is_retried_once_in_low_vram_mode_then_restored(kind, monkeypatch, capsys):
    pipeline = FakePipeline(outcomes=[kind, "mesh"])
    emptied = []  # where the weights were each time torch's cache was emptied
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(pipeline.weights()))

    assert runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7) == "mesh"

    first, retry = pipeline.runs
    assert first["low_vram"] is False and first["weights"] == {"cuda"}
    # Upstream's low-VRAM mode: every model off the GPU, each moved onto it only while it runs
    assert retry["low_vram"] is True and retry["weights"] == {"cpu"} and retry["device"] == "cuda"
    assert retry["image"] is first["image"] and retry["seed"] == 7 and retry["pipeline_type"] == "1024_cascade"
    # The failed run's tensors were already let go, and the cache went back to CUDA after the weights left
    assert retry["live_runs"] == 0 and emptied == [{"cpu"}]
    # Then everything is back on the GPU, the way __init__ left it
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"} and pipeline.device.type == "cuda"
    log = capsys.readouterr().out
    assert log.startswith("[forge3d] out of GPU memory in 1024_cascade, retrying in low-VRAM mode: ")
    assert log.count("\n") == 1 and log.endswith("(run 1)\n")


@pytest.mark.parametrize(
    "mode, outcomes",
    [
        ("final", ["oom", "bug"]),  # not running out of memory again: nothing to fall back for
        ("final", ["cumesh-oom", "cuda-error"]),
        ("preview", ["oom", "oom", "mesh"]),  # the preview has nothing cheaper to fall back to
        ("preview", ["cumesh-oom", "cumesh-oom", "mesh"]),
    ],
)
def test_a_failed_retry_is_raised_after_restoring(mode, outcomes, capsys):
    pipeline = FakePipeline(outcomes=outcomes)
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError, ValueError), match=r"\(run 2\)$") as raised:
        runtime_around(pipeline).generate(cutout(), PRESETS[mode], seed=7)

    assert len(pipeline.runs) == 2 and raised.traceback[-1].name == "run"
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"}
    # The retry's tensors were let go before the weights went back onto the GPU
    assert pipeline.live_at_cuda == [0, 0]
    assert capsys.readouterr().out.count("\n") == 1


@pytest.mark.parametrize("first, second", [("oom", "oom"), ("cumesh-oom", "cumesh-oom"), ("oom", "cumesh-oom")])
def test_a_final_still_out_of_memory_in_low_vram_mode_falls_back_to_the_preview_pipeline(first, second, monkeypatch, capsys):
    pipeline = FakePipeline(outcomes=[first, second, "mesh"])
    emptied = []  # where the weights were, and how many runs' tensors were alive, at each emptying
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append((pipeline.weights(), pipeline.live_runs())))
    runtime = runtime_around(pipeline)

    assert runtime.generate(cutout(), PRESETS["final"], seed=7) == "mesh"

    initial, retry, fallback = pipeline.runs
    assert [run["pipeline_type"] for run in pipeline.runs] == ["1024_cascade", "1024_cascade", "512"]
    # Still in low-VRAM mode, on the same picture with the same seed: the shape the preview showed
    assert fallback["low_vram"] is True and fallback["weights"] == {"cpu"} and fallback["device"] == "cuda"
    assert fallback["image"] is initial["image"] and fallback["seed"] == 7 and fallback["preprocess_image"] is False
    # Neither failed run's tensors were alive when it started, and the cache went back to CUDA after the
    # retry's were freed
    assert fallback["live_runs"] == 0 and emptied == [({"cpu"}, 0), ({"cpu"}, 0)]
    assert runtime.pipeline_used == "512"
    # Then everything is back on the GPU, the way __init__ left it
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"} and pipeline.live_at_cuda == [0, 0]
    log = capsys.readouterr().out.splitlines()
    assert len(log) == 2 and log[0].startswith("[forge3d] out of GPU memory in 1024_cascade, retrying in low-VRAM mode: ")
    assert log[1].startswith("[forge3d] out of GPU memory in 1024_cascade even in low-VRAM mode, falling back to 512: ")
    assert log[1].endswith("(run 2)")


@pytest.mark.parametrize("kind", ["oom", "bug"])
def test_a_failed_fallback_is_raised_after_restoring(kind, capsys):
    pipeline = FakePipeline(outcomes=["oom", "oom", kind])
    runtime = runtime_around(pipeline)
    with pytest.raises((torch.cuda.OutOfMemoryError, ValueError), match=r"\(run 3\)$") as raised:
        runtime.generate(cutout(), PRESETS["final"], seed=7)

    assert len(pipeline.runs) == 3 and raised.traceback[-1].name == "run"
    assert pipeline.runs[2]["pipeline_type"] == "512"
    # Its tensors were let go before the weights went back onto the GPU, and nothing tries a fourth time
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"} and pipeline.live_at_cuda == [0, 0]
    assert capsys.readouterr().out.count("\n") == 2


@pytest.mark.parametrize("outcomes, low_vram", [(["mesh"], [False]), (["oom", "mesh"], [False, True])])
def test_a_shape_made_again_for_a_final_that_fell_back_runs_its_fallback_pipeline_at_once(outcomes, low_vram):
    """A textures job for a final that fell back ("pipeline": "512") gets the final's preset with that pipeline."""
    pipeline = FakePipeline(outcomes=outcomes)
    runtime = runtime_around(pipeline)
    final = PRESETS["final"]
    fell_back = dataclasses.replace(final, pipeline_type=FALLBACK_PIPELINE[final.pipeline_type])

    assert runtime.generate(cutout(), fell_back, seed=7) == "mesh"

    # The run the final's fallback made, on the same picture with the same seed, and never the cascade first.
    # Running out of memory gets the one retry in low-VRAM mode any job gets, and nothing cheaper after it
    assert [run["pipeline_type"] for run in pipeline.runs] == ["512"] * len(outcomes)
    assert [run["low_vram"] for run in pipeline.runs] == low_vram and pipeline.runs[-1]["seed"] == 7
    assert runtime.pipeline_used == "512"
    # Its shape latent is the 512 one, so retexture() samples on it with the 512 texture flow, as run() did
    assert runtime.last_latent.pipeline_type == "512" and runtime.last_latent.resolution == 512
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"}


def test_each_job_reports_the_pipeline_that_made_its_mesh():
    pipeline = FakePipeline(outcomes=["oom", "oom", "mesh", "mesh", "mesh"])
    runtime = runtime_around(pipeline)
    runtime.generate(cutout(), PRESETS["final"], seed=7)
    assert runtime.pipeline_used == "512"
    # A fallback doesn't stick to the worker: the next jobs run, and report, their own pipelines
    runtime.generate(cutout(), PRESETS["final"], seed=8)
    assert runtime.pipeline_used == "1024_cascade" and pipeline.runs[-1]["pipeline_type"] == "1024_cascade"
    runtime.generate(cutout(), PRESETS["preview"], seed=9)
    assert runtime.pipeline_used == "512"


@pytest.mark.parametrize("kind", ["cuda-error", "bug"])
def test_other_errors_are_not_retried(kind, capsys):
    pipeline = FakePipeline(outcomes=[kind, "mesh"])
    with pytest.raises((RuntimeError, ValueError), match=r"\(run 1\)$"):
        runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7)

    assert len(pipeline.runs) == 1
    assert pipeline.live_at_cuda == [0] and pipeline.weights() == {"cuda"}  # nothing moved
    assert capsys.readouterr().out == ""


def test_a_preview_out_of_memory_in_low_vram_mode_is_not_retried(capsys):
    pipeline = FakePipeline(outcomes=["oom", "mesh"], low_vram=True)
    with pytest.raises(torch.cuda.OutOfMemoryError, match=r"\(run 1\)$"):
        runtime_around(pipeline).generate(cutout(), PRESETS["preview"], seed=7)

    # Nothing cheaper to fall back to, and no retry: the models are already off the GPU
    assert len(pipeline.runs) == 1 and pipeline.low_vram is True and pipeline.weights() == {"cpu"}
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kind", ["oom", "cumesh-oom"])
@pytest.mark.parametrize("why", ["deployed in low-VRAM mode", "asleep"])
def test_a_final_out_of_memory_in_low_vram_mode_goes_straight_to_the_preview_pipeline(kind, why, capsys):
    """An A10 deployment (TRELLIS2_LOW_VRAM=1), or TRELLIS.2 asleep beside Pixal3D in production's container."""
    if why == "asleep":
        pipeline = FakePipeline(outcomes=[kind, "mesh"])
        runtime = runtime_around(pipeline)
        runtime.sleep()
    else:
        pipeline = FakePipeline(outcomes=[kind, "mesh"], low_vram=True)
        runtime = runtime_around(pipeline)

    assert runtime.generate(cutout(), PRESETS["final"], seed=7) == "mesh"

    assert [run["pipeline_type"] for run in pipeline.runs] == ["1024_cascade", "512"]
    assert pipeline.runs[1]["low_vram"] is True and pipeline.runs[1]["live_runs"] == 0
    assert runtime.pipeline_used == "512"
    # The models stay off the GPU afterwards: an A10 keeps its low-VRAM mode, a sleeping runtime sleeps on
    assert pipeline.low_vram is True and pipeline.weights() == {"cpu"} and pipeline.live_at_cuda == [0]
    log = capsys.readouterr().out.splitlines()
    assert len(log) == 1
    assert log[0].startswith("[forge3d] out of GPU memory in 1024_cascade with the models already off the GPU, falling back to 512: ")


def test_a_final_out_of_memory_in_low_vram_mode_that_fails_again_is_raised(capsys):
    pipeline = FakePipeline(outcomes=["oom", "oom"], low_vram=True)
    with pytest.raises(torch.cuda.OutOfMemoryError, match=r"\(run 2\)$"):
        runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7)
    assert [run["pipeline_type"] for run in pipeline.runs] == ["1024_cascade", "512"]
    assert capsys.readouterr().out.count("\n") == 1


def test_sleep_puts_the_models_off_the_gpu_for_good_and_wake_brings_them_back():
    pipeline = FakePipeline(outcomes=["oom", "mesh", "mesh"])
    runtime = runtime_around(pipeline)
    runtime.sleep()
    assert runtime.asleep and pipeline.low_vram is True and pipeline.weights() == {"cpu"}
    # A retry's restore leaves a sleeping runtime where it is
    runtime._restore()
    assert pipeline.low_vram is True and pipeline.weights() == {"cpu"}
    # Asleep, it still works, in low-VRAM mode (an out-of-memory final falls back to 512 directly)
    assert runtime.generate(cutout(), PRESETS["final"], seed=7) == "mesh"
    assert [run["low_vram"] for run in pipeline.runs] == [True, True] and runtime.pipeline_used == "512"
    runtime.wake()
    assert not runtime.asleep and pipeline.low_vram is False and pipeline.weights() == {"cuda"}
    assert runtime.generate(cutout(), PRESETS["final"], seed=7) == "mesh"
    assert pipeline.runs[-1]["low_vram"] is False


@pytest.mark.parametrize("why", ["deployed in low-VRAM mode", "asleep"])
def test_a_model_a_failed_stage_left_on_the_gpu_goes_back_to_the_cpu(why, capsys):
    """Upstream's low-VRAM stages move their model back without a finally; the restore does it instead."""
    pipeline = FakePipeline(outcomes=["stranded-oom", "mesh"], low_vram=why != "asleep")
    runtime = runtime_around(pipeline)
    if why == "asleep":
        runtime.sleep()

    assert runtime.generate(cutout(), PRESETS["final"], seed=7) == "mesh"
    assert runtime.pipeline_used == "512"
    assert pipeline.low_vram is True and pipeline.weights() == {"cpu"}


def test_freeing_gpu_memory_clears_a_stale_cuda_error(monkeypatch, capsys):
    """CuMesh never resets CUDA's error flag; the next run's first kernel would report it."""
    import forge3d_worker.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "clear_cuda_error", lambda: "CUDA error: out of memory")
    pipeline = FakePipeline(outcomes=["cumesh-oom", "mesh"], low_vram=True)
    assert runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7) == "mesh"
    assert "[forge3d] cleared a stale CUDA error before going on: CUDA error: out of memory" in capsys.readouterr().out
    monkeypatch.undo()
    assert pipeline_module.clear_cuda_error() is None  # without CUDA (these tests) there is nothing to clear


@pytest.mark.parametrize("kind", ["oom", "cumesh-oom"])
def test_running_out_of_memory_while_exporting_is_retried_with_the_models_off_the_gpu(kind, capsys):
    pipeline = FakePipeline()
    runtime = runtime_around(pipeline)
    attempts = []

    def export(mesh, preset):
        attempts.append({"low_vram": pipeline.low_vram, "weights": pipeline.weights()})
        if len(attempts) == 1:
            raise failure(kind, 1)
        return b"glb", 12

    runtime._export = export
    assert runtime.export("mesh", PRESETS["final"]) == (b"glb", 12)
    assert attempts == [
        {"low_vram": False, "weights": {"cuda"}},
        {"low_vram": True, "weights": {"cpu"}},
    ]
    # Back the way __init__ left it
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"}
    assert "retrying with the models off the GPU" in capsys.readouterr().out


def test_other_export_errors_are_not_retried():
    runtime = runtime_around(FakePipeline())
    calls = []

    def export(mesh, preset):
        calls.append(1)
        raise failure("cuda-error", 1)

    runtime._export = export
    with pytest.raises(RuntimeError, match="illegal memory access"):
        runtime.export("mesh", PRESETS["final"])
    assert len(calls) == 1


# --- The picture travels from generate() to export(), where it is painted on (final only) -----------


class Remover(FakeModel):
    """Upstream's BiRefNet wrapper reduced to what it returns: the picture with an alpha channel."""

    def __call__(self, image):
        image.putalpha(Image.new("L", image.size, 255))
        return image


class Mesh:
    """Upstream's MeshWithVoxel takes attributes; a str (FakePipeline's mesh) doesn't."""


class CroppingPipeline(FakePipeline):
    """preprocess_image as upstream does it: the remover runs (unless the picture has alpha), then a crop."""

    def preprocess_image(self, image):
        if image.mode != "RGBA":
            if self.low_vram:
                self.rembg_model.to("cuda")
            image = self.rembg_model(image.convert("RGB"))
        return image.crop((8, 8, 40, 40)).convert("RGB")

    def run(self, image, **options):
        super().run(image, **options)
        return returned([Mesh()], options)


def test_generate_keeps_the_full_frame_cutout_on_the_mesh():
    pipeline = CroppingPipeline(low_vram=True)
    remover = pipeline.rembg_model = Remover()
    picture = Image.new("RGB", (64, 48), (200, 30, 30))
    mesh = runtime_around(pipeline).generate(picture, PRESETS["final"], seed=7)

    cutout = mesh.forge3d_cutout
    assert cutout.mode == "RGBA" and cutout.size == (64, 48)  # before the crop
    assert pipeline.runs[0]["image"].size == (32, 32)  # the model still got the crop
    assert pipeline.rembg_model is remover and remover.device == "cuda"  # put back; to() went through


def test_a_picture_with_its_own_alpha_is_its_own_cutout():
    pipeline = CroppingPipeline()
    picture = cutout()  # RGBA, transparent around a red box: upstream skips background removal
    mesh = runtime_around(pipeline).generate(picture, PRESETS["final"], seed=7)
    assert mesh.forge3d_cutout is picture


def test_meshes_that_take_no_attributes_still_generate():
    pipeline = CroppingPipeline()
    pipeline.rembg_model = Remover()
    pipeline.run = lambda image, **options: returned(["mesh"], options)
    assert runtime_around(pipeline).generate(Image.new("RGB", (64, 64)), PRESETS["final"], seed=7) == "mesh"


def textured_glb():
    import trimesh

    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGBA", (8, 8), (100, 100, 100, 128)))
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices), 2)), material=material)
    return box


@pytest.mark.parametrize("mode, projects", [("final", True), ("preview", False)])
def test_export_paints_the_picture_on_finals(mode, projects, monkeypatch, capsys):
    import types

    from forge3d_worker import pipeline as pipeline_module

    calls = []

    report = {"applied": True, "reason": "applied", "iou": 0.97, "pose": {"azimuth": 10.0}, "timings": {"total_s": 1.5}}

    def project_picture(glb, picture):
        calls.append((glb, picture, glb.visual.material.baseColorTexture.mode))
        return glb, report

    monkeypatch.setattr(pipeline_module.projection, "project_picture", project_picture)
    glb = textured_glb()
    runtime = runtime_around(FakePipeline())
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: glb))
    mesh = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=None)
    mesh.forge3d_cutout = cutout()

    data, triangles = runtime.export(mesh, PRESETS[mode])
    assert data[:4] == b"glTF" and triangles == 12
    if projects:
        # After unpremultiplying (RGB by then), with the picture generate() kept
        assert calls == [(glb, mesh.forge3d_cutout, "RGB")]
        summary = {"applied": True, "reason": "applied", "iou": 0.97, "pose": {"azimuth": 10.0}, "seconds": 1.5}
        assert runtime.last_projection == summary
        assert "\n[forge3d] projection: {" in "\n" + capsys.readouterr().out  # after the glass line
    else:
        assert calls == [] and runtime.last_projection is None


def test_an_experiment_can_change_the_texture_before_the_projection(monkeypatch):
    import types

    from forge3d_worker import pipeline as pipeline_module

    steps = []

    def project_picture(glb, picture):
        steps.append(("projection", glb.visual.material.baseColorTexture.getpixel((0, 0))))
        return glb, {"applied": True, "reason": "applied"}

    def bake(glb, mesh):
        # After unpremultiplying (RGB by then); handed the generated mesh, which carries the cutout
        steps.append(("bake", glb.visual.material.baseColorTexture.mode, mesh.forge3d_cutout.size))
        glb.visual.material.baseColorTexture = Image.new("RGB", (8, 8), (1, 2, 3))
        return {"seconds": 0.1}

    monkeypatch.setattr(pipeline_module.projection, "project_picture", project_picture)
    glb = textured_glb()
    runtime = runtime_around(FakePipeline())
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: glb))
    runtime.before_projection = bake
    mesh = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=None)
    mesh.forge3d_cutout = cutout()
    runtime.export(mesh, PRESETS["final"])
    assert steps == [("bake", "RGB", (64, 64)), ("projection", (1, 2, 3))]  # the projection painted on the bake
    assert runtime.last_before_projection == {"seconds": 0.1}


def test_production_changes_nothing_before_the_projection():
    assert Trellis2Runtime.before_projection is None


def test_export_without_a_picture_says_so():
    import types

    glb = textured_glb()
    runtime = runtime_around(FakePipeline())
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: glb))
    mesh = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=None)
    runtime.export(mesh, PRESETS["final"])
    assert runtime.last_projection == {"applied": False, "reason": "no picture came with the mesh"}


def test_a_final_made_by_the_fallback_still_carries_the_picture_to_paint_on():
    class MeshPipeline(FakePipeline):
        def run(self, image, **options):
            super().run(image, **options)  # raises for the scripted failures
            return returned([types.SimpleNamespace(pipeline_type=options["pipeline_type"])], options)

    pipeline = MeshPipeline(outcomes=["oom", "oom", "mesh"])
    picture = cutout()  # has its own alpha, so it is its own cutout
    mesh = runtime_around(pipeline).generate(picture, PRESETS["final"], seed=7)
    assert mesh.pipeline_type == "512" and mesh.forge3d_cutout is picture


# --- retexture(): where the models are, running out of memory, what is kept -------------------------------


class TexturePipeline(FakePipeline):
    """
    FakePipeline with upstream's texture stages reduced to their device handling: in low-VRAM mode each
    moves its model onto the GPU and back without a ``finally`` (as upstream), and the texture stage
    fails as scripted (``texture``: per attempt, "texture" or a failure()'s kind, "stranded-" first to
    fail with its model left on the GPU).
    """

    tex_slat_sampler_params = {"steps": 12, "guidance_strength": 1.0}

    def __init__(self, outcomes=("mesh",), low_vram: bool = False, texture=("texture",)) -> None:
        super().__init__(outcomes, low_vram)
        for name in ("tex_slat_flow_model_512", "tex_slat_flow_model_1024", "shape_slat_decoder"):
            self.models[name] = FakeModel()
        self.to(self._device)  # where __init__'s cuda() put the others
        self.texture = list(texture)
        self.attempts = []  # per texture attempt: the mode, where the weights were, the latent's device

    def get_cond(self, image, resolution, include_neg_cond=True):
        return {"cond": torch.ones(1, 2) * resolution, "neg_cond": torch.zeros(1, 2)}

    def sample_tex_slat(self, cond, flow_model, shape_slat, sampler_params={}):  # noqa: B006 - upstream's
        state = {"low_vram": self.low_vram, "weights": self.weights(), "latent": shape_slat.device}
        self.attempts.append({**state, "flow": flow_model, "params": sampler_params, "live_runs": self.live_runs()})
        activations = Activations()
        self.activations.append(weakref.ref(activations))
        outcome = self.texture[len(self.attempts) - 1]
        if self.low_vram:
            flow_model.to("cuda")
        if outcome != "texture":
            raise failure(outcome.replace("stranded-", ""), len(self.attempts))
        if self.low_vram:
            flow_model.cpu()
        return "texture latent"

    def decode_latent(self, shape_slat, tex_slat, resolution):
        return [types.SimpleNamespace(shape=shape_slat, tex=tex_slat, resolution=resolution)]


def retexturing(deployment: str, **options):
    pipeline = TexturePipeline(low_vram=deployment == "deployed in low-VRAM mode", **options)
    runtime = runtime_around(pipeline)
    if deployment == "asleep":
        runtime.sleep()
    runtime.generate(cutout(), PRESETS["final"], seed=7)
    return pipeline, runtime


@pytest.mark.parametrize("deployment", ["on the GPU", "deployed in low-VRAM mode", "asleep"])
def test_retexture_runs_wherever_the_models_live_and_leaves_them_there(deployment, capsys):
    pipeline, runtime = retexturing(deployment)
    mesh = runtime.retexture()

    (attempt,) = pipeline.attempts
    off = deployment != "on the GPU"
    # Upstream's own mode: on the GPU all along, or each model visiting it for its stage
    assert attempt["low_vram"] is off and attempt["weights"] == ({"cpu"} if off else {"cuda"})
    # The 1024 texture flow, the defaults, and the shape latent on the pipeline's device for it
    assert attempt["flow"] is pipeline.models["tex_slat_flow_model_1024"] and attempt["params"] == {"steps": 12, "guidance_strength": 1.0}
    assert attempt["latent"] == "cuda" and mesh.shape.device == "cuda" and mesh.resolution == 1024
    assert runtime.last_latent.shape_slat.device == "cpu"  # the kept one stays where it was
    assert pipeline.low_vram is off and pipeline.weights() == ({"cpu"} if off else {"cuda"})
    assert runtime.asleep is (deployment == "asleep")


@pytest.mark.parametrize("kind", ["oom", "cumesh-oom"])
def test_retexture_out_of_memory_is_retried_once_in_low_vram_mode_then_restored(kind, monkeypatch, capsys):
    pipeline, runtime = retexturing("on the GPU", texture=[kind, "texture"])
    emptied = []  # where the weights were each time torch's cache was emptied
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(pipeline.weights()))
    capsys.readouterr()

    assert runtime.retexture(sampler_params={"steps": 20}).tex == "texture latent"

    first, retry = pipeline.attempts
    assert first["low_vram"] is False and first["weights"] == {"cuda"}
    assert retry["low_vram"] is True and retry["weights"] == {"cpu"} and retry["params"]["steps"] == 20
    # The failed attempt's tensors were let go before the retry, and the cache went back to CUDA
    assert retry["live_runs"] == 0 and emptied[0] == {"cpu"}
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"}
    log = capsys.readouterr().out.splitlines()
    assert len(log) == 2 and log[0].startswith("[forge3d] retexturing the 1024_cascade shape: ")
    assert log[1].startswith("[forge3d] out of GPU memory retexturing, retrying in low-VRAM mode: ")
    assert log[1].endswith("(run 1)")


@pytest.mark.parametrize(
    "deployment, texture",
    [
        ("on the GPU", ["bug"]),  # not out of memory: not retried
        ("on the GPU", ["oom", "oom"]),  # out of memory again in low-VRAM mode
        ("deployed in low-VRAM mode", ["stranded-oom"]),  # nothing cheaper; its model left on the GPU
        ("asleep", ["stranded-cumesh-oom"]),
    ],
)
def test_a_failed_retexture_is_raised_with_the_models_put_back(deployment, texture, capsys):
    pipeline, runtime = retexturing(deployment, texture=texture)
    latent = runtime.last_latent
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError, ValueError), match=rf"\(run {len(texture)}\)$"):
        runtime.retexture()

    assert len(pipeline.attempts) == len(texture)
    off = deployment != "on the GPU"
    assert pipeline.low_vram is off and pipeline.weights() == ({"cpu"} if off else {"cuda"})
    # The failed attempts' tensors were let go before the weights went back
    assert pipeline.live_runs() == 0
    # The shape is still there to try again
    assert runtime.last_latent is latent


def test_retexture_needs_a_shape():
    runtime = runtime_around(TexturePipeline(outcomes=["bug", "mesh"]))
    with pytest.raises(RuntimeError, match="no shape to retexture"):
        runtime.retexture()
    # A failed generation leaves none either, even after a successful one
    with pytest.raises(ValueError):
        runtime.generate(cutout(), PRESETS["final"], seed=7)
    with pytest.raises(RuntimeError, match="no shape to retexture"):
        runtime.retexture()
    runtime.generate(cutout(), PRESETS["final"], seed=8)
    assert runtime.last_latent is not None and runtime.last_latent.seed == 8


def test_generate_keeps_its_shape_latent_small():
    """On the CPU, without the caches it shared with the run's tensors (upstream's flows fill them on the GPU)."""
    pipeline = TexturePipeline()
    runtime = runtime_around(pipeline)
    returns = []
    stock = pipeline.run

    def run(image, **options):
        result = stock(image, **options)
        returns.append(result)
        return result

    pipeline.run = run
    picture = cutout()
    mesh = runtime.generate(picture, PRESETS["final"], seed=7)

    assert pipeline.runs[0]["return_latent"] is True
    meshes, (shape_slat, _, resolution) = returns[0]
    assert mesh is meshes[0]  # what generate() returns is unchanged
    latent = runtime.last_latent
    assert latent.shape_slat.device == "cpu" and latent.shape_slat.cache == {}
    assert shape_slat.cache == {"neighbours at 1024": "big"}  # the run's own cache is left alone
    assert (latent.resolution, latent.pipeline_type, latent.seed) == (1024, "1024_cascade", 7)
    assert latent.image is pipeline.runs[0]["image"] and latent.cutout is picture


def test_a_fallback_keeps_the_fallbacks_shape_and_its_512_texture_flow():
    pipeline = TexturePipeline(outcomes=["oom", "oom", "mesh"])
    runtime = runtime_around(pipeline)
    runtime.generate(cutout(), PRESETS["final"], seed=7)
    assert (runtime.last_latent.pipeline_type, runtime.last_latent.resolution) == ("512", 512)
    mesh = runtime.retexture()
    assert pipeline.attempts[-1]["flow"] is pipeline.models["tex_slat_flow_model_512"] and mesh.resolution == 512


def upstream_sparse_tensor():
    source = os.environ.get("TRELLIS2_SRC")
    if source and source not in sys.path:
        sys.path.append(source)
    try:
        from trellis2.modules.sparse import SparseTensor
    except ImportError:
        return None
    return SparseTensor


@pytest.mark.skipif(upstream_sparse_tensor() is None, reason="TRELLIS2_SRC: TRELLIS.2's source")
def test_the_kept_latent_drops_upstreams_shared_spatial_cache():
    from forge3d_worker.pipeline import _detached

    SparseTensor = upstream_sparse_tensor()
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int32)
    run = SparseTensor(feats=torch.randn(3, 8), coords=coords)
    latent = run * 2.0 + 1.0  # derived, as upstream's de-normalised latent is: it shares the run's cache
    latent.register_spatial_cache("neighbours", torch.zeros(1000))
    assert latent._spatial_cache is run._spatial_cache

    kept = _detached(latent, "cpu")
    assert kept._spatial_cache == {} and "neighbours" in str(run._spatial_cache)
    assert torch.equal(kept.feats, latent.feats) and torch.equal(kept.coords, latent.coords)
    assert kept.shape == latent.shape and kept._scale == latent._scale
    # Moved for a retexture, it gets its own cache again: what the texture flow keeps never lands on the kept one
    moved = _detached(kept, "cpu")
    moved.register_spatial_cache("texture flow", torch.zeros(10))
    assert kept._spatial_cache == {}


# --- The floater cleanup in export(), when the preset asks for it ------------------------------------------


def floater_glb():
    """A textured box with a small cube floating beside it, as to_glb would return it (premultiplied texture)."""
    import trimesh

    body = trimesh.creation.box()
    speck = trimesh.creation.box(extents=(0.05, 0.05, 0.05)).apply_translation((1.0, 0, 0))
    vertices = np.concatenate([body.vertices, speck.vertices])
    faces = np.concatenate([body.faces, speck.faces + len(body.vertices)])
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGBA", (8, 8), (100, 100, 100, 128)))
    visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(vertices), 2)), material=material)
    return trimesh.Trimesh(vertices, faces, visual=visual, process=False)


def exporting(glb):
    runtime = runtime_around(FakePipeline())
    runtime._o_voxel = types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=lambda **kwargs: glb))
    mesh = types.SimpleNamespace(vertices=None, faces=None, attrs=None, coords=None, layout=None, voxel_size=None)
    mesh.forge3d_cutout = cutout()
    return runtime, mesh


def test_export_drops_floaters_after_unpremultiplying_and_before_the_projection(monkeypatch, capsys):
    import dataclasses

    from forge3d_worker import pipeline as pipeline_module

    order = []
    real = pipeline_module.cleanup.drop_floaters

    def drop_floaters(glb):
        order.append(("cleanup", len(glb.faces), glb.visual.material.baseColorTexture.mode))
        return real(glb)

    def project_picture(glb, picture):
        order.append(("projection", len(glb.faces)))
        return glb, {"applied": False, "reason": "test"}

    monkeypatch.setattr(pipeline_module.cleanup, "drop_floaters", drop_floaters)
    monkeypatch.setattr(pipeline_module.projection, "project_picture", project_picture)
    runtime, mesh = exporting(floater_glb())
    preset = dataclasses.replace(PRESETS["final"], drop_floaters=True)

    _, triangles = runtime.export(mesh, preset)

    assert order == [("cleanup", 24, "RGB"), ("projection", 12)] and triangles == 12
    assert runtime.last_cleanup["pieces"] == 2 and runtime.last_cleanup["dropped"] == 1
    assert runtime.last_cleanup["floaters"][0]["faces"] == 12
    assert "[forge3d] floaters: {" in capsys.readouterr().out


def test_production_presets_keep_every_piece(monkeypatch):
    from forge3d_worker import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.projection, "project_picture", lambda glb, picture: (glb, {"applied": False}))
    assert not PRESETS["final"].drop_floaters and not PRESETS["preview"].drop_floaters
    runtime, mesh = exporting(floater_glb())
    _, triangles = runtime.export(mesh, PRESETS["final"])
    assert triangles == 24 and runtime.last_cleanup is None


def test_a_failed_cleanup_leaves_the_model_whole(monkeypatch, capsys):
    import dataclasses

    from forge3d_worker import pipeline as pipeline_module

    def broken(glb):
        raise IndexError("index 7 is out of bounds\nfor axis 0")

    monkeypatch.setattr(pipeline_module.cleanup, "drop_floaters", broken)
    runtime, mesh = exporting(floater_glb())
    preset = dataclasses.replace(PRESETS["preview"], drop_floaters=True)
    _, triangles = runtime.export(mesh, preset)
    assert triangles == 24 and runtime.last_cleanup == {"error": "IndexError: index 7 is out of bounds for axis 0"}
    assert '[forge3d] floaters: {"error": ' in capsys.readouterr().out
