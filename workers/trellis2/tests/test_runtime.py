"""Trellis2Runtime on a fake pipeline: the retry after running out of GPU memory."""

import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from forge3d_worker.pipeline import Trellis2Runtime
from forge3d_worker.settings import PRESETS


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
        if outcome != "mesh":
            raise failure(outcome, len(self.runs))
        return [outcome]


def runtime_around(pipeline: FakePipeline) -> Trellis2Runtime:
    """A Trellis2Runtime holding a fake pipeline; __init__ would load the real one onto a GPU."""
    runtime = Trellis2Runtime.__new__(Trellis2Runtime)
    runtime.pipeline = pipeline
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


@pytest.mark.parametrize("kind", ["oom", "bug"])
def test_a_failed_retry_is_raised_after_restoring(kind, capsys):
    pipeline = FakePipeline(outcomes=["oom", kind])
    with pytest.raises((torch.cuda.OutOfMemoryError, ValueError), match=r"\(run 2\)$") as raised:
        runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7)

    assert len(pipeline.runs) == 2 and raised.traceback[-1].name == "run"
    assert pipeline.low_vram is False and pipeline.weights() == {"cuda"}
    # The retry's tensors were let go before the weights went back onto the GPU
    assert pipeline.live_at_cuda == [0, 0]
    assert capsys.readouterr().out.count("\n") == 1


@pytest.mark.parametrize("kind", ["cuda-error", "bug"])
def test_other_errors_are_not_retried(kind, capsys):
    pipeline = FakePipeline(outcomes=[kind, "mesh"])
    with pytest.raises((RuntimeError, ValueError), match=r"\(run 1\)$"):
        runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7)

    assert len(pipeline.runs) == 1
    assert pipeline.live_at_cuda == [0] and pipeline.weights() == {"cuda"}  # nothing moved
    assert capsys.readouterr().out == ""


def test_out_of_memory_in_low_vram_mode_is_not_retried(capsys):
    pipeline = FakePipeline(outcomes=["oom", "mesh"], low_vram=True)
    with pytest.raises(torch.cuda.OutOfMemoryError, match=r"\(run 1\)$"):
        runtime_around(pipeline).generate(cutout(), PRESETS["final"], seed=7)

    assert len(pipeline.runs) == 1 and pipeline.low_vram is True and pipeline.weights() == {"cpu"}
    assert capsys.readouterr().out == ""
