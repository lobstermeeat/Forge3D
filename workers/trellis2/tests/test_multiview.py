"""
Several pictures per sample (multiview.py) on CPU: upstream's samplers, or a faithful reduction of them,
driven by mock flow models, and a fake pipeline whose run() calls them the way upstream's does.
"""

import dataclasses
import os
import sys
import types

import numpy as np
import pytest
import torch

from forge3d_worker import multiview
from forge3d_worker.inputs import View
from forge3d_worker.pipeline import Trellis2Runtime
from forge3d_worker.settings import PRESETS, MultiView

# --- Upstream's samplers ------------------------------------------------------------------------------
# trellis2/pipelines/samplers reduced to their call structure, kept to upstream's arithmetic: sample() ->
# sample_once() -> _inference_model(), where the guidance-interval and CFG mixins wrap the Euler
# sampler's call of the model. With TRELLIS.2's source on the path (TRELLIS2_SRC), the real ones run too.


class FlowEuler:
    def __init__(self, sigma_min: float) -> None:
        self.sigma_min = sigma_min

    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        return types.SimpleNamespace(pred_x_prev=x_t - (t - t_prev) * pred_v)

    def sample(self, model, noise, cond=None, steps=50, rescale_t=1.0, verbose=True, tqdm_desc="", **kwargs):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
        for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
            sample = self.sample_once(model, sample, t, t_prev, cond, **kwargs).pred_x_prev
        return types.SimpleNamespace(samples=sample)


class Guidance:
    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        if guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
        pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred)
            std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
            std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
            x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
            x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
            pred = self._xstart_to_pred(x_t, t, x_0)
        return pred


class Interval:
    def _inference_model(self, model, x_t, t, cond, guidance_strength, guidance_interval, **kwargs):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return super()._inference_model(model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs)
        return super()._inference_model(model, x_t, t, cond, guidance_strength=1, **kwargs)


class FlowEulerGuidanceInterval(Interval, Guidance, FlowEuler):
    def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0, guidance_strength=3.0,
               guidance_interval=(0.0, 1.0), verbose=True, **kwargs):  # fmt: skip
        guidance = dict(neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval)
        return super().sample(model, noise, cond, steps, rescale_t, verbose, **guidance, **kwargs)


def upstream_sampler_class():
    source = os.environ.get("TRELLIS2_SRC")
    if source and source not in sys.path:
        sys.path.append(source)
    try:
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    except ImportError:
        return None
    return FlowEulerGuidanceIntervalSampler


SAMPLER_CLASSES = [
    pytest.param(FlowEulerGuidanceInterval, id="reduced"),
    pytest.param(
        upstream_sampler_class(),
        id="upstream",
        marks=pytest.mark.skipif(upstream_sampler_class() is None, reason="TRELLIS2_SRC: TRELLIS.2's source"),
    ),
]

# TRELLIS.2-4B's pipeline.json: the sparse structure and shape flows use guidance with rescale inside an
# interval, the texture flow none (strength 1) and takes the shape latent as concat_cond
STRUCTURE = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=[0.6, 1.0], rescale_t=5.0)
TEXTURE = dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=[0.6, 0.9], rescale_t=3.0)


class Model:
    """A flow model's interface: velocity from (x_t, t, cond) and maybe concat_cond. Logs each call's cond."""

    def __init__(self) -> None:
        self.calls = []  # (t, the cond's first value) per call; 0.0 is the unconditional pass

    def __call__(self, x, t, cond, concat_cond=None):
        # Cross-attention batches x with cond, so a cond of several pictures can't reach a model at once
        assert cond.shape[0] == x.shape[0], "one condition per sample"
        self.calls.append((float(t[0]), float(cond.flatten()[0])))
        extra = 0 if concat_cond is None else concat_cond.mean()
        return torch.tanh(0.3 * x + cond.mean() * (1 + x.square()) + 1e-3 * t.view(-1, 1, 1)) + extra


def picture(value: float) -> torch.Tensor:
    """A picture's DINOv3 tokens (1, tokens, channels), distinguishable by value."""
    return torch.full((1, 5, 3), value) + torch.linspace(0, 0.01, 15).view(1, 5, 3)


def sample(sampler, model, cond, params, **extra):
    torch.manual_seed(0)
    noise = torch.randn(1, 5, 3)
    unconditional = torch.zeros_like(cond[:1])
    return sampler.sample(model, noise, cond=cond, neg_cond=unconditional, verbose=False, **params, **extra).samples


@pytest.mark.parametrize("cls", SAMPLER_CLASSES)
@pytest.mark.parametrize("mode", multiview.MODES)
@pytest.mark.parametrize("params, extra", [(STRUCTURE, {}), (TEXTURE, {"concat_cond": torch.ones(1, 5, 2)})])
def test_one_picture_samples_exactly_as_upstream(cls, mode, params, extra):
    stock = cls(sigma_min=1e-5)
    ref = sample(stock, Model(), picture(0.2), params, **extra)
    out = sample(multiview.multi_view_sampler(stock, [3.0], mode), Model(), picture(0.2), params, **extra)
    assert torch.equal(out, ref)  # bit for bit


@pytest.mark.parametrize("cls", SAMPLER_CLASSES)
def test_copies_of_one_picture_blend_back_to_it(cls):
    stock = cls(sigma_min=1e-5)
    ref = sample(stock, Model(), picture(0.2), STRUCTURE)
    copies = torch.cat([picture(0.2)] * 3)
    blend = multiview.multi_view_sampler(stock, [2.0, 1.0, 1.0], "multidiffusion")
    assert torch.allclose(sample(blend, Model(), copies, STRUCTURE), ref, atol=1e-6)
    turns = multiview.multi_view_sampler(stock, [2.0, 1.0, 1.0], "stochastic")
    assert torch.equal(sample(turns, Model(), copies, STRUCTURE), ref)


def test_the_sampler_keeps_its_class_and_settings_with_the_pictures_just_above_the_model_call():
    stock = FlowEulerGuidanceInterval(sigma_min=1e-5)
    sampler = multiview.multi_view_sampler(stock, [2, 1], "stochastic")
    assert isinstance(sampler, FlowEulerGuidanceInterval) and sampler.sigma_min == 1e-5
    names = [cls.__name__ for cls in type(sampler).__mro__]
    assert names.index("Guidance") < names.index("MultiViewMixin") < names.index("FlowEuler")
    assert sampler.mv_weights == (2.0, 1.0) and stock.__class__ is FlowEulerGuidanceInterval  # untouched
    # One class per sampler class
    assert type(multiview.multi_view_sampler(stock, [1, 1, 1], "multidiffusion")) is type(sampler)


def conditional_calls(model):
    return [value for _, value in model.calls if value != 0.0]


def unconditional_calls(model):
    return [t for t, value in model.calls if value == 0.0]


def test_multidiffusion_averages_the_pictures_by_weight_and_guides_once():
    pictures = torch.cat([picture(0.2), picture(-0.4), picture(0.6)])
    model = Model()
    sampler = multiview.multi_view_sampler(FlowEulerGuidanceInterval(1e-5), [2.0, 1.0, 1.0], "multidiffusion")
    sample(sampler, model, pictures, STRUCTURE)
    # Every step asks about every picture, in order; the unconditional pass runs once per guided step
    assert len(conditional_calls(model)) == 3 * 12
    assert np.allclose(conditional_calls(model)[:3], [0.2, -0.4, 0.6])
    guided = [t for t in sorted({t for t, _ in model.calls}, reverse=True) if 600 <= t <= 1000]
    assert unconditional_calls(model) == guided and 0 < len(guided) < 12

    # One step's prediction is the weighted mean of the pictures' predictions
    x, t = torch.randn(1, 5, 3), 0.3  # outside the guidance interval: the conditional prediction alone
    guidance = dict(neg_cond=torch.zeros(1, 5, 3), guidance_strength=7.5, guidance_interval=[0.6, 1.0])
    stock = FlowEulerGuidanceInterval(1e-5)
    single = [stock._inference_model(Model(), x, t, p[None], **guidance) for p in pictures]
    blended = sampler._inference_model(Model(), x, t, pictures, **guidance)
    assert torch.allclose(blended, (2 * single[0] + single[1] + single[2]) / 4)


def test_multidiffusion_only_scales_and_adds_predictions():
    """Upstream's SparseTensor (the shape and texture latents) supports `* float` and `+`; nothing else is used."""

    class Prediction:
        def __init__(self, value):
            self.value = value

        def __mul__(self, other):
            assert isinstance(other, float)
            return Prediction(self.value * other)

        def __add__(self, other):
            return Prediction(self.value + other.value)

    sampler = multiview.multi_view_sampler(FlowEuler(1e-5), [3.0, 1.0], "multidiffusion")
    model = lambda x, t, cond: Prediction(float(cond[0, 0, 0]))  # noqa: E731
    out = sampler._inference_model(model, torch.zeros(1, 2), 0.5, torch.cat([picture(1.0), picture(5.0)]))
    assert out.value == pytest.approx(0.75 * 1.0 + 0.25 * 5.0)


@pytest.mark.parametrize(
    "weights, turns",
    [
        ([1, 1, 1], [0, 1, 2] * 4),
        # The original twice as often as each view, spread out, and first
        ([2, 1, 1, 1], [0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1]),
        ([1, 1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]),
    ],
)
def test_stochastic_steps_take_turns_by_weight_starting_with_the_original(weights, turns):
    pictures = torch.cat([picture(0.1 * (i + 1)) for i in range(len(weights))])
    model = Model()
    sampler = multiview.multi_view_sampler(FlowEulerGuidanceInterval(1e-5), weights, "stochastic")
    sample(sampler, model, pictures, STRUCTURE)
    assert sampler.mv_schedule == turns
    # One conditional pass per step, on that step's picture; guidance's unconditional pass as upstream
    assert np.allclose(conditional_calls(model), [0.1 * (i + 1) for i in turns])
    assert 0 < len(unconditional_calls(model)) < 12


def test_each_sample_call_takes_its_own_turns():
    """The cascade samples the shape twice with one sampler: both stages start from the original."""
    sampler = multiview.multi_view_sampler(FlowEulerGuidanceInterval(1e-5), [1, 1, 1, 1, 1], "stochastic")
    pictures = torch.cat([picture(0.1 * (i + 1)) for i in range(5)])
    sample(sampler, Model(), pictures, STRUCTURE)
    first = list(sampler.mv_schedule)
    sample(sampler, Model(), pictures, STRUCTURE)
    assert first == sampler.mv_schedule == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


@pytest.mark.parametrize(
    "weights, mode, message",
    [
        ([1, 1], "average", "mode must be one of"),
        ([1, 0], "stochastic", "positive numbers"),
        ([1, -2], "multidiffusion", "positive numbers"),
        ([1, float("nan")], "multidiffusion", "positive numbers"),
        ([], "multidiffusion", "positive numbers"),
    ],
)
def test_bad_settings_are_refused(weights, mode, message):
    with pytest.raises(ValueError, match=message):
        multiview.multi_view_sampler(FlowEulerGuidanceInterval(1e-5), weights, mode)


def test_a_condition_that_does_not_match_the_weights_is_refused():
    sampler = multiview.multi_view_sampler(FlowEulerGuidanceInterval(1e-5), [1, 1], "multidiffusion")
    with pytest.raises(ValueError, match="3 pictures to condition on but 2 weights"):
        sample(sampler, Model(), torch.cat([picture(0.1)] * 3), STRUCTURE)


# --- The pipeline: run() as upstream's calls get_cond and the three samplers --------------------------


class Flow:
    """A flow model on the pipeline: callable, and moved about in low-VRAM mode."""

    def __init__(self, scale: float) -> None:
        self.model = Model()
        self.scale = scale

    def __call__(self, x, t, cond, concat_cond=None):
        return self.scale * self.model(x, t, cond, concat_cond)

    def to(self, device) -> None:
        pass

    def cpu(self) -> None:
        pass


class SamplingPipeline:
    """
    Upstream Trellis2ImageTo3DPipeline's run() reduced to its use of get_cond and the samplers: conditions
    at 512 (and 1024 unless '512'), the sparse structure, the shape (both cascade stages with one sampler)
    and the texture with the shape as concat_cond. A picture is a number; its tokens are picture(number).
    """

    device = "cpu"

    def __init__(self, sampler_class=FlowEulerGuidanceInterval) -> None:
        self.sparse_structure_sampler = sampler_class(sigma_min=1e-5)
        self.shape_slat_sampler = sampler_class(sigma_min=1e-5)
        self.tex_slat_sampler = sampler_class(sigma_min=1e-5)
        self.tex_slat_sampler_params = dict(TEXTURE)  # from pipeline.json
        scales = [("ss", 1.0), ("shape", 0.9), ("tex", 1.1), ("tex512", 1.2)]
        self.flows = {name: Flow(scale) for name, scale in scales}
        self.models = {"tex_slat_flow_model_1024": self.flows["tex"], "tex_slat_flow_model_512": self.flows["tex512"]}
        self.cond_calls = []  # (pictures, resolution) per get_cond call
        self.sampler_calls = []  # (sampler, cond rows, neg_cond rows) per sample() call
        self.tex_calls = []  # per sample_tex_slat call: its flow, params, shape and sampler
        self.rembg_model = None

    def get_cond(self, image, resolution, include_neg_cond=True):
        self.cond_calls.append((list(image), resolution))
        cond = torch.cat([picture(value) * (1 + resolution / 4096) for value in image])
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)} if include_neg_cond else {"cond": cond}

    def preprocess_image(self, image):
        return image

    def _sample(self, name, flow, noise, cond, params, **extra):
        sampler = getattr(self, name)
        self.sampler_calls.append((name, cond["cond"].shape[0], cond["neg_cond"].shape[0]))
        return sampler.sample(flow, noise, **cond, **params, verbose=False, **extra).samples

    def sample_tex_slat(self, cond, flow_model, shape_slat, sampler_params={}):  # noqa: B006 - upstream's
        """Upstream's order: the noise first, from torch's CPU generator, then the sampler on params over the defaults."""
        noise = torch.randn(1, 5, 3)
        params = {**self.tex_slat_sampler_params, **sampler_params}
        call = {"flow": flow_model, "params": params, "shape": shape_slat, "sampler": self.tex_slat_sampler, "noise": noise}
        self.tex_calls.append(call)
        return self._sample("tex_slat_sampler", flow_model, noise, cond, params, concat_cond=shape_slat)

    def decode_latent(self, shape_slat, tex_slat, resolution):
        return [types.SimpleNamespace(shape=shape_slat, tex=tex_slat, resolution=resolution)]

    def run(self, image, seed=42, pipeline_type="1024_cascade", preprocess_image=True, return_latent=False, **_):
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != "512" else None
        coords = self._sample("sparse_structure_sampler", self.flows["ss"], torch.randn(1, 5, 3), cond_512, STRUCTURE)
        if pipeline_type == "512":
            shape = self._sample("shape_slat_sampler", self.flows["shape"], torch.randn(1, 5, 3), cond_512, STRUCTURE)
            tex = self.sample_tex_slat(cond_512, self.models["tex_slat_flow_model_512"], shape)
            resolution = 512
        else:
            low = self._sample("shape_slat_sampler", self.flows["shape"], torch.randn(1, 5, 3), cond_512, STRUCTURE)
            shape = self._sample("shape_slat_sampler", self.flows["shape"], torch.randn(1, 5, 3) + low, cond_1024, STRUCTURE)
            tex = self.sample_tex_slat(cond_1024, self.models["tex_slat_flow_model_1024"], shape)
            resolution = 1024
        meshes = self.decode_latent(shape, tex, resolution)
        meshes[0].coords = coords  # upstream's shape latent carries its coordinates; this one doesn't
        return (meshes, (shape, tex, resolution)) if return_latent else meshes


def same_mesh(a, b, parts=("coords", "shape", "tex")) -> bool:
    return all(torch.equal(getattr(a, k), getattr(b, k)) for k in parts)


@pytest.mark.parametrize("cls", SAMPLER_CLASSES)
@pytest.mark.parametrize("pipeline_type", ["1024_cascade", "512"])
def test_one_picture_through_the_pipeline_is_bit_identical(cls, pipeline_type):
    ref = SamplingPipeline(cls).run(0.3, seed=5, pipeline_type=pipeline_type)[0]
    pipeline = SamplingPipeline(cls)
    with multiview.conditioned_on(pipeline, [0.3], [2.0], "multidiffusion"):
        out = pipeline.run(0.3, seed=5, pipeline_type=pipeline_type)[0]
    assert same_mesh(out, ref)


@pytest.mark.parametrize("pipeline_type, resolutions", [("1024_cascade", [512, 1024]), ("512", [512])])
def test_every_flow_is_conditioned_on_every_picture_built_one_by_one(pipeline_type, resolutions):
    pipeline = SamplingPipeline()
    stock = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
    with multiview.conditioned_on(pipeline, [0.3, -0.2, 0.5], [2.0, 1.0, 0.5], "stochastic"):
        swapped = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
        pipeline.run(0.3, seed=5, pipeline_type=pipeline_type)
    # Conditions per picture, as a single-picture run builds them, at each resolution run() asks for
    assert pipeline.cond_calls == [([v], r) for r in resolutions for v in (0.3, -0.2, 0.5)]
    stages = 4 if pipeline_type == "1024_cascade" else 3
    # Three pictures in every flow (both cascade stages), one unconditional row
    assert [rows for _, *rows in pipeline.sampler_calls] == [[3, 1]] * stages
    for name in multiview.SAMPLERS:
        assert swapped[name].mv_weights == (2.0, 1.0, 0.5) and swapped[name].mv_mode == "stochastic"
        assert getattr(pipeline, name) is stock[name]  # upstream's back
    assert "get_cond" not in vars(pipeline)
    # The original's own condition is the single-picture one, bit for bit
    single = SamplingPipeline().get_cond([0.3], 1024)["cond"]
    with multiview.conditioned_on(pipeline, [0.3, -0.2], [1.0, 1.0], "stochastic"):
        both = pipeline.get_cond([0.3], 1024)
    assert torch.equal(both["cond"][:1], single) and both["neg_cond"].shape[0] == 1


def test_the_pipeline_is_put_back_when_run_fails():
    pipeline = SamplingPipeline()
    stock = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with multiview.conditioned_on(pipeline, [0.3, -0.2], [2.0, 1.0], "multidiffusion"):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert all(getattr(pipeline, name) is stock[name] for name in multiview.SAMPLERS)
    assert "get_cond" not in vars(pipeline) and pipeline.get_cond.__func__ is SamplingPipeline.get_cond


def test_more_pictures_change_the_sample():
    ref = SamplingPipeline().run(0.3, seed=5)[0]
    for mode in multiview.MODES:
        pipeline = SamplingPipeline()
        with multiview.conditioned_on(pipeline, [0.3, -0.6, 0.9], [2.0, 1.0, 1.0], mode):
            out = pipeline.run(0.3, seed=5)[0]
        assert not torch.allclose(out.tex, ref.tex)


# --- Trellis2Runtime with views -------------------------------------------------------------------------


class Picture:
    """A picture whose cut-out, cropped form is a number (the fake pipeline's picture)."""

    def __init__(self, value, mode="RGB") -> None:
        self.value = value
        self.mode = mode


class RuntimePipeline(SamplingPipeline):
    """SamplingPipeline with upstream's preprocess_image (cut out and crop) and device handling."""

    def __init__(self, sampler_class=FlowEulerGuidanceInterval) -> None:
        super().__init__(sampler_class)
        self.low_vram = False
        self.prepared = []

    def preprocess_image(self, image):
        if image.value is None:  # nothing stands out from the background: upstream's bbox of nothing
            raise ValueError("zero-size array to reduction operation minimum which has no identity")
        self.prepared.append(image.value)
        return image.value

    def cuda(self) -> None:
        pass


def runtime_around(pipeline, settings: MultiView = MultiView()) -> Trellis2Runtime:
    runtime = Trellis2Runtime.__new__(Trellis2Runtime)
    runtime.pipeline = pipeline
    runtime.multiview = settings
    return runtime


@pytest.mark.parametrize("mode", ["preview", "final"])
def test_without_views_generate_is_upstreams_single_picture_run(mode):
    pipeline_type = PRESETS[mode].pipeline_type
    ref = SamplingPipeline().run(0.3, seed=11, pipeline_type=pipeline_type, preprocess_image=False)[0]
    for views in ([], ()):
        pipeline = RuntimePipeline()
        runtime = runtime_around(pipeline)
        out = runtime.generate(Picture(0.3), PRESETS[mode], seed=11, views=views)
        assert same_mesh(out, ref) and runtime.views_used == 0
        assert pipeline.cond_calls[0] == ([0.3], 512)
        assert {tuple(rows) for _, *rows in pipeline.sampler_calls} == {(1, 1)}
    assert same_mesh(runtime_around(RuntimePipeline()).generate(Picture(0.3), PRESETS[mode], seed=11), ref)


def views(*values, weights=None):
    """Views at azimuths 90, 180, 270..."""
    weights = weights or [1.0] * len(values)
    pairs = enumerate(zip(values, weights), 1)
    return [View(Picture(v), azimuth=90.0 * i, elevation=0.0, weight=w) for i, (v, w) in pairs]


@pytest.mark.parametrize("mode", multiview.MODES)
def test_views_are_cut_out_like_the_picture_and_weighed_against_it(mode, capsys):
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline, MultiView(mode=mode, picture_weight=3.0))
    stock = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
    seen = {}

    def run(image, **options):
        seen.update(image=image, weights=pipeline.shape_slat_sampler.mv_weights, mode=pipeline.tex_slat_sampler.mv_mode)
        return SamplingPipeline.run(pipeline, image, **options)

    pipeline.run = run
    mesh = runtime.generate(Picture(0.3), PRESETS["final"], seed=11, views=views(-0.2, 0.5, weights=[1.0, 0.5]))
    assert pipeline.prepared == [0.3, -0.2, 0.5]  # the picture, then each view, the same way
    assert seen == {"image": 0.3, "weights": (3.0, 1.0, 0.5), "mode": mode}
    assert runtime.views_used == 2 and mesh.tex is not None
    assert all(getattr(pipeline, name) is stock[name] for name in multiview.SAMPLERS)
    assert capsys.readouterr().out == f"[forge3d] 2 views besides the picture ({mode}; weights 3, 1, 0.5)\n"


def test_a_view_with_no_object_is_left_out(capsys):
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    runtime.generate(Picture(0.3), PRESETS["preview"], seed=11, views=views(-0.2, None, 0.5))
    assert runtime.views_used == 2
    assert [rows for _, *rows in pipeline.sampler_calls] == [[3, 1]] * 3
    log = capsys.readouterr().out.splitlines()
    assert log[0] == "[forge3d] views[1] (azimuth 180) left out: no object found in it"


def test_the_projection_still_paints_the_picture():
    class Mesh:
        pass

    pipeline = RuntimePipeline()
    pipeline.run = lambda image, **options: ([Mesh()], (torch.zeros(1, 5, 3), None, 1024))
    picture = Picture(0.3, mode="RGBA")  # its own alpha: its own cutout
    mesh = runtime_around(pipeline).generate(picture, PRESETS["final"], seed=11, views=views(-0.2))
    assert mesh.forge3d_cutout is picture


def test_views_carry_through_the_out_of_memory_fallback(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    pipeline = RuntimePipeline()  # its models (the texture flows) move about like the others
    pipeline.image_cond_model = Flow(1.0)
    pipeline.rembg_model = Flow(1.0)
    stock = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
    runs = []

    def run(image, **options):
        runs.append((options["pipeline_type"], pipeline.low_vram, pipeline.sparse_structure_sampler.mv_weights))
        if len(runs) < 3:
            # Upstream's samplers are back between attempts
            raise torch.cuda.OutOfMemoryError(f"CUDA out of memory (run {len(runs)})")
        return SamplingPipeline.run(pipeline, image, **options)

    pipeline.run = run
    runtime = runtime_around(pipeline)
    runtime.generate(Picture(0.3), PRESETS["final"], seed=11, views=views(-0.2, 0.5))
    assert runs == [
        ("1024_cascade", False, (2.0, 1.0, 1.0)),
        ("1024_cascade", True, (2.0, 1.0, 1.0)),
        ("512", True, (2.0, 1.0, 1.0)),
    ]
    assert runtime.pipeline_used == "512" and runtime.views_used == 2
    assert all(getattr(pipeline, name) is stock[name] for name in multiview.SAMPLERS)
    assert "get_cond" not in vars(pipeline) and pipeline.low_vram is False


def cutout(size=64, box=(8, 8, 56, 56), opaque=255):
    """An RGBA picture of one solid square (``box``) on a transparent frame."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=(200, 10, 10, opaque))
    return image


@pytest.mark.parametrize(
    "box, edges",
    [
        ((8, 8, 56, 56), ()),  # whole object inside the frame
        ((0, 8, 56, 56), ("left",)),  # runs off the left edge
        ((-5, 8, 70, 56), ("left", "right")),  # MV-Adapter's side view of a wide object
        ((8, -3, 56, 70), ("top", "bottom")),
        ((8, 8, 56, 63), ("bottom",)),
        ((30, 8, 63, 9), ()),  # a 2-pixel brush of the right edge, under 4 % of it: not clipped
    ],
)
def test_clipped_edges_names_the_frame_edges_the_object_runs_off(box, edges):
    from forge3d_worker.pipeline import clipped_edges

    assert clipped_edges(cutout(box=box)) == edges


def test_a_translucent_fringe_on_the_edge_is_not_clipping():
    from forge3d_worker.pipeline import clipped_edges

    # Upstream's bounding box ignores alpha at or under 80 %, and so does this
    assert clipped_edges(cutout(box=(0, 0, 63, 63), opaque=200)) == ()
    assert clipped_edges(cutout(box=(0, 0, 63, 63), opaque=205)) == ("left", "right", "top", "bottom")


def test_a_job_with_views_conditions_every_flow_and_reports_them(capsys):
    """
    handle_job end to end: decoded views, cut out like the picture, every flow. Of four views, one has
    nothing in it and one's object runs off the frame: both are left out.
    """
    import base64
    import io

    from forge3d_worker.service import handle_job

    class PicturePipeline(RuntimePipeline):
        """A picture's number is its object's red channel over 255; a fully transparent one has no object."""

        def preprocess_image(self, image):
            if image.mode == "RGBA" and image.getextrema()[3][1] == 0:
                raise ValueError("zero-size array to reduction operation minimum which has no identity")
            value = image.convert("RGB").getpixel((32, 32))[0] / 255
            self.prepared.append(value)
            return value

    def png(red, **options):
        image = cutout(**options)
        pixels = image.load()
        for x in range(image.width):
            for y in range(image.height):
                r, g, b, a = pixels[x, y]
                pixels[x, y] = (red, g, b, a)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    pipeline = PicturePipeline()
    runtime = runtime_around(pipeline, MultiView(mode="multidiffusion", picture_weight=2.0))
    weights = []
    stock_run = pipeline.run

    def run(image, **options):
        weights.append(pipeline.tex_slat_sampler.mv_weights)
        return stock_run(image, **options)

    pipeline.run = run
    runtime.export = lambda mesh, preset: (b"glb" * 10, 1234)
    views = [
        {"image_base64": png(51), "azimuth": 90},
        {"image_base64": png(0, opaque=0), "azimuth": 180},  # nothing in it
        {"image_base64": png(153, box=(-5, 8, 70, 56)), "azimuth": 225},  # runs off both sides
        {"image_base64": png(204), "azimuth": 270, "weight": 0.5},
    ]
    job = {"id": "j", "input": {"image_base64": png(102), "mode": "final", "seed": 3, "views": views}}
    out = handle_job(job, runtime, types.SimpleNamespace(put=lambda key, data, kind: {"key": key}), lambda raw, size: raw)

    assert "error" not in out and out["views_used"] == 2 and out["pipeline"] == "1024_cascade"
    # The picture, then every view with an object (the clipped one is cut out before it is judged)
    assert pipeline.prepared == pytest.approx([0.4, 0.2, 0.6, 0.8])
    assert weights == [(2.0, 1.0, 0.5)]
    assert [rows for _, *rows in pipeline.sampler_calls] == [[3, 1]] * 4  # structure, both shape stages, texture
    log = capsys.readouterr().out.splitlines()
    assert log[0] == "[forge3d] views[1] (azimuth 180) left out: no object found in it"
    assert log[1] == "[forge3d] views[2] (azimuth 225) left out: its object runs off the frame (left, right)"


# --- retexture(): the texture flow again, alone, on the generation's shape --------------------------------


def flow_calls(pipeline) -> dict:
    return {name: len(flow.model.calls) for name, flow in pipeline.flows.items()}


@pytest.mark.parametrize("cls", SAMPLER_CLASSES)
@pytest.mark.parametrize("mode, flow, resolution", [("final", "tex", 1024), ("preview", "tex512", 512)])
def test_retexture_draws_the_generations_texture_again_from_its_own_noise(cls, mode, flow, resolution, capsys):
    pipeline = RuntimePipeline(cls)
    runtime = runtime_around(pipeline)
    mesh = runtime.generate(Picture(0.3), PRESETS[mode], seed=11)
    before = flow_calls(pipeline)
    pipeline.cond_calls.clear()

    again = runtime.retexture()

    # Bit for bit: the same shape latent, condition, noise and sampler settings
    assert same_mesh(again, mesh, parts=("shape", "tex")) and again.resolution == resolution
    assert torch.equal(pipeline.tex_calls[-1]["noise"], pipeline.tex_calls[0]["noise"])
    # Only the texture flow ran (no structure, no shape), on the picture's condition at its resolution
    after = flow_calls(pipeline)
    assert {name for name in after if after[name] != before[name]} == {flow}
    assert pipeline.cond_calls == [([0.3], resolution)]
    assert pipeline.tex_calls[-1]["flow"] is pipeline.flows[flow] and pipeline.tex_calls[-1]["params"] == TEXTURE
    pipeline_type = PRESETS[mode].pipeline_type
    assert runtime.last_retexture == {"pipeline": pipeline_type, "sampler": TEXTURE, "noise": "generation", "views_used": 0}
    assert capsys.readouterr().out.startswith(f"[forge3d] retexturing the {pipeline_type} shape: {{")


def test_retexture_puts_its_sampler_params_over_the_defaults():
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    mesh = runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    calls = len(pipeline.flows["tex"].model.calls)

    again = runtime.retexture(sampler_params={"steps": 20, "guidance_strength": 3.0})

    expected = {**TEXTURE, "steps": 20, "guidance_strength": 3.0}
    assert pipeline.tex_calls[-1]["params"] == expected and runtime.last_retexture["sampler"] == expected
    # 20 steps; with guidance, the unconditional pass inside the guidance interval
    steps = pipeline.flows["tex"].model.calls[calls:]
    assert len([c for c in steps if c[1] != 0.0]) == 20 and 0 < len([c for c in steps if c[1] == 0.0]) < 20
    # The same noise and shape, another texture; the pipeline's defaults untouched
    assert torch.equal(pipeline.tex_calls[-1]["noise"], pipeline.tex_calls[0]["noise"])
    assert torch.equal(again.shape, mesh.shape) and not torch.allclose(again.tex, mesh.tex)
    assert pipeline.tex_slat_sampler_params == TEXTURE


def test_retexture_seeds_its_noise_so_variants_share_it():
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    mesh = runtime.generate(Picture(0.3), PRESETS["final"], seed=11)

    five = runtime.retexture(seed=5)
    assert runtime.last_retexture["noise"] == "seed 5" and not torch.allclose(five.tex, mesh.tex)
    assert torch.equal(runtime.retexture(seed=5).tex, five.tex)
    runtime.retexture(seed=5, sampler_params={"steps": 30})
    assert torch.equal(pipeline.tex_calls[-1]["noise"], pipeline.tex_calls[-3]["noise"])  # other settings, same noise
    assert not torch.allclose(runtime.retexture(seed=6).tex, five.tex)
    # A latent whose texture noise wasn't seen: seed None means the generation's seed
    latent = dataclasses.replace(runtime.last_latent, noise=None)
    assert torch.equal(runtime.retexture(latent=latent).tex, runtime.retexture(seed=11).tex)
    assert runtime.last_retexture["noise"] == "seed 11"


def test_retexture_takes_an_older_generations_latent():
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    first = runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    latent = runtime.last_latent
    second = runtime.generate(Picture(-0.4), PRESETS["final"], seed=12)
    assert runtime.last_latent is not latent and not torch.allclose(second.shape, first.shape)
    assert same_mesh(runtime.retexture(latent=latent), first, parts=("shape", "tex"))


@pytest.mark.parametrize("mode", multiview.MODES)
def test_views_steer_the_texture_flow_alone(mode, capsys):
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline, MultiView(mode=mode, picture_weight=2.0))
    mesh = runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    plain = runtime.retexture()
    stock = {name: getattr(pipeline, name) for name in multiview.SAMPLERS}
    before = flow_calls(pipeline)
    for log in (pipeline.cond_calls, pipeline.sampler_calls, pipeline.prepared):
        log.clear()
    capsys.readouterr()

    steered = runtime.retexture(views=views(-0.2, 0.5, weights=[1.0, 0.5]))

    assert pipeline.prepared == [-0.2, 0.5]  # cut out and cropped as generate() does its views
    assert pipeline.cond_calls == [([v], 1024) for v in (0.3, -0.2, 0.5)]  # each on its own, at 1024
    assert [rows for _, *rows in pipeline.sampler_calls] == [[3, 1]]  # one flow, on the three pictures
    sampler = pipeline.tex_calls[-1]["sampler"]
    assert sampler.mv_weights == (2.0, 1.0, 0.5) and sampler.mv_mode == mode
    # The structure and shape flows never ran; the shape and the noise are the generation's
    after = flow_calls(pipeline)
    assert {name for name in after if after[name] != before[name]} == {"tex"}
    assert torch.equal(pipeline.tex_calls[-1]["noise"], pipeline.tex_calls[0]["noise"])
    assert torch.equal(steered.shape, mesh.shape) and not torch.allclose(steered.tex, plain.tex)
    # Upstream's samplers and get_cond are back for the next run
    assert all(getattr(pipeline, name) is stock[name] for name in multiview.SAMPLERS)
    assert "get_cond" not in vars(pipeline) and "sample_tex_slat" not in vars(pipeline)
    assert runtime.last_retexture["views_used"] == 2
    assert capsys.readouterr().out.endswith(f", 2 views besides the picture ({mode}; weights 2, 1, 0.5)\n")


def test_a_view_with_no_object_is_left_out_of_a_retexture(capsys):
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    pipeline.sampler_calls.clear()
    runtime.retexture(views=views(None, 0.5))
    assert [rows for _, *rows in pipeline.sampler_calls] == [[2, 1]] and runtime.last_retexture["views_used"] == 1
    assert "[forge3d] views[0] (azimuth 90) left out: no object found in it" in capsys.readouterr().out


def test_retexture_keeps_the_picture_for_the_projection():
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    picture = Picture(0.3, mode="RGBA")  # its own alpha: its own cutout
    assert runtime.generate(picture, PRESETS["final"], seed=11).forge3d_cutout is picture
    assert runtime.last_latent.cutout is picture and runtime.last_latent.image == 0.3
    assert runtime.retexture().forge3d_cutout is picture
    assert runtime.retexture(views=views(-0.2)).forge3d_cutout is picture


def test_generate_leaves_upstreams_texture_sampling_as_it_was():
    """The runtime notes the texture's noise by wrapping sample_tex_slat during run(), and only then."""
    pipeline = RuntimePipeline()
    runtime = runtime_around(pipeline)
    runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    assert "sample_tex_slat" not in vars(pipeline) and isinstance(runtime.last_latent.noise, torch.Tensor)

    def run(image, **options):
        assert "sample_tex_slat" in vars(pipeline)  # wrapped while upstream runs
        raise ValueError("Invalid pipeline type")

    pipeline.run = run
    with pytest.raises(ValueError):
        runtime.generate(Picture(0.3), PRESETS["final"], seed=11)
    assert "sample_tex_slat" not in vars(pipeline) and runtime.last_latent is None
