"""
Several pictures of one object for TRELLIS.2, whose flows each take one.

TRELLIS.2 conditions its three flows (sparse structure, shape, texture) on one picture's DINOv3 tokens,
and invents whatever that picture doesn't show: the back of a shield, of an arcade cabinet. Its flow
models were trained on single pictures, but each sampling step only asks a model for a velocity given
the current latent and one picture's tokens, so other pictures of the object (its sides and back, drawn
by a multi-view model) can steer the same sample, as TRELLIS (v1) does in ``run_multi_image``:

- ``"stochastic"``: each step is conditioned on one picture, taken in turn with the original first, so it
  costs what a single picture costs. Weights set how often each picture's turn comes (smooth weighted
  round robin, so the turns spread evenly).
- ``"multidiffusion"``: each step averages every picture's prediction, weighted, so the conditional
  passes cost as many times more as there are pictures.

The pictures replace only the conditional prediction. Classifier-free guidance, its rescale and its
interval stay upstream's: they sit above this in the sampler's class hierarchy, and the unconditional
pass (on zeros, the same whatever the picture) runs once per step. Each ``sample`` call starts its own
turn order, so the shape cascade's two stages each start from the original picture.

With a single picture everything here passes straight through, bit for bit.
"""

from __future__ import annotations

import contextlib
import math
import types
from typing import Any, Iterator, Sequence

import torch

MODES = ("stochastic", "multidiffusion")
# The pipeline's three samplers; the shape sampler runs twice in a cascade (512, then 1024)
SAMPLERS = ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler")


def _rows(cond: Any) -> int:
    return int(cond.shape[0]) if isinstance(cond, torch.Tensor) and cond.ndim > 0 else 1


class MultiViewMixin:
    """
    Goes between classifier-free guidance and the Euler sampler that calls the model (see
    ``multi_view_sampler``). Set ``mv_mode`` and ``mv_weights`` (one per row of the condition).
    """

    mv_mode: str = "multidiffusion"
    mv_weights: Sequence[float] = (1.0,)
    # The picture each step of the last sample() call was conditioned on (stochastic), for logs and tests
    mv_schedule: list

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        # A fresh turn order per call: the cascade samples the shape twice
        self._mv_credit = [0.0] * len(self.mv_weights)
        self._mv_view = 0
        self.mv_schedule = []
        return super().sample(*args, **kwargs)  # type: ignore[misc]

    def sample_once(self, *args: Any, **kwargs: Any) -> Any:
        self._mv_view = self._next_view()
        self.mv_schedule.append(self._mv_view)
        return super().sample_once(*args, **kwargs)  # type: ignore[misc]

    def _next_view(self) -> int:
        """Smooth weighted round robin: view i gets weight_i / total of the steps, spread out, view 0 first."""
        weights = self.mv_weights
        credit = self._mv_credit
        for i, weight in enumerate(weights):
            credit[i] += weight
        best = max(range(len(weights)), key=lambda i: (credit[i], -i))
        credit[best] -= sum(weights)
        return best

    def _inference_model(self, model: Any, x_t: Any, t: float, cond: Any = None, **kwargs: Any) -> Any:
        count = _rows(cond)
        stock = super()._inference_model  # type: ignore[misc]
        if count == 1:
            # One picture, or guidance's unconditional pass
            return stock(model, x_t, t, cond, **kwargs)
        weights = self.mv_weights
        if count != len(weights):
            raise ValueError(f"{count} pictures to condition on but {len(weights)} weights")
        if self.mv_mode == "stochastic":
            i = self._mv_view
            return stock(model, x_t, t, cond[i : i + 1], **kwargs)
        total = float(sum(weights))
        blended = None
        for i, weight in enumerate(weights):
            term = stock(model, x_t, t, cond[i : i + 1], **kwargs) * (weight / total)
            blended = term if blended is None else blended + term
        return blended


_CLASSES: dict[type, type] = {}


def _multi_view_class(cls: type) -> type:
    """
    ``cls`` with MultiViewMixin just above the last class in its MRO that defines _inference_model (the
    Euler sampler, which calls the model), so the mixins above it (guidance interval, CFG) stay as they
    are: FlowEulerGuidanceIntervalSampler -> GuidanceInterval, CFG, MultiView, FlowEuler.
    """
    if cls not in _CLASSES:
        callers = [base for base in cls.__mro__ if "_inference_model" in vars(base)]
        if not callers:
            raise TypeError(f"{cls.__name__} has no _inference_model to condition on several pictures")
        leaf = callers[-1]
        mixed = types.new_class(f"MultiView{leaf.__name__}", (MultiViewMixin, leaf))
        # A plain Euler sampler (no guidance) is the leaf itself
        _CLASSES[cls] = mixed if leaf is cls else types.new_class(f"MultiView{cls.__name__}", (cls, mixed))
    return _CLASSES[cls]


def multi_view_sampler(sampler: Any, weights: Sequence[float], mode: str) -> Any:
    """A copy of ``sampler`` (same class and settings) that conditions on len(weights) pictures."""
    check(weights, mode)
    cls = _multi_view_class(type(sampler))
    copy = cls.__new__(cls)
    copy.__dict__.update(vars(sampler))
    copy.mv_weights = tuple(float(w) for w in weights)
    copy.mv_mode = mode
    copy.mv_schedule = []
    return copy


def check(weights: Sequence[float], mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"multi-view mode must be one of {MODES}, not {mode!r}")
    if not weights or not all(isinstance(w, (int, float)) and math.isfinite(w) and w > 0 for w in weights):
        raise ValueError(f"multi-view weights must be positive numbers, not {list(weights)!r}")


@contextlib.contextmanager
def conditioned_on(pipeline: Any, pictures: Sequence[Any], weights: Sequence[float], mode: str) -> Iterator[None]:
    """
    While active, ``pipeline.run(pictures[0], preprocess_image=False, ...)`` samples every flow from all
    ``pictures`` (prepared as preprocess_image prepares them, the original first), weighted by ``weights``.
    Conditions are built per picture at each resolution run() asks for (512, and 1024 for the cascade's
    second stage and its texture), so the original's are exactly what a single-picture run gets; the
    unconditional condition keeps one row. Upstream's samplers and get_cond are back afterwards, also
    when run() fails.
    """
    if len(pictures) != len(weights):
        raise ValueError(f"{len(pictures)} pictures but {len(weights)} weights")
    check(weights, mode)
    stock_samplers = {name: getattr(pipeline, name) for name in SAMPLERS}
    had_own_get_cond = "get_cond" in vars(pipeline)
    stock_get_cond = pipeline.get_cond

    def get_cond(image: Any, resolution: int, include_neg_cond: bool = True) -> dict:
        # run() passes [pictures[0]]; every picture is used instead, each on its own (as run() would)
        parts = [stock_get_cond([picture], resolution, include_neg_cond) for picture in pictures]
        cond = {"cond": torch.cat([part["cond"] for part in parts])}
        if include_neg_cond:
            cond["neg_cond"] = parts[0]["neg_cond"][:1]
        return cond

    try:
        for name, sampler in stock_samplers.items():
            setattr(pipeline, name, multi_view_sampler(sampler, weights, mode))
        pipeline.get_cond = get_cond
        yield
    finally:
        for name, sampler in stock_samplers.items():
            setattr(pipeline, name, sampler)
        if had_own_get_cond:
            pipeline.get_cond = stock_get_cond
        else:
            vars(pipeline).pop("get_cond", None)
