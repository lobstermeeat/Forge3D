"""
2-D neighborhood attention in plain PyTorch: the one NATTEN call Pixal3D makes, through NAF.

Pixal3D upsamples DINOv3's patch features with NAF (valeoai/NAF, Apache-2.0), whose only compiled
dependency is NATTEN (MIT). NATTEN 0.20+ ships wheels for PyTorch 2.7+ only, production runs 2.6, and
the wheel Pixal3D's own demo installs carries kernels for Hopper (sm_90) alone, so it can't run on an
L40S or A10. Building NATTEN for Ada and Ampere would work, at the cost of a long CUDA compile in the
image. NAF only ever calls ``natten.na2d(q, k, v, kernel_size, dilation, stride=1)``, non-causal, on
one picture at a time, which this module computes directly: each query attends to the k x k keys of its
neighborhood, softmax over them. ``install()`` makes it the ``natten`` NAF imports, unless the real
NATTEN is installed.

The neighborhood is NATTEN's (see its Flex Attention backend, ``get_na_flex_mask`` in NATTEN 0.21.0,
for the definition every backend shares). Along each axis of length L, with dilation d, position i
belongs to dilation group r = i mod d, of length L_r = ceil((L - r) / d), at index g = i // d within
it. Its window is the k positions of that group centred on g, shifted inwards at the group's ends:
start = clamp(g - k // 2, 0, L_r - k), so a query near an edge still sees k keys.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Optional, Sequence, Union

import torch

Size2 = Union[int, Sequence[int]]

# Queries handled per chunk: bounds the gathered keys and values (rows x width x k^2 x heads x dims)
CHUNK_ELEMENTS = 1 << 27


def _pair(value: Size2, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    pair = tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int or a pair, got {value!r}")
    return pair  # type: ignore[return-value]


def neighbor_index(length: int, kernel_size: int, dilation: int = 1, device=None) -> torch.Tensor:
    """
    ``[length, kernel_size]`` positions each query along one axis attends to, in NATTEN's order
    (ascending, the window's start first).
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and positive, got {kernel_size}")
    if dilation < 1:
        raise ValueError(f"dilation must be positive, got {dilation}")
    if kernel_size * dilation > length:
        raise ValueError(f"kernel_size x dilation ({kernel_size} x {dilation}) exceeds the length {length}")
    position = torch.arange(length, device=device)
    group = position % dilation
    index = position // dilation
    group_length = (length - group + dilation - 1) // dilation
    start = torch.minimum((index - kernel_size // 2).clamp(min=0), group_length - kernel_size)
    offsets = torch.arange(kernel_size, device=device)
    return group[:, None] + dilation * (start[:, None] + offsets[None, :])


def na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Size2,
    stride: Size2 = 1,
    dilation: Size2 = 1,
    is_causal: Union[bool, Sequence[bool]] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[torch.Tensor] = None,
    additional_values: Optional[torch.Tensor] = None,
    backend: Optional[str] = None,  # NAF asks for "cutlass-fna"; there is only one way here
    **_tuning,  # tile shapes and other kernel options: nothing to tune
) -> torch.Tensor:
    """
    NATTEN's ``na2d`` for ``[batch, X, Y, heads, head_dim]`` tensors: stride 1, not causal, no extra
    keys (all NAF uses). Softmax runs in float32 whatever the inputs' type, as NATTEN's kernels do.
    """
    if _pair(stride, "stride") != (1, 1):
        raise NotImplementedError("only stride 1 is supported")
    causal = (is_causal, is_causal) if isinstance(is_causal, bool) else tuple(is_causal)
    if any(causal):
        raise NotImplementedError("causal neighborhood attention is not supported")
    if additional_keys is not None or additional_values is not None:
        raise NotImplementedError("additional keys and values are not supported")
    if query.dim() != 5 or key.shape != query.shape or value.shape[:-1] != query.shape[:-1]:
        raise ValueError(
            "expected query, key and value of shape [batch, X, Y, heads, head_dim], got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    kx, ky = _pair(kernel_size, "kernel_size")
    dx, dy = _pair(dilation, "dilation")
    batch, height, width, heads, dim = query.shape
    rows = neighbor_index(height, kx, dx, device=query.device)  # [X, kx]
    cols = neighbor_index(width, ky, dy, device=query.device)  # [Y, ky]
    scale = dim**-0.5 if scale is None else scale

    out = torch.empty(*query.shape[:-1], value.shape[-1], dtype=query.dtype, device=query.device)
    per_row = width * kx * ky * heads * max(dim, value.shape[-1])
    step = max(1, CHUNK_ELEMENTS // max(1, batch * per_row))
    for top in range(0, height, step):
        r = rows[top : top + step]  # [n, kx]
        n = r.shape[0]
        # [B, n, kx, Y, heads, D] -> each query column's ky neighbours: [B, n, kx, Y, ky, heads, D]
        k_rows = key[:, r]
        v_rows = value[:, r]
        k_win = k_rows[:, :, :, cols]
        v_win = v_rows[:, :, :, cols]
        q = query[:, top : top + n].float()  # [B, n, Y, heads, D]
        logits = torch.einsum("bnyhd,bnaychd->bnyhac", q, k_win.float()) * scale
        weights = logits.reshape(batch, n, width, heads, kx * ky).softmax(dim=-1)
        weights = weights.reshape(batch, n, width, heads, kx, ky)
        result = torch.einsum("bnyhac,bnaychd->bnyhd", weights, v_win.float())
        out[:, top : top + n] = result.to(out.dtype)
    return out


def available() -> bool:
    """Whether the real NATTEN is importable (then install() leaves it alone)."""
    module = sys.modules.get("natten")
    if module is not None:
        return not getattr(module, "_forge3d_stand_in", False)
    return importlib.util.find_spec("natten") is not None


def install() -> bool:
    """
    Makes ``import natten`` give this module's ``na2d`` when NATTEN isn't installed. Returns whether
    the stand-in was installed. NAF tries NATTEN's pre-0.20 API first (``natten.functional.na2d_qk``)
    and falls back to ``natten.na2d``; the stand-in has only the latter, so NAF takes that path.
    torch.hub's dependency check (NAF's hubconf lists natten) sees the module's spec.
    """
    if "natten" in sys.modules or available():  # already installed: the real one, or this one
        return False
    module = types.ModuleType("natten")
    module.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None)
    module.__version__ = "0.21.0+forge3d-pytorch"
    module._forge3d_stand_in = True  # type: ignore[attr-defined]
    module.na2d = na2d  # type: ignore[attr-defined]
    sys.modules["natten"] = module
    return True
