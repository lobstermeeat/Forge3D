"""The PyTorch na2d against NATTEN's own definition of a neighborhood, and its installation."""

import importlib
import sys

import pytest
import torch

from pixal3d_worker import neighborhood


def natten_mask_1d(length: int, kernel: int, dilation: int) -> torch.Tensor:
    """
    [query, key] mask of one axis, transcribed from NATTEN 0.21.0's Flex Attention backend
    (backends/flex.py, get_na_flex_mask.single_dim_tiling_mask; stride 1, not causal), position by
    position.
    """
    mask = torch.zeros(length, length, dtype=torch.bool)
    left, right = kernel // 2, kernel // 2 + (kernel % 2 - 1)
    for q in range(length):
        q_group, q_index = q % dilation, q // dilation
        padding = 1 - ((q_group + (dilation - (length % dilation))) // dilation)
        group_length = length // dilation + padding
        leader = min(q_index, group_length - 1)
        centre = min(max(leader, left), group_length - 1 - right)
        for k in range(length):
            if k % dilation != q_group:
                continue
            w0, w1 = centre - k // dilation, k // dilation - centre
            mask[q, k] = (0 <= w0 <= left) or (0 <= w1 <= right)
    return mask


def dense_reference(q, k, v, kernel, dilation, scale=None):
    """Full attention over every key, masked to NATTEN's neighborhoods."""
    batch, height, width, heads, dim = q.shape
    scale = dim**-0.5 if scale is None else scale
    rows = natten_mask_1d(height, kernel[0], dilation[0])
    cols = natten_mask_1d(width, kernel[1], dilation[1])
    mask = rows[:, None, :, None] & cols[None, :, None, :]  # [X, Y, X', Y']
    logits = torch.einsum("bxyhd,buvhd->bxyhuv", q.double(), k.double()) * scale
    logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))
    weights = logits.reshape(batch, height, width, heads, -1).softmax(-1).reshape(logits.shape)
    return torch.einsum("bxyhuv,buvhd->bxyhd", weights, v.double())


@pytest.mark.parametrize(
    "shape, kernel, dilation",
    [
        ((9, 9), (3, 3), (1, 1)),
        ((13, 11), (5, 3), (2, 3)),  # lengths not a multiple of the dilation: groups of two lengths
        ((16, 16), (3, 3), (4, 4)),  # NAF's case in small: keys upsampled 4x, dilation 4
        ((7, 12), (7, 5), (1, 2)),  # a kernel as long as the axis
    ],
)
def test_matches_natten_definition(shape, kernel, dilation):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, *shape, 3, 8) for _ in range(3))
    got = neighborhood.na2d(q, k, v, kernel_size=kernel, dilation=dilation)
    expected = dense_reference(q, k, v, kernel, dilation)
    assert got.shape == q.shape
    torch.testing.assert_close(got.double(), expected, rtol=1e-5, atol=1e-5)


def test_neighbor_index_shifts_windows_inwards_at_the_edges():
    index = neighborhood.neighbor_index(10, 3, 1)
    assert index[0].tolist() == [0, 1, 2]  # the first query still sees three keys
    assert index[5].tolist() == [4, 5, 6]
    assert index[9].tolist() == [7, 8, 9]
    # Dilation 2: even and odd positions are separate groups
    index = neighborhood.neighbor_index(10, 3, 2)
    assert index[1].tolist() == [1, 3, 5]
    assert index[8].tolist() == [4, 6, 8]


def test_chunking_gives_the_same_result(monkeypatch):
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 12, 10, 2, 4) for _ in range(3))
    whole = neighborhood.na2d(q, k, v, kernel_size=3, dilation=2)
    monkeypatch.setattr(neighborhood, "CHUNK_ELEMENTS", 1)  # one row at a time
    torch.testing.assert_close(neighborhood.na2d(q, k, v, kernel_size=3, dilation=2), whole)


def test_scale_and_dtype():
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 6, 6, 1, 4) for _ in range(3))
    got = neighborhood.na2d(q.half(), k.half(), v.half(), kernel_size=3, scale=0.3)
    assert got.dtype == torch.float16
    expected = dense_reference(q.half().float(), k.half().float(), v.half().float(), (3, 3), (1, 1), scale=0.3)
    torch.testing.assert_close(got.double(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "kwargs",
    [{"stride": 2}, {"is_causal": True}, {"additional_keys": torch.zeros(1)}],
)
def test_refuses_what_it_does_not_implement(kwargs):
    q = torch.zeros(1, 6, 6, 1, 4)
    with pytest.raises(NotImplementedError):
        neighborhood.na2d(q, q, q, kernel_size=3, **kwargs)


def test_refuses_a_window_longer_than_the_axis():
    q = torch.zeros(1, 6, 6, 1, 4)
    with pytest.raises(ValueError):
        neighborhood.na2d(q, q, q, kernel_size=5, dilation=2)


def test_install_gives_naf_the_stand_in(monkeypatch):
    monkeypatch.delitem(sys.modules, "natten", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "natten" else object())
    assert neighborhood.install()
    import natten

    assert natten.na2d is neighborhood.na2d
    assert natten.__spec__ is not None  # torch.hub's dependency check looks for it
    # NAF's import: the pre-0.20 API is missing, so it takes the na2d path
    with pytest.raises(ImportError):
        from natten.functional import na2d_qk  # noqa: F401
    assert not neighborhood.install()  # once is enough
    monkeypatch.delitem(sys.modules, "natten")


def test_install_leaves_real_natten_alone(monkeypatch):
    monkeypatch.delitem(sys.modules, "natten", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    assert not neighborhood.install()
    assert "natten" not in sys.modules
