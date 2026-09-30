"""The review gallery's page, with a reviewer's verdicts (no rendering needed)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gallery"))

import make_gallery  # noqa: E402


def runs():
    return [make_gallery.Run(name=f"s-0{i}", subject=f"thing {i}") for i in range(1, 5)]


def test_verdicts_are_counted_against_the_bar_overall_and_per_group():
    board = runs()
    make_gallery.apply_verdicts(
        board,
        {
            "s-01": {"verdict": "publish", "note": "Clean", "group": "Everyday"},
            "s-02": {"verdict": "publish", "group": "Everyday"},
            "s-03": {"verdict": "edits", "note": "The <handle> is fused", "group": "Weak spot"},
            "s-04": {"verdict": "reject", "note": "Two objects", "group": "Weak spot"},
        },
    )
    html = make_gallery.page(board, "Set", "Sub")
    assert "<dt>Publishable (bar 80%)</dt><dd>2 of 4 (50%)</dd>" in html
    assert "<dt>Everyday</dt><dd>2 of 2 (100%)</dd>" in html
    assert "<dt>Weak spot</dt><dd>0 of 2 (0%)</dd>" in html
    assert '<span class="chip warn">Needs edits</span>' in html
    assert '<span class="chip fail">Not usable</span>' in html
    assert '<p class="note">The &lt;handle&gt; is fused</p>' in html  # notes are escaped


def test_without_verdicts_the_page_is_unchanged():
    html = make_gallery.page(runs(), "Set", "Sub")
    assert "Publishable" not in html and '<span class="chip">Preview only</span>' in html


def test_verdicts_must_name_real_runs_and_known_verdicts():
    with pytest.raises(SystemExit, match="aren't in the gallery: s-09"):
        make_gallery.apply_verdicts(runs(), {"s-09": {"verdict": "publish"}})
    with pytest.raises(SystemExit, match="verdict must be one of"):
        make_gallery.apply_verdicts(runs(), {"s-01": {"verdict": "great"}})
