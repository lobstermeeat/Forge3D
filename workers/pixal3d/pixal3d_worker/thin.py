"""
Thin, flat objects get Pixal3D's single-view weights.

The production recipe builds a picture with Pixal3D's multi-view weights and the picture as their one
view: on the side the picture doesn't show they invent far less than the single-view weights (a plain
back on the helmet, a proper rear on the car, a white bowl all round the ramen). On a thin, flat object
they fail instead: the shield came out as a hollow tray and the skateboard as two decks, where the
single-view weights built a thin clean shield and a clean skateboard.

TRELLIS.2's '512' preview of the picture, made anyway to level the model, tells the two apart: the
smallest extent of its axis-aligned bounding box over the largest ("thin ratio"). On the twenty Phase 2
previews: shield 0.10, skateboard 0.14, pistol 0.15, then arcade 0.37, sneaker 0.40 and everything else
at least 0.46. The box is axis-aligned on purpose: a flat object pictured on the diagonal (the guitar,
0.89 axis-aligned but 0.15 along its principal axes) is one the multi-view weights built cleanly.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# At or below this thin ratio, the single-view weights build the picture
THIN_RATIO = 0.20


def extents(glb: Any) -> list[float]:
    """The mesh's axis-aligned bounding-box extents, smallest first."""
    bounds = np.asarray(glb.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("the mesh has no bounding box")
    return sorted(float(v) for v in bounds[1] - bounds[0])


def ratio(sizes: list[float]) -> Optional[float]:
    """Smallest extent over largest, or None for a degenerate box."""
    smallest, largest = min(sizes), max(sizes)
    if largest <= 0:
        return None
    return smallest / largest


def is_thin(value: Optional[float], threshold: float = THIN_RATIO) -> bool:
    return value is not None and value <= threshold


def measure(glb: Any, threshold: float = THIN_RATIO) -> dict:
    """
    The rule's report for a (level) mesh: ``{"extents", "ratio", "threshold", "thin"}``. ``thin`` says
    whether the single-view weights should build it.
    """
    sizes = extents(glb)
    value = ratio(sizes)
    shown = None if value is None else round(value, 4)  # the reported ratio is the one compared
    return {
        "extents": [round(v, 4) for v in sizes],
        "ratio": shown,
        "threshold": threshold,
        "thin": is_thin(shown, threshold),
    }
