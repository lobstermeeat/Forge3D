"""Generation presets for the two passes of the funnel: a cheap preview and the final asset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["preview", "final"]


@dataclass(frozen=True)
class Preset:
    # TRELLIS.2 pipeline type: '512' takes ~3 s on an H100, '1024_cascade' ~17 s
    pipeline_type: str
    # Target triangle count after simplification (CuMesh counts faces)
    max_faces: int
    # Baked PBR texture size before KTX2 compression
    texture_size: int
    # Use TRELLIS.2's narrow-band remesh (cleaner, watertight topology)
    remesh: bool
    # Paint the input picture onto the side of the model it shows (forge3d_worker.projection)
    project_picture: bool = False


PRESETS: dict[str, Preset] = {
    # Shown while the user decides; small enough to stream instantly
    "preview": Preset(pipeline_type="512", max_faces=30_000, texture_size=1024, remesh=True),
    # What gets published: detailed, but still light enough for phones
    "final": Preset(
        pipeline_type="1024_cascade", max_faces=100_000, texture_size=2048, remesh=True, project_picture=True
    ),
}

ViewMode = Literal["stochastic", "multidiffusion"]


@dataclass(frozen=True)
class MultiView:
    """How a job's extra views of the object (its other sides) steer TRELLIS.2 (see multiview.py)."""

    # "stochastic": each sampling step sees one picture in turn (a single picture's cost);
    # "multidiffusion": each step averages every picture's prediction (that many times the cost).
    # Phase 6 tests with MV-Adapter views: multidiffusion kept fronts crisp and copied the back from the
    # views; stochastic washed fronts out, made two fronts, and wrecked the fox control
    mode: ViewMode = "multidiffusion"
    # The original picture's weight; each view weighs 1 unless the job gives it a weight. The original is
    # what the user picked and what the projection paints from; generated views may drift from it.
    # With four views, 1 and 4 both did a little worse than 2 (at 4 the arcade's back became a second front)
    picture_weight: float = 2.0


# Used for previews and finals alike, so a final keeps the shape of the preview made with the same views
MULTIVIEW = MultiView()

# Attribution the DINOv3 license requires wherever generated assets are offered
CREDITS = ("Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)")
