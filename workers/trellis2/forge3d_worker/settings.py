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


PRESETS: dict[str, Preset] = {
    # Shown while the user decides; small enough to stream instantly
    "preview": Preset(pipeline_type="512", max_faces=30_000, texture_size=1024, remesh=True),
    # What gets published: detailed, but still light enough for phones
    "final": Preset(pipeline_type="1024_cascade", max_faces=100_000, texture_size=2048, remesh=True),
}

# Attribution the DINOv3 license requires wherever generated assets are offered
CREDITS = ("Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)")
