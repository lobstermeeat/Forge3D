"""Job handling, kept free of GPU code so it can be tested anywhere."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Optional, Protocol

from .inputs import Fetch, InputError, parse_job
from .settings import CREDITS, PRESETS, Preset
from .storage import Storage


class Runtime(Protocol):
    def generate(self, image: Any, preset: Preset, seed: int) -> Any: ...

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]: ...


Pack = Callable[[bytes, int], bytes]


def handle_job(
    job: dict,
    runtime: Runtime,
    storage: Storage,
    pack: Pack,
    fetch: Optional[Fetch] = None,
) -> dict:
    """
    Input: ``{"image_url" | "image_base64", "mode": "preview" | "final", "seed"?, "request_id"?}``.
    Reuse the preview's ``seed`` for the final pass so the final refines the approved shape.
    """
    try:
        spec = parse_job(job.get("input"), fallback_id=str(job.get("id", "job")), fetch=fetch)
    except InputError as err:
        return {"error": f"invalid input: {err}"}

    preset = PRESETS[spec.mode]
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now

    try:
        mesh = runtime.generate(spec.image, preset, spec.seed)
        lap("generate_s")
        raw, triangles = runtime.export(mesh, preset)
        lap("export_s")
        packed = pack(raw, preset.texture_size)
        lap("compress_s")
        stored = storage.put(spec.output_key, packed, "model/gltf-binary")
        lap("upload_s")
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        traceback.print_exc()
        # A fresh process is the safest recovery after a GPU error (e.g. out of memory)
        return {"error": f"generation failed: {type(err).__name__}: {err}", "refresh_worker": True}

    return {
        "request_id": spec.request_id,
        "mode": spec.mode,
        "seed": spec.seed,
        "glb": stored,
        "bytes": len(packed),
        "raw_bytes": len(raw),
        "triangles": triangles,
        "timings": timings,
        "credits": list(CREDITS),
    }
