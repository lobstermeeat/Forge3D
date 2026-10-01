"""Job handling, kept free of GPU code so it can be tested anywhere."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Optional, Protocol, Sequence

from .inputs import Fetch, InputError, View, parse_job
from .settings import CREDITS, PRESETS, Preset
from .storage import Storage


class Runtime(Protocol):
    """
    The GPU side. After generate() it may set ``pipeline_used`` to the TRELLIS.2 pipeline that actually
    made the mesh (a final that runs out of GPU memory falls back to the preview's); without it, the
    preset's pipeline is reported. It gets ``views`` only when the job has some, and may then set
    ``views_used`` to how many it used (all of them, if it doesn't say).
    """

    def generate(self, image: Any, preset: Preset, seed: int, views: Sequence[View] = ()) -> Any: ...

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]: ...


Pack = Callable[[bytes, int], bytes]

_GPU_FAULTS = ("cuda", "out of memory", "outofmemory", "cublas", "cudnn", "device-side assert")


def needs_restart(err: BaseException) -> bool:
    """A GPU fault can leave CUDA unusable in this process; other failures (storage, network) can't."""
    text = f"{type(err).__name__}: {err}".lower()
    return any(marker in text for marker in _GPU_FAULTS)


def handle_job(
    job: dict,
    runtime: Runtime,
    storage: Storage,
    pack: Pack,
    fetch: Optional[Fetch] = None,
) -> dict:
    """
    Input: ``{"image_url" | "image_base64", "mode": "preview" | "final", "seed"?, "request_id"?, "views"?}``,
    where ``views`` are other pictures of the object: ``[{"image_url" | "image_base64", "azimuth",
    "elevation", "weight"?}]``. Reuse the preview's ``seed`` and ``views`` for the final pass so the final
    refines the approved shape.
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
        if spec.views:
            mesh = runtime.generate(spec.image, preset, spec.seed, views=spec.views)
            views_used = getattr(runtime, "views_used", len(spec.views))
        else:
            # Exactly as before views existed (and for runtimes that don't take them)
            mesh = runtime.generate(spec.image, preset, spec.seed)
            views_used = 0
        pipeline = getattr(runtime, "pipeline_used", None) or preset.pipeline_type
        lap("generate_s")
        raw, triangles = runtime.export(mesh, preset)
        lap("export_s")
        packed = pack(raw, preset.texture_size)
        lap("compress_s")
        stored = storage.put(spec.output_key, packed, "model/gltf-binary")
        lap("upload_s")
    except InputError as err:
        # e.g. no object found in the image
        return {"error": f"invalid input: {err}"}
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        traceback.print_exc()
        result = {"error": f"generation failed: {type(err).__name__}: {err}"}
        if needs_restart(err):
            result["refresh_worker"] = True
        return result

    result = {
        "request_id": spec.request_id,
        "mode": spec.mode,
        "seed": spec.seed,
        "glb": stored,
        "bytes": len(packed),
        "raw_bytes": len(raw),
        "triangles": triangles,
        # "512" for a final made with the preview's pipeline because the cascade ran out of GPU memory
        "pipeline": pipeline,
        # Extra views of the object that helped make the model (0: the picture alone)
        "views_used": views_used,
        "timings": timings,
        "credits": list(CREDITS),
    }
    # Optional: whether the picture was painted onto the model, and why not (Trellis2Runtime)
    projection = getattr(runtime, "last_projection", None)
    if isinstance(projection, dict) and projection:
        result["projection"] = projection
        if projection.get("gpu_fault"):
            result["refresh_worker"] = True  # the model went out unprojected; CUDA may not be usable
    return result
