"""Job handling, kept free of GPU code so it can be tested anywhere."""

from __future__ import annotations

import dataclasses
import inspect
import time
import traceback
from typing import Any, Callable, Optional, Protocol, Sequence

from .inputs import Fetch, InputError, Job, View, parse_job
from .settings import CREDITS, PRESETS, TEXTURE_SEED_STEP, Preset
from .storage import Storage

# A textures job's error where TRELLIS.2 doesn't make the finals: here, from a runtime that can't retexture
# (the Pixal3D worker's); in modal_app.py, from a deployment with the recipe on
TEXTURES_NEED_TRELLIS2 = "texture options need TRELLIS.2 finals"


class Runtime(Protocol):
    """
    The GPU side. After generate() it may set ``pipeline_used`` to the TRELLIS.2 pipeline that actually
    made the mesh (a final that runs out of GPU memory falls back to the preview's); without it, the
    preset's pipeline is reported. It gets ``views`` only when the job has some and its generate() takes
    them (takes_views), and may then set ``views_used`` to how many it used (all of them, if it doesn't say).
    A textures job also needs ``retexture(seed=…)``: a new texture for the last generation's shape
    (Trellis2Runtime's).
    """

    def generate(self, image: Any, preset: Preset, seed: int, views: Sequence[View] = ()) -> Any: ...

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]: ...


Pack = Callable[[bytes, int], bytes]


def takes_views(runtime: Any) -> bool:
    """
    Whether the runtime's generate() accepts ``views``. A runtime from before views, or one that binds a
    job's views itself (the Pixal3D worker's service wraps its runtime that way), is called as it always
    was, with the picture alone.
    """
    try:
        parameters = inspect.signature(runtime.generate).parameters
    except (TypeError, ValueError):  # a callable the inspector can't read
        return False
    return "views" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

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
    refines the approved shape. ``"mode": "textures"`` with the final's input (its ``seed`` required),
    ``"count"``? and ``"pipeline"``? makes more textures for the final's shape instead: see make_textures.
    """
    try:
        spec = parse_job(job.get("input"), fallback_id=str(job.get("id", "job")), fetch=fetch)
    except InputError as err:
        return {"error": f"invalid input: {err}"}
    if spec.mode == "textures":
        return make_textures(spec, runtime, storage, pack)

    preset = PRESETS[spec.mode]
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now

    try:
        mesh, views_used = _generate(spec, runtime, preset)
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
        return _failed(err)

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
    notes = _export_notes(runtime)
    result.update(notes)
    if _gpu_fault(notes):
        result["refresh_worker"] = True
    return result


def texture_seed(seed: int, number: int) -> int:
    """The noise seed of texture ``number`` (1 to count) in a textures job for the final made with ``seed``."""
    return seed + TEXTURE_SEED_STEP * number


def make_textures(spec: Job, runtime: Runtime, storage: Storage, pack: Pack) -> dict:
    """
    A textures job (texture options): the final's shape made again exactly as the final was (its preset,
    seed and views), then ``spec.count`` new textures for it from TRELLIS.2's texture flow alone
    (``runtime.retexture``). Texture k draws its noise from texture_seed(seed, k), and is exported, packed
    and stored as a final is, at ``spec.texture_key(k)``. The final's own texture isn't exported again: the
    creator has it. A texture that fails is listed in ``texture_errors`` and the others go on; when none is
    made, the job fails. A runtime that can't retexture (the Pixal3D worker's) is refused before anything runs.
    Each texture's ``export`` says how its export ran (Trellis2Runtime's last_export): the first runs to_glb
    in full and keeps the shape's texture layout, the others only sample their texture onto it ("rebake").

    A final that fell back to the preview's pipeline (``spec.pipeline``, "512") has that pipeline's shape, so
    the shape is made with it at once, as the fallback made it: the final's preset with that pipeline, run by
    the same generate(). The cascade would make another shape, or run out of memory again first.
    """
    if not callable(getattr(runtime, "retexture", None)):
        return {"error": TEXTURES_NEED_TRELLIS2}
    preset = PRESETS["final"]
    if spec.pipeline is not None:
        preset = dataclasses.replace(preset, pipeline_type=spec.pipeline)
    # Each step's seconds, summed over the textures. A failed step counts too: its GPU time was spent
    timings = dict.fromkeys(("generate_s", "retexture_s", "export_s", "pack_s", "upload_s"), 0.0)

    def timed(name: str, work: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return work(*args, **kwargs)
        finally:
            timings[name] = round(timings[name] + time.perf_counter() - started, 3)

    try:
        mesh, views_used = timed("generate_s", _generate, spec, runtime, preset)
        # Not exported: its texture is the final's own. Let go at once, so its GPU memory is free for the
        # textures; the runtime keeps the shape latent that retexture() samples them on
        del mesh
        pipeline = getattr(runtime, "pipeline_used", None) or preset.pipeline_type
    except InputError as err:
        # e.g. no object found in the image
        return {"error": f"invalid input: {err}"}
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        return _failed(err)

    textures: list[dict] = []
    errors: list[dict] = []
    refresh = False
    for number in range(1, spec.count + 1):
        seed = texture_seed(spec.seed, number)
        try:
            mesh = timed("retexture_s", runtime.retexture, seed=seed)
            try:
                raw, triangles = timed("export_s", runtime.export, mesh, preset)
            finally:
                del mesh  # off the GPU before the next texture is sampled
            notes = {**_export_notes(runtime), **_export_path(runtime)}
            refresh = refresh or _gpu_fault(notes)
            packed = timed("pack_s", pack, raw, preset.texture_size)
            stored = timed("upload_s", storage.put, spec.texture_key(number), packed, "model/gltf-binary")
        except Exception as err:  # noqa: BLE001 - one texture failing must not lose the others
            print(f"[forge3d] texture {number} of {spec.count} (seed {seed}) failed:")
            traceback.print_exc()
            errors.append({"texture_seed": seed, "error": f"{type(err).__name__}: {err}"})
            refresh = refresh or needs_restart(err)
            # Its traceback holds the failed texture's mesh, and so its GPU memory, for as long as the error
            # lives: let go of it before the next texture
            traceback.clear_frames(err.__traceback__)
            continue
        textures.append(
            {
                "texture_seed": seed,
                "glb": stored,
                "bytes": len(packed),
                "raw_bytes": len(raw),
                "triangles": triangles,
                **notes,  # "projection" and "floaters", as a final has them, and "export"
            }
        )

    if textures:
        result = {
            "request_id": spec.request_id,
            "mode": spec.mode,
            "seed": spec.seed,
            "textures": textures,
            # The shape's pipeline: "512" when the cascade ran out of GPU memory (see handle_job)
            "pipeline": pipeline,
            "views_used": views_used,
            "timings": timings,
            "credits": list(CREDITS),
        }
        if errors:
            result["texture_errors"] = errors
    else:
        result = {"error": f"generation failed: none of the {spec.count} textures was made: {errors[0]['error']}"}
    if refresh:
        result["refresh_worker"] = True
    return result


def _generate(spec: Job, runtime: Runtime, preset: Preset) -> tuple[Any, int]:
    """The job's mesh, and how many of its views helped make it."""
    if spec.views and takes_views(runtime):
        mesh = runtime.generate(spec.image, preset, spec.seed, views=spec.views)
        return mesh, getattr(runtime, "views_used", len(spec.views))
    # Exactly as before views existed. A runtime that doesn't take views gets the picture alone and
    # reports the views it used itself, if it bound them some other way (else none)
    mesh = runtime.generate(spec.image, preset, spec.seed)
    return mesh, int(getattr(runtime, "views_used", 0) or 0) if spec.views else 0


def _failed(err: Exception) -> dict:
    """A failed job's result. After a GPU fault the worker is replaced too: CUDA may be unusable in it."""
    traceback.print_exc()
    result = {"error": f"generation failed: {type(err).__name__}: {err}"}
    if needs_restart(err):
        result["refresh_worker"] = True
    return result


def _export_notes(runtime: Any) -> dict:
    """
    What the runtime's last export() reported besides the model (Trellis2Runtime keeps both): ``projection``,
    whether the picture was painted onto the model and why not, and ``floaters``, the small pieces floating
    apart it dropped (presets with drop_floaters). Each only when there is one.
    """
    notes = {}
    for key, attribute in (("projection", "last_projection"), ("floaters", "last_cleanup")):
        value = getattr(runtime, attribute, None)
        if isinstance(value, dict) and value:
            notes[key] = value
    return notes


def _export_path(runtime: Any) -> dict:
    """
    How the runtime's last export() made its model (Trellis2Runtime's last_export), as ``export``:
    ``{"path": "to_glb" | "rebake", "seconds": ...}``. Only when there is one; textures jobs only.
    """
    value = getattr(runtime, "last_export", None)
    return {"export": dict(value)} if isinstance(value, dict) and value else {}


def _gpu_fault(notes: dict) -> bool:
    """A GPU fault while projecting: the model went out unprojected, and CUDA may not be usable."""
    return bool(notes.get("projection", {}).get("gpu_fault"))
