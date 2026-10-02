"""Job handling, kept free of GPU code so it can be tested anywhere."""

from __future__ import annotations

import base64
import inspect
import io
import json
import math
import random
import time
import traceback
from typing import Any, Callable, Optional, Protocol, Sequence

from PIL import Image

from .inputs import Fetch, InputError, Job, View, parse_job
from .settings import CREDITS, PRESETS, TEXTURE_SEED_STEP, Preset
from .storage import Storage

# A textures job's error where TRELLIS.2 doesn't make the finals: here, from a runtime that can't retexture
# (the Pixal3D worker's); in modal_app.py, from a deployment with the recipe on
TEXTURES_NEED_TRELLIS2 = "texture options need TRELLIS.2 finals"
# A textures job's judge_error when the job asks for the judge and the worker has none (RunPod's)
NO_JUDGE = "no judge in this worker"
# The timings a textures job adds when it asks for the judge (each 0 when the judge's part didn't run)
JUDGE_TIMINGS = ("own_export_s", "render_s", "judge_s")
# The judge's verdicts (judge_worker/parse.py); anything else is passed on as null
VERDICTS = ("publish", "edits", "reject")
# The judge looks at the picture at most this big on its longest side, with any transparency on white
# (judge_worker/judge.py's fit and PICTURE_SIDE). Sent that way, the picture is what the judge would make
# of the full one, in a fraction of the bytes (a large photo as PNG can pass the judge's 20 MB limit)
JUDGE_PICTURE_SIDE = 768
JUDGE_UNDERLAY = (255, 255, 255)


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


class Judge(Protocol):
    """
    What a textures job asks which texture is best, when its input says ``"judge": true``. Called with the
    judge worker's job (workers/judge/judge_worker/service.py): ``{"picture_png", "candidates_png": [...],
    "prompt", "order"}``, base64 PNGs, it returns that worker's result (``best``, ``verdicts``, ``why``,
    ``seconds``, ``model``, and ``parse_error`` when the reply named no pick) or ``{"error": ...}``, and it may
    raise. Two methods are optional: ``unavailable()``, why it can't judge here (its weights are missing,
    say), which the job checks before any work for the judge; and ``warm()``, which starts it up without
    waiting while the textures are made. modal_app.py's is Judge8B, called through Modal.
    """

    def __call__(self, request: dict) -> dict: ...


# A candidate's exported GLB (before gltfpack) to the grid of views the judge looks at
Render = Callable[[bytes], Image.Image]


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
    judge: Optional[Judge] = None,
    render: Optional[Render] = None,
) -> dict:
    """
    Input: ``{"image_url" | "image_base64", "mode": "preview" | "final", "seed"?, "request_id"?, "views"?}``,
    where ``views`` are other pictures of the object: ``[{"image_url" | "image_base64", "azimuth",
    "elevation", "weight"?}]``. Reuse the preview's ``seed`` and ``views`` for the final pass so the final
    refines the approved shape. ``"mode": "textures"`` with the final's input (its ``seed`` required),
    ``"count"``? and ``"judge"``? (with ``"prompt"``?) makes more textures for the final's shape instead,
    and with ``"judge": true`` asks ``judge`` which is best: see make_textures. Previews and finals never
    use ``judge`` or ``render``.
    """
    try:
        spec = parse_job(job.get("input"), fallback_id=str(job.get("id", "job")), fetch=fetch)
    except InputError as err:
        return {"error": f"invalid input: {err}"}
    if spec.mode == "textures":
        return make_textures(spec, runtime, storage, pack, judge=judge, render=render)

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


def make_textures(
    spec: Job,
    runtime: Runtime,
    storage: Storage,
    pack: Pack,
    judge: Optional[Judge] = None,
    render: Optional[Render] = None,
) -> dict:
    """
    A textures job (texture options): the final's shape made again exactly as the final was (its preset,
    seed and views), then ``spec.count`` new textures for it from TRELLIS.2's texture flow alone
    (``runtime.retexture``). Texture k draws its noise from texture_seed(seed, k), and is exported, packed
    and stored as a final is, at ``spec.texture_key(k)``. A texture that fails is listed in
    ``texture_errors`` and the others go on; when none is made, the job fails. A runtime that can't
    retexture (the Pixal3D worker's) is refused before anything runs. Each texture's ``export`` says how its
    export ran (Trellis2Runtime's last_export): the first export of the shape runs to_glb in full and keeps
    its texture layout, the later ones only sample their texture onto it ("rebake").

    Without ``spec.judge`` the final's own texture isn't exported: the creator has it. With it, the job
    asks ``judge`` which texture a creator would rather use (judge_textures): the own texture is exported
    first, as the final was, but keeping the shape's texture layout (``runtime.keep_layout``), so every new
    texture rebakes; it is never packed, stored or returned. Then each candidate, the own texture and the
    new ones, is drawn by ``render`` (judgeviews' grid of its exported GLB) and the judge gets them all. The
    result then has ``judge`` (its pick and verdicts) or ``judge_error`` (why there is none), and
    ``own_texture`` (how the own texture's export went) when it was exported. The judge's part never fails
    the textures, and when the judge can't run here (``unavailable()``, or no judge at all) none of its
    part runs: the job goes on as without it.
    """
    if not callable(getattr(runtime, "retexture", None)):
        return {"error": TEXTURES_NEED_TRELLIS2}
    preset = PRESETS["final"]
    # Each step's seconds, summed over the textures. A failed step counts too: its GPU time was spent
    steps = ("generate_s", "retexture_s", "export_s", "pack_s", "upload_s")
    timings = dict.fromkeys(steps + (JUDGE_TIMINGS if spec.judge else ()), 0.0)

    def timed(name: str, work: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return work(*args, **kwargs)
        finally:
            timings[name] = round(timings[name] + time.perf_counter() - started, 3)

    # Why the judge can't be asked, known before any work for it
    judge_error = _unavailable(judge) if spec.judge else None
    try:
        mesh, views_used = timed("generate_s", _generate, spec, runtime, preset)
        pipeline = getattr(runtime, "pipeline_used", None) or preset.pipeline_type
    except InputError as err:
        # e.g. no object found in the image
        return {"error": f"invalid input: {err}"}
    except Exception as err:  # noqa: BLE001 - report any failure to the caller
        return _failed(err)

    refresh = False
    own: Optional[dict] = None  # how the own texture's export went, when it was exported for the judge
    candidates: list[bytes] = []  # the GLBs the judge compares: the own texture's, then each new one's
    if spec.judge and judge_error is None:
        _warm(judge)  # its container starts while the textures are made
        try:
            keep = getattr(runtime, "keep_layout", None)
            if callable(keep):
                keep(mesh)
            raw, triangles = timed("own_export_s", runtime.export, mesh, preset)
        except Exception as err:  # noqa: BLE001 - the judge goes without; the textures go on
            print("[forge3d] the final's own texture could not be exported for the judge:")
            traceback.print_exc()
            judge_error = f"the final's own texture could not be exported: {type(err).__name__}: {err}"
            refresh = needs_restart(err)
            traceback.clear_frames(err.__traceback__)  # it holds the mesh, and so its GPU memory
        else:
            notes = {**_export_notes(runtime), **_export_path(runtime)}
            refresh = _gpu_fault(notes)
            own = {"raw_bytes": len(raw), "triangles": triangles, **notes}
            candidates.append(raw)
    # Let go at once, so its GPU memory is free for the textures; the runtime keeps the shape latent that
    # retexture() samples them on
    del mesh

    textures: list[dict] = []
    errors: list[dict] = []
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
        if own is not None:
            candidates.append(raw)

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
        if spec.judge:
            if own is not None:
                result["own_texture"] = own
                answer, judge_error, fault = judge_textures(spec, judge, render, candidates, textures, timed)
                refresh = refresh or fault
                if answer is not None:
                    result["judge"] = answer
            if judge_error is not None:
                print(f"[forge3d] no judgement: {judge_error}")
                result["judge_error"] = judge_error
    else:
        result = {"error": f"generation failed: none of the {spec.count} textures was made: {errors[0]['error']}"}
    if refresh:
        result["refresh_worker"] = True
    return result


def judge_textures(
    spec: Job,
    judge: Judge,
    render: Optional[Render],
    candidates: Sequence[bytes],
    textures: Sequence[dict],
    timed: Callable[..., Any],
) -> tuple[Optional[dict], Optional[str], bool]:
    """
    The judge's answer about ``candidates`` (exported GLBs: the final's own texture, then the job's
    ``textures`` in order) as a textures job returns it: ``{"pick": 0 for the own texture or k for the k-th
    of textures, "verdicts": one per candidate, own first, "why", "model", "seconds", "order"}``. Each
    candidate is drawn by ``render`` (judgeviews' grid by default) and sent as a PNG, with the job's picture
    and prompt. The judge sees them in an order shuffled with the job's seed (``order``: the candidates as
    shown, the first under the letter K), as it favours some places over others. Returns the answer, or None
    and why there is none (a candidate that can't be drawn, the judge failing or naming no candidate), and
    whether drawing hit a GPU fault.
    """
    draw = render or draw_candidate
    grids: list[str] = []
    for number, raw in enumerate(candidates):
        try:
            grids.append(_base64_png(timed("render_s", draw, raw)))
        except Exception as err:  # noqa: BLE001 - the judge goes without; the textures are made
            which = "the final's own texture"
            if number:
                which = f"the texture of seed {textures[number - 1]['texture_seed']}"
            reason = f"{which} could not be drawn for the judge: {type(err).__name__}: {err}"
            print(f"[forge3d] {reason}")
            traceback.print_exc()
            return None, reason, needs_restart(err)
    order = list(range(len(grids)))
    random.Random(spec.seed).shuffle(order)  # what the judge worker's own "seed" would give
    request = {
        "picture_png": _base64_png(judge_picture(spec.image)),
        "candidates_png": grids,
        "prompt": spec.prompt,
        "order": order,
    }
    started = time.perf_counter()
    try:
        reply = timed("judge_s", judge, request)
    except Exception as err:  # noqa: BLE001 - a timeout, the judge's container failing: the textures stand
        return None, f"the judge failed: {type(err).__name__}: {err}", False
    waited = time.perf_counter() - started
    answer, reason = _judgement(reply, len(grids), order, getattr(judge, "model", None), waited)
    if answer is not None:
        print(f"[forge3d] judge: {json.dumps(answer)}")
    return answer, reason, False


def _judgement(
    answer: Any, count: int, order: list, model: Optional[str], waited: float
) -> tuple[Optional[dict], Optional[str]]:
    """
    The judge worker's result for ``count`` candidates as a textures job's ``judge``, or None and why not.
    ``seconds`` is the judge's own time (``waited``, the job's wait for it, when it doesn't say).
    """
    if not isinstance(answer, dict):
        return None, f"the judge answered with {type(answer).__name__}, not an object"
    if answer.get("error"):
        return None, f"the judge returned an error: {answer['error']}"
    if answer.get("parse_error"):
        # The judge then says best 0 (the own texture), but nothing was picked
        return None, f"the judge's reply named no pick: {answer['parse_error']}"
    pick = answer.get("best")
    if not isinstance(pick, int) or isinstance(pick, bool) or not 0 <= pick < count:
        return None, f"the judge's pick {json.dumps(pick, default=str)} is none of the {count} candidates"
    verdicts = answer.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != count:
        return None, f"the judge's verdicts {json.dumps(verdicts, default=str)[:80]} are not one per candidate"
    seconds = answer.get("seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        seconds = waited
    why = answer.get("why")
    return {
        "pick": pick,
        "verdicts": [verdict if verdict in VERDICTS else None for verdict in verdicts],
        "why": why if isinstance(why, str) else "",
        "model": answer.get("model") or model,
        "seconds": round(float(seconds), 2),
        "order": list(order),
    }, None


def draw_candidate(raw: bytes) -> Image.Image:
    """A candidate as the judge sees it: judgeviews' six views of its exported GLB (GPU code, so imported here)."""
    from .judgeviews import from_glb

    return from_glb(raw)


def judge_picture(image: Image.Image) -> Image.Image:
    """
    The job's picture as the judge looks at it: RGB, any transparency on white, at most JUDGE_PICTURE_SIDE on
    its longest side. The judge's own fit (judge_worker/judge.py) does exactly this, so it leaves it as it is.
    """
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, JUDGE_UNDERLAY + (255,)), rgba)
    image = image.convert("RGB")
    scale = min(1.0, JUDGE_PICTURE_SIDE / max(image.size))
    if scale < 1.0:
        # Rounded down as the judge rounds, so it finds nothing to shrink
        size = tuple(max(1, math.floor(length * scale + 1e-6)) for length in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _unavailable(judge: Optional[Judge]) -> Optional[str]:
    """Why ``judge`` can't be asked here, or None: no judge, or its own ``unavailable()`` says why."""
    if judge is None:
        return NO_JUDGE
    check = getattr(judge, "unavailable", None)
    if not callable(check):
        return None
    try:
        reason = check()
    except Exception as err:  # noqa: BLE001 - a judge that can't be checked isn't asked
        return f"the judge could not be checked: {type(err).__name__}: {err}"
    return str(reason) if reason else None


def _warm(judge: Judge) -> None:
    """Starts the judge up without waiting, if it can be (``warm()``): its cold start then overlaps the textures."""
    warm = getattr(judge, "warm", None)
    if not callable(warm):
        return
    try:
        warm()
    except Exception as err:  # noqa: BLE001 - only time is at stake: the judge starts when it is asked
        print(f"[forge3d] the judge was not warmed up: {type(err).__name__}: {err}")


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
