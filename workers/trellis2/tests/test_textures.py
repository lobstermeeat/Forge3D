"""
Texture options (mode "textures") with a fake runtime: the shape, the seeds, the keys, the failures, and
the judge ("judge": true) with fakes for the renderer and the judge: the final's own texture exported first,
what the judge is sent, its pick, its failures.
"""

import base64
import gc
import io
import json
import math
import pathlib
import random
import types
import weakref

import pytest
from PIL import Image

from forge3d_worker import inputs, service
from forge3d_worker.compress import CompressionError
from forge3d_worker.inputs import InputError, parse_job
from forge3d_worker.service import NO_JUDGE, TEXTURES_NEED_TRELLIS2, handle_job, judge_picture, texture_seed
from forge3d_worker.settings import CREDITS, MAX_TEXTURES, MODES, PRESETS, TEXTURE_COUNT, TEXTURE_SEED_STEP

WORKERS = pathlib.Path(__file__).resolve().parents[2]


def png(size=(64, 48)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 80, 40)).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


PICTURE = png()


def payload(**extra) -> dict:
    return {"image_base64": PICTURE, "mode": "textures", "seed": 1234, "request_id": "gen_42", **extra}


def textures_job(**extra) -> dict:
    return {"id": "rp-1", "input": payload(**extra)}


class Mesh:
    """What generate() and retexture() return: the noise its texture came from ("final": the generation's)."""

    def __init__(self, texture) -> None:
        self.texture = texture


class FakeRuntime:
    """
    Trellis2Runtime as a textures job drives it: generate() makes the shape, retexture(seed=…) a new texture
    for it, export() a GLB of either. ``fail`` maps a step ("generate", or (step, texture seed)) to what it
    raises. retexture() notes the meshes still alive, which should be none: one model on the GPU at a time.
    """

    def __init__(self, fail=None) -> None:
        self.fail = dict(fail or {})
        self.calls = []
        self.meshes = []  # a weak reference to each mesh handed out

    def _mesh(self, texture) -> Mesh:
        mesh = Mesh(texture)
        self.meshes.append(weakref.ref(mesh))
        return mesh

    def alive(self) -> list:
        gc.collect()
        return [ref().texture for ref in self.meshes if ref() is not None]

    def generate(self, image, preset, seed, views=()):
        self.calls.append(("generate", preset, seed, image.size, [(v.azimuth, v.weight) for v in views]))
        if "generate" in self.fail:
            raise self.fail["generate"]
        return self._mesh("final")

    def retexture(self, *, seed):  # no views: texture options come from the picture alone, as in the rolls
        self.calls.append(("retexture", seed, self.alive()))
        if ("retexture", seed) in self.fail:
            raise self.fail[("retexture", seed)]
        return self._mesh(seed)

    def export(self, mesh, preset):
        self.calls.append(("export", mesh.texture, preset))
        if ("export", mesh.texture) in self.fail:
            raise self.fail[("export", mesh.texture)]
        return raw_glb(mesh.texture), 99_000


def raw_glb(texture) -> bytes:
    return f"glb of texture {texture}".encode() * 3


class Packer:
    """Stands in for gltfpack; fails for the textures in ``fail``."""

    def __init__(self, fail=()) -> None:
        self.fail = set(fail)
        self.calls = []

    def __call__(self, raw: bytes, texture_limit: int) -> bytes:
        self.calls.append((raw, texture_limit))
        if any(raw == raw_glb(texture) for texture in self.fail):
            raise CompressionError("gltfpack failed: Error loading in.glb: data is corrupt")
        return raw[:24]


class FakeStorage:
    def __init__(self, fail=None) -> None:
        self.fail = dict(fail or {})
        self.saved = {}

    def put(self, key, data, content_type):
        if key in self.fail:
            raise self.fail[key]
        self.saved[key] = (data, content_type)
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def key(number: int) -> str:
    return f"ai/gen_42/final-1234-texture-{number}.glb"


def test_a_textures_job_makes_the_finals_shape_once_then_one_texture_per_seed():
    runtime, storage, packer = FakeRuntime(), FakeStorage(), Packer()
    out = handle_job(textures_job(count=3), runtime, storage, packer)

    assert "error" not in out
    final = PRESETS["final"]
    # The final's shape, as a final job makes it: the final's preset, the seed and the picture
    assert runtime.calls[0] == ("generate", final, 1234, (64, 48), [])
    # Then a texture per seed, each exported with the final's settings. The generation's own texture (the
    # final's, which the creator has) is never exported, and every mesh is let go before the next is made
    assert runtime.calls[1:] == [
        ("retexture", 2234, []),
        ("export", 2234, final),
        ("retexture", 3234, []),
        ("export", 3234, final),
        ("retexture", 4234, []),
        ("export", 4234, final),
    ]
    # Packed and stored as a final is: its texture size, glTF binaries, beside the final's model
    assert [(raw, limit) for raw, limit in packer.calls] == [(raw_glb(s), 2048) for s in (2234, 3234, 4234)]
    assert storage.saved == {key(k): (raw_glb(1234 + 1000 * k)[:24], "model/gltf-binary") for k in (1, 2, 3)}
    assert out["textures"] == [
        {
            "texture_seed": seed,
            "glb": {"key": key(k), "url": f"https://assets.example.com/{key(k)}"},
            "bytes": 24,
            "raw_bytes": len(raw_glb(seed)),
            "triangles": 99_000,
        }
        for k, seed in ((1, 2234), (2, 3234), (3, 4234))
    ]


def test_the_result_mirrors_a_finals_fields():
    out = handle_job(textures_job(), FakeRuntime(), FakeStorage(), Packer())
    assert set(out) == {"request_id", "mode", "seed", "textures", "pipeline", "views_used", "timings", "credits"}
    assert (out["request_id"], out["mode"], out["seed"]) == ("gen_42", "textures", 1234)
    assert out["pipeline"] == "1024_cascade" and out["views_used"] == 0 and out["credits"] == list(CREDITS)
    assert set(out["timings"]) == {"generate_s", "retexture_s", "export_s", "pack_s", "upload_s"}
    assert all(set(texture) == {"texture_seed", "glb", "bytes", "raw_bytes", "triangles"} for texture in out["textures"])
    assert "glb" not in out  # the final's own texture isn't sent again
    assert json.loads(json.dumps(out)) == out  # what the job API passes on


def test_the_texture_seeds_are_the_rolls():
    assert TEXTURE_SEED_STEP == 1000
    assert [texture_seed(1234, k) for k in (1, 2, 3, 4)] == [2234, 3234, 4234, 5234]
    # At the top of the seed range too: torch.manual_seed takes seeds well past 2^31
    top = 2**31 - 1
    runtime = FakeRuntime()
    out = handle_job(textures_job(seed=top, count=MAX_TEXTURES), runtime, FakeStorage(), Packer())
    assert [texture["texture_seed"] for texture in out["textures"]] == [top + 1000 * k for k in (1, 2, 3, 4)]
    assert [call[1] for call in runtime.calls if call[0] == "retexture"] == [top + 1000 * k for k in (1, 2, 3, 4)]


@pytest.mark.parametrize("count, made", [("absent", 3), (None, 3), (1, 1), (2, 2), (4, 4)])
def test_three_textures_unless_the_job_asks_for_one_to_four(count, made):
    assert TEXTURE_COUNT == 3 and MAX_TEXTURES == 4
    extra = {} if count == "absent" else {"count": count}
    runtime = FakeRuntime()
    out = handle_job(textures_job(**extra), runtime, FakeStorage(), Packer())
    assert [texture["texture_seed"] for texture in out["textures"]] == [1234 + 1000 * k for k in range(1, made + 1)]
    assert [call[0] for call in runtime.calls].count("retexture") == made


@pytest.mark.parametrize("count", [0, 5, -1, 2.5, 3.0, "3", True, [3]])
def test_other_counts_are_refused_before_anything_runs(count):
    runtime = FakeRuntime()
    out = handle_job(textures_job(count=count), runtime, FakeStorage(), Packer())
    assert out == {"error": "invalid input: count must be an integer from 1 to 4"}
    assert runtime.calls == []


@pytest.mark.parametrize("mode", ["preview", "final"])
def test_previews_and_finals_ignore_count(mode):
    job = parse_job({"image_base64": PICTURE, "mode": mode, "count": 99}, fallback_id="job")
    assert job.mode == mode and job.count == 0


def test_textures_is_a_mode():
    assert MODES == ("preview", "final", "textures")
    assert parse_job(payload(), fallback_id="job").count == 3
    with pytest.raises(InputError, match=r"^mode must be one of \['final', 'preview', 'textures'\]$"):
        parse_job(payload(mode="texture"), fallback_id="job")


@pytest.mark.parametrize("seed", ["absent", None])
def test_a_textures_job_needs_the_finals_seed(seed):
    """A seed picked at random, as a preview's is, would make another shape than the final's."""
    job = textures_job()
    if seed == "absent":
        del job["input"]["seed"]
    else:
        job["input"]["seed"] = seed
    runtime = FakeRuntime()
    out = handle_job(job, runtime, FakeStorage(), Packer())
    assert out == {"error": "invalid input: seed is required for textures: send the final's seed"}
    assert runtime.calls == []


def test_the_seed_is_checked_as_a_finals_is():
    assert parse_job(payload(seed=0), fallback_id="job").seed == 0
    for seed in (-1, 2**31, True, "7"):
        with pytest.raises(InputError, match="^seed must be an integer between 0 and 2"):
            parse_job(payload(seed=seed), fallback_id="job")


def test_textures_are_stored_beside_the_finals_model():
    final = parse_job(payload(mode="final"), fallback_id="job")
    textures = parse_job(payload(), fallback_id="job")
    assert final.output_key == "ai/gen_42/final-1234.glb"
    assert [textures.texture_key(k) for k in (1, 2, 4)] == [key(1), key(2), key(4)]
    # Without a request id, the job's own id, as for a final
    textures = parse_job({k: v for k, v in payload().items() if k != "request_id"}, fallback_id="fc-01K6ABC")
    assert textures.texture_key(3) == "ai/fc-01K6ABC/final-1234-texture-3.glb"


@pytest.mark.parametrize("step", ["retexture", "export", "pack", "upload"])
def test_one_texture_failing_does_not_lose_the_others(step, capsys):
    runtime, storage, packer = FakeRuntime(), FakeStorage(), Packer()
    if step == "retexture":
        runtime = FakeRuntime(fail={("retexture", 3234): ValueError("Invalid pipeline type: 2048")})
        error = "ValueError: Invalid pipeline type: 2048"
    elif step == "export":
        runtime = FakeRuntime(fail={("export", 3234): IndexError("index 7 is out of bounds")})
        error = "IndexError: index 7 is out of bounds"
    elif step == "pack":
        packer = Packer(fail={3234})
        error = "CompressionError: gltfpack failed: Error loading in.glb: data is corrupt"
    else:
        storage = FakeStorage(fail={key(2): ConnectionError("R2 unreachable")})
        error = "ConnectionError: R2 unreachable"

    out = handle_job(textures_job(), runtime, storage, packer)

    assert "error" not in out and "refresh_worker" not in out  # nothing went wrong with the GPU
    assert [texture["texture_seed"] for texture in out["textures"]] == [2234, 4234]
    assert [texture["glb"]["key"] for texture in out["textures"]] == [key(1), key(3)]
    assert out["texture_errors"] == [{"texture_seed": 3234, "error": error}]
    assert list(storage.saved) == [key(1), key(3)]
    # The failed texture's mesh was let go before the next one was made, though the error lives on (the fake
    # raises the same error object each time, and that keeps its traceback)
    assert [call[2] for call in runtime.calls if call[0] == "retexture"] == [[], [], []]
    assert "[forge3d] texture 2 of 3 (seed 3234) failed:" in capsys.readouterr().out


def test_a_gpu_fault_in_one_texture_keeps_the_others_and_replaces_the_worker():
    fault = RuntimeError("CUDA error: an illegal memory access was encountered")
    out = handle_job(textures_job(), FakeRuntime(fail={("retexture", 3234): fault}), FakeStorage(), Packer())
    assert [texture["texture_seed"] for texture in out["textures"]] == [2234, 4234]
    assert out["texture_errors"] == [{"texture_seed": 3234, "error": f"RuntimeError: {fault}"}]
    assert out["refresh_worker"] is True  # modal_app's run_job takes it out of the result


@pytest.mark.parametrize(
    "error, refresh",
    [
        (RuntimeError("no shape to retexture: generate() one first, or pass a latent"), False),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
    ],
)
def test_when_no_texture_is_made_the_job_fails(error, refresh):
    runtime = FakeRuntime(fail={("retexture", seed): error for seed in (2234, 3234, 4234)})
    out = handle_job(textures_job(), runtime, FakeStorage(), Packer())
    expected = {"error": f"generation failed: none of the 3 textures was made: RuntimeError: {error}"}
    assert out == ({**expected, "refresh_worker": True} if refresh else expected)
    assert [call[0] for call in runtime.calls] == ["generate", "retexture", "retexture", "retexture"]


def test_a_shape_that_fails_fails_the_job_as_a_final_does():
    runtime = FakeRuntime(fail={"generate": RuntimeError("CUDA out of memory")})
    out = handle_job(textures_job(), runtime, FakeStorage(), Packer())
    assert out == {"error": "generation failed: RuntimeError: CUDA out of memory", "refresh_worker": True}
    assert [call[0] for call in runtime.calls] == ["generate"]

    nothing = InputError("no object found in the image: use one object on a plain background")
    out = handle_job(textures_job(), FakeRuntime(fail={"generate": nothing}), FakeStorage(), Packer())
    assert out == {"error": f"invalid input: {nothing}"}


def test_a_runtime_that_cannot_retexture_is_refused_before_anything_runs():
    class Pixal3DLike:
        """The Pixal3D worker's runtime as production's handle_job sees it: generate and export only."""

        def __init__(self) -> None:
            self.calls = []

        def generate(self, image, preset, seed):
            self.calls.append("generate")
            return "mesh"

        def export(self, mesh, preset):
            self.calls.append("export")
            return b"glb", 1

    runtime, storage = Pixal3DLike(), FakeStorage()
    out = handle_job(textures_job(), runtime, storage, Packer())
    assert out == {"error": TEXTURES_NEED_TRELLIS2} == {"error": "texture options need TRELLIS.2 finals"}
    assert runtime.calls == [] and storage.saved == {}


def view(azimuth, **extra) -> dict:
    return {"image_base64": PICTURE, "azimuth": azimuth, **extra}


def test_the_shape_gets_the_views_the_final_got():
    views = [view(90), view(180, weight=2)]
    final, textures = FakeRuntime(), FakeRuntime()
    handle_job({"id": "f", "input": payload(mode="final", views=views)}, final, FakeStorage(), Packer())
    out = handle_job(textures_job(views=views), textures, FakeStorage(), Packer())
    assert textures.calls[0] == final.calls[0] == ("generate", PRESETS["final"], 1234, (64, 48), [(90, 1.0), (180, 2.0)])
    assert out["views_used"] == 2
    # The textures themselves come from the picture alone, as the rolls' did: retexture() gets a seed only
    # (FakeRuntime.retexture takes nothing else, so views would have failed every texture)
    assert len(out["textures"]) == 3 and "texture_errors" not in out


def test_the_shape_reports_its_pipeline():
    class FellBack(FakeRuntime):
        def generate(self, image, preset, seed, views=()):
            self.pipeline_used = "512"  # the cascade ran out of GPU memory even in low-VRAM mode
            return super().generate(image, preset, seed, views)

    out = handle_job(textures_job(), FellBack(), FakeStorage(), Packer())
    assert out["pipeline"] == "512" and len(out["textures"]) == 3


class Projecting(FakeRuntime):
    """Notes what each export did, as Trellis2Runtime does: the picture's projection and the floaters."""

    def __init__(self, notes, **options) -> None:
        super().__init__(**options)
        self.notes = notes  # texture seed -> (last_projection, last_cleanup)

    def export(self, mesh, preset):
        self.last_projection, self.last_cleanup = self.notes.get(mesh.texture, (None, None))
        return super().export(mesh, preset)


def test_each_texture_says_whether_the_picture_was_painted_on():
    applied = {"applied": True, "reason": "applied", "iou": 0.97, "seconds": 0.4}
    missed = {"applied": False, "reason": "the silhouettes don't match (IoU 0.91)", "iou": 0.91}
    floaters = {"pieces": 2, "dropped": 1, "faces_dropped": 40, "area_dropped": 0.01, "floaters": [], "seconds": 0.2}
    runtime = Projecting({2234: (applied, None), 3234: (missed, floaters), 4234: (None, None)})
    out = handle_job(textures_job(), runtime, FakeStorage(), Packer())
    first, second, third = out["textures"]
    assert first["projection"] == applied and "floaters" not in first
    assert second["projection"] == missed and second["floaters"] == floaters
    assert "projection" not in third and "floaters" not in third
    assert "refresh_worker" not in out


def test_a_gpu_fault_while_projecting_keeps_the_texture_but_replaces_the_worker():
    fault = {"applied": False, "reason": "error: RuntimeError: CUDA error: an illegal memory access", "gpu_fault": True}
    applied = {"applied": True, "reason": "applied"}
    runtime = Projecting({2234: (applied, None), 3234: (fault, None), 4234: (applied, None)})
    out = handle_job(textures_job(), runtime, FakeStorage(), Packer())
    assert [texture["projection"] for texture in out["textures"]] == [applied, fault, applied]
    assert out["refresh_worker"] is True

    # Noted as soon as the export is done: the texture failing afterwards doesn't hide it
    runtime = Projecting({2234: (applied, None), 3234: (fault, None), 4234: (applied, None)})
    storage = FakeStorage(fail={key(2): ConnectionError("R2 unreachable")})
    out = handle_job(textures_job(), runtime, storage, Packer())
    assert out["texture_errors"] == [{"texture_seed": 3234, "error": "ConnectionError: R2 unreachable"}]
    assert out["refresh_worker"] is True


class Rebaking(FakeRuntime):
    """Notes how each export ran, as Trellis2Runtime does: to_glb in full for the first texture, then rebakes."""

    def export(self, mesh, preset):
        if mesh.texture in ("final", 2234):
            self.last_export = {"path": "to_glb", "seconds": 14.2, "captured": True}
        else:
            self.last_export = {"path": "rebake", "seconds": 1.3}
        return super().export(mesh, preset)


def test_each_texture_says_how_its_export_ran():
    out = handle_job(textures_job(), Rebaking(), FakeStorage(), Packer())
    assert [texture["export"] for texture in out["textures"]] == [
        {"path": "to_glb", "seconds": 14.2, "captured": True},
        {"path": "rebake", "seconds": 1.3},
        {"path": "rebake", "seconds": 1.3},
    ]
    assert json.loads(json.dumps(out)) == out
    # A final's result says nothing of it: a final always runs to_glb in full, as it always has
    final = handle_job({"id": "f", "input": payload(mode="final")}, Rebaking(), FakeStorage(), Packer())
    assert "error" not in final and "export" not in final


def test_timings_sum_each_step_over_the_textures(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(service, "time", types.SimpleNamespace(perf_counter=lambda: now[0]))

    def taking(seconds, work):
        def timed(*args, **kwargs):
            now[0] += seconds
            return work(*args, **kwargs)

        return timed

    runtime = FakeRuntime(fail={("export", 3234): RuntimeError("[CuMesh] remesh failed")})
    runtime.generate = taking(30.0, runtime.generate)
    runtime.retexture = taking(10.0, runtime.retexture)
    runtime.export = taking(20.0, runtime.export)
    storage = FakeStorage()
    storage.put = taking(0.25, storage.put)

    out = handle_job(textures_job(), runtime, storage, taking(1.5, Packer()))

    # A failed step counts as well (the second export): the GPU was busy all the same
    assert out["timings"] == {"generate_s": 30.0, "retexture_s": 30.0, "export_s": 60.0, "pack_s": 3.0, "upload_s": 0.5}
    assert sum(out["timings"].values()) == pytest.approx(now[0])  # the job's whole time


# --- The judge ("judge": true): which texture is best ---------------------------------------------------------

# Each candidate's grid, as Draw makes it, is a small image in its texture's tint
TINTS = {
    "final": (230, 40, 40),
    "2234": (40, 200, 60),
    "3234": (50, 70, 220),
    "4234": (230, 210, 40),
    "5234": (40, 200, 210),
}
PROMPT = "a red lamp"


def judged_job(**extra) -> dict:
    return textures_job(judge=True, prompt=PROMPT, **extra)


def texture_of(raw: bytes) -> str:
    """Which texture a fake GLB is of: "final" (the generation's own) or its seed."""
    return raw.decode().split("glb of texture ")[1]


def tint_of(encoded: str) -> str:
    """Which texture a base64 PNG Draw made shows."""
    colour = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB").getpixel((0, 0))
    return next(name for name, tint in TINTS.items() if tint == colour)


class LayoutRuntime(FakeRuntime):
    """
    FakeRuntime with Trellis2Runtime's export paths: a mesh marked with its shape (by keep_layout, as the
    generation's own texture is, or by retexture) runs to_glb in full and keeps the layout when none is kept
    yet, and rebakes on it after that; an unmarked mesh runs to_glb, unwatched. A new shape drops the layout.
    """

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.layout = False

    def generate(self, image, preset, seed, views=()):
        self.layout = False
        return super().generate(image, preset, seed, views)

    def keep_layout(self, mesh) -> bool:
        self.calls.append(("keep_layout", mesh.texture))
        mesh.shape = True
        return True

    def retexture(self, *, seed):
        mesh = super().retexture(seed=seed)
        mesh.shape = True
        return mesh

    def export(self, mesh, preset):
        self.last_export = None
        exported = super().export(mesh, preset)
        if not getattr(mesh, "shape", False):
            self.last_export = {"path": "to_glb", "seconds": 20.0}
        elif self.layout:
            self.last_export = {"path": "rebake", "seconds": 3.0}
        else:
            self.layout = True
            self.last_export = {"path": "to_glb", "seconds": 20.0, "captured": True}
        return exported


class Draw:
    """Stands in for judgeviews: a candidate's grid is a small image in its texture's tint (grey for others)."""

    def __init__(self, fail=None) -> None:
        self.fail = dict(fail or {})  # texture -> what drawing it raises
        self.drawn = []

    def __call__(self, raw: bytes) -> Image.Image:
        texture = texture_of(raw)
        self.drawn.append(texture)
        if texture in self.fail:
            raise self.fail[texture]
        return Image.new("RGB", (12, 8), TINTS.get(texture, (128, 128, 128)))


class FakeJudge:
    """
    Stands in for Judge8B: it likes ``favourite`` best (a texture: "final" or a seed) and calls the others
    "edits", or gives ``answer``, or raises ``fail``. ``missing`` is what unavailable() says. Notes each
    request, and its warm-ups (in ``log`` too, when given one).
    """

    model = "8b"

    def __init__(self, favourite="3234", answer=None, fail=None, missing=None, warm_fail=None, log=None) -> None:
        self.favourite, self.answer, self.fail, self.missing, self.warm_fail = favourite, answer, fail, missing, warm_fail
        self.log = log if log is not None else []
        self.requests = []
        self.warmed = 0

    def unavailable(self):
        return self.missing

    def warm(self) -> None:
        self.warmed += 1
        self.log.append(("warm",))
        if self.warm_fail is not None:
            raise self.warm_fail

    def __call__(self, request: dict) -> dict:
        self.requests.append(request)
        self.log.append(("judge",))
        if self.fail is not None:
            raise self.fail
        if self.answer is not None:
            return self.answer
        seen = [tint_of(grid) for grid in request["candidates_png"]]
        best = seen.index(self.favourite)
        return {
            "verdicts": ["publish" if texture == self.favourite else "edits" for texture in seen],
            "best": best,
            "why": f"{self.favourite} has the cleanest back",
            "raw": "{…}",
            "seconds": 8.6,
            "letters": ["K", "L", "M", "N", "P"][: len(seen)],
            "problems": [None] * len(seen),
            "model": "8b",
        }


@pytest.fixture
def still_clock(monkeypatch):
    """No time passes, so two jobs' timings compare equal."""
    monkeypatch.setattr(service, "time", types.SimpleNamespace(perf_counter=lambda: 0.0))


def seeded_order(count: int, seed: int = 1234) -> list:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


@pytest.fixture
def judge_worker(monkeypatch):
    """The judge worker's own package (workers/judge), for checks across the two."""
    monkeypatch.syspath_prepend(str(WORKERS / "judge"))
    from judge_worker import judge, parse, prompt
    from judge_worker import service as judge_service

    return types.SimpleNamespace(judge=judge, parse=parse, prompt=prompt, service=judge_service)


@pytest.mark.parametrize("judge", ["absent", None, False])
def test_without_the_judge_a_textures_job_is_as_before(judge, still_clock):
    # Without the judge, a prompt isn't read: one that isn't a string goes unchecked, as before
    extra = {} if judge == "absent" else {"judge": judge, "prompt": 42}
    before = LayoutRuntime()
    expected = handle_job(textures_job(), before, FakeStorage(), Packer())
    runtime, storage, packer, asked, draw = LayoutRuntime(), FakeStorage(), Packer(), FakeJudge(), Draw()

    out = handle_job(textures_job(**extra), runtime, storage, packer, judge=asked, render=draw)

    assert out == expected and runtime.calls == before.calls
    assert [call[0] for call in runtime.calls] == ["generate"] + ["retexture", "export"] * 3
    # The first new texture runs to_glb in full and keeps the layout, as it always has
    assert [texture["export"]["path"] for texture in out["textures"]] == ["to_glb", "rebake", "rebake"]
    assert set(out["timings"]) == {"generate_s", "retexture_s", "export_s", "pack_s", "upload_s"}
    assert not {"judge", "judge_error", "own_texture"} & set(out)
    assert asked.requests == [] and asked.warmed == 0 and draw.drawn == []


def test_with_the_judge_the_finals_own_texture_is_exported_first_and_keeps_the_layout(still_clock):
    runtime, storage, packer = LayoutRuntime(), FakeStorage(), Packer()
    out = handle_job(judged_job(), runtime, storage, packer, judge=FakeJudge(), render=Draw())

    final = PRESETS["final"]
    assert runtime.calls == [
        ("generate", final, 1234, (64, 48), []),
        # Marked with its shape, then exported as the final was: to_glb in full, its layout kept
        ("keep_layout", "final"),
        ("export", "final", final),
        # Let go before the first new texture is sampled, as each texture is before the next
        ("retexture", 2234, []),
        ("export", 2234, final),
        ("retexture", 3234, []),
        ("export", 3234, final),
        ("retexture", 4234, []),
        ("export", 4234, final),
    ]
    assert out["own_texture"] == {
        "raw_bytes": len(raw_glb("final")),
        "triangles": 99_000,
        "export": {"path": "to_glb", "seconds": 20.0, "captured": True},
    }
    # So every new texture rebakes
    assert [texture["export"] for texture in out["textures"]] == [{"path": "rebake", "seconds": 3.0}] * 3
    # The own texture is never packed, stored or returned: the final has it
    assert [raw for raw, _ in packer.calls] == [raw_glb(seed) for seed in (2234, 3234, 4234)]
    assert list(storage.saved) == [key(1), key(2), key(3)]
    assert [texture["texture_seed"] for texture in out["textures"]] == [2234, 3234, 4234]
    assert set(out["timings"]) == {
        "generate_s", "retexture_s", "export_s", "pack_s", "upload_s", "own_export_s", "render_s", "judge_s"
    }
    assert json.loads(json.dumps(out)) == out  # what the job API passes on


@pytest.mark.parametrize("favourite, pick", [("final", 0), ("2234", 1), ("3234", 2), ("4234", 3)])
def test_the_judge_sees_every_candidate_in_a_seeded_order_and_its_pick_names_one(favourite, pick, still_clock):
    judge, draw = FakeJudge(favourite), Draw()
    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=draw)

    # Every candidate drawn from its exported GLB: the own texture first, then the new ones in order
    assert draw.drawn == ["final", "2234", "3234", "4234"]
    (request,) = judge.requests
    assert set(request) == {"picture_png", "candidates_png", "prompt", "order"}
    assert [tint_of(grid) for grid in request["candidates_png"]] == ["final", "2234", "3234", "4234"]
    # Shown in an order shuffled with the job's seed (the judge favours some places over others)
    order = seeded_order(4)
    assert request["order"] == order and order != [0, 1, 2, 3]
    assert request["prompt"] == PROMPT
    # 0 is the final's own texture, k the k-th of "textures"
    assert out["judge"] == {
        "pick": pick,
        "verdicts": ["publish" if candidate == pick else "edits" for candidate in range(4)],
        "why": f"{favourite} has the cleanest back",
        "model": "8b",
        "seconds": 8.6,
        "order": order,
    }
    if pick:
        assert out["textures"][pick - 1]["texture_seed"] == int(favourite)
    assert "judge_error" not in out


def test_the_order_is_the_jobs_own():
    """The same final shows the judge its candidates in the same order every time; another seed may not."""
    orders = {}
    for seed in (1234, 1234, 7, 99):
        judge = FakeJudge(answer={"verdicts": ["edits"] * 4, "best": 0, "why": "", "seconds": 8.0})
        out = handle_job(judged_job(seed=seed), LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=Draw())
        orders.setdefault(seed, []).append(judge.requests[0]["order"])
        assert out["judge"]["order"] == judge.requests[0]["order"]
    assert orders[1234][0] == orders[1234][1] == seeded_order(4, 1234)
    assert orders[7] == [seeded_order(4, 7)] and orders[99] == [seeded_order(4, 99)]
    assert len({tuple(order[0]) for order in orders.values()}) > 1


def test_the_order_is_what_the_judges_own_seed_gives(judge_worker):
    for seed, count in ((1234, 4), (7, 2), (2**31 - 1, 5)):
        assert seeded_order(count, seed) == judge_worker.service.shuffled(count, seed)


def test_a_texture_that_failed_is_left_out_and_the_pick_counts_the_textures_returned(still_clock):
    runtime = LayoutRuntime(fail={("export", 3234): IndexError("index 7 is out of bounds")})
    judge = FakeJudge("4234")
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=Draw())

    assert out["texture_errors"] == [{"texture_seed": 3234, "error": "IndexError: index 7 is out of bounds"}]
    assert [tint_of(grid) for grid in judge.requests[0]["candidates_png"]] == ["final", "2234", "4234"]
    assert judge.requests[0]["order"] == seeded_order(3)
    assert out["judge"]["pick"] == 2 and out["textures"][1]["texture_seed"] == 4234
    assert out["judge"]["verdicts"] == ["edits", "edits", "publish"]


def test_a_textures_job_that_makes_no_texture_fails_as_before_and_asks_no_judge():
    error = RuntimeError("no shape to retexture: generate() one first, or pass a latent")
    runtime = LayoutRuntime(fail={("retexture", seed): error for seed in (2234, 3234, 4234)})
    judge, draw = FakeJudge(), Draw()
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=draw)
    assert out == {"error": f"generation failed: none of the 3 textures was made: RuntimeError: {error}"}
    assert judge.requests == [] and draw.drawn == []


# A judge's failure: what the fake does, and the job's judge_error
JUDGE_FAILURES = {
    "timeout": (
        {"fail": TimeoutError("the judge didn't answer within 180 s")},
        "the judge failed: TimeoutError: the judge didn't answer within 180 s",
    ),
    "error": (
        {"answer": {"error": "judge failed: RuntimeError: CUDA error: out of memory"}},
        "the judge returned an error: judge failed: RuntimeError: CUDA error: out of memory",
    ),
    "no pick": (
        # The judge says best 0 when its reply names none, and flags it
        {"answer": {"verdicts": [None] * 4, "best": 0, "why": "", "parse_error": "no JSON object or pick in the reply"}},
        "the judge's reply named no pick: no JSON object or pick in the reply",
    ),
    "a pick out of range": (
        {"answer": {"verdicts": ["edits"] * 4, "best": 4, "why": ""}},
        "the judge's pick 4 is none of the 4 candidates",
    ),
    "a pick that is no number": (
        {"answer": {"verdicts": ["edits"] * 4, "best": "L", "why": ""}},
        'the judge\'s pick "L" is none of the 4 candidates',
    ),
    "verdicts missing": (
        {"answer": {"verdicts": ["edits"] * 3, "best": 1, "why": ""}},
        'the judge\'s verdicts ["edits", "edits", "edits"] are not one per candidate',
    ),
    "not an object": ({"answer": ["K"]}, "the judge answered with list, not an object"),
}


@pytest.mark.parametrize("failure", JUDGE_FAILURES)
def test_a_judge_that_fails_keeps_the_textures(failure, still_clock, capsys):
    options, reason = JUDGE_FAILURES[failure]
    expected = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=FakeJudge(), render=Draw())
    storage = FakeStorage()
    out = handle_job(judged_job(), LayoutRuntime(), storage, Packer(), judge=FakeJudge(**options), render=Draw())

    assert out["judge_error"] == reason and "judge" not in out
    assert {k: v for k, v in out.items() if k != "judge_error"} == {k: v for k, v in expected.items() if k != "judge"}
    assert list(storage.saved) == [key(1), key(2), key(3)] and "refresh_worker" not in out
    assert f"[forge3d] no judgement: {reason}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "texture, which",
    [("final", "the final's own texture"), ("3234", "the texture of seed 3234")],
)
def test_a_candidate_that_cannot_be_drawn_keeps_the_textures_and_asks_no_judge(texture, which, still_clock):
    judge = FakeJudge()
    draw = Draw(fail={texture: ValueError("the mesh has no base colour texture or UVs")})
    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=draw)

    assert out["judge_error"] == f"{which} could not be drawn for the judge: ValueError: the mesh has no base colour texture or UVs"
    assert "judge" not in out and judge.requests == [] and draw.drawn[-1] == texture
    assert len(out["textures"]) == 3 and "own_texture" in out and "refresh_worker" not in out


def test_a_gpu_fault_while_drawing_keeps_the_textures_and_replaces_the_worker(still_clock):
    fault = RuntimeError("CUDA error: an illegal memory access was encountered")
    out = handle_job(
        judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=FakeJudge(), render=Draw(fail={"2234": fault})
    )
    assert out["judge_error"] == f"the texture of seed 2234 could not be drawn for the judge: RuntimeError: {fault}"
    assert len(out["textures"]) == 3 and out["refresh_worker"] is True


@pytest.mark.parametrize(
    "error, refresh",
    [(RuntimeError("[CuMesh] remesh failed"), False), (RuntimeError("CUDA error: an illegal memory access"), True)],
)
def test_when_the_own_texture_cannot_be_exported_the_textures_are_made_as_without_the_judge(error, refresh, still_clock):
    runtime = LayoutRuntime(fail={("export", "final"): error})
    judge, draw = FakeJudge(), Draw()
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=draw)

    assert out["judge_error"] == f"the final's own texture could not be exported: RuntimeError: {error}"
    assert "judge" not in out and "own_texture" not in out
    assert judge.requests == [] and draw.drawn == [] and judge.warmed == 1
    # The first new texture then keeps the layout, as without the judge, and the mesh was let go first
    assert [texture["export"]["path"] for texture in out["textures"]] == ["to_glb", "rebake", "rebake"]
    assert [call[2] for call in runtime.calls if call[0] == "retexture"] == [[], [], []]
    assert out.get("refresh_worker", False) is refresh


class Unreadable(FakeJudge):
    def unavailable(self):
        raise OSError("Read-only file system: '/models'")


@pytest.mark.parametrize(
    "judge, reason",
    [
        (None, NO_JUDGE),
        (
            FakeJudge(missing="the judge's weights are missing: modal run workers/modal_app.py::download_models --which judge8b"),
            "the judge's weights are missing: modal run workers/modal_app.py::download_models --which judge8b",
        ),
        (Unreadable(), "the judge could not be checked: OSError: Read-only file system: '/models'"),
    ],
    ids=["no judge", "weights missing", "unreadable"],
)
def test_a_judge_that_cannot_run_here_costs_the_job_nothing(judge, reason, still_clock):
    runtime, draw = LayoutRuntime(), Draw()
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=draw)

    assert NO_JUDGE == "no judge in this worker"
    assert out["judge_error"] == reason and not {"judge", "own_texture"} & set(out)
    # None of the judge's part ran: no own export, no warm-up, nothing drawn, nothing asked
    assert [call[0] for call in runtime.calls] == ["generate"] + ["retexture", "export"] * 3
    assert [texture["export"]["path"] for texture in out["textures"]] == ["to_glb", "rebake", "rebake"]
    assert draw.drawn == [] and (judge is None or (judge.requests == [] and judge.warmed == 0))
    assert {name: out["timings"][name] for name in ("own_export_s", "render_s", "judge_s")} == dict.fromkeys(
        ("own_export_s", "render_s", "judge_s"), 0.0
    )


def test_the_judge_starts_up_once_the_shape_is_made(still_clock, capsys):
    runtime = LayoutRuntime()
    judge = FakeJudge(log=runtime.calls)
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=Draw())
    # Warmed as soon as the shape is there, so its container starts while the textures are made
    assert [call[0] for call in runtime.calls] == (
        ["generate", "warm", "keep_layout", "export"] + ["retexture", "export"] * 3 + ["judge"]
    )
    assert judge.warmed == 1 and "judge" in out

    # A shape that fails starts nothing
    runtime = LayoutRuntime(fail={"generate": RuntimeError("CUDA out of memory")})
    judge = FakeJudge()
    out = handle_job(judged_job(), runtime, FakeStorage(), Packer(), judge=judge, render=Draw())
    assert out["error"] == "generation failed: RuntimeError: CUDA out of memory" and judge.warmed == 0

    # A warm-up that fails changes nothing but the time the judge takes
    judge = FakeJudge(warm_fail=ConnectionError("Modal is unreachable"))
    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=Draw())
    assert out["judge"]["pick"] == 2
    assert "[forge3d] the judge was not warmed up: ConnectionError: Modal is unreachable" in capsys.readouterr().out


def test_what_the_judge_says_is_passed_on_as_the_contract_has_it(still_clock):
    answer = {"verdicts": ["publish", None, "maybe", "reject"], "best": 1, "seconds": 9, "notes": ["no verdict for L"]}
    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=FakeJudge(answer=answer), render=Draw())
    # Unknown verdicts become null; no "why" is ""; "model" is the judge's own when the answer doesn't say
    assert out["judge"] == {
        "pick": 1,
        "verdicts": ["publish", None, None, "reject"],
        "why": "",
        "model": "8b",
        "seconds": 9.0,
        "order": seeded_order(4),
    }
    # An answer without its seconds: how long the job waited for it
    answer = {"verdicts": ["edits"] * 4, "best": 3, "why": "M", "model": "8b"}
    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=FakeJudge(answer=answer), render=Draw())
    assert out["judge"]["seconds"] == 0.0 and isinstance(out["judge"]["seconds"], float)


def test_timings_with_the_judge(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(service, "time", types.SimpleNamespace(perf_counter=lambda: now[0]))

    def taking(seconds, work):
        def timed(*args, **kwargs):
            now[0] += seconds(*args) if callable(seconds) else seconds
            return work(*args, **kwargs)

        return timed

    runtime = LayoutRuntime()
    runtime.generate = taking(30.0, runtime.generate)
    runtime.retexture = taking(10.0, runtime.retexture)
    # The own texture runs to_glb in full; the new ones rebake
    runtime.export = taking(lambda mesh, preset: 25.0 if mesh.texture == "final" else 5.0, runtime.export)
    storage = FakeStorage()
    storage.put = taking(0.25, storage.put)
    judge = FakeJudge()

    out = handle_job(judged_job(), runtime, storage, taking(1.5, Packer()), judge=taking(9.0, judge), render=taking(0.5, Draw()))

    assert out["timings"] == {
        "generate_s": 30.0,
        "retexture_s": 30.0,
        "export_s": 15.0,
        "pack_s": 4.5,
        "upload_s": 0.75,
        "own_export_s": 25.0,
        "render_s": 2.0,
        "judge_s": 9.0,
    }
    assert sum(out["timings"].values()) == pytest.approx(now[0])  # the job's whole time
    assert out["judge"]["seconds"] == 8.6  # the judge's own time, not the job's wait


def test_the_picture_the_judge_gets_is_what_its_own_fit_makes_of_the_jobs(judge_worker):
    """Shrunk and on white before it is sent, so the judge sees what it would have made of the full picture."""
    import numpy as np

    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, (1000, 1600, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(rng.random((1000, 1600)) < 0.5, 0, rgba[..., 3])  # transparent where it isn't the object
    picture = Image.fromarray(rgba, "RGBA")
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    judge = FakeJudge()
    job = {"id": "rp-1", "input": {**judged_job()["input"], "image_base64": base64.b64encode(buffer.getvalue()).decode()}}
    handle_job(job, LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=Draw())

    sent = Image.open(io.BytesIO(base64.b64decode(judge.requests[0]["picture_png"])))
    expected = judge_worker.judge.fit(picture, judge_worker.judge.PICTURE_SIDE)
    assert sent.mode == "RGB" and sent.size == expected.size == (768, 480)
    assert np.array_equal(np.asarray(sent), np.asarray(expected))
    # The judge's own fit leaves it as it is
    again = judge_worker.judge.fit(sent, judge_worker.judge.PICTURE_SIDE)
    assert again.size == sent.size and np.array_equal(np.asarray(again), np.asarray(sent))
    # A small picture without transparency goes as it is
    small = Image.new("RGB", (300, 200), (10, 20, 30))
    assert np.array_equal(np.asarray(judge_picture(small)), np.asarray(small))


def test_the_worker_and_the_judge_agree_on_their_limits(judge_worker):
    assert service.JUDGE_PICTURE_SIDE == judge_worker.judge.PICTURE_SIDE
    assert service.JUDGE_UNDERLAY == judge_worker.judge.UNDERLAY
    assert service.VERDICTS == judge_worker.parse.VERDICTS
    assert inputs.MAX_PROMPT == judge_worker.service.MAX_PROMPT == 500  # and the Studio's MAX_PROMPT_LENGTH
    # Own texture and the most new textures a job makes: the judge has letters for them all
    assert 1 + MAX_TEXTURES <= len(judge_worker.prompt.LETTERS)


class Chat:
    """
    Stands in for Qwen3-VL in the judge worker: it reads the versions in the order they are shown (each
    image after its "Version X:" line) and answers, as the model is asked to, in JSON with the letter of the
    one in ``favourite``'s tint.
    """

    def __init__(self, favourite: str) -> None:
        self.favourite = favourite
        self.shown = []

    def __call__(self, messages, max_new_tokens=1024):
        content = messages[0]["content"]
        letters, tints = [], []
        for item, following in zip(content, content[1:]):
            text = item.get("text", "").strip() if item["type"] == "text" else ""
            if text.startswith("Version ") and following["type"] == "image":
                letters.append(text.removeprefix("Version ").rstrip(":"))
                tints.append(following["image"].convert("RGB").getpixel((0, 0)))
        self.shown = [next(name for name, tint in TINTS.items() if tint == colour) for colour in tints]
        best = letters[self.shown.index(self.favourite)]
        answer = {letter: {"problems": "none", "verdict": "publish" if letter == best else "edits"} for letter in letters}
        return json.dumps({**answer, "best": best, "why": f"{best} has the cleanest back"})


@pytest.mark.parametrize("favourite, pick", [("final", 0), ("4234", 3)])
def test_through_the_judge_workers_own_job_handling(judge_worker, favourite, pick, still_clock):
    """The request and the answer as the judge worker reads and writes them, with a stand-in for its model."""
    chat = Chat(favourite)

    def judge(request: dict) -> dict:
        return judge_worker.service.handle_job({"id": "fc-01K8", **request}, chat, name="8b")

    out = handle_job(judged_job(), LayoutRuntime(), FakeStorage(), Packer(), judge=judge, render=Draw())

    order = seeded_order(4)
    candidates = ["final", "2234", "3234", "4234"]
    # The model saw them in the job's order, the first under K
    assert chat.shown == [candidates[number] for number in order]
    assert out["judge"]["pick"] == pick and out["judge"]["order"] == order
    assert out["judge"]["verdicts"] == ["publish" if number == pick else "edits" for number in range(4)]
    assert out["judge"]["model"] == "8b" and isinstance(out["judge"]["seconds"], float)
    letter = judge_worker.prompt.LETTERS[order.index(pick)]
    assert out["judge"]["why"] == f"{letter} has the cleanest back"


# --- The judge's inputs --------------------------------------------------------------------------------------


@pytest.mark.parametrize("judge, asked", [("absent", False), (None, False), (False, False), (True, True)])
def test_judge_is_true_or_false(judge, asked):
    extra = {} if judge == "absent" else {"judge": judge}
    job = parse_job(payload(**extra), fallback_id="job")
    assert job.judge is asked and job.prompt == ""


@pytest.mark.parametrize("judge", [1, 0, "true", "yes", [True], {"model": "8b"}])
def test_anything_else_for_judge_is_refused_before_anything_runs(judge):
    runtime = FakeRuntime()
    out = handle_job(textures_job(judge=judge), runtime, FakeStorage(), Packer(), judge=FakeJudge())
    assert out == {"error": "invalid input: judge must be true or false"} and runtime.calls == []


def test_the_prompt_is_what_the_user_typed_for_the_judge():
    assert parse_job(payload(judge=True, prompt=PROMPT), fallback_id="job").prompt == PROMPT
    assert parse_job(payload(judge=True, prompt=""), fallback_id="job").prompt == ""  # a photo: none typed
    assert parse_job(payload(judge=True, prompt=None), fallback_id="job").prompt == ""
    assert parse_job(payload(judge=True, prompt="x" * 500), fallback_id="job").prompt == "x" * 500
    for prompt in ("x" * 501, 42, ["a lamp"]):
        with pytest.raises(InputError, match="^prompt must be a string of at most 500 characters$"):
            parse_job(payload(judge=True, prompt=prompt), fallback_id="job")
    # Without the judge it isn't read
    assert parse_job(payload(judge=False, prompt=42), fallback_id="job").prompt == ""


@pytest.mark.parametrize("mode", ["preview", "final"])
def test_previews_and_finals_ignore_judge_and_prompt(mode):
    job = parse_job({"image_base64": PICTURE, "mode": mode, "judge": "maybe", "prompt": 42}, fallback_id="job")
    assert job.judge is False and job.prompt == ""
    judge = FakeJudge()
    out = handle_job({"id": "f", "input": payload(mode=mode, judge=True)}, LayoutRuntime(), FakeStorage(), Packer(), judge=judge)
    assert "error" not in out and not {"judge", "judge_error"} & set(out) and judge.warmed == 0
