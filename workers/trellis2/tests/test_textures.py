"""Texture options (mode "textures") with a fake runtime: the shape, the seeds, the keys, the failures."""

import base64
import dataclasses
import gc
import io
import json
import types
import weakref

import pytest
from PIL import Image

from forge3d_worker import service
from forge3d_worker.compress import CompressionError
from forge3d_worker.inputs import InputError, parse_job
from forge3d_worker.service import TEXTURES_NEED_TRELLIS2, handle_job, texture_seed
from forge3d_worker.settings import (
    CREDITS,
    FALLBACK_PIPELINE,
    MAX_TEXTURES,
    MODES,
    PRESETS,
    TEXTURE_COUNT,
    TEXTURE_SEED_STEP,
)


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


def test_a_final_that_fell_back_to_512_has_its_shape_made_with_512_at_once():
    """The final's own fallback made its shape: the cascade would make another, or run out of memory again."""
    runtime = FakeRuntime()
    out = handle_job(textures_job(pipeline="512"), runtime, FakeStorage(), Packer())
    final = PRESETS["final"]
    fell_back = dataclasses.replace(final, pipeline_type=FALLBACK_PIPELINE[final.pipeline_type])
    assert fell_back.pipeline_type == "512"
    # The final's preset with the fallback's pipeline, through the generate() every job runs, then exported
    # with the final's settings, as the final that fell back was
    assert runtime.calls[0] == ("generate", fell_back, 1234, (64, 48), [])
    assert [call[2] for call in runtime.calls if call[0] == "export"] == [fell_back] * 3
    assert dataclasses.replace(fell_back, pipeline_type=final.pipeline_type) == final
    assert out["pipeline"] == "512" and len(out["textures"]) == 3 and "texture_errors" not in out


def test_the_finals_own_pipeline_is_the_same_as_none():
    runtime = FakeRuntime()
    out = handle_job(textures_job(pipeline="1024_cascade"), runtime, FakeStorage(), Packer())
    assert runtime.calls[0] == ("generate", PRESETS["final"], 1234, (64, 48), [])
    assert out["pipeline"] == "1024_cascade"
    for pipeline in ("1024_cascade", None):
        assert parse_job(payload(pipeline=pipeline), fallback_id="job").pipeline is None
    assert parse_job(payload(pipeline="512"), fallback_id="job").pipeline == "512"


@pytest.mark.parametrize("pipeline", ["1536_cascade", "1024", "512 ", "", 512, True, ["512"]])
def test_other_pipelines_are_refused_before_anything_runs(pipeline):
    runtime = FakeRuntime()
    out = handle_job(textures_job(pipeline=pipeline), runtime, FakeStorage(), Packer())
    assert out == {"error": "invalid input: pipeline must be the final's: '1024_cascade' or '512'"}
    assert runtime.calls == []


@pytest.mark.parametrize("mode", ["preview", "final"])
def test_previews_and_finals_ignore_pipeline(mode):
    job = parse_job({"image_base64": PICTURE, "mode": mode, "pipeline": "2048"}, fallback_id="job")
    assert job.mode == mode and job.pipeline is None


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
