"""modal_app.py without Modal's servers: its pins, its shape and the per-job wrapper."""

import ast
import base64
import io
import json
import pathlib
import re
import subprocess

import pytest

modal = pytest.importorskip("modal")
import modal.experimental  # noqa: E402

import modal_app  # noqa: E402

WORKERS = pathlib.Path(__file__).parent.parent


def test_pins_match_the_runpod_dockerfile():
    dockerfile = (WORKERS / "trellis2" / "Dockerfile").read_text()
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", dockerfile, re.M))
    assert args["TRELLIS2_COMMIT"] == modal_app.TRELLIS2_COMMIT
    assert args["CUMESH_COMMIT"] == modal_app.CUMESH_COMMIT
    assert args["FLEXGEMM_COMMIT"] == modal_app.FLEXGEMM_COMMIT
    assert args["GLTFPACK_VERSION"] == modal_app.GLTFPACK_VERSION
    assert f"pip install {' '.join(modal_app.TORCH)} --index-url {modal_app.TORCH_INDEX}" in dockerfile
    assert f"pip install {modal_app.FLASH_ATTN_WHEEL}" in dockerfile


def test_pixal3d_pins_match_its_notice():
    """The commits the image clones are the ones pixal3d/NOTICE.md records (and its download script's)."""
    notice = (WORKERS / "pixal3d" / "NOTICE.md").read_text()
    for commit in (modal_app.PIXAL3D_COMMIT, modal_app.NAF_COMMIT, modal_app.MOGE_COMMIT, modal_app.UTILS3D_COMMIT):
        assert f"`{commit}`" in notice, commit
    assert modal_app.PIXAL3D_WEIGHTS == "all"  # the recipe needs the multi-view set and the single-view one
    assert modal_app.FINAL_MODEL in modal_app.FINAL_MODELS


def test_finals_are_trellis2s_unless_deployed_with_pixal3d(monkeypatch):
    """Phase 6's re-test kept TRELLIS.2 as the default; ORAINGE_FINAL_MODEL=pixal3d switches to the recipe."""
    assert modal_app.FINAL_MODELS[0] == "trellis2"
    if "ORAINGE_FINAL_MODEL" not in __import__("os").environ:
        assert modal_app.FINAL_MODEL == "trellis2" and modal_app.FINAL_WEIGHTS == ()


def test_defines_the_workers_the_api_and_the_helpers():
    assert isinstance(modal_app.Trellis2, modal.Cls)
    assert isinstance(modal_app.FluxSchnell, modal.Cls)
    assert isinstance(modal_app.MultiView, modal.Cls)
    assert isinstance(modal_app.GeometryViews, modal.Cls)
    assert list(modal_app.WEIGHT_SCRIPTS) == ["trellis2", "reference", "multiview", "pixal3d"]  # download order
    assert isinstance(modal_app.download_models, modal.Function)
    assert isinstance(modal_app.api, modal.Function)
    assert isinstance(modal_app.make_model, modal.Function)
    assert isinstance(modal_app.make, modal.app.LocalEntrypoint)
    assert isinstance(modal_app.make_set, modal.app.LocalEntrypoint)


def test_run_job_passes_the_call_id_as_the_job_id(monkeypatch):
    stops = []
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "fc-01K6ABC")
    monkeypatch.setattr(modal.experimental, "stop_fetching_inputs", lambda: stops.append(True))
    jobs = []

    def handle(job):
        jobs.append(job)
        return {"request_id": job["id"], "seed": 7}

    assert modal_app.run_job(handle, {"input": {"mode": "preview"}}) == {"request_id": "fc-01K6ABC", "seed": 7}
    assert jobs == [{"id": "fc-01K6ABC", "input": {"mode": "preview"}}]
    assert stops == []


def test_run_job_replaces_the_container_after_a_gpu_fault(monkeypatch):
    stops = []
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "fc-01K6ABC")
    monkeypatch.setattr(modal.experimental, "stop_fetching_inputs", lambda: stops.append(True))
    error = "generation failed: RuntimeError: CUDA error: an illegal memory access was encountered"

    def handle(job):
        return {"error": error, "refresh_worker": True}

    # The flag is for this container only; the caller just sees the error
    assert modal_app.run_job(handle, {"input": {}}) == {"error": error}
    assert stops == [True]


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """A stand-in for the models volume, with download scripts that succeed or fail on demand."""
    runs = tmp_path / "runs.txt"
    script = tmp_path / "trellis2_weights.py"
    script.write_text(
        "import os, pathlib, sys\n"
        f"with open({str(runs)!r}, 'a') as f: f.write('run\\n')\n"
        "root = pathlib.Path(os.environ['MODELS_ROOT'], 'TRELLIS.2-4B')\n"
        "root.mkdir(exist_ok=True)\n"
        "(root / 'pipeline.json').write_text('{}')  # written early, like the real script\n"
        "sys.exit(1 if os.environ.get('FAIL_DOWNLOAD') else 0)\n"
    )
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(modal_app, "MODELS", str(models))
    monkeypatch.setattr(modal_app, "WEIGHT_SCRIPTS", {"trellis2": str(script)})
    return models, runs


def test_workers_refuse_partial_downloads(volume, monkeypatch):
    models, runs = volume
    monkeypatch.setenv("FAIL_DOWNLOAD", "1")
    with pytest.raises(subprocess.CalledProcessError):
        modal_app._download("trellis2", force=False)
    # pipeline.json is there, but the download never finished
    assert (models / "TRELLIS.2-4B" / "pipeline.json").exists()
    with pytest.raises(RuntimeError, match="download_models --which trellis2"):
        modal_app._require_weights("trellis2")


def test_complete_downloads_are_kept_and_not_repeated(volume, monkeypatch):
    models, runs = volume
    modal_app._download("trellis2", force=False)
    modal_app._require_weights("trellis2")
    modal_app._download("trellis2", force=False)
    assert runs.read_text().count("run") == 1

    # A forced download that fails leaves the weights marked incomplete again
    monkeypatch.setenv("FAIL_DOWNLOAD", "1")
    with pytest.raises(subprocess.CalledProcessError):
        modal_app._download("trellis2", force=True)
    with pytest.raises(RuntimeError):
        modal_app._require_weights("trellis2")


# GeometryViews: MV-Adapter's image+geometry model, for experiments only


def test_geometry_views_is_a_small_experiment_worker():
    """MultiView's image and GPU, one container that soon scales down, and nothing in production calls it."""
    tree = ast.parse((WORKERS / "modal_app.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GeometryViews")
    options = {keyword.arg: keyword.value for keyword in cls.decorator_list[0].keywords}
    assert ast.unparse(options["image"]) == "multiview_image" and ast.unparse(options["gpu"]) == "MULTIVIEW_GPU"
    assert modal_app.MULTIVIEW_GPU == "A10G"
    assert ast.literal_eval(options["max_containers"]) == 1
    assert ast.literal_eval(options["scaledown_window"]) < 60  # MultiView's
    assert "secrets" not in options  # the views come back inline, never to R2
    assert [node.name for node in cls.body if isinstance(node, ast.FunctionDef)] == ["load", "generate"]
    # The job API routes nothing to it
    api = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "api")
    assert "GeometryViews" not in ast.unparse(api)


def test_geometry_views_loads_the_model_once_and_handles_jobs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(WORKERS / "multiview"))
    np = pytest.importorskip("numpy")
    from PIL import Image

    from multiview_worker import geometry

    built = []

    class Model:
        last_timings = {"views_s": 1.0}

        def __init__(self, models_root):
            built.append(models_root)

        def draw_views(self, reference, control, prompt, seed, **settings):
            return [Image.new("RGB", (768, 768), (10 * i, 0, 0)) for i in range(6)]

    monkeypatch.setattr(geometry, "GeometryViewGenerator", Model)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(modal_app, "MODELS", str(models))
    with pytest.raises(RuntimeError, match="download_models --which multiview"):
        modal_app.geometry_views_handler()

    (models / ".multiview-weights").write_text("digest")
    handle = modal_app.geometry_views_handler()
    assert built == [str(models)]
    picture = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(picture, format="PNG")
    control = geometry.pack_control(np.full(geometry.CONTROL_SHAPE, 0.5, dtype=np.float32))
    # run_job adds the call's id to the job
    out = handle({"id": "fc-01K6", "image_base64": base64.b64encode(picture.getvalue()).decode(), "control_pngs": control})
    assert len(out["views"]) == 6 and out["timings"] == {"views_s": 1.0} and built == [str(models)]
    assert handle({"id": "fc-01K7", "control_pngs": control}) == {"error": "invalid input: image_base64 must be a base64 string"}


# Two models in one container: TRELLIS.2 for previews, Pixal3D for finals


def job(mode="final", **extra):
    return {"id": "j", "input": {"image_base64": "aGk=", "mode": mode, **extra}}


@pytest.mark.parametrize("payload", [{"mode": "preview"}, {"mode": "preview", "views": [{"azimuth": 90}]}])
def test_previews_are_trellis2s_whatever_is_loaded(payload):
    assert modal_app.choose_model({"input": payload}, "pixal3d", {"trellis2", "pixal3d"}) == "trellis2"


@pytest.mark.parametrize("payload", [{"mode": "final"}, {}, {"mode": "final", "views": [{"azimuth": 90}]}])
def test_finals_are_the_final_models_when_it_is_loaded(payload):
    assert modal_app.choose_model({"input": payload}, "pixal3d", {"trellis2", "pixal3d"}) == "pixal3d"
    assert modal_app.choose_model({"input": payload}, "pixal3d", {"trellis2"}) == "trellis2"  # not loaded
    assert modal_app.choose_model({"input": payload}, "trellis2", {"trellis2", "pixal3d"}) == "trellis2"
    assert modal_app.choose_model({"input": "garbage"}, "pixal3d", {"trellis2", "pixal3d"}) == "pixal3d"


class FakeRuntime:
    def __init__(self, name):
        self.name = name
        self.asleep = False
        self.log = []

    def sleep(self):
        self.asleep = True
        self.log.append("sleep")

    def wake(self):
        self.asleep = False
        self.log.append("wake")


def test_the_pool_builds_pixal3d_in_the_background_and_puts_trellis2_to_sleep_once():
    trellis2, pixal3d = FakeRuntime("trellis2"), FakeRuntime("pixal3d")
    pool = modal_app.ModelPool(trellis2)
    assert pool.loaded == {"trellis2"} and pool.use("trellis2") is trellis2

    pool.load_later("pixal3d", lambda: pixal3d)
    assert pool.loaded == {"trellis2", "pixal3d"}  # a job may ask for it already; use() waits
    assert pool.use("pixal3d") is pixal3d
    assert trellis2.log == ["sleep"] and pool.resident == "pixal3d"
    # From then on TRELLIS.2 serves asleep; nothing is swapped back and forth between jobs
    assert pool.use("trellis2") is trellis2 and pool.use("pixal3d") is pixal3d
    assert trellis2.log == ["sleep"] and pixal3d.log == [] and pool.resident == "pixal3d"


def test_a_model_that_fails_to_load_is_counted_out(capsys):
    trellis2 = FakeRuntime("trellis2")
    pool = modal_app.ModelPool(trellis2)

    def broken():
        raise RuntimeError("pipeline_mv.json is missing")

    pool.load_later("pixal3d", broken)
    assert pool.use("pixal3d") is None
    assert pool.loaded == {"trellis2"} and trellis2.log == []
    assert "pixal3d could not be loaded, so finals are made with trellis2: RuntimeError: pipeline_mv.json" in capsys.readouterr().out

    pool.count_out("pixal3d", "its weights are missing")
    assert pool.loaded == {"trellis2"} and pool.use("pixal3d") is None


class Handlers:
    """Each model's handle_job, scripted: the result, or an error."""

    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.calls = []

    def __getitem__(self, name):
        def handle(job, runtime, storage, pack):
            self.calls.append((name, runtime.name, job["input"]["mode"], storage, pack))
            outcome = self.outcomes[name]
            return dict(outcome.pop(0) if isinstance(outcome, list) else outcome)

        return handle


def pool_with_both():
    trellis2, pixal3d = FakeRuntime("trellis2"), FakeRuntime("pixal3d")
    pool = modal_app.ModelPool(trellis2)
    pool.load_later("pixal3d", lambda: pixal3d)
    return pool


def test_finals_go_through_pixal3d_and_previews_through_trellis2():
    pool = pool_with_both()
    handlers = Handlers({"trellis2": {"glb": "t"}, "pixal3d": {"glb": "p", "weights": "multiview"}})
    final = modal_app.handle_with_models(job("final"), pool, "pixal3d", handlers, "storage", "pack")
    preview = modal_app.handle_with_models(job("preview"), pool, "pixal3d", handlers, "storage", "pack")
    assert final == {"glb": "p", "weights": "multiview", "model": "pixal3d"}
    assert preview == {"glb": "t", "model": "trellis2"}
    assert [call[:3] for call in handlers.calls] == [("pixal3d", "pixal3d", "final"), ("trellis2", "trellis2", "preview")]
    assert handlers.calls[0][3:] == ("storage", "pack")


def test_a_pixal3d_final_out_of_memory_is_made_with_trellis2_and_says_so(capsys):
    pool = pool_with_both()
    oom = {"error": "generation failed: OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"}
    handlers = Handlers({"pixal3d": oom, "trellis2": {"glb": "t", "pipeline": "512"}})
    result = modal_app.handle_with_models(job("final"), pool, "pixal3d", handlers, None, None)
    assert result == {"glb": "t", "pipeline": "512", "model": "trellis2", "fallback": oom["error"]}
    assert [call[0] for call in handlers.calls] == ["pixal3d", "trellis2"]
    assert "pixal3d ran out of GPU memory; making this final with trellis2 instead" in capsys.readouterr().out


def test_other_pixal3d_failures_are_reported_as_they_are():
    pool = pool_with_both()
    handlers = Handlers({"pixal3d": {"error": "invalid input: no object found in the image"}, "trellis2": {"glb": "t"}})
    result = modal_app.handle_with_models(job("final"), pool, "pixal3d", handlers, None, None)
    assert result == {"error": "invalid input: no object found in the image"}  # no "model" on an error
    assert [call[0] for call in handlers.calls] == ["pixal3d"]


def test_finals_fall_back_to_trellis2_when_pixal3d_is_not_there(capsys):
    trellis2 = FakeRuntime("trellis2")
    pool = modal_app.ModelPool(trellis2)
    pool.count_out("pixal3d", "its weights are missing")
    handlers = Handlers({"trellis2": {"glb": "t", "pipeline": "1024_cascade"}})
    result = modal_app.handle_with_models(job("final"), pool, "pixal3d", handlers, None, None)
    # The result says why the final isn't Pixal3D's, so a silent switch shows in every result
    assert result == {"glb": "t", "pipeline": "1024_cascade", "model": "trellis2", "fallback": "its weights are missing"}
    assert trellis2.log == []  # TRELLIS.2 keeps the GPU to itself
    # Previews never say so: they are TRELLIS.2's anyway
    preview = modal_app.handle_with_models(job("preview"), pool, "pixal3d", handlers, None, None)
    assert "fallback" not in preview

    # Deployed with ORAINGE_FINAL_MODEL=trellis2: Pixal3D is never asked for
    pool = pool_with_both()
    result = modal_app.handle_with_models(job("final"), pool, "trellis2", handlers, None, None)
    assert result["model"] == "trellis2" and pool.resident == "trellis2" and "fallback" not in result


@pytest.mark.parametrize("final_model", ["trellis2", "pixal3d"])
def test_texture_options_are_trellis2s(final_model):
    payload = {"mode": "textures", "seed": 7}
    assert modal_app.choose_model({"input": payload}, final_model, {"trellis2", "pixal3d"}) == "trellis2"


def test_texture_options_for_trellis2s_finals_go_to_trellis2():
    pool = modal_app.ModelPool(FakeRuntime("trellis2"))
    handlers = Handlers({"trellis2": {"mode": "textures", "textures": [{"texture_seed": 1007}]}})
    result = modal_app.handle_with_models(job("textures", seed=7), pool, "trellis2", handlers, "storage", "pack")
    assert result == {"mode": "textures", "textures": [{"texture_seed": 1007}], "model": "trellis2"}
    assert handlers.calls == [("trellis2", "trellis2", "textures", "storage", "pack")]
    # An error is passed on as it is, with no model named
    handlers = Handlers({"trellis2": {"error": "invalid input: seed is required for textures: send the final's seed"}})
    result = modal_app.handle_with_models(job("textures"), pool, "trellis2", handlers, None, None)
    assert result == {"error": "invalid input: seed is required for textures: send the final's seed"}


@pytest.mark.parametrize("pixal3d", ["loaded", "missing"])
def test_texture_options_are_refused_with_the_recipe_on(pixal3d, capsys):
    """A Pixal3D final has another shape than TRELLIS.2 would make, and only TRELLIS.2 retextures."""
    if pixal3d == "loaded":
        pool = pool_with_both()
    else:  # its finals are TRELLIS.2's here, but the creator's final may come from a container where it loaded
        pool = modal_app.ModelPool(FakeRuntime("trellis2"))
        pool.count_out("pixal3d", "its weights are missing")
    handlers = Handlers({"trellis2": {"glb": "t"}, "pixal3d": {"glb": "p"}})
    result = modal_app.handle_with_models(job("textures", seed=7), pool, "pixal3d", handlers, None, None)
    assert result == {"error": "texture options need TRELLIS.2 finals"}
    assert handlers.calls == []  # nothing ran
    assert pool.resident == "trellis2" and pool.runtimes["trellis2"].log == []  # nor did TRELLIS.2 go to sleep
    assert "[orainge] a textures job refused: the finals are pixal3d's" in capsys.readouterr().out


def test_the_refusal_is_the_trellis2_workers_own():
    """The TRELLIS.2 worker's service refuses a runtime that can't retexture with the same words."""
    service = (WORKERS / "trellis2" / "forge3d_worker" / "service.py").read_text()
    assert f'TEXTURES_NEED_TRELLIS2 = "{modal_app.TEXTURES_NEED_TRELLIS2}"' in service


class MovableRuntime(FakeRuntime):
    """A runtime whose models can go to the CPU and back (Pixal3D's _offload and _restore)."""

    def _offload(self):
        self.log.append("offload")

    def _restore(self):
        self.log.append("restore")


def test_a_pixal3d_out_of_memory_fallback_runs_with_pixal3ds_models_off_the_gpu():
    """Its _offload also clears the stale CUDA error a CuMesh failure leaves; _restore brings it back after."""
    trellis2, pixal3d = FakeRuntime("trellis2"), MovableRuntime("pixal3d")
    pool = modal_app.ModelPool(trellis2)
    pool.load_later("pixal3d", lambda: pixal3d)
    order = []
    oom = {"error": "export failed: RuntimeError: [CuMesh] CUDA error: out of memory"}

    class Recording(Handlers):
        def __getitem__(self, name):
            handle = super().__getitem__(name)

            def recorded(job, runtime, storage, pack):
                order.append((name, list(pixal3d.log)))
                return handle(job, runtime, storage, pack)

            return recorded

    handlers = Recording({"pixal3d": oom, "trellis2": {"glb": "t"}})
    result = modal_app.handle_with_models(job("final"), pool, "pixal3d", handlers, None, None)
    assert result == {"glb": "t", "model": "trellis2", "fallback": oom["error"]}
    assert order == [("pixal3d", []), ("trellis2", ["offload"])]
    assert pixal3d.log == ["offload", "restore"]


def test_build_pixal3d_shares_the_containers_trellis2(monkeypatch):
    import sys
    import types

    built = {}

    class Runtime:
        def __init__(self, **options):
            built.update(options)

    fake = types.ModuleType("pixal3d_worker.pipeline")
    fake.Pixal3DRuntime = Runtime
    fake.LEVEL_PREVIEW = "preview"
    fake.thin_ratio_from_env = lambda raw: 0.25 if raw else 0.2
    package = types.ModuleType("pixal3d_worker")
    package.pipeline = fake
    monkeypatch.setitem(sys.modules, "pixal3d_worker", package)
    monkeypatch.setitem(sys.modules, "pixal3d_worker.pipeline", fake)
    monkeypatch.delenv("PIXAL3D_THIN_RATIO", raising=False)

    trellis2 = FakeRuntime("trellis2")
    modal_app.build_pixal3d(trellis2)
    # The multi-view weights with the single-view set for thin objects, resident, levelling against the
    # recipe's preview made by this container's TRELLIS.2 (no second copy)
    assert built == {"multiview": True, "low_vram": False, "level": "preview", "trellis2": trellis2, "thin_ratio": 0.2}
    monkeypatch.setenv("PIXAL3D_THIN_RATIO", "0.25")
    modal_app.build_pixal3d(trellis2)
    assert built["thin_ratio"] == 0.25


# run_pipeline: the whole flow `make` runs in the cloud, with fake workers


class FakeWorkers:
    def __init__(self):
        self.calls = []
        self.fail_final = False

    def ensure_weights(self, name):
        self.calls.append(("weights", name))

    def reference(self, job):
        self.calls.append(("reference", job))
        pictures = [b"png-5", b"png-6"]
        return {
            "images": [
                {"key": f"ai/x/reference-{5 + i}.png", "url": None, "base64": base64.b64encode(p).decode(), "seed": 5 + i}
                for i, p in enumerate(pictures)
            ]
        }

    def trellis(self, job):
        self.calls.append(("trellis", job))
        mode = job["input"]["mode"]
        if mode == "final" and self.fail_final:
            raise RuntimeError("container exited with code 137")
        seed = job["input"].get("seed", 42)
        result = {
            "seed": seed,
            "mode": mode,
            "glb": {"key": "k", "url": None, "base64": base64.b64encode(f"glb-{mode}".encode()).decode()},
            "triangles": 30_000 if mode == "preview" else 100_000,
            "pipeline": "512" if mode == "preview" else "1024_cascade",
            "timings": {"generate_s": 1.0},
        }
        if mode == "final":  # the worker says whether it painted the picture onto the model
            result["projection"] = {"applied": False, "reason": "the silhouettes don't match well enough"}
            result["views_used"] = 0
            # The Pixal3D recipe's notes (production's container)
            result["pipeline"] = "pixal3d-1024_cascade"
            result["model"] = "pixal3d"
            result["weights"] = "multiview"
            result["thin"] = {"extents": [0.4, 0.8, 1.0], "ratio": 0.4, "threshold": 0.2, "thin": False, "decided": True}
            result["camera"] = {"fov_deg": 31.2, "tilt": {"elevation": 21.2, "roll": 0.0, "source": "preview"}}
            result["pose"] = {"applied": True, "reason": "found", "iou": 0.98}
            result["level"] = {"applied": True, "elevation": 21.2, "roll": 0.0}
        else:
            result["model"] = "trellis2"
        return result


def run(folder, workers, **kwargs):
    options = dict(prompt=None, image=None, image_name="", final=False, seed=None)
    options.update(kwargs)
    return modal_app.run_pipeline(
        folder,
        ensure_weights=workers.ensure_weights,
        reference=workers.reference,
        trellis=workers.trellis,
        fetch=modal_app._asset_bytes,
        save=lambda: None,
        **options,
    )


def test_prompt_run_saves_every_step(tmp_path):
    workers = FakeWorkers()
    folder = tmp_path / "run-1"
    state = run(folder, workers, prompt="a brass pocket watch", final=True, final_weights=("pixal3d",))

    # The weights for every step, Pixal3D's because it makes the final
    assert workers.calls[:3] == [("weights", "trellis2"), ("weights", "reference"), ("weights", "pixal3d")]
    assert workers.calls[3] == (
        "reference",
        {"input": {"prompt": "a brass pocket watch", "count": 4, "request_id": "run-1"}},
    )
    preview, final = workers.calls[4][1]["input"], workers.calls[5][1]["input"]
    # Unscored pictures: the first goes to 3D, and the final reuses the preview's seed
    assert base64.b64decode(preview["image_base64"]) == b"png-5"
    assert (preview["mode"], preview["request_id"], "seed" in preview) == ("preview", "run-1", False)
    assert (final["mode"], final["seed"]) == ("final", 42)

    assert (folder / "reference-5.png").read_bytes() == b"png-5"
    assert (folder / "preview-42.glb").read_bytes() == b"glb-preview"
    assert (folder / "final-42.glb").read_bytes() == b"glb-final"
    saved = json.loads((folder / "progress.json").read_text())
    assert saved == state
    assert "scores" not in saved["steps"]["reference"]
    assert {step: info["status"] for step, info in saved["steps"].items()} == {
        "weights": "done",
        "reference": "done",
        "preview": "done",
        "final": "done",
    }
    assert saved["steps"]["final"]["triangles"] == 100_000
    final_step, preview_step = saved["steps"]["final"], saved["steps"]["preview"]
    assert final_step["pipeline"] == "pixal3d-1024_cascade" and preview_step["pipeline"] == "512"
    assert final_step["projection"]["applied"] is False
    assert "projection" not in preview_step
    # What made the final and how: the model, Pixal3D's weights, what the preview measured, the camera, the
    # pose search and the levelling; nothing of that on a TRELLIS.2 preview, and no "views_used": 0
    assert (final_step["model"], final_step["weights"]) == ("pixal3d", "multiview")
    assert final_step["thin"]["ratio"] == 0.4 and final_step["level"]["elevation"] == 21.2
    assert final_step["camera"]["tilt"]["source"] == "preview" and final_step["pose"]["iou"] == 0.98
    assert preview_step["model"] == "trellis2"
    assert "views_used" not in final_step and "fallback" not in final_step
    assert not {"weights", "thin", "camera", "pose", "level"} & set(preview_step)


def test_a_run_continued_with_a_final_fetches_the_final_models_weights(tmp_path):
    """`make --run NAME --final` after a preview-only run: its weights step is done, Pixal3D's aren't."""
    workers = FakeWorkers()
    folder = tmp_path / "run-1"
    run(folder, workers, image=b"img", image_name="pic.png", final_weights=("pixal3d",))
    assert ("weights", "pixal3d") not in workers.calls  # no final, so no Pixal3D

    workers.calls.clear()
    run(folder, workers, final=True, final_weights=("pixal3d",))
    assert [call[0] if call[0] != "weights" else call for call in workers.calls] == [("weights", "pixal3d"), "trellis"]


def test_finals_by_trellis2_fetch_nothing_more(tmp_path):
    workers = FakeWorkers()
    run(tmp_path / "run-1", workers, image=b"img", image_name="pic.png", final=True)
    assert [call for call in workers.calls if call[0] == "weights"] == [("weights", "trellis2")]


def test_an_interrupted_run_continues_where_it_stopped(tmp_path):
    workers = FakeWorkers()
    workers.fail_final = True
    folder = tmp_path / "run-2"
    with pytest.raises(RuntimeError, match="code 137"):
        run(folder, workers, prompt="a teapot", final=True)
    saved = json.loads((folder / "progress.json").read_text())
    assert saved["steps"]["final"] == {"status": "failed", "error": "RuntimeError: container exited with code 137"}

    # Continuing (no prompt needed) only redoes the final
    workers.calls.clear()
    workers.fail_final = False
    run(folder, workers)
    assert [name for name, _ in workers.calls if name != "weights"] == ["trellis"]
    assert workers.calls[-1][1]["input"]["seed"] == 42
    assert (folder / "final-42.glb").exists()


def test_image_runs_skip_the_reference_step(tmp_path):
    workers = FakeWorkers()
    folder = tmp_path / "run-3"
    run(folder, workers, image=b"jpeg-bytes", image_name="cat.jpg", seed=7)
    assert workers.calls[0] == ("weights", "trellis2")
    assert [name for name, _ in workers.calls] == ["weights", "trellis"]
    job = workers.calls[1][1]["input"]
    assert (base64.b64decode(job["image_base64"]), job["seed"]) == (b"jpeg-bytes", 7)
    assert (folder / "input-cat.jpg").read_bytes() == b"jpeg-bytes"
    assert (folder / "preview-7.glb").exists()


def test_worker_errors_stop_the_run_and_are_recorded(tmp_path):
    workers = FakeWorkers()
    workers.trellis = lambda job: {"error": "invalid input: no object found in the image"}
    folder = tmp_path / "run-4"
    with pytest.raises(RuntimeError, match="no object found"):
        run(folder, workers, image=b"x", image_name="blank.png")
    saved = json.loads((folder / "progress.json").read_text())
    assert saved["steps"]["preview"]["status"] == "failed"


def test_a_continued_run_keeps_its_inputs(tmp_path):
    workers = FakeWorkers()
    folder = tmp_path / "run-5"
    run(folder, workers, prompt="a lamp")
    for change in ({"prompt": "a chair"}, {"image": b"x", "image_name": "a.png"}, {"seed": 1}):
        with pytest.raises(ValueError, match="start a new run"):
            run(folder, workers, **change)
    with pytest.raises(ValueError, match="needs a prompt or an image"):
        run(tmp_path / "run-6", workers)


def test_an_image_run_started_again_with_the_same_image_carries_on(tmp_path):
    # Modal runs an input again when its container is replaced mid-run
    workers = FakeWorkers()
    workers.fail_final = True
    folder = tmp_path / "run-10"
    with pytest.raises(RuntimeError, match="code 137"):
        run(folder, workers, image=b"jpeg-bytes", image_name="cat.jpg", seed=7, final=True)
    workers.fail_final = False
    workers.calls.clear()
    run(folder, workers, image=b"jpeg-bytes", image_name="cat.jpg", seed=7, final=True)
    assert [job["input"]["mode"] for name, job in workers.calls if name == "trellis"] == ["final"]
    with pytest.raises(ValueError, match="start a new run"):
        run(folder, workers, image=b"other-bytes", image_name="cat.jpg", seed=7)


def test_pictures_first_then_the_picked_one_becomes_3d(tmp_path):
    workers = FakeWorkers()
    folder = tmp_path / "run-7"
    state = run(folder, workers, prompt="a lamp", final=True, pictures_only=True)
    assert workers.calls[:2] == [("weights", "trellis2"), ("weights", "reference")]
    assert [name for name, _ in workers.calls] == ["weights", "weights", "reference"]
    assert state["input"] == "reference-5.png" and "preview" not in state["steps"]

    # Like a user in the Studio: the second picture goes on to 3D, preview and final
    workers.calls.clear()
    run(folder, workers, prompt="a lamp", final=True, pick=2)
    jobs = [job["input"] for name, job in workers.calls if name == "trellis"]
    assert [(job["mode"], base64.b64decode(job["image_base64"])) for job in jobs] == [
        ("preview", b"png-6"),
        ("final", b"png-6"),
    ]
    assert [name for name, _ in workers.calls if name == "reference"] == []  # the pictures are kept
    assert json.loads((folder / "progress.json").read_text())["input"] == "reference-6.png"

    # Once there is a model, another picture needs a new run; the same pick again is fine
    with pytest.raises(ValueError, match="start a new run to use picture 1"):
        run(folder, workers, pick=1)
    run(folder, workers, pick=2)


def test_a_pick_needs_a_picture_to_pick(tmp_path):
    workers = FakeWorkers()
    with pytest.raises(ValueError, match="pick must be 1 to 2"):
        run(tmp_path / "run-8", workers, prompt="a lamp", pick=4)
    with pytest.raises(ValueError, match="no pictures to pick from"):
        run(tmp_path / "run-9", workers, image=b"x", image_name="a.png", pick=1)


class ScoringWorkers(FakeWorkers):
    """Like the FLUX worker: each picture comes with its framing score and issues (None: not scored)."""

    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def reference(self, job):
        self.calls.append(("reference", job))
        images = []
        for i, score in enumerate(self.scores):
            data = base64.b64encode(f"png-{5 + i}".encode()).decode()
            picture = {"key": f"ai/x/reference-{5 + i}.png", "url": None, "base64": data, "seed": 5 + i}
            if score is not None:
                picture.update(score=score, issues=[] if score >= 0.5 else ["small in the frame"])
            images.append(picture)
        return {"images": images}


def test_the_best_scored_picture_becomes_3d(tmp_path):
    workers = ScoringWorkers([0.41, 0.93, 0.93, 0.7])
    folder = tmp_path / "run-10"
    state = run(folder, workers, prompt="a lamp")

    # The second and third tie, and the earlier one wins
    assert state["input"] == "reference-6.png"
    preview = next(job["input"] for name, job in workers.calls if name == "trellis")
    assert base64.b64decode(preview["image_base64"]) == b"png-6"
    # The scores are kept next to the pictures they belong to
    step = json.loads((folder / "progress.json").read_text())["steps"]["reference"]
    assert step["files"] == ["reference-5.png", "reference-6.png", "reference-7.png", "reference-8.png"]
    assert step["scores"] == [0.41, 0.93, 0.93, 0.7]
    assert step["issues"] == [["small in the frame"], [], [], []]


def test_a_pick_still_beats_the_scores(tmp_path):
    workers = ScoringWorkers([0.2, 0.9])
    folder = tmp_path / "run-11"
    assert run(folder, workers, prompt="a lamp", pictures_only=True)["input"] == "reference-6.png"

    run(folder, workers, prompt="a lamp", pick=1)
    jobs = [job["input"] for name, job in workers.calls if name == "trellis"]
    assert [base64.b64decode(job["image_base64"]) for job in jobs] == [b"png-5"]


def test_pictures_the_worker_couldnt_score_are_passed_over(tmp_path):
    workers = ScoringWorkers([None, 0.3, 0.8])
    state = run(tmp_path / "run-12", workers, prompt="a lamp", pictures_only=True)
    assert state["input"] == "reference-7.png"
    assert state["steps"]["reference"]["scores"] == [None, 0.3, 0.8]
    assert state["steps"]["reference"]["issues"] == [[], ["small in the frame"], []]


def test_status_says_what_is_ready_and_what_is_left(monkeypatch, capsys):
    import modal.exception as mx
    from modal.types import FileEntry, FileEntryType

    progress = {
        "prompt": "a teapot",
        "steps": {
            "weights": {"status": "done"},
            "reference": {"status": "done", "files": ["reference-5.png"]},
            "preview": {"status": "failed", "error": "RuntimeError: boom"},
        },
        "updated": "2026-09-30 01:00:00 UTC",
    }

    class FakeVolume:
        def __init__(self, name):
            self.name = name

        def listdir(self, path):
            if self.name == "orainge-models":
                return [
                    FileEntry(".trellis2-weights", FileEntryType.FILE, 1, 64),
                    FileEntry("TRELLIS.2-4B", FileEntryType.DIRECTORY, 1, 0),
                ]
            return [FileEntry("run-1", FileEntryType.DIRECTORY, 5, 0)]

        def read_file(self, path):
            assert path == "run-1/progress.json"
            yield json.dumps(progress).encode()

    class NotDeployed:
        def get_web_url(self):
            raise mx.NotFoundError("App 'orainge-ai' not found")

    monkeypatch.setattr(modal.Volume, "from_name", staticmethod(lambda name, **kw: FakeVolume(name)))
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(lambda app, name, **kw: NotDeployed()))
    modal_app.print_status()
    out = capsys.readouterr().out
    assert "Weights trellis2: ready" in out
    assert "Weights reference: not downloaded yet" in out
    assert "Job API: not deployed" in out
    assert "run-1 (a teapot): weights done, reference done, preview failed" in out
    assert "preview failed: RuntimeError: boom" in out
    assert "make --run run-1" in out


class FakeOutputs:
    """The outputs volume's read calls, over a dict of stored files."""

    def __init__(self, stored):
        self.stored = stored

    def read_file(self, path):
        if path not in self.stored:
            raise FileNotFoundError(path)
        yield self.stored[path]

    def read_file_into_fileobj(self, path, fileobj):
        if path not in self.stored:
            raise FileNotFoundError(path)
        fileobj.write(self.stored[path])


def test_copy_run_brings_back_the_progress_and_every_saved_file(tmp_path, monkeypatch):
    progress = {
        "prompt": None,
        "input": "input-cat.jpg",
        "steps": {
            "weights": {"status": "done"},
            "preview": {"status": "done", "files": ["preview-7.glb"]},
            "final": {"status": "done", "files": ["final-7.glb"]},  # lost from the volume
        },
    }
    stored = {
        "run-1/progress.json": json.dumps(progress).encode(),
        "run-1/input-cat.jpg": b"jpeg",
        "run-1/preview-7.glb": b"glb",
    }
    monkeypatch.setattr(modal_app, "outputs", FakeOutputs(stored))
    target = tmp_path / "copy" / "run-1"
    assert modal_app._copy_run("run-1", target) == ["preview-7.glb", "input-cat.jpg"]
    assert (target / "preview-7.glb").read_bytes() == b"glb"
    assert (target / "input-cat.jpg").read_bytes() == b"jpeg"
    assert not (target / "final-7.glb").exists()
    # progress.json comes along, so make_gallery.py can read the copy
    assert json.loads((target / "progress.json").read_text()) == progress
    # A run that never saved anything leaves nothing behind
    assert modal_app._copy_run("run-2", tmp_path / "copy" / "run-2") == []
    assert not (tmp_path / "copy" / "run-2").exists()


# Test sets


def test_a_set_lists_prompts_and_photos(tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "Chair 1.JPG").write_bytes(b"jpeg")
    listing = tmp_path / "set.txt"
    listing.write_text("# notes\n\na brass pocket watch\n  photos/Chair 1.JPG  \na 3.5 inch floppy disk\n")
    assert modal_app.read_set(listing) == [
        {"prompt": "a brass pocket watch"},
        {"image": tmp_path / "photos" / "Chair 1.JPG"},
        {"prompt": "a 3.5 inch floppy disk"},
    ]
    # A missing photo stops the set before anything runs
    listing.write_text("a lamp\nphotos/missing.png\n")
    with pytest.raises(SystemExit, match="line 2"):
        modal_app.read_set(listing)


def test_set_runs_are_named_after_their_prompts(tmp_path):
    runs = [
        {"prompt": "A Brass Pocket-Watch!"},
        {"image": tmp_path / "Chair 1.JPG"},
        {"prompt": "a medieval longsword with a leather-wrapped grip"},
        {"prompt": "보물 상자"},  # no Latin letters to name it by
    ]
    assert modal_app.set_run_names("s", runs) == [
        "s-01-brass-pocket-watch",
        "s-02-chair-1",
        "s-03-medieval-longsword",
        "s-04-run",
    ]
    # The longest set names and prompts still make valid run names
    names = modal_app.set_run_names("x" * 24, [{"prompt": "y" * 80}] + [{"prompt": "word " * 40}] * 120)
    assert all(modal_app.RUN_NAME.fullmatch(name) for name in names)


def test_the_starter_set_is_valid():
    runs = modal_app.read_set(WORKERS / "test-sets" / "starter.txt")
    assert len(runs) == 8 and all("prompt" in run for run in runs)
    names = modal_app.set_run_names("starter", runs)
    assert names[0] == "starter-01-wooden-treasure-chest-with-iron"
    assert len(set(names)) == len(names)
    assert all(modal_app.RUN_NAME.fullmatch(name) for name in names)


def test_the_phase2_set_is_valid():
    runs = modal_app.read_set(WORKERS / "test-sets" / "phase2.txt")
    assert len(runs) == 20 and all("prompt" in run for run in runs)
    names = modal_app.set_run_names("phase2", runs)
    assert names[0] == "phase2-01-cute-low-poly-fox-sitting-down"
    assert len(set(names)) == len(names)
    assert all(modal_app.RUN_NAME.fullmatch(name) for name in names)


def test_the_products_set_is_valid():
    runs = modal_app.read_set(WORKERS / "test-sets" / "products.txt")
    assert len(runs) == 8 and all("prompt" in run for run in runs)
    names = modal_app.set_run_names("products", runs)
    assert names[0] == "products-01-make-a-bmw-car-m3-model-blue"
    assert len(set(names)) == len(names)
    assert all(modal_app.RUN_NAME.fullmatch(name) for name in names)


def test_the_renders_set_lists_its_images():
    runs = modal_app.read_set(WORKERS / "test-sets" / "renders.txt")
    assert len(runs) == 6 and all(run["image"].is_file() for run in runs)
    assert modal_app.set_run_names("renders", runs)[0] == "renders-01-lantern"


def test_set_arguments_continue_runs_already_started(tmp_path):
    photo = tmp_path / "chair.jpg"
    photo.write_bytes(b"jpeg")
    runs = [{"prompt": "a lamp"}, {"image": photo}, {"image": photo}]
    names = ["s-01-lamp", "s-02-chair", "s-03-chair"]
    assert modal_app.set_arguments(names, runs, {"s-01-lamp", "s-02-chair"}, True) == [
        # The prompt goes again, so run_pipeline refuses it if the file was edited since
        ("s-01-lamp", "a lamp", b"", "", True, -1, 0, False),
        # A started photo run continues from the photo it saved
        ("s-02-chair", "", b"", "", True, -1, 0, False),
        ("s-03-chair", "", b"jpeg", "chair.jpg", True, -1, 0, False),
    ]
    # Picks go to their runs by number; pictures_only goes to every run
    assert modal_app.set_arguments(names[:1], runs[:1], set(), True, {1: 3}, True) == [
        ("s-01-lamp", "a lamp", b"", "", True, -1, 3, True)
    ]


def test_picks_name_a_prompt_run_and_one_of_its_pictures(tmp_path):
    photo = tmp_path / "chair.jpg"
    photo.write_bytes(b"jpeg")
    runs = [{"prompt": "a lamp"}, {"image": photo}, {"prompt": "a mug"}]
    assert modal_app.parse_picks("", runs) == {}
    assert modal_app.parse_picks(" 3=2, 1 = 4 ", runs) == {3: 2, 1: 4}
    for bad in ["4=1", "1=5", "0=1", "one=2", "1:2"]:
        with pytest.raises(SystemExit, match="should be"):
            modal_app.parse_picks(bad, runs)
    with pytest.raises(SystemExit, match="starts from a photo"):
        modal_app.parse_picks("2=1", runs)


def test_make_set_runs_every_line_and_reports_failures(tmp_path, monkeypatch, capsys):
    listing = tmp_path / "set.txt"
    listing.write_text("a lamp\na chair\n")
    progress = json.dumps({"prompt": "a lamp", "steps": {"final": {"status": "done", "files": ["final-1.glb"]}}})
    stored = {"s-01-lamp/progress.json": progress.encode(), "s-01-lamp/final-1.glb": b"glb"}

    class Outputs(FakeOutputs):
        def listdir(self, path):
            return []

    calls = []

    class Download:
        def remote(self, which):
            calls.append(("download", which))

    class MakeModel:
        def starmap(self, arguments, return_exceptions):
            calls.append(("starmap", [args[:2] for args in arguments], return_exceptions))
            return iter([{"steps": {}}, RuntimeError("invalid input: no object found\ntraceback...")])

    monkeypatch.setattr(modal_app, "outputs", Outputs(stored))
    monkeypatch.setattr(modal_app, "download_models", Download())
    monkeypatch.setattr(modal_app, "make_model", MakeModel())
    monkeypatch.chdir(tmp_path)
    modal_app.make_set.info.raw_f(prompts=str(listing), final=True, name="s")

    # Weights once, up front (prompts need FLUX's too; finals by TRELLIS.2 nothing more), then every run
    # in one map
    assert calls == [
        ("download", "trellis2"),
        ("download", "reference"),
        ("starmap", [("s-01-lamp", "a lamp"), ("s-02-chair", "a chair")], True),
    ]
    assert (tmp_path / "orainge-outputs" / "s" / "s-01-lamp" / "final-1.glb").read_bytes() == b"glb"
    out = capsys.readouterr().out
    assert "s-01-lamp: final-1.glb" in out
    assert "s-02-chair: failed (RuntimeError: invalid input: no object found)" in out
    assert "Done: 1 of 2 runs finished" in out
    assert "make_set --prompts" in out and "--name s" in out


@pytest.mark.parametrize("final_weights", [(), ("pixal3d",)])
def test_a_set_of_photos_fetches_the_weights_its_finals_need(final_weights, tmp_path, monkeypatch):
    photo = tmp_path / "cat.png"
    photo.write_bytes(b"png")
    listing = tmp_path / "set.txt"
    listing.write_text("cat.png\n")
    calls = []

    class Outputs(FakeOutputs):
        def listdir(self, path):
            return []

    class Download:
        def remote(self, which):
            calls.append(which)

    class MakeModel:
        def starmap(self, arguments, return_exceptions):
            return iter([{"steps": {}}])

    monkeypatch.setattr(modal_app, "outputs", Outputs({}))
    monkeypatch.setattr(modal_app, "download_models", Download())
    monkeypatch.setattr(modal_app, "make_model", MakeModel())
    monkeypatch.setattr(modal_app, "FINAL_WEIGHTS", final_weights)  # deployed with ORAINGE_FINAL_MODEL=pixal3d, or not
    monkeypatch.chdir(tmp_path)
    modal_app.make_set.info.raw_f(prompts=str(listing), final=True, name="s")
    # No FLUX for photos; Pixal3D's after TRELLIS.2's (which it reuses) when Pixal3D makes the finals
    assert calls == ["trellis2", *final_weights]
    calls.clear()
    modal_app.make_set.info.raw_f(prompts=str(listing), final=False, name="s")
    assert calls == ["trellis2"]


def test_make_set_can_stop_at_the_pictures(tmp_path, monkeypatch, capsys):
    listing = tmp_path / "set.txt"
    listing.write_text("a lamp\na chair\n")
    seen = []

    class Outputs(FakeOutputs):
        def listdir(self, path):
            return []

    class Download:
        def remote(self, which):
            pass

    class MakeModel:
        def starmap(self, arguments, return_exceptions):
            seen.extend(args[-2:] for args in arguments)
            return iter([{"steps": {}}, {"steps": {}}])

    monkeypatch.setattr(modal_app, "outputs", Outputs({}))
    monkeypatch.setattr(modal_app, "download_models", Download())
    monkeypatch.setattr(modal_app, "make_model", MakeModel())
    monkeypatch.chdir(tmp_path)
    modal_app.make_set.info.raw_f(prompts=str(listing), final=True, name="s", pictures_only=True, picks="2=3")
    assert seen == [(0, True), (3, True)]
    assert '--name s --picks "1=2,3=4"' in capsys.readouterr().out


def test_tuning_results_are_merged_without_losing_any(tmp_path):
    shared, local = tmp_path / "shared" / "tuning.json", tmp_path / "local.json"
    local.write_text(json.dumps({"NVIDIA L40S": {"conv": {"(10, 64)": {"B": 64}}}, "NVIDIA A100": {"conv": {}}}))
    # Nothing shared yet: the first container's results become the shared ones
    assert modal_app.merge_tuning(shared, local)
    assert json.loads(shared.read_text()) == json.loads(local.read_text())
    assert not modal_app.merge_tuning(shared, local)  # nothing new, nothing written

    # Another container tuned a new shape and a new kernel meanwhile
    other = {"NVIDIA L40S": {"conv": {"(12, 64)": {"B": 128}}, "sample": {"(9,)": {"B": 32}}}}
    shared.write_text(json.dumps(other))
    assert modal_app.merge_tuning(shared, local)
    assert json.loads(shared.read_text())["NVIDIA L40S"] == {
        "conv": {"(12, 64)": {"B": 128}, "(10, 64)": {"B": 64}},
        "sample": {"(9,)": {"B": 32}},
    }
    # A missing or broken source changes nothing
    assert not modal_app.merge_tuning(shared, tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{")
    assert not modal_app.merge_tuning(shared, tmp_path / "broken.json")


def test_saving_the_kernel_caches_never_fails_a_job(tmp_path, monkeypatch, capsys):
    local, shared = tmp_path / "local.json", tmp_path / "cache" / "tuning.json"
    local.write_text(json.dumps({"NVIDIA L40S": {"conv": {"(10, 64)": {"B": 64}}}}))
    monkeypatch.setattr(modal_app, "LOCAL_TUNING", local)
    monkeypatch.setattr(modal_app, "SHARED_TUNING", shared)
    commits = []

    class Cache:
        def commit(self):
            commits.append(True)

        def reload(self):  # Modal refuses while Triton's launchers are open from the volume
            raise RuntimeError("there are open files preventing the operation")

    monkeypatch.setattr(modal_app, "cache", Cache())
    modal_app.share_caches()
    assert json.loads(shared.read_text()) == json.loads(local.read_text()) and commits == [True]

    class BrokenCache:
        def commit(self):
            raise RuntimeError("volume unavailable")

    monkeypatch.setattr(modal_app, "cache", BrokenCache())
    modal_app.share_caches()  # reported, not raised
    assert "kernel caches not saved: RuntimeError: volume unavailable" in capsys.readouterr().out
