"""modal_app.py without Modal's servers: its pins, its shape and the per-job wrapper."""

import base64
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


def test_defines_both_workers_the_api_and_the_helpers():
    assert isinstance(modal_app.Trellis2, modal.Cls)
    assert isinstance(modal_app.FluxSchnell, modal.Cls)
    assert isinstance(modal_app.download_models, modal.Function)
    assert isinstance(modal_app.api, modal.Function)
    assert isinstance(modal_app.make_model, modal.Function)
    assert isinstance(modal_app.make, modal.app.LocalEntrypoint)


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
        return {
            "seed": seed,
            "mode": mode,
            "glb": {"key": "k", "url": None, "base64": base64.b64encode(f"glb-{mode}".encode()).decode()},
            "triangles": 30_000 if mode == "preview" else 100_000,
            "timings": {"generate_s": 1.0},
        }


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
    state = run(folder, workers, prompt="a brass pocket watch", final=True)

    assert workers.calls[:2] == [("weights", "trellis2"), ("weights", "reference")]
    assert workers.calls[2] == (
        "reference",
        {"input": {"prompt": "a brass pocket watch", "count": 4, "request_id": "run-1"}},
    )
    preview, final = workers.calls[3][1]["input"], workers.calls[4][1]["input"]
    # The first reference picture goes to 3D, and the final reuses the preview's seed
    assert base64.b64decode(preview["image_base64"]) == b"png-5"
    assert (preview["mode"], preview["request_id"], "seed" in preview) == ("preview", "run-1", False)
    assert (final["mode"], final["seed"]) == ("final", 42)

    assert (folder / "reference-5.png").read_bytes() == b"png-5"
    assert (folder / "preview-42.glb").read_bytes() == b"glb-preview"
    assert (folder / "final-42.glb").read_bytes() == b"glb-final"
    saved = json.loads((folder / "progress.json").read_text())
    assert saved == state
    assert {step: info["status"] for step, info in saved["steps"].items()} == {
        "weights": "done",
        "reference": "done",
        "preview": "done",
        "final": "done",
    }
    assert saved["steps"]["final"]["triangles"] == 100_000


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
