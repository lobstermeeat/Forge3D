"""modal_app.py without Modal's servers: its pins, its shape and the per-job wrapper."""

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
    assert isinstance(modal_app.try_image, modal.app.LocalEntrypoint)
    assert isinstance(modal_app.try_prompt, modal.app.LocalEntrypoint)


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
