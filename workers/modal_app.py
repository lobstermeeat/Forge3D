"""
Orainge's AI workers on Modal (https://modal.com): serverless GPUs billed by the second.

    modal run --detach workers/modal_app.py::make --prompt "a brass pocket watch" --final
    python workers/modal_app.py status                   # what's ready, what's running, what's left
    python workers/modal_app.py build                    # build the images ahead of time
    modal deploy workers/modal_app.py                    # the job API the server calls

`make` runs the whole flow in Modal's cloud (weights on first use, reference images, preview,
final) and saves every step in the orainge-outputs volume. With --detach it keeps going when
this computer sleeps or goes offline, and `--run NAME` continues a run from its last finished
step.

The GPU code is the same as in the RunPod images (trellis2/, flux-schnell/); only the entry
points differ. The job API (job_api.py) speaks RunPod's protocol, so the server's client works
with both. One-time setup (Hugging Face access, Modal secrets) is in README.md.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import time
from typing import Any, Callable, Optional

import modal

WORKERS = pathlib.Path(__file__).parent
MODELS = "/models"
OUTPUTS = "/outputs"
APP_NAME = "orainge-ai"

# Pinned exactly like trellis2/Dockerfile (tests/test_modal_app.py keeps them in sync)
TORCH = ("torch==2.6.0", "torchvision==0.21.0")
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
TRELLIS2_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CUMESH_COMMIT = "12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_COMMIT = "6dd94a859c26ee8246888502eada3dd8ad85532e"
GLTFPACK_VERSION = "1.3"

# 48 GB cards keep every TRELLIS.2 model on the GPU. An A10 (24 GB) costs about half as much per
# second but is slower and needs TRELLIS2_LOW_VRAM = "1"; 1024³ finals may run out of memory there.
TRELLIS2_GPU = "L40S"
TRELLIS2_LOW_VRAM = "0"
FLUX_GPU = "L40S"  # FLUX.1 [schnell] needs about 34 GB

app = modal.App(APP_NAME)
models = modal.Volume.from_name("orainge-models", create_if_missing=True)
# What `make` produces, one folder per run, so results outlive the computer that asked for them
outputs = modal.Volume.from_name("orainge-outputs", create_if_missing=True)

# Results are returned inline (base64, 8 MB limit) unless R2 is configured: create a Modal secret
# with the R2_* variables (see README) and deploy with ORAINGE_R2_SECRET set to its name.
R2_SECRET = os.environ.get("ORAINGE_R2_SECRET")
storage_secrets = [modal.Secret.from_name(R2_SECRET)] if R2_SECRET else []
if modal.is_local():
    print(f"[orainge] results: {f'R2 (secret {R2_SECRET})' if R2_SECRET else 'inline; set ORAINGE_R2_SECRET for R2'}")


def _git(url: str, commit: str, target: str) -> str:
    return (
        f"git clone {url} {target} && git -C {target} checkout {commit}"
        f" && git -C {target} submodule update --init --recursive"
    )


trellis2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .entrypoint([])  # skip the CUDA image's banner script
    .apt_install("git", "curl", "unzip", "build-essential", "libgl1", "libglib2.0-0", "libjpeg-dev")
    .pip_install(*TORCH, index_url=TORCH_INDEX)
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install_from_requirements(str(WORKERS / "trellis2" / "requirements.txt"))
    # CUDA extensions are compiled for Ampere, Ada and Hopper (A10, A100, L4, L40S, H100)
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "MAX_JOBS": "4"})
    # TRELLIS.2 and its CUDA extensions (all MIT). nvdiffrast is deliberately not installed:
    # its license allows research use only (see trellis2/forge3d_worker/uv_raster.py).
    .run_commands(
        _git("https://github.com/microsoft/TRELLIS.2", TRELLIS2_COMMIT, "/opt/trellis2"),
        # Importable as `trellis2` without touching PYTHONPATH, which Modal manages
        'echo /opt/trellis2 > "$(python -c "import site; print(site.getsitepackages()[0])")/trellis2-repo.pth"',
    )
    .run_commands(
        _git("https://github.com/JeffreyXiang/CuMesh", CUMESH_COMMIT, "/tmp/CuMesh"),
        "python -m pip install /tmp/CuMesh --no-build-isolation && rm -rf /tmp/CuMesh",
    )
    .run_commands(
        _git("https://github.com/JeffreyXiang/FlexGEMM", FLEXGEMM_COMMIT, "/tmp/FlexGEMM"),
        "python -m pip install /tmp/FlexGEMM --no-build-isolation && rm -rf /tmp/FlexGEMM",
    )
    .run_commands("python -m pip install /opt/trellis2/o-voxel --no-build-isolation")
    # gltfpack (meshoptimizer, MIT; embeds Basis Universal, Apache-2.0) for meshopt + KTX2
    .run_commands(
        "curl -fsSL -o /tmp/gltfpack.zip https://github.com/zeux/meshoptimizer/releases/download/"
        f"v{GLTFPACK_VERSION}/gltfpack-ubuntu.zip && unzip /tmp/gltfpack.zip -d /usr/local/bin"
        " && chmod +x /usr/local/bin/gltfpack && rm /tmp/gltfpack.zip"
    )
    .env(
        {
            "ATTN_BACKEND": "flash_attn",
            "HF_HUB_OFFLINE": "1",
            "TRELLIS2_MODEL_DIR": f"{MODELS}/TRELLIS.2-4B",
            "TRELLIS2_LOW_VRAM": TRELLIS2_LOW_VRAM,
        }
    )
    .add_local_dir(WORKERS / "trellis2" / "forge3d_worker", "/root/forge3d_worker")
)

flux_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(TORCH[0], index_url=TORCH_INDEX)
    .pip_install_from_requirements(str(WORKERS / "flux-schnell" / "requirements.txt"))
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(WORKERS / "flux-schnell" / "reference_worker", "/root/reference_worker")
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.34,<2")
    .add_local_file(WORKERS / "trellis2" / "scripts" / "download_weights.py", "/root/weights/trellis2.py")
    .add_local_file(
        WORKERS / "flux-schnell" / "scripts" / "download_weights.py", "/root/weights/reference.py"
    )
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]>=0.115,<1")
    .add_local_file(WORKERS / "job_api.py", "/root/job_api.py")
)

# For `make`, which only coordinates the GPU workers
light_image = modal.Image.debian_slim(python_version="3.11")


def run_job(handle: Callable[[dict], dict], job: dict) -> dict:
    """Runs one job in a GPU container."""
    from modal.experimental import stop_fetching_inputs

    # The call id doubles as the fallback request id, so stored files match the API's job ids
    result = handle({"id": modal.current_function_call_id() or "job", **job})
    if result.pop("refresh_worker", False):
        # CUDA may be unusable after a GPU fault: finish this job, then Modal starts a fresh container
        stop_fetching_inputs()
    return result


def _weights_marker(name: str) -> pathlib.Path:
    """Written by download_models after a complete download (a failed one leaves partial files)."""
    return pathlib.Path(MODELS) / f".{name}-weights"


def _require_weights(name: str) -> None:
    if not _weights_marker(name).exists():
        raise RuntimeError(
            f"The {name} weights are missing or incomplete in the orainge-models volume. "
            f"Run `modal run workers/modal_app.py::download_models --which {name}` first."
        )


@app.cls(
    image=trellis2_image,
    gpu=TRELLIS2_GPU,
    cpu=4.0,
    memory=16384,
    volumes={MODELS: models},
    secrets=storage_secrets,
    timeout=600,
    startup_timeout=600,
    scaledown_window=60,  # idle containers are billed; a cold start takes about a minute
    max_containers=2,  # caps spending; raise it for more parallel jobs
)
class Trellis2:
    """Image to textured, web-packed GLB. Input and output as in README.md (Job contracts)."""

    @modal.enter()
    def load(self) -> None:
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import Trellis2Runtime
        from forge3d_worker.service import handle_job
        from forge3d_worker.storage import storage_from_env

        _require_weights("trellis2")
        storage = storage_from_env()
        runtime = Trellis2Runtime(f"{MODELS}/TRELLIS.2-4B")
        self.handle = lambda job: handle_job(job, runtime, storage, pack_glb)

    @modal.method()
    def generate(self, job: dict) -> dict:
        return run_job(self.handle, job)


@app.cls(
    image=flux_image,
    gpu=FLUX_GPU,
    cpu=2.0,
    memory=32768,  # the weights pass through RAM on their way to the GPU
    volumes={MODELS: models},
    secrets=storage_secrets,
    timeout=300,
    startup_timeout=600,
    scaledown_window=60,
    max_containers=1,
)
class FluxSchnell:
    """Text prompt to reference images (FLUX.1 [schnell], Apache-2.0)."""

    @modal.enter()
    def load(self) -> None:
        from reference_worker.service import handle_job, make_flux_generator
        from reference_worker.storage import storage_from_env

        _require_weights("reference")
        storage = storage_from_env()
        generate = make_flux_generator(f"{MODELS}/FLUX.1-schnell", cpu_offload=False)
        self.handle = lambda job: handle_job(job, generate, storage)

    @modal.method()
    def generate(self, job: dict) -> dict:
        return run_job(self.handle, job)


WEIGHT_SCRIPTS = {"trellis2": "/root/weights/trellis2.py", "reference": "/root/weights/reference.py"}


@app.function(
    image=download_image,
    volumes={MODELS: models},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    timeout=3 * 3600,  # generous for slow Hugging Face transfers; an interrupted run resumes
)
def download_models(which: str = "all", force: bool = False) -> None:
    """Downloads the pinned weights (about 50 GB) into the orainge-models volume. CPU only."""
    if which != "all" and which not in WEIGHT_SCRIPTS:
        raise SystemExit("--which must be all, trellis2 or reference")
    for name in WEIGHT_SCRIPTS if which == "all" else [which]:
        _download(name, force)
        models.commit()


def _download(name: str, force: bool) -> None:
    import hashlib
    import subprocess
    import sys

    script = WEIGHT_SCRIPTS[name]
    # Each script pins every revision, so its hash identifies what the volume holds
    digest = hashlib.sha256(pathlib.Path(script).read_bytes()).hexdigest()
    marker = _weights_marker(name)
    if not force and marker.exists() and marker.read_text() == digest:
        print(f"{name}: already downloaded (--force to fetch again)")
        return
    # Modal keeps whatever a failed run wrote, so the marker must not outlive a partial download
    marker.unlink(missing_ok=True)
    env = {**os.environ, "MODELS_ROOT": MODELS, "FLUX_MODEL_DIR": f"{MODELS}/FLUX.1-schnell"}
    subprocess.run([sys.executable, script], env=env, check=True)
    marker.write_text(digest)


@app.function(
    image=api_image,
    secrets=[modal.Secret.from_name("orainge-worker-token", required_keys=["ORAINGE_WORKER_TOKEN"])],
    scaledown_window=300,
    max_containers=3,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def api():
    """https://<workspace>--orainge-ai-api.modal.run: set it as the server's AI_WORKERS_URL."""
    from job_api import ModalCalls, create_app

    calls = ModalCalls({"trellis2": Trellis2().generate, "reference": FluxSchnell().generate})
    return create_app(os.environ["ORAINGE_WORKER_TOKEN"], calls)


RUN_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _asset_bytes(asset: dict) -> bytes:
    """A stored file as the workers report it: inline base64, or a public R2 URL."""
    if asset.get("base64"):
        return base64.b64decode(asset["base64"])
    if asset.get("url"):
        import urllib.request

        with urllib.request.urlopen(asset["url"], timeout=120) as response:
            return response.read()
    raise RuntimeError(f"{asset.get('key')} was stored without a public URL; set R2_PUBLIC_BASE_URL")


def run_pipeline(
    folder: pathlib.Path,
    *,
    prompt: Optional[str],
    image: Optional[bytes],
    image_name: str,
    final: bool,
    seed: Optional[int],
    ensure_weights: Callable[[str], Any],
    reference: Callable[[dict], dict],
    trellis: Callable[[dict], dict],
    fetch: Callable[[dict], bytes],
    save: Callable[[], Any],
) -> dict:
    """
    Prompt or image -> reference images -> preview GLB -> (optionally) final GLB, written into
    `folder`. Each finished step is recorded in progress.json, so running it again on the same
    folder skips what is done and continues where it stopped.
    """
    folder.mkdir(parents=True, exist_ok=True)
    progress_file = folder / "progress.json"
    run = folder.name
    if progress_file.exists():
        # A continued run keeps its prompt, image and seed; it can only add the final step
        state: dict = json.loads(progress_file.read_text())
        if (prompt and prompt != state.get("prompt")) or image is not None or (
            seed is not None and seed != state.get("seed")
        ):
            raise ValueError(f"{run} already exists; start a new run to change its prompt, image or seed")
    else:
        if not prompt and image is None:
            raise ValueError("a run needs a prompt or an image")
        state = {"prompt": prompt, "seed": seed, "steps": {}}
        if image is not None:
            state["input"] = f"input-{image_name or 'image'}"
            (folder / state["input"]).write_bytes(image)
    state["final"] = final = final or bool(state.get("final"))
    prompt = state.get("prompt")
    steps: dict = state["steps"]

    def record(step: str, **fields: Any) -> None:
        steps[step] = {**steps.get(step, {}), **fields}
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        progress_file.write_text(json.dumps(state, indent=2))
        save()

    def done(step: str) -> bool:
        entry = steps.get(step, {})
        return entry.get("status") == "done" and all((folder / f).exists() for f in entry.get("files", []))

    def attempt(step: str, work: Callable[[], dict]) -> None:
        if done(step):
            return
        record(step, status="running", error=None)
        started = time.monotonic()
        try:
            fields = work()
        except Exception as err:
            record(step, status="failed", error=f"{type(err).__name__}: {err}")
            raise
        record(step, status="done", seconds=round(time.monotonic() - started, 1), **fields)

    def checked(result: dict) -> dict:
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def weights() -> dict:
        for name in ["trellis2", "reference"] if prompt else ["trellis2"]:
            ensure_weights(name)
        return {}

    def reference_images() -> dict:
        result = checked(reference({"input": {"prompt": prompt, "count": 4, "request_id": run}}))
        files = []
        for picture in result["images"]:
            name = f"reference-{picture['seed']}.png"
            (folder / name).write_bytes(fetch(picture))
            files.append(name)
        # The first picture goes on to 3D; `--image` on another one makes that one instead
        state["input"] = files[0]
        return {"files": files, "gpu_seconds": result.get("seconds")}

    def model(mode: str) -> Callable[[], dict]:
        def work() -> dict:
            job: dict = {
                "image_base64": base64.b64encode((folder / state["input"]).read_bytes()).decode("ascii"),
                "mode": mode,
                "request_id": run,
            }
            if state.get("seed") is not None:
                job["seed"] = state["seed"]
            result = checked(trellis({"input": job}))
            # The final reuses the preview's seed, so it refines the shape the preview showed
            state["seed"] = result["seed"]
            name = f"{mode}-{result['seed']}.glb"
            (folder / name).write_bytes(fetch(result["glb"]))
            return {
                "files": [name],
                "triangles": result["triangles"],
                "bytes": result.get("bytes"),
                "timings": result["timings"],
                "gpu_seconds": round(sum(result["timings"].values()), 1),
            }

        return work

    attempt("weights", weights)
    if prompt:
        attempt("reference", reference_images)
    attempt("preview", model("preview"))
    if final:
        attempt("final", model("final"))
    return state


@app.function(image=light_image, volumes={OUTPUTS: outputs}, timeout=4 * 3600)
def make_model(
    run: str, prompt: str = "", image: bytes = b"", image_name: str = "", final: bool = False, seed: int = -1
) -> dict:
    """The whole flow in the cloud; see run_pipeline. Results land in orainge-outputs/<run>/."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError("the run name must be 1-64 letters, digits, '-' or '_'")
    return run_pipeline(
        pathlib.Path(OUTPUTS) / run,
        prompt=prompt or None,
        image=image or None,
        image_name=pathlib.Path(image_name).name,
        final=final,
        seed=seed if seed >= 0 else None,
        ensure_weights=lambda name: download_models.remote(which=name),
        reference=lambda job: FluxSchnell().generate.remote(job),
        trellis=lambda job: Trellis2().generate.remote(job),
        fetch=_asset_bytes,
        save=outputs.commit,
    )


@app.local_entrypoint()
def make(prompt: str = "", image: str = "", final: bool = False, seed: int = -1, run: str = "") -> None:
    """
    Prompt or image to 3D in Modal's cloud. Start it with `modal run --detach` and it finishes
    even if this computer sleeps or goes offline; `--run NAME` continues an earlier run.
    """
    if not (prompt or image or run):
        raise SystemExit("Give --prompt or --image, or --run NAME to continue a run")
    if prompt and image:
        raise SystemExit("Give either --prompt or --image, not both")
    run = run or time.strftime("run-%Y%m%d-%H%M%S")
    if not RUN_NAME.fullmatch(run):
        raise SystemExit("--run must be 1-64 letters, digits, '-' or '_'")
    source = pathlib.Path(image) if image else None
    print(f"[orainge] {run}: saving each step in the orainge-outputs volume under {run}/")
    print("  Progress, any time:  python workers/modal_app.py status")
    print(f"  Pick up an unfinished run:  modal run --detach workers/modal_app.py::make --run {run}")
    state = make_model.remote(
        run=run,
        prompt=prompt,
        image=source.read_bytes() if source else b"",
        image_name=source.name if source else "",
        final=final,
        seed=seed,
    )
    # Still connected: copy the results here too
    target = pathlib.Path("orainge-outputs") / run
    target.mkdir(parents=True, exist_ok=True)
    files = [f for step in state["steps"].values() for f in step.get("files", [])]
    for name in files:
        with open(target / name, "wb") as f:
            outputs.read_file_into_fileobj(f"{run}/{name}", f)
    print(f"Done: {', '.join(files)} in {target}")
    if not state.get("final"):
        print(f"Final quality, same shape: modal run --detach workers/modal_app.py::make --run {run} --final")


def print_status() -> None:
    """Weights, deployment and recent runs, read straight from Modal (nothing is built or started)."""
    import modal.exception as mx
    from modal.types import FileEntryType

    def names(volume: str, path: str = "/") -> list:
        try:
            return modal.Volume.from_name(volume).listdir(path)
        except mx.NotFoundError:
            return []

    ready = {entry.path.strip("/") for entry in names("orainge-models")}
    for name in WEIGHT_SCRIPTS:
        print(f"Weights {name}: {'ready' if f'.{name}-weights' in ready else 'not downloaded yet'}")
    try:
        print(f"Job API: {modal.Function.from_name(APP_NAME, 'api').get_web_url()}")
    except mx.NotFoundError:
        print("Job API: not deployed (modal deploy workers/modal_app.py)")

    runs = sorted(
        (entry for entry in names("orainge-outputs") if entry.type == FileEntryType.DIRECTORY),
        key=lambda entry: entry.mtime,
    )[-10:]
    if not runs:
        print("Runs: none yet")
    volume = modal.Volume.from_name("orainge-outputs")
    for entry in runs:
        run = entry.path.strip("/")
        try:
            state = json.loads(b"".join(volume.read_file(f"{run}/progress.json")))
        except (mx.NotFoundError, FileNotFoundError, ValueError):
            print(f"{run}: no progress recorded")
            continue
        steps = ", ".join(f"{step} {info.get('status')}" for step, info in state["steps"].items())
        print(f"{run} ({state.get('prompt') or state.get('input')}): {steps}; updated {state.get('updated')}")
        for step, info in state["steps"].items():
            if info.get("status") == "failed":
                print(f"  {step} failed: {info.get('error')}")
                print(f"  Continue it: modal run --detach workers/modal_app.py::make --run {run}")
            if info.get("status") == "running":
                print("  (a step marked running with no recent update was interrupted; continue it the same way)")
        print(f"  Download: modal volume get orainge-outputs {run} .")


def build_images() -> None:
    """Builds every image ahead of the first run, so a later deploy or run starts at once.
    Needs no secrets, so it can run while Hugging Face access is still pending."""
    builder = modal.App.lookup("orainge-ai-images", create_if_missing=True)
    images = [("download", download_image), ("api", api_image), ("reference", flux_image), ("trellis2", trellis2_image)]
    with modal.enable_output():
        for name, image in images:
            started = time.monotonic()
            image.build(builder)
            print(f"[orainge] {name} image ready after {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    import sys

    commands = {"status": print_status, "build": build_images}
    if len(sys.argv) == 2 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        print(__doc__)
