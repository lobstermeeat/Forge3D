"""
Orainge's AI workers on Modal (https://modal.com): serverless GPUs billed by the second.

    modal run workers/modal_app.py::download_models      # once: weights into a Modal volume
    modal run workers/modal_app.py::try_prompt --prompt "a brass pocket watch"
    modal run workers/modal_app.py::try_image --image reference-123.png
    modal deploy workers/modal_app.py                    # the job API the server calls

The GPU code is the same as in the RunPod images (trellis2/, flux-schnell/); only the entry
points differ. The job API (job_api.py) speaks RunPod's protocol, so the server's client works
with both. One-time setup (Hugging Face access, Modal secrets) is in README.md.
"""

from __future__ import annotations

import base64
import os
import pathlib
from typing import Any, Callable

import modal

WORKERS = pathlib.Path(__file__).parent
MODELS = "/models"

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

app = modal.App("orainge-ai")
models = modal.Volume.from_name("orainge-models", create_if_missing=True)

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
    timeout=3600,
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


def _save(asset: dict, target: pathlib.Path) -> None:
    if asset.get("base64"):
        target.write_bytes(base64.b64decode(asset["base64"]))
    elif asset.get("url"):
        import urllib.request

        with urllib.request.urlopen(asset["url"], timeout=120) as response:
            target.write_bytes(response.read())
    else:
        raise SystemExit(f"{asset['key']} was stored without a public URL; set R2_PUBLIC_BASE_URL")


@app.local_entrypoint()
def try_prompt(prompt: str, count: int = 4, seed: int = -1) -> None:
    """Saves reference images for a prompt in the current folder."""
    job: dict[str, Any] = {"prompt": prompt, "count": count}
    if seed >= 0:
        job["seed"] = seed
    result = FluxSchnell().generate.remote({"input": job})
    if "error" in result:
        raise SystemExit(result["error"])
    for image in result["images"]:
        target = pathlib.Path(f"reference-{image['seed']}.png")
        _save(image, target)
        print(f"Saved {target}")
    first = result["images"][0]["seed"]
    print(f"Make one 3D: modal run workers/modal_app.py::try_image --image reference-{first}.png")


@app.local_entrypoint()
def try_image(image: str, mode: str = "preview", seed: int = -1) -> None:
    """Turns an image into a GLB saved next to it."""
    source = pathlib.Path(image)
    job: dict[str, Any] = {"image_base64": base64.b64encode(source.read_bytes()).decode("ascii"), "mode": mode}
    if seed >= 0:
        job["seed"] = seed
    result = Trellis2().generate.remote({"input": job})
    if "error" in result:
        raise SystemExit(result["error"])
    target = source.with_name(f"{source.stem}-{result['mode']}-{result['seed']}.glb")
    _save(result["glb"], target)
    print(
        f"Saved {target}: {result['triangles']:,} triangles, {result['bytes'] / 1e6:.1f} MB, "
        f"timings {result['timings']}"
    )
    if result["mode"] == "preview":
        print(
            "Same shape at final quality: modal run workers/modal_app.py::try_image "
            f"--image {image} --mode final --seed {result['seed']}"
        )
