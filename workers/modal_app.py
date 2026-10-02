"""
Orainge's AI workers on Modal (https://modal.com): serverless GPUs billed by the second.

    modal run --detach workers/modal_app.py::make --prompt "a brass pocket watch" --final
    modal run --detach workers/modal_app.py::make_set --prompts workers/test-sets/starter.txt
    python workers/modal_app.py status                   # what's ready, what's running, what's left
    python workers/modal_app.py build                    # build the images ahead of time
    modal deploy workers/modal_app.py                    # the job API the server calls

`make` runs the whole flow in Modal's cloud (weights on first use, reference images, preview,
final) and saves every step in the orainge-outputs volume. With --detach it keeps going when
this computer sleeps or goes offline, and `--run NAME` continues a run from its last finished
step. `make_set` does the same for every line of a test set, and workers/gallery/ turns the
results into a review page.

Previews are TRELLIS.2's '512' pipeline, and so are finals ('1024_cascade') unless the app is deployed
with ORAINGE_FINAL_MODEL=pixal3d: then finals go through the Pixal3D recipe (README.md, "The recipe"),
TRELLIS.2's '512' preview at the job's seed, the picture's camera found against it, Pixal3D's
multi-view weights with the picture as their one view (the single-view weights for a thin, flat
object), levelled, then the usual export. Both run in the Trellis2 container, so the server's contract
(worker "trellis2", mode "preview" | "final") is the same either way. Phase 6's re-test kept TRELLIS.2
as the default: the recipe fixed made-up backs but lost more on textures (README.md, "The recipe").
Texture options (mode "textures": more textures for a final's shape) are TRELLIS.2's alone, so with the
recipe on they are refused. A textures job that says "judge": true also asks the judge, Judge8B (Qwen3-VL
picking the texture a creator would rather use; judge/judge_worker/), which the Trellis2 container calls.

Experiments call more workers that production never does: GeometryViews (MV-Adapter's views of a given
mesh) and Judge30B (the judge's larger size).

The GPU code is the same as in the RunPod images (trellis2/, flux-schnell/); only the entry
points differ. The job API (job_api.py) speaks RunPod's protocol, so the server's client works
with both. One-time setup (Hugging Face access, Modal secrets) is in README.md.
"""

from __future__ import annotations

import base64
import functools
import json
import os
import pathlib
import re
import threading
import time
import traceback
from typing import Any, Callable, Collection, Optional, Sequence

import modal

WORKERS = pathlib.Path(__file__).parent
MODELS = "/models"
OUTPUTS = "/outputs"
# Production's app. ORAINGE_APP_NAME runs the same code under another name, for a staging copy that
# never touches production's deployment (a re-test as an ephemeral `modal run`, say)
APP_NAME = os.environ.get("ORAINGE_APP_NAME", "orainge-ai")

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
# Pixal3D (TencentARC, MIT since 2026-05-21) makes the finals: pixel-aligned image features on TRELLIS.2's
# models and decoders. Pinned as pixal3d/NOTICE.md records, with MoGe-2 (the picture's field of view),
# the utils3d it pins, and NAF (DINOv3 feature upsampling); NATTEN is not installed (pixal3d_worker/
# neighborhood.py computes its one call in PyTorch), nor nvdiffrast or RMBG-2.0.
PIXAL3D_COMMIT = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"
NAF_COMMIT = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
MOGE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"  # MoGe-2, before MoGe-3 changed its dependencies
UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"  # what that MoGe pins
# Pixal3D's weight sets download_models fetches: the recipe needs both (the multi-view set for most
# pictures, the single-view set for thin, flat objects: pixal3d_worker/thin.py)
PIXAL3D_WEIGHTS = "all"

# Which model makes finals: TRELLIS.2 (the default), or Pixal3D (the recipe), which Phase 6's re-test
# found no better overall (README.md, "The recipe"). Read when the app is deployed and baked into the
# images, so switching is one env change:
#     ORAINGE_FINAL_MODEL=pixal3d modal deploy workers/modal_app.py
# Previews are always TRELLIS.2's, and so are finals while Pixal3D's weights or models can't be loaded
# (the result's "fallback" says why).
FINAL_MODELS = ("trellis2", "pixal3d")
FINAL_MODEL = os.environ.get("ORAINGE_FINAL_MODEL", FINAL_MODELS[0])
if FINAL_MODEL not in FINAL_MODELS:
    raise SystemExit(f"ORAINGE_FINAL_MODEL must be one of {', '.join(FINAL_MODELS)}, not {FINAL_MODEL!r}")
# The weights a final needs besides TRELLIS.2's
FINAL_WEIGHTS = ("pixal3d",) if FINAL_MODEL == "pixal3d" else ()

# 48 GB cards keep every TRELLIS.2 model on the GPU. An A10 (24 GB) costs about half as much per
# second but is slower and needs TRELLIS2_LOW_VRAM = "1"; 1024³ finals may run out of memory there.
TRELLIS2_GPU = "L40S"
TRELLIS2_LOW_VRAM = "0"
FLUX_GPU = "L40S"  # FLUX.1 [schnell] needs about 34 GB
MULTIVIEW_GPU = "A10G"  # MV-Adapter on SDXL needs about 14 GB: a 24 GB A10G, about half an L40S's price
# The judge: Qwen3-VL-8B is 17.5 GB in bf16 (an A10G might do, untested; an L40S leaves room, and is where
# Phase 7 measured it); Qwen3-VL-30B-A3B, the experiments' larger size, is 62 GB, so an H100 (80 GB)
JUDGE_GPUS = {"8b": "L40S", "30b": "H100"}

app = modal.App(APP_NAME)
models = modal.Volume.from_name("orainge-models", create_if_missing=True)
# What `make` produces, one folder per run, so results outlive the computer that asked for them
outputs = modal.Volume.from_name("orainge-outputs", create_if_missing=True)
# Compiled Triton kernels and FlexGEMM's kernel tuning, kept between containers. Without them each
# new container spends most of its first minute compiling and benchmarking kernel variants.
cache = modal.Volume.from_name("orainge-cache", create_if_missing=True)
CACHE = "/cache"

# Results are returned inline (base64, 8 MB limit) unless R2 is configured: create a Modal secret
# with the R2_* variables (see README) and deploy with ORAINGE_R2_SECRET set to its name.
R2_SECRET = os.environ.get("ORAINGE_R2_SECRET")
storage_secrets = [modal.Secret.from_name(R2_SECRET)] if R2_SECRET else []
if modal.is_local():
    print(f"[orainge] results: {f'R2 (secret {R2_SECRET})' if R2_SECRET else 'inline; set ORAINGE_R2_SECRET for R2'}")
    other = "trellis2" if FINAL_MODEL == "pixal3d" else "pixal3d"
    print(f"[orainge] finals: {FINAL_MODEL} (ORAINGE_FINAL_MODEL={other} for the other)")


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
    # --no-build-isolation builds use this environment's tools, and bdist_wheel needs `wheel`
    .run_commands("python -m pip install --upgrade pip 'setuptools>=70' wheel")
    # add_python's interpreter was built with clang, so setuptools would link with clang++, which
    # this image doesn't have. Compile and link with the image's gcc, as PyTorch itself is built.
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "LDSHARED": "gcc -pthread -shared",
            "LDCXXSHARED": "g++ -pthread -shared",
        }
    )
    .run_commands(
        _git("https://github.com/JeffreyXiang/CuMesh", CUMESH_COMMIT, "/tmp/CuMesh"),
        "python -m pip install /tmp/CuMesh --no-build-isolation --no-deps && rm -rf /tmp/CuMesh",
    )
    .run_commands(
        _git("https://github.com/JeffreyXiang/FlexGEMM", FLEXGEMM_COMMIT, "/tmp/FlexGEMM"),
        "python -m pip install /tmp/FlexGEMM --no-build-isolation --no-deps && rm -rf /tmp/FlexGEMM",
    )
    # --no-deps: o-voxel lists CuMesh and FlexGEMM as unpinned git dependencies, which would
    # replace the pinned builds above
    .run_commands("python -m pip install /opt/trellis2/o-voxel --no-build-isolation --no-deps")
    # gltfpack (meshoptimizer, MIT; embeds Basis Universal, Apache-2.0) for meshopt + KTX2
    .run_commands(
        "curl -fsSL -o /tmp/gltfpack.zip https://github.com/zeux/meshoptimizer/releases/download/"
        f"v{GLTFPACK_VERSION}/gltfpack-ubuntu.zip && unzip /tmp/gltfpack.zip -d /usr/local/bin"
        " && chmod +x /usr/local/bin/gltfpack && rm /tmp/gltfpack.zip"
    )
    # Pixal3D (MIT) and NAF (Apache-2.0) at pinned commits, Pixal3D importable as `pixal3d`. The same
    # steps as the Phase 6 experiments' image (ops/exp_pixal3d.py on ai-ops-pixal3d), so Modal reuses them
    .run_commands(
        _git("https://github.com/TencentARC/Pixal3D", PIXAL3D_COMMIT, "/opt/pixal3d"),
        'echo /opt/pixal3d > "$(python -c "import site; print(site.getsitepackages()[0])")/pixal3d-repo.pth"',
        _git("https://github.com/valeoai/NAF", NAF_COMMIT, "/opt/naf"),
    )
    # MoGe-2 and the utils3d it pins (both MIT). --no-deps: their requirements would replace pinned
    # packages (MoGe lists opencv-python, gradio and more that inference never imports)
    .run_commands(
        "python -m pip install --no-deps"
        f" 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT}'"
        f" 'moge @ git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT}'"
    )
    .pip_install_from_requirements(str(WORKERS / "pixal3d" / "requirements.txt"))
    .env(
        {
            "ATTN_BACKEND": "flash_attn",
            "HF_HUB_OFFLINE": "1",
            "TRELLIS2_MODEL_DIR": f"{MODELS}/TRELLIS.2-4B",
            "TRELLIS2_LOW_VRAM": TRELLIS2_LOW_VRAM,
            "TRITON_CACHE_DIR": f"{CACHE}/triton",
            "PIXAL3D_MODEL_DIR": f"{MODELS}/pixal3d",
            "PIXAL3D_DINO_DIR": f"{MODELS}/dinov3-vitl16",
            "PIXAL3D_NAF_DIR": "/opt/naf",
            "ORAINGE_FINAL_MODEL": FINAL_MODEL,
        }
    )
    .add_local_dir(WORKERS / "trellis2" / "forge3d_worker", "/root/forge3d_worker")
    .add_local_dir(WORKERS / "pixal3d" / "pixal3d_worker", "/root/pixal3d_worker")
)

flux_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(TORCH[0], index_url=TORCH_INDEX)
    .pip_install_from_requirements(str(WORKERS / "flux-schnell" / "requirements.txt"))
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(WORKERS / "flux-schnell" / "reference_worker", "/root/reference_worker")
)

multiview_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*TORCH, index_url=TORCH_INDEX)
    .pip_install_from_requirements(str(WORKERS / "multiview" / "requirements.txt"))
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(WORKERS / "multiview" / "multiview_worker", "/root/multiview_worker")
    # MV-Adapter's pipeline code (Apache-2.0), vendored without its nvdiffrast-based mesh tools
    .add_local_dir(WORKERS / "multiview" / "mvadapter", "/root/mvadapter")
)

# The judge: Qwen3-VL through transformers, on MultiView's base (the same Python and torch, so Modal reuses
# those layers)
judge_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*TORCH, index_url=TORCH_INDEX)
    .pip_install_from_requirements(str(WORKERS / "judge" / "requirements.txt"))
    .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(WORKERS / "judge" / "judge_worker", "/root/judge_worker")
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.34,<2")
    .add_local_file(WORKERS / "trellis2" / "scripts" / "download_weights.py", "/root/weights/trellis2.py")
    .add_local_file(
        WORKERS / "flux-schnell" / "scripts" / "download_weights.py", "/root/weights/reference.py"
    )
    .add_local_file(WORKERS / "multiview" / "scripts" / "download_weights.py", "/root/weights/multiview.py")
    .add_local_file(WORKERS / "pixal3d" / "scripts" / "download_weights.py", "/root/weights/pixal3d.py")
    .add_local_file(WORKERS / "judge" / "scripts" / "download_weights.py", "/root/weights/judge.py")
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]>=0.115,<1")
    .add_local_file(WORKERS / "job_api.py", "/root/job_api.py")
)

# For `make`, which only coordinates the GPU workers
# make_model runs here, and fetches Pixal3D's weights for finals only when they make them
light_image = modal.Image.debian_slim(python_version="3.11").env({"ORAINGE_FINAL_MODEL": FINAL_MODEL})


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


# FlexGEMM keeps its tuning results in a local file (device -> kernel -> input shape -> config)
LOCAL_TUNING = pathlib.Path.home() / ".flex_gemm" / "autotune_cache.json"
SHARED_TUNING = pathlib.Path(CACHE) / "flex_gemm" / "autotune_cache.json"


def merge_tuning(into: pathlib.Path, source: pathlib.Path) -> bool:
    """Adds FlexGEMM tuning results from `source` to `into`. Returns whether `into` changed."""
    try:
        found = json.loads(source.read_text())
    except (OSError, ValueError):
        return False
    try:
        merged = json.loads(into.read_text())
    except (OSError, ValueError):
        merged = {}
    changed = False
    for device, kernels in found.items():
        for kernel, shapes in kernels.items():
            known = merged.setdefault(device, {}).setdefault(kernel, {})
            for shape, config in shapes.items():
                if known.get(shape) != config:
                    known[shape] = config
                    changed = True
    if changed:
        into.parent.mkdir(parents=True, exist_ok=True)
        partial = into.with_name(into.name + ".partial")
        partial.write_text(json.dumps(merged))
        partial.replace(into)
    return changed


def share_caches() -> None:
    """
    After a job: saves new compiled kernels and tuning results for the containers to come. There is
    no reload first: Triton keeps its compiled launchers open from the volume, which Modal refuses
    to reload under, so results another container saved meanwhile are merged in by the next one.
    """
    try:
        merge_tuning(SHARED_TUNING, LOCAL_TUNING)
        cache.commit()  # also saves the kernels Triton wrote straight into the volume
    except Exception as err:  # noqa: BLE001 - only speed is at stake, so this never fails a job
        print(f"[orainge] kernel caches not saved: {type(err).__name__}: {err}")


# --- Two models in one container: TRELLIS.2 for previews, Pixal3D for finals --------------------------


def choose_model(job: dict, final_model: str, loaded: Collection[str]) -> str:
    """
    Which model makes a job. Previews are TRELLIS.2's ('512', quick; the final keeps the preview's seed).
    Finals are the final model's (FINAL_MODEL) when it is loaded or still loading, else TRELLIS.2's. A
    final carrying ``views`` (the Studio's AI_MULTIVIEW step, off by default) is Pixal3D's too: its
    multi-view weights build from the views (the Pixal3D worker's contract), while TRELLIS.2 steers its
    flows with them for the preview. Texture options ("textures") are TRELLIS.2's (handle_with_models
    refuses them when the finals are another model's).
    """
    payload = job.get("input")
    payload = payload if isinstance(payload, dict) else {}
    if payload.get("mode", "final") != "final":
        return "trellis2"
    return final_model if final_model in loaded else "trellis2"


class ModelPool:
    """
    A container's runtimes. TRELLIS.2 is built first and resident on the GPU, so previews start as soon
    as it is up; Pixal3D is built in a background thread (its models take about 90 s to load) and the
    first final waits for it. Pixal3D's finals peaked at 28-31 GB on an L40S (48 GB) with its weights
    resident, which leaves no room for TRELLIS.2's 1024 cascade beside them, so when Pixal3D takes the
    first final TRELLIS.2 goes to sleep (its models to RAM, upstream's low-VRAM mode) and stays there:
    previews then cost a few seconds more, and that is the state the Pixal3D recipe runs TRELLIS.2 in
    anyway for its '512' preview. Nothing is swapped back and forth between jobs.
    """

    def __init__(self, trellis2: Any) -> None:
        self.runtimes: dict[str, Any] = {"trellis2": trellis2}
        self.resident = "trellis2"
        self._loading: dict[str, threading.Thread] = {}
        self._failed: dict[str, str] = {}

    def load_later(self, name: str, build: Callable[[], Any]) -> None:
        """Builds a runtime in a background thread. If that fails, the model is counted out."""

        def work() -> None:
            try:
                self.runtimes[name] = build()
            except Exception as err:  # noqa: BLE001 - the model is counted out, the container goes on
                traceback.print_exc()
                self._failed[name] = f"{type(err).__name__}: {err}"
                print(f"[orainge] {name} could not be loaded, so finals are made with trellis2: {self._failed[name]}")

        thread = threading.Thread(target=work, name=f"load-{name}", daemon=True)
        self._loading[name] = thread
        thread.start()

    def count_out(self, name: str, reason: str) -> None:
        """A model that isn't there (its weights were never downloaded): finals use TRELLIS.2."""
        self._failed[name] = reason
        print(f"[orainge] {name} is not available, so finals are made with trellis2: {reason}")

    def failure(self, name: str) -> Optional[str]:
        """Why `name` was counted out (its weights missing, or its build failed), or None."""
        return self._failed.get(name)

    @property
    def loaded(self) -> set[str]:
        """The models a job can ask for: those built, and those still being built (use() waits for them)."""
        return {name for name in (*self.runtimes, *self._loading) if name not in self._failed}

    def use(self, name: str) -> Optional[Any]:
        """The runtime for `name`, ready for a job, or None if it could not be loaded."""
        thread = self._loading.pop(name, None)
        if thread is not None:
            thread.join()
        runtime = self.runtimes.get(name)
        if runtime is None:
            return None
        if name != "trellis2" and self.resident == "trellis2":
            # From now on the GPU is this model's; TRELLIS.2 serves asleep (its weights in RAM), for good
            self.runtimes["trellis2"].sleep()
            self.resident = name
        return runtime


def build_pixal3d(trellis2: Any) -> Any:
    """
    The Pixal3D runtime for finals (pixal3d_worker.pipeline.Pixal3DRuntime): the multi-view weights with
    the single-view set beside them for thin objects (PIXAL3D_THIN_RATIO), its models on the GPU, and
    this container's TRELLIS.2 for the recipe's '512' preview instead of a copy of its own.
    """
    from pixal3d_worker import pipeline

    return pipeline.Pixal3DRuntime(
        multiview=True,
        low_vram=False,
        level=pipeline.LEVEL_PREVIEW,
        trellis2=trellis2,
        thin_ratio=pipeline.thin_ratio_from_env(os.environ.get("PIXAL3D_THIN_RATIO")),
    )


def _ran_out_of_memory(result: dict) -> bool:
    return "out of memory" in str(result.get("error", "")).lower()


# What a textures job gets with the recipe on (forge3d_worker/service.py gives the same to a runtime that
# can't retexture): texture options remake the final's shape with TRELLIS.2 and retexture it, and a
# Pixal3D final has another shape. Even a container whose Pixal3D isn't loaded refuses them: the final
# may have come from one whose Pixal3D was
TEXTURES_NEED_TRELLIS2 = "texture options need TRELLIS.2 finals"


def handle_with_models(
    job: dict,
    pool: ModelPool,
    final_model: str,
    handlers: dict[str, Callable[..., dict]],
    storage: Any,
    pack: Any,
) -> dict:
    """
    One job through the model choose_model picks, with that model's name in the result (``"model"``).
    ``handlers`` holds each model's handle_job (production's for TRELLIS.2; the Pixal3D worker's, which
    binds a job's views and reports the recipe's camera, pose, level, weights and thin). A final that
    Pixal3D can't make for want of GPU memory, even after its own low-VRAM retry, is made with TRELLIS.2
    instead, with Pixal3D's models off the GPU meanwhile (which also clears the stale CUDA error a
    CuMesh failure leaves), and says so in ``"fallback"``: a model the user keeps is worth more than an
    error. A final made with TRELLIS.2 because the final model isn't there says why in ``"fallback"`` too.
    A textures job is refused unless the finals are TRELLIS.2's (TEXTURES_NEED_TRELLIS2); nothing runs.
    """
    if _is_textures(job) and final_model != "trellis2":
        print(f"[orainge] a textures job refused: the finals are {final_model}'s")
        return {"error": TEXTURES_NEED_TRELLIS2}
    name = choose_model(job, final_model, pool.loaded)
    runtime = pool.use(name)
    if runtime is None:  # it failed to load since the choice was made
        name, runtime = "trellis2", pool.use("trellis2")
    result = handlers[name](job, runtime, storage, pack)
    if name != "trellis2" and _ran_out_of_memory(result):
        reason = str(result["error"])
        print(f"[orainge] {name} ran out of GPU memory; making this final with trellis2 instead: {reason}")
        failed = runtime
        _call_quietly(failed, "_offload")
        try:
            name, runtime = "trellis2", pool.use("trellis2")
            result = handlers[name](job, runtime, storage, pack)
        finally:
            _call_quietly(failed, "_restore")
        if "error" not in result:
            result["fallback"] = reason
    elif name == "trellis2" and final_model != "trellis2" and _is_final(job) and "error" not in result:
        result["fallback"] = pool.failure(final_model) or f"{final_model} is not loaded"
    if "error" not in result:
        result["model"] = name
    return result


def _is_final(job: dict) -> bool:
    payload = job.get("input")
    return not isinstance(payload, dict) or payload.get("mode", "final") == "final"


def _is_textures(job: dict) -> bool:
    payload = job.get("input")
    return isinstance(payload, dict) and payload.get("mode") == "textures"


def _call_quietly(runtime: Any, method: str) -> None:
    """Moves a runtime's models (``_offload``/``_restore``) if it can; a failure there only costs speed."""
    move = getattr(runtime, method, None)
    if not callable(move):
        return
    try:
        move()
    except Exception as err:  # noqa: BLE001 - the job's own result matters more
        print(f"[orainge] {method} failed: {type(err).__name__}: {err}")


def trellis2_handler(
    trellis2: Any, storage: Any, pack: Any, final_model: str, judge: Any = None
) -> Callable[[dict], dict]:
    """
    The Trellis2 container's job handler around its loaded TRELLIS.2 runtime: every job goes through
    handle_with_models. TRELLIS.2's handle_job gets the judge a textures job asks for with "judge": true
    (``judge``, by default TextureJudge: Judge8B, called through Modal); no other job calls it. With the
    recipe on (``final_model`` "pixal3d"), Pixal3D is built in the background when its weights are there.
    """
    from forge3d_worker.service import handle_job

    pool = ModelPool(trellis2)
    judge = TextureJudge() if judge is None else judge
    handlers: dict[str, Callable[..., dict]] = {"trellis2": functools.partial(handle_job, judge=judge)}
    if final_model == "pixal3d":
        from pixal3d_worker.service import handle_job as pixal3d_handle_job

        handlers["pixal3d"] = pixal3d_handle_job
        if _weights_marker("pixal3d").exists():
            # Built while this container's first previews run; use() waits for it at the first final
            pool.load_later("pixal3d", lambda: build_pixal3d(trellis2))
        else:
            pool.count_out(
                "pixal3d",
                "its weights are missing: modal run workers/modal_app.py::download_models --which pixal3d",
            )
    return lambda job: handle_with_models(job, pool, final_model, handlers, storage, pack)


@app.cls(
    image=trellis2_image,
    gpu=TRELLIS2_GPU,
    cpu=4.0,
    # TRELLIS.2 asleep (about 10 GB) and Pixal3D's single-view flow models wait in RAM during a final,
    # and Pixal3D's own low-VRAM retry moves its multi-view models there too; what the Phase 6 runs used
    memory=49152,
    volumes={MODELS: models, CACHE: cache},
    secrets=storage_secrets,
    # A Pixal3D final takes 1-2 minutes, 3-5 with out-of-memory retries, and may then be made again with
    # TRELLIS.2 (1-3 minutes); a stuck job is still cut off well inside a quarter of an hour
    timeout=900,
    startup_timeout=600,
    scaledown_window=60,  # idle containers are billed; a cold start takes about a minute
    max_containers=2,  # caps spending; raise it for more parallel jobs
)
class Trellis2:
    """
    Image to textured, web-packed GLB. Input and output as in README.md (Job contracts). Previews are
    TRELLIS.2's, and so are finals unless FINAL_MODEL is "pixal3d" (the recipe), in one container, so a
    preview's seed carries into its final the way it always has. Texture options for a TRELLIS.2 final
    (mode "textures") are made here too.
    """

    @modal.enter()
    def load(self) -> None:
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.pipeline import Trellis2Runtime
        from forge3d_worker.storage import storage_from_env

        _require_weights("trellis2")
        # FlexGEMM reads its tuning results when it is imported, so bring in the shared ones first
        merge_tuning(LOCAL_TUNING, SHARED_TUNING)
        storage = storage_from_env()
        trellis2 = Trellis2Runtime(f"{MODELS}/TRELLIS.2-4B")
        final_model = os.environ.get("ORAINGE_FINAL_MODEL", FINAL_MODEL)
        self.handle = trellis2_handler(trellis2, storage, pack_glb, final_model)

    @modal.method()
    def generate(self, job: dict) -> dict:
        try:
            return run_job(self.handle, job)
        finally:
            share_caches()

    @modal.method()
    def warm(self) -> bool:
        """Does nothing: calling it starts a container (load() runs first) before a job needs one."""
        return True


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

    @modal.method()
    def warm(self) -> bool:
        """Does nothing: calling it starts a container (load() runs first) before a job needs one."""
        return True


@app.cls(
    image=multiview_image,
    gpu=MULTIVIEW_GPU,
    cpu=2.0,
    memory=16384,  # SDXL's fp16 weights and the adapter pass through RAM on their way to the GPU
    volumes={MODELS: models},
    secrets=storage_secrets,
    timeout=300,
    startup_timeout=600,
    scaledown_window=60,
    max_containers=1,
)
class MultiView:
    """One picture to six views around its object (MV-Adapter, Apache-2.0, on SDXL). See README.md."""

    @modal.enter()
    def load(self) -> None:
        from multiview_worker.generator import MultiViewGenerator
        from multiview_worker.service import handle_job
        from multiview_worker.storage import storage_from_env

        _require_weights("multiview")
        storage = storage_from_env()
        generate = MultiViewGenerator(MODELS)
        self.handle = lambda job: handle_job(job, generate, storage)

    @modal.method()
    def generate(self, job: dict) -> dict:
        return run_job(self.handle, job)

    @modal.method()
    def warm(self) -> bool:
        """Does nothing: calling it starts a container (load() runs first) before a job needs one."""
        return True


def geometry_views_handler() -> Callable[[dict], dict]:
    """GeometryViews' job handler: MV-Adapter's image+geometry model, loaded, behind handle_geometry_job."""
    from multiview_worker.geometry import GeometryViewGenerator
    from multiview_worker.service import handle_geometry_job

    _require_weights("multiview")
    generator = GeometryViewGenerator(MODELS)
    return lambda job: handle_geometry_job(job, generator)


@app.cls(
    image=multiview_image,
    gpu=MULTIVIEW_GPU,
    cpu=2.0,
    memory=16384,  # as MultiView: SDXL's fp16 weights and the adapter pass through RAM
    volumes={MODELS: models},
    timeout=300,
    startup_timeout=600,
    scaledown_window=30,  # experiments call it in bursts; idle time is billed
    max_containers=1,
)
class GeometryViews:
    """
    Experiments only, nothing in production calls it: six views of a given mesh drawn from a picture
    by MV-Adapter's image+geometry model (ig2mv, Apache-2.0, on SDXL). The caller renders the mesh's
    position and normal maps; the job and its result are in multiview/multiview_worker/service.py
    (handle_geometry_job), and README.md has the conventions:

        modal_app.GeometryViews().generate.remote({"image_base64": …, "control_pngs": […], "seed": 0})
    """

    @modal.enter()
    def load(self) -> None:
        self.handle = geometry_views_handler()

    @modal.method()
    def generate(self, job: dict) -> dict:
        return run_job(self.handle, job)


# --- The judge: which texture of a shape a creator would rather use --------------------------------------
# Production asks Judge8B when a textures job says "judge": true (TextureJudge, below); experiments ask
# either size (ops/exp_judge.py). One class per size, because a class parameter can't choose the GPU (Modal
# sets it per class; with_options would leave Judge(model="30b") on a card it doesn't fit).
# judge_class("8b" | "30b") gives the class.

# The downloads a size's weights come from: the 8B's own, or the one with both sizes
JUDGE_WEIGHTS = {"8b": ("judge8b", "judge"), "30b": ("judge",)}


def judge_weights_missing(size: str) -> Optional[str]:
    """Why ``size``'s judge can't load here (no complete download of its weights), or None. Reads markers only."""
    names = JUDGE_WEIGHTS[size]
    if any(_weights_marker(name).exists() for name in names):
        return None
    return (
        f"the judge's {size} weights are missing or incomplete in the orainge-models volume: "
        f"modal run workers/modal_app.py::download_models --which {names[0]}"
    )


def judge_handler(which: str) -> Callable[[dict], dict]:
    """A judge job handler (judge/judge_worker/service.py) with that size of Qwen3-VL loaded on the GPU."""
    from judge_worker.model import load
    from judge_worker.service import handle_job

    missing = judge_weights_missing(which)
    if missing is not None:
        raise RuntimeError(missing)
    model = load(which, MODELS)
    print(f"[orainge] judge {which}: {model.path} loaded in {model.load_seconds} s")
    return lambda job: handle_job(job, model, name=which)


JUDGE_OPTIONS = dict(
    image=judge_image,
    cpu=4.0,
    memory=32768,  # the weights go straight to the GPU; RAM holds a file's worth at a time
    volumes={MODELS: models},
    max_containers=1,
)


@app.cls(
    gpu=JUDGE_GPUS["8b"],
    # A judgement takes about 9 s; one still running after 5 minutes is stuck
    timeout=300,
    # Its 17.5 GB loaded in 16 s in Phase 7; the rest is room for a slow volume or a slow image pull
    startup_timeout=600,
    # A textures job warms it once the shape is made and asks it about a minute later: it must outlast
    # that wait. Idle time is billed (about $0.04 a minute), so not much longer
    scaledown_window=120,
    **JUDGE_OPTIONS,
)
class Judge8B:
    """
    Qwen3-VL-8B looks at the picture a model was made from and turntable grids of several textures of its
    shape (trellis2/forge3d_worker/judgeviews.py) and says which a creator would rather use. Production's
    judge: the Trellis2 container asks it when a textures job says "judge": true (TextureJudge), and only
    then; experiments ask it too. The job and its result are judge/judge_worker/service.py's:

        modal_app.Judge8B().judge.remote({"picture_png": …, "candidates_png": […], "prompt": "…"})
    """

    @modal.enter()
    def load(self) -> None:
        self.handle = judge_handler("8b")

    @modal.method()
    def judge(self, job: dict) -> dict:
        return run_job(self.handle, job)

    @modal.method()
    def warm(self) -> bool:
        """Does nothing: calling it starts a container (load() runs first) before a judgement needs one."""
        return True


@app.cls(
    gpu=JUDGE_GPUS["30b"],
    timeout=900,
    startup_timeout=1200,  # its 62 GB come off the volume before the first job
    # Experiments call it as each object is made, a minute or so apart: idle time is billed, but a cold
    # start (62 GB off the volume) costs more
    scaledown_window=300,
    **JUDGE_OPTIONS,
)
class Judge30B:
    """Judge8B with Qwen3-VL-30B-A3B (a mixture of experts, 62 GB in bf16) on an H100. Experiments only."""

    @modal.enter()
    def load(self) -> None:
        self.handle = judge_handler("30b")

    @modal.method()
    def judge(self, job: dict) -> dict:
        return run_job(self.handle, job)

    @modal.method()
    def warm(self) -> bool:
        """Does nothing: calling it starts a container (load() runs first) before a judgement needs one."""
        return True


JUDGES = {"8b": Judge8B, "30b": Judge30B}


def judge_class(model: str) -> Any:
    """The judge class for a size, "8b" or "30b": ``judge_class("30b")().judge.spawn(job)``."""
    if model not in JUDGES:
        raise ValueError(f"the judge model is one of {', '.join(JUDGES)}, not {model!r}")
    return JUDGES[model]


# Production's judge for texture options. In Phase 7's validation (20 objects, four textures each) its pick
# was publishable for 15 of 20 in each of two orders, the 30B's for 13 to 14, the final's own texture for 11
JUDGE_SIZE = "8b"
# How long a textures job waits for the judge's answer: about 9 s when it is warm, plus up to a minute or
# so when its container has to start (TextureJudge warms it early, so it rarely has to). After that the
# textures go back without a pick
JUDGE_WAIT = 180.0


class TextureJudge:
    """
    The judge a textures job asks (trellis2/forge3d_worker/service.py's Judge), from the Trellis2 container:
    Judge8B (JUDGE_SIZE), through Modal. ``unavailable()`` says, without calling anything, why it can't
    judge (its weights were never downloaded), so such a job skips the judge's part at once instead of
    starting a container that can't load. ``warm()`` starts its container and doesn't wait. Called with the
    judge worker's job, it returns that worker's result; after ``wait`` seconds it cancels the call and
    raises TimeoutError.
    """

    def __init__(self, size: str = JUDGE_SIZE, wait: float = JUDGE_WAIT) -> None:
        judge_class(size)  # a wrong size fails here, not in a job
        self.size = size
        self.model = size  # what the job's "judge" says made the pick when the answer doesn't
        self.wait = wait

    def unavailable(self) -> Optional[str]:
        return judge_weights_missing(self.size)

    def warm(self) -> None:
        judge_class(self.size)().warm.spawn()

    def __call__(self, request: dict) -> dict:
        missing = self.unavailable()
        if missing is not None:
            raise RuntimeError(missing)
        call = judge_class(self.size)().judge.spawn(request)
        try:
            return call.get(timeout=self.wait)
        except (TimeoutError, modal.exception.TimeoutError) as err:
            _cancel_quietly(call)
            raise TimeoutError(f"the judge didn't answer within {self.wait:g} s") from err


def _cancel_quietly(call: Any) -> None:
    """Cancels a call nobody waits for any more, so its GPU time stops; failing to only costs that time."""
    try:
        call.cancel()
    except Exception as err:  # noqa: BLE001 - the job's own result matters more
        print(f"[orainge] the judge's call was not cancelled: {type(err).__name__}: {err}")


# In download order: Pixal3D's script reuses the TRELLIS.2 worker's DINOv3, BiRefNet and decoders. The
# judge's only when asked for: the 8B alone (17.5 GB) for production's texture options, or both sizes
# (about 80 GB) for the experiments
WEIGHT_SCRIPTS = {
    "trellis2": "/root/weights/trellis2.py",
    "reference": "/root/weights/reference.py",
    "multiview": "/root/weights/multiview.py",
    "pixal3d": "/root/weights/pixal3d.py",
    "judge8b": "/root/weights/judge.py",
    "judge": "/root/weights/judge.py",
}
ON_REQUEST_WEIGHTS = ("judge8b", "judge")  # left out of --which all
# What WHICH tells a script: which of Pixal3D's weight sets, which of the judge's sizes (the others ignore it)
WEIGHT_SETS = {"pixal3d": PIXAL3D_WEIGHTS, "judge8b": "8b", "judge": "all"}


@app.function(
    image=download_image,
    volumes={MODELS: models},
    # The name Modal's Hugging Face secret template suggests
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    timeout=3 * 3600,  # generous for slow Hugging Face transfers; an interrupted run resumes
)
def download_models(which: str = "all", force: bool = False) -> None:
    """
    Downloads the pinned weights (about 119 GB, 44 GB of it Pixal3D's two sets) into the orainge-models
    volume. CPU only. "all" leaves the judge out: ``--which judge8b`` fetches production's (Qwen3-VL-8B,
    17.5 GB), which texture options ask when a job says "judge": true, and ``--which judge`` both sizes
    (about 80 GB) for the experiments.
    """
    if which != "all" and which not in WEIGHT_SCRIPTS:
        raise SystemExit(f"--which must be all, {', '.join(WEIGHT_SCRIPTS)}")
    names = [name for name in WEIGHT_SCRIPTS if name not in ON_REQUEST_WEIGHTS] if which == "all" else [which]
    for name in names:
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
    env = {
        **os.environ,
        "MODELS_ROOT": MODELS,
        "FLUX_MODEL_DIR": f"{MODELS}/FLUX.1-schnell",
        "WHICH": WEIGHT_SETS.get(name, "all"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # progress bars flood non-interactive logs
    }
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
    """
    https://<workspace>--orainge-ai-api.modal.run: set it as the server's AI_WORKERS_URL. Worker
    "trellis2" takes previews, finals and texture options alike (the Trellis2 container routes finals to
    the Pixal3D recipe); "multiview" is there for the Studio's AI_MULTIVIEW step, which is off by default.
    """
    from job_api import ModalCalls, app_for_token

    trellis2, flux, multiview = Trellis2(), FluxSchnell(), MultiView()
    calls = ModalCalls(
        {"trellis2": trellis2.generate, "reference": flux.generate, "multiview": multiview.generate},
        warm={"trellis2": trellis2.warm, "reference": flux.warm, "multiview": multiview.warm},
    )
    return app_for_token(os.environ.get("ORAINGE_WORKER_TOKEN"), calls)


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
    pick: Optional[int] = None,
    pictures_only: bool = False,
    final_weights: Sequence[str] = (),
) -> dict:
    """
    Prompt or image -> reference images -> preview GLB -> (optionally) final GLB, written into
    `folder`. Each finished step is recorded in progress.json, so running it again on the same
    folder skips what is done and continues where it stopped.

    The reference picture the worker scored best goes on to 3D (the first, if it didn't score
    them) unless `pick` names another (1-4), the way a user picks one in the Studio.
    `pictures_only` stops before 3D, so the pictures can be looked at first; running again with
    `pick` then makes the model. `final_weights` are what a final needs besides TRELLIS.2's (Pixal3D's
    when it makes finals), fetched before the final even when the run continues an earlier one.
    """
    folder.mkdir(parents=True, exist_ok=True)
    progress_file = folder / "progress.json"
    run = folder.name
    if progress_file.exists():
        # A continued run keeps its prompt, image and seed; it can only add the final step
        state: dict = json.loads(progress_file.read_text())
        # Modal runs an input again when its container is replaced mid-run, with the same image: carry on
        saved = folder / state["input"] if state.get("input") else None
        same_image = image is not None and saved is not None and saved.exists() and saved.read_bytes() == image
        if (prompt and prompt != state.get("prompt")) or (image is not None and not same_image) or (
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

    def persist() -> None:
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        progress_file.write_text(json.dumps(state, indent=2))
        save()

    def record(step: str, **fields: Any) -> None:
        steps[step] = {**steps.get(step, {}), **fields}
        persist()

    def use_picture(number: int) -> None:
        pictures = steps.get("reference", {}).get("files", [])
        if not 1 <= number <= len(pictures):
            raise ValueError(f"{run} has {len(pictures)} pictures, so its pick must be 1 to {len(pictures)}")
        chosen = pictures[number - 1]
        if chosen == state.get("input"):
            return
        if any(steps.get(step, {}).get("status") == "done" for step in ("preview", "final")):
            made_from = state.get("input")
            raise ValueError(f"{run} already has a model of {made_from}; start a new run to use picture {number}")
        state["input"] = chosen
        persist()

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

    ensured: set[str] = set()

    def weights() -> dict:
        names = ["trellis2", "reference"] if prompt else ["trellis2"]
        if final:
            names.extend(final_weights)
        for name in names:
            ensure_weights(name)
            ensured.add(name)
        return {}

    def reference_images() -> dict:
        result = checked(reference({"input": {"prompt": prompt, "count": 4, "request_id": run}}))
        files = []
        for picture in result["images"]:
            name = f"reference-{picture['seed']}.png"
            (folder / name).write_bytes(fetch(picture))
            files.append(name)
        fields: dict = {"files": files, "gpu_seconds": result.get("seconds")}
        scores = [picture.get("score") for picture in result["images"]]
        if any(score is not None for score in scores):
            fields["scores"] = scores
            fields["issues"] = [picture.get("issues", []) for picture in result["images"]]
        # The best-framed picture goes on to 3D: the earlier one on a tie, the first if nothing was
        # scored. `pick` makes another one instead
        best = max(range(len(files)), key=lambda i: -1 if scores[i] is None else scores[i])
        state["input"] = files[best]
        return fields

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
            step = {
                "files": [name],
                "triangles": result["triangles"],
                "bytes": result.get("bytes"),
                "timings": result["timings"],
                "gpu_seconds": round(sum(result["timings"].values()), 1),
            }
            # Which pipeline made it ("512", "1024_cascade", "pixal3d-1024_cascade"; a TRELLIS.2 final that
            # ran out of GPU memory is made with "512")
            if "pipeline" in result:
                step["pipeline"] = result["pipeline"]
            # Which model made it ("pixal3d" | "trellis2"), why not the one asked for (fallback), and the
            # recipe's notes: Pixal3D's weights ("multiview" | "single"), what the preview measured (thin),
            # the picture's camera, the pose search that placed it, and what the export levelled
            for key in ("model", "fallback", "weights", "thin", "camera", "pose", "level"):
                if result.get(key):
                    step[key] = result[key]
            if result.get("views_used"):  # extra views of the object that helped (0 is left out)
                step["views_used"] = result["views_used"]
            if result.get("projection"):  # finals: whether the picture was painted on, and why not
                step["projection"] = result["projection"]
            return step

        return work

    if pick and not prompt:
        raise ValueError(f"{run} started from an image, so it has no pictures to pick from")
    attempt("weights", weights)
    if prompt:
        attempt("reference", reference_images)
        if pick:
            use_picture(pick)
    if pictures_only:
        return state
    attempt("preview", model("preview"))
    if final:
        # A run continued with --final (its weights step done earlier, for a preview) still needs them
        for name in final_weights:
            if name not in ensured:
                ensure_weights(name)
        attempt("final", model("final"))
    return state


@app.function(image=light_image, volumes={OUTPUTS: outputs}, timeout=4 * 3600)
def make_model(
    run: str,
    prompt: str = "",
    image: bytes = b"",
    image_name: str = "",
    final: bool = False,
    seed: int = -1,
    pick: int = 0,
    pictures_only: bool = False,
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
        pick=pick or None,
        pictures_only=pictures_only,
        final_weights=FINAL_WEIGHTS,
    )


@app.local_entrypoint()
def make(
    prompt: str = "",
    image: str = "",
    final: bool = False,
    seed: int = -1,
    run: str = "",
    pick: int = 0,
    pictures_only: bool = False,
) -> None:
    """
    Prompt or image to 3D in Modal's cloud. Start it with `modal run --detach` and it finishes
    even if this computer sleeps or goes offline; `--run NAME` continues an earlier run.
    `--pictures-only` stops at the four pictures; `--run NAME --pick 3` makes the third into 3D.
    """
    if not (prompt or image or run):
        raise SystemExit("Give --prompt or --image, or --run NAME to continue a run")
    if prompt and image:
        raise SystemExit("Give either --prompt or --image, not both")
    if not 0 <= pick <= 4:
        raise SystemExit("--pick must be 1 to 4")
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
        pick=pick,
        pictures_only=pictures_only,
    )
    # Still connected: copy the results here too
    target = pathlib.Path("orainge-outputs") / run
    files = _copy_run(run, target)
    print(f"Done: {', '.join(files)} in {target}")
    if pictures_only:
        print(f"Make one into 3D: modal run --detach workers/modal_app.py::make --run {run} --pick <1-4>")
    elif not state.get("final"):
        print(f"Final quality, same shape: modal run --detach workers/modal_app.py::make --run {run} --final")


def _copy_run(run: str, target: pathlib.Path) -> list[str]:
    """
    Copies a run's progress.json and finished files from the outputs volume into `target`, which
    make_gallery.py can then read. Returns the files copied besides progress.json.
    """
    import modal.exception as mx

    try:
        state = json.loads(b"".join(outputs.read_file(f"{run}/progress.json")))
    except (mx.NotFoundError, FileNotFoundError):
        return []  # the run stopped before it saved anything
    target.mkdir(parents=True, exist_ok=True)
    (target / "progress.json").write_text(json.dumps(state, indent=2))
    files = [name for step in state["steps"].values() for name in step.get("files", [])]
    if state.get("input") and state["input"] not in files:
        files.append(state["input"])  # the photo an image run started from
    copied = []
    for name in files:
        try:
            with open(target / name, "wb") as f:
                outputs.read_file_into_fileobj(f"{run}/{name}", f)
        except (mx.NotFoundError, FileNotFoundError):
            (target / name).unlink(missing_ok=True)
            continue
        copied.append(name)
    return copied


# Test sets: many runs from one file, for judging quality across kinds of assets

SET_NAME = re.compile(r"[A-Za-z0-9_-]{1,24}")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def read_set(path: pathlib.Path) -> list[dict]:
    """
    One run per line: a prompt, or the path of an image relative to the file. Blank lines and
    lines starting with # are skipped.
    """
    runs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if pathlib.PurePath(line).suffix.lower() in IMAGE_SUFFIXES:
            image = path.parent / line
            if not image.is_file():
                raise SystemExit(f"{path}, line {number}: {image} doesn't exist")
            runs.append({"image": image})
        else:
            runs.append({"prompt": line})
    return runs


SLUG_FILLER = {"a", "an", "the", "and", "with", "of", "on", "in", "to", "for"}


def _slug(text: str, limit: int = 32) -> str:
    """The first words of a prompt, whole, without a leading article or a dangling 'with'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) > 1 and words[0] in {"a", "an", "the"}:
        words = words[1:]
    kept: list[str] = []
    for word in words:
        if len("-".join(kept + [word])) > limit:
            break
        kept.append(word)
    while len(kept) > 1 and kept[-1] in SLUG_FILLER:
        kept.pop()
    return "-".join(kept) or (words[0][:limit] if words else "")


def set_run_names(name: str, runs: list[dict]) -> list[str]:
    """<set>-<nn>-<words>, so runs sort like the file and read well in `status` and galleries."""
    return [
        f"{name}-{number:02d}-{_slug(run['image'].stem if 'image' in run else run['prompt']) or 'run'}"
        for number, run in enumerate(runs, 1)
    ]


def parse_picks(text: str, runs: list[dict]) -> dict[int, int]:
    """
    '3=2, 7=4' -> {3: 2, 7: 4}: the set's third run is made from its second picture and the
    seventh from its fourth. Runs left out use their best-scored picture.
    """
    picks = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        match = re.fullmatch(r"(\d+)\s*=\s*([1-4])", item)
        if not match or not 1 <= int(match[1]) <= len(runs):
            raise SystemExit(f"--picks: {item!r} should be <run 1-{len(runs)}>=<picture 1-4>, e.g. 3=2")
        if "prompt" not in runs[int(match[1]) - 1]:
            raise SystemExit(f"--picks: run {match[1]} starts from a photo, so it has no pictures to pick from")
        picks[int(match[1])] = int(match[2])
    return picks


def set_arguments(
    names: list[str],
    runs: list[dict],
    existing: set,
    final: bool,
    picks: Optional[dict[int, int]] = None,
    pictures_only: bool = False,
) -> list[tuple]:
    """
    make_model's arguments for each run of a set. A run already in the volume continues from its
    saved image; a prompt is passed again so that an edited prompt is refused, not ignored.
    """
    picks = picks or {}
    arguments = []
    for number, (run, entry) in enumerate(zip(names, runs), 1):
        choice = (picks.get(number, 0), pictures_only)
        if "prompt" in entry:
            arguments.append((run, entry["prompt"], b"", "", final, -1, *choice))
        elif run in existing:
            arguments.append((run, "", b"", "", final, -1, *choice))
        else:
            arguments.append((run, "", entry["image"].read_bytes(), entry["image"].name, final, -1, *choice))
    return arguments


@app.local_entrypoint()
def make_set(prompts: str, final: bool = True, name: str = "", pictures_only: bool = False, picks: str = "") -> None:
    """
    A test set in one go: one run per line of a text file (see workers/test-sets/). The runs share
    warm GPU containers, so a set costs less than the same runs one by one. With --detach it
    finishes without this computer, and the same --name continues an interrupted set. The results
    are copied to orainge-outputs/<name>/, ready for workers/gallery/make_gallery.py.

    Like a user in the Studio, you can choose which picture becomes 3D: run the set with
    --pictures-only, look at the pictures, then run it again with --picks "3=2,7=4" (run number =
    picture number; runs left out use their best-scored picture).
    """
    import modal.exception as mx

    source = pathlib.Path(prompts)
    runs = read_set(source)
    if not runs:
        raise SystemExit(f"{source} lists no prompts or images")
    name = name or time.strftime("set-%Y%m%d-%H%M")
    if not SET_NAME.fullmatch(name):
        raise SystemExit("--name must be 1-24 letters, digits, '-' or '_'")
    chosen = parse_picks(picks, runs)
    names = set_run_names(name, runs)
    try:
        existing = {entry.path.strip("/") for entry in outputs.listdir("/")}
    except mx.NotFoundError:
        existing = set()
    print(f"[orainge] {name}: {len(runs)} runs, saved in the orainge-outputs volume as {names[0]} to {names[-1]}")
    print("  Progress, any time:  python workers/modal_app.py status")
    # Any missing weights are fetched once here, not by every run at the same time: TRELLIS.2's, FLUX's
    # for prompts, and Pixal3D's for finals when it makes them
    needed = ["trellis2"]
    if any("prompt" in run for run in runs):
        needed.append("reference")
    if final:
        needed.extend(FINAL_WEIGHTS)
    for which in needed:
        download_models.remote(which=which)
    # All at once, so every run finishes before the copies start and the map closes cleanly
    arguments = set_arguments(names, runs, existing, final, chosen, pictures_only)
    outcomes = list(make_model.starmap(arguments, return_exceptions=True))
    target = pathlib.Path("orainge-outputs") / name
    failed = 0
    for run, outcome in zip(names, outcomes):
        files = _copy_run(run, target / run)
        if isinstance(outcome, BaseException):
            failed += 1
            reason = next((line for line in str(outcome).splitlines() if line.strip()), "")
            print(f"{run}: failed ({type(outcome).__name__}: {reason})")
        else:
            print(f"{run}: {', '.join(files)}")
    print(f"Done: {len(runs) - failed} of {len(runs)} runs finished; files in {target}")
    again = f"modal run --detach workers/modal_app.py::make_set --prompts {prompts} --name {name}"
    if failed:
        print(f"Retry what failed: {again}{' --pictures-only' if pictures_only else ''}")
    if pictures_only:
        print(f'Make the models, choosing pictures by run number: {again} --picks "1=2,3=4"')


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
    )[-20:]
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
    images = [
        ("download", download_image),
        ("api", api_image),
        ("reference", flux_image),
        ("multiview", multiview_image),
        ("trellis2", trellis2_image),
        ("judge", judge_image),
    ]
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
