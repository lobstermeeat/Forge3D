"""
Phase 6, PIXAL3D line: Pixal3D (TencentARC, MIT) on Modal. Does its multi-view model fix the invented backs?

    modal run ops/exp_pixal3d.py::download                     # the weights, once (CPU)
    modal run ops/exp_pixal3d.py::experiment --plan single --out ops-out/private
    modal run ops/exp_pixal3d.py::experiment --plan synthetic --out ops-out/private
    modal run ops/exp_pixal3d.py::experiment --plan mvadapter --out ops-out/private
    modal run ops/exp_pixal3d.py::experiment --plan controls --out ops-out/private     # run 2
    modal run ops/exp_pixal3d.py::experiment --plan mvchoice --out ops-out/private
    modal run ops/exp_pixal3d.py::experiment --plan posed --out ops-out/private        # run 3
    modal run ops/exp_pixal3d.py::experiment --plan final20 --out ops-out/private      # run 4

An ephemeral app (orainge-exp-pixal3d; nothing is deployed) with one or two L40S, running workers/pixal3d
through its handle_job as a worker would: the Phase 2 picture and its phase2d seed, mode "final",
production's export and packing. Plans:

- single: Pixal3D's single-view weights on the picture alone (its camera from MoGe-2).
- synthetic: the multi-view weights on views rendered here from the object's Phase 5 final, orthographic
  and level at MV-Adapter's azimuths and framing (Pixal3D's rebuild must line up with them: this checks
  the cameras before real views arrive).
- mvadapter: the multi-view weights on the MV line's views, /outputs/phase6/views/<phase2 run>/.
- controls: the single-view weights on the brief's other eight controls.
- mvchoice: the 11 books' mv6 again, the car's synthetic views framed to fit the cube, then four views
  against six and the picture against the redrawn front as the main view (see PLANS).
- posed (run 3, the tilt): the multi-view weights with the picture as one view at its real camera
  elevation (Phase 5's pose), as an absolute pose the model has to honour ("posed"), against the same
  weights levelled afterwards by that elevation ("mvlevel"). Nothing else is turned.
- final20 (run 4): production's recipe on all twenty prompts at the phase2d seeds: TRELLIS.2's '512'
  preview first (written as preview-<seed>.glb), the picture's camera found against it, Pixal3D's
  multi-view weights with the picture as their one view, the model levelled by that elevation, then
  the full export. Runs are named phase2f-NN-<slug> and written to /outputs/phase6/pixal3d/final20/.
  final20single is the same on the single-view weights (phase2fs-NN-<slug>).

Every multi-view result is checked against its views: the packed final is rendered from each view's
camera and its silhouette compared with the view's (IoU). Results go to the volume,
/outputs/phase6/pixal3d/<plan>/<run>/ (progress.json as make_model writes it, the packed final, the
picture, the views), and to <out>/<plan>/ here; numbers only to ops-out/pixal3d-<plan>.json.
"""


import base64
import io
import json
import pathlib
import re
import sys
import time
from typing import Optional

import modal

HERE = pathlib.Path(__file__).resolve()
# Here: the checkout's workers/. In the container: /root, where the image puts modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - images, volumes and constants only; nothing is deployed

app = modal.App("orainge-exp-pixal3d")

PIXAL3D_COMMIT = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"
NAF_COMMIT = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
MOGE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"  # MoGe-2, before MoGe-3 changed its dependencies
UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"  # what that MoGe pins
WORKERS = prod.WORKERS
RESULTS = "phase6/pixal3d"
VIEWS = "phase6/views"
WEIGHTS_MARKER = pathlib.Path(prod.MODELS) / ".pixal3d-weights"


def trellis2_base() -> modal.Image:
    """
    workers/modal_app.py's trellis2_image, step for step (so Modal reuses its built layers), without
    the worker files it ends with, which no build step may follow.
    """
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
        .entrypoint([])
        .apt_install("git", "curl", "unzip", "build-essential", "libgl1", "libglib2.0-0", "libjpeg-dev")
        .pip_install(*prod.TORCH, index_url=prod.TORCH_INDEX)
        .pip_install(prod.FLASH_ATTN_WHEEL)
        .pip_install_from_requirements(str(WORKERS / "trellis2" / "requirements.txt"))
        .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "MAX_JOBS": "4"})
        .run_commands(
            prod._git("https://github.com/microsoft/TRELLIS.2", prod.TRELLIS2_COMMIT, "/opt/trellis2"),
            'echo /opt/trellis2 > "$(python -c "import site; print(site.getsitepackages()[0])")/trellis2-repo.pth"',
        )
        .run_commands("python -m pip install --upgrade pip 'setuptools>=70' wheel")
        .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -pthread -shared", "LDCXXSHARED": "g++ -pthread -shared"})
        .run_commands(
            prod._git("https://github.com/JeffreyXiang/CuMesh", prod.CUMESH_COMMIT, "/tmp/CuMesh"),
            "python -m pip install /tmp/CuMesh --no-build-isolation --no-deps && rm -rf /tmp/CuMesh",
        )
        .run_commands(
            prod._git("https://github.com/JeffreyXiang/FlexGEMM", prod.FLEXGEMM_COMMIT, "/tmp/FlexGEMM"),
            "python -m pip install /tmp/FlexGEMM --no-build-isolation --no-deps && rm -rf /tmp/FlexGEMM",
        )
        .run_commands("python -m pip install /opt/trellis2/o-voxel --no-build-isolation --no-deps")
        .run_commands(
            "curl -fsSL -o /tmp/gltfpack.zip https://github.com/zeux/meshoptimizer/releases/download/"
            f"v{prod.GLTFPACK_VERSION}/gltfpack-ubuntu.zip && unzip /tmp/gltfpack.zip -d /usr/local/bin"
            " && chmod +x /usr/local/bin/gltfpack && rm /tmp/gltfpack.zip"
        )
        .env(
            {
                "ATTN_BACKEND": "flash_attn",
                "HF_HUB_OFFLINE": "1",
                "TRELLIS2_MODEL_DIR": f"{prod.MODELS}/TRELLIS.2-4B",
                "TRELLIS2_LOW_VRAM": prod.TRELLIS2_LOW_VRAM,
                "TRITON_CACHE_DIR": f"{prod.CACHE}/triton",
            }
        )
    )


pixal3d_image = (
    trellis2_base()
    # Pixal3D (MIT) and NAF (Apache-2.0) at pinned commits; Pixal3D importable as `pixal3d`
    .run_commands(
        prod._git("https://github.com/TencentARC/Pixal3D", PIXAL3D_COMMIT, "/opt/pixal3d"),
        'echo /opt/pixal3d > "$(python -c "import site; print(site.getsitepackages()[0])")/pixal3d-repo.pth"',
        prod._git("https://github.com/valeoai/NAF", NAF_COMMIT, "/opt/naf"),
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
            "PIXAL3D_MODEL_DIR": f"{prod.MODELS}/pixal3d",
            "PIXAL3D_DINO_DIR": f"{prod.MODELS}/dinov3-vitl16",
            "PIXAL3D_NAF_DIR": "/opt/naf",
        }
    )
    .add_local_dir(WORKERS / "trellis2" / "forge3d_worker", "/root/forge3d_worker")
    .add_local_dir(WORKERS / "pixal3d" / "pixal3d_worker", "/root/pixal3d_worker")
    .add_local_file(WORKERS / "modal_app.py", "/root/modal_app.py")
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.34,<2")
    .add_local_file(WORKERS / "pixal3d" / "scripts" / "download_weights.py", "/root/weights/pixal3d.py")
    .add_local_file(WORKERS / "modal_app.py", "/root/modal_app.py")
)


@app.function(
    image=download_image,
    volumes={prod.MODELS: prod.models},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    memory=4096,
    timeout=3 * 3600,
)
def fetch_weights(which: str = "all", force: bool = False) -> str:
    """The Pixal3D worker's pinned weights into orainge-models (/models/pixal3d). CPU only."""
    import hashlib
    import os
    import subprocess

    script = pathlib.Path("/root/weights/pixal3d.py")
    digest = hashlib.sha256(script.read_bytes()).hexdigest() + f":{which}"
    if not force and WEIGHTS_MARKER.exists() and WEIGHTS_MARKER.read_text() == digest:
        return "already downloaded"
    WEIGHTS_MARKER.unlink(missing_ok=True)
    env = {**os.environ, "MODELS_ROOT": prod.MODELS, "WHICH": which, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    started = time.monotonic()
    subprocess.run(["python", str(script)], env=env, check=True)
    WEIGHTS_MARKER.write_text(digest)
    prod.models.commit()
    return f"downloaded in {time.monotonic() - started:.0f} s"


@app.local_entrypoint()
def download(which: str = "all", force: bool = False) -> None:
    print(f"[pixal3d] weights ({which}): {fetch_weights.remote(which=which, force=force)}")


# --- Packed GLBs, rendered from known cameras (GPU) -----------------------------------------------------


def plain_mesh(packed: bytes) -> dict:
    """
    A packed GLB (meshopt geometry, WebP colour) as plain arrays: gltfpack decodes it (-noq: no
    quantisation), then positions, glTF UVs and indices are read straight from its buffer.
    """
    import subprocess
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        source, target = pathlib.Path(tmp, "in.glb"), pathlib.Path(tmp, "out.glb")
        source.write_bytes(packed)
        subprocess.run(["gltfpack", "-i", str(source), "-o", str(target), "-noq"], check=True, capture_output=True)
        data = target.read_bytes()
    length = int.from_bytes(data[12:16], "little")
    gltf = json.loads(data[20 : 20 + length])
    binary = data[20 + length + 8 :]
    kinds = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}

    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        kind, width = np.dtype(kinds[acc["componentType"]]), widths[acc["type"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride", kind.itemsize * width)
        raw = np.frombuffer(binary, np.uint8, count=stride * acc["count"], offset=start).reshape(acc["count"], stride)
        return raw[:, : kind.itemsize * width].copy().view(kind).reshape(acc["count"], width)

    primitive = gltf["meshes"][0]["primitives"][0]
    node = next(n for n in gltf["nodes"] if "mesh" in n)
    positions = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
    positions = positions * node.get("scale", [1, 1, 1]) + node.get("translation", [0, 0, 0])
    uv = accessor(primitive["attributes"]["TEXCOORD_0"]).astype(np.float64)
    reference = gltf["materials"][primitive["material"]]["pbrMetallicRoughness"]["baseColorTexture"]
    transform = reference.get("extensions", {}).get("KHR_texture_transform", {})
    uv = uv * transform.get("scale", [1, 1]) + transform.get("offset", [0, 0])
    texture = gltf["textures"][reference["index"]]
    image_index = texture.get("source", texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
    view = gltf["bufferViews"][gltf["images"][image_index]["bufferView"]]
    start = view.get("byteOffset", 0)
    image = Image.open(io.BytesIO(binary[start : start + view["byteLength"]])).convert("RGB")
    faces = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
    return {"positions": positions, "uv": uv, "faces": faces, "texture": image}


def render_views(
    mesh: dict,
    azimuths: list,
    size: int = 768,
    half_extent: float = 0.55,
    fill: Optional[float] = 0.9,
    fit: str = "front",
) -> tuple[list, list]:
    """
    RGBA renders (PIL) and silhouettes (bool arrays) of a mesh from MV-Adapter's cameras (orthographic,
    level, +-half_extent) at the given azimuths (pixal3d_worker.cameras' convention: 0 on the GLB's +Z).
    With ``fill``, the mesh is first centred and scaled: ``fit`` "front" frames it as MV-Adapter frames a
    picture (its longer side in the 0-degree view at that share of the frame), "cube" as Pixal3D's
    training renders did (its longest side, depth included, at that share). Without, it is drawn where
    it is.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from forge3d_worker import projection as P
    from pixal3d_worker import cameras

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = cameras.glb_to_world(mesh["positions"])
    if fill is not None:
        world = world - (world.min(0) + world.max(0)) / 2
        if fit == "cube":
            extent = max(np.ptp(world, axis=0))
        else:
            extent = max(np.ptp(world[:, 0]), np.ptp(world[:, 2]))  # right and up in the 0-degree view
        world = world * (fill * 2 * half_extent / extent)
    verts = torch.tensor(world, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh["faces"], dtype=torch.long, device=device)
    uv = torch.tensor(np.c_[mesh["uv"][:, 0], 1 - mesh["uv"][:, 1]], dtype=torch.float32, device=device)
    texture = torch.tensor(np.asarray(mesh["texture"]), device=device).float().permute(2, 0, 1)[None] / 255
    normals = P._welded_normals(verts, faces)
    big = size * 2
    pictures, silhouettes = [], []
    for azimuth in azimuths:
        c2w = torch.tensor(cameras.orbit_camera(float(azimuth), 0.0, 3.0), dtype=torch.float32, device=device)
        local = (verts - c2w[:3, 3]) @ c2w[:3, :3]  # right, up, back
        xy = torch.stack([big / 2 * (1 + local[:, 0] / half_extent), big / 2 * (1 - local[:, 1] / half_extent)], -1)
        depth = -local[:, 2]
        ones = torch.ones_like(depth)
        zbuf = P.rasterize_depth(xy, depth, ones, faces, big, big)
        face, bary = P.rasterize_faces(xy, depth, ones, faces, zbuf)
        hit = face >= 0
        corners = faces[face.clamp(min=0)]
        colour = P._sample_texture(texture, (bary[..., None] * uv[corners]).sum(-2).view(-1, 2)).view(big, big, 3)
        normal = (bary[..., None] * normals[corners]).sum(-2)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        light = c2w[:3, 2] + 0.5 * c2w[:3, 1] - 0.3 * c2w[:3, 0]  # from the camera, above, a little left
        shade = 0.62 + 0.38 * (normal @ (light / light.norm())).clamp(min=0)
        alpha = hit.float()[..., None]
        rgba = torch.cat([colour * shade[..., None] * alpha, alpha], -1).permute(2, 0, 1)[None]
        rgba = F.avg_pool2d(rgba, 2)[0].permute(1, 2, 0)  # 2x supersampled, premultiplied
        a = rgba[..., 3:]
        pixels = torch.cat([rgba[..., :3] / a.clamp_min(1e-6), a], -1).clamp(0, 1)
        pictures.append(Image.fromarray(np.ascontiguousarray((pixels * 255).round().byte().cpu().numpy()), "RGBA"))
        silhouettes.append((a[..., 0] > 0.5).cpu().numpy())
    return pictures, silhouettes


def iou(a, b) -> float:
    import numpy as np

    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# --- The GPU class ----------------------------------------------------------------------------------------

STARTED = time.time()


@app.cls(
    image=pixal3d_image,
    gpu="L40S",
    cpu=4.0,
    memory=49152,  # TRELLIS.2 for the previews lives on the CPU between uses (level="preview")
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=20,  # the next job is sent as soon as this one is saved
    max_containers=2,
)
class Pixal3D:
    weights: str = modal.parameter(default="multiview")
    # How a single picture's model is levelled: "preview" (TRELLIS.2's '512' preview, the picture's camera
    # found against it: production's way), "given" (the job's Phase 5 pose) or "none"
    level: str = modal.parameter(default="none")

    @modal.enter()
    def load(self) -> None:
        import torch

        if not WEIGHTS_MARKER.exists():
            raise RuntimeError("Pixal3D weights missing: modal run ops/exp_pixal3d.py::download")
        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from pixal3d_worker.pipeline import Pixal3DRuntime

        started = time.time()
        self.runtime = Pixal3DRuntime(multiview=self.weights == "multiview", level=self.level)
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[pixal3d] {self.weights} weights (level {self.level}) on {self.gpu} after {time.time() - STARTED:.0f} s")

    @modal.method()
    def make(self, job: dict) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        from forge3d_worker.compress import pack_glb
        from forge3d_worker.storage import InlineStorage
        from pixal3d_worker.service import handle_job

        root = pathlib.Path(prod.OUTPUTS)
        prod.outputs.reload()
        source = root / job["source"]
        state = json.loads((source / "progress.json").read_text())
        picture = (source / state["input"]).read_bytes()
        seed = state["seed"]
        run = job.get("run_name") or f"{job['variant']}-{job['source']}"
        out = root / RESULTS / job["plan"] / run
        out.mkdir(parents=True, exist_ok=True)

        views, camera, summary = [], None, {"run": run, "variant": job["variant"], "seed": seed}
        tilt = job.get("tilt")  # Phase 5's (elevation, roll) for level="given"
        self.runtime.given_tilt = tuple(tilt) if tilt else (0.0, 0.0)
        # The picture's camera elevation into its camera as an absolute pose (multi-view weights only)
        self.runtime.posed = bool(job.get("posed")) and self.weights == "multiview"
        if tilt:
            summary["given_tilt"] = list(tilt)
        if self.runtime.posed:
            summary["posed"] = True
        azimuths = job.get("azimuths") or []
        if job.get("views") == "synthetic":  # the object's Phase 5 final (phase2d: the same picture and seed)
            finals = root / job["finals"]
            final = json.loads((finals / "progress.json").read_text())["steps"]["final"]["files"][0]
            fit = job.get("fit", "front")
            pictures, _ = render_views(plain_mesh((finals / final).read_bytes()), azimuths, fit=fit)
            summary["views_from"] = f"{job['finals']}/{final} (framed by its {fit})"
        elif job.get("views") == "mvadapter":
            folder = root / VIEWS / job["source"]
            pictures = [Image.open(folder / f"view-{a}.png").convert("RGBA") for a in azimuths]
            info = json.loads((folder / "views.json").read_text()) if (folder / "views.json").exists() else {}
            camera = info.get("camera")
            summary["views_from"] = f"{VIEWS}/{job['source']}"
        else:
            pictures = []
        for azimuth, view in zip(azimuths, pictures):
            (out / "views").mkdir(exist_ok=True)
            buffer = io.BytesIO()
            view.save(buffer, "PNG")
            (out / "views" / f"view-{azimuth}.png").write_bytes(buffer.getvalue())
            views.append({"image_base64": base64.b64encode(buffer.getvalue()).decode(), "azimuth": azimuth, "elevation": 0})

        self.runtime.main = job.get("main", "view")
        self.runtime.azimuths = None
        request = {"image_base64": base64.b64encode(picture).decode(), "mode": job.get("mode", "final"), "seed": seed}
        request["request_id"] = run[:64]
        if views:
            request["views"] = views
        if camera:
            request["camera"] = camera
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        result = handle_job({"id": run, "input": request}, self.runtime, InlineStorage(), pack_glb)
        seconds = round(time.time() - started, 1)
        if result.get("error"):
            raise RuntimeError(f"{run}: {result['error']}")
        packed = base64.b64decode(result["glb"]["base64"])

        if views:  # the rebuild seen from each view's camera, against the view's silhouette
            mesh = plain_mesh(packed)
            # The rebuild is in Pixal3D's cube: the views' frame there is the one the worker fitted
            half = float((result.get("camera") or {}).get("half_extent") or (camera or {}).get("half_extent", 0.55))
            _, rebuilt = render_views(mesh, azimuths, half_extent=half, fill=None)
            given = [np.asarray(v.getchannel("A")) > 127 for v in pictures]
            summary["silhouette_iou"] = {str(a): round(iou(g, r), 3) for a, g, r in zip(azimuths, given, rebuilt)}

        name = f"{job.get('mode', 'final')}-{seed}.glb"
        (out / name).write_bytes(packed)
        (out / state["input"]).write_bytes(picture)
        step = {
            "status": "done",
            "files": [name],
            "triangles": result["triangles"],
            "bytes": result["bytes"],
            "timings": result["timings"],
            "gpu_seconds": round(sum(result["timings"].values()), 1),
            "pipeline": result.get("pipeline"),
            "views_used": result.get("views_used"),
        }
        for key in ("projection", "camera", "pose", "level"):
            if result.get(key):
                step[key] = result[key]
        steps = {"final": step}
        files = {name: packed, state["input"]: picture}
        preview_raw = getattr(self.runtime, "last_preview", None)
        if preview_raw:  # TRELLIS.2's '512' preview the pose was found against, packed like production's
            preview_name = f"preview-{seed}.glb"
            preview_packed = pack_glb(preview_raw, 1024)
            (out / preview_name).write_bytes(preview_packed)
            files[preview_name] = preview_packed
            pose = result.get("pose") or {}
            steps["preview"] = {
                "status": "done",
                "files": [preview_name],
                "bytes": len(preview_packed),
                "pipeline": (pose.get("preview") or {}).get("pipeline", "512"),
                "seconds": pose.get("seconds"),
                "made_by": "TRELLIS.2, for the picture's camera (pixal3d_worker.level)",
            }
        progress = {
            "prompt": state.get("prompt"),
            "seed": seed,
            "input": state["input"],
            "final": True,
            "steps": steps,
            "made_by": {"worker": "pixal3d", "plan": job["plan"], "variant": job["variant"], "picture_from": job["source"]},
            "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        (out / "progress.json").write_text(json.dumps(progress, indent=2))
        prod.outputs.commit()
        prod.share_caches()
        summary.update(
            {
                "seconds": seconds,
                "timings": result["timings"],
                "triangles": result["triangles"],
                "views_used": result.get("views_used"),
                "pipeline": result.get("pipeline"),
                "projection": result.get("projection"),
                "camera": result.get("camera"),
                "pose": result.get("pose"),
                "level": result.get("level"),
                "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
                "load_seconds": self.load_seconds,
                "container_seconds": round(time.time() - STARTED, 1),
                "gpu": self.gpu,
            }
        )
        files["progress.json"] = json.dumps(progress, indent=2).encode()
        for azimuth in azimuths:
            files[f"views/view-{azimuth}.png"] = (out / "views" / f"view-{azimuth}.png").read_bytes()
        return {"summary": summary, "files": files}


# --- Plans -------------------------------------------------------------------------------------------------

BACKS = ["04", "06", "08", "11"]
CONTROLS = ["01", "13"]
# The brief's other controls, which must not get worse either
MORE_CONTROLS = ["02", "05", "07", "09", "10", "17", "19", "20"]
ALL = [f"{n:02d}" for n in range(1, 21)]
SIX = [0, 45, 90, 180, 270, 315]
FOUR = [0, 90, 180, 270]
# The picture's camera (elevation, roll) as Phase 5's projection placed it against the TRELLIS.2 final
# (phase2d-NN progress.json, steps.final.projection.pose); 03 and 18 failed its gate (IoU 0.91, 0.66)
PHASE5_POSES: dict[str, tuple[float, float]] = {
    "01": (-3.1, 0.0), "02": (5.0, 0.0), "03": (1.2, 0.0), "04": (5.6, 0.0), "05": (7.5, 0.0),
    "06": (6.9, 0.0), "07": (5.0, 0.0), "08": (0.0, 0.0), "09": (28.8, 0.0), "10": (21.2, 0.5),
    "11": (10.0, 0.0), "12": (47.5, 0.0), "13": (10.0, -1.5), "14": (10.0, 0.0), "15": (7.5, 0.0),
    "16": (10.0, 0.0), "17": (45.0, -7.9), "18": (-5.0, 4.0), "19": (0.0, 0.0), "20": (4.4, 0.5),
}  # fmt: skip
# Run 3: the objects the tilt shows on, the backs and the car
TILTED = ["09", "10", "17", "04", "13"]

PLANS: dict[str, dict] = {
    "single": {"weights": "single", "jobs": [{"number": n, "variant": "single"} for n in BACKS + CONTROLS]},
    # Run 3, the picture's elevation into Pixal3D itself (multi-view weights): the picture's camera as an
    # absolute pose above the front ("posed"), against levelling the same weights' camera-aligned model
    # afterwards ("mvlevel", experiment B on these weights)
    "posed": {
        "weights": "multiview",
        "level": "given",
        "jobs": [{"number": n, "variant": "posed", "tilt": PHASE5_POSES[n], "posed": 1} for n in TILTED]
        + [{"number": n, "variant": "mvlevel", "tilt": PHASE5_POSES[n]} for n in ["09", "17"]],
    },
    # Run 4: production's recipe on all twenty, at the phase2d seeds. The multi-view weights with the
    # picture as their one view: in run 3 they gave the ramen a white bowl all round and the donut its
    # frosting where the single-view weights (runs 1-2) invented a dark outside
    "final20": {
        "weights": "multiview",
        "level": "preview",
        "jobs": [{"number": n, "variant": "phase2f", "name_like_source": "phase2f"} for n in ALL],
    },
    # The same recipe on the single-view weights, for the comparison
    "final20single": {
        "weights": "single",
        "level": "preview",
        "jobs": [{"number": n, "variant": "phase2fs", "name_like_source": "phase2fs"} for n in ALL],
    },
    # The single-view weights on the rest of the controls (run 2)
    "controls": {"weights": "single", "jobs": [{"number": n, "variant": "single"} for n in MORE_CONTROLS]},
    # Consistent views of known models: the rebuild must line up with them (cameras and framing)
    "synthetic": {
        "weights": "multiview",
        "jobs": [{"number": n, "variant": "syn6", "views": "synthetic", "azimuths": SIX} for n in ("04", "13")]
        + [{"number": "04", "variant": "syn4", "views": "synthetic", "azimuths": FOUR}],
    },
    # The MV line's MV-Adapter views: all six, the redrawn front as the main view
    "mvadapter": {
        "weights": "multiview",
        "jobs": [{"number": n, "variant": "mv6", "views": "mvadapter", "azimuths": SIX} for n in BACKS],
    },
    # Run 2, the multi-view weights: the 11 books' mv6 again (its export died on a stale CUDA error),
    # the car's synthetic views framed so it fits the cube (the first run's side views were cut off),
    # then which views: four (Pixal3D's own rig, no 45/315, which wash out or cut off), and the picture
    # instead of the redrawn front
    "mvchoice": {
        "weights": "multiview",
        "jobs": [{"number": "11", "variant": "mv6", "views": "mvadapter", "azimuths": SIX}]
        + [{"number": "13", "variant": "syn6fit", "views": "synthetic", "azimuths": SIX, "fit": "cube"}]
        + [{"number": n, "variant": "mv4", "views": "mvadapter", "azimuths": FOUR} for n in BACKS]
        + [{"number": n, "variant": "mv6pic", "views": "mvadapter", "azimuths": SIX, "main": "picture"} for n in BACKS],
    },
}


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the pictures; phase2d: Phase 5's finals)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


@app.local_entrypoint()
def experiment(plan: str, out: str = "ops-out/private", only: str = "", containers: int = 1) -> None:
    if plan not in PLANS:
        raise SystemExit(f"--plan must be one of {sorted(PLANS)}")
    settings = PLANS[plan]
    pictures = runs_by_number(prod.outputs, "phase2")
    finals = runs_by_number(prod.outputs, "phase2d")
    jobs = []
    for job in settings["jobs"]:
        number = job["number"]
        if only and number not in only.split(","):
            continue
        if number not in pictures or (job.get("views") == "synthetic" and number not in finals):
            print(f"[pixal3d] no phase2 run (or no phase2d final) for {number}, skipped")
            continue
        job = {**job, "plan": plan, "source": pictures[number], "finals": finals.get(number)}
        if job.get("name_like_source"):  # phase2f-NN-<slug>: Phase 6's final of the phase2 picture
            job["run_name"] = pictures[number].replace("phase2-", job["name_like_source"] + "-", 1)
        jobs.append(job)
    print(f"[pixal3d] {plan}: {len(jobs)} jobs on {settings['weights']} weights (level {settings.get('level', 'none')})")
    worker = Pixal3D(weights=settings["weights"], level=settings.get("level", "none"))
    summaries = []
    failures = 0

    def save(job: dict, made) -> None:
        nonlocal failures
        name = job.get("run_name") or f"{job['variant']}-{job['source']}"
        if isinstance(made, BaseException):
            print(f"[pixal3d] {name}: failed: {type(made).__name__}: {made}")
            summaries.append({"run": name, "error": f"{type(made).__name__}: {made}"})
            failures += 1
            return
        summary = made["summary"]
        target = pathlib.Path(out) / plan / summary["run"]
        for file_name, data in made["files"].items():
            (target / file_name).parent.mkdir(parents=True, exist_ok=True)
            (target / file_name).write_bytes(data)
        summaries.append(summary)
        print(f"[pixal3d] {json.dumps(summary)}")

    if containers > 1:  # the jobs spread over up to that many warm containers (the class caps them at 2)
        for job, made in zip(jobs, worker.make.map(jobs, return_exceptions=True, order_outputs=True)):
            save(job, made)
    else:
        for job in jobs:  # one at a time: one container, warm between jobs
            try:
                made = worker.make.remote(job)
            except Exception as err:  # noqa: BLE001 - report it and go on with the next object
                made = err
            save(job, made)
            if failures == 2 and len(summaries) == 2:  # the first two failed: something general, stop paying
                print("[pixal3d] the first two jobs failed; not trying the rest")
                break
    pathlib.Path("ops-out").mkdir(exist_ok=True)
    pathlib.Path(f"ops-out/pixal3d-{plan}.json").write_text(json.dumps(summaries, indent=2))
    ages = [s.get("container_seconds", 0) for s in summaries if "error" not in s]
    print(f"[pixal3d] {plan}: {len(ages)} of {len(jobs)} made; the container was up {max(ages, default=0):.0f} s")
