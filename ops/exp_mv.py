"""
Phase 6, multiview line: MV-Adapter views of the Phase 2 pictures, on an app of its own
(orainge-exp-mv). It never deploys anything and never touches the production app orainge-ai.

    modal run ops/exp_mv.py::views --only 04,06,08,11        # views of those Phase 2 runs
    modal run ops/exp_mv.py::views --only all
    modal run ops/exp_mv.py::views --only 04 --variants "caption:prompt=caption;s30:steps=30"

The weights go into orainge-models first if they aren't there (download, CPU only, the same script
and marker as modal_app.download_models). Each Phase 2 run's picture is read from the orainge-outputs volume (/outputs/phase2-NN-*/
progress.json -> "input") and goes through the worker's own handle_job. The views land in
/outputs/phase6/views/<run>/ (view-<azimuth>.png, views.json, reference.png, raw-<azimuth>.jpg);
a variant other than "views" lands in /outputs/phase6/mv/<variant>/<run>/. Everything is copied
to ops-out/private/views/... for local use. Variant keys: prompt (default | caption, the Phase 2
prompt | back, BACK_PROMPTS below), steps, guidance, fill, seed (an offset added to the run's seed),
elevation (degrees, every view), extent (the orthographic half-extent, 0.55 trained), negative
(replaces the negative prompt; spaces as _), runs (a subset of --only for this variant, 04+06).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Locally modal_app.py is in workers/; in the containers it is added to /root (below)
for candidate in (ROOT / "workers", pathlib.Path("/root")):
    try:
        found = (candidate / "modal_app.py").is_file()
    except OSError:  # /root on GitHub's runners can't even be looked into
        found = False
    if found:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
from modal_app import MODELS, MULTIVIEW_GPU, OUTPUTS, WORKERS, download_image, models, multiview_image, outputs  # noqa: E402

app = modal.App("orainge-exp-mv")
CANONICAL = "views"
# prompt=back: the Phase 2 prompt plus what the hidden side should look like, by run number
BACK_PROMPTS = {
    "04": "a retro arcade machine, a plain flat back panel, smooth sides",
    "06": "a wooden shield with a lion painted on the front, the back is plain wooden planks",
    "08": "a vintage film camera; the back is a flat film door with a small viewfinder eyepiece, the lens only on the front",
    "11": "a stack of old books with a candle on top; spines on one side, page edges on the others",
    "16": "a green cartoon dragon sitting, seen from all sides, its back has wings and a spiky spine",
    "18": "an electric guitar; the back of the body is plain painted wood, pickups only on the front",
}
# The production images (this checkout's worker code), plus modal_app itself, which this file imports
# again in the container
gpu_image = multiview_image.add_local_file(WORKERS / "modal_app.py", "/root/modal_app.py")
weights_image = download_image.add_local_file(WORKERS / "modal_app.py", "/root/modal_app.py")


@app.function(
    image=weights_image,
    volumes={MODELS: models},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    cpu=2.0,
    timeout=3600,
)
def download(force: bool = False) -> str:
    """modal_app.download_models --which multiview, run from this app (same script, same marker)."""
    import hashlib
    import os
    import subprocess

    script = pathlib.Path("/root/weights/multiview.py")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    marker = pathlib.Path(MODELS) / ".multiview-weights"
    if not force and marker.exists() and marker.read_text() == digest:
        return "multiview: already downloaded"
    marker.unlink(missing_ok=True)
    started = time.monotonic()
    env = {**os.environ, "MODELS_ROOT": MODELS, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    subprocess.run([sys.executable, str(script)], env=env, check=True)
    marker.write_text(digest)
    models.commit()
    sizes = {}
    for name in ("mv-adapter", "stable-diffusion-xl-base-1.0", "sdxl-vae-fp16-fix", "BiRefNet"):
        files = [path for path in (pathlib.Path(MODELS) / name).rglob("*") if path.is_file()]
        sizes[name] = f"{len(files)} files, {sum(path.stat().st_size for path in files) / 1e9:.2f} GB"
    return f"multiview: downloaded in {time.monotonic() - started:.0f} s; {sizes}"


class Capture:
    """Storage that keeps the PNGs in memory."""

    def __init__(self) -> None:
        self.saved: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        self.saved[key] = data
        return {"key": key, "url": None}


def silhouette(view) -> dict:
    """Where the cutout is: its box (alpha over half) in pixels, and its share of the frame."""
    import numpy as np

    found = np.asarray(view)[..., 3] > 127
    if not found.any():
        return {"bbox": None, "coverage": 0.0}
    ys, xs = np.nonzero(found)
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "coverage": round(float(found.mean()), 4),
    }


@app.function(
    image=gpu_image,
    gpu=MULTIVIEW_GPU,
    cpu=2.0,
    memory=16384,
    volumes={MODELS: models, OUTPUTS: outputs},
    timeout=3600,
)
def make_views(runs: list, variants: list, pins: dict) -> dict:
    import base64
    import io

    import torch
    from PIL import Image

    import multiview_worker.generator as generator_module
    from multiview_worker.cameras import ELEVATION, HALF_EXTENT, IMAGE_SIZE, camera_to_world
    from multiview_worker.generator import FILL, GUIDANCE, NEGATIVE_PROMPT, STEPS, MultiViewGenerator, control_images
    from multiview_worker.service import handle_job

    started = time.monotonic()
    if not (pathlib.Path(MODELS) / ".multiview-weights").exists():
        raise RuntimeError("run ops/exp_mv.py::download first")
    generator = MultiViewGenerator(MODELS)
    load_s = round(time.monotonic() - started, 1)
    loaded_gb = round(torch.cuda.memory_allocated() / 2**30, 2)
    print(f"[mv] {torch.cuda.get_device_name()}: loaded in {load_s} s, {loaded_gb} GiB allocated", flush=True)
    trained_control = generator.control

    class Variant:
        def __init__(self, spec: dict) -> None:
            self.steps = int(spec.get("steps", STEPS))
            self.guidance = float(spec.get("guidance", GUIDANCE))
            self.fill = float(spec.get("fill", FILL))
            self.elevation = float(spec.get("elevation", ELEVATION))
            self.extent = float(spec.get("extent", HALF_EXTENT))
            self.negative = spec.get("negative", "").replace("_", " ") or NEGATIVE_PROMPT
            self.control = trained_control
            if self.elevation != ELEVATION or self.extent != HALF_EXTENT:
                self.control = control_images(generator.device, elevation=self.elevation, half_extent=self.extent)

        def __call__(self, image, seed, prompt):
            # Experiment-only knobs: the camera maps and the negative prompt are the worker's constants
            generator.control = self.control
            generator_module.NEGATIVE_PROMPT = self.negative
            try:
                return generator(image, seed, prompt, steps=self.steps, guidance=self.guidance, fill=self.fill)
            finally:
                generator.control = trained_control
                generator_module.NEGATIVE_PROMPT = NEGATIVE_PROMPT

        @property
        def last_timings(self):
            return generator.last_timings

    results: dict = {}
    for spec in variants:  # the canonical views first: other lines of work wait for them
        name = spec["name"]
        variant = Variant(spec)
        wanted = {number.zfill(2) for number in spec.get("runs", "").split("+") if number}
        for run in runs:
            if wanted and run.split("-")[1] not in wanted:
                continue
            source = pathlib.Path(OUTPUTS) / run
            state = json.loads((source / "progress.json").read_text())
            seed = (int(state["seed"]) + int(spec.get("seed", 0))) % 2**31
            job = {
                "image_base64": base64.b64encode((source / state["input"]).read_bytes()).decode("ascii"),
                "seed": seed,
                "request_id": run[:64],
            }
            if spec.get("prompt") == "caption" and state.get("prompt"):
                job["prompt"] = state["prompt"]
            elif spec.get("prompt") == "back":
                job["prompt"] = BACK_PROMPTS.get(run.split("-")[1], state.get("prompt", "high quality"))
            torch.cuda.reset_peak_memory_stats()
            storage = Capture()
            out = handle_job({"id": run, "input": job}, variant, storage)
            key = f"{name}/{run}"
            if out.get("error"):
                print(f"[mv] {key}: {out['error']}", flush=True)
                results[key] = {"error": out["error"]}
                continue
            folder = pathlib.Path(OUTPUTS) / "phase6" / ("views" if name == CANONICAL else f"mv/{name}") / run
            folder.mkdir(parents=True, exist_ok=True)
            views = []
            for entry, raw in zip(out["views"], generator.last_views):
                azimuth = entry["azimuth"]
                data = storage.saved[entry["key"]]
                (folder / f"view-{azimuth}.png").write_bytes(data)
                raw.save(folder / f"raw-{azimuth}.jpg", quality=92)
                views.append(
                    {
                        "azimuth": azimuth,
                        "elevation": variant.elevation,
                        "file": f"view-{azimuth}.png",
                        "raw": f"raw-{azimuth}.jpg",
                        "c2w": [
                            [round(value, 6) for value in row]
                            for row in camera_to_world(azimuth, variant.elevation).tolist()
                        ],
                        "silhouette": silhouette(Image.open(io.BytesIO(data))),
                    }
                )
            generator.last_reference.save(folder / "reference.png")
            record = {
                "run": run,
                "picture": f"/outputs/{run}/{state['input']}",
                "seed": seed,
                "prompt": job.get("prompt", "high quality"),
                "settings": {
                    "steps": variant.steps,
                    "guidance": variant.guidance,
                    "fill": variant.fill,
                    "elevation": variant.elevation,
                    "half_extent": variant.extent,
                    "negative_prompt": variant.negative,
                },
                "models": pins,
                "camera": {
                    **out["camera"],
                    "elevation": variant.elevation,
                    "half_extent": variant.extent,
                    "pixels_per_unit": round(IMAGE_SIZE / (2 * variant.extent), 3),
                },
                "views": views,
                "reference": "reference.png: the 768 x 768 picture the views were conditioned on",
                "background": (
                    "raw-*.jpg are the views as drawn, on MV-Adapter's flat mid-gray (127) training background; "
                    "view-*.png are the same pixels with BiRefNet's cutout as alpha"
                ),
                "seconds": out["seconds"],
                "timings": out["timings"],
                "vram_peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            }
            (folder / "views.json").write_text(json.dumps(record, indent=2))
            outputs.commit()
            files = sorted(path.name for path in folder.iterdir())
            results[key] = {
                "folder": str(folder.relative_to(OUTPUTS)),
                "files": files,
                "seconds": out["seconds"],
                "timings": out["timings"],
                "vram_peak_gib": record["vram_peak_gib"],
                "silhouettes": {view["azimuth"]: view["silhouette"] for view in views},
            }
            print(f"[mv] {key}: {out['seconds']} s, peak {record['vram_peak_gib']} GiB reserved", flush=True)
    return {"runs": results, "load_s": load_s, "total_s": round(time.monotonic() - started, 1), "gpu": torch.cuda.get_device_name()}


def parse_variants(text: str) -> list[dict]:
    """'views;caption:prompt=caption;s30:steps=30' -> [{"name": "views"}, {"name": "caption", "prompt": "caption"}, ...]"""
    variants = []
    for part in filter(None, (piece.strip() for piece in text.split(";"))):
        name, _, settings = part.partition(":")
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", name):
            raise SystemExit(f"bad variant name {name!r}")
        spec: dict = {"name": name}
        for item in filter(None, (piece.strip() for piece in settings.split(","))):
            key, _, value = item.partition("=")
            if key not in {"prompt", "steps", "guidance", "fill", "seed", "elevation", "extent", "negative", "runs"}:
                raise SystemExit(f"unknown variant setting {key!r}")
            spec[key] = value
        variants.append(spec)
    return variants


def pins() -> dict:
    spec = importlib.util.spec_from_file_location("mv_weights", ROOT / "workers/multiview/scripts/download_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "adapter": f"{module.MVADAPTER[0]}@{module.MVADAPTER[1]} {module.ADAPTER_FILE}",
        "base": f"{module.SDXL[0]}@{module.SDXL[1]} (fp16)",
        "vae": f"{module.VAE[0]}@{module.VAE[1]}",
        "cutout": f"{module.BIREFNET[0]}@{module.BIREFNET[1]}",
        "code": "huanngzh/MV-Adapter@4277e0018232bac82bb2c103caf0893cedb711be (workers/multiview/mvadapter)",
    }


@app.local_entrypoint()
def views(only: str = "04,06,08,11", variants: str = CANONICAL, out: str = "ops-out/private/views") -> None:
    names = sorted(
        entry.path.strip("/") for entry in outputs.listdir("/") if re.fullmatch(r"phase2-\d\d-[a-z0-9-]+", entry.path.strip("/"))
    )
    if only != "all":
        wanted = {number.strip().zfill(2) for number in only.split(",") if number.strip()}
        names = [name for name in names if name.split("-")[1] in wanted]
    chosen = parse_variants(variants)
    print(download.remote())  # quick when the weights are there already
    print(f"[mv] {len(names)} runs x {len(chosen)} variants: {', '.join(names)}; {chosen}")
    result = make_views.remote(names, chosen, pins())
    target = pathlib.Path(out)
    for key, entry in result["runs"].items():
        if "files" not in entry:
            continue
        variant, run = key.split("/", 1)
        local = target / run if variant == CANONICAL else target / variant / run
        local.mkdir(parents=True, exist_ok=True)
        for name in entry["files"]:
            (local / name).write_bytes(b"".join(outputs.read_file(f"{entry['folder']}/{name}")))
    summary = pathlib.Path("ops-out") / f"mv-summary-{int(time.time())}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2))
    for key, entry in result["runs"].items():
        print(f"{key}: {entry.get('error') or entry['seconds']}")
    print(f"[mv] {result['gpu']}: load {result['load_s']} s, total {result['total_s']} s")
