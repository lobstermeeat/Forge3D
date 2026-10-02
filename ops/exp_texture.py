"""
Phase 7's texture experiment, in a staging app (ORAINGE_APP_NAME; nothing is deployed). For each
picked Phase 2 picture at its Phase 4 seed, production's final (TRELLIS.2 '1024_cascade', exported as
production does) and, on that same shape, textures made other ways:

- `t7base`: production's final as it is, the control.
- `t7cfg`: TRELLIS.2's texture flow again with CFG 3 (guidance rescale 0.5); its default is 1 (no CFG).
- `t7steps`: the texture flow again with 24 steps instead of 12.
- `t7bake`: MV-Adapter's image+geometry model (`GeometryViews`) draws six views of this shape from the
  picture, and `mvtexture.bake_views` bakes them into the texture before the picture's projection.
- `t7views`: the same views' right, back and left sides steer TRELLIS.2's texture flow alone
  (`Trellis2Runtime.retexture(views=...)`), so TRELLIS.2 still writes its own texture.
- `t7replay` (with --replay): the generation's texture sampled again with nothing changed, which must
  match t7base: the check that only the method differs between variants.

Every variant shares the generation's texture noise. Results land in the orainge-outputs volume under
phase7/tex/ and in --out, one folder per variant and object, named like the Phase 2 runs.

    ORAINGE_APP_NAME=orainge-p7-tex modal run ops/exp_texture.py::experiment --only 04 --replay 04

`rolls` is the follow-up: no texture method won, but the texture is a lottery (best of five beat one),
so each of the 20 pictures gets its final (`t8roll0`, production's) and `--count` more textures on the
same shape, each from new noise (`t8roll<k>`: torch.manual_seed(seed + 1000 k)), all exported as
production does. Each is also rendered from ig2mv's six cameras (renders/<run>/view-<i>.jpg, no light),
and ig2mv draws its six views of the shape once (views/<source>/), for pickers scored afterwards.

    ORAINGE_APP_NAME=orainge-p7-tex modal run ops/exp_texture.py::rolls --count 3
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import re
import sys
import time

import modal

if modal.is_local() and os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
# Here: the checkout's workers/. In the container: /root, where the image puts modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app
RESULTS = "phase7/tex"
# The texture misses first (04 arcade, 06 shield, 08 camera, 11 books, 12 skateboard, 14 bubble tea,
# 18 guitar), then controls (01 fox, 02 sneaker, 13 car, 16 dragon, 20 balloon)
OBJECTS = ["04", "06", "08", "11", "12", "14", "18", "01", "02", "13", "16", "20"]
CFG = {"guidance_strength": 3.0, "guidance_rescale": 0.5}
STEPS = {"steps": 24}
# The views that steer the texture flow: ig2mv's right, back and left (its front is the picture's side;
# straight up and down are views TRELLIS.2 rarely saw)
SIDE_VIEWS = ((1, 90.0), (2, 180.0), (3, 270.0))
# rolls: every Phase 2 prompt, the slowest (the books) first so the containers finish together
ROLL_OBJECTS = ["11", *(f"{n:02d}" for n in range(1, 21) if n != 11)]
ROLL_SEED = 1000  # roll k's texture noise: torch.manual_seed(seed + ROLL_SEED * k)

tex_image = (
    prod.trellis2_image
    # The control maps' PNG packing (numpy and PIL only)
    .add_local_dir(prod.WORKERS / "multiview" / "multiview_worker", "/root/multiview_worker")
    .add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")
)

STARTED = time.time()


@app.cls(
    image=tex_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=4,
)
class Texture:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        defaults = dict(self.runtime.pipeline.tex_slat_sampler_params)
        print(f"[tex] TRELLIS.2 on {self.gpu} in {self.load_seconds} s; texture sampler {json.dumps(defaults)}")

    @modal.method()
    def make(self, job: dict) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        from forge3d_worker import mvtexture
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.inputs import View
        from forge3d_worker.settings import PRESETS
        from multiview_worker.geometry import pack_control

        runtime = self.runtime
        root = pathlib.Path(prod.OUTPUTS)
        prod.outputs.reload()
        source = root / job["source"]
        state = json.loads((source / "progress.json").read_text())
        picture_bytes = (source / state["input"]).read_bytes()
        picture = Image.open(io.BytesIO(picture_bytes))
        picture.load()
        seed = int(state["seed"])
        prompt = state.get("prompt") or "high quality"
        subject = job["source"].split("-", 1)[1]  # phase2-04-retro-arcade-machine -> 04-retro-arcade-machine
        preset = PRESETS["final"]
        files: dict[str, bytes] = {}
        raws: dict[str, bytes] = {}  # each variant's GLB as exported, before gltfpack
        summary: dict = {"source": job["source"], "seed": seed, "prompt": prompt, "variants": {}}
        rolls = job.get("plan") == "rolls"
        torch.cuda.reset_peak_memory_stats()

        started = time.time()
        mesh = runtime.generate(picture, preset, seed)
        summary["generate_s"] = round(time.time() - started, 1)
        summary["pipeline"] = getattr(runtime, "pipeline_used", None)

        def export(variant: str, made: object, hook=None, **notes) -> None:
            runtime.before_projection = hook
            clock = time.time()
            try:
                raw, triangles = runtime.export(made, preset)
            finally:
                runtime.before_projection = None
            if rolls:
                raws[variant] = raw
            packed = pack_glb(raw, preset.texture_size)
            run = f"{variant}-{subject}"
            out = root / RESULTS / run
            out.mkdir(parents=True, exist_ok=True)
            name = f"final-{seed}.glb"
            (out / name).write_bytes(packed)
            (out / state["input"]).write_bytes(picture_bytes)
            step = {
                "status": "done",
                "files": [name],
                "triangles": triangles,
                "bytes": len(packed),
                "export_s": round(time.time() - clock, 1),
                "variant": variant,
                **notes,
            }
            if runtime.last_projection:
                step["projection"] = runtime.last_projection
            hook_report = getattr(runtime, "last_before_projection", None)
            if hook is not None and isinstance(hook_report, dict):
                step["hook"] = {k: v for k, v in hook_report.items() if k != "debug"}
            progress = {
                "prompt": state.get("prompt"),
                "seed": seed,
                "input": state["input"],
                "final": True,
                "steps": {"final": step},
                "made_by": {"experiment": "phase7 texture", "variant": variant, "picture_from": job["source"]},
                "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
            (out / "progress.json").write_text(json.dumps(progress, indent=2, default=str))
            files[f"{run}/{name}"] = packed
            files[f"{run}/{state['input']}"] = picture_bytes
            files[f"{run}/progress.json"] = json.dumps(progress, indent=2, default=str).encode()
            summary["variants"][variant] = {k: v for k, v in step.items() if k not in ("files", "status")}
            print(f"[tex] {run}: {json.dumps(summary['variants'][variant], default=str)}")

        # 1. Production's final, and the control maps of to_glb's mesh (before the projection)
        captured: dict = {}

        def capture(glb, made):
            frame = mvtexture.frame_for(glb, 0.0)
            maps = mvtexture.control_maps(glb, frame)
            captured["frame"] = frame
            captured["control"] = mvtexture.as_control(maps)
            captured["mask"] = maps.mask
            return {"frame": frame.as_dict(), "coverage": [round(float(m.mean()), 4) for m in maps.mask]}

        export("t8roll0" if rolls else "t7base", mesh, capture)

        # 2. Six views of this shape, drawn from the picture by MV-Adapter's image+geometry model
        clock = time.time()
        pngs = pack_control(captured["control"])
        request = {"image_base64": base64.b64encode(picture_bytes).decode(), "control_pngs": pngs, "prompt": prompt, "seed": 0}
        if rolls:
            return self._rolls(job, request, pngs, captured, export, raws, files, summary, seed)
        drawn = prod.GeometryViews().generate.remote(request)
        if drawn.get("error"):
            raise RuntimeError(f"{job['source']}: GeometryViews: {drawn['error']}")
        views = [Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB") for v in drawn["views"]]
        summary["views_s"] = round(time.time() - clock, 1)
        summary["views_timings"] = drawn.get("timings")
        folder = root / RESULTS / "views" / job["source"]
        folder.mkdir(parents=True, exist_ok=True)
        for index, (view, png) in enumerate(zip(views, drawn["views"])):
            data = base64.b64decode(png)
            (folder / f"view-{index}.png").write_bytes(data)
            files[f"views/{job['source']}/view-{index}.png"] = data
        for index, png in enumerate(pngs):
            data = base64.b64decode(png)
            files[f"views/{job['source']}/control-{index:02d}.png"] = data

        # 3. Those views baked into the texture, before the picture's projection
        def bake(glb, made):
            frame = mvtexture.frame_for(glb, 0.0)
            return mvtexture.bake_views(glb, frame, views)

        export("t7bake", mesh, bake)

        # 4. The views' sides steering TRELLIS.2's texture flow alone
        cutouts = []
        for index, azimuth in SIDE_VIEWS:
            rgba = views[index].convert("RGBA")
            rgba.putalpha(Image.fromarray(np.asarray(captured["mask"][index], dtype=np.uint8) * 255))
            cutouts.append(View(rgba, azimuth, 0.0, 1.0))
        clock = time.time()
        made = runtime.retexture(views=cutouts)
        export("t7views", made, retexture_s=round(time.time() - clock, 1), retexture=runtime.last_retexture)

        # 5. The texture sampler's settings
        for variant, params in (("t7cfg", CFG), ("t7steps", STEPS)):
            clock = time.time()
            made = runtime.retexture(sampler_params=params)
            export(variant, made, retexture_s=round(time.time() - clock, 1), retexture=runtime.last_retexture)
        if job.get("replay"):
            clock = time.time()
            made = runtime.retexture()
            export("t7replay", made, retexture_s=round(time.time() - clock, 1), retexture=runtime.last_retexture)

        prod.outputs.commit()
        prod.share_caches()
        summary["peak_gpu_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        summary["load_seconds"] = self.load_seconds
        summary["container_seconds"] = round(time.time() - STARTED, 1)
        summary["gpu"] = self.gpu
        return {"summary": summary, "files": files}

    def _rolls(self, job, request, pngs, captured, export, raws, files, summary, seed) -> dict:
        """
        The rolls plan, after the final's export (t8roll0): ig2mv's views are asked for without waiting
        (the entrypoint collects them by the call's id), the texture is drawn again from new noise
        job["count"] times, and every candidate is rendered from ig2mv's six cameras.
        """
        import torch
        import trimesh

        from forge3d_worker import mvtexture

        runtime = self.runtime
        source = job["source"]
        call = prod.GeometryViews().generate.spawn(request)
        summary["views_call"] = call.object_id
        for index, png in enumerate(pngs):
            files[f"views/{source}/control-{index:02d}.png"] = base64.b64decode(png)
        for k in range(1, int(job.get("count", 3)) + 1):
            clock = time.time()
            made = runtime.retexture(seed=seed + ROLL_SEED * k)
            export(f"t8roll{k}", made, retexture_s=round(time.time() - clock, 1), retexture=runtime.last_retexture)
        # Each candidate as delivered (the projection painted on, before gltfpack) from ig2mv's cameras,
        # base colour only on the views' grey. One that fails to render keeps its GLB
        subject = source.split("-", 1)[1]
        clock = time.time()
        for variant, raw in raws.items():
            try:
                loaded = trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh")
                for index, view in enumerate(mvtexture.render_views(loaded, captured["frame"])):
                    buffer = io.BytesIO()
                    view.save(buffer, "JPEG", quality=92)
                    files[f"renders/{variant}-{subject}/view-{index}.jpg"] = buffer.getvalue()
            except Exception as err:  # noqa: BLE001 - the GLB is the result; renders are for the pickers
                print(f"[tex] {variant}-{subject}: not rendered: {type(err).__name__}: {err}")
                summary.setdefault("render_errors", {})[variant] = f"{type(err).__name__}: {err}"
        summary["renders_s"] = round(time.time() - clock, 1)
        prod.outputs.commit()
        prod.share_caches()
        summary["peak_gpu_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        summary["load_seconds"] = self.load_seconds
        summary["container_seconds"] = round(time.time() - STARTED, 1)
        summary["gpu"] = self.gpu
        return {"summary": summary, "files": files}


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the picked pictures)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


@app.local_entrypoint()
def experiment(only: str = "", out: str = "ops-out/private/tex", replay: str = "") -> None:
    print(f"[tex] app {prod.APP_NAME}")
    # ig2mv's adapter (and the sha256 checks the multiview script now makes); quick when already there
    prod.download_models.remote(which="multiview")
    pictures = runs_by_number(prod.outputs, "phase2")
    numbers = [n.strip().zfill(2) for n in only.split(",") if n.strip()] if only else OBJECTS
    replays = {n.strip().zfill(2) for n in replay.split(",") if n.strip()}
    jobs = [{"number": n, "source": pictures[n], "replay": n in replays} for n in numbers if n in pictures]
    print(f"[tex] {len(jobs)} objects: {', '.join(job['source'] for job in jobs)}")
    summaries, failed = [], 0
    for job, made in zip(jobs, Texture().make.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[tex] {job['source']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"source": job["source"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for name, data in made["files"].items():
            target = pathlib.Path(out) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
        print(f"[tex] {job['source']}: {json.dumps(made['summary'], default=str)}")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out) / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[tex] done: {len(jobs) - failed} of {len(jobs)} objects")
    if jobs and failed == len(jobs):
        raise SystemExit("[tex] nothing was made")


@app.local_entrypoint()
def rolls(only: str = "", out: str = "ops-out/private/rolls", count: int = 3) -> None:
    print(f"[tex] app {prod.APP_NAME}: the final and {count} more textures per picture")
    prod.download_models.remote(which="multiview")
    pictures = runs_by_number(prod.outputs, "phase2")
    numbers = [n.strip().zfill(2) for n in only.split(",") if n.strip()] if only else ROLL_OBJECTS
    jobs = [{"number": n, "source": pictures[n], "plan": "rolls", "count": count} for n in numbers if n in pictures]
    print(f"[tex] {len(jobs)} objects: {', '.join(job['source'] for job in jobs)}")
    root = pathlib.Path(out)
    summaries, failed = [], 0
    for job, made in zip(jobs, Texture().make.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[tex] {job['source']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"source": job["source"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for name, data in made["files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
        brief = {k: v for k, v in made["summary"].items() if k != "variants"}
        print(f"[tex] {job['source']}: {json.dumps(brief, default=str)}")
    # ig2mv's views, asked for by each container without waiting
    for summary in summaries:
        call_id = summary.get("views_call")
        if not call_id:
            continue
        try:
            drawn = modal.FunctionCall.from_id(call_id).get(timeout=1200)
            if drawn.get("error"):
                raise RuntimeError(drawn["error"])
            for index, png in enumerate(drawn["views"]):
                target = root / "views" / summary["source"] / f"view-{index}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(png))
            summary["views_seconds"] = drawn.get("seconds")
            print(f"[tex] {summary['source']}: views in {drawn.get('seconds')} s")
        except Exception as err:  # noqa: BLE001 - the candidates are made; the views only feed one picker
            summary["views_error"] = f"{type(err).__name__}: {err}"
            print(f"[tex] {summary['source']}: no views: {summary['views_error']}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[tex] done: {len(jobs) - failed} of {len(jobs)} objects")
    if jobs and failed == len(jobs):
        raise SystemExit("[tex] nothing was made")
