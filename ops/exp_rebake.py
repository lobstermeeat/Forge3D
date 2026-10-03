"""
A check of the cached texture layout (workers/trellis2/forge3d_worker/rebake.py) on a real GPU, in a staging
app (ORAINGE_APP_NAME; nothing is deployed). For each picked Phase 2 picture at its seed, the final's shape is
made as a textures job makes it (Trellis2Runtime.generate with the final's preset), then two textures of it
(Trellis2Runtime.retexture at the job's seeds, seed + 1000 k), and each texture is exported both ways from the
same retextured mesh:

- texture 1: first the way a textures job's first texture runs (to_glb in full, its layout kept), then
  rebaked on that layout. The two must give the same texture.
- texture 2: first the full way with the cache off (Trellis2Runtime.rebake_textures = False: to_glb in full,
  no layout used or kept), then rebaked on texture 1's layout, as a textures job's later textures are.

Each pair is compared on the surface: both GLBs (before gltfpack) rendered from MV-Adapter's six cameras in
the same frame (mvtexture.render_views: base colour only, no light), and the mean absolute difference taken
over the pixels either covers (0 to 255), with the triangles. Each export's seconds, and those of its to_glb
or rebake step (Trellis2Runtime.last_export), are noted. The packed GLBs, a sheet per texture (the full way
above, the rebake in the middle, the difference times four below) and summary.json land in --out; each object
gets a one-line verdict.

    ORAINGE_APP_NAME=orainge-p7-rebake modal run ops/exp_rebake.py::check --only 04,12
"""

from __future__ import annotations

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
# Base colour differences (mean over the covered pixels of the worst texture, 0 to 255) for the verdict:
# up to SAME the rebake gives the full way's texture, up to CLOSE a texture no one could tell apart. Not 0:
# render_views breaks ties between triangles at their shared edges its own way, so two renders of one GLB
# can differ in a few edge pixels (a CPU run of the tests' box: 0.003)
SAME = 1.0
CLOSE = 4.0
# A pixel whose channels differ by more than this counts as visibly different
VISIBLE = 16
SHEET_VIEW = 256  # pixels per view on the comparison sheets

rebake_image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")

STARTED = time.time()


@app.cls(
    image=rebake_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=3,
)
class Rebake:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[rebake] TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def compare(self, job: dict) -> dict:
        import torch
        from PIL import Image

        from forge3d_worker.service import texture_seed
        from forge3d_worker.settings import PRESETS

        runtime = self.runtime
        root = pathlib.Path(prod.OUTPUTS)
        prod.outputs.reload()
        name = job["source"]
        state = json.loads((root / name / "progress.json").read_text())
        picture_bytes = (root / name / state["input"]).read_bytes()
        picture = Image.open(io.BytesIO(picture_bytes))
        picture.load()
        seed = int(state["seed"])
        preset = PRESETS["final"]
        files: dict[str, bytes] = {f"{name}/picture.png": picture_bytes}
        summary: dict = {"source": name, "seed": seed, "textures": []}
        torch.cuda.reset_peak_memory_stats()

        # The final's shape, as a textures job makes it; its own texture isn't exported (the creator has it)
        clock = time.time()
        mesh = runtime.generate(picture, preset, seed)
        del mesh
        summary["generate_s"] = round(time.time() - clock, 1)
        summary["pipeline"] = runtime.pipeline_used

        for number in range(1, int(job.get("textures", 2)) + 1):
            texture = texture_seed(seed, number)
            clock = time.time()
            mesh = runtime.retexture(seed=texture)
            retexture_s = round(time.time() - clock, 1)
            # Texture 1 the way a textures job's first texture runs (its layout kept); later ones with the
            # cache off. Then the same mesh rebaked, on texture 1's layout
            full = self._export(mesh, preset, rebake=number == 1)
            rebaked = self._export(mesh, preset, rebake=True)
            del mesh
            entry = {"texture_seed": texture, "retexture_s": retexture_s}
            for way, made in (("full", full), ("rebake", rebaked)):
                files[f"{name}/{way}-{texture}.glb"] = made.pop("packed")
                entry[way] = {k: v for k, v in made.items() if k != "raw"}
            entry["compare"], sheet = compare_on_surface(full["raw"], rebaked["raw"])
            entry["compare"]["same_triangles"] = full["triangles"] == rebaked["triangles"]
            entry["compare"]["same_raw_glb"] = full["raw"] == rebaked["raw"]
            files[f"{name}/compare-{texture}.jpg"] = sheet
            summary["textures"].append(entry)
            print(f"[rebake] {name} texture {texture}: {json.dumps(entry, default=str)}")

        prod.share_caches()
        summary["verdict"] = verdict(summary)
        summary["peak_gpu_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        summary["load_seconds"] = self.load_seconds
        summary["container_seconds"] = round(time.time() - STARTED, 1)
        summary["gpu"] = self.gpu
        return {"summary": summary, "files": files}

    def _export(self, mesh, preset, rebake: bool) -> dict:
        """One export of ``mesh`` with the cache on or off, packed as production packs it, with its seconds."""
        from forge3d_worker.compress import pack_glb

        runtime = self.runtime
        runtime.rebake_textures = rebake
        clock = time.time()
        try:
            raw, triangles = runtime.export(mesh, preset)
        finally:
            runtime.rebake_textures = True
        export_s = round(time.time() - clock, 2)
        clock = time.time()
        packed = pack_glb(raw, preset.texture_size)
        return {
            "raw": raw,
            "packed": packed,
            "export_s": export_s,
            "step": dict(runtime.last_export or {}),  # to_glb or the rebake, and its seconds
            "pack_s": round(time.time() - clock, 2),
            "triangles": triangles,
            "raw_bytes": len(raw),
            "packed_bytes": len(packed),
            "projection": (runtime.last_projection or {}).get("reason"),
        }


def compare_on_surface(full: bytes, rebaked: bytes) -> tuple[dict, bytes]:
    """
    Both GLBs' base colour from MV-Adapter's six cameras, in the frame of the full way's mesh: how much they
    differ over the pixels either covers (0 to 255), and a sheet (the full way, the rebake, the difference x4).
    """
    import numpy as np
    import trimesh
    from PIL import Image

    from forge3d_worker import mvtexture

    meshes = [trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh") for raw in (full, rebaked)]
    frame = mvtexture.frame_for(meshes[0], 0.0)
    views = [[np.asarray(view, dtype=np.int16) for view in mvtexture.render_views(mesh, frame)] for mesh in meshes]
    background = int(mvtexture.BACKGROUND * 255 + 0.5)  # render_views' grey where the mesh isn't
    means, visible, covered_total, diff_total = [], 0, 0, 0.0
    rows: list[list[np.ndarray]] = [[], [], []]
    for one, two in zip(*views):
        covered = (one != background).any(-1) | (two != background).any(-1)
        difference = np.abs(one - two)
        count = int(covered.sum())
        means.append(float(difference[covered].mean()) if count else 0.0)
        visible += int((difference.max(-1) > VISIBLE)[covered].sum())
        covered_total += count
        diff_total += float(difference[covered].sum())
        for row, image in zip(rows, (one, two, np.clip(difference * 4, 0, 255))):
            row.append(np.asarray(Image.fromarray(image.astype(np.uint8)).resize((SHEET_VIEW, SHEET_VIEW))))
    sheet = Image.fromarray(np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0))
    buffer = io.BytesIO()
    sheet.save(buffer, "JPEG", quality=90)
    summary = {
        "diff_mean": round(diff_total / max(1, covered_total * 3), 3),
        "diff_worst_view": round(max(means), 3),
        "visible_share": round(visible / max(1, covered_total), 5),
        "covered_pixels": covered_total,
    }
    return summary, buffer.getvalue()


def verdict(summary: dict) -> str:
    """One line: whether the rebake gave the full way's texture, and how much faster it was."""
    textures = summary["textures"]
    paths = [(t["full"]["step"].get("path"), t["rebake"]["step"].get("path")) for t in textures]
    if any(rebaked != "rebake" for _, rebaked in paths):
        reasons = [t["rebake"]["step"].get("fallback") or t["rebake"]["step"].get("capture_error") for t in textures]
        return f"NO REBAKE: {paths}; {reasons}"
    worst = max(t["compare"]["diff_mean"] for t in textures)
    label = "SAME" if worst <= SAME else "CLOSE" if worst <= CLOSE else "DIFFERENT"
    each = ", ".join(f"{t['texture_seed']}: {t['compare']['diff_mean']:.2f}" for t in textures)
    triangles = "triangles equal" if all(t["compare"]["same_triangles"] for t in textures) else "TRIANGLES DIFFER"
    # Speed from the later textures, the ones a textures job rebakes (all of them if there is only one)
    later = textures[1:] or textures
    full_s = sum(t["full"]["export_s"] for t in later) / len(later)
    rebake_s = sum(t["rebake"]["export_s"] for t in later) / len(later)
    to_glb_s = sum(t["full"]["step"].get("seconds", 0.0) for t in later) / len(later)
    step_s = sum(t["rebake"]["step"].get("seconds", 0.0) for t in later) / len(later)
    return (
        f"{label}: base colour differs by {worst:.2f} of 255 at most ({each}), {triangles}; "
        f"export {full_s:.1f} s in full, {rebake_s:.1f} s rebaked (to_glb {to_glb_s:.1f} s, rebake {step_s:.2f} s)"
    )


def runs_by_number(volume: modal.Volume, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2: the picked pictures)."""
    pattern = re.compile(rf"{prefix}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


@app.local_entrypoint()
def check(only: str = "04,12", textures: int = 2, out: str = "ops-out/private/rebake") -> None:
    print(f"[rebake] app {prod.APP_NAME}: {textures} textures per object, each exported in full and rebaked")
    if textures < 2:
        raise SystemExit("[rebake] --textures must be at least 2: texture 2 is the one rebaked on another's layout")
    pictures = runs_by_number(prod.outputs, "phase2")
    numbers = [n.strip().zfill(2) for n in only.split(",") if n.strip()]
    jobs = [{"source": pictures[n], "textures": textures} for n in numbers if n in pictures]
    missing = [n for n in numbers if n not in pictures]
    print(f"[rebake] {len(jobs)} objects: {', '.join(job['source'] for job in jobs)}")
    if missing:
        print(f"[rebake] no Phase 2 picture for {', '.join(missing)}")
    root = pathlib.Path(out)
    summaries, failed = [], 0
    for job, made in zip(jobs, Rebake().compare.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[rebake] {job['source']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"source": job["source"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for name, data in made["files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
        print(f"[rebake] {job['source']}: {made['summary']['verdict']}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[rebake] done: {len(jobs) - failed} of {len(jobs)} objects")
    if not jobs or failed:
        raise SystemExit(1)
