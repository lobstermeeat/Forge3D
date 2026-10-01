"""
Phase 6, TRELLIS-multiview line: does giving TRELLIS.2 the object's other sides fix its invented backs?

    modal run ops/exp_tmv.py --plan synthetic --out ops-out/private
    modal run ops/exp_tmv.py --plan real --out ops-out/private

An ephemeral app (orainge-exp-tmv) with one L40S class built like workers/modal_app.py's Trellis2, running
the branch's forge3d_worker through handle_job, as production would. Each object's runs happen in one
call, in order, on its Phase 2 picture and seed (the phase2d seeds): first the single-picture final
(today's production), then the multi-view variants.

- "synthetic" views are rendered from that single-picture final itself, orthographic and level, at
  azimuths relative to the camera its projection found for the picture, so they agree with it exactly:
  given them, a multi-view run should keep that shape (not fatten or thin it).
- "real" views are the MV line's MV-Adapter drawings, /outputs/phase6/views/<phase2 run>/view-<az>.png.

Every run is compared with the object's single-picture final: bounding-box extents, surface area, volume,
silhouette IoU and area ratio from eight level views and the top (in TRELLIS.2's canonical frame, no
alignment), and the Chamfer distance. Results go to the volume, /outputs/phase6/trellis-mv/<plan>/<run>/
(progress.json as make_model writes it, the packed final, the picture, the views, metrics.json), and to
<out>/<plan>/ here; numbers only (no pictures) to ops-out/tmv-<plan>.json.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import sys
import time
from typing import Any, Optional

import modal

HERE = pathlib.Path(__file__).resolve()
for candidate in (HERE.parents[1] / "workers", pathlib.Path("/root")):
    if (candidate / "modal_app.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from modal_app import CACHE, MODELS, OUTPUTS, WORKERS, cache, models, outputs, trellis2_image  # noqa: E402

app = modal.App("orainge-exp-tmv")
# The production image (the branch's forge3d_worker is added from this checkout), plus modal_app itself,
# which this file imports again in the container
image = trellis2_image.add_local_file(WORKERS / "modal_app.py", "/root/modal_app.py")

RESULTS = "phase6/trellis-mv"
VIEWS = "phase6/views"

# --- Plans --------------------------------------------------------------------------------------------

SINGLE = {"name": "single", "views": "none"}


def _mv(name: str, views: str, azimuths: list, mode: str, weight: float, **extra: Any) -> dict:
    return {"name": name, "views": views, "azimuths": azimuths, "view_mode": mode, "picture_weight": weight, **extra}


SYNTHETIC_RUNS = [
    SINGLE,
    _mv("syn3-stoch-w2", "synthetic", [90, 180, 270], "stochastic", 2.0),
    _mv("syn3-multi-w2", "synthetic", [90, 180, 270], "multidiffusion", 2.0),
]

PLANS: dict[str, list[dict]] = {
    # Consistent views first: does the mechanism keep a shape it is shown from every side?
    "synthetic": [
        {"number": "06", "runs": SYNTHETIC_RUNS + [{**SYNTHETIC_RUNS[1], "name": "syn3-stoch-w2-preview", "mode": "preview"}]},
        {"number": "04", "runs": SYNTHETIC_RUNS},
        {"number": "08", "runs": SYNTHETIC_RUNS},
        {"number": "13", "runs": SYNTHETIC_RUNS},
    ],
}


# --- Synthetic views and shape metrics (GPU if there is one) -------------------------------------------


def load_mesh(glb: bytes):
    import trimesh

    return trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)


def render_views(glb: bytes, azimuth: float, relative: list, size: int = 768) -> list[bytes]:
    """
    RGBA PNG cutouts of a textured GLB, orthographic and level, at azimuth + each of ``relative``
    (degrees, projection.py's convention: 0 puts the camera on +Z), the object filling 92 % of the frame,
    lit softly from the camera's upper left.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from forge3d_worker import projection as P

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = P._load_model(load_mesh(glb), device)
    texture = model.texture.permute(2, 0, 1)[None]
    big = size * 2
    pngs = []
    for rel in relative:
        params = torch.tensor([[azimuth + rel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        x, y, depth, s = P.project(model.verts, params)
        half = big / 2 * 0.92
        xy = torch.stack([big / 2 + x[0] * half, big / 2 - y[0] * half], -1)
        zbuf = P.rasterize_depth(xy, depth[0], s[0], model.faces, big, big)
        face, bary = P.rasterize_faces(xy, depth[0], s[0], model.faces, zbuf)
        hit = face >= 0
        corners = model.faces[face.clamp(min=0)]  # (H, W, 3)
        uv = (bary[..., None] * model.uv[corners]).sum(-2)
        normal = (bary[..., None] * model.normals[corners]).sum(-2)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        colour = P._sample_texture(texture, uv.view(-1, 2)).view(big, big, 3)
        right, up, back = P.view_axes(params)
        light = back[0] + 0.5 * up[0] - 0.3 * right[0]
        shade = 0.62 + 0.38 * (normal @ (light / light.norm())).clamp(min=0)
        alpha = hit.float()[..., None]
        rgba = torch.cat([colour * shade[..., None] * alpha, alpha], -1).permute(2, 0, 1)[None]
        rgba = F.avg_pool2d(rgba, 2)[0].permute(1, 2, 0)  # 2x supersampled, premultiplied
        a = rgba[..., 3:]
        rgb = rgba[..., :3] / a.clamp_min(1e-6)
        pixels = (torch.cat([rgb, a], -1).clamp(0, 1) * 255).round().byte().cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(pixels), "RGBA").save(buffer, "PNG")
        pngs.append(buffer.getvalue())
    return pngs


def _silhouette(verts, faces, azimuth: float, elevation: float, size: int):
    """A mesh's silhouette (size, size) from an orthographic view, in TRELLIS.2's canonical frame unscaled."""
    import torch

    from forge3d_worker import projection as P

    params = torch.tensor([[azimuth, elevation, 0.0, 0.0, 0.0, 0.0, 0.0]], device=verts.device)
    x, y, depth, s = P.project(verts * 1.3, params)  # the [-0.5, 0.5] box's corners stay inside the frame
    half = size / 2
    xy = torch.stack([half + x[0] * half, half - y[0] * half], -1)
    return torch.isfinite(P.rasterize_depth(xy, depth[0], s[0], faces, size, size))


SILHOUETTE_VIEWS = [(az, 0.0) for az in range(0, 360, 45)] + [(0.0, 89.0)]


def shape_metrics(glb: bytes, reference: Optional[bytes]) -> dict:
    """The shape's size, and how it differs from ``reference`` (the single-picture final), if given."""
    import numpy as np
    import torch

    mesh = load_mesh(glb)
    lo, hi = mesh.bounds
    out: dict = {
        "extents": [round(float(v), 4) for v in hi - lo],
        "area": round(float(mesh.area), 4),
        "watertight": bool(mesh.is_watertight),
        "faces": int(len(mesh.faces)),
    }
    if mesh.is_watertight:
        out["volume"] = round(float(mesh.volume), 5)
    if reference is None:
        return out
    ref = load_mesh(reference)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def tensors(m):
        return (
            torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(m.faces), dtype=torch.long, device=device),
        )

    (va, fa), (vb, fb) = tensors(mesh), tensors(ref)
    ious, ratios = [], []
    for azimuth, elevation in SILHOUETTE_VIEWS:
        a, b = _silhouette(va, fa, azimuth, elevation, 384), _silhouette(vb, fb, azimuth, elevation, 384)
        ious.append(float((a & b).sum() / (a | b).sum().clamp_min(1)))
        ratios.append(float(a.sum() / b.sum().clamp_min(1)))
    ref_lo, ref_hi = ref.bounds
    out["vs_single"] = {
        "extent_ratio": [round(float(v), 3) for v in (hi - lo) / np.maximum(ref_hi - ref_lo, 1e-6)],
        "area_ratio": round(float(mesh.area / max(ref.area, 1e-9)), 3),
        "silhouette_iou": [round(v, 3) for v in ious],
        "silhouette_area_ratio": [round(v, 3) for v in ratios],
        "mean_iou": round(float(np.mean(ious)), 3),
        "mean_area_ratio": round(float(np.mean(ratios)), 3),
    }
    if mesh.is_watertight and ref.is_watertight:
        out["vs_single"]["volume_ratio"] = round(float(mesh.volume / max(ref.volume, 1e-9)), 3)
    # Chamfer distance, in units of the canonical box (1 = its side)
    def points(m):
        try:
            samples = m.sample(30000, seed=0)
        except TypeError:  # an older trimesh
            samples = m.sample(30000)
        return torch.tensor(np.asarray(samples), dtype=torch.float32, device=device)

    pa, pb = points(mesh), points(ref)

    def nearest(p, q):
        return torch.cat([torch.cdist(p[i : i + 4096], q).min(dim=1).values for i in range(0, len(p), 4096)])

    ab, ba = nearest(pa, pb), nearest(pb, pa)
    out["vs_single"]["chamfer_mean"] = round(float((ab.mean() + ba.mean()) / 2), 5)
    out["vs_single"]["chamfer_p90"] = round(float(torch.quantile(torch.cat([ab, ba]), 0.9)), 5)
    return out


# --- The GPU side -------------------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=24576,
    volumes={MODELS: models, CACHE: cache, OUTPUTS: outputs},
    timeout=3600,
    startup_timeout=900,
    scaledown_window=20,
    max_containers=2,
)
class Lab:
    @modal.enter()
    def load(self) -> None:
        started = time.monotonic()
        from forge3d_worker.pipeline import Trellis2Runtime
        from modal_app import LOCAL_TUNING, SHARED_TUNING, merge_tuning

        merge_tuning(LOCAL_TUNING, SHARED_TUNING)
        self.runtime = Trellis2Runtime(f"{MODELS}/TRELLIS.2-4B")
        stock = self.runtime.export

        def export(mesh: Any, preset: Any):
            raw, triangles = stock(mesh, preset)
            self.raw = raw  # the GLB before packing, which trimesh can read
            return raw, triangles

        self.runtime.export = export
        self.load_s = round(time.monotonic() - started, 1)
        self.first = True

    @modal.method()
    def object(self, plan: str, spec: dict) -> dict:
        import torch
        from forge3d_worker.compress import pack_glb
        from forge3d_worker.service import handle_job
        from forge3d_worker.settings import MultiView
        from forge3d_worker.storage import InlineStorage
        from modal_app import share_caches

        started = time.monotonic()
        root = pathlib.Path(OUTPUTS)
        source = next(p for p in sorted(root.iterdir()) if p.name.startswith(f"phase2-{spec['number']}-"))
        state = json.loads((source / "progress.json").read_text())
        picture = (source / state["input"]).read_bytes()
        seed = int(spec.get("seed", state["seed"]))
        label = source.name.split("-", 1)[1]  # 04-retro-arcade-machine
        report: dict = {"source": source.name, "seed": seed, "label": label, "runs": {}, "files": {}}
        report["load_s"] = self.load_s if self.first else 0.0
        self.first = False
        reference: Optional[dict] = None  # the single-picture final: raw GLB and the picture's camera
        for run in spec["runs"]:
            name = f"{label}-{run['name']}"
            folder = root / RESULTS / plan / name
            folder.mkdir(parents=True, exist_ok=True)
            entry: dict = {"run": run}
            try:
                views, view_files = self._views(run, source, reference)
            except Exception as err:  # noqa: BLE001 - one bad run doesn't stop the others
                report["runs"][name] = {**entry, "error": f"views: {type(err).__name__}: {err}"}
                continue
            job: dict = {"image_base64": base64.b64encode(picture).decode(), "seed": seed, "request_id": name[:64]}
            job["mode"] = run.get("mode", "final")
            if views:
                job["views"] = views
            self.runtime.multiview = MultiView(
                mode=run.get("view_mode", "multidiffusion"), picture_weight=float(run.get("picture_weight", 2.0))
            )
            torch.cuda.reset_peak_memory_stats()
            clock = time.monotonic()
            self.raw = None
            result = handle_job({"id": name, "input": job}, self.runtime, InlineStorage(), pack_glb)
            entry["wall_s"] = round(time.monotonic() - clock, 1)
            entry["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            if result.get("error"):
                report["runs"][name] = {**entry, "error": result["error"]}
                print(f"[tmv] {name}: {result['error']}")
                continue
            packed = base64.b64decode(result["glb"]["base64"])
            entry.update({k: result.get(k) for k in ("timings", "pipeline", "views_used", "triangles", "projection")})
            entry["gpu_s"] = round(sum(result["timings"].values()), 1)
            if run["name"] == "single" and job["mode"] == "final":
                pose = (result.get("projection") or {}).get("pose") or {"azimuth": 0.0}
                reference = {"raw": self.raw, "azimuth": float(pose["azimuth"])}
            try:
                compare = None if run["name"] == "single" or reference is None else reference["raw"]
                entry["metrics"] = shape_metrics(self.raw, compare)
            except Exception as err:  # noqa: BLE001
                entry["metrics"] = {"error": f"{type(err).__name__}: {err}"}
            # The run folder as make_model leaves one, for grade_renders.py
            files = {f"{job['mode']}-{seed}.glb": packed, f"input-{label}.png": picture, **view_files}
            for file, data in files.items():
                (folder / file).write_bytes(data)
            step = {"status": "done", "files": [f"{job['mode']}-{seed}.glb"], "timings": entry.get("timings")}
            progress = {"seed": seed, "input": f"input-{label}.png", "steps": {"final": step}, "exp": entry}
            (folder / "progress.json").write_text(json.dumps(progress, indent=2))
            outputs.commit()
            report["runs"][name] = entry
            report["files"][name] = files
            print(f"[tmv] {name}: {entry['gpu_s']} s on the GPU, views {entry.get('views_used')}, "
                  f"{json.dumps(entry['metrics'].get('vs_single', entry['metrics']))[:400]}")  # fmt: skip
        share_caches()
        report["seconds"] = round(time.monotonic() - started, 1)
        return report

    def _views(self, run: dict, source: pathlib.Path, reference: Optional[dict]) -> tuple[list, dict]:
        """The job's views, and their PNGs by file name (view-<az>.png)."""
        kind = run.get("views", "none")
        if kind == "none":
            return [], {}
        azimuths = run["azimuths"]
        weights = run.get("view_weights", {})
        if kind == "synthetic":
            if reference is None:
                raise RuntimeError("synthetic views need the single-picture final first")
            pngs = render_views(reference["raw"], reference["azimuth"], azimuths)
        elif kind == "real":
            folder = pathlib.Path(OUTPUTS) / VIEWS / source.name
            pngs = [(folder / f"view-{az}.png").read_bytes() for az in azimuths]
        else:
            raise ValueError(f"unknown views {kind!r}")
        views, files = [], {}
        for az, png in zip(azimuths, pngs):
            view = {"image_base64": base64.b64encode(png).decode(), "azimuth": az, "elevation": 0}
            if str(az) in weights:
                view["weight"] = float(weights[str(az)])
            views.append(view)
            files[f"view-{az}.png"] = png
        return views, files


# --- Here ---------------------------------------------------------------------------------------------


@app.local_entrypoint()
def main(plan: str = "synthetic", out: str = "ops-out/private", only: str = "") -> None:
    specs = PLANS[plan]
    if only:
        wanted = {n.strip().zfill(2) for n in only.split(",")}
        specs = [s for s in specs if s["number"] in wanted]
    print(f"[tmv] {plan}: {len(specs)} objects, {sum(len(s['runs']) for s in specs)} runs")
    started = time.monotonic()
    reports = list(Lab().object.starmap([(plan, spec) for spec in specs], return_exceptions=True))
    summary: dict = {"plan": plan, "objects": {}, "wall_s": None}
    gpu_s = 0.0
    for spec, report in zip(specs, reports):
        if isinstance(report, BaseException):
            reason = next((line for line in str(report).splitlines() if line.strip()), "")
            print(f"[tmv] {spec['number']}: failed ({type(report).__name__}: {reason})")
            summary["objects"][spec["number"]] = {"error": f"{type(report).__name__}: {reason}"}
            continue
        for name, files in report.pop("files").items():
            folder = pathlib.Path(out) / plan / name
            folder.mkdir(parents=True, exist_ok=True)
            for file, data in files.items():
                (folder / file).write_bytes(data)
            entry = report["runs"][name]
            glbs = [n for n in files if n.endswith(".glb")]
            step = {"status": "done", "files": glbs, "timings": entry.get("timings")}
            progress = {"seed": report["seed"], "input": f"input-{report['label']}.png", "steps": {"final": step}}
            (folder / "progress.json").write_text(json.dumps({**progress, "exp": entry}, indent=2))
        gpu_s += report["load_s"] + report["seconds"]
        summary["objects"][spec["number"]] = report
    summary["wall_s"] = round(time.monotonic() - started, 1)
    summary["gpu_minutes"] = round(gpu_s / 60, 1)
    pathlib.Path("ops-out").mkdir(exist_ok=True)
    pathlib.Path(f"ops-out/tmv-{plan}.json").write_text(json.dumps(summary, indent=2))
    print(f"[tmv] done in {summary['wall_s']} s; about {summary['gpu_minutes']} GPU minutes (calls plus loading)")
