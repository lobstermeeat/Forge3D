"""
Phase 7's judge experiment, in a staging app (ORAINGE_APP_NAME; nothing is deployed). For each picked
picture of a Phase 2 run prefix (--prefix, "phase2e" by default: the next seeds), TRELLIS.2's final and
--count more textures on its shape, each exported as production does, the way the rolls experiment
(exp_texture.py's `rolls`) made them: `t9roll0` is production's final, `t9roll<k>` the texture drawn
again from torch.manual_seed(seed + 1000 k). Each candidate is rendered as the judge sees it
(judgeviews.turntable: six views in a 3 x 2 grid), and the self-hosted judges, Qwen3-VL 8B and 30B-A3B
(Judge8B and Judge30B in modal_app.py), are asked about that object at once, without waiting, for a
verdict per candidate and the one a creator would rather use. Each judge sees the candidates in --orders
orders: shuffled by the object's number, then that order reversed (so a bias towards a position shows).

    ORAINGE_APP_NAME=orainge-p7-judge modal run ops/exp_judge.py::judges --count 3
    ORAINGE_APP_NAME=orainge-p7-judge modal run ops/exp_judge.py::judges --only 04,06 --sizes 8b

Results land in the orainge-outputs volume under phase7/judge/ and in --out: a folder per candidate
(t9roll<k>-<subject>: the packed GLB, the picture and progress.json, as the rolls), the grids the judges
saw (judge/<subject>/t9roll<k>.jpg) and summary.json, with every object's candidates and each judge's
answers by candidate name (verdicts, pick, why, the raw reply).

The phase2e runs started from their pictures, so their progress has no prompt: the prompt the user
typed comes from the test set (--prompts, workers/test-sets/phase2.txt), by number.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import random
import re
import sys
import time
from typing import Any, Callable

import modal

if modal.is_local() and os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
# Here: the checkout's workers/. In the container: /root, where the image puts modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code under the staging app name

app = prod.app
RESULTS = "phase7/judge"
LOG = "[judge]"
# Every Phase 2 prompt, the slowest (the books) first so the containers finish together
ROLL_OBJECTS = ["11", *(f"{n:02d}" for n in range(1, 21) if n != 11)]
ROLL_SEED = 1000  # roll k's texture noise: torch.manual_seed(seed + ROLL_SEED * k)
SIZES = tuple(prod.JUDGE_GPUS)  # "8b", "30b"
PROMPTS = CHECKOUT / "test-sets" / "phase2.txt"
GRID_QUALITY = 92  # the saved grids; the judges get PNGs

rolls_image = prod.trellis2_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")

STARTED = time.time()


# --- Plain functions (tests/test_exp_judge.py runs them with stand-ins) ------------------------------


def runs_by_number(volume: Any, prefix: str) -> dict[str, str]:
    """Prompt number -> the run named <prefix>-NN-<slug> (phase2e: the picked pictures at the next seeds)."""
    pattern = re.compile(rf"{re.escape(prefix)}-(\d\d)-[a-z0-9-]+")
    names = sorted(entry.path.strip("/") for entry in volume.listdir("/"))
    return {match[1]: name for name in names if (match := pattern.fullmatch(name))}


def prompts_by_number(path: pathlib.Path) -> dict[str, str]:
    """Prompt number -> what the user typed, from a test set."""
    return {f"{number:02d}": run["prompt"] for number, run in enumerate(prod.read_set(path), 1) if "prompt" in run}


def orders_for(number: str, count: int, how_many: int) -> list[list[int]]:
    """
    The orders the judges see an object's candidates in (lists of their indices, shown as K, L, M, …):
    shuffled by the object's number, then that order reversed, so every candidate is also seen at the
    mirrored position; beyond two, more seeded shuffles.
    """
    first = list(range(count))
    random.Random(int(number)).shuffle(first)
    orders = [first, first[::-1]]
    while len(orders) < how_many:
        more = list(range(count))
        random.Random(int(number) * 100 + len(orders)).shuffle(more)
        orders.append(more)
    return orders[: max(1, how_many)]


def _encoded(image: Any, kind: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=kind, **options)
    return buffer.getvalue()


def make_object(
    job: dict,
    runtime: Any,
    root: pathlib.Path,
    *,
    pack: Callable[[bytes, int], bytes],
    render: Callable[[bytes], Any],
    spawn: Callable[[str, dict], str],
) -> dict:
    """
    One object: production's final and job["count"] more textures of its shape, each exported, packed and
    saved as the rolls are (under root/RESULTS); each GLB as exported (before gltfpack) drawn as a grid by
    ``render``; then every judge size asked about it in each order through ``spawn`` (which returns the
    call's id, so nothing waits). Returns {"summary", "files"}: the files are also written under root.
    """
    from PIL import Image

    from forge3d_worker.settings import PRESETS

    source = job["source"]
    state = json.loads((root / source / "progress.json").read_text())
    picture_bytes = (root / source / state["input"]).read_bytes()
    picture = Image.open(io.BytesIO(picture_bytes))
    picture.load()
    seed = int(state["seed"])
    prompt = job.get("prompt") or state.get("prompt") or ""
    subject = source.split("-", 1)[1]  # phase2e-04-retro-arcade-machine -> 04-retro-arcade-machine
    preset = PRESETS["final"]
    files: dict[str, bytes] = {}
    raws: dict[str, bytes] = {}  # each candidate's GLB as exported, before gltfpack
    summary: dict = {"source": source, "number": job["number"], "seed": seed, "prompt": prompt, "variants": {}}

    def save(name: str, data: bytes) -> None:
        target = root / RESULTS / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = data

    def export(variant: str, made: Any, **notes: Any) -> None:
        clock = time.time()
        raw, triangles = runtime.export(made, preset)
        raws[variant] = raw
        packed = pack(raw, preset.texture_size)
        run = f"{variant}-{subject}"
        name = f"final-{seed}.glb"
        step = {
            "status": "done",
            "files": [name],
            "triangles": triangles,
            "bytes": len(packed),
            "export_s": round(time.time() - clock, 1),
            "variant": variant,
            **notes,
        }
        if getattr(runtime, "last_projection", None):
            step["projection"] = runtime.last_projection
        progress = {
            "prompt": prompt or None,
            "seed": seed,
            "input": state["input"],
            "final": True,
            "steps": {"final": step},
            "made_by": {"experiment": "phase7 judge", "variant": variant, "picture_from": source},
            "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        save(f"{run}/{name}", packed)
        save(f"{run}/{state['input']}", picture_bytes)
        save(f"{run}/progress.json", json.dumps(progress, indent=2, default=str).encode())
        summary["variants"][variant] = {k: v for k, v in step.items() if k not in ("files", "status")}
        print(f"{LOG} {run}: {json.dumps(summary['variants'][variant], default=str)}")

    # 1. Production's final, then the same shape with new texture noise
    clock = time.time()
    mesh = runtime.generate(picture, preset, seed)
    summary["generate_s"] = round(time.time() - clock, 1)
    summary["pipeline"] = getattr(runtime, "pipeline_used", None)
    export("t9roll0", mesh)
    for k in range(1, int(job.get("count", 3)) + 1):
        clock = time.time()
        made = runtime.retexture(seed=seed + ROLL_SEED * k)
        export(f"t9roll{k}", made, retexture_s=round(time.time() - clock, 1), retexture=runtime.last_retexture)

    # 2. Each candidate as the judges see it
    variants = list(raws)
    summary["candidates"] = variants
    grids: dict[str, bytes] = {}
    clock = time.time()
    for variant in variants:
        try:
            grid = render(raws[variant])
        except Exception as err:  # noqa: BLE001 - the GLBs are made; the object goes unjudged
            summary.setdefault("render_errors", {})[variant] = f"{type(err).__name__}: {err}"
            print(f"{LOG} {variant}-{subject}: not rendered: {type(err).__name__}: {err}")
            continue
        save(f"judge/{subject}/{variant}.jpg", _encoded(grid.convert("RGB"), "JPEG", quality=GRID_QUALITY))
        grids[variant] = _encoded(grid.convert("RGB"), "PNG")
    summary["render_s"] = round(time.time() - clock, 1)
    if len(grids) != len(variants):
        summary["judge_error"] = "not every candidate could be rendered, so the judges weren't asked"
        return {"summary": summary, "files": files}

    # 3. The judges, every size and order at once; the entrypoint collects their answers
    request = {
        "picture_png": base64.b64encode(picture_bytes).decode("ascii"),
        "candidates_png": [base64.b64encode(grids[variant]).decode("ascii") for variant in variants],
        "prompt": prompt,
    }
    summary["judge_calls"] = []
    for order in orders_for(job["number"], len(variants), int(job.get("orders", 2))):
        for size in job.get("sizes", SIZES):
            call = spawn(size, {**request, "order": order})
            summary["judge_calls"].append({"size": size, "order": order, "call": call})
    return {"summary": summary, "files": files}


def judged(result: dict, variants: list, order: list) -> dict:
    """A judge's result with candidates named (t9roll<k>) instead of numbered."""
    shown = [variants[number] for number in order]  # as K, L, M, … in this order
    if result.get("error"):
        return {"order": shown, "error": result["error"]}
    record = {
        "order": shown,
        "best": variants[result["best"]],
        "verdicts": dict(zip(variants, result["verdicts"])),
        "why": result.get("why"),
        "problems": dict(zip(variants, result.get("problems") or [None] * len(variants))),
    }
    for key in ("parse_error", "notes", "seconds", "tokens", "model", "raw"):
        if result.get(key) is not None:
            record[key] = result[key]
    return record


def collect(summary: dict, get: Callable[[str], dict]) -> None:
    """Every judge call's answer into the summary ("judges": size -> one record per order) and the picks."""
    variants = summary.get("candidates", [])
    judges: dict = summary.setdefault("judges", {})
    for call in summary.get("judge_calls", []):
        try:
            record = judged(get(call["call"]), variants, call["order"])
        except Exception as err:  # noqa: BLE001 - one judge call failing leaves the rest
            record = {"order": [variants[number] for number in call["order"]], "error": f"{type(err).__name__}: {err}"}
        judges.setdefault(call["size"], []).append(record)
    summary["picks"] = {size: [record.get("best") for record in records] for size, records in judges.items()}


# --- Modal ----------------------------------------------------------------------------------------------


@app.cls(
    image=rolls_image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    volumes={prod.MODELS: prod.models, prod.OUTPUTS: prod.outputs, prod.CACHE: prod.cache},
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=4,
)
class JudgeRolls:
    @modal.enter()
    def load(self) -> None:
        import torch

        prod.merge_tuning(prod.LOCAL_TUNING, prod.SHARED_TUNING)
        from forge3d_worker.pipeline import Trellis2Runtime

        started = time.time()
        self.runtime = Trellis2Runtime(f"{prod.MODELS}/TRELLIS.2-4B")
        self.load_seconds = round(time.time() - started, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"{LOG} TRELLIS.2 on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def make(self, job: dict) -> dict:
        import torch
        import trimesh

        from forge3d_worker import judgeviews
        from forge3d_worker.compress import pack_glb

        prod.outputs.reload()
        torch.cuda.reset_peak_memory_stats()

        def render(raw: bytes) -> Any:
            return judgeviews.turntable(trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh"))

        def spawn(size: str, request: dict) -> str:
            return prod.judge_class(size)().judge.spawn(request).object_id

        made = make_object(job, self.runtime, pathlib.Path(prod.OUTPUTS), pack=pack_glb, render=render, spawn=spawn)
        prod.outputs.commit()
        prod.share_caches()
        made["summary"].update(
            peak_gpu_gb=round(torch.cuda.max_memory_reserved() / 2**30, 1),
            load_seconds=self.load_seconds,
            container_seconds=round(time.time() - STARTED, 1),
            gpu=self.gpu,
        )
        return made


@app.local_entrypoint()
def judges(
    prefix: str = "phase2e",
    only: str = "",
    out: str = "ops-out/private/judge",
    count: int = 3,
    sizes: str = ",".join(SIZES),
    orders: int = 2,
    prompts: str = str(PROMPTS),
) -> None:
    chosen = [size.strip() for size in sizes.split(",") if size.strip()]
    for size in chosen:
        prod.judge_class(size)  # a wrong size stops here, before any GPU runs
    print(f"{LOG} app {prod.APP_NAME}: {prefix}'s finals and {count} more textures each, judged by {', '.join(chosen)}")
    # The judge's weights (both sizes, checked against their sha256); quick when already there
    prod.download_models.remote(which="judge")
    pictures = runs_by_number(prod.outputs, prefix)
    typed = prompts_by_number(pathlib.Path(prompts))
    numbers = [n.strip().zfill(2) for n in only.split(",") if n.strip()] if only else ROLL_OBJECTS
    missing = [number for number in numbers if number not in pictures]
    if missing:
        print(f"{LOG} no {prefix} run for {', '.join(missing)}")
    jobs = [
        {"number": n, "source": pictures[n], "prompt": typed.get(n, ""), "count": count, "sizes": chosen, "orders": orders}
        for n in numbers
        if n in pictures
    ]
    print(f"{LOG} {len(jobs)} objects: {', '.join(job['source'] for job in jobs)}")
    root = pathlib.Path(out)
    summaries, failed = [], 0
    for job, made in zip(jobs, JudgeRolls().make.map(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"{LOG} {job['source']}: failed: {type(made).__name__}: {reason}")
            summaries.append({"source": job["source"], "number": job["number"], "error": f"{type(made).__name__}: {reason}"})
            continue
        for name, data in made["files"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        summaries.append(made["summary"])
        brief = {k: v for k, v in made["summary"].items() if k not in ("variants", "judge_calls")}
        print(f"{LOG} {job['source']}: {json.dumps(brief, default=str)}")
    # The judges' answers, asked for by each container without waiting
    for summary in summaries:
        collect(summary, lambda call_id: modal.FunctionCall.from_id(call_id).get(timeout=1800))
        if summary.get("picks"):
            print(f"{LOG} {summary['source']}: picks {json.dumps(summary['picks'])}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"{LOG} done: {len(jobs) - failed} of {len(jobs)} objects")
    if jobs and failed == len(jobs):
        raise SystemExit(f"{LOG} nothing was made")
