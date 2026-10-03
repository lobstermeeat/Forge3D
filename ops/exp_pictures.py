"""
Which way FLUX.1 [schnell]'s reference pictures face, on a real GPU, in a staging app (ORAINGE_APP_NAME;
nothing is deployed). Each template variant below draws four pictures per prompt at fixed seeds (the
prompt's seed and the next three, as a reference job does) with production's generator and scoring
(workers/flux-schnell/reference_worker/service.py: make_flux_generator, score_picture). Every picture is
saved at the scoring size (512 px, as score_picture shrinks it) as lossless WebP, so views can be labelled
by eye and checks run offline on the very pixels the worker scores, with its score, issues and seconds in
summary.json.

    ORAINGE_APP_NAME=orainge-p8-pics modal run ops/exp_pictures.py::check --plan "a=v0,v1,v2;b=v0"

Prompts: workers/test-sets/phase2.txt (20) and products.txt (8), numbered 1-28 in that order. Seed set s
gives prompt k (1-28) the seeds SEED_SETS[s] + 10 k + 0..3.

ops/exp_klein.py draws the same pictures with FLUX.2 [klein] 4B, for comparison.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
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

PROMPT_SETS = ("test-sets/phase2.txt", "test-sets/products.txt")
SEED_SETS = {"a": 810_000, "b": 820_000, "c": 830_000}
COUNT = 4
SCORE_SIZE = 512  # reference_worker.service.SCORE_SIZE

# The wrappers compared. v0 is production's template as Phase 8 found it
TEMPLATES = {
    "v0": (
        "{prompt}. A single object, centered and fully in frame, on a plain light gray background, "
        "soft even studio lighting, three-quarter view from slightly above, sharp focus, "
        "no text, no other objects"
    ),
    # The view first, as concrete numbers
    "v1": (
        "Three-quarter front view, the camera 35 degrees to the left and 20 degrees above: {prompt}. "
        "A single object, centered and fully in frame, on a plain light gray background, soft even studio "
        "lighting, sharp focus, no text, no other objects"
    ),
    # The view first, as what the picture shows
    "v2": (
        "A three-quarter view of {prompt}, seen from the front left corner and slightly above, so its front, "
        "its left side and its top all show. A single object, fully in frame, on a plain light gray "
        "background, soft even studio lighting, sharp focus, no text, no other objects"
    ),
    # Short
    "v3": (
        "{prompt}, three-quarter view from the front left, slightly above, on a plain light gray background, "
        "studio lighting"
    ),
    # A realistic product photo, taken at an angle
    "v4": (
        "A realistic studio product photo of {prompt}, taken at an angle from the front left corner and "
        "slightly above, showing its front and one side. One whole object on a plain light gray background, "
        "soft even lighting, sharp focus, no text"
    ),
    # Production's order, with stronger view words and without "centered"
    "v5": (
        "{prompt}. Front three-quarter view, turned 45 degrees to the camera, seen from slightly above. "
        "A single object, fully in frame, on a plain light gray background, soft even studio lighting, "
        "sharp focus, no text, no other objects"
    ),
    # No "three-quarter": an angled shot
    "v6": (
        "An angled perspective shot of {prompt} from the front left, slightly above, showing the front and "
        "the side. A single object, fully in frame, on a plain light gray background, soft even studio "
        "lighting, sharp focus, no text, no other objects"
    ),
    # Round 2, after v6 (most three-quarter views) and v4 (kept styles such as "low poly"; cartoons in 3D).
    # v4's opening with v6's angle
    "v7": (
        "A realistic studio product photo of {prompt}: an angled perspective shot from the front left corner "
        "and slightly above, showing its front and its side. A single object, fully in frame, on a plain light "
        "gray background, soft even studio lighting, sharp focus, no text, no other objects"
    ),
    # v6 from the corner, with room around the object (v6 cut a few off at the edge)
    "v8": (
        "An angled perspective shot of {prompt} from the front left corner, slightly above, showing its front "
        "and its side. A single object, fully in frame with space around it, on a plain light gray background, "
        "soft even studio lighting, sharp focus, no text, no other objects"
    ),
    # v6, then v4's realism
    "v9": (
        "An angled perspective shot of {prompt} from the front left, slightly above, showing the front and the "
        "side. A realistic studio product photo of one whole object, fully in frame, on a plain light gray "
        "background, soft even lighting, sharp focus, no text, no other objects"
    ),
    # Round 3: v6's angle (front three-quarters, v10's "turned" drew backs) as a studio product photo (v4: kept
    # "low poly" and drew cartoons in 3D, where v7's "realistic" lost the low poly), with v4 and v10's "one whole
    # object" (fewer cut off at the edge)
    "v11": (
        "A studio product photo of {prompt}: an angled perspective shot from the front left corner, slightly "
        "above, showing its front and its side. One whole object, fully in frame, on a plain light gray "
        "background, soft even lighting, sharp focus, no text, no other objects"
    ),
    # The object turned rather than the camera moved
    "v10": (
        "A studio product photo of {prompt} at a three-quarter angle, turned so its front and its left side both "
        "face the camera, seen from slightly above. One whole object, fully in frame, on a plain light gray "
        "background, soft even lighting, sharp focus, no text, no other objects"
    ),
}

pictures_image = prod.flux_image.add_local_file(prod.WORKERS / "modal_app.py", "/root/modal_app.py")

STARTED = time.time()


@app.cls(
    image=pictures_image,
    gpu=prod.FLUX_GPU,
    cpu=4.0,
    memory=32768,  # the weights pass through RAM on their way to the GPU
    volumes={prod.MODELS: prod.models},  # read only: nothing is written to the shared volume
    timeout=3600,
    startup_timeout=1200,
    scaledown_window=30,
    max_containers=2,
)
class Pictures:
    @modal.enter()
    def load(self) -> None:
        import torch

        from reference_worker.service import make_flux_generator

        prod._require_weights("reference")
        clock = time.time()
        self.generate = make_flux_generator(f"{prod.MODELS}/FLUX.1-schnell", cpu_offload=False)
        self.load_seconds = round(time.time() - clock, 1)
        self.gpu = torch.cuda.get_device_name()
        print(f"[pictures] FLUX.1 [schnell] on {self.gpu} in {self.load_seconds} s")

    @modal.method()
    def draw(self, job: dict) -> dict:
        return draw_with(self.generate, job, self.gpu, self.load_seconds)


def draw_with(generate, job: dict, gpu: str, load_seconds: float) -> dict:
    """Four pictures of one prompt in one template, each with production's score and its seconds."""
    import torch
    from PIL import Image

    from reference_worker.service import score_picture

    full = job["template"].format(prompt=job["prompt"].strip().rstrip("."))
    torch.cuda.reset_peak_memory_stats()
    pictures = []
    for seed in job["seeds"]:
        clock = time.time()
        image = generate(full, seed)
        torch.cuda.synchronize()
        seconds = time.time() - clock
        clock = time.time()
        framing = score_picture(image)
        score_seconds = time.time() - clock
        small = image.convert("RGB")
        small.thumbnail((SCORE_SIZE, SCORE_SIZE), Image.Resampling.BOX)  # as score_picture shrinks it
        buffer = io.BytesIO()
        small.save(buffer, "WEBP", lossless=True, method=4)
        pictures.append(
            {
                "seed": seed,
                **framing,
                "seconds": round(seconds, 3),
                "score_seconds": round(score_seconds, 3),
                "webp": buffer.getvalue(),
            }
        )
    return {
        "prompt": full,
        "pictures": pictures,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu": gpu,
        "load_seconds": load_seconds,
    }


def prompts() -> list[str]:
    found = []
    for name in PROMPT_SETS:
        found.extend(run["prompt"] for run in prod.read_set(prod.WORKERS / name) if "prompt" in run)
    return found


@app.local_entrypoint()
def check(plan: str = "a=v0", only: str = "", out: str = "ops-out/private/pictures") -> None:
    """--plan "a=v0,v1;b=v0": variants v0 and v1 with seed set a, and v0 with seed set b."""
    run_plan(plan, only, out, Pictures(), "schnell")


def parse_plan(plan: str, model: str) -> list[tuple[str, str, str]]:
    pairs = []
    for part in filter(None, (p.strip() for p in plan.split(";"))):
        seed_set, _, listed = part.partition("=")
        for variant in filter(None, (v.strip() for v in listed.split(","))):
            if seed_set.strip() not in SEED_SETS or variant not in TEMPLATES:
                raise SystemExit(f"[pictures] --plan: no seed set {seed_set!r} or variant {variant!r}")
            pairs.append((model, seed_set.strip(), variant))
    return pairs


def run_plan(plan: str, only: str, out: str, drawer, model: str) -> None:
    """Draws every (seed set, variant) of `plan` for every prompt with `drawer` (Pictures() or Klein())."""
    pairs = parse_plan(plan, model)
    texts = prompts()
    numbers = [int(n) for n in only.split(",") if n.strip()] or list(range(1, len(texts) + 1))
    jobs = []
    for _, seed_set, variant in pairs:
        for number in numbers:
            base = SEED_SETS[seed_set] + 10 * number
            jobs.append(
                {
                    "model": model,
                    "set": seed_set,
                    "variant": variant,
                    "number": number,
                    "prompt": texts[number - 1],
                    "template": TEMPLATES[variant],
                    "seeds": list(range(base, base + COUNT)),
                }
            )
    print(f"[pictures] app {prod.APP_NAME}: {len(jobs)} prompts x {COUNT} pictures ({plan})")
    root = pathlib.Path(out)
    rows, failed = [], 0
    clock = time.time()
    for job, made in zip(jobs, drawer.draw.map(jobs, return_exceptions=True, order_outputs=True)):
        prefix = "" if model == "schnell" else f"{model}/"
        where = f"{prefix}{job['set']}/{job['variant']}/{job['number']:02d}-{prod._slug(job['prompt'])}"
        if isinstance(made, BaseException):
            failed += 1
            reason = next((line for line in str(made).splitlines() if line.strip()), "")
            print(f"[pictures] {where}: failed: {type(made).__name__}: {reason}")
            error = f"{type(made).__name__}: {reason}"
            rows.append({**{k: job[k] for k in ("model", "set", "variant", "number", "prompt")}, "error": error})
            continue
        folder = root / where
        folder.mkdir(parents=True, exist_ok=True)
        for picture in made["pictures"]:
            (folder / f"{picture['seed']}.webp").write_bytes(picture.pop("webp"))
        rows.append(
            {
                **{k: job[k] for k in ("model", "set", "variant", "number", "prompt")},
                "full_prompt": made["prompt"],
                "folder": where,
                "pictures": made["pictures"],
                **{k: made[k] for k in ("peak_allocated_gb", "peak_reserved_gb", "gpu", "load_seconds")},
            }
        )
        scores = " ".join(f"{p['score']:.2f}" for p in made["pictures"])
        seconds = sum(p["seconds"] for p in made["pictures"]) / len(made["pictures"])
        print(f"[pictures] {where}: {scores} ({seconds:.2f} s each, {made['peak_reserved_gb']} GB)")
    root.mkdir(parents=True, exist_ok=True)
    templates = {variant: TEMPLATES[variant] for _, _, variant in pairs}
    (root / "summary.json").write_text(json.dumps({"plan": plan, "templates": templates, "rows": rows}, indent=1))
    for key in pairs:
        _, seed_set, variant = key
        done = [r for r in rows if "pictures" in r and (r["model"], r["set"], r["variant"]) == key]
        pictures = [p for r in done for p in r["pictures"]]
        if not pictures:
            continue
        issues: dict[str, int] = {}
        for p in pictures:
            for issue in p.get("issues", []):
                issues[issue] = issues.get(issue, 0) + 1
        mean = sum(p.get("score", 0) for p in pictures) / len(pictures)
        seconds = sorted(p["seconds"] for p in pictures)
        memory = max(r["peak_reserved_gb"] for r in done)
        print(
            f"[pictures] {model}:{seed_set}/{variant}: {len(pictures)} pictures, mean score {mean:.3f}, "
            f"median {seconds[len(seconds) // 2]:.2f} s, peak {memory} GB reserved, issues {json.dumps(issues)}"
        )
    print(f"[pictures] done in {time.time() - clock:.0f} s: {len(jobs) - failed} of {len(jobs)} prompts")
    if not jobs or failed:
        raise SystemExit(1)
