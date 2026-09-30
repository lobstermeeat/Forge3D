"""
Builds a review gallery from `make` runs: turntables of every preview and final GLB, the
reference images, and the time and GPU cost of each step, so a test set can be judged at a
glance.

    python workers/gallery/make_gallery.py <folder with runs> -o <output folder> --title "..."

The input is any folder holding run folders (each with its progress.json), such as a
`modal volume get orainge-outputs` download or a checkout of the ai-results branch. The output
is index.html plus img/, ready to publish as one page.

Needs Playwright's Chromium (pip install playwright && playwright install chromium) and the
repository's node_modules (pnpm install), which provide three.js and its decoders.
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import io
import json
import pathlib
import statistics
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
# One worker container as configured in modal_app.py: L40S + CPU + memory
DOLLARS_PER_GPU_SECOND = 2.30 / 3600
VIEWS = 6
THUMB = 256


def three_dir() -> pathlib.Path:
    for candidate in ("apps/client/node_modules/three", "node_modules/three", "packages/engine/node_modules/three"):
        path = REPO / candidate
        if (path / "build" / "three.module.js").exists():
            return path.resolve()
    raise SystemExit("three.js not found: run pnpm install at the repository root")


class _Files(http.server.SimpleHTTPRequestHandler):
    """Serves the renderer, three.js and the GLBs from their own folders (and nothing else)."""

    routes: dict[str, pathlib.Path] = {}
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".glb": "model/gltf-binary",
    }

    def translate_path(self, path: str) -> str:
        path = urllib.parse.unquote(path.split("?", 1)[0])
        for prefix, folder in self.routes.items():
            if path.startswith(prefix):
                target = (folder / path[len(prefix):]).resolve()
                if target == folder or folder in target.parents:
                    return str(target)
        return str(HERE / "__not_served__")

    def log_message(self, *args) -> None:  # quiet
        pass


@dataclass
class Model:
    mode: str
    file: str
    strip: Optional[str] = None
    triangles: Optional[int] = None
    megabytes: Optional[float] = None
    gpu_seconds: Optional[float] = None
    seconds: Optional[float] = None


@dataclass
class Run:
    name: str
    subject: str
    references: list[str] = field(default_factory=list)
    chosen: Optional[str] = None
    models: list[Model] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    gpu_seconds: float = 0.0

    @property
    def cost(self) -> float:
        return self.gpu_seconds * DOLLARS_PER_GPU_SECOND


def _save_webp(image, target: pathlib.Path, quality: int = 82) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "WEBP", quality=quality, method=5)


def _thumbnail(source: pathlib.Path, target: pathlib.Path, size: int = 200) -> None:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        _save_webp(image, target)


def _strip(shots: list[str], target: pathlib.Path) -> None:
    import base64

    from PIL import Image

    strip = Image.new("RGB", (THUMB * len(shots), THUMB))
    for i, shot in enumerate(shots):
        data = base64.b64decode(shot.split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as view:
            strip.paste(view.convert("RGB").resize((THUMB, THUMB), Image.LANCZOS), (i * THUMB, 0))
    _save_webp(strip, target)


def load_runs(root: pathlib.Path, out: pathlib.Path) -> list[tuple[Run, pathlib.Path]]:
    runs = []
    for progress in sorted(root.rglob("progress.json")):
        folder = progress.parent
        state = json.loads(progress.read_text())
        steps = state.get("steps", {})
        run = Run(name=folder.name, subject=state.get("prompt") or f"Image: {state.get('input', '')}")
        for step, info in steps.items():
            if info.get("status") == "failed":
                run.failures.append(f"{step}: {info.get('error')}")
            elif info.get("status") == "running":
                run.failures.append(f"{step}: interrupted")
            run.gpu_seconds += info.get("gpu_seconds") or 0
        for name in steps.get("reference", {}).get("files", []):
            if (folder / name).exists():
                thumb = f"img/{run.name}-{pathlib.Path(name).stem}.webp"
                _thumbnail(folder / name, out / thumb)
                run.references.append(thumb)
                if name == state.get("input"):
                    run.chosen = thumb
        if not run.references and state.get("input") and (folder / state["input"]).exists():
            thumb = f"img/{run.name}-input.webp"
            _thumbnail(folder / state["input"], out / thumb)
            run.references.append(thumb)
            run.chosen = thumb
        for mode in ("preview", "final"):
            info = steps.get(mode, {})
            for name in info.get("files", []):
                if (folder / name).exists():
                    run.models.append(
                        Model(
                            mode=mode,
                            file=str((folder / name).relative_to(root)),
                            triangles=info.get("triangles"),
                            megabytes=round((folder / name).stat().st_size / 1e6, 2),
                            gpu_seconds=info.get("gpu_seconds"),
                            seconds=info.get("seconds"),
                        )
                    )
        runs.append((run, folder))
    return runs


def render_models(runs: list[tuple[Run, pathlib.Path]], root: pathlib.Path, out: pathlib.Path) -> None:
    from playwright.sync_api import sync_playwright

    handler = functools.partial(_Files)
    _Files.routes = {"/three/": three_dir(), "/gallery/": HERE, "/files/": root.resolve()}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
            )
            page = browser.new_page(viewport={"width": 600, "height": 600})
            page.goto(f"http://127.0.0.1:{port}/gallery/render.html")
            page.wait_for_function("window.rendererReady === true", timeout=60_000)
            for run, _ in runs:
                for model in run.models:
                    started = time.monotonic()
                    url = "/files/" + urllib.parse.quote(model.file)
                    result = page.evaluate("([url, views]) => window.renderTurntable(url, views)", [url, VIEWS])
                    model.strip = f"img/{run.name}-{model.mode}.webp"
                    _strip(result["shots"], out / model.strip)
                    model.triangles = model.triangles or result["triangles"]
                    print(f"{run.name} {model.mode}: rendered in {time.monotonic() - started:.1f} s")
            browser.close()
    finally:
        server.shutdown()


def _seconds(value: Optional[float]) -> str:
    if value is None:
        return "–"
    # Non-breaking spaces keep a number with its unit when captions wrap on a phone
    return f"{value:.0f}\u00a0s" if value < 90 else f"{value / 60:.1f}\u00a0min"


def _median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def page(runs: list[Run], title: str, subtitle: str) -> str:
    e = html.escape
    finals = [m for r in runs for m in r.models if m.mode == "final"]
    previews = [m for r in runs for m in r.models if m.mode == "preview"]
    failed = [r for r in runs if r.failures]
    costs = [r.cost for r in runs if r.models]
    summary = [
        ("Runs", str(len(runs))),
        ("Finals", str(len(finals))),
        ("Failed", str(len(failed))),
        ("Median preview", _seconds(_median([m.gpu_seconds for m in previews if m.gpu_seconds]))),
        ("Median final", _seconds(_median([m.gpu_seconds for m in finals if m.gpu_seconds]))),
        ("GPU cost per run", f"${statistics.mean(costs):.3f}" if costs else "–"),
    ]
    cards = []
    for run in runs:
        refs = "".join(
            f'<img class="ref{" chosen" if src == run.chosen else ""}" src="{e(src)}" alt="Reference image'
            f'{" used for 3D" if src == run.chosen else ""}" loading="lazy" width="200" height="200">'
            for src in run.references
        )
        models = []
        for model in sorted(run.models, key=lambda m: m.mode != "final"):
            facts = " · ".join(
                x
                for x in (
                    f"{model.triangles:,}\u00a0triangles" if model.triangles else "",
                    f"{model.megabytes}\u00a0MB" if model.megabytes else "",
                    f"{_seconds(model.gpu_seconds)}\u00a0GPU" if model.gpu_seconds else "",
                )
                if x
            )
            strip = (
                f'<img src="{e(model.strip)}" alt="{e(model.mode)} model from six angles" loading="lazy" '
                f'width="{THUMB * VIEWS}" height="{THUMB}">'
                if model.strip
                else '<p class="missing">Not rendered</p>'
            )
            models.append(
                f'<figure class="model {e(model.mode)}"><div class="strip">{strip}</div>'
                f"<figcaption><b>{e(model.mode.capitalize())}</b> {e(facts)}</figcaption></figure>"
            )
        status = (
            '<span class="chip fail">Failed</span>'
            if run.failures
            else ('<span class="chip ok">Final ready</span>' if any(m.mode == "final" for m in run.models) else '<span class="chip">Preview only</span>')
        )
        failures = "".join(f'<p class="error">{e(f)}</p>' for f in run.failures)
        cards.append(
            f'<article class="run" id="{e(run.name)}"><header><h2>{e(run.subject)}</h2>{status}</header>'
            f'<div class="refs">{refs}</div>{"".join(models)}{failures}'
            f'<p class="meta"><span>{e(run.name)}</span><span>GPU ${run.cost:.3f}</span></p></article>'
        )
    stats = "".join(f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in summary)
    return TEMPLATE.format(title=e(title), subtitle=e(subtitle), stats=stats, cards="".join(cards))


TEMPLATE = """<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=Barlow:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* A review board: a summary strip, then one card per prompt with its turntables */
:root {{
  --bg: #f4f3f1; --panel: #ffffff; --ink: #1d1c1a; --muted: #6b6760; --line: #e2dfd9;
  --accent: #d9601f; --good: #2f7d4f; --bad: #b3372b; --stage: #24252c;
  --display: "Barlow Semi Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Barlow", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #141417; --panel: #1c1c21; --ink: #ecebe8; --muted: #a09c95; --line: #2d2d34;
  --accent: #f08a4b; --good: #5fbf85; --bad: #ec7b6f; --stage: #24252c; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #141417; --panel: #1c1c21; --ink: #ecebe8; --muted: #a09c95; --line: #2d2d34;
  --accent: #f08a4b; --good: #5fbf85; --bad: #ec7b6f; --stage: #24252c; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 400 15px/1.5 var(--body); }}
.page {{ max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 32px 48px; display: grid; gap: 28px; }}
.intro {{ display: grid; gap: 10px; }}
.eyebrow {{ margin: 0; font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ margin: 0; font: 600 clamp(28px, 4vw, 40px)/1.1 var(--display); text-wrap: balance; }}
.intro p.lead {{ margin: 0; max-width: 68ch; color: var(--muted); }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1px; margin: 0;
  background: var(--line); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
.stats div {{ background: var(--panel); padding: 12px 14px; display: grid; gap: 4px; }}
.stats dt {{ font: 500 11px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.stats dd {{ margin: 0; font: 500 22px/1.1 var(--mono); font-variant-numeric: tabular-nums; }}
.runs {{ display: grid; gap: 18px; }}
.run {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; display: grid; gap: 14px; min-width: 0; }}
.run header {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 16px; }}
.run h2 {{ margin: 0; font: 600 20px/1.2 var(--display); text-wrap: balance; min-width: 0; }}
.chip {{ font: 500 12px/1 var(--mono); padding: 6px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }}
.chip.ok {{ color: var(--good); border-color: currentColor; }}
.chip.fail {{ color: var(--bad); border-color: currentColor; }}
.refs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.ref {{ width: 96px; height: 96px; object-fit: cover; border-radius: 8px; border: 2px solid transparent; opacity: .72; }}
.ref.chosen {{ border-color: var(--accent); opacity: 1; }}
.model {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
.strip {{ overflow-x: auto; border-radius: 8px; background: var(--stage); }}
.strip img {{ display: block; width: 100%; min-width: 720px; height: auto; }}
.model.preview .strip img {{ min-width: 540px; }}
figcaption {{ font: 400 13px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }}
figcaption b {{ color: var(--ink); font-weight: 500; margin-right: 6px; }}
.error {{ margin: 0; font: 400 13px/1.4 var(--mono); color: var(--bad); overflow-wrap: anywhere; }}
.missing {{ margin: 0; padding: 24px; color: var(--muted); }}
.meta {{ margin: 0; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; font: 400 12px/1 var(--mono); color: var(--muted); }}
footer {{ font-size: 13px; color: var(--muted); }}
</style>
<div class="page">
  <section class="intro">
    <p class="eyebrow">Orainge · AI 3D test set</p>
    <h1>{title}</h1>
    <p class="lead">{subtitle}</p>
    <dl class="stats">{stats}</dl>
  </section>
  <main class="runs">{cards}</main>
  <footer>Built with DINOv3. 3D generation: TRELLIS.2 (Microsoft, MIT). Reference images: FLUX.1 [schnell] (Black Forest Labs, Apache-2.0).</footer>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=pathlib.Path, help="folder holding run folders with progress.json")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("gallery"))
    parser.add_argument("--title", default="Orainge test gallery")
    parser.add_argument(
        "--subtitle",
        default="Each prompt's reference images (the one used for 3D is outlined), then its final and preview "
        "turned through six angles. Would you publish the final as it stands?",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs, args.out)
    if not runs:
        raise SystemExit(f"No runs (progress.json) under {args.runs}")
    render_models(runs, args.runs, args.out)
    (args.out / "index.html").write_text(page([run for run, _ in runs], args.title, args.subtitle))
    print(f"Wrote {args.out / 'index.html'} with {len(runs)} runs")


if __name__ == "__main__":
    main()
