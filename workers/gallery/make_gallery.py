"""
Builds a review gallery from `make` runs: turntables of every preview and final GLB, the
reference images, and the time and GPU cost of each step, so a test set can be judged at a
glance.

    python workers/gallery/make_gallery.py <folder with runs> -o <output folder> --title "..."

The input is any folder holding run folders (each with its progress.json), such as the
orainge-outputs/ folder that make and make_set fill. The output is index.html with img/, plus
each final GLB in models/ for the page's 3D viewer (--no-3d leaves them out), ready to publish as
one page. The viewer loads three.js from jsDelivr, so the page must be served over http(s).

--verdicts adds a reviewer's judgement from a JSON file: {"<run>": {"verdict": "publish" | "edits" |
"reject", "note": "why", "group": "Everyday"}}. Each card shows it, and the summary counts the
runs that are publishable as they stand, overall and per group, against the go/no-go bar.

Needs Playwright's Chromium (pip install playwright && playwright install chromium) and the
repository's node_modules (pnpm install), which provide three.js and its decoders.
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import http.server
import io
import json
import pathlib
import shutil
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
# The go/no-go bar: the share of a test set that must be publishable without editing
BAR = 0.8
VERDICTS = {"publish": ("Publishable", "ok"), "edits": ("Needs edits", "warn"), "reject": ("Not usable", "fail")}


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
    viewer: Optional[str] = None  # the copy the page's 3D viewer opens
    timings: dict = field(default_factory=dict)  # the worker's seconds per stage


@dataclass
class Run:
    name: str
    subject: str
    references: list[str] = field(default_factory=list)
    chosen: Optional[str] = None
    models: list[Model] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    gpu_seconds: float = 0.0
    photo: bool = False  # started from a photo rather than a prompt
    verdict: Optional[str] = None  # a key of VERDICTS, from --verdicts
    note: str = ""
    group: str = ""

    @property
    def cost(self) -> float:
        return self.gpu_seconds * DOLLARS_PER_GPU_SECOND


def _photo_name(file: str) -> str:
    """A heading for a photo run: input-toy-car.jpg -> Toy car."""
    words = pathlib.Path(file).stem.removeprefix("input-").replace("-", " ").replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else "Photo"


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
        prompt = state.get("prompt")
        run = Run(name=folder.name, subject=prompt or _photo_name(state.get("input", "")), photo=not prompt)
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
                            timings=info.get("timings") or {},
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


def apply_verdicts(runs: list[Run], verdicts: dict) -> None:
    """Attaches a reviewer's verdicts ({run: {"verdict", "note", "group"}}) to the runs they name."""
    known = {run.name for run in runs}
    unknown = sorted(set(verdicts) - known)
    if unknown:
        raise SystemExit(f"--verdicts names runs that aren't in the gallery: {', '.join(unknown)}")
    for run in runs:
        entry = verdicts.get(run.name)
        if not entry:
            continue
        if entry.get("verdict") not in VERDICTS:
            raise SystemExit(f"{run.name}: verdict must be one of {', '.join(VERDICTS)}")
        run.verdict, run.note, run.group = entry["verdict"], entry.get("note", ""), entry.get("group", "")


def _share(runs: list[Run]) -> str:
    passed = sum(run.verdict == "publish" for run in runs)
    return f"{passed} of {len(runs)} ({passed / len(runs):.0%})"


def page(runs: list[Run], title: str, subtitle: str, three_version: Optional[str] = None) -> str:
    e = html.escape
    finals = [m for r in runs for m in r.models if m.mode == "final"]
    previews = [m for r in runs for m in r.models if m.mode == "preview"]
    failed = [r for r in runs if r.failures]
    costs = [r.cost for r in runs if r.models]
    judged = [r for r in runs if r.verdict]
    verdicts: list[tuple[str, str]] = []
    if judged:
        verdicts.append((f"Publishable (bar {BAR:.0%})", _share(judged)))
        for group in dict.fromkeys(r.group for r in judged if r.group):
            verdicts.append((group, _share([r for r in judged if r.group == group])))
    summary = verdicts + [
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
            f'<img class="ref{" chosen" if src == run.chosen else ""}" src="{e(src)}" '
            f'alt="{"Input photo" if run.photo else "Reference image" + (" used for 3D" if src == run.chosen else "")}" '
            'loading="lazy" width="200" height="200">'
            for src in run.references
        )
        models = []
        for model in sorted(run.models, key=lambda m: m.mode != "final"):
            # Hovering the GPU time shows where it went (generate, export, compress, upload)
            stages = ", ".join(f"{k.removesuffix('_s')} {v:.0f} s" for k, v in model.timings.items())
            facts = " · ".join(
                x
                for x in (
                    e(f"{model.triangles:,}\u00a0triangles") if model.triangles else "",
                    e(f"{model.megabytes}\u00a0MB") if model.megabytes else "",
                    f'<span title="{e(stages)}">{e(_seconds(model.gpu_seconds))}\u00a0GPU</span>'
                    if model.gpu_seconds
                    else "",
                )
                if x
            )
            strip = (
                f'<img src="{e(model.strip)}" alt="{e(model.mode)} model from six angles" loading="lazy" '
                f'width="{THUMB * VIEWS}" height="{THUMB}">'
                if model.strip
                else '<p class="missing">Not rendered</p>'
            )
            view = (
                f'<button type="button" class="view3d" data-model="{e(model.viewer)}" '
                f'data-title="{e(run.subject)}">View in 3D</button>'
                if model.viewer
                else ""
            )
            models.append(
                f'<figure class="model {e(model.mode)}"><div class="strip">{strip}</div>'
                f"<figcaption><span><b>{e(model.mode.capitalize())}</b> {facts}</span>{view}</figcaption>"
                "</figure>"
            )
        if run.verdict:
            label, tone = VERDICTS[run.verdict]
            status = f'<span class="chip {tone}">{e(label)}</span>'
        elif run.failures:
            status = '<span class="chip fail">Failed</span>'
        elif any(m.mode == "final" for m in run.models):
            status = '<span class="chip ok">Final ready</span>'
        else:
            status = '<span class="chip">Preview only</span>'
        group = f'<span class="chip">{e(run.group)}</span>' if run.group else ""
        failures = "".join(f'<p class="error">{e(f)}</p>' for f in run.failures)
        note = f'<p class="note">{e(run.note)}</p>' if run.note else ""
        cards.append(
            f'<article class="run" id="{e(run.name)}"><header><h2>{e(run.subject)}</h2>'
            f'<span class="chips">{group}{status}</span></header>{note}'
            f'<div class="refs">{refs}</div>{"".join(models)}{failures}'
            f'<p class="meta"><span>{e(run.name)}</span><span>GPU ${run.cost:.3f}</span></p></article>'
        )
    stats = "".join(f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in summary)
    body = TEMPLATE.format(title=e(title), subtitle=e(subtitle), stats=stats, cards="".join(cards))
    if three_version and any(m.viewer for r in runs for m in r.models):
        body += VIEWER.replace("__THREE__", three_version)
    return body


def copy_models(
    runs: list[tuple[Run, pathlib.Path]], root: pathlib.Path, out: pathlib.Path, as_text: bool = False
) -> Optional[str]:
    """
    Copies each final GLB next to the page, with the KTX2 transcoder the page's viewer needs,
    and returns the three.js version the viewer should load (None when there is nothing to view).
    With as_text the copies are base64 .txt files, for hosts that serve only web file types.
    """
    finals = [(run, model) for run, _ in runs for model in run.models if model.mode == "final"]
    if not finals:
        return None
    (out / "models").mkdir(parents=True, exist_ok=True)
    for run, model in finals:
        model.viewer = f"models/{run.name}-final.glb" + (".txt" if as_text else "")
        if as_text:
            (out / model.viewer).write_bytes(base64.b64encode((root / model.file).read_bytes()))
        else:
            shutil.copyfile(root / model.file, out / model.viewer)
    basis = out / "vendor" / "basis"
    basis.mkdir(parents=True, exist_ok=True)
    for name in ("basis_transcoder.js", "basis_transcoder.wasm"):
        shutil.copyfile(three_dir() / "examples" / "jsm" / "libs" / "basis" / name, basis / name)
    return json.loads((three_dir() / "package.json").read_text())["version"]


TEMPLATE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
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
.chip.warn {{ color: var(--accent); border-color: currentColor; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.note {{ margin: 0; max-width: 80ch; }}
.refs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.ref {{ width: 96px; height: 96px; object-fit: cover; border-radius: 8px; border: 2px solid transparent; opacity: .72; }}
.ref.chosen {{ border-color: var(--accent); opacity: 1; }}
.model {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
.strip {{ overflow-x: auto; border-radius: 8px; background: var(--stage); }}
.strip img {{ display: block; width: 100%; min-width: 720px; height: auto; }}
.model.preview .strip img {{ min-width: 540px; }}
figcaption {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px 12px;
  font: 400 13px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }}
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


# A viewer for the finals, appended when they are copied next to the page. three.js comes from
# jsDelivr (the same version the renderer used); the KTX2 transcoder is served with the page.
# The buttons stay hidden unless the viewer loads, so the page never offers a dead control.
VIEWER = """
<dialog id="viewer" aria-labelledby="viewer-title">
  <header>
    <h2 id="viewer-title">Model</h2>
    <div class="tools">
      <button type="button" id="viewer-spin" aria-pressed="true">Turning</button>
      <button type="button" id="viewer-wire" aria-pressed="false">Wireframe</button>
      <button type="button" id="viewer-close">Close</button>
    </div>
  </header>
  <canvas id="viewer-stage"></canvas>
  <p id="viewer-status" role="status"></p>
</dialog>
<style>
.view3d { display: none; font: 500 12px/1 var(--mono); color: var(--ink); background: var(--panel);
  border: 1px solid var(--line); border-radius: 999px; padding: 7px 11px; cursor: pointer; }
:root[data-viewer="ready"] .view3d { display: inline-block; }
.view3d:hover, .view3d:focus-visible, #viewer button:hover, #viewer button:focus-visible { border-color: var(--accent); color: var(--accent); }
:is(.view3d, #viewer button):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
#viewer { width: min(960px, calc(100vw - 32px)); max-height: calc(100dvh - 32px); padding: 0; border: 1px solid var(--line);
  border-radius: 12px; background: var(--panel); color: var(--ink); overflow: hidden; }
#viewer::backdrop { background: rgb(10 10 12 / .6); }
#viewer header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 16px; padding: 12px 14px; }
#viewer h2 { margin: 0; font: 600 18px/1.2 var(--display); min-width: 0; }
#viewer .tools { display: flex; flex-wrap: wrap; gap: 6px; }
#viewer button { font: 500 12px/1 var(--mono); color: var(--ink); background: transparent; border: 1px solid var(--line);
  border-radius: 999px; padding: 7px 11px; cursor: pointer; }
#viewer button[aria-pressed="true"] { background: var(--ink); color: var(--panel); border-color: var(--ink); }
#viewer-stage { display: block; width: 100%; aspect-ratio: 4 / 3; max-height: calc(100dvh - 150px); background: var(--stage); touch-action: none; }
#viewer-status { margin: 0; padding: 10px 14px 14px; font: 400 13px/1.4 var(--mono); color: var(--muted); }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.min.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const dialog = document.getElementById('viewer');
const canvas = document.getElementById('viewer-stage');
const heading = document.getElementById('viewer-title');
const status = document.getElementById('viewer-status');
const spin = document.getElementById('viewer-spin');
const wire = document.getElementById('viewer-wire');
let renderer, scene, camera, controls, loader, model, frame, loading = 0;

function setup() {
  // The same lighting and tone mapping as the editor and the turntables
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x24252c);
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.autoRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  spin.setAttribute('aria-pressed', String(controls.autoRotate));
  const ktx2 = new KTX2Loader().setTranscoderPath('vendor/basis/').detectSupport(renderer);
  loader = new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);
}

function draw() {
  frame = requestAnimationFrame(draw);
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * renderer.getPixelRatio()) || canvas.height !== Math.round(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  controls.update();
  renderer.render(scene, camera);
}

// Some hosts serve only web file types, so a GLB may come as base64 text (--glb-as-text)
async function fromBase64(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} for ${url}`);
  const binary = atob((await response.text()).trim());
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function clear() {
  if (!model) return;
  scene.remove(model);
  model.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry.dispose();
    for (const material of [object.material].flat()) {
      for (const value of Object.values(material)) if (value && value.isTexture) value.dispose();
      material.dispose();
    }
  });
  model = null;
}

async function show(button) {
  if (!renderer) setup();
  const ticket = ++loading;
  heading.textContent = button.dataset.title;
  status.textContent = 'Loading the model…';
  wire.setAttribute('aria-pressed', 'false');
  clear();
  dialog.showModal();
  cancelAnimationFrame(frame);
  draw();
  try {
    const url = button.dataset.model;
    const gltf = url.endsWith('.txt') ? await loader.parseAsync(await fromBase64(url), '') : await loader.loadAsync(url);
    if (ticket !== loading || !dialog.open) return;
    model = gltf.scene;
    const sphere = new THREE.Box3().setFromObject(model).getBoundingSphere(new THREE.Sphere());
    model.position.sub(sphere.center);
    scene.add(model);
    const distance = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.1;
    camera.near = distance / 100;
    camera.far = distance * 10;
    camera.position.set(Math.SQRT1_2, 0.4, Math.SQRT1_2).normalize().multiplyScalar(distance);
    controls.target.set(0, 0, 0);
    controls.minDistance = distance * 0.15;
    controls.maxDistance = distance * 4;
    controls.update();
    let triangles = 0;
    model.traverse((object) => {
      if (object.isMesh) triangles += (object.geometry.index ? object.geometry.index.count : object.geometry.attributes.position.count) / 3;
    });
    status.textContent = `${Math.round(triangles).toLocaleString()} triangles. Drag to turn, scroll or pinch to zoom.`;
  } catch (error) {
    if (ticket === loading) status.textContent = `This model couldn't be shown here (${error.message || error}). Its turntable on the page shows the same model.`;
  }
}

document.addEventListener('click', (event) => {
  const button = event.target.closest('button.view3d');
  if (button) show(button);
});
dialog.addEventListener('close', () => { loading++; cancelAnimationFrame(frame); clear(); });
document.getElementById('viewer-close').addEventListener('click', () => dialog.close());
spin.addEventListener('click', () => {
  controls.autoRotate = !controls.autoRotate;
  spin.setAttribute('aria-pressed', String(controls.autoRotate));
});
wire.addEventListener('click', () => {
  const on = wire.getAttribute('aria-pressed') !== 'true';
  wire.setAttribute('aria-pressed', String(on));
  model?.traverse((object) => { if (object.isMesh) for (const material of [object.material].flat()) material.wireframe = on; });
});
document.documentElement.dataset.viewer = 'ready';
</script>
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
    parser.add_argument("--no-3d", action="store_true", help="leave out the final GLBs and the 3D viewer")
    parser.add_argument(
        "--glb-as-text", action="store_true", help="copy the GLBs as base64 .txt, for hosts that don't serve .glb"
    )
    parser.add_argument("--verdicts", type=pathlib.Path, help="a reviewer's verdicts as JSON (see above)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs, args.out)
    if not runs:
        raise SystemExit(f"No runs (progress.json) under {args.runs}")
    if args.verdicts:
        apply_verdicts([run for run, _ in runs], json.loads(args.verdicts.read_text()))
    render_models(runs, args.runs, args.out)
    three_version = None if args.no_3d else copy_models(runs, args.runs, args.out, args.glb_as_text)
    (args.out / "index.html").write_text(page([run for run, _ in runs], args.title, args.subtitle, three_version))
    print(f"Wrote {args.out / 'index.html'} with {len(runs)} runs")


if __name__ == "__main__":
    main()
