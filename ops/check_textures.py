"""
Checks texture options on the deployed job API, the way the Orainge server uses them: a final from a
Phase 2 picture at its seed, then a "textures" job for that final (3 more textures of its shape). It runs
inside Modal, where the worker token is, so the token never reaches GitHub or its logs. Costs about as
much as two finals.

    modal run ops/check_textures.py --url https://<workspace>--orainge-ai-api.modal.run
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

import modal

app = modal.App("orainge-textures-check")
image = modal.Image.debian_slim(python_version="3.11")
SECRET = modal.Secret.from_name("orainge-worker-token", required_keys=["ORAINGE_WORKER_TOKEN"])
OUTPUTS = modal.Volume.from_name("orainge-outputs")

POLL_S = 5.0  # the Studio's panel polls textures this often
LIMIT_S = 20 * 60


def request(url: str, token: Optional[str], method: str, path: str, body: Any = None) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(errors="replace")[:300]


def run_job(url: str, token: str, worker: str, job: dict, log: Callable[[str], None]) -> tuple[dict, float]:
    """POST /{worker}/run, then poll its status like the server does. Returns (output, seconds)."""
    started = time.monotonic()
    status, queued = request(url, token, "POST", f"/{worker}/run", {"input": job})
    if status != 200:
        raise RuntimeError(f"{worker}/run: HTTP {status} {queued}")
    log(f"{job['mode']}: queued as {queued['id']}")
    while True:
        time.sleep(POLL_S)
        status, state = request(url, token, "GET", f"/{worker}/status/{queued['id']}")
        if status != 200:
            raise RuntimeError(f"{worker}/status: HTTP {status} {state}")
        if state["status"] == "COMPLETED":
            return state["output"], time.monotonic() - started
        if state["status"] not in ("IN_QUEUE", "IN_PROGRESS"):
            raise RuntimeError(f"{job['mode']}: {state['status']}: {state.get('error')}")
        if time.monotonic() - started > LIMIT_S:
            request(url, token, "POST", f"/{worker}/cancel/{queued['id']}")
            raise RuntimeError(f"{job['mode']}: still running after {LIMIT_S // 60} minutes, so it was cancelled")


@app.function(image=image, secrets=[SECRET], volumes={"/outputs": OUTPUTS}, timeout=50 * 60)
def check(url: str, source: str) -> dict:
    url = url.rstrip("/")
    token = os.environ["ORAINGE_WORKER_TOKEN"].strip()
    folder = pathlib.Path("/outputs") / source
    state = json.loads((folder / "progress.json").read_text())
    picture = (folder / state["input"]).read_bytes()
    seed = int(state["seed"])
    request_id = f"textures-check-{int(time.time())}"
    base = {"image_base64": base64.b64encode(picture).decode(), "seed": seed, "request_id": request_id}

    final, final_seconds = run_job(url, token, "trellis2", {**base, "mode": "final"}, print)
    if final.get("error"):
        raise RuntimeError(f"final: {final['error']}")
    glb = base64.b64decode(final["glb"]["base64"])
    print(
        f"Final: {final['triangles']:,} triangles, {len(glb) / 1e6:.1f} MB, pipeline {final.get('pipeline')}, "
        f"model {final.get('model')}, after {final_seconds:.0f} s ({sum(final['timings'].values()):.0f} s of GPU work)"
    )

    textures, textures_seconds = run_job(url, token, "trellis2", {**base, "mode": "textures", "count": 3}, print)
    if textures.get("error"):
        raise RuntimeError(f"textures: {textures['error']}")
    made = textures.get("textures") or []
    glbs = {t["texture_seed"]: base64.b64decode(t["glb"]["base64"]) for t in made}
    for t in made:
        export = t.get("export") or {}
        print(
            f"Texture {t['texture_seed']}: {t['triangles']:,} triangles, {t['bytes'] / 1e6:.1f} MB, "
            f"export {export.get('path')} in {export.get('seconds')} s, key {t['glb'].get('key')}"
        )
    if textures.get("texture_errors"):
        print(f"Texture errors: {json.dumps(textures['texture_errors'])}")
    print(
        f"Textures: {len(made)} of 3 after {textures_seconds:.0f} s; pipeline {textures.get('pipeline')}; "
        f"timings {json.dumps(textures.get('timings'))}"
    )
    if len(made) != 3 or not all(g[:4] == b"glTF" for g in glbs.values()):
        raise RuntimeError(f"expected 3 GLBs, got {len(made)}")
    if textures.get("pipeline") != final.get("pipeline"):
        raise RuntimeError(f"the textures' pipeline {textures.get('pipeline')} isn't the final's {final.get('pipeline')}")
    return {
        "picture": picture,
        "seed": seed,
        "final": glb,
        "textures": glbs,
        "seconds": {"final": round(final_seconds, 1), "textures": round(textures_seconds, 1)},
    }


@app.local_entrypoint()
def main(url: str, source: str = "phase2-12-skateboard-with-graffiti-art", out: str = "ops-out/private/textures-check") -> None:
    result = check.remote(url, source)
    folder = pathlib.Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "picture.png").write_bytes(result["picture"])
    (folder / f"final-{result['seed']}.glb").write_bytes(result["final"])
    for texture_seed, glb in result["textures"].items():
        (folder / f"texture-{texture_seed}.glb").write_bytes(glb)
    print(f"Texture options work on the deployed job API ({result['seconds']}). Saved the models in {folder}/")
