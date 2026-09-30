"""
Checks the deployed job API end to end, the way the Orainge server uses it: a prompt, four
pictures, then a preview made from the first picture. It runs inside Modal, where the worker
token is, so the token never reaches GitHub or its logs. Costs about as much as one preview.

    modal run ops/check_api.py --url https://<workspace>--orainge-ai-api.modal.run
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

app = modal.App("orainge-api-check")
image = modal.Image.debian_slim(python_version="3.11")

POLL_S = 2.0  # the Studio's panel polls this often too
LIMIT_S = 20 * 60
MIN_TOKEN_LENGTH = 32  # as in workers/job_api.py
SECRET = modal.Secret.from_name("orainge-worker-token", required_keys=["ORAINGE_WORKER_TOKEN"])


def token_problems(token: str) -> list[str]:
    """What would stop the job API from accepting this token (never the token itself)."""
    problems = []
    if len(token.strip()) < MIN_TOKEN_LENGTH:
        problems.append(f"is shorter than the {MIN_TOKEN_LENGTH} characters the job API needs")
    if token != token.strip():
        problems.append("has spaces or a line break around it")
    return problems


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
    log(f"{worker}: queued as {queued['id']} ({time.monotonic() - started:.1f} s)")
    while True:
        time.sleep(POLL_S)
        status, state = request(url, token, "GET", f"/{worker}/status/{queued['id']}")
        if status != 200:
            raise RuntimeError(f"{worker}/status: HTTP {status} {state}")
        if state["status"] == "COMPLETED":
            return state["output"], time.monotonic() - started
        if state["status"] not in ("IN_QUEUE", "IN_PROGRESS"):
            raise RuntimeError(f"{worker}: {state['status']}: {state.get('error')}")
        if time.monotonic() - started > LIMIT_S:
            request(url, token, "POST", f"/{worker}/cancel/{queued['id']}")
            raise RuntimeError(f"{worker}: still running after {LIMIT_S // 60} minutes, so it was cancelled")


def check_flow(url: str, token: str, prompt: str, log: Callable[[str], None] = print) -> dict:
    url = url.rstrip("/")
    status, _ = request(url, None, "GET", "/")
    if status != 401:
        raise RuntimeError(f"without the token the API answered {status}, not 401")
    status, _ = request(url, "not-the-token-" + "x" * 32, "GET", "/")
    if status != 401:
        raise RuntimeError(f"with a wrong token the API answered {status}, not 401")
    status, index = request(url, token, "GET", "/")
    if status != 200 or sorted(index.get("workers", [])) != ["reference", "trellis2"]:
        raise RuntimeError(f"GET / answered {status} {index}")
    log("Token: refused without it or with a wrong one; accepted with it")

    refs, ref_seconds = run_job(
        url, token, "reference", {"prompt": prompt, "count": 4, "request_id": "api-check"}, log
    )
    pictures = [base64.b64decode(picture["base64"]) for picture in refs["images"]]
    if len(pictures) != 4 or not all(p.startswith(b"\x89PNG") for p in pictures):
        raise RuntimeError(f"expected 4 PNG pictures, got {len(pictures)}")
    sizes = ", ".join(f"{len(p) // 1024} KB" for p in pictures)
    log(f"Pictures: 4 PNGs ({sizes}) after {ref_seconds:.0f} s, {refs.get('seconds')} s of it on the GPU")

    preview, preview_seconds = run_job(
        url,
        token,
        "trellis2",
        {"image_base64": refs["images"][0]["base64"], "mode": "preview", "request_id": "api-check"},
        log,
    )
    glb = base64.b64decode(preview["glb"]["base64"])
    if glb[:4] != b"glTF":
        raise RuntimeError("the preview is not a GLB")
    gpu = sum(preview["timings"].values())
    log(
        f"Preview: {preview['triangles']:,} triangles, {len(glb) / 1e6:.1f} MB, seed {preview['seed']}, "
        f"after {preview_seconds:.0f} s, {gpu:.0f} s of it on the GPU"
    )
    log("Credits: " + "; ".join(preview.get("credits") or ["(none)"]))
    return {
        "pictures": pictures,
        "glb": glb,
        "seed": preview["seed"],
        "seconds": {"pictures": round(ref_seconds, 1), "preview": round(preview_seconds, 1)},
    }


@app.function(image=image, secrets=[SECRET])
def token_check() -> None:
    problems = token_problems(os.environ["ORAINGE_WORKER_TOKEN"])
    print("ORAINGE_WORKER_TOKEN " + (" and ".join(problems) if problems else "is fine: long enough, nothing around it"))


@app.function(image=image, secrets=[SECRET], timeout=45 * 60)
def check(url: str, prompt: str) -> dict:
    token = os.environ["ORAINGE_WORKER_TOKEN"]
    problems = token_problems(token)
    if problems:
        raise RuntimeError("ORAINGE_WORKER_TOKEN in the Modal secret orainge-worker-token " + " and ".join(problems))
    return check_flow(url, token, prompt)


@app.local_entrypoint()
def main(url: str, prompt: str = "a red fire hydrant", out: str = "ops-out/private/api-check") -> None:
    result = check.remote(url, prompt)
    folder = pathlib.Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for number, picture in enumerate(result["pictures"], start=1):
        (folder / f"picture-{number}.png").write_bytes(picture)
    (folder / f"preview-{result['seed']}.glb").write_bytes(result["glb"])
    print(f"The job API works end to end ({result['seconds']}). Saved the pictures and the preview in {folder}/")
