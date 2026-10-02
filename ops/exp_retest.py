"""
Phase 6's re-test in a staging app: a finished test set run again from the same pictures with this
branch's workers (as ops/rerun_set.py does against production), as an ephemeral `modal run` app
named by ORAINGE_APP_NAME. Production's deployed orainge-ai is never touched: nothing is deployed,
and the app goes away when the run ends. Results land in the orainge-outputs volume like any run.

    ORAINGE_APP_NAME=orainge-p6-retest-g \
      modal run ops/exp_retest.py::retest --source phase2 --names phase2g --out ops-out/private/all

``--only 04,06`` limits it to those prompt numbers (a smoke test), ``--offset 1`` adds one to every
seed (a second try), and ``--fresh`` stops if any of the runs already exists (make_model would carry
on from what an older version left).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import modal

if os.environ.get("ORAINGE_APP_NAME", "orainge-ai") == "orainge-ai":
    raise SystemExit("Set ORAINGE_APP_NAME to a staging name: this never runs as production's orainge-ai")

HERE = pathlib.Path(__file__).resolve()
# Here: the checkout's workers/. In a container: /root, where the images put modal_app.py
CHECKOUT = HERE.parents[1] / "workers"
sys.path.insert(0, str(CHECKOUT) if (CHECKOUT / "modal_app.py").is_file() else "/root")
import modal_app as prod  # noqa: E402 - production's code, under the staging app name

app = prod.app
OUTPUTS = "orainge-outputs"


def _read(volume: modal.Volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _copy(volume: modal.Volume, run: str, target: pathlib.Path) -> list[str]:
    """The run's progress.json and every file it lists, as make_set copies them."""
    try:
        state = json.loads(_read(volume, f"{run}/progress.json"))
    except Exception:  # noqa: BLE001 - the run stopped before saving anything
        return []
    target.mkdir(parents=True, exist_ok=True)
    (target / "progress.json").write_text(json.dumps(state, indent=2))
    names = [name for step in state["steps"].values() for name in step.get("files", [])]
    if state.get("input") and state["input"] not in names:
        names.append(state["input"])
    copied = []
    for name in names:
        try:
            (target / name).write_bytes(_read(volume, f"{run}/{name}"))
            copied.append(name)
        except Exception:  # noqa: BLE001
            continue
    return copied


@app.local_entrypoint()
def retest(
    source: str, names: str, out: str = "ops-out/private", only: str = "", offset: int = 0, fresh: bool = False
) -> None:
    print(f"[retest] app {prod.APP_NAME}; finals made by {prod.FINAL_MODEL}")
    volume = modal.Volume.from_name(OUTPUTS)
    existing = {entry.path.strip("/") for entry in volume.listdir("/")}
    runs = sorted(run for run in existing if run.startswith(f"{source}-"))
    if only:
        wanted = {number.strip().zfill(2) for number in only.split(",") if number.strip()}
        runs = [run for run in runs if run.split("-")[1] in wanted]
    print(f"{len(runs)} runs in {source}")
    jobs = []
    for index, name in enumerate(names.split(",")):
        for run in runs:
            state = json.loads(_read(volume, f"{run}/progress.json"))
            if state.get("seed") is None or not state.get("input"):
                print(f"{run}: no picture or seed recorded; skipped")
                continue
            picture = _read(volume, f"{run}/{state['input']}")
            subject = run.split("-", 1)[1]  # phase2-01-cute-low-poly-fox -> 01-cute-low-poly-fox
            image_name = subject.split("-", 1)[1] + ".png"
            jobs.append((f"{name}-{subject}", "", picture, image_name, True, state["seed"] + offset + index))
    if fresh and (taken := sorted(job[0] for job in jobs if job[0] in existing)):
        raise SystemExit(f"These runs already exist, from an earlier version: {', '.join(taken)}")
    # Missing weights are fetched once here, not by every run at the same time
    for which in ("trellis2", "pixal3d"):
        prod.download_models.remote(which=which)
    print(f"Starting {len(jobs)} runs")
    outcomes = list(prod.make_model.starmap(jobs, return_exceptions=True))
    failed = 0
    for job, outcome in zip(jobs, outcomes):
        run = job[0]
        files = _copy(volume, run, pathlib.Path(out) / f"{run.split('-', 1)[0]}-runs" / run)
        if isinstance(outcome, BaseException):
            failed += 1
            reason = next((line for line in str(outcome).splitlines() if line.strip()), "")
            print(f"{run}: failed ({type(outcome).__name__}: {reason})")
        else:
            print(f"{run}: {', '.join(files)}")
    print(f"Done: {len(jobs) - failed} of {len(jobs)} runs finished")
