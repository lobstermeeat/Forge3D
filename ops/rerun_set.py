"""
Runs a finished test set again from the same pictures, so pipeline changes can be compared fairly:
each run starts from the picture its original used, as an image run, with the original's seed
(or the next seed, for a second try). Uses the deployed orainge-ai app.

    modal run ops/rerun_set.py --source phase2 --names phase2b,phase2c --out ops-out/private

``--only 06,14`` limits it to those prompt numbers (a smoke test before the whole set).
"""

from __future__ import annotations

import json
import pathlib

import modal

app = modal.App("orainge-rerun")
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
def main(source: str, names: str, out: str = "ops-out/private", only: str = "") -> None:
    volume = modal.Volume.from_name(OUTPUTS)
    make_model = modal.Function.from_name("orainge-ai", "make_model")
    runs = sorted(
        entry.path.strip("/")
        for entry in volume.listdir("/")
        if entry.path.strip("/").startswith(f"{source}-")
    )
    if only:
        wanted = {number.strip().zfill(2) for number in only.split(",") if number.strip()}
        runs = [run for run in runs if run.split("-")[1] in wanted]
    print(f"{len(runs)} runs in {source}")
    jobs = []
    for offset, name in enumerate(names.split(",")):
        for run in runs:
            state = json.loads(_read(volume, f"{run}/progress.json"))
            if state.get("seed") is None or not state.get("input"):
                print(f"{run}: no picture or seed recorded; skipped")
                continue
            picture = _read(volume, f"{run}/{state['input']}")
            subject = run.split("-", 1)[1]  # phase2-01-cute-low-poly-fox -> 01-cute-low-poly-fox
            new_run = f"{name}-{subject}"
            # A readable file name, so galleries head the run with its subject
            image_name = subject.split("-", 1)[1] + ".png"
            jobs.append((new_run, "", picture, image_name, True, state["seed"] + offset))
    print(f"Starting {len(jobs)} runs")
    outcomes = list(make_model.starmap(jobs, return_exceptions=True))
    failed = 0
    for job, outcome in zip(jobs, outcomes):
        run = job[0]
        name = run.split("-", 1)[0]
        files = _copy(volume, run, pathlib.Path(out) / f"{name}-runs" / run)
        if isinstance(outcome, BaseException):
            failed += 1
            reason = next((line for line in str(outcome).splitlines() if line.strip()), "")
            print(f"{run}: failed ({type(outcome).__name__}: {reason})")
        else:
            print(f"{run}: {', '.join(files)}")
    print(f"Done: {len(jobs) - failed} of {len(jobs)} runs finished")
