"""
ops/exp_judge.py without Modal's servers or a GPU: what it makes and saves for an object, what it asks the
judges, and how it reads their answers, with stand-ins for TRELLIS.2, the turntable and the judge calls.
"""

import base64
import importlib.util
import io
import json
import pathlib
from types import SimpleNamespace

import pytest
from PIL import Image

modal = pytest.importorskip("modal")

WORKERS = pathlib.Path(__file__).parent.parent
SCRIPT = WORKERS.parent / "ops" / "exp_judge.py"
SOURCE = "phase2e-04-retro-arcade-machine"
SUBJECT = "04-retro-arcade-machine"
SEED = 949054594
COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]


def load(app_name):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ORAINGE_APP_NAME", app_name)
        patch.syspath_prepend(str(WORKERS / "trellis2"))  # forge3d_worker.settings, for make_object
        spec = importlib.util.spec_from_file_location("exp_judge", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="module")
def exp():
    module = load("orainge-test-judge")
    import sys

    sys.path.insert(0, str(WORKERS / "trellis2"))
    yield module
    sys.path.remove(str(WORKERS / "trellis2"))


def test_it_never_runs_as_production():
    with pytest.raises(SystemExit, match="staging name"):
        load("orainge-ai")


def test_it_runs_in_productions_app_with_a_class_of_its_own(exp):
    import modal_app

    assert exp.app is modal_app.app and isinstance(exp.JudgeRolls, modal.Cls)
    assert exp.SIZES == ("8b", "30b") and exp.RESULTS == "phase7/judge"
    assert exp.ROLL_OBJECTS[0] == "11" and sorted(exp.ROLL_OBJECTS) == [f"{n:02d}" for n in range(1, 21)]


def test_the_orders_mirror_each_other_and_stay_put(exp):
    first, second = exp.orders_for("04", 4, 2)
    assert sorted(first) == [0, 1, 2, 3] and second == first[::-1]
    assert exp.orders_for("04", 4, 2) == [first, second] and exp.orders_for("04", 4, 1) == [first]
    third = exp.orders_for("04", 4, 3)[2]
    assert sorted(third) == [0, 1, 2, 3]
    assert len({tuple(exp.orders_for(f"{n:02d}", 4, 1)[0]) for n in range(1, 21)}) > 5  # objects differ


def test_prompts_come_from_the_test_set_by_number(exp):
    typed = exp.prompts_by_number(WORKERS / "test-sets" / "phase2.txt")
    assert len(typed) == 20 and typed["04"] == "a retro arcade machine" and typed["18"] == "electric guitar"


def test_runs_are_found_by_prefix_and_number(exp):
    names = [SOURCE, "phase2-04-retro-arcade-machine", "/phase2e-11-stack-of-old-books-with-a-candle", "phase2e-x", "phase7"]

    class Volume:
        def listdir(self, path):
            return [SimpleNamespace(path=name) for name in names]

    assert exp.runs_by_number(Volume(), "phase2e") == {"04": SOURCE, "11": "phase2e-11-stack-of-old-books-with-a-candle"}
    assert exp.runs_by_number(Volume(), "phase2") == {"04": "phase2-04-retro-arcade-machine"}


class Runtime:
    """Trellis2Runtime's stand-in: meshes are names, a GLB is its mesh's name."""

    pipeline_used = "1024_cascade"

    def __init__(self):
        self.calls = []
        self.last_projection = None
        self.last_retexture = None

    def generate(self, picture, preset, seed):
        self.calls.append(("generate", picture.size, preset.pipeline_type, seed))
        return "generated"

    def retexture(self, seed=None):
        self.calls.append(("retexture", seed))
        self.last_retexture = {"noise": f"seed {seed}"}
        return f"retextured-{seed}"

    def export(self, made, preset):
        self.calls.append(("export", made))
        self.last_projection = {"applied": True}
        return f"glb:{made}".encode(), 1234


def picture_png():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (250, 250, 250)).save(buffer, format="PNG")
    return buffer.getvalue()


def source_run(root):
    folder = root / SOURCE
    folder.mkdir()
    (folder / "input-retro-arcade-machine.png").write_bytes(picture_png())
    progress = {"prompt": None, "seed": SEED, "input": "input-retro-arcade-machine.png", "final": True, "steps": {}}
    (folder / "progress.json").write_text(json.dumps(progress))


def candidate(raw: bytes) -> int:
    """Which roll a GLB is: 0 for the generation's texture, k for seed + 1000 k."""
    made = raw.decode().removeprefix("glb:")
    return 0 if made == "generated" else (int(made.removeprefix("retextured-")) - SEED) // 1000


def job(**extra):
    return {"number": "04", "source": SOURCE, "prompt": "a retro arcade machine", "count": 3, "sizes": ["8b", "30b"], "orders": 2, **extra}


def test_an_object_is_made_saved_drawn_and_sent_to_both_judges(exp, tmp_path):
    source_run(tmp_path)
    runtime, spawned = Runtime(), []

    def spawn(size, request):
        spawned.append((size, request))
        return f"fc-{len(spawned)}"

    def pack(raw, texture_size):
        return b"packed:" + raw + b":" + str(texture_size).encode()

    made = exp.make_object(
        job(), runtime, tmp_path, pack=pack, render=lambda raw: Image.new("RGB", (1152, 768), COLOURS[candidate(raw)]), spawn=spawn
    )
    summary, files = made["summary"], made["files"]
    # Production's final, then three textures of its shape from seed + 1000 k, each exported
    assert runtime.calls == [
        ("generate", (64, 64), "1024_cascade", SEED),
        ("export", "generated"),
        *[call for k in (1, 2, 3) for call in (("retexture", SEED + 1000 * k), ("export", f"retextured-{SEED + 1000 * k}"))],
    ]
    assert summary["candidates"] == ["t9roll0", "t9roll1", "t9roll2", "t9roll3"]
    assert summary["pipeline"] == "1024_cascade" and summary["prompt"] == "a retro arcade machine" and summary["seed"] == SEED
    for k, variant in enumerate(summary["candidates"]):
        run = f"{variant}-{SUBJECT}"
        # Saved as the rolls are: the packed GLB, the picture, progress.json; in the volume and the results
        packed = files[f"{run}/final-{SEED}.glb"]
        assert candidate(packed.removeprefix(b"packed:").rsplit(b":", 1)[0]) == k and packed.endswith(b":2048")
        assert (tmp_path / exp.RESULTS / run / f"final-{SEED}.glb").read_bytes() == packed
        assert files[f"{run}/input-retro-arcade-machine.png"] == picture_png()
        progress = json.loads(files[f"{run}/progress.json"])
        assert progress["steps"]["final"]["files"] == [f"final-{SEED}.glb"] and progress["final"] is True
        assert progress["prompt"] == "a retro arcade machine" and progress["input"] == "input-retro-arcade-machine.png"
        assert progress["made_by"] == {"experiment": "phase7 judge", "variant": variant, "picture_from": SOURCE}
        assert progress["steps"]["final"]["projection"] == {"applied": True}
        # The grid the judges saw
        grid = Image.open(io.BytesIO(files[f"judge/{SUBJECT}/{variant}.jpg"]))
        assert grid.size == (1152, 768) and max(abs(a - b) for a, b in zip(grid.getpixel((9, 9)), COLOURS[k])) <= 6
        assert (tmp_path / exp.RESULTS / "judge" / SUBJECT / f"{variant}.jpg").exists()
    assert summary["variants"]["t9roll2"]["retexture"] == {"noise": f"seed {SEED + 2000}"}

    # Both sizes in both orders, all at once, each with the picture and the four grids in roll order
    orders = exp.orders_for("04", 4, 2)
    assert [(size, request["order"]) for size, request in spawned] == [
        ("8b", orders[0]), ("30b", orders[0]), ("8b", orders[1]), ("30b", orders[1])
    ]
    for _, request in spawned:
        assert set(request) == {"picture_png", "candidates_png", "prompt", "order"}
        assert request["prompt"] == "a retro arcade machine"
        assert base64.b64decode(request["picture_png"]) == picture_png()
        grids = [Image.open(io.BytesIO(base64.b64decode(item))) for item in request["candidates_png"]]
        assert [grid.format for grid in grids] == ["PNG"] * 4 and [grid.getpixel((9, 9)) for grid in grids] == COLOURS
    assert summary["judge_calls"] == [
        {"size": size, "order": order, "call": f"fc-{i}"}
        for i, (size, order) in enumerate([("8b", orders[0]), ("30b", orders[0]), ("8b", orders[1]), ("30b", orders[1])], 1)
    ]
    json.dumps(summary)


def test_the_test_sets_prompt_is_used_and_the_runs_own_otherwise(exp, tmp_path):
    source_run(tmp_path)
    spawned = []
    made = exp.make_object(
        job(prompt="", count=1, sizes=["30b"], orders=1), Runtime(), tmp_path, pack=lambda raw, size: raw,
        render=lambda raw: Image.new("RGB", (8, 8)), spawn=lambda size, request: spawned.append(request) or "fc-1",
    )
    assert made["summary"]["candidates"] == ["t9roll0", "t9roll1"] and len(spawned) == 1
    assert spawned[0]["prompt"] == "" and made["summary"]["prompt"] == ""  # a photo run: no prompt anywhere


def test_an_object_that_cant_be_drawn_goes_unjudged(exp, tmp_path):
    source_run(tmp_path)
    spawned = []

    def render(raw):
        if candidate(raw) == 2:
            raise RuntimeError("the mesh has no base colour texture or UVs")
        return Image.new("RGB", (16, 8))

    made = exp.make_object(job(), Runtime(), tmp_path, pack=lambda raw, size: raw, render=render, spawn=lambda *a: spawned.append(a))
    summary = made["summary"]
    assert summary["render_errors"] == {"t9roll2": "RuntimeError: the mesh has no base colour texture or UVs"}
    assert summary["judge_error"] and spawned == [] and "judge_calls" not in summary
    assert f"t9roll2-{SUBJECT}/final-{SEED}.glb" in made["files"]  # the GLBs are kept


def test_the_judges_answers_are_read_by_candidate_name(exp):
    summary = {
        "candidates": ["t9roll0", "t9roll1", "t9roll2"],
        "judge_calls": [
            {"size": "8b", "order": [2, 0, 1], "call": "fc-1"},
            {"size": "30b", "order": [2, 0, 1], "call": "fc-2"},
            {"size": "8b", "order": [1, 0, 2], "call": "fc-3"},
            {"size": "30b", "order": [1, 0, 2], "call": "fc-4"},
        ],
    }
    answers = {
        "fc-1": {
            "verdicts": ["edits", "publish", "reject"], "best": 1, "why": "clean back", "problems": ["smear", "none", "ghost"],
            "raw": "{…}", "seconds": 3.2, "letters": ["L", "M", "K"], "model": "8b", "tokens": {"prompt": 4100, "reply": 140},
        },
        "fc-2": {"error": "judge failed: RuntimeError: CUDA out of memory"},
        "fc-3": {"verdicts": [None, None, None], "best": 0, "why": "", "parse_error": "no JSON object or pick in the reply", "raw": "??"},
    }

    def get(call):
        if call not in answers:
            raise TimeoutError("no result in 1800 s")
        return answers[call]

    exp.collect(summary, get)
    first, second = summary["judges"]["8b"]
    assert first == {
        "order": ["t9roll2", "t9roll0", "t9roll1"],
        "best": "t9roll1",
        "verdicts": {"t9roll0": "edits", "t9roll1": "publish", "t9roll2": "reject"},
        "why": "clean back",
        "problems": {"t9roll0": "smear", "t9roll1": "none", "t9roll2": "ghost"},
        "seconds": 3.2,
        "tokens": {"prompt": 4100, "reply": 140},
        "model": "8b",
        "raw": "{…}",
    }
    assert second["best"] == "t9roll0" and second["parse_error"] == "no JSON object or pick in the reply"
    assert second["problems"] == {"t9roll0": None, "t9roll1": None, "t9roll2": None}
    assert summary["judges"]["30b"] == [
        {"order": ["t9roll2", "t9roll0", "t9roll1"], "error": "judge failed: RuntimeError: CUDA out of memory"},
        {"order": ["t9roll1", "t9roll0", "t9roll2"], "error": "TimeoutError: no result in 1800 s"},
    ]
    assert summary["picks"] == {"8b": ["t9roll1", "t9roll0"], "30b": [None, None]}
    json.dumps(summary)
    # An object that was never judged gets an empty record
    unjudged = {"candidates": ["t9roll0"], "judge_error": "not every candidate could be rendered"}
    exp.collect(unjudged, get)
    assert unjudged["judges"] == {} and unjudged["picks"] == {}
