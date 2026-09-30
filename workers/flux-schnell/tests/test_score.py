"""score_picture on pictures drawn like FLUX's product shots: objects on light grey, soft shadows."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from reference_worker import service
from reference_worker.service import handle_job, score_picture

SIZE = 1024
RED = (196, 52, 44)


def backdrop(size=(SIZE, SIZE), spot=0.0):
    """Light grey, darker at the top like a wall meeting a floor; `spot` adds a lit centre and dark corners."""
    y, x = np.indices((size[1], size[0])) / SIZE
    grey = 210 + 24 * y + spot * (0.35 - np.hypot(x - 0.5, y - 0.45))
    return Image.fromarray(np.repeat(grey[..., None], 3, axis=2).clip(0, 255).astype(np.uint8))


def studio(*boxes, colour=RED, size=(SIZE, SIZE), spot=0.0):
    """Each box (x0, y0, x1, y1) as a rounded object with a stripe of detail, on a soft shadow."""
    picture = backdrop(size, spot)
    shadow = Image.new("L", size, 0)
    for x0, y0, x1, y1 in boxes:  # falling to the right and a little in front
        w, h = x1 - x0, y1 - y0
        ImageDraw.Draw(shadow).ellipse((x0, y1 - 0.1 * h, x1 + 0.4 * w, y1 + 0.12 * h), fill=80)
    picture = Image.composite(Image.new("RGB", size), picture, shadow.filter(ImageFilter.GaussianBlur(32)))
    draw = ImageDraw.Draw(picture)
    for x0, y0, x1, y1 in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=(x1 - x0) // 5, fill=colour)
        middle = (y0 + y1) // 2
        draw.rectangle((x0 + (x1 - x0) // 5, middle - 12, x1 - (x1 - x0) // 5, middle + 12), fill=(70, 40, 36))
    return picture


LARGE = (170, 150, 854, 880)  # 72% of the picture tall, centred


def test_one_large_centred_object_scores_best():
    result = score_picture(studio(LARGE))
    assert result == {"score": 1.0, "issues": []}
    json.dumps(result)  # plain floats and strings, ready for the job's output


def test_small_cut_off_and_split_pictures_score_lower():
    best = score_picture(studio(LARGE))["score"]
    tiny = score_picture(studio((452, 440, 572, 580)))
    cut = score_picture(studio((420, 150, 1104, 880)))  # runs off the right edge
    split = score_picture(studio((110, 300, 450, 720), (590, 300, 930, 720)))
    corner = score_picture(studio((40, 40, 420, 420)))

    assert tiny["issues"] == ["small in the frame"] and tiny["score"] < 0.1
    assert "cut off at the edge" in cut["issues"]
    assert split["issues"] == ["several separate objects"]
    assert corner["issues"] == ["off-centre"]
    for worse in (tiny, cut, split, corner):
        assert worse["score"] < best - 0.2


def test_bigger_is_better_until_the_object_fills_the_frame():
    scores = [score_picture(studio((512 - r, 512 - r, 512 + r, 512 + r)))["score"] for r in (100, 200, 300, 370)]
    assert scores[0] < scores[1] < scores[2] < scores[3] == 1.0  # 20%, 39%, 59%, 72% of the picture
    edge_to_edge = score_picture(studio((12, 12, 1012, 1012)))
    assert edge_to_edge["issues"] == ["fills the frame edge to edge"] and edge_to_edge["score"] < 0.8


def test_studio_light_is_background():
    # A lit centre, dark corners and a wall darker than the floor are all smooth, so none of it
    # is mistaken for the object, and soft shadows that reach the edge don't cut it off
    for spot in (0.0, 90.0, -90.0):
        assert score_picture(studio(LARGE, spot=spot)) == {"score": 1.0, "issues": []}
    assert score_picture(studio((300, 420, 900, 880)))["issues"] == []  # its shadow runs off the right


def test_white_and_dark_objects_are_found_on_light_grey():
    for colour in [(248, 248, 246), (28, 28, 30), (150, 150, 152)]:
        assert score_picture(studio(LARGE, colour=colour)) == {"score": 1.0, "issues": []}


def test_a_part_floating_apart_is_a_separate_object_but_a_faint_gap_is_not():
    def bowl_with_sticks(gap):
        picture = studio((250, 480, 774, 800))
        draw = ImageDraw.Draw(picture)
        for start, end in ((400, 640), (640, 400)):  # crossed chopsticks above the bowl
            draw.line((start, 480 - gap, end, 480 - gap - 300), fill=(120, 72, 40), width=20)
        return score_picture(picture)

    assert "several separate objects" in bowl_with_sticks(gap=90)["issues"]
    assert "several separate objects" not in bowl_with_sticks(gap=10)["issues"]


def test_pictures_with_nothing_in_them():
    assert score_picture(backdrop()) == {"score": 0.0, "issues": ["no clear object"]}
    assert score_picture(Image.new("RGB", (8, 8), (40, 0, 0)))["issues"] == ["no clear object"]
    assert score_picture(Image.new("RGB", (1, 1)))["score"] == 0.0


def test_any_size_and_mode_is_measured_the_same():
    small = studio(LARGE).resize((300, 300))
    assert score_picture(small) == {"score": 1.0, "issues": []}
    wide = score_picture(studio(LARGE, size=(2048, SIZE)))  # the object sits in the left half
    assert wide["issues"] == ["off-centre"] and wide["score"] < 0.9
    assert score_picture(studio(LARGE).convert("RGBA")) == {"score": 1.0, "issues": []}


class FakeStorage:
    def put(self, key, data, content_type):
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def test_jobs_score_every_picture():
    pictures = {10: studio(LARGE), 11: studio((452, 440, 572, 580))}
    job = {"id": "j", "input": {"prompt": "a mug", "count": 2, "seed": 10}}
    out = handle_job(job, lambda prompt, seed: pictures[seed], FakeStorage())

    good, tiny = out["images"]
    assert set(good) == {"key", "url", "seed", "score", "issues"}
    assert (good["seed"], good["score"], good["issues"]) == (10, 1.0, [])
    assert tiny["issues"] == ["small in the frame"] and tiny["score"] < good["score"]
    json.dumps(out)


def test_a_scoring_bug_never_costs_the_pictures(monkeypatch, capsys):
    def broken(image):
        raise ValueError("unexpected picture")

    monkeypatch.setattr(service, "score_picture", broken)
    out = handle_job({"id": "j", "input": {"prompt": "a mug", "count": 2}}, lambda p, s: studio(LARGE), FakeStorage())
    assert "error" not in out
    assert [set(image) for image in out["images"]] == [{"key", "url", "seed"}] * 2
    assert "ValueError: unexpected picture" in capsys.readouterr().err
