"""score_picture on pictures drawn like FLUX's product shots: objects on light grey, soft shadows."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from reference_worker import service
from reference_worker.service import STRAIGHT_PENALTY, handle_job, score_picture, straight_on

SIZE = 1024
RED = (196, 52, 44)
TYRE = (34, 32, 36)
GLASS = (70, 80, 96)
LAMP = (250, 236, 200)


def backdrop(size=(SIZE, SIZE), spot=0.0):
    """Light grey, darker at the top like a wall meeting a floor; `spot` adds a lit centre and dark corners."""
    y, x = np.indices((size[1], size[0])) / SIZE
    grey = 210 + 24 * y + spot * (0.35 - np.hypot(x - 0.5, y - 0.45))
    return Image.fromarray(np.repeat(grey[..., None], 3, axis=2).clip(0, 255).astype(np.uint8))


def turned(draw, box, colour):
    """A toy car turned to three quarters, filling `box`: its nose to the left and nearer, its side running back."""
    at, oval = _placing(box)
    body = [(0.0, 0.52), (0.1, 0.4), (0.38, 0.34), (0.52, 0.02), (0.84, 0.0)]  # nose, bonnet, windscreen, roof
    body += [(1.0, 0.3), (1.0, 0.66), (0.8, 0.8), (0.3, 0.92), (0.02, 0.78)]  # the back, then along the bottom
    draw.polygon([at(u, v) for u, v in body], fill=colour)
    draw.polygon([at(0.5, 0.08), at(0.82, 0.06), at(0.93, 0.32), at(0.55, 0.36)], fill=GLASS)  # its side windows
    nose = [(0.0, 0.52), (0.1, 0.4), (0.3, 0.42), (0.26, 0.6), (0.02, 0.7)]
    draw.polygon([at(u, v) for u, v in nose], fill=tuple(int(c * 0.78) for c in colour))  # in shade
    draw.ellipse(oval(0.08, 0.52, 0.04, 0.04), fill=LAMP)
    draw.ellipse(oval(0.22, 0.87, 0.09, 0.13), fill=TYRE)  # the near wheel, bigger than the far one
    draw.ellipse(oval(0.74, 0.8, 0.07, 0.1), fill=TYRE)


def front(draw, box, colour):
    """The same car straight from the front, filling `box`: everything mirrors about its middle."""
    at, oval = _placing(box)
    draw.ellipse(oval(0.17, 0.86, 0.1, 0.14), fill=TYRE)
    draw.ellipse(oval(0.83, 0.86, 0.1, 0.14), fill=TYRE)
    body = [(0.0, 0.5), (0.12, 0.38), (0.24, 0.0), (0.76, 0.0), (0.88, 0.38), (1.0, 0.5), (1.0, 0.86), (0.0, 0.86)]
    draw.polygon([at(u, v) for u, v in body], fill=colour)
    draw.polygon([at(0.28, 0.06), at(0.72, 0.06), at(0.8, 0.36), at(0.2, 0.36)], fill=GLASS)
    for u in (0.16, 0.84):
        draw.ellipse(oval(u, 0.55, 0.07, 0.06), fill=LAMP)
    draw.rectangle((*at(0.34, 0.6), *at(0.66, 0.72)), fill=TYRE)  # the grille


def side(draw, box, colour):
    """A bus square from its side, filling `box`: a row of windows and a wheel at each end."""
    at, oval = _placing(box)
    draw.rounded_rectangle((*at(0.0, 0.0), *at(1.0, 0.84)), radius=int(0.05 * (box[2] - box[0])), fill=colour)
    for u in np.arange(0.06, 0.9, 0.12):
        draw.rectangle((*at(u, 0.12), *at(u + 0.08, 0.4)), fill=GLASS)
    for u in (0.2, 0.8):
        draw.ellipse(oval(u, 0.86, 0.09, 0.14), fill=TYRE)


def _placing(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    def at(u, v):
        return (x0 + u * w, y0 + v * h)

    def oval(u, v, ru, rv):
        return (x0 + (u - ru) * w, y0 + (v - rv) * h, x0 + (u + ru) * w, y0 + (v + rv) * h)

    return at, oval


def studio(*boxes, colour=RED, size=(SIZE, SIZE), spot=0.0, view=turned):
    """Each box (x0, y0, x1, y1) holds an object drawn by `view` (three-quarters unless asked), on a soft shadow."""
    picture = backdrop(size, spot)
    shadow = Image.new("L", size, 0)
    for x0, y0, x1, y1 in boxes:  # falling to the right and a little in front
        w, h = x1 - x0, y1 - y0
        ImageDraw.Draw(shadow).ellipse((x0, y1 - 0.1 * h, x1 + 0.4 * w, y1 + 0.12 * h), fill=80)
    picture = Image.composite(Image.new("RGB", size), picture, shadow.filter(ImageFilter.GaussianBlur(32)))
    draw = ImageDraw.Draw(picture)
    for box in boxes:
        view(draw, box, colour)
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


def test_a_straight_on_view_is_marked_and_scores_lower():
    # Image-to-3D makes up the sides a straight-on view hides, so a three-quarter view goes first
    assert score_picture(studio(LARGE)) == {"score": 1.0, "issues": []}
    straight = {"score": round(1 - STRAIGHT_PENALTY, 3), "issues": ["seen straight on"]}
    assert score_picture(studio(LARGE, view=front)) == straight
    assert score_picture(studio(LARGE, view=side))["issues"] == ["seen straight on"]


@pytest.mark.parametrize("colour", [RED, (248, 248, 246), (28, 28, 30), (150, 150, 152)])
@pytest.mark.parametrize("spot", [0.0, 90.0, -90.0])
def test_light_and_colour_dont_change_the_view(colour, spot):
    assert "seen straight on" not in score_picture(studio(LARGE, colour=colour, spot=spot))["issues"]
    assert "seen straight on" in score_picture(studio(LARGE, colour=colour, spot=spot, view=front))["issues"]


def test_the_view_is_judged_at_any_size_and_place():
    for box in [(312, 312, 712, 712), (40, 40, 420, 420), (452, 440, 572, 580)]:
        assert "seen straight on" not in score_picture(studio(box))["issues"]
        assert "seen straight on" in score_picture(studio(box, view=front))["issues"]
    assert "seen straight on" in score_picture(studio(LARGE, size=(2048, SIZE), view=front))["issues"]


def test_a_turned_picture_is_suggested_over_a_straight_one_that_frames_better():
    pictures = {20: studio(LARGE, view=front), 21: studio((240, 300, 784, 840))}  # 72% and 53% of the picture
    job = {"id": "j", "input": {"prompt": "a toy car", "count": 2, "seed": 20}}
    out = handle_job(job, lambda prompt, seed: pictures[seed], FakeStorage())
    straight, turned_ = out["images"]
    assert straight["issues"] == ["seen straight on"] and turned_["issues"] == []
    assert turned_["score"] > straight["score"]
    # ...but not one that frames its object much worse
    tiny = score_picture(studio((452, 440, 572, 580)))
    assert tiny["score"] < straight["score"]


def test_straight_on_is_how_much_the_object_mirrors_itself():
    rng = np.random.default_rng(7)
    half = rng.uniform(0, 255, (64, 32, 3)).astype(np.float32)
    mirrored = np.concatenate([half, half[:, ::-1]], axis=1)  # a pattern that mirrors exactly about its middle
    lopsided = np.concatenate([half, rng.uniform(0, 255, (64, 32, 3)).astype(np.float32)], axis=1)
    mask = np.ones((64, 64), bool)
    assert straight_on(mirrored, mask) == 1.0
    assert straight_on(lopsided, mask) == 0.0
    assert straight_on(mirrored, np.zeros((64, 64), bool)) == 0.0
    assert straight_on(np.full((64, 64, 3), 90.0, np.float32), mask) == 0.0  # no edges to mirror


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
