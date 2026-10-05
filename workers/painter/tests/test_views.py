"""How a view is asked for: the subject, the prompt, the square and the editor round a stand-in painter."""

from PIL import Image

from painter_worker import views


def test_the_subject_is_what_the_object_is():
    assert views.subject_of("make a bmw car m3 model blue") == "bmw car m3 model blue"
    assert views.subject_of("Please create me an iPhone 15 Pro") == "iphone 15 pro"
    assert views.subject_of("a 3d model of a red Nike Air Jordan 1") == "red nike air jordan 1"
    assert views.subject_of("") == "object" and views.subject_of(None) == "object"
    assert len(views.subject_of("x" * 500)) <= views.MAX_SUBJECT


def test_the_prompt_names_the_side_and_the_colours():
    text = views.prompt_for("bmw car", {"name": "a045", "side": "from its left side", "colours": ["steel blue", "black", "grey"]})
    assert "a rough 3D render of a bmw car seen from its left side" in text
    assert "(its main colours are steel blue, black and grey)" in text
    assert "Picture 1" in text and "Picture 2" not in text
    assert views.colour_words([]) == "" and views.colour_words(["red"]) == " (its main colours are red)"
    assert "seen from the front" in views.prompt_for("mug")


def test_a_render_is_squared_on_the_backdrop_and_cropped_back():
    render = Image.new("RGB", (120, 60), (10, 200, 30))
    squared = views.square(render)
    assert squared.size == (120, 120)
    assert squared.getpixel((60, 5)) == views.BACKDROP and squared.getpixel((60, 60)) == (10, 200, 30)
    assert views.square(Image.new("RGB", (64, 64))).size == (64, 64)


class StandIn:
    """QwenPainter's paint(): returns its first picture as painted, at 1024 x 1024 as the real one does."""

    def __init__(self):
        self.calls = []
        self.last_seconds, self.last_peak_gb = 4.2, 61.0

    def paint(self, images, prompt, *, seed, steps=None, **options):
        self.calls.append({"images": [im.size for im in images], "prompt": prompt, "seed": seed, "steps": steps, **options})
        return images[0].resize((1024, 1024))


def test_the_editor_paints_each_view_at_its_seed_and_keeps_what_it_asked():
    painter = StandIn()
    paint_view = views.editor(painter, "make a yellow Lamborghini Huracan", steps=8)
    render = Image.new("RGB", (160, 80), (200, 20, 20))
    out = paint_view(render, None, None, 300, {"name": "a090", "side": "from its right side", "colours": ["yellow"]})
    assert out.size == (160, 80) and out.getpixel((80, 40)) == (200, 20, 20)
    assert painter.calls == [
        {"images": [(160, 160)], "prompt": painter.calls[0]["prompt"], "seed": 300, "steps": 8}
    ]
    assert "yellow lamborghini huracan seen from its right side" in painter.calls[0]["prompt"]
    assert paint_view.subject == "yellow lamborghini huracan"
    assert paint_view.asked == [
        {"view": "a090", "attempt": 0, "skip": 0, "seed": 300, "prompt": painter.calls[0]["prompt"],
         "seconds": 4.2, "peak_gb": 61.0}
    ]
    assert out.info["skip"] == 0


def test_later_tries_start_part_way():
    painter = StandIn()
    paint_view = views.editor(painter, "a red Nike Air Jordan 1 sneaker")
    render = Image.new("RGB", (64, 64), (200, 20, 20))
    skips = []
    for attempt in range(4):
        out = paint_view(render, None, None, 10 + attempt, {"name": "a000", "attempt": attempt})
        skips.append(out.info["skip"])
    # The first from pure noise (no skip asked for), then further in; past the last, the last again
    assert skips == [0, 1, 2, 2] and list(views.SKIPS) == [0, 1, 2]
    assert "skip" not in painter.calls[0] and [call.get("skip") for call in painter.calls[1:]] == [1, 2, 2]
    assert [entry["skip"] for entry in paint_view.asked] == [0, 1, 2, 2]
    # A view without "attempt" is a first try; skips=(0,) never starts part way
    plain = views.editor(painter, "mug", skips=(0,))
    assert plain(render, None, None, 1, {"name": "a000", "attempt": 2}).info["skip"] == 0
    assert paint_view(render, None, None, 1, {"name": "top"}).info["skip"] == 0
