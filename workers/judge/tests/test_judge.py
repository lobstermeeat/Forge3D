"""judge() with a fake model: what the model is shown, and the reply mapped back to the caller's candidates."""

import json

import pytest
from PIL import Image

from judge_worker import judge as J
from judge_worker import model as M

COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]


class FakeChat:
    """Answers with ``reply`` (a string, or a function of the chat) and records each chat."""

    def __init__(self, reply):
        self.reply = reply
        self.chats = []
        self.last_tokens = {"prompt": 4321, "reply": 87}

    def __call__(self, messages, max_new_tokens=J.MAX_NEW_TOKENS):
        self.chats.append((messages, max_new_tokens))
        return self.reply(messages) if callable(self.reply) else self.reply


def grids(count=4, size=(96, 64)):
    return [Image.new("RGB", size, COLOURS[i]) for i in range(count)]


def picture(size=(64, 64)):
    return Image.new("RGB", size, (250, 250, 250))


def shown_images(chat):
    """The images the model saw, in order: the picture, then the grids."""
    return [item["image"] for item in chat[0]["content"] if item["type"] == "image"]


def shown_letters(chat):
    return [item["text"].strip().removeprefix("Version ").rstrip(":") for item in chat[0]["content"][2::2][:-1]]


def test_candidates_are_shown_in_the_order_asked_and_mapped_back():
    reply = json.dumps(
        {
            "K": {"problems": "a red ghost on the back", "verdict": "reject"},
            "L": {"problems": "olive panels", "verdict": "edits"},
            "M": {"problems": "none", "verdict": "publish"},
            "N": {"problems": "a light smear", "verdict": "edits"},
            "best": "M",
            "why": "M is clean all round.",
        }
    )
    chat = FakeChat(reply)
    result = J.judge(picture(), grids(), "a retro arcade machine", model=chat, order=[2, 0, 3, 1])
    messages, max_new_tokens = chat.chats[0]
    # K is candidate 2, L candidate 0, M candidate 3, N candidate 1
    assert [image.getpixel((0, 0)) for image in shown_images(messages)[1:]] == [COLOURS[2], COLOURS[0], COLOURS[3], COLOURS[1]]
    assert shown_letters(messages) == ["K", "L", "M", "N"]
    assert max_new_tokens == J.MAX_NEW_TOKENS
    assert result["verdicts"] == ["edits", "edits", "reject", "publish"]
    assert result["best"] == 3 and result["why"] == "M is clean all round."
    assert result["letters"] == ["L", "N", "K", "M"]
    assert result["problems"] == ["olive panels", "a light smear", "a red ghost on the back", "none"]
    assert result["raw"] == reply and result["tokens"] == {"prompt": 4321, "reply": 87}
    assert result["seconds"] >= 0 and "parse_error" not in result and "notes" not in result


def test_by_default_the_candidates_go_in_as_given():
    chat = FakeChat('{"K": "edits", "L": "publish", "best": "L", "why": "w"}')
    result = J.judge(picture(), grids(2), "", model=chat)
    assert [image.getpixel((0, 0)) for image in shown_images(chat.chats[0][0])[1:]] == COLOURS[:2]
    assert result["verdicts"] == ["edits", "publish"] and result["best"] == 1 and result["letters"] == ["K", "L"]


@pytest.mark.parametrize("raw", ["I cannot judge these.", '{"K": "publish", "L": "edits"}', '{"best": "Q"}'])
def test_no_valid_pick_falls_back_to_the_generations_own_texture(raw):
    result = J.judge(picture(), grids(2), "a lamp", model=FakeChat(raw), order=[1, 0])
    assert result["best"] == 0 and result["parse_error"] and result["raw"] == raw
    assert "notes" in result or result["verdicts"] != [None, None]


def test_whatever_was_read_is_kept_with_the_parse_error():
    result = J.judge(picture(), grids(2), "a lamp", model=FakeChat('{"K": "publish", "L": "edits"}'), order=[1, 0])
    assert result["verdicts"] == ["edits", "publish"] and result["best"] == 0
    assert result["parse_error"] == "no valid pick in the reply"


def test_notes_say_what_was_off():
    result = J.judge(picture(), grids(2), "a lamp", model=FakeChat("K: publish\nL: edits\nBest: K"))
    assert result["best"] == 0 and "parse_error" not in result
    assert result["notes"] == ["no JSON object in the reply: read from its text"]


def test_images_are_fitted_to_the_models_budget():
    chat = FakeChat('{"K": "publish", "best": "K"}')
    see_through = Image.new("RGBA", (1536, 1024), (0, 0, 0, 0))
    J.judge(Image.new("RGB", (2000, 1500), "white"), [see_through], "x", model=chat)
    shown = shown_images(chat.chats[0][0])
    assert shown[0].size == (768, 576) and shown[0].mode == "RGB"
    assert shown[1].size == (1152, 768) and shown[1].getpixel((5, 5)) == J.UNDERLAY  # transparency on white
    # Small images stay as they are; the area cap holds for odd shapes
    assert J.fit(Image.new("RGB", (300, 200)), J.GRID_SIDE, J.GRID_PIXELS).size == (300, 200)
    width, height = J.fit(Image.new("RGB", (1100, 1100)), J.GRID_SIDE, J.GRID_PIXELS).size
    assert width * height <= J.GRID_PIXELS and abs(width - height) <= 1


def test_bad_arguments_are_refused():
    chat = FakeChat("{}")
    with pytest.raises(ValueError, match="1 to 8 candidates, not 0"):
        J.judge(picture(), [], "x", model=chat)
    with pytest.raises(ValueError, match="1 to 8 candidates, not 9"):
        J.judge(picture(), grids(1) * 9, "x", model=chat)
    for order in ([0, 1, 1], [0, 1], [1, 2, 3]):
        with pytest.raises(ValueError, match="order must list each of the 3 candidates once"):
            J.judge(picture(), grids(3), "x", model=chat, order=order)
    assert chat.chats == []


def test_the_models_own_errors_go_through():
    def broken(messages):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        J.judge(picture(), grids(2), "x", model=FakeChat(broken))


def test_without_a_model_it_uses_the_one_loaded(monkeypatch):
    monkeypatch.setattr(M, "_loaded", None)
    with pytest.raises(RuntimeError, match="no judge model is loaded"):
        J.judge(picture(), grids(2), "x")
    built = []

    class Loaded(FakeChat):
        def __init__(self, which, models_root, **options):
            built.append((which, models_root, options))
            super().__init__('{"K": "edits", "L": "publish", "best": "L"}')

    monkeypatch.setattr(M, "QwenVL", Loaded)
    model = M.load("30b", "/models", device="cuda")
    assert built == [("30b", "/models", {"device": "cuda"})] and M.loaded() is model
    assert J.judge(picture(), grids(2), "x")["best"] == 1 and len(model.chats) == 1
