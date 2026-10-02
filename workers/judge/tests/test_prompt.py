"""The judge's chat: the instructions around the picture and the lettered grids, every placeholder filled."""

import json
import re

import pytest
from PIL import Image

from judge_worker import parse, prompt

PLACEHOLDER = re.compile(r"\{(prompt|versions|letter|answer|text)\}")


def colour(name, size=(12, 8)):
    return Image.new("RGB", size, name)


def texts(chat):
    return [item["text"] for item in chat[0]["content"] if item["type"] == "text"]


def test_each_image_comes_after_the_text_that_names_it():
    picture = colour("red", (8, 8))
    grids = [colour("blue"), colour("green"), colour("white")]
    chat = prompt.messages(picture, grids, "a retro arcade machine", ["K", "L", "M"])
    assert len(chat) == 1 and chat[0]["role"] == "user"
    content = chat[0]["content"]
    assert [item["type"] for item in content] == ["text", "image"] * 4 + ["text"]
    assert content[1]["image"] is picture
    assert [content[i]["image"] for i in (3, 5, 7)] == grids
    assert content[0]["text"].endswith("The reference picture:\n")
    assert [content[i]["text"] for i in (2, 4, 6)] == ["\n\nVersion K:\n", "\n\nVersion L:\n", "\n\nVersion M:\n"]


def test_the_intro_names_the_prompt_the_versions_and_the_views():
    chat = prompt.messages(colour("red"), [colour("blue")] * 4, "a retro arcade machine", list("KLMN"))
    intro = texts(chat)[0]
    assert 'for this prompt the user typed: "a retro arcade machine"' in intro
    assert "It comes in 4 versions, K, L, M and N." in intro
    assert "the SAME shape" in intro and "surface texture" in intro
    # The grids as judgeviews.turntable draws them
    assert "3x2 grid, camera 20 degrees above" in intro and "top-left view looks at the model's front" in intro
    rubric = texts(chat)[-1]
    for verdict in parse.VERDICTS:
        assert f'- "{verdict}":' in rubric
    assert "texture problems often hide on the back and sides" in rubric
    assert rubric.rstrip().endswith(prompt.answer_template("KLMN"))
    for text in texts(chat):
        assert not PLACEHOLDER.search(text), text


@pytest.mark.parametrize("typed", [None, "", "   "])
def test_a_photo_run_has_no_prompt(typed):
    intro = texts(prompt.messages(colour("red"), [colour("blue")], typed, ["K"]))[0]
    assert "without a typed prompt" in intro and "the user typed:" not in intro
    assert "It comes in 1 version, K." in intro


def test_the_users_prompt_goes_in_once_and_as_it_is():
    typed = 'a mug that says "{count}"\nwith  {letter} on it'
    intro = texts(prompt.messages(colour("red"), [colour("blue")] * 2, typed, ["K", "L"]))[0]
    assert 'typed: "a mug that says "{count}" with {letter} on it"' in intro  # whitespace folded, not filled


def test_the_answer_template_is_json_with_a_slot_for_each_version():
    template = prompt.answer_template(["K", "L", "M"])
    answer = json.loads(template)
    assert list(answer) == ["K", "L", "M", "best", "why"]
    assert all(set(answer[letter]) == {"problems", "verdict"} for letter in "KLM")
    # A reply that only echoes it reads as no verdicts and no pick
    echoed = parse.parse_reply(template, ["K", "L", "M"])
    assert echoed.verdicts == {} and echoed.best is None


def test_letters_and_listings():
    assert prompt.LETTERS[:4] == ("K", "L", "M", "N") and "O" not in prompt.LETTERS
    assert len(set(prompt.LETTERS)) == len(prompt.LETTERS) == 8
    assert prompt.listing(["K"]) == "K" and prompt.listing(["K", "L"]) == "K and L"
    assert prompt.listing(list("KLMN")) == "K, L, M and N"
    with pytest.raises(ValueError, match="2 versions but 1 letters"):
        prompt.messages(colour("red"), [colour("blue")] * 2, "x", ["K"])
