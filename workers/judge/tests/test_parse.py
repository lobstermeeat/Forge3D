"""Reading the judge's reply however it strays from the JSON it was asked for."""

import json

import pytest

from judge_worker.parse import parse_reply, pick, position, verdict

KLMN = ["K", "L", "M", "N"]


def answer(**entries):
    return json.dumps(entries)


def test_the_answer_as_asked():
    text = answer(
        K={"problems": "a dark smear on the back", "verdict": "edits"},
        L={"problems": "none", "verdict": "publish"},
        M={"problems": "a ghost of the lion on the back", "verdict": "edits"},
        N={"problems": "blotches all over", "verdict": "reject"},
        best="L",
        why="L's back is clean wood; K and M carry smears.",
    )
    reply = parse_reply(text, KLMN)
    assert reply.verdicts == {0: "edits", 1: "publish", 2: "edits", 3: "reject"}
    assert reply.problems == {0: "a dark smear on the back", 1: "none", 2: "a ghost of the lion on the back", 3: "blotches all over"}
    assert reply.best == 1 and reply.why.startswith("L's back") and reply.error is None and reply.notes == []


def test_flat_verdicts_in_a_code_fence_after_some_words():
    text = 'Here is my grading.\n```json\n{"K": "publish", "L": "Edits", "M": "reject", "N": "edits", "best": "K", "why": "clean"}\n```\nThanks!'
    reply = parse_reply(text, KLMN)
    assert reply.verdicts == {0: "publish", 1: "edits", 2: "reject", 3: "edits"} and reply.best == 0 and reply.why == "clean"


def test_versions_named_other_ways():
    text = answer(**{"Version K": "publish", "l": "edits", "3": "reject", "Model N": {"Verdict": "edits"}, "Best": "version m"})
    reply = parse_reply(text, KLMN)
    assert reply.verdicts == {0: "publish", 1: "edits", 2: "reject", 3: "edits"} and reply.best == 2


@pytest.mark.parametrize(
    "said, expected",
    [
        ("publish", "publish"),
        ("Publishable", "publish"),
        ("PUBLISH.", "publish"),
        ("edit", "edits"),
        ("needs edits", "edits"),
        ("Edits (back smear)", "edits"),
        ("reject", "reject"),
        ("Rejected", "reject"),
        ("publish after edits", None),  # two verdicts in one: none
        ("great", None),
        ("<publish, edits or reject>", None),
        (3, None),
        (None, None),
    ],
)
def test_verdict_spellings(said, expected):
    assert verdict(said) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("L", 1),
        ("l", 1),
        ("Version M", 2),
        ("**N**", 3),
        ("M (tied with K)", 2),
        ("I'd pick L, it's cleaner", 1),  # the capital L, not the lower-case s of "it's"
        (2, 1),
        ("2", 1),
        (4.0, 3),
        (["K", "L"], 0),
        ({"letter": "N"}, 3),
        ("Z", None),
        (5, None),
        (0, None),
        (True, None),
        ("none", None),
        ("", None),
    ],
)
def test_picks_by_letter_or_number(value, expected):
    assert pick(value, KLMN) == expected


def test_a_name_counts_the_versions_shown_from_one():
    assert position("1", KLMN) == 0 and position(4, KLMN) == 3 and position("5", KLMN) is None
    assert position("K", ["K", "L"]) == 0 and position("M", ["K", "L"]) is None


def test_a_pick_of_a_version_not_shown_is_no_pick():
    reply = parse_reply(answer(K="publish", L="edits", best="N"), ["K", "L"])
    assert reply.best is None and reply.verdicts == {0: "publish", 1: "edits"}
    assert reply.error == 'the pick "N" names no version shown'


def test_versions_inside_a_container():
    nested = answer(versions={"K": {"verdict": "edits"}, "L": {"verdict": "publish"}}, best="L", why="w")
    listed = answer(versions=[{"letter": "K", "verdict": "edits"}, {"letter": "L", "verdict": "publish"}], best="L")
    for text in (nested, listed):
        reply = parse_reply(text, ["K", "L"])
        assert reply.verdicts == {0: "edits", 1: "publish"} and reply.best == 1


def test_common_slips_are_forgiven():
    trailing = '{"K": "publish", "L": "edits", "best": "K",}'
    curly = "{“K”: “publish”, “L”: “edits”, “best”: “L”}"
    single = "{'K': 'reject', 'L': 'publish', 'best': 'L'}"
    assert parse_reply(trailing, ["K", "L"]).best == 0
    assert parse_reply(curly, ["K", "L"]).best == 1
    reply = parse_reply(single, ["K", "L"])
    assert reply.verdicts == {0: "reject", 1: "publish"} and reply.best == 1


def test_a_reply_cut_short_keeps_its_complete_entries():
    text = '{"K": {"problems": "none", "verdict": "publish"}, "L": {"problems": "a smear on the bac'
    reply = parse_reply(text, ["K", "L"])
    assert reply.verdicts == {0: "publish"} and reply.best is None
    assert reply.error == "no valid pick in the reply"
    assert "the reply was cut short: read its complete entries" in reply.notes and "no verdict for L" in reply.notes


def test_the_object_that_reads_best_wins():
    template = '{"K": {"problems": "<…>", "verdict": "<publish, edits or reject>"}, "best": "<one letter>"}'
    text = f"The format is {template}. My answer: " + answer(K={"problems": "none", "verdict": "publish"}, best="K")
    reply = parse_reply(text, ["K"])
    assert reply.verdicts == {0: "publish"} and reply.best == 0 and reply.error is None


def test_without_json_the_text_is_read():
    text = "K: publish\nL - needs edits (the back is smeared)\nM: reject\nBest: K\nWhy: the cleanest back."
    reply = parse_reply(text, ["K", "L", "M"])
    assert reply.verdicts == {0: "publish", 1: "edits", 2: "reject"} and reply.best == 0
    assert reply.why == "the cleanest back." and reply.notes[0] == "no JSON object in the reply: read from its text"


@pytest.mark.parametrize("text", ["", "I can't see the images.", None, "{not json", '{"note": "nothing useful"}'])
def test_a_reply_with_nothing_to_read(text):
    reply = parse_reply(text, ["K", "L"])
    assert reply.best is None and reply.verdicts == {}
    assert reply.error == "no JSON object or pick in the reply"
    assert reply.notes == ["no verdict for K", "no verdict for L"]


def test_unreadable_verdicts_are_noted():
    reply = parse_reply(answer(K="fine", L={"problems": "x"}, best="K"), ["K", "L"])
    assert reply.verdicts == {} and reply.best == 0 and reply.problems == {1: "x"}
    assert reply.notes == ['no verdict read for K from "fine"', "no verdict read for L from null"]
