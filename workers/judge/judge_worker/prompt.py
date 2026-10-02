"""
The judge's instructions, in one place: edit the wording here. The code only fills in the placeholders
({prompt}, {versions}, {letter}, {answer}) and puts each image after the text that names it.

The rubric is the one Phase 7's reviewers graded the texture rolls with (phase7/rolls/grader-prompt.md
in the scratchpad), adapted to one object at a time: the versions share one shape and differ only in
texture, and each is shown as judgeviews.turntable draws it (six views, the front at the top left, the
back at the bottom left). Change the view description if the grids are drawn another way.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

# The letters versions are shown under, in the order they are shown: the reviewers' K, L, M and N first.
# No O, which reads like a zero. Their number caps the versions one call can compare
LETTERS = ("K", "L", "M", "N", "P", "Q", "R", "S")

INTRO = """You are a strict reviewer grading AI-generated 3D models for a product's go/no-go decision.

A 3D model was made from the reference picture below{prompt}. It comes in {versions}. All of them have the SAME shape; they differ only in their surface texture (colours, painted detail, and any shading or smears baked into the surface). You don't know how any version was made, and the letters are in no particular order. Judge only from the images.

Each version is one image of six views around it: a 3x2 grid, camera 20 degrees above, plain light, dark background. The top-left view looks at the model's front and the bottom-left view at its back; the other views turn round between them.

"""

# How the user's prompt reads in the intro; a run that started from a photo has none
WITH_PROMPT = ', for this prompt the user typed: "{text}"'
WITHOUT_PROMPT = " (the user started from this picture, without a typed prompt)"

PICTURE = "The reference picture:\n"
VERSION = "\n\nVersion {letter}:\n"

RUBRIC = """

Look at all six views of every version: texture problems often hide on the back and sides, which the picture doesn't show.

The question for each version: would a hobbyist creator (16-24, building 3D scenes and simple games in a browser) put this model into their scene as it stands, without editing it?

Verdicts:
- "publish": yes, as is. Clearly what was asked for, one coherent object (or the group asked for), complete from every angle, surfaces and textures clean at normal viewing distance. Slight softness in tiny details is fine.
- "edits": the right object and mostly good, but with a visible flaw a creator would want fixed first: a smeared or wrongly coloured patch, a messy back side, a fused, missing or broken part, a hole, missing thin parts such as strings or straps, or clearly wrong proportions.
- "reject": not usable: wrong or unrecognisable, several objects merged into a blob, badly broken geometry, or texture failures over much of the model.

The texture flaws reviewers punish most: patches of a colour the object shouldn't have on the sides the picture doesn't show, smears and blotches, a surface that changes colour towards the back (silver turning black, say), a ghost of the front's art copied onto the back, and painted art that comes out washed out or garbled. The shape is the same in every version, so a shape flaw counts the same against each: the textures decide.

Be strict and consistent: a lenient verdict would mislead the founder. Grade each version on its own against the rubric (they can all get the same verdict, or all different ones). Use the picture only to confirm the intended object and spot what got lost; the versions needn't match it exactly. Glass or liquid won't look see-through in these renders; judge whether the object still reads as what was asked.

Then the pick: which one version would that creator rather put in their scene? Name one letter; if two or more are truly equal at the top, name any one of them and say so in "why".

Reply with only this JSON object, no other text:
{answer}
"""

# One version's entry in the answer, and the end of the answer
ANSWER_VERSION = '"{letter}": {"problems": "<the texture flaws you see on it, and where; or none>", "verdict": "<publish, edits or reject>"}'
ANSWER_END = '"best": "<one letter>", "why": "<one plain sentence, under 30 words, naming the texture differences that decided the pick>"'


def _fill(template: str, /, **values: str) -> str:
    """
    The template with each {name} replaced, in one pass (what goes in is never filled again); other
    braces (the answer's JSON) stay as they are.
    """
    return re.sub(r"\{(\w+)\}", lambda match: values.get(match[1], match[0]), template)


def listing(letters: Sequence[str]) -> str:
    """'K', 'K and L', 'K, L, M and N'."""
    letters = list(letters)
    return letters[0] if len(letters) == 1 else f"{', '.join(letters[:-1])} and {letters[-1]}"


def versions_phrase(letters: Sequence[str]) -> str:
    """'1 version, K', '4 versions, K, L, M and N'."""
    count = len(letters)
    return f"{count} version{'' if count == 1 else 's'}, {listing(letters)}"


def answer_template(letters: Sequence[str]) -> str:
    """The JSON object the reply should be, with a placeholder for each value."""
    parts = [_fill(ANSWER_VERSION, letter=letter) for letter in letters] + [ANSWER_END]
    return "{" + ", ".join(parts) + "}"


def prompt_clause(prompt_text: Optional[str]) -> str:
    text = " ".join((prompt_text or "").split())
    return _fill(WITH_PROMPT, text=text) if text else WITHOUT_PROMPT


def messages(picture: Any, versions: Sequence[Any], prompt_text: Optional[str], letters: Sequence[str]) -> list:
    """
    The chat for the model: one user turn, the instructions around the picture and each version's grid (in
    the order given, each under its letter). Images go in as they are (PIL images); transformers' chat
    template puts each where it stands.
    """
    if len(versions) != len(letters):
        raise ValueError(f"{len(versions)} versions but {len(letters)} letters")
    intro = _fill(INTRO, prompt=prompt_clause(prompt_text), versions=versions_phrase(letters))
    content: list = [{"type": "text", "text": intro + PICTURE}, {"type": "image", "image": picture}]
    for letter, image in zip(letters, versions):
        content.append({"type": "text", "text": _fill(VERSION, letter=letter)})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": _fill(RUBRIC, answer=answer_template(letters))})
    return [{"role": "user", "content": content}]
