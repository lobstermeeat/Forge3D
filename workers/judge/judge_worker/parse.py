"""
Reading the judge's reply: a verdict per version and the pick, by the letters the versions were shown under.

The model is asked for one JSON object (prompt.answer_template). Replies stray from it, and each of these
is still read:
- the JSON in a code fence or after a few words, with trailing commas, curly or single quotes, or cut
  short by the token limit (its complete entries are kept);
- a verdict as a plain string ("K": "edits") or nested ("K": {"verdict": "edits"}), under a key such as
  "Version K", "k" or "1" (the first version shown), or inside a "versions" object or list;
- verdicts spelt "Edit", "needs edits" or "Publishable";
- a pick written "L", "Version L", "L (tied with K)" or 2 (the second shown);
- no JSON at all: lines such as "K: publish" and "Best: L" are looked for in the text.
What can't be read is left out and noted. Positions count the versions in the order they were shown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

VERDICTS = ("publish", "edits", "reject")
# Whole answers that mean a verdict; otherwise a verdict is the one of these stems a value contains
SYNONYMS = {
    "publish": "publish",
    "publishable": "publish",
    "published": "publish",
    "yes": "publish",
    "edit": "edits",
    "edits": "edits",
    "needs edit": "edits",
    "needs edits": "edits",
    "needs editing": "edits",
    "reject": "reject",
    "rejected": "reject",
    "no": "reject",
}
STEMS = {"publish": "publish", "edit": "edits", "reject": "reject"}
BEST_KEYS = {"best", "pick", "choice", "winner", "best version", "best pick", "preferred"}
WHY_KEYS = {"why", "reason", "reasoning", "because"}
VERDICT_KEYS = ("verdict", "rating", "grade", "decision", "result")
PROBLEM_KEYS = ("problems", "issues", "flaws", "notes", "problem")
NAME_KEYS = ("letter", "version", "name", "id", "model", "candidate")
# Words a version's name may come with ("Version K", "Model 2")
NAME_WORDS = re.compile(r"^(?:version|model|candidate|option|texture)\b[\s:#.-]*", re.I)


@dataclass
class Reply:
    """What a reply says, by shown position (0 for the first version shown)."""

    verdicts: dict = field(default_factory=dict)  # position -> "publish" | "edits" | "reject"
    problems: dict = field(default_factory=dict)  # position -> what the model saw
    best: Optional[int] = None  # the pick's position; None when the reply names no valid pick
    why: str = ""
    error: Optional[str] = None  # why there is no pick
    notes: list = field(default_factory=list)  # what was off but read anyway


def verdict(value: Any) -> Optional[str]:
    """'publish', 'edits' or 'reject' for a verdict however it's spelt; None when it isn't one (or is two)."""
    if not isinstance(value, str):
        return None
    text = " ".join(re.findall(r"[a-z]+", value.lower()))
    if text in SYNONYMS:
        return SYNONYMS[text]
    found = {verdict for stem, verdict in STEMS.items() if stem in text}
    return found.pop() if len(found) == 1 else None


def position(name: Any, letters: Sequence[str]) -> Optional[int]:
    """The shown position a version's name stands for: its letter ("K", "k", "Version K") or number (1 for the first)."""
    if isinstance(name, bool):
        return None
    if isinstance(name, float) and name.is_integer():
        name = int(name)
    if isinstance(name, int):
        return name - 1 if 1 <= name <= len(letters) else None
    if not isinstance(name, str):
        return None
    text = NAME_WORDS.sub("", name.strip().strip("\"'*`()[]<>").strip()).strip(" \"'*`()[]<>:.-")
    if text.upper() in letters:
        return list(letters).index(text.upper())
    if text.isdigit():
        number = int(text)
        return number - 1 if 1 <= number <= len(letters) else None
    return None


def pick(value: Any, letters: Sequence[str]) -> Optional[int]:
    """The shown position of the pick: a name as ``position`` reads it, or the first capital letter or number in a phrase."""
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = next((value[key] for key in NAME_KEYS if key in value), None)
    found = position(value, letters)
    if found is not None or not isinstance(value, str):
        return found
    # "L (tied with K)", "Version M, because…": the first capital that is a letter shown (a lower-case one
    # is likely a word: the "s" of "it's"), else the first number
    for token in re.findall(r"(?<![A-Za-z0-9])[A-Z](?![A-Za-z0-9])", value):
        if token in letters:
            return list(letters).index(token)
    for token in re.findall(r"\d+", value):
        number = int(token)
        if 1 <= number <= len(letters):
            return number - 1
    return None


def _key(key: Any) -> str:
    return " ".join(re.findall(r"[a-z]+", str(key).lower()))


def _read_version(where: int, value: Any, letters: Sequence[str], reply: Reply) -> None:
    letter = letters[where]
    if isinstance(value, dict):
        named = {_key(key): item for key, item in value.items()}
        said = next((named[key] for key in VERDICT_KEYS if key in named), None)
        problems = next((named[key] for key in PROBLEM_KEYS if key in named), None)
        if problems is not None:
            reply.problems[where] = problems if isinstance(problems, str) else json.dumps(problems)
    else:
        said = value
    found = verdict(said)
    if found is None:
        reply.notes.append(f"no verdict read for {letter} from {json.dumps(said)[:80]}")
    else:
        reply.verdicts[where] = found


def _read_object(found: dict, letters: Sequence[str], reply: Reply, depth: int = 0) -> None:
    for key, value in found.items():
        name = _key(key)
        if name in BEST_KEYS:
            reply.best = pick(value, letters)
            if reply.best is None:
                reply.error = f"the pick {json.dumps(value)[:80]} names no version shown"
            continue
        if name in WHY_KEYS:
            reply.why = value if isinstance(value, str) else json.dumps(value)
            continue
        where = position(key, letters)
        if where is not None:
            _read_version(where, value, letters, reply)
        elif depth < 2 and isinstance(value, dict):
            _read_object(value, letters, reply, depth + 1)  # {"versions": {"K": …}}
        elif depth < 2 and isinstance(value, list):
            for item in value:  # {"versions": [{"letter": "K", "verdict": …}, …]}
                if isinstance(item, dict):
                    where = next((position(item[k], letters) for k in NAME_KEYS if k in item), None)
                    if where is not None:
                        _read_version(where, item, letters, reply)


def _objects(text: str):
    """Each JSON object in the text, outermost ones only, in order."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            yield value
        start = text.find("{", end)


def _cleaned(text: str) -> str:
    """Common slips fixed: curly quotes, trailing commas, Python's literals, single quotes (when there are no double ones)."""
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", text)))
    if '"' not in text:
        text = text.replace("'", '"')
    return text


def _truncated(text: str) -> Optional[dict]:
    """The complete entries of an object the token limit cut short: the text up to an inner '}', closed."""
    start = text.find("{")
    if start < 0:
        return None
    ends = [i for i, char in enumerate(text) if char == "}" and i > start][-64:]
    for end in reversed(ends):
        try:
            value = json.loads(text[start : end + 1] + "}")
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _score(reply: Reply) -> int:
    return (100 if reply.best is not None else 0) + len(reply.verdicts)


def _read_text(text: str, letters: Sequence[str], reply: Reply) -> None:
    """No JSON: "K: publish"-like lines and "best: L"-like phrases (letters in capitals)."""
    words = r"(?i:(publish\w*|needs? edits?|edits?|reject\w*))"
    for where, letter in enumerate(letters):
        match = re.search(rf"(?<![A-Za-z0-9]){letter}(?![A-Za-z0-9])[^\n]{{0,40}}?\b{words}", text)
        if match:
            reply.verdicts[where] = verdict(match[1])
    shown = "".join(letters)
    match = re.search(rf"(?i:\b(?:best|pick|choice|winner|prefer\w*)\b)[^\n]{{0,40}}?(?<![A-Za-z0-9])([{shown}])(?![A-Za-z0-9])", text)
    if match:
        reply.best = list(letters).index(match[1])
    match = re.search(r"(?i:\bwhy\b)\W+([^\n]+)", text)
    if match:
        reply.why = match[1].strip().strip('"')


def parse_reply(text: str, letters: Sequence[str]) -> Reply:
    """
    The verdicts and pick in a reply, for versions shown under ``letters`` (in that order). The object that
    reads best is used when there are several (a reply that repeats the template before answering). Never
    raises: ``error`` says why there is no pick.
    """
    letters = [str(letter).upper() for letter in letters]
    text = text if isinstance(text, str) else ""
    best: Optional[Reply] = None
    for source in (text, _cleaned(text)):
        for found in _objects(source):
            reply = Reply()
            _read_object(found, letters, reply)
            if best is None or _score(reply) >= _score(best):
                best = reply
        if best is not None and _score(best) > 0:
            break
    if best is None or _score(best) == 0:
        cut = _truncated(_cleaned(text))
        if cut is not None:
            reply = Reply(notes=["the reply was cut short: read its complete entries"])
            _read_object(cut, letters, reply)
            if _score(reply) > 0:
                best = reply
    if best is None or _score(best) == 0:
        reply = Reply()
        _read_text(text, letters, reply)
        if _score(reply) > 0:
            reply.notes.insert(0, "no JSON object in the reply: read from its text")
            best = reply
    reply = best if best is not None else Reply()
    for where, letter in enumerate(letters):
        if where not in reply.verdicts and not any(note.startswith(f"no verdict read for {letter} ") for note in reply.notes):
            reply.notes.append(f"no verdict for {letter}")
    if reply.best is None and reply.error is None:
        reply.error = "no valid pick in the reply" if _score(reply) > 0 else "no JSON object or pick in the reply"
    return reply
