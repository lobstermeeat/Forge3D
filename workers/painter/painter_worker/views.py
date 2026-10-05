"""
How the painter asks Qwen-Image-Edit-2511 for one view of the model (forge3d_worker/paint.py's ``paint(render, picture,
neighbour, seed, view)``): the render of the model's current texture (Picture 1, its only picture) turned into a
product photo of the same object from the same viewpoint, every part where the render has it.

Phase 8 tried three ways on six objects (ops/exp_paint.py, runs 4 and 5): the render alone, the render with the
picture, and with the picture and the nearest view painted before. Given the picture, the model copied the
picture's viewpoint instead of the render's (most views failed paint_views' outline check); the render alone kept
the viewpoint (silhouette IoU 0.92-0.99) and made clean, photo-real views. The render already carries the
picture's colours where the picture reaches, and the prompt names the picture's main colours.

``editor(painter, subject)`` is that Painter for one object: each render is padded to a square on the renders'
own light grey (the size QwenPainter needs, see qwen.py), painted at the view's seed, and cropped back.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence

from PIL import Image

# paint.py's BACKGROUND (0.92 grey) in 8 bits: what the renders are drawn on
BACKDROP = (235, 235, 235)
PROMPT = (
    "Turn Picture 1, a rough 3D render of a {subject} seen {side}, into a clean, photorealistic studio product "
    "photo of the same {subject} from exactly the same viewpoint. Keep the camera angle, framing, outline, "
    "proportions and position of every part exactly as they are in Picture 1: do not move, add, remove or reshape "
    "anything. Replace the blotchy, smeared surface with clean, crisp, realistic materials and fine details in the "
    "same colours{colours}. Where Picture 1 shows a plain surface, keep it plain: do not invent screens, buttons, "
    "doors, handles, text, logos or patterns that Picture 1 doesn't show. Soft, even, diffused studio lighting from "
    "all around, with no cast shadows and no strong reflections. Plain light grey background."
)
MAX_SUBJECT = 200


def subject_of(prompt: Optional[str]) -> str:
    """What the object is, from the creator's prompt: "make a bmw car m3 model blue" -> "bmw car m3 model blue"."""
    text = " ".join((prompt or "").lower().split())[:MAX_SUBJECT]
    text = re.sub(r"^(please\s+)?(make|create|generate|draw|build|give me|i want)\s+(me\s+)?", "", text)
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s*\b(3d\s+)?model\s+of\s+(a|an|the)\s+", " ", text).strip()
    return text or "object"


def colour_words(names: Sequence[str]) -> str:
    """paint.main_colours' names as the prompt's aside: " (its main colours are steel blue, black and grey)"."""
    names = [str(name) for name in names if name]
    if not names:
        return ""
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f" (its main colours are {listed})"


def prompt_for(subject: str, view: Optional[dict] = None, template: str = PROMPT) -> str:
    """The prompt for one view: ``view`` is paint_views' {"name", "side", "colours"}."""
    view = view or {}
    return template.format(
        subject=subject, side=view.get("side") or "from the front", colours=colour_words(view.get("colours") or [])
    )


def square(image: Image.Image, backdrop: Sequence[int] = BACKDROP) -> Image.Image:
    """``image`` centred on a square of ``backdrop`` (as qwen.from_square expects to crop it back)."""
    image = image.convert("RGB")
    if image.width == image.height:
        return image
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), tuple(backdrop))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def editor(painter: Any, prompt: Optional[str], *, steps: Optional[int] = None, template: str = PROMPT) -> Callable:
    """
    paint_views' Painter for the object ``prompt`` describes (subject_of), on ``painter`` (a QwenPainter). It keeps
    what it asked in ``asked`` (one entry per view: view, seed, prompt, seconds, peak GB).
    """
    from .qwen import from_square

    subject = subject_of(prompt)
    asked: list = []

    def paint_view(render: Image.Image, picture: Any, neighbour: Any, seed: int, view: dict) -> Image.Image:
        text = prompt_for(subject, view, template)
        painted = painter.paint([square(render)], text, seed=int(seed), steps=steps)
        asked.append(
            {"view": (view or {}).get("name"), "seed": int(seed), "prompt": text,
             "seconds": getattr(painter, "last_seconds", None), "peak_gb": getattr(painter, "last_peak_gb", None)}
        )
        return from_square(painted, render.size)

    paint_view.asked = asked  # type: ignore[attr-defined]
    paint_view.subject = subject  # type: ignore[attr-defined]
    return paint_view
