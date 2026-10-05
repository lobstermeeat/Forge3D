"""
The painter in production (Phase 8): a final's texture painted again from views round the model.

TRELLIS.2 gets the shape right but its texture is soft and blotchy, pale where the picture is vivid, and made up
on the sides the picture doesn't show. paint.py turns renders of the model into product photos from ten sides
with an image-editing model (Qwen-Image-Edit-2511 on a GPU of its own: workers/painter) and bakes them back. In
production it runs inside a final's export, as Trellis2Runtime.before_projection (``hook``): to_glb's mesh, the
picture cut out and what the object is go to the painter as one ``kit``; the texture that comes back goes into
the mesh's material; then the picture's projection, the shading normals, the glass and the packing run as for
any final, the texture packed up to SIZE.

The painter's side is ``paint_kit``: the mesh's base colour scaled up to SIZE (the same UV layout), painted, and
sent back as a PNG with paint_views' report.

A painter that fails costs the final nothing but the time it took: the final keeps TRELLIS.2's texture, and its
"paint" note says why.
"""

from __future__ import annotations

import io
import json
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

# The texture the views are baked into and shipped at: twice the final's 2048 each way. Phase 8's lab: the views
# have about 630 pixels per unit of the model where a 2048 atlas of TRELLIS.2's layout has 360 texels, so at
# 2048 the bake loses detail the views drew (wheel spokes, grille slats); at 4096 (720) it keeps it
SIZE = 4096
# paint_views' options in production (its defaults otherwise: glare out, tone held to the picture where it sees the
# surface face on, each texel mostly from its best view, a dark bottom left alone). Three tries a view, the second
# and third started from the render noised part way (views.SKIPS): from pure noise the editing model turned half of
# run 7's sneaker, watch and controller views to a catalogue angle; started part way it keeps the render's
OPTIONS: dict = {"attempts": 3}
# Which of paint_views' textures is shipped: "robust" (views that disagree with the others at a texel left out,
# so a highlight one view drew doesn't go in) or "plain"
OUTPUT = "robust"
KIT_VERSION = 1


class PaintError(Exception):
    """The painter couldn't paint this kit (the final keeps its own texture)."""


def pack_kit(glb: Any, cutout: Any, subject: str, options: Optional[dict] = None) -> bytes:
    """
    What the painter needs as bytes: to_glb's textured mesh (paint.pack_mesh), the picture cut out (RGBA PNG),
    what the object is (``subject``: the creator's words, or "object") and paint_views' options.
    """
    from . import paint

    if cutout is None:
        raise PaintError("the final has no cut-out picture")
    image = cutout if isinstance(cutout, Image.Image) else Image.fromarray(np.asarray(cutout))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    arrays = {
        "version": np.asarray(KIT_VERSION),
        "mesh": np.frombuffer(paint.pack_mesh(glb), np.uint8),
        "cutout": np.frombuffer(buffer.getvalue(), np.uint8),
        "subject": np.asarray(str(subject or "object")),
        "options": np.asarray(json.dumps(options or {})),
    }
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def unpack_kit(data: bytes) -> tuple[Any, Image.Image, str, dict]:
    """``pack_kit``'s mesh (a trimesh, as it went in), cutout, subject and options."""
    from . import paint

    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        version = int(arrays["version"])
        if version != KIT_VERSION:
            raise PaintError(f"a kit of version {version}; this painter reads {KIT_VERSION}")
        mesh = paint.unpack_mesh(arrays["mesh"].tobytes())
        cutout = Image.open(io.BytesIO(arrays["cutout"].tobytes()))
        cutout.load()
        subject = str(arrays["subject"])
        options = json.loads(str(arrays["options"]))
    return mesh, cutout, subject, options


def paint_kit(
    data: bytes,
    editor: Callable[[str], Any],
    *,
    size: int = SIZE,
    output: str = OUTPUT,
    device: Any = "cuda",
    log: Callable[[str], None] = print,
) -> dict:
    """
    The painter's side: ``data`` (pack_kit's) painted by ``editor(subject)`` (paint_views' Painter for that
    object), its base colour first scaled up to ``size`` on the same layout. Returns ``{"texture": PNG bytes,
    "size": side, "output": which texture, "report": paint_views' report, "seconds": ...}``.
    """
    from . import paint

    started = time.perf_counter()
    mesh, cutout, subject, options = unpack_kit(data)
    material = mesh.visual.material
    base = material.baseColorTexture
    if base is None:
        raise PaintError("the mesh has no base colour texture")
    if max(base.size) < size:
        material.baseColorTexture = base.resize((size, size), Image.Resampling.LANCZOS)
    result = paint.paint_views(mesh, cutout, editor(subject), device=device, log=log, **{**OPTIONS, **options})
    texture = result.robust if output == "robust" and result.robust is not None else result.texture
    buffer = io.BytesIO()
    texture.convert("RGB").save(buffer, "PNG", compress_level=1)
    return {
        "texture": buffer.getvalue(),
        "size": texture.size[0],
        "output": "robust" if texture is result.robust else "plain",
        "report": result.report,
        "seconds": round(time.perf_counter() - started, 1),
    }


def hook(call: Callable[[bytes], dict], subject: str, options: Optional[dict] = None) -> Callable[[Any, Any], dict]:
    """
    Trellis2Runtime.before_projection for one final: sends to_glb's mesh and the generated mesh's cutout to
    ``call`` (the painter, paint_kit on its own GPU) and puts the texture that comes back in the mesh's material.
    Returns the final's "paint" note: ``{"applied": True, "size", "views", "seconds", ...}``, or ``{"applied":
    False, "reason"}`` when the painter failed or kept none of its views, which leaves the texture as it was.
    """
    from .pipeline import CUTOUT

    def before_projection(glb: Any, mesh: Any) -> dict:
        started = time.perf_counter()
        try:
            kit = pack_kit(glb, getattr(mesh, CUTOUT, None), subject, options)
            painted = call(kit)
            if not isinstance(painted, dict) or "texture" not in painted:
                raise PaintError(f"the painter answered {type(painted).__name__}, not a texture")
            views = (painted.get("report") or {}).get("views") or []
            if not any(view.get("accepted") for view in views):
                # paint_views then hands back TRELLIS.2's own texture, only scaled up: the final keeps its own,
                # at its own size, and the server offers it texture options as for any unpainted final
                kept_none = f"the painter kept none of its {len(views)} views"
                note = {
                    "applied": False,
                    "reason": kept_none if views else "the painter reported no views",
                    "views": 0,
                    "of": len(views),
                    "painter_s": painted.get("seconds"),
                    "seconds": round(time.perf_counter() - started, 1),
                }
                print(f"[forge3d] paint: not applied, {json.dumps(note)}")
                return note
            texture = Image.open(io.BytesIO(painted["texture"]))
            texture.load()
            material = glb.visual.material
            old = material.baseColorTexture
            if old is not None and texture.size[0] * old.size[1] != texture.size[1] * old.size[0]:
                raise PaintError(f"the painted texture is {texture.size}, the final's {old.size}")
            material.baseColorTexture = texture.convert("RGB")
        except Exception as err:  # noqa: BLE001 - the final goes on with its own texture
            traceback.print_exc()
            reason = f"{type(err).__name__}: {err}".splitlines()[0][:300]
            print(f"[forge3d] paint: not applied, {reason}")
            return {"applied": False, "reason": reason, "seconds": round(time.perf_counter() - started, 1)}
        report = painted.get("report") or {}
        views = report.get("views") or []
        note = {
            "applied": True,
            "size": int(painted.get("size") or texture.size[0]),
            "output": painted.get("output"),
            "views": sum(1 for view in views if view.get("accepted")),
            "of": len(views),
            "joint": (report.get("joint_gains") or {}).get("anchor"),
            "painter_s": painted.get("seconds"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"[forge3d] paint: {json.dumps(note)}")
        return note

    return before_projection
