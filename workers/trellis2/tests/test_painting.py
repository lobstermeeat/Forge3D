"""The painter in production: the kit, the painter's side on a fake editing model, and the final's hook."""

import base64
import io

import numpy as np
import pytest
import torch
import trimesh
import trimesh.visual
from PIL import Image

from forge3d_worker import paint as P
from forge3d_worker import painting
from forge3d_worker.pipeline import CUTOUT
from forge3d_worker.service import handle_job
from forge3d_worker.settings import PRESETS


def textured_box(colour=(90, 60, 30), size=64) -> trimesh.Trimesh:
    """A box with a textured material, as to_glb's mesh has (UVs, a base colour texture, PBR factors)."""
    box = trimesh.creation.box(extents=(1.0, 0.7, 0.5))
    box = box.subdivide()
    uv = np.stack([(box.vertices[:, 0] + 0.5), (box.vertices[:, 1] + 0.35) / 0.7], 1).clip(0, 1)
    texture = Image.new("RGB", (size, size), colour)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0)
    return trimesh.Trimesh(box.vertices, box.faces, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def cutout(size=48) -> Image.Image:
    image = np.zeros((size, size, 4), np.uint8)
    image[..., :3] = 200
    image[12:36, 8:40] = (40, 160, 220, 255)
    return Image.fromarray(image, "RGBA")


def test_a_kit_carries_the_mesh_the_picture_and_what_the_object_is():
    mesh = textured_box()
    kit = painting.pack_kit(mesh, cutout(), "red sneaker", {"around": 4})
    again, picture, subject, options = painting.unpack_kit(kit)
    np.testing.assert_array_equal(again.vertices, mesh.vertices)
    np.testing.assert_array_equal(again.visual.uv, mesh.visual.uv)
    assert again.visual.material.baseColorTexture.size == (64, 64)
    assert picture.mode == "RGBA" and picture.size == (48, 48)
    assert subject == "red sneaker" and options == {"around": 4}
    with pytest.raises(painting.PaintError):
        painting.pack_kit(mesh, None, "red sneaker")


def flat_editor(colour=(40, 160, 220)):
    """paint_views' Painter that paints the render's object one colour (as the tests of paint.py do)."""
    asked = []

    def editor(subject):
        asked.append(subject)

        def paint_view(render, picture, neighbour, seed, view):
            array = np.asarray(render).astype(int)
            background = np.round(np.asarray(P.BACKGROUND) * 255).astype(int)
            mask = np.abs(array - background).max(-1) > 6
            out = np.empty_like(array)
            out[:] = background
            out[mask] = colour
            return Image.fromarray(out.astype(np.uint8), "RGB")

        return paint_view

    return editor, asked


def test_the_painter_paints_the_kit_at_its_size_and_sends_the_texture_back():
    editor, asked = flat_editor()
    kit = painting.pack_kit(textured_box(), cutout(), "bmw car", {"around": 4, "top": False, "bottom": False, "pixels": 96 * 96})
    out = painting.paint_kit(kit, editor, size=128, device="cpu", log=lambda _: None)
    texture = Image.open(io.BytesIO(out["texture"]))
    assert texture.size == (128, 128) and out["size"] == 128
    assert asked == ["bmw car"]
    assert out["output"] in ("robust", "plain")
    assert out["report"]["accepted"] >= 1
    # The views' colour went in where they saw the box
    assert np.abs(np.asarray(texture).reshape(-1, 3).astype(int) - (40, 160, 220)).sum(-1).min() < 30


class Mesh:
    """What the runtime's generate() returns, as far as the hook reads it: the cut-out picture."""

    def __init__(self, picture):
        setattr(self, CUTOUT, picture)


def test_the_hook_puts_the_painted_texture_in_the_material():
    glb = textured_box(size=32)
    sent = []

    def call(kit):
        sent.append(kit)
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, "PNG")
        return {"texture": buffer.getvalue(), "size": 64, "output": "robust", "seconds": 1.0,
                "report": {"views": [{"accepted": True}, {"accepted": False}], "joint_gains": {"anchor": "picture"}}}

    note = painting.hook(call, "watch")(glb, Mesh(cutout()))
    assert note["applied"] and note["size"] == 64 and note["views"] == 1 and note["of"] == 2 and note["joint"] == "picture"
    assert glb.visual.material.baseColorTexture.size == (64, 64)
    assert glb.visual.material.baseColorTexture.getpixel((5, 5)) == (10, 20, 30)
    assert painting.unpack_kit(sent[0])[2] == "watch"


@pytest.mark.parametrize(
    "answer, reason",
    [
        (RuntimeError("the painter's GPU ran out of memory"), "RuntimeError: the painter's GPU ran out of memory"),
        ({"report": {}}, "not a texture"),
        ({"texture": b"not a png"}, "UnidentifiedImageError"),
    ],
)
def test_a_painter_that_fails_leaves_the_texture_alone(answer, reason):
    glb = textured_box(size=32)
    before = np.asarray(glb.visual.material.baseColorTexture).copy()

    def call(kit):
        if isinstance(answer, Exception):
            raise answer
        return answer

    note = painting.hook(call, "watch")(glb, Mesh(cutout()))
    assert not note["applied"] and reason in note["reason"]
    np.testing.assert_array_equal(np.asarray(glb.visual.material.baseColorTexture), before)


def test_a_final_without_a_cutout_is_not_painted():
    note = painting.hook(lambda kit: pytest.fail("the painter was called"), "watch")(textured_box(), Mesh(None))
    assert not note["applied"] and "cut-out" in note["reason"]


# --- The final's job -----------------------------------------------------------------------------------


def png_bytes(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 80, 40)).save(buf, "PNG")
    return buf.getvalue()


class PaintingRuntime:
    """A fake runtime whose export runs before_projection on a real textured mesh, as Trellis2Runtime's does."""

    def __init__(self, painter=None):
        if painter is not None:
            self.painter = painter
        self.before_projection = None
        self.last_before_projection = None
        self.exported = []

    def generate(self, image, preset, seed):
        return Mesh(cutout())

    def export(self, mesh, preset):
        glb = textured_box(size=32)
        self.last_before_projection = None
        if self.before_projection is not None:
            self.last_before_projection = self.before_projection(glb, mesh)
        self.exported.append(glb.visual.material.baseColorTexture.size)
        return b"raw-glb" * 10, 1000


class Storage:
    def put(self, key, data, content_type):
        return {"key": key, "url": f"https://assets.example.com/{key}"}


def painted_texture(kit):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, "PNG")
    return {"texture": buffer.getvalue(), "size": 64, "report": {"views": [{"accepted": True}]}}


def job(**extra):
    return {"id": "j", "input": {"image_base64": base64.b64encode(png_bytes()).decode(), "seed": 7, **extra}}


def test_a_final_asked_to_be_painted_is_painted_and_packed_at_the_painters_size():
    limits = []
    runtime = PaintingRuntime(painter=painted_texture)
    out = handle_job(job(paint=True, subject="a red Nike sneaker"), runtime, Storage(), lambda raw, limit: limits.append(limit) or raw)
    assert "error" not in out
    assert out["paint"]["applied"] and out["paint"]["views"] == 1
    assert runtime.exported == [(64, 64)] and limits == [painting.SIZE]
    assert runtime.before_projection is None  # only for that export


def test_a_final_not_asked_or_a_preview_is_not_painted():
    limits = []
    runtime = PaintingRuntime(painter=lambda kit: pytest.fail("the painter was called"))
    out = handle_job(job(), runtime, Storage(), lambda raw, limit: limits.append(limit) or raw)
    assert "paint" not in out and limits == [PRESETS["final"].texture_size]
    out = handle_job(job(mode="preview", paint=True), runtime, Storage(), lambda raw, limit: limits.append(limit) or raw)
    assert "paint" not in out and limits[-1] == PRESETS["preview"].texture_size


def test_a_final_on_a_worker_without_a_painter_says_so():
    limits = []
    out = handle_job(job(paint=True), PaintingRuntime(), Storage(), lambda raw, limit: limits.append(limit) or raw)
    assert out["paint"] == {"applied": False, "reason": "this worker has no painter"}
    assert limits == [PRESETS["final"].texture_size]


def test_a_painter_that_fails_keeps_the_final_and_its_size():
    def broken(kit):
        raise TimeoutError("the painter took too long")

    limits = []
    out = handle_job(job(paint=True), PaintingRuntime(painter=broken), Storage(), lambda raw, limit: limits.append(limit) or raw)
    assert "error" not in out and not out["paint"]["applied"] and "took too long" in out["paint"]["reason"]
    assert limits == [PRESETS["final"].texture_size]


@pytest.mark.parametrize(
    "extra, message",
    [({"paint": "yes"}, "paint must be true or false"), ({"paint": True, "subject": 5}, "subject must be a string"),
     ({"paint": True, "subject": "x" * 201}, "at most 200")],
)
def test_bad_paint_requests_are_input_errors(extra, message):
    out = handle_job(job(**extra), PaintingRuntime(painter=painted_texture), Storage(), lambda raw, limit: raw)
    assert message in out["error"]


def test_the_subject_is_tidied_and_defaults_to_object():
    from forge3d_worker.inputs import parse_job

    image = {"image_base64": base64.b64encode(png_bytes()).decode()}
    assert parse_job({**image, "paint": True, "subject": "  a\tred\n sneaker "}, "x").paint == "a red sneaker"
    assert parse_job({**image, "paint": True}, "x").paint == "object"
    assert parse_job({**image, "paint": False}, "x").paint is None
