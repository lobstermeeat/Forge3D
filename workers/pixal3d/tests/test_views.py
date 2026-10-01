"""A job's views: parsing, framing the picture like the redrawn front, and Pixal3D's bundle."""

import base64
import io
import math

import numpy as np
import pytest
import torch
from PIL import Image

from forge3d_worker.inputs import InputError
from pixal3d_worker import cameras, views


def png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def cutout(size=64, box=(16, 20, 40, 52), colour=(200, 40, 40)) -> Image.Image:
    """A transparent square with an opaque rectangle (x0, y0, x1, y1, exclusive ends)."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (*colour, 255)), box[:2])
    return image


def view(azimuth, elevation=0, image=None):
    return {"image_base64": png(image or cutout()), "azimuth": azimuth, "elevation": elevation}


def test_parse_views():
    parsed = views.parse_views([view(0), view(90), view(-45, 10)])
    assert [(v.azimuth, v.elevation) for v in parsed] == [(0, 0), (90, 0), (315, 10)]
    assert parsed[0].is_front and not parsed[2].is_front
    assert parsed[0].image.mode == "RGBA" and parsed[0].image.size == (64, 64)


def test_parse_views_by_url():
    fetched = []

    def fetch(url):
        fetched.append(url)
        return base64.b64decode(png(cutout()))

    parsed = views.parse_views([{"image_url": "https://r2.example/v0.png", "azimuth": 0}], fetch=fetch)
    assert fetched == ["https://r2.example/v0.png"] and parsed[0].elevation == 0


@pytest.mark.parametrize(
    "raw, message",
    [
        ("nope", "non-empty list"),
        ([], "non-empty list"),
        ([view(a) for a in range(0, 360, 40)], "at most"),
        (["x"], "must be an object"),
        ([{"azimuth": 0}], "exactly one of"),
        ([{**view(0), "image_url": "https://x"}], "exactly one of"),
        ([view("front")], "azimuth must be a number"),
        ([view(True)], "azimuth must be a number"),
        ([view(float("nan"))], "azimuth must be a number"),
        ([view(0, 85)], "elevation"),
        ([view(0), view(360)], "repeats"),
        ([{"image_base64": "%%%", "azimuth": 0}], "not valid base64"),
        ([view(0, image=Image.new("RGBA", (64, 32)))], "square"),
    ],
)
def test_parse_views_refuses(raw, message):
    with pytest.raises(InputError, match=message):
        views.parse_views(raw)


def test_parse_camera():
    assert views.parse_camera(None).half_extent == cameras.MV_ADAPTER_HALF_WIDTH
    # The multiview worker's camera_info(), passed along as it is
    info = {"type": "orthographic", "image_size": 768, "half_extent": 0.6, "distance": 1.8, "up": [0, 0, 1]}
    assert views.parse_camera(info).half_extent == 0.6
    for bad in ("x", {"type": "perspective"}, {"half_extent": 0}, {"half_extent": "wide"}):
        with pytest.raises(InputError):
            views.parse_camera(bad)


def test_object_box_and_edges():
    image = cutout(box=(16, 20, 40, 52))
    assert views.object_box(views.alpha_of(image)) == (16, 20, 40, 52)
    assert views.object_box(np.zeros((4, 4))) is None
    assert not views.touches_edge(image)
    assert views.touches_edge(cutout(box=(0, 20, 40, 52)))


def test_place_like_matches_the_reference_views_object():
    picture = cutout(size=200, box=(10, 30, 110, 80))  # 100 x 50
    reference = cutout(size=64, box=(12, 22, 52, 42))  # 40 x 20, centred at (32, 32)
    placed = views.place_like(picture, 64, reference)
    assert placed.size == (64, 64)
    x0, y0, x1, y1 = views.object_box(views.alpha_of(placed))
    assert (x1 - x0, y1 - y0) == (40, 20)
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == (32, 32)
    assert placed.getpixel((32, 32))[:3] == (200, 40, 40)


def test_place_like_defaults_to_mv_adapters_framing():
    placed = views.place_like(cutout(size=100, box=(0, 0, 50, 25)), 768)
    x0, y0, x1, y1 = views.object_box(views.alpha_of(placed))
    assert x1 - x0 == round(0.9 * 768)  # the longer side at 90 % of the frame
    assert abs((x0 + x1) / 2 - 384) <= 1 and abs((y0 + y1) / 2 - 384) <= 1


def test_place_like_needs_an_object():
    with pytest.raises(InputError):
        views.place_like(Image.new("RGBA", (8, 8)), 8)


def test_order_puts_the_front_first_and_filters():
    parsed = [views.View(cutout(), a, 0) for a in (315, 90, 0, 180, 45, 270)]
    assert [v.azimuth for v in views.order(parsed)] == [0, 45, 90, 180, 270, 315]
    assert [v.azimuth for v in views.order(parsed, [0, 90, 180, 270])] == [0, 90, 180, 270]
    assert [v.azimuth for v in views.order(parsed, [90, 180])] == [90, 180]  # no front: the picture's job


def test_cond_tensor_premultiplies_onto_black():
    image = cutout(size=8, box=(2, 2, 6, 6), colour=(255, 128, 0))
    tensor = views.cond_tensor(image, 8)
    assert tensor.shape == (3, 8, 8)
    assert torch.all(tensor[:, 0, 0] == 0)  # transparent: black
    assert torch.allclose(tensor[:, 4, 4], torch.tensor([1.0, 128 / 255, 0.0]))


def test_bundle_is_run_mvs_layout():
    images = [cutout(), cutout(), cutout()]
    angles = [(0, 0), (90, 0), (180, 0)]
    packed = views.bundle(images, angles, views.ViewCamera())
    assert set(packed["images"]) == {512, 1024}
    assert packed["images"][512].shape == (1, 3, 3, 512, 512)
    assert packed["images"][1024].shape == (1, 3, 3, 1024, 1024)
    assert packed["transform_matrix"].shape == (1, 3, 4, 4)
    distance = cameras.distance_for_half_width(cameras.MV_ADAPTER_HALF_WIDTH, cameras.NEAR_ORTHO_FOV_DEG)
    assert torch.allclose(packed["camera_distance"], torch.full((1, 3), distance))
    assert torch.allclose(packed["camera_angle_x"], torch.full((1, 3), math.radians(cameras.NEAR_ORTHO_FOV_DEG)))
    assert packed["mesh_scale"] == 1.0
    with pytest.raises(ValueError):
        views.bundle(images, angles[:2], views.ViewCamera())


def test_fit_camera_keeps_mv_adapters_framing_when_the_front_is_the_longest_side():
    # The 0-degree view's longer side at 0.9 of the frame (MV-Adapter's fill): 0.99 of a +-0.55 frame
    front = cutout(size=100, box=(30, 5, 70, 95))  # 40 wide, 90 tall
    side = cutout(size=100, box=(35, 5, 65, 95))  # 30 deep
    camera, note = views.fit_camera([front, side, front, side], [(0, 0), (90, 0), (180, 0), (270, 0)], views.ViewCamera())
    assert camera.half_extent == pytest.approx(0.55)
    assert note["extent"] == pytest.approx(0.99) and not note["rescaled"] and "cut_off" not in note


def test_fit_camera_shrinks_an_object_deeper_than_its_front():
    # A car seen from the front: 0.9 of the frame wide, but 1.2 times that deep in the side views
    front = cutout(size=100, box=(5, 30, 95, 70))  # 90 wide, 40 tall
    side = cutout(size=100, box=(0, 30, 100, 70))  # 100 deep: cut off at both edges
    camera, note = views.fit_camera([front, side, front, side], [(0, 0), (90, 0), (180, 0), (270, 0)], views.ViewCamera())
    assert note["extent"] == pytest.approx(1.1) and note["rescaled"] and note["cut_off"] == [90, 270]
    assert camera.half_extent == pytest.approx(0.99 / 2)  # the side views' full frame is now 0.99 of a unit
    # A frame given in other units changes the reported extent, not the rig: the object fills the cube either way
    other, note = views.fit_camera([front, side], [(0, 0), (90, 0)], views.ViewCamera(half_extent=1.0))
    assert other.half_extent == pytest.approx(camera.half_extent) and note["extent"] == pytest.approx(2.0)


def test_fit_camera_ignores_diagonal_widths_and_empty_views():
    front = cutout(size=100, box=(30, 10, 70, 90))  # 40 wide, 80 tall: the height sets the scale
    diagonal = cutout(size=100, box=(0, 10, 100, 90))  # wider than the box's sides, as a diagonal view is
    empty = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    camera, note = views.fit_camera([front, diagonal, empty], [(0, 0), (45, 0), (90, 0)], views.ViewCamera())
    assert camera.half_extent == pytest.approx(0.99 / 1.6) and note["extent"] == pytest.approx(0.88)
    assert note["cut_off"] == [45]  # reported, but its width never counted
    same, note = views.fit_camera([empty], [(0, 0)], views.ViewCamera())
    assert same.half_extent == 0.55 and "extent" not in note
