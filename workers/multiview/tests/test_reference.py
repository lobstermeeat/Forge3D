"""Preparing the picture MV-Adapter is conditioned on: cutout, crop, scale, centre, gray background."""

import numpy as np
import pytest
from PIL import Image

from multiview_worker.generator import MAX_CUTOUT_SIDE, check_adapter_weights, cut_out, has_transparency, prepare_reference
from multiview_worker.inputs import InputError


def cutout(size=(1024, 1024), box=(400, 300, 600, 700), colour=(200, 40, 40)):
    """A transparent picture with one opaque rectangle (x0, y0, x1, y1)."""
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    image.paste((*colour, 255), box)
    return image


def object_box(reference, background=127):
    pixels = np.asarray(reference).astype(int)
    found = np.abs(pixels - background).sum(axis=2) > 6
    ys, xs = np.nonzero(found)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_the_object_is_centred_with_its_longer_side_at_90_percent_on_mid_gray():
    reference = prepare_reference(cutout())
    assert reference.size == (768, 768) and reference.mode == "RGB"
    assert reference.getpixel((5, 5)) == (127, 127, 127)  # 0.5 gray, as MV-Adapter was trained
    x0, y0, x1, y1 = object_box(reference)
    assert y1 - y0 == pytest.approx(691, abs=2)  # 0.9 * 768: the box is taller than wide
    # 200 x 400 (+1 px margin top and left) -> width about half the height
    assert x1 - x0 == pytest.approx(691 * 201 / 401, abs=3)
    assert (x0 + x1) / 2 == pytest.approx(384, abs=2) and (y0 + y1) / 2 == pytest.approx(384, abs=2)
    assert reference.getpixel((384, 384)) == (200, 40, 40)


def test_wide_objects_are_fitted_by_their_width():
    reference = prepare_reference(cutout(box=(100, 450, 900, 550)))
    x0, y0, x1, y1 = object_box(reference)
    assert x1 - x0 == pytest.approx(691, abs=2) and y1 - y0 < 100


def test_soft_edges_are_blended_into_the_gray():
    image = cutout()
    pixels = np.array(image)
    pixels[300:700, 400:600, 3] = 128  # half transparent everywhere
    reference = prepare_reference(Image.fromarray(pixels))
    red, green, _ = reference.getpixel((384, 384))
    assert 160 <= red <= 165 and 80 <= green <= 85  # about halfway between the colour and the gray


def test_the_fill_can_be_changed():
    reference = prepare_reference(cutout(), fill=0.75)
    x0, y0, x1, y1 = object_box(reference)
    assert y1 - y0 == pytest.approx(576, abs=2)


def test_an_empty_cutout_is_an_input_error():
    with pytest.raises(InputError, match="no object found"):
        prepare_reference(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))


def test_a_picture_with_its_own_transparency_skips_background_removal():
    calls = []

    def remover(image):
        calls.append(image.size)
        return image.convert("RGBA")

    picture = cutout()
    assert has_transparency(picture)
    assert cut_out(picture, remover) is picture and calls == []
    # Fully opaque RGBA counts as no transparency, like TRELLIS.2's preprocess_image
    opaque = Image.new("RGBA", (300, 200), (10, 20, 30, 255))
    assert not has_transparency(opaque)
    cut_out(opaque, remover)
    assert calls == [(300, 200)]


def test_large_pictures_are_scaled_down_before_background_removal():
    sizes = []

    def remover(image):
        sizes.append((image.size, image.mode))
        return image.convert("RGBA")

    cut_out(Image.new("RGB", (3000, 1500), (90, 90, 90)), remover)
    assert sizes == [((MAX_CUTOUT_SIDE, 512), "RGB")]


def test_adapter_weights_must_cover_every_adapter_layer():
    unet = {"down.attn1.to_q.weight", "down.attn1.processor.to_q_mv.weight", "down.attn1.processor.to_k_ref.weight"}
    encoder = {"adapter.conv_in.weight"}
    complete = {"down.attn1.processor.to_q_mv.weight": 0, "down.attn1.processor.to_k_ref.weight": 0, "adapter.conv_in.weight": 0}
    check_adapter_weights(complete, unet, encoder)

    with pytest.raises(RuntimeError, match="1 adapter layers without weights"):
        check_adapter_weights({k: v for k, v in complete.items() if "_ref" not in k}, unet, encoder)
    with pytest.raises(RuntimeError, match="1 unused tensors"):
        check_adapter_weights({**complete, "unet.down.attn1.processor.to_q_mv.weight": 0}, unet, encoder)
