"""
The painter around a stand-in pipeline: how pictures are made into the squares the pipeline leaves alone, what
paint() passes to the pipeline, seeds, and what loading asks of diffusers (the model, the scheduler, the LoRA),
through a fake diffusers module. Runs with Pillow and pytest; the tests that build torch generators need torch.
tests/test_diffusers.py checks the same against the real diffusers, when it is installed.
"""

import json
import math
import struct
import sys
import types

import numpy as np
import pytest
from PIL import Image

from painter_worker import qwen as Q

try:
    import torch
except ImportError:  # the pure tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch isn't installed")


def noise(size, mode="RGB", seed=0):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], channels), dtype=np.uint8)
    return Image.fromarray(pixels[..., 0] if channels == 1 else pixels)


# --- Sizes: what the pipeline's own resize leaves alone -------------------------------------------------


def test_only_1024_squares_are_left_alone_by_the_pipeline():
    alone = [side for side in range(64, 2049, 16) if Q.pipeline_size(side, side, Q.VAE_AREA) == (side, side)]
    assert alone == [1024] == [Q.SIZE]
    # The text encoder's copy is always about 384 x 384, whatever the size
    assert {Q.pipeline_size(side, side, Q.CONDITION_AREA) for side in (256, 1024, 3000)} == {(384, 384)}
    # Other shapes keep their aspect ratio at about a megapixel, sides in multiples of 32
    assert Q.pipeline_size(1920, 1080, Q.VAE_AREA) == (1376, 768)
    assert Q.pipeline_size(1080, 1920, Q.VAE_AREA) == (768, 1376)


def test_other_sizes_are_refused():
    Q.check_size(1024)
    for size in (512, 768, 1008, 1040, 2048):
        with pytest.raises(ValueError, match="size must be 1024"):
            Q.check_size(size)


# --- Pictures into squares ------------------------------------------------------------------------------


def test_a_square_rgb_picture_of_the_size_is_unchanged():
    picture = noise((1024, 1024))
    square = Q.to_square(picture)
    assert square.mode == "RGB" and square.size == (1024, 1024)
    assert square.tobytes() == picture.tobytes()


def test_other_squares_are_resized_with_lanczos():
    picture = noise((512, 512), seed=1)
    expected = picture.resize((1024, 1024), Image.Resampling.LANCZOS)
    assert Q.to_square(picture).tobytes() == expected.tobytes()
    big = noise((1536, 1536), seed=2)
    assert Q.to_square(big).tobytes() == big.resize((1024, 1024), Image.Resampling.LANCZOS).tobytes()


@pytest.mark.parametrize("size, offset", [((6, 4), (0, 1)), ((4, 6), (1, 0)), ((5, 2), (0, 1))])
def test_other_shapes_are_centred_on_white_squares(size, offset):
    red = Image.new("RGB", size, (200, 10, 10))
    side = max(size)
    square = np.asarray(Q.to_square(red, side))  # the square's own size: padding only, no resize
    expected = np.full((side, side, 3), 255, dtype=np.uint8)
    left, top = offset
    expected[top : top + size[1], left : left + size[0]] = (200, 10, 10)
    assert np.array_equal(square, expected)


def test_padding_then_resizing():
    wide = Image.new("RGB", (2048, 1024), (10, 120, 200))
    square = np.asarray(Q.to_square(wide))
    assert square.shape == (1024, 1024, 3)
    # A white band above and below the picture, 256 rows each
    assert (square[:250] == 255).all() and (square[-250:] == 255).all()
    assert (square[262:762] == (10, 120, 200)).all()


def pixels(image):
    return [tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3).tolist()]


def test_transparency_goes_over_white():
    picture = Image.fromarray(np.array([[(0, 0, 0, 0), (200, 0, 0, 255), (0, 0, 200, 128), (10, 20, 30, 0)]], np.uint8))
    assert pixels(Q.to_rgb(picture)) == [(255, 255, 255), (200, 0, 0), (127, 127, 227), (255, 255, 255)]
    grey = Image.new("LA", (2, 2), (90, 0))
    assert set(pixels(Q.to_rgb(grey))) == {(255, 255, 255)}
    palette = Image.new("P", (2, 1), 0)
    palette.putpixel((1, 0), 1)
    palette.putpalette([0, 0, 0, 255, 0, 0])
    palette.info["transparency"] = 0
    assert pixels(Q.to_rgb(palette)) == [(255, 255, 255), (255, 0, 0)]


@pytest.mark.parametrize("mode", ["L", "1", "P", "CMYK", "RGB", "RGBA", "LA"])
def test_every_mode_comes_out_rgb(mode):
    square = Q.to_square(Image.new(mode, (30, 20)), 64)
    assert square.mode == "RGB" and square.size == (64, 64)


def test_from_square_undoes_to_square():
    x = np.linspace(0, 255, 300)
    y = np.linspace(0, 255, 200)
    gradient = np.stack([np.add.outer(y, x) / 2, np.add.outer(y, 0 * x), np.add.outer(0 * y, x)], -1).astype(np.uint8)
    picture = Image.fromarray(gradient)
    back = Q.from_square(Q.to_square(picture), picture.size)
    assert back.size == picture.size
    error = np.abs(np.asarray(back, dtype=int) - gradient)
    # LANCZOS up and back down: the same picture, but for the rows next to the white padding
    assert error[5:-5].mean() < 0.1 and error.mean() < 1.0
    square = noise((1024, 1024), seed=3)
    assert Q.from_square(square, (1024, 1024)).tobytes() == square.tobytes()


# --- Prompts and memory ---------------------------------------------------------------------------------


def test_the_prompts_name_the_pictures_as_the_pipeline_labels_them():
    # The pipeline puts "Picture 1: <image>Picture 2: <image>..." in front of the prompt
    assert Q.PROMPTS == {2: Q.PROMPT_PAINT_ONE_REFERENCE, 3: Q.PROMPT_PAINT}
    for count, prompt in Q.PROMPTS.items():
        named = {n for n in range(1, 5) if f"Picture {n}" in prompt}
        assert named == set(range(1, count + 1)), count
        for wording in ("image 1", "Image 1", "Figure", "picture 1"):
            assert wording not in prompt
        for asked in ("outline", "camera angle", "framing", "plain white background"):
            assert asked in prompt


def test_estimated_memory_is_the_weights_in_bf16():
    assert Q.estimate_memory() == {
        "transformer": 40.86,
        "text_encoder": 16.58,
        "vae": 0.25,
        "weights": 57.7,
        "lora_while_fusing": 0.85,
    }


# --- LoRA files ------------------------------------------------------------------------------------------

TARGETS = [f"transformer_blocks.{block}.{layer}" for block in range(2) for layer in ("attn.to_q", "attn.to_k", "img_mlp.net.2")]


def write_safetensors(path, names, shape=(2, 4)):
    """A .safetensors file of float32 zeros, written by hand: 8 bytes of header length, the JSON header, the data."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name in names:
        scalar = name.endswith(".alpha")
        size = 4 * (1 if scalar else shape[0] * shape[1])
        header[name] = {"dtype": "F32", "shape": [] if scalar else list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return path


def lora_names(style, targets=TARGETS):
    parts = {
        "lightning": ("lora_down.weight", "lora_up.weight", "alpha"),
        "default": ("lora_A.default.weight", "lora_B.default.weight"),
        "diffusers": ("lora_A.weight", "lora_B.weight"),
    }
    prefix = {"lightning": "", "comfy": "diffusion_model.", "diffusers": "transformer.", "default": ""}
    names = parts["lightning" if style == "comfy" else style]
    return [f"{prefix[style]}{target}.{part}" for target in targets for part in names]


@pytest.mark.parametrize("style", ["lightning", "comfy", "default", "diffusers"])
def test_lora_layers_are_counted_in_every_format(tmp_path, style):
    path = write_safetensors(tmp_path / "lora.safetensors", lora_names(style))
    assert sorted(Q.safetensors_keys(str(path))) == sorted(lora_names(style))
    assert Q.lora_targets(Q.safetensors_keys(str(path))) == len(TARGETS) == 6


@needs_torch
def test_safetensors_keys_agree_with_the_library(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tensors = {name: torch.zeros(2, 3) for name in lora_names("diffusers")}
    safetensors_torch.save_file(tensors, str(tmp_path / "lora.safetensors"))
    assert sorted(Q.safetensors_keys(str(tmp_path / "lora.safetensors"))) == sorted(tensors)


def test_the_lora_file_is_found_in_its_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="no LoRA file"):
        Q.lora_file(tmp_path)
    write_safetensors(tmp_path / Q.LORA_FILE, lora_names("lightning"))
    assert Q.lora_file(tmp_path) == str(tmp_path / Q.LORA_FILE)
    other = write_safetensors(tmp_path / "other.safetensors", lora_names("lightning"))
    assert Q.lora_file(other) == str(other)


# --- paint(), around a stand-in pipeline ------------------------------------------------------------------


class FakePipeline:
    """Stands in for a loaded pipeline: records each call and paints noise drawn from the call's generator."""

    def __init__(self, result_size=None):
        self.calls = []
        self.result_size = result_size

    def __call__(self, **arguments):
        self.calls.append(arguments)
        tile = torch.rand((8, 8, 3), generator=arguments["generator"])
        picture = Image.fromarray((tile.numpy() * 255).astype(np.uint8))
        size = self.result_size or (arguments["width"], arguments["height"])
        return types.SimpleNamespace(images=[picture.resize(size, Image.Resampling.NEAREST)])


def painter(fake=None, **options):
    options.setdefault("scheduler", "lightning")
    return Q.QwenPainter("/models/Qwen-Image-Edit-2511", device="cpu", pipeline=fake or FakePipeline(), **options)


@needs_torch
def test_paint_passes_three_squares_in_order_and_asks_for_1024():
    render = Image.new("RGBA", (300, 200), (128, 128, 128, 255))
    reference = noise((1024, 1024), seed=4)
    other = Image.new("L", (512, 768), 30)
    fake = FakePipeline()
    result = painter(fake).paint([render, reference, other], Q.PROMPT_PAINT, seed=11)
    assert result.mode == "RGB" and result.size == (1024, 1024)
    (call,) = fake.calls
    pictures = call.pop("image")
    assert [(p.mode, p.size) for p in pictures] == [("RGB", (1024, 1024))] * 3
    assert [p.tobytes() for p in pictures] == [Q.to_square(p).tobytes() for p in (render, reference, other)]
    assert pictures[1].tobytes() == reference.tobytes()  # already a 1024 square: untouched
    generator = call.pop("generator")
    assert isinstance(generator, torch.Generator) and generator.device.type == "cpu"
    assert call == {
        "prompt": Q.PROMPT_PAINT,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
        "num_images_per_prompt": 1,
        "output_type": "pil",
    }  # no negative prompt: without CFG the pipeline would only warn about it


@needs_torch
def test_the_same_seed_paints_the_same_picture():
    fake = FakePipeline()
    paint = painter(fake)
    pictures = [noise((640, 480), seed=5), noise((1024, 1024), seed=6)]
    first = paint.paint(pictures, Q.PROMPT_PAINT_ONE_REFERENCE, seed=7)
    again = paint.paint(pictures, Q.PROMPT_PAINT_ONE_REFERENCE, seed=7)
    other = paint.paint(pictures, Q.PROMPT_PAINT_ONE_REFERENCE, seed=8)
    assert first.tobytes() == again.tobytes() and first.tobytes() != other.tobytes()
    assert [call["generator"].initial_seed() for call in fake.calls] == [7, 7, 8]


@needs_torch
def test_steps_and_cfg():
    fake = FakePipeline()
    paint = painter(fake)
    paint.paint([noise((64, 64))], "Make it red", seed=1, steps=4)
    assert fake.calls[-1]["num_inference_steps"] == 4 and "negative_prompt" not in fake.calls[-1]
    paint.paint([noise((64, 64))], "Make it red", seed=1, true_cfg_scale=4.0)
    assert fake.calls[-1]["true_cfg_scale"] == 4.0 and fake.calls[-1]["negative_prompt"] == " "
    paint.paint([noise((64, 64))], "Make it red", seed=1, true_cfg_scale=3, negative_prompt="blurry")
    assert fake.calls[-1]["negative_prompt"] == "blurry"
    with pytest.raises(ValueError, match="needs a negative prompt"):
        paint.paint([noise((64, 64))], "Make it red", seed=1, true_cfg_scale=4.0, negative_prompt=None)


@needs_torch
def test_without_the_lora_the_model_cards_recipe():
    fake = FakePipeline()
    base = painter(fake, scheduler=None)  # no LoRA: the repository's scheduler
    assert (base.scheduler, base.steps, base.true_cfg_scale) == ("default", 40, 4.0)
    base.paint([noise((64, 64))], "Make it red", seed=1)
    assert fake.calls[-1]["num_inference_steps"] == 40
    assert fake.calls[-1]["true_cfg_scale"] == 4.0 and fake.calls[-1]["negative_prompt"] == " "
    assert painter(fake, steps=12).steps == 12


@needs_torch
def test_paint_checks_its_inputs():
    paint = painter()
    picture = noise((64, 64))
    with pytest.raises(ValueError, match="1 to 3 pictures, not 0"):
        paint.paint([], "x", seed=1)
    with pytest.raises(ValueError, match="1 to 3 pictures, not 4"):
        paint.paint([picture] * 4, "x", seed=1)
    with pytest.raises(TypeError, match="PIL images"):
        paint.paint([np.zeros((8, 8, 3))], "x", seed=1)
    with pytest.raises(ValueError, match="size must be 1024"):
        paint.paint([picture], "x", seed=1, size=512)
    with pytest.raises(ValueError, match="some text"):
        paint.paint([picture], "  ", seed=1)
    with pytest.raises(TypeError):
        paint.paint([picture], "x")  # the seed is required
    assert paint.paint(picture, "x", seed=1).size == (1024, 1024)  # one picture needn't be in a list


@needs_torch
def test_each_call_is_timed_and_checked():
    paint = painter()
    assert paint.last_seconds is None and paint.load_seconds >= 0
    paint.paint([noise((64, 64))], "x", seed=1)
    assert isinstance(paint.last_seconds, float) and paint.last_seconds >= 0
    assert paint.last_peak_gb is None  # measured on CUDA only
    with pytest.raises(RuntimeError, match="returned 512 x 512, not 1024 x 1024"):
        painter(FakePipeline(result_size=(512, 512))).paint([noise((64, 64))], "x", seed=1)


# --- Loading, through a fake diffusers module -----------------------------------------------------------


def fake_diffusers(calls, landing=None):
    """
    diffusers' stand-in. load_lora_weights gives a LoRA to as many of the transformer's layers as the file has
    weights for (or to `landing` layers), the way peft's LoRA layers hold it: in lora_A[adapter].
    """

    class FlowMatchEulerDiscreteScheduler:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_config(cls, config):
            calls.append(("scheduler", dict(config)))
            return cls(dict(config))

    class Layer:
        def __init__(self):
            self.lora_A = {}

    class Transformer:
        def __init__(self):
            self.layers = [Layer() for _ in range(20)]

        def modules(self):
            return [self, *self.layers]

    class QwenImageEditPlusPipeline:
        def __init__(self, options):
            self.options = options
            self.transformer = Transformer()

        @classmethod
        def from_pretrained(cls, path, **options):
            calls.append(("from_pretrained", path, options))
            return cls(options)

        def to(self, device):
            calls.append(("to", device))
            return self

        def set_progress_bar_config(self, **options):
            calls.append(("progress bar", options))

        def load_lora_weights(self, folder, **options):
            calls.append(("load_lora_weights", folder, options))
            names = Q.safetensors_keys(f"{folder}/{options['weight_name']}")
            count = Q.lora_targets(names) if landing is None else landing
            for layer in self.transformer.layers[:count]:
                layer.lora_A[options["adapter_name"]] = object()

        def fuse_lora(self, **options):
            calls.append(("fuse_lora", options))

        def unload_lora_weights(self):
            calls.append(("unload_lora_weights",))

    module = types.ModuleType("diffusers")
    module.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler
    module.QwenImageEditPlusPipeline = QwenImageEditPlusPipeline
    return module


@pytest.fixture
def folders(tmp_path):
    model = tmp_path / Q.MODEL_FOLDER
    model.mkdir()
    (model / "model_index.json").write_text('{"_class_name": "QwenImageEditPlusPipeline"}')
    lora = tmp_path / Q.LORA_FOLDER
    lora.mkdir()
    for name in Q.LORA_FILES:
        write_safetensors(lora / name, lora_names("lightning"))
    return model, lora


@needs_torch
def test_loading_with_the_lora(monkeypatch, folders):
    model, lora = folders
    calls = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls))
    paint = Q.QwenPainter(model, lora)
    scheduler, loaded, progress, load_lora, fuse, unload = calls
    assert scheduler == ("scheduler", Q.LIGHTNING_SCHEDULER)
    assert Q.LIGHTNING_SCHEDULER["base_shift"] == Q.LIGHTNING_SCHEDULER["max_shift"] == math.log(3)
    assert Q.LIGHTNING_SCHEDULER["shift_terminal"] is None and Q.LIGHTNING_SCHEDULER["use_dynamic_shifting"]
    options = loaded[2]
    assert loaded[1] == str(model)
    assert options.pop("scheduler").config == Q.LIGHTNING_SCHEDULER
    # Local files only, every weight straight onto the GPU in bf16
    assert options == {"torch_dtype": torch.bfloat16, "local_files_only": True, "device_map": "cuda"}
    assert progress == ("progress bar", {"disable": True})
    # The fp32 8-step file by name (with local_files_only diffusers won't guess one), fused and unloaded
    assert load_lora == (
        "load_lora_weights",
        str(lora),
        {"weight_name": Q.LORA_FILE, "adapter_name": "lightning", "local_files_only": True},
    )
    assert fuse == ("fuse_lora", {"components": ["transformer"], "lora_scale": 1.0, "adapter_names": ["lightning"]})
    assert unload == ("unload_lora_weights",)
    assert (paint.scheduler, paint.steps, paint.true_cfg_scale, paint.lora_layers) == ("lightning", 8, 1.0, 6)
    assert paint.lora_path == str(lora / Q.LORA_FILE) and isinstance(paint.load_seconds, float)


@needs_torch
def test_loading_a_lora_file_on_another_gpu(monkeypatch, folders):
    model, lora = folders
    calls = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls))
    bf16 = lora / Q.LORA_FILES[1]
    Q.QwenPainter(model, bf16, lora_scale=0.75, device="cuda:1", dtype="float16", steps=6)
    loaded = calls[1]
    assert loaded[2]["torch_dtype"] == torch.float16 and "device_map" not in loaded[2]  # cuda:1 isn't a device_map
    assert calls[2] == ("to", "cuda:1")
    options = {"weight_name": Q.LORA_FILES[1], "adapter_name": "lightning", "local_files_only": True}
    assert calls[4] == ("load_lora_weights", str(lora), options)
    assert calls[5][1]["lora_scale"] == 0.75


@needs_torch
def test_loading_without_the_lora_keeps_the_repositorys_scheduler(monkeypatch, folders):
    model, _ = folders
    calls = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls))
    paint = Q.QwenPainter(model)
    assert [call[0] for call in calls] == ["from_pretrained", "progress bar"]
    assert calls[0][2] == {"torch_dtype": torch.bfloat16, "local_files_only": True, "device_map": "cuda"}
    assert (paint.scheduler, paint.steps, paint.lora_layers) == ("default", 40, 0)
    # The Lightning schedule can be tried on its own
    calls.clear()
    Q.QwenPainter(model, scheduler="lightning", device="cpu")
    assert [call[0] for call in calls] == ["scheduler", "from_pretrained", "progress bar"]
    assert calls[1][2]["device_map"] == "cpu"


@needs_torch
def test_a_lora_that_does_not_land_stops_the_load(monkeypatch, folders):
    model, lora = folders
    calls = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls, landing=0))
    with pytest.raises(RuntimeError, match="weights for 6 layers, but 0 layers"):
        Q.QwenPainter(model, lora)
    assert "fuse_lora" not in [call[0] for call in calls]
    calls.clear()
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls, landing=5))
    with pytest.raises(RuntimeError, match="weights for 6 layers, but 5 layers"):
        Q.QwenPainter(model, lora)
    empty = write_safetensors(lora / "empty.safetensors", ["transformer_blocks.0.norm.weight"])
    with pytest.raises(ValueError, match="holds no LoRA weights"):
        Q.QwenPainter(model, empty)


@needs_torch
def test_loading_checks_its_arguments_before_reading_anything(monkeypatch, folders, tmp_path):
    model, lora = folders
    calls = []
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers(calls))
    with pytest.raises(FileNotFoundError, match="no LoRA file"):
        Q.QwenPainter(model, tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="no diffusers pipeline"):
        Q.QwenPainter(tmp_path, lora)
    with pytest.raises(ValueError, match="scheduler is one of lightning, default, not 'fast'"):
        Q.QwenPainter(model, lora, scheduler="fast")
    with pytest.raises(ValueError, match="dtype is one of"):
        Q.QwenPainter(model, lora, dtype="int8")
    with pytest.raises(ValueError, match="used as it is"):
        Q.QwenPainter(model, lora, pipeline=FakePipeline())
    assert calls == []
