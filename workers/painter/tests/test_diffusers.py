"""
The painter against the real diffusers (the pinned 0.37.1, with transformers, peft and torchvision), when they are
installed: that pipeline_size() is the pipeline's own resize, that a 1024 x 1024 picture reaches the VAE pixel for
pixel, the Lightning schedule, the text encoder's tokens a picture, and a tiny random QwenImageEditPlusPipeline
saved like the real one, loaded with a LoRA in each format diffusers converts for Qwen-Image, fused, and painting
on the CPU. The real weights never load here.
"""

import importlib.util
import math

import numpy as np
import pytest
from PIL import Image

from painter_worker import qwen as Q


def installed(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


needs_diffusers = pytest.mark.skipif(not installed("torch", "diffusers"), reason="diffusers isn't installed")
needs_stack = pytest.mark.skipif(
    not installed("torch", "diffusers", "transformers", "peft", "torchvision"),
    reason="diffusers, transformers, peft and torchvision aren't all installed",
)


@needs_diffusers
def test_pipeline_size_is_the_pipelines_own():
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as plus

    assert (plus.VAE_IMAGE_SIZE, plus.CONDITION_IMAGE_SIZE) == (Q.VAE_AREA, Q.CONDITION_AREA)
    shapes = [(w, h) for w in (1, 7, 300, 512, 1000, 1024, 1080, 1920, 4000) for h in (1, 9, 200, 768, 1024, 1920)]
    for area in (Q.VAE_AREA, Q.CONDITION_AREA, 128 * 128):
        for width, height in shapes:
            assert Q.pipeline_size(width, height, area) == plus.calculate_dimensions(area, width / height)


@needs_diffusers
def test_a_1024_square_reaches_the_vae_pixel_for_pixel():
    import torch
    from diffusers.image_processor import VaeImageProcessor

    # As the pipeline makes it: 2 ** len(temperal_downsample) for 2511's VAE ([false, true, true]), times 2
    processor = VaeImageProcessor(vae_scale_factor=8 * 2)
    picture = Q.to_square(Image.fromarray(np.random.default_rng(0).integers(0, 256, (700, 900, 3), dtype=np.uint8)))
    vae_side = processor.preprocess(picture, 1024, 1024)
    expected = torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0 * 2.0 - 1.0).permute(2, 0, 1)[None]
    assert vae_side.shape == (1, 3, 1024, 1024) and torch.equal(vae_side, expected)
    # The text encoder's copy is the same square at 384 x 384, Pillow's LANCZOS
    condition = processor.resize(picture, 384, 384)
    assert condition.tobytes() == picture.resize((384, 384), Image.Resampling.LANCZOS).tobytes()


@needs_diffusers
def test_the_lightning_schedule_is_a_fixed_shift_of_3():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(Q.LIGHTNING_SCHEDULER)
    config = scheduler.config
    # As the pipeline sets it up for a 1024 x 1024 output (4,096 latent tokens), and for any other size
    for tokens in (4096, 1024, 8192):
        mu = calculate_shift(tokens, config.base_image_seq_len, config.max_image_seq_len, config.base_shift, config.max_shift)
        assert mu == pytest.approx(math.log(3))
    sigmas = np.linspace(1.0, 1 / 8, 8)
    scheduler.set_timesteps(sigmas=sigmas, mu=math.log(3))
    expected = [3 * s / (1 + 2 * s) for s in sigmas] + [0.0]
    assert scheduler.sigmas.tolist() == pytest.approx(expected, abs=1e-6)
    assert scheduler.timesteps.tolist() == pytest.approx([1000 * s for s in expected[:-1]], abs=1e-3)


@needs_stack
def test_the_text_encoder_sees_196_tokens_a_picture():
    from transformers import Qwen2VLImageProcessor

    # Qwen-Image-Edit-2511's processor/preprocessor_config.json (its fast twin needs torchvision, same sizes)
    processor = Qwen2VLImageProcessor(
        min_pixels=3136, max_pixels=12845056, patch_size=14, merge_size=2, temporal_patch_size=2
    )
    picture = Q.to_square(Image.new("RGB", (900, 700), (90, 90, 90))).resize((384, 384), Image.Resampling.LANCZOS)
    grid = processor(images=[picture], return_tensors="pt")["image_grid_thw"]
    assert grid.tolist() == [[1, 28, 28]]  # 392 x 392 in 14-pixel patches
    assert int(grid.prod()) // 4 == 196


# --- A tiny random pipeline, end to end -----------------------------------------------------------------

SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
LAYERS = ["attn.to_q", "attn.to_k", "attn.add_v_proj", "attn.to_out.0", "img_mlp.net.2"]
TARGETS = [f"transformer_blocks.{block}.{layer}" for block in range(2) for layer in LAYERS]
SIDE = 128  # the tests make the pipeline's VAE-side area 128 x 128, so that this is the size it leaves alone


def tiny_pipeline(folder):
    """A random QwenImageEditPlusPipeline two layers deep, with 2511's zero_cond_t, saved like the real one."""
    import torch
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2TokenizerFast,
        Qwen2VLImageProcessor,
        Qwen2VLProcessor,
    )
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    byte_level = Tokenizer(models.BPE(vocab={char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}, merges=[]))
    byte_level.pre_tokenizer = ByteLevel(add_prefix_space=False)
    byte_level.decoder = decoders.ByteLevel()
    byte_level.add_special_tokens(SPECIALS)
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=byte_level, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token=None, bos_token=None
    )
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIALS}
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=512 * 512),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
    )
    torch.manual_seed(0)
    text_encoder = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(
            text_config={
                "vocab_size": len(tokenizer),
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
                "rope_theta": 1000000.0,
            },
            vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 16},
            image_token_id=ids["<|image_pad|>"],
            video_token_id=ids["<|video_pad|>"],
            vision_start_token_id=ids["<|vision_start|>"],
            vision_end_token_id=ids["<|vision_end|>"],
        )
    )
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=3,
        joint_attention_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(8, 4, 4),
        zero_cond_t=True,
    )
    vae = AutoencoderKLQwenImage(
        base_dim=24,
        z_dim=4,
        dim_mult=[1, 2, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    pipe = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    pipe.save_pretrained(folder)
    return transformer


def write_lora(path, transformer, style, rank=2, alpha=1.0):
    """A random LoRA for TARGETS in one of the formats diffusers converts for Qwen-Image. Returns each target's delta."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(1)
    tensors, deltas = {}, {}
    for target in TARGETS:
        layer = transformer.get_submodule(target)
        down = torch.randn(rank, layer.in_features, generator=generator) * 0.5
        up = torch.randn(layer.out_features, rank, generator=generator) * 0.5
        if style in ("lightning", "comfy"):  # lora_down/lora_up/alpha, as Qwen-Image-Lightning's own merge reads them
            prefix = "diffusion_model." if style == "comfy" else ""
            tensors[f"{prefix}{target}.lora_down.weight"] = down
            tensors[f"{prefix}{target}.lora_up.weight"] = up
            tensors[f"{prefix}{target}.alpha"] = torch.tensor(alpha)
            deltas[target] = alpha / rank * up @ down
        elif style == "default":  # peft's adapter names left in
            tensors[f"{target}.lora_A.default.weight"] = down
            tensors[f"{target}.lora_B.default.weight"] = up
            deltas[target] = up @ down
        elif style in ("diffusers", "unprefixed"):  # diffusers' own, and the same without its "transformer." prefix
            prefix = "transformer." if style == "diffusers" else ""
            tensors[f"{prefix}{target}.lora_A.weight"] = down
            tensors[f"{prefix}{target}.lora_B.weight"] = up
            deltas[target] = up @ down
    save_file(tensors, str(path))
    return deltas


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as plus

    # The pipeline's VAE-side area (and the painter's copy of it) made 128 x 128, its text-encoder side 64 x 64,
    # to keep the tiny model quick
    monkeypatch.setattr(plus, "VAE_IMAGE_SIZE", SIDE * SIDE)
    monkeypatch.setattr(Q, "VAE_AREA", SIDE * SIDE)
    monkeypatch.setattr(plus, "CONDITION_IMAGE_SIZE", 56 * 56)
    folder = tmp_path / Q.MODEL_FOLDER
    transformer = tiny_pipeline(folder)
    return folder, transformer


def painter_with_lora(folder, transformer, lora, style="lightning"):
    """The tiny pipeline with a LoRA in `style` fused at half strength. Returns the painter and each layer's delta."""
    lora.mkdir()
    deltas = write_lora(lora / Q.LORA_FILE, transformer, style)
    return Q.QwenPainter(folder, lora, lora_scale=0.5, device="cpu", dtype="float32"), deltas


@needs_stack
@pytest.mark.parametrize("style", ["lightning", "comfy", "default", "diffusers"])
def test_a_lora_in_each_format_is_fused(tiny, tmp_path, style):
    import torch

    folder, transformer = tiny
    painter, deltas = painter_with_lora(folder, transformer, tmp_path / Q.LORA_FOLDER, style)
    pipe = painter.pipeline
    assert painter.lora_layers == len(TARGETS) == 10
    # Fused at half strength, and no LoRA layers left behind
    for target, delta in deltas.items():
        fused = pipe.transformer.get_submodule(target).weight
        base = transformer.get_submodule(target).weight
        torch.testing.assert_close(fused, base + 0.5 * delta, atol=1e-5, rtol=1e-4)
    assert Q.count_lora_layers(pipe.transformer) == 0 and not hasattr(pipe.transformer, "peft_config")
    assert type(pipe.transformer.get_submodule(TARGETS[0])).__name__ == "Linear"
    assert pipe.transformer.config.zero_cond_t
    # The Lightning schedule
    assert pipe.scheduler.config.base_shift == pipe.scheduler.config.max_shift == pytest.approx(math.log(3))
    assert pipe.scheduler.config.shift_terminal is None


@needs_stack
def test_a_tiny_pipeline_paints_on_the_cpu(tiny, tmp_path, monkeypatch):
    import torch

    folder, transformer = tiny
    painter, _ = painter_with_lora(folder, transformer, tmp_path / Q.LORA_FOLDER)
    pipe = painter.pipeline
    # What reaches the VAE is each square exactly
    seen = []
    preprocess = pipe.image_processor.preprocess

    def spy(image, height=None, width=None, **options):
        out = preprocess(image, height, width, **options)
        seen.append((image, height, width, out))
        return out

    monkeypatch.setattr(pipe.image_processor, "preprocess", spy)
    render = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
    render.paste(Image.new("RGBA", (40, 30), (120, 120, 120, 255)), (28, 17))
    reference = Image.fromarray(np.random.default_rng(2).integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8))
    result = painter.paint([render, reference], Q.PROMPT_PAINT_ONE_REFERENCE, seed=3, size=SIDE, steps=2)
    assert result.mode == "RGB" and result.size == (SIDE, SIDE)
    assert [(h, w) for _, h, w, _ in seen] == [(SIDE, SIDE)] * 2
    for (image, _, _, out), picture in zip(seen, (render, reference)):
        assert image.tobytes() == Q.to_square(picture, SIDE).tobytes()
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0).permute(2, 0, 1)[None]
        assert torch.equal(out, pixels)
    # The same seed paints the same picture, another seed another
    again = painter.paint([render, reference], Q.PROMPT_PAINT_ONE_REFERENCE, seed=3, size=SIDE, steps=2)
    other = painter.paint([render, reference], Q.PROMPT_PAINT_ONE_REFERENCE, seed=4, size=SIDE, steps=2)
    assert result.tobytes() == again.tobytes() != other.tobytes()
    assert painter.last_seconds > 0


@needs_stack
def test_a_lora_diffusers_would_leave_out_stops_the_load(tiny, tmp_path):
    folder, transformer = tiny
    path = tmp_path / "unprefixed.safetensors"
    write_lora(path, transformer, "unprefixed")
    # diffusers loads only names under "transformer.", and says no more than a warning about the rest
    with pytest.raises(RuntimeError, match="weights for 10 layers, but 0 layers"):
        Q.QwenPainter(folder, path, device="cpu", dtype="float32")


@needs_stack
def test_three_pictures_and_the_base_recipe(tiny, monkeypatch):
    folder, _ = tiny
    painter = Q.QwenPainter(folder, device="cpu", dtype="float32")  # no LoRA: the scheduler saved with it
    assert (painter.scheduler, painter.steps, painter.true_cfg_scale) == ("default", 40, 4.0)
    assert painter.pipeline.scheduler.config.base_shift == 0.5  # FlowMatchEulerDiscreteScheduler's own default
    calls = []
    transformer_forward = painter.pipeline.transformer.forward

    def count(*args, **options):
        calls.append(options["img_shapes"])
        return transformer_forward(*args, **options)

    monkeypatch.setattr(painter.pipeline.transformer, "forward", count)
    pictures = [Image.new("RGB", (SIDE, SIDE), colour) for colour in ((90, 90, 90), (200, 30, 30), (30, 30, 200))]
    result = painter.paint(pictures, Q.PROMPT_PAINT, seed=1, size=SIDE, steps=2)
    assert result.size == (SIDE, SIDE)
    # CFG (true_cfg_scale 4): two passes a step; the output and the three pictures at the same latent size
    assert len(calls) == 4
    assert calls[0] == [[(1, SIDE // 8, SIDE // 8)] * 4]
