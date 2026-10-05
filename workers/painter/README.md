# View painter (Phase 8)

Repaints a plain grey render of our mesh, seen from a given camera, so that it looks like the object in a reference
picture, keeping the render's exact outline and framing. It is [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
through diffusers' `QwenImageEditPlusPipeline`, with lightx2v's
[8-step Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) fused in, both Apache-2.0
([NOTICE.md](NOTICE.md)), on one H100 80GB in BF16 with every model resident. Not wired into `modal_app.py` yet.

```python
from painter_worker.qwen import LORA_DIR, MODEL_DIR, PROMPT_PAINT, QwenPainter, from_square

painter = QwenPainter(MODEL_DIR, LORA_DIR)  # once per container; painter.load_seconds
painted = painter.paint([render, picture, other_view], PROMPT_PAINT, seed=7)  # painter.last_seconds
painted_like_render = from_square(painted, render.size)  # only needed when the render isn't square
```

**In:** 1 to 3 PIL images. Picture 1 is the grey render, Picture 2 the reference picture, Picture 3 (optional)
another reference, such as a view painted before (Qwen-Image-Edit-2509's model card: "optimal performance is
currently achieved with 1 to 3 input images"). **Out:** an RGB 1024 x 1024 image that lines up pixel for pixel with
Picture 1, the same for the same pictures, prompt and seed (a `torch.Generator` on the GPU) on the same hardware and
software.

## Why every picture becomes a 1024 x 1024 square

What diffusers 0.37.1 does with the pictures
([pipeline_qwenimage_edit_plus.py](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)):

- Each picture goes in twice. The Qwen2.5-VL text encoder gets it resized to about 384 x 384
  (`calculate_dimensions(384 * 384, ratio)`; its processor makes that 392 x 392, 196 tokens), the transformer its VAE
  latents at about 1024 x 1024 (`calculate_dimensions(1024 * 1024, ratio)`), whatever its own size: aspect ratio
  kept, sides rounded to multiples of 32, Pillow's LANCZOS.
- The output is `height` x `width` (floored to multiples of 16); left out, about a megapixel at the _last_ picture's
  aspect ratio.
- 2511 sets `zero_cond_t` in its transformer config (new in diffusers 0.37.0): the pictures' latents are modulated as
  clean, only the output's at the current timestep. Each picture gets its own frame index but the same centred 2D
  positions, so a picture with the output's latent size lines up with the output token for token.

So `paint()` lays each picture over white, pads it to a white square and resizes it to 1024 x 1024 with LANCZOS, and
asks for 1024 x 1024. For a square, `calculate_dimensions` gives 1024 x 1024, Pillow's resize to an image's own size
is a copy, and 1024 is a multiple of 16: the VAE gets Picture 1's exact pixels. No other square size is left alone
(`pipeline_size()`), so `paint()` refuses other sizes. `tests/test_diffusers.py` checks all of this against diffusers.

## Sampling

The LoRA's recipe, from [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning)'s
`generate_with_diffusers.py` (also in diffusers' Qwen-Image docs): **8 steps, `true_cfg_scale` 1** (the LoRA is
step- and CFG-distilled: one transformer pass a step, no negative prompt), and `FlowMatchEulerDiscreteScheduler`
with `base_shift = max_shift = log 3` ("we use shift=3 in distillation"), `shift_terminal` off, dynamic shifting on:
a fixed shift of 3, sigmas 1, 0.95, 0.9, 0.83, 0.75, 0.64, 0.5, 0.3. The LoRA is fused into the transformer's
weights (as the Lightning repository's own first script merged it), so a step costs what the base model's does.
The 8-step files came out on 6 January 2026 (LightX2V's README); a user reported a red hue with the 4-step one
([discussion](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/discussions/2)). The repository's model
card still describes only the 4-step files; LightX2V's own 8-step config for 2511 names the fp32 8-step file, at
strength 1, with CFG off. Without a LoRA the painter uses the model card's recipe: the repository's scheduler,
40 steps, `true_cfg_scale` 4, negative prompt " ".

## Memory and speed

| BF16 weights (pinned files)      | GB    |
| -------------------------------- | ----- |
| Transformer (20.4B parameters)   | 40.86 |
| Qwen2.5-VL-7B text encoder       | 16.58 |
| VAE                              | 0.25  |
| All, resident together           | 57.70 |
| The LoRA, only while it is fused | 0.85  |

Activations add a few GB at most (the VAE's 1024 x 1024 decode is the largest); `painter.last_peak_gb` measures the
peak. Speed, estimated rather than measured: [LightX2V's benchmark](https://github.com/ModelTC/LightX2V/tree/main/examples/qwen_image)
of 2511 on one H100 (December 2025) has diffusers at 46.8 s of transformer time for 50 steps with CFG, 100 passes,
so about 0.47 s a pass; it doesn't say how many pictures or what size, so take it as one picture at about a
megapixel. 8 steps without CFG are 8 passes, about 4 s. Three pictures double the transformer's image tokens
(4 x 4,096 instead of 2 x 4,096), so expect about twice that or a little more, plus about a second for the text
encoder and the VAE. `painter.last_seconds` says what it really is.

## The prompt

The pipeline shows the pictures to the text encoder as "Picture 1: … Picture 2: …", so the prompts name them that
way, as Qwen's own prompt rewriter does. `PROMPT_PAINT` (three pictures; `PROMPT_PAINT_ONE_REFERENCE` without
Picture 3) is a starting point to tune on real renders:

> Repaint the grey 3D model in Picture 1 as the exact object shown in Picture 2: the same colours, paint,
> materials, logos and details. Keep the shape, outline, camera angle and framing of Picture 1 exactly. Where
> Picture 2 doesn't show a part, continue its design consistently with Picture 2 and Picture 3. Clean, even studio
> lighting with soft reflections, plain white background.

## Weights and tests

`MODELS_ROOT=/models python scripts/download_weights.py` puts the pinned repositories in `/models/Qwen-Image-Edit-2511`
(the whole diffusers repository, 57.7 GB) and `/models/Qwen-Image-Edit-2511-Lightning` (the two 8-step LoRA files),
checking every weight's sha256; `--check` checks what is there without downloading. The painter reads only those
folders (`local_files_only`).

`python -m pytest tests -q` runs everything with Pillow and torch; `tests/test_diffusers.py` (the pipeline's own
resize, the schedule, and a tiny random pipeline with a LoRA in each format diffusers converts) also needs
`requirements.txt` and torchvision installed.
