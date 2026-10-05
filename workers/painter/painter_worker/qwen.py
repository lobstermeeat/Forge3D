"""
The view painter: Qwen-Image-Edit-2511 (Apache-2.0) through diffusers' QwenImageEditPlusPipeline, with lightx2v's
8-step Lightning LoRA (Apache-2.0) fused in. It repaints a plain grey render of our mesh (Picture 1) as the object
in a reference picture (Picture 2, and optionally a third picture), keeping the render's outline and framing.

What the pinned diffusers (0.37.1, src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py) does with
the pictures:

- The Qwen2.5-VL text encoder reads them as "Picture 1: <image>Picture 2: <image>..." in front of the instruction,
  so prompts call them Picture 1, 2 and 3, as Qwen's own prompt rewriter does (src/examples/tools/prompt_utils.py).
  Qwen-Image-Edit-2509's model card (2511 is its successor): "optimal performance is currently achieved with 1 to
  3 input images".
- Every picture goes in twice. The text encoder sees it resized to about 384 x 384 pixels
  (calculate_dimensions(384 * 384, aspect ratio)), which Qwen2.5-VL's processor rounds to 392 x 392: 196 tokens.
  The transformer sees its VAE latents at about 1024 x 1024 (calculate_dimensions(1024 * 1024, aspect ratio)),
  whatever the picture's own size. Both resizes are Pillow's LANCZOS; calculate_dimensions keeps the aspect ratio
  and rounds each side to a multiple of 32.
- The output is height x width when they are given (floored to multiples of 16), otherwise about one megapixel at
  the *last* picture's aspect ratio.
- 2511's transformer config sets zero_cond_t, which diffusers learnt in 0.37.0: the input pictures' tokens are
  modulated as clean (timestep 0) and only the output's at the current timestep. Each picture's latents get their
  own frame index but the same centred 2D positions, so a picture with the output's latent size lines up with the
  output token for token.

So paint() makes every picture an RGB 1024 x 1024 square and asks for a 1024 x 1024 output. For a square,
calculate_dimensions(1024 * 1024) is 1024 x 1024, Pillow's resize to an image's own size returns a copy, and 1024
is a multiple of 16: the VAE gets Picture 1's exact pixels and the output lines up with them pixel for pixel. No
other square size does that (pipeline_size()); non-square pictures are padded to squares with white first.

Sampling follows the LoRA's recipe (github.com/ModelTC/Qwen-Image-Lightning, generate_with_diffusers.py, quoted
by diffusers' Qwen-Image docs): 8 steps, true_cfg_scale 1 (the LoRA is CFG-distilled: one transformer pass a step
and no negative prompt), and FlowMatchEulerDiscreteScheduler with a fixed shift of 3 (base_shift = max_shift =
log 3, no shift_terminal) in place of the repository's resolution-dependent schedule. Without the LoRA the
defaults are the model card's: the repository's scheduler, 40 steps, true_cfg_scale 4 and the negative prompt " ".

Memory, in BF16 with everything resident (estimate_memory()): 40.86 GB of transformer, 16.58 GB of Qwen2.5-VL
text encoder and 0.25 GB of VAE, 57.7 GB of weights in all, on an 80 GB H100.

diffusers and torch are imported only inside QwenPainter, so the rest of the package (and its tests) runs with
Pillow alone. QwenPainter(pipeline=...) wraps a pipeline that is already built, which is how tests use a stand-in.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from typing import Any, Optional, Sequence

from PIL import Image

# Folders under the models root, filled by scripts/download_weights.py
MODELS_ROOT = os.environ.get("PAINTER_MODELS_ROOT", "/models")
MODEL_FOLDER = "Qwen-Image-Edit-2511"  # Qwen/Qwen-Image-Edit-2511: the whole diffusers repository
LORA_FOLDER = "Qwen-Image-Edit-2511-Lightning"  # lightx2v/Qwen-Image-Edit-2511-Lightning: its 8-step files only
MODEL_DIR = os.path.join(MODELS_ROOT, MODEL_FOLDER)
LORA_DIR = os.path.join(MODELS_ROOT, LORA_FOLDER)
# The LoRA in fp32, the file LightX2V's own 8-step config for 2511 names (configs/qwen_image/
# qwen_image_i2i_2511_distill.json). The bf16 file is the same LoRA stored in bf16: diffusers holds the adapter in
# the transformer's dtype either way, and fuses it into the bf16 weights
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
LORA_FILES = (LORA_FILE, "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
ADAPTER = "lightning"

# Qwen-Image-Lightning's scheduler settings, verbatim ("We use shift=3 in distillation")
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
SCHEDULERS = ("lightning", "default")  # the LoRA's recipe, or the repository's scheduler and the model card's
LIGHTNING_STEPS = 8
BASE_STEPS = 40  # Qwen-Image-Edit-2511's model card: 40 steps, true_cfg_scale 4, negative prompt " "
BASE_TRUE_CFG = 4.0
DTYPES = ("bfloat16", "float16", "float32")
# The device_map values the pipeline itself accepts: "cpu" and the accelerator's type ("balanced" splits models)
DEVICE_MAPS = ("cuda", "cpu")

# QwenImageEditPlusPipeline's module constants (diffusers 0.37.1): the areas it resizes every picture to
VAE_AREA = 1024 * 1024  # VAE_IMAGE_SIZE: the transformer's view
CONDITION_AREA = 384 * 384  # CONDITION_IMAGE_SIZE: the text encoder's view
SIZE = 1024  # the one square side its resize leaves alone
MAX_IMAGES = 3
WHITE = (255, 255, 255)

# Picture 1 is the grey render; Picture 2 the reference picture; Picture 3 (optional) another reference, such as a
# view painted before. These are starting points to tune on real renders.
PROMPT_PAINT = (
    "Repaint the grey 3D model in Picture 1 as the exact object shown in Picture 2: the same colours, paint, "
    "materials, logos and details. Keep the shape, outline, camera angle and framing of Picture 1 exactly. Where "
    "Picture 2 doesn't show a part, continue its design consistently with Picture 2 and Picture 3. Clean, even "
    "studio lighting with soft reflections, plain white background."
)
PROMPT_PAINT_ONE_REFERENCE = (
    "Repaint the grey 3D model in Picture 1 as the exact object shown in Picture 2: the same colours, paint, "
    "materials, logos and details. Keep the shape, outline, camera angle and framing of Picture 1 exactly. Where "
    "Picture 2 doesn't show a part, continue its design consistently with Picture 2. Clean, even studio lighting "
    "with soft reflections, plain white background."
)
PROMPTS = {2: PROMPT_PAINT_ONE_REFERENCE, 3: PROMPT_PAINT}  # by the number of pictures

# Bytes of the pinned weight files (scripts/download_weights.py), by component
WEIGHT_BYTES = {
    "transformer": 40_861_028_288,  # 20.4B parameters in five files
    "text_encoder": 16_584_414_544,  # Qwen2.5-VL-7B, 8.3B parameters in four files
    "vae": 253_806_966,
}
LORA_BYTES = 849_608_296  # the LoRA's 425M parameters in bf16, the transformer's dtype, while it is being fused

# The tensor that each LoRA'd layer has exactly one of, in the formats diffusers converts for Qwen-Image
# (lora_down/lora_up/alpha, peft's "default" adapter, diffusers' own lora_A/lora_B)
LORA_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight", ".lora_A.default.weight")


def pipeline_size(width: int, height: int, area: int) -> tuple:
    """
    The size QwenImageEditPlusPipeline resizes a width x height picture to: its calculate_dimensions(area, ratio)
    (diffusers 0.37.1), which keeps the aspect ratio at about `area` pixels and rounds each side to a multiple of 32.
    Returns (width, height).
    """
    ratio = width / height
    new_width = math.sqrt(area * ratio)
    new_height = new_width / ratio
    return round(new_width / 32) * 32, round(new_height / 32) * 32


def check_size(size: int) -> None:
    """Raises unless the pipeline's VAE-side resize leaves a size x size picture as it is."""
    if pipeline_size(size, size, VAE_AREA) != (size, size):
        side = pipeline_size(1, 1, VAE_AREA)[0]
        raise ValueError(
            f"size must be {side}: QwenImageEditPlusPipeline resizes every picture to about {VAE_AREA} pixels, so a "
            f"{size} x {size} picture would be resized and the output wouldn't line up with it"
        )


def to_rgb(image: Image.Image) -> Image.Image:
    """The image in RGB, anything transparent laid over white."""
    if image.mode in ("RGBA", "LA", "PA", "RGBa") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, WHITE + (255,)), rgba).convert("RGB")
    return image.convert("RGB")


def to_square(image: Image.Image, size: int = SIZE) -> Image.Image:
    """
    The picture as an RGB size x size square: centred on a white square as wide as its longer side, then resized
    with LANCZOS. A size x size RGB picture comes back unchanged.
    """
    rgb = to_rgb(image)
    side = max(rgb.size)
    if rgb.width != rgb.height:
        square = Image.new("RGB", (side, side), WHITE)
        square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
        rgb = square
    return rgb.resize((size, size), Image.Resampling.LANCZOS) if side != size else rgb


def from_square(image: Image.Image, original_size: Sequence[int]) -> Image.Image:
    """
    Undoes to_square() on a picture that lines up with its result (paint()'s output lines up with Picture 1 as
    to_square() made it): the part where the original was, resized back to the original's size with LANCZOS.
    """
    width, height = original_size
    side = max(width, height)
    scale = image.width / side
    left, top = (side - width) // 2, (side - height) // 2
    box = (left * scale, top * scale, (left + width) * scale, (top + height) * scale)
    return image.resize((width, height), Image.Resampling.LANCZOS, box=box)


def estimate_memory() -> dict:
    """
    GPU memory in GB (10^9 bytes) for the painter in BF16 with every model resident: the weights' exact sizes at
    the pinned revisions, plus the LoRA's adapter weights while they are being fused (freed straight after).

    Activations come on top and stay small next to the weights: with three pictures the transformer works on
    4 x 4,096 latent tokens plus a few hundred text tokens, a 3,072-wide hidden state is about 0.1 GB of them, and
    PyTorch's fused attention never builds the full attention matrix; decoding one 1024 x 1024 latent takes the VAE
    a few GB more. QwenPainter.last_peak_gb measures the real peak on the first run.
    """
    gb = {name: round(size / 1e9, 2) for name, size in WEIGHT_BYTES.items()}
    gb["weights"] = round(sum(WEIGHT_BYTES.values()) / 1e9, 2)
    gb["lora_while_fusing"] = round(LORA_BYTES / 1e9, 2)
    return gb


def lora_file(path: Any) -> str:
    """The LoRA file: `path` itself, or LORA_FILE in it when it is a folder (LORA_DIR holds both 8-step files)."""
    path = os.path.abspath(os.fspath(path))
    if os.path.isdir(path):
        path = os.path.join(path, LORA_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no LoRA file at {path} (run scripts/download_weights.py)")
    return path


def safetensors_keys(path: str) -> list:
    """A .safetensors file's tensor names, from its header alone: 8 bytes of length, then that much JSON."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > 100_000_000:  # the format's own limit
            raise ValueError(f"{path} is not a .safetensors file")
        header = json.loads(f.read(length))
    return [key for key in header if key != "__metadata__"]


def lora_targets(keys: Sequence[str]) -> int:
    """How many layers a LoRA state dict's tensor names hold weights for."""
    return len({key[: -len(suffix)] for key in keys for suffix in LORA_DOWN_SUFFIXES if key.endswith(suffix)})


def count_lora_layers(model: Any, adapter: str = ADAPTER) -> int:
    """How many of a model's layers hold the adapter's LoRA (peft's LoRA layers keep it in lora_A[adapter])."""
    return sum(1 for module in model.modules() if adapter in (getattr(module, "lora_A", None) or {}))


def apply_lora(pipe: Any, path: str, scale: float = 1.0, adapter: str = ADAPTER) -> int:
    """
    Loads a LoRA file into the pipeline's transformer, checks that every layer it has weights for took them, fuses
    it into the transformer's weights at `scale` and drops the LoRA layers. Returns how many layers it changed.

    diffusers converts the LoRA formats it knows for Qwen-Image to its own and loads only names under
    "transformer."; any other name is left out with no more than a log line, and a LoRA that doesn't load would
    leave the base model sampling 8 steps. Hence the count. The file is passed with its weight_name: with
    local_files_only diffusers won't guess one, and the folder holds two.
    """
    folder, name = os.path.split(path)
    expected = lora_targets(safetensors_keys(path))
    if not expected:
        raise ValueError(f"{name} holds no LoRA weights (no lora_down or lora_A tensors)")
    pipe.load_lora_weights(folder, weight_name=name, adapter_name=adapter, local_files_only=True)
    landed = count_lora_layers(pipe.transformer, adapter)
    if landed != expected:
        raise RuntimeError(
            f"{name} has LoRA weights for {expected} layers, but {landed} layers of the transformer took them: "
            "diffusers doesn't recognise this file's tensor names"
        )
    pipe.fuse_lora(components=["transformer"], lora_scale=scale, adapter_names=[adapter])
    # The fused weights stay: unloading swaps each LoRA layer back for its base layer, which now holds them
    pipe.unload_lora_weights()
    return landed


class QwenPainter:
    """
    Qwen-Image-Edit-2511 loaded once per container from local folders (no network), then one paint() per view.

    lora_path is the Lightning LoRA, a file or a folder holding LORA_FILE. scheduler picks the sampling recipe:
    "lightning" (the LoRA's scheduler settings, 8 steps, no CFG; the default with a LoRA) or "default" (the
    repository's scheduler, 40 steps, true_cfg_scale 4; the default without one). steps overrides the recipe's.
    load_seconds is how long loading took, LoRA included; lora_layers how many layers the LoRA changed.
    """

    def __init__(
        self,
        model_dir: Any,
        lora_path: Any = None,
        *,
        lora_scale: float = 1.0,
        steps: Optional[int] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        scheduler: Optional[str] = None,
        pipeline: Any = None,
    ) -> None:
        started = time.perf_counter()
        if pipeline is not None and lora_path is not None:
            raise ValueError("a pipeline passed in is used as it is: load the LoRA into it first")
        if dtype not in DTYPES:
            raise ValueError(f"dtype is one of {', '.join(DTYPES)}, not {dtype!r}")
        self.model_dir = os.path.abspath(os.fspath(model_dir))
        self.lora_path = lora_file(lora_path) if lora_path is not None else None
        self.scheduler = scheduler or ("lightning" if self.lora_path else "default")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler is one of {', '.join(SCHEDULERS)}, not {self.scheduler!r}")
        lightning = self.scheduler == "lightning"
        self.steps = int(steps) if steps is not None else (LIGHTNING_STEPS if lightning else BASE_STEPS)
        self.true_cfg_scale = 1.0 if lightning else BASE_TRUE_CFG
        self.lora_scale = lora_scale
        self.device = device
        self.dtype = dtype
        self.lora_layers = 0
        self.pipeline = pipeline if pipeline is not None else self._load()
        self.load_seconds = round(time.perf_counter() - started, 1)
        self.last_seconds: Optional[float] = None
        self.last_peak_gb: Optional[float] = None

    def _load(self) -> Any:
        # A folder, never a Hugging Face repository name: nothing is fetched
        if not os.path.isfile(os.path.join(self.model_dir, "model_index.json")):
            raise FileNotFoundError(f"no diffusers pipeline in {self.model_dir} (run scripts/download_weights.py)")
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

        options: dict = {"torch_dtype": getattr(torch, self.dtype), "local_files_only": True}
        if self.scheduler == "lightning":
            options["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
        if self.device in DEVICE_MAPS:
            # accelerate puts each weight straight on the device as it is read, not all 58 GB through RAM first
            options["device_map"] = self.device
        pipe = QwenImageEditPlusPipeline.from_pretrained(self.model_dir, **options)
        if "device_map" not in options:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.lora_path is not None:
            self.lora_layers = apply_lora(pipe, self.lora_path, self.lora_scale)
        return pipe

    def paint(
        self,
        images: Any,
        prompt: str,
        *,
        seed: int,
        steps: Optional[int] = None,
        true_cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = " ",
        size: int = SIZE,
    ) -> Image.Image:
        """
        Repaints Picture 1 (the render) to look like the object in Pictures 2 and 3 (the references): 1 to 3 PIL
        images, each made an RGB size x size square (to_square()). Returns the size x size RGB result, which lines up pixel for pixel with Picture 1 as
        to_square() made it. The same pictures, prompt, seed and settings give the same result on the same GPU and
        software. steps and true_cfg_scale default to the recipe's; negative_prompt is used only when
        true_cfg_scale > 1. last_seconds is how long the call took, last_peak_gb the GPU memory peak (on CUDA).
        """
        import torch

        pictures = [images] if isinstance(images, Image.Image) else list(images)
        if not 1 <= len(pictures) <= MAX_IMAGES:
            raise ValueError(f"paint takes 1 to {MAX_IMAGES} pictures, not {len(pictures)}")
        if not all(isinstance(picture, Image.Image) for picture in pictures):
            raise TypeError("the pictures must be PIL images")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("the prompt must be some text")
        check_size(size)
        steps = self.steps if steps is None else int(steps)
        true_cfg_scale = self.true_cfg_scale if true_cfg_scale is None else float(true_cfg_scale)
        arguments = {
            "image": [to_square(picture, size) for picture in pictures],
            "prompt": prompt,
            # Given, because the pipeline would otherwise take the output's shape from the last picture
            "height": size,
            "width": size,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "generator": torch.Generator(device=self.device).manual_seed(int(seed)),
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }
        if true_cfg_scale > 1:
            # The pipeline runs CFG only with a negative prompt; without CFG it would only warn about one
            if negative_prompt is None:
                raise ValueError("true_cfg_scale > 1 needs a negative prompt (the model card's is ' ')")
            arguments["negative_prompt"] = negative_prompt
        on_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipeline(**arguments).images[0]
        self.last_seconds = round(time.perf_counter() - started, 2)
        self.last_peak_gb = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2) if on_cuda else None
        if result.size != (size, size):
            raise RuntimeError(f"the pipeline returned {result.size[0]} x {result.size[1]}, not {size} x {size}")
        return result.convert("RGB")
