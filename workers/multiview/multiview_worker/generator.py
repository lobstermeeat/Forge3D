"""
MV-Adapter's image-to-multiview SDXL model: one picture of an object in, six views around it out.

It does what MV-Adapter's own scripts/inference_i2mv_sdxl.py (Apache-2.0) does, adapted here with
every model loaded from local folders (scripts/download_weights.py):

1. Cut the object out of the picture with BiRefNet, as the inference script's --remove_bg does and
   as the TRELLIS.2 worker does (MIT). A picture with its own transparency keeps its alpha.
2. Crop to the cutout's box, scale its longer side to 90% of 768 px, centre it on a 768 x 768
   canvas and flatten it onto mid-gray (0.5), the background of MV-Adapter's training renders.
3. Draw the six views (azimuths 0, 45, 90, 180, 270 and 315 at elevation 0; see cameras.py) with
   SDXL and the adapter: 50 steps, guidance 3, the caption "high quality" unless the job describes
   the object. The views come out on the same mid-gray.
4. Cut each view out of its gray background with BiRefNet, so the views are RGBA cutouts.

Everything but the GPU part (the cutout and reference preparation) runs on a CPU, for tests.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .cameras import AZIMUTH_OFFSET, AZIMUTHS, DISTANCE, ELEVATION, FILL, HALF_EXTENT, IMAGE_SIZE
from .inputs import DEFAULT_PROMPT, InputError

NEGATIVE_PROMPT = "watermark, ugly, deformed, noisy, blurry, low contrast"
STEPS = 50
GUIDANCE = 3.0
REFERENCE_SCALE = 1.0
SHIFT_SCALE = 8.0  # the noise-schedule shift MV-Adapter was trained with ("interpolated" ShiftSNR)
BACKGROUND = 0.5  # mid-gray, as in MV-Adapter's training renders and its inference script
MAX_CUTOUT_SIDE = 1024  # larger pictures are scaled down before background removal, as TRELLIS.2 does

# Folders under the models root, filled by scripts/download_weights.py
SDXL_DIR = "stable-diffusion-xl-base-1.0"
VAE_DIR = "sdxl-vae-fp16-fix"
ADAPTER_DIR = "mv-adapter"
ADAPTER_FILE = "mvadapter_i2mv_sdxl.safetensors"
BIREFNET_DIR = "BiRefNet"  # shared with the TRELLIS.2 worker

Remover = Callable[[Image.Image], Image.Image]  # RGB picture -> the same picture with a cutout alpha


def has_transparency(image: Image.Image) -> bool:
    return image.mode == "RGBA" and bool((np.asarray(image)[..., 3] < 255).any())


def cut_out(image: Image.Image, remove_background: Remover) -> Image.Image:
    """The picture as RGBA with the object cut out: its own alpha if it has one, else BiRefNet's."""
    if has_transparency(image):
        return image
    rgb = image.convert("RGB")
    scale = min(1.0, MAX_CUTOUT_SIDE / max(rgb.size))
    if scale < 1:
        rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)
    return remove_background(rgb)


def prepare_reference(
    cutout: Image.Image, size: int = IMAGE_SIZE, fill: float = FILL, background: float = BACKGROUND
) -> Image.Image:
    """
    MV-Adapter's preprocess_image: the cutout's box (alpha > 0, one pixel more at the top and left),
    its longer side scaled to `fill` of `size`, centred on a `size` x `size` canvas and flattened
    onto gray `background`.
    """
    image = np.array(cutout.convert("RGBA"))
    found = image[..., 3] > 0
    if not found.any():
        raise InputError("no object found in the picture")
    height, width = found.shape
    ys, xs = np.nonzero(found)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 1, height)
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 1, width)
    crop = image[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    if h > w:
        w, h = max(1, int(w * (size * fill) / h)), int(size * fill)
    else:
        h, w = max(1, int(h * (size * fill) / w)), int(size * fill)
    crop = np.array(Image.fromarray(crop).resize((w, h)))
    top, left = (size - h) // 2, (size - w) // 2
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[top : top + h, left : left + w] = crop
    canvas = canvas.astype(np.float32) / 255.0
    rgb = canvas[..., :3] * canvas[..., 3:4] + (1 - canvas[..., 3:4]) * background
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


def control_images(device: str) -> Any:
    """The six views' camera conditions, as MV-Adapter's inference script makes them."""
    from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
    from mvadapter.utils.mesh_utils import get_orthogonal_camera

    count = len(AZIMUTHS)
    cameras = get_orthogonal_camera(
        elevation_deg=[ELEVATION] * count,
        distance=[DISTANCE] * count,
        left=-HALF_EXTENT,
        right=HALF_EXTENT,
        bottom=-HALF_EXTENT,
        top=HALF_EXTENT,
        azimuth_deg=[azimuth + AZIMUTH_OFFSET for azimuth in AZIMUTHS],
        device=device,
    )
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [2 * HALF_EXTENT] * count, IMAGE_SIZE)
    return ((plucker + 1.0) / 2.0).clamp(0, 1)


class BiRefNetRemover:
    """The TRELLIS.2 worker's background remover (trellis2/pipelines/rembg/BiRefNet.py), from a local folder."""

    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.model = AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True)
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        import torch
        from torchvision import transforms

        rgb = image.convert("RGB")
        batch = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(batch)[-1].sigmoid().cpu()[0].squeeze()
        rgb.putalpha(transforms.ToPILImage()(prediction).resize(rgb.size))
        return rgb


def check_adapter_weights(weights: dict, unet_keys: set, encoder_keys: set) -> None:
    """
    The pipeline loads the adapter with strict=False, which would leave a mismatched file's layers
    as copies of SDXL's own and draw six unrelated pictures. Every tensor in the file must land, and
    every multi-view, reference and camera-encoder layer must get one.
    """
    unused = sorted(key for key in weights if key not in unet_keys and key not in encoder_keys)
    adapter_layers = {key for key in unet_keys if "_mv" in key or "_ref" in key} | set(encoder_keys)
    missing = sorted(adapter_layers - set(weights))
    if unused or missing:
        raise RuntimeError(
            f"{ADAPTER_FILE} doesn't match the pipeline: {len(unused)} unused tensors "
            f"(e.g. {unused[:2]}), {len(missing)} adapter layers without weights (e.g. {missing[:2]})"
        )


class MultiViewGenerator:
    """Loads everything once (about 10 GB of weights); each call turns a picture into six RGBA views."""

    def __init__(
        self,
        models_root: str,
        device: str = "cuda",
        steps: int = STEPS,
        guidance: float = GUIDANCE,
        remover: Optional[Remover] = None,
    ) -> None:
        import safetensors.torch
        import torch
        from diffusers import AutoencoderKL

        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained(os.path.join(models_root, VAE_DIR), torch_dtype=dtype)
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            os.path.join(models_root, SDXL_DIR),
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            add_watermarker=False,  # SDXL's invisible watermark would alter the views' pixels
        )
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler, shift_mode="interpolated", shift_scale=SHIFT_SCALE
        )
        # Upstream first copies SDXL's attention weights into the new layers (a training start) by
        # cloning the whole UNet, 5 GB more RAM. The adapter file overwrites every one of those layers
        # (check_adapter_weights), so skipping the copy gives the same weights.
        pipe.init_custom_adapter(num_views=len(AZIMUTHS), copy_attn_weights=False)
        weights = safetensors.torch.load_file(os.path.join(models_root, ADAPTER_DIR, ADAPTER_FILE))
        check_adapter_weights(weights, set(pipe.unet.state_dict()), set(pipe.cond_encoder.state_dict()))
        pipe.load_custom_adapter(weights, weight_name=ADAPTER_FILE)
        del weights
        pipe.to(device=device, dtype=dtype)
        pipe.cond_encoder.to(device=device, dtype=dtype)
        pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)  # 50 lines a job in the logs otherwise

        self.pipe = pipe
        self.device = device
        self.steps = steps
        self.guidance = guidance
        self.remover = remover or BiRefNetRemover(os.path.join(models_root, BIREFNET_DIR), device)
        self.control = control_images(device)
        # The last job's inputs and outputs, for experiments and debugging
        self.last_reference: Optional[Image.Image] = None
        self.last_views: list[Image.Image] = []
        self.last_timings: dict[str, float] = {}

    def __call__(
        self,
        image: Image.Image,
        seed: int,
        prompt: str = DEFAULT_PROMPT,
        *,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        fill: float = FILL,
    ) -> list[Image.Image]:
        """The six views (AZIMUTHS order) as 768 x 768 RGBA cutouts."""
        import torch

        clock = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round(now - clock, 3)
            clock = now

        reference = prepare_reference(cut_out(image, self.remover), fill=fill)
        lap("cutout_s")
        views = self.pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=steps or self.steps,
            guidance_scale=self.guidance if guidance is None else guidance,
            num_images_per_prompt=len(AZIMUTHS),
            control_image=self.control,
            control_conditioning_scale=1.0,
            reference_image=reference,
            reference_conditioning_scale=REFERENCE_SCALE,
            negative_prompt=NEGATIVE_PROMPT,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images
        lap("views_s")
        cutouts = [self.remover(view) for view in views]
        lap("view_cutouts_s")
        self.last_reference, self.last_views, self.last_timings = reference, views, timings
        return cutouts
