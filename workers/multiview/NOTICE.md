# Third-party notices: multiview worker

| Component                                                                                                         | Used for                                       | License                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [MV-Adapter](https://github.com/huanngzh/MV-Adapter) code (vendored subset in `mvadapter/`, see its `NOTICE.md`)  | Image-to-multiview pipeline                    | Apache-2.0                                                                                                |
| [MV-Adapter weights](https://huggingface.co/huanngzh/mv-adapter) (`mvadapter_i2mv_sdxl.safetensors`)              | The multi-view adapter                         | Apache-2.0                                                                                                |
| [Stable Diffusion XL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (fp16 files)      | The image model the adapter extends            | [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) |
| [SDXL-VAE-FP16-Fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)                                         | Decoding the views in fp16                     | MIT                                                                                                       |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)                                                            | Cutting out the picture and the views          | MIT                                                                                                       |
| [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate) | Runtime | Apache-2.0                                            |
| [PyTorch](https://github.com/pytorch/pytorch), [timm](https://github.com/huggingface/pytorch-image-models), [kornia](https://github.com/kornia/kornia), [einops](https://github.com/arogozhnikov/einops) | Runtime | BSD-3-Clause, Apache-2.0, Apache-2.0, MIT |

**SDXL's license carries use-based restrictions**: its Attachment A lists uses that are not allowed
(unlawful uses, harming minors, harassment, harmful disinformation and others). Running it in a
service is allowed, but paragraph 5 requires Orainge to bind its users to those restrictions, so
the terms of service must include them before this worker serves users. Nothing in the license
asks for attribution on the outputs.

Not used: MV-Adapter's mesh tools and texturing pipeline, which need nvdiffrast (research use
only) and can segment with RMBG-2.0 (CC BY-NC).

This summary is not legal advice; have counsel confirm it before launch.
