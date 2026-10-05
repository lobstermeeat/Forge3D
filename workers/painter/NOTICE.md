# Third-party notices: painter worker

The view painter repaints grey renders of Orainge's own meshes after a reference picture (Phase 8). Its outputs
are images; it redistributes no weights.

| Component                                                                                                                                                                                                                                  | Used for                                                    | License      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------ |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) weights, revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`: the MMDiT transformer, the Qwen2.5-VL-7B text encoder, the VAE, the tokenizer and the processor | Repainting the renders                                      | Apache-2.0   |
| [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) weights, revision `d74eba145674fd7e31b949324e148e21e7118abd`: the 8-step LoRA only (`…-8steps-V1.0-fp32.safetensors`, and its bf16 copy) | 8-step sampling, fused into the transformer                 | Apache-2.0   |
| [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning)'s scheduler settings (`generate_with_diffusers.py`), copied into `painter_worker/qwen.py` as `LIGHTNING_SCHEDULER`                                                | The LoRA's sampling schedule                                | Apache-2.0   |
| [diffusers](https://github.com/huggingface/diffusers) 0.37.1 (`QwenImageEditPlusPipeline`), [transformers](https://github.com/huggingface/transformers) 4.57.6 (Qwen2.5-VL), [peft](https://github.com/huggingface/peft), [accelerate](https://github.com/huggingface/accelerate), [tokenizers](https://github.com/huggingface/tokenizers), [safetensors](https://github.com/huggingface/safetensors), [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Runtime | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision)                                                                                                                                           | Runtime                                                     | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow)                                                                                                                                                                                          | Image handling                                              | MIT-CMU      |

Both model cards give `license: apache-2.0` at the pinned revisions (Qwen's card: "Qwen-Image is licensed under
Apache 2.0"); neither repository has a separate license file, and both hold weights and configs only (the model
code is diffusers' and transformers'). The text encoder inside Qwen-Image-Edit-2511 is Qwen2.5-VL-7B-Instruct, which
is Apache-2.0 on its own too. Apache-2.0 allows commercial use; redistributing the weights would need the license
and notices passed on, and the painter doesn't redistribute them. Not used: the Lightning repository's 4-step LoRAs
and its fp8 models, and Qwen models under the Qwen or Qwen Research licenses, which restrict commercial use.

The prompts in `painter_worker/qwen.py` are Orainge's own.

This summary is not legal advice; have counsel confirm it before launch.
