# Third-party notices: judge worker

Production runs the 8B for texture options, only when a textures job asks for the judge's pick
(`"judge": true`): its pick, verdicts and one-sentence reason reach users as a preselected texture, and it
makes no models or images. The 30B is for experiments only (Phase 7).

| Component                                                                                                                                                                       | Used for                                    | License                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------- |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) weights, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`                                           | Judging textures (the small judge)          | Apache-2.0                  |
| [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) weights, revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`                                 | Judging textures (the large judge)          | Apache-2.0                  |
| [transformers](https://github.com/huggingface/transformers) 4.57.6 (Qwen3-VL's model code), [accelerate](https://github.com/huggingface/accelerate), [tokenizers](https://github.com/huggingface/tokenizers) | Runtime | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision)                                                                                 | Runtime                                     | BSD-3-Clause                |

Both models' cards give `license: apache-2.0` at the pinned revisions; the repositories hold weights,
configs and the tokenizer only (the model code is transformers'). Apache-2.0 allows commercial use;
redistributing the weights would need the license and notices passed on, and the judge doesn't
redistribute them. Not used: Qwen models under the Qwen or Qwen Research licenses, which restrict
commercial use.

The rubric in `judge_worker/prompt.py` is Orainge's own (Phase 7's reviewers' grading prompt, adapted).

This summary is not legal advice; have counsel confirm it before launch.
