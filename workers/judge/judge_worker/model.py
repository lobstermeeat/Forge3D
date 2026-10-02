"""
Qwen3-VL (Apache-2.0) through transformers: loaded once per container, then one chat call per judgement.

Two sizes behind one setting: "8b" (Qwen3-VL-8B-Instruct, dense, 17.5 GB in bf16: an L40S) and "30b"
(Qwen3-VL-30B-A3B-Instruct, a mixture of experts with 3B of its 31B parameters active per token, 62 GB in
bf16: an H100). Their weights come pinned from scripts/download_weights.py into the models volume. The
usage is the model cards': ``<class>.from_pretrained(path, dtype=…, device_map=…)``, ``AutoProcessor``,
``processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
return_tensors="pt")`` and ``generate``.

Decoding is greedy. Both models ship a generation config that samples (temperature 0.7, top-p 0.8,
top-k 20); the arguments to ``generate`` override it (transformers applies them last), so the same
images and prompt always give the same answer on the same hardware and software. The mixture of experts
runs only the experts each token uses (``only_used_experts``), which keeps a judgement's memory to the
weights and a few GB.

transformers is imported only when a model is built, so the rest of the package (and its tests) runs
without it. ``QwenVL(path=…)`` loads any Qwen3-VL folder, which is how the tests run a tiny random one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

MODELS_ROOT = os.environ.get("JUDGE_MODELS_ROOT", "/models")
MAX_NEW_TOKENS = 1024


@dataclass(frozen=True)
class Spec:
    repo: str  # the Hugging Face repository (its revision is pinned in scripts/download_weights.py)
    folder: str  # where the download script puts it under the models root
    moe: bool  # a mixture of experts: Qwen3VLMoeForConditionalGeneration


MODELS = {
    "8b": Spec("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct", moe=False),
    "30b": Spec("Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-30B-A3B-Instruct", moe=True),
}


def spec(which: str) -> Spec:
    if which not in MODELS:
        raise ValueError(f"the judge model is one of {', '.join(MODELS)}, not {which!r}")
    return MODELS[which]


def only_used_experts(model: Any) -> int:
    """
    Makes a mixture of experts run only the experts each token uses. At inference transformers 4.57 runs
    every expert on every token (all 128 of Qwen3-VL-30B-A3B's, for the 8 a token uses) and weighs the
    unused ones by zero: about 10 GB of temporaries for a 5,000-token prompt, on top of 62 GB of weights
    on an 80 GB card. Its other branch loops over the experts the tokens use, the same sum in another
    order; the experts' training flag picks it, and nothing else in them reads that flag (no dropout).
    Returns how many experts modules were switched.
    """
    switched = 0
    for module in model.modules():
        if type(module).__name__.endswith("TextExperts"):
            module.train()
            switched += 1
    return switched


class QwenVL:
    """
    A loaded Qwen3-VL: call it with a chat (transformers' message format, images as PIL images) for the
    reply's text. ``last_tokens`` then holds the prompt's and the reply's lengths in tokens.
    """

    def __init__(
        self,
        which: str = "8b",
        models_root: str = MODELS_ROOT,
        *,
        path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attention: str = "sdpa",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        self.which = which
        chosen = spec(which)
        self.path = path or os.path.join(models_root, chosen.folder)
        model_class = Qwen3VLMoeForConditionalGeneration if chosen.moe else Qwen3VLForConditionalGeneration
        started = time.perf_counter()
        # device_map puts each weight straight on the GPU as it is read (accelerate), not through RAM first
        self.model = model_class.from_pretrained(
            self.path, dtype=getattr(torch, dtype), device_map=device, attn_implementation=attention
        ).eval()
        if chosen.moe:
            only_used_experts(self.model)
        self.processor = AutoProcessor.from_pretrained(self.path)
        self.load_seconds = round(time.perf_counter() - started, 1)
        self.last_tokens: Optional[dict] = None

    def __call__(self, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        import torch

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # The model's sampling settings would only be ignored, with a warning each call
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        reply = output[0, prompt_tokens:]
        self.last_tokens = {"prompt": prompt_tokens, "reply": int(reply.shape[0])}
        return self.processor.decode(reply, skip_special_tokens=True, clean_up_tokenization_spaces=False)


_loaded: Optional[Any] = None


def load(which: str = "8b", models_root: str = MODELS_ROOT, **options: Any) -> QwenVL:
    """Loads a model as this process's judge (``judge()`` uses it when given none) and returns it."""
    global _loaded
    _loaded = QwenVL(which, models_root, **options)
    return _loaded


def loaded() -> Any:
    """This process's judge model, from load()."""
    if _loaded is None:
        raise RuntimeError("no judge model is loaded: call judge_worker.model.load() first, or pass model=")
    return _loaded
