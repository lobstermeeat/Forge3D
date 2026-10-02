"""
The model wrapper. Always: its calls into transformers, through a stand-in. With transformers 4.57 or later
installed (and torchvision): a tiny random Qwen3-VL, dense and mixture-of-experts, saved and loaded the way
the real ones are and judged with end to end on the CPU, and the image token budget under the pinned
models' image settings. The real weights never load here.
"""

import importlib.util
import json
import sys
import types

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from judge_worker import judge as J  # noqa: E402
from judge_worker import model as M  # noqa: E402


def _qwen3_vl():
    if importlib.util.find_spec("torchvision") is None:  # the processors' image and video transforms
        return None
    try:
        import transformers
    except ImportError:
        return None
    return transformers if hasattr(transformers, "Qwen3VLForConditionalGeneration") else None


needs_qwen3_vl = pytest.mark.skipif(_qwen3_vl() is None, reason="transformers >= 4.57 (Qwen3-VL) and torchvision aren't installed")


# --- The calls into transformers, through a stand-in ---------------------------------------------------


def fake_transformers(calls):
    class Batch(dict):
        def to(self, device):
            calls.append(("to", device))
            return self

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, path, **options):
            calls.append(("model", cls.__name__, path, options))
            return cls()

        def eval(self):
            calls.append(("eval",))
            return self

        def generate(self, **options):
            calls.append(("generate", {k: v for k, v in options.items() if k != "input_ids"}))
            return torch.cat([options["input_ids"], torch.tensor([[5, 6, 7]])], 1)

        def modules(self):
            return [self]

    class Qwen3VLMoeTextExperts:
        def train(self):
            calls.append(("experts train",))

    class Qwen3VLForConditionalGeneration(Model):
        pass

    class Qwen3VLMoeForConditionalGeneration(Model):
        def modules(self):
            return [self, Qwen3VLMoeTextExperts(), Qwen3VLMoeTextExperts()]

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path):
            calls.append(("processor", path))
            return cls()

        def apply_chat_template(self, messages, **options):
            calls.append(("template", messages, options))
            return Batch(input_ids=torch.zeros((1, 11), dtype=torch.long), attention_mask=torch.ones((1, 11)))

        def decode(self, ids, **options):
            calls.append(("decode", ids.tolist(), options))
            return '{"K": "publish", "best": "K"}'

    module = types.ModuleType("transformers")
    module.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    module.Qwen3VLMoeForConditionalGeneration = Qwen3VLMoeForConditionalGeneration
    module.AutoProcessor = AutoProcessor
    return module


@pytest.mark.parametrize("which, model_class", [("8b", "Qwen3VLForConditionalGeneration"), ("30b", "Qwen3VLMoeForConditionalGeneration")])
def test_each_size_loads_its_class_from_its_folder(monkeypatch, which, model_class):
    calls = []
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers(calls))
    qwen = M.QwenVL(which, "/models")
    folder = f"/models/{M.MODELS[which].folder}"
    assert calls[0] == ("model", model_class, folder, {"dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": "sdpa"})
    # The mixture of experts runs only the experts its tokens use (only_used_experts)
    switched = [("experts train",)] * 2 if which == "30b" else []
    assert calls[1:] == [("eval",), *switched, ("processor", folder)]
    assert qwen.path == folder and qwen.which == which and qwen.last_tokens is None


def test_a_call_is_one_greedy_generation_decoded_from_the_reply_on(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers(calls))
    qwen = M.QwenVL("8b", "/models", device="cuda:1", dtype="float16", attention="eager")
    assert calls[0][3] == {"dtype": torch.float16, "device_map": "cuda:1", "attn_implementation": "eager"}
    calls.clear()
    chat = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert qwen(chat, max_new_tokens=99) == '{"K": "publish", "best": "K"}'
    template, moved, generate, decode = calls
    assert template == ("template", chat, {"tokenize": True, "add_generation_prompt": True, "return_dict": True, "return_tensors": "pt"})
    assert moved == ("to", "cuda:0")  # the inputs go where the model is
    # Greedy whatever the model's own generation config says (Qwen3-VL's samples)
    assert generate == ("generate", {"attention_mask": generate[1]["attention_mask"], "max_new_tokens": 99, "do_sample": False, "temperature": None, "top_p": None, "top_k": None})
    assert decode == ("decode", [5, 6, 7], {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})
    assert qwen.last_tokens == {"prompt": 11, "reply": 3}


def test_only_the_two_sizes(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers([]))
    with pytest.raises(ValueError, match="8b, 30b, not '70b'"):
        M.QwenVL("70b")
    assert M.spec("30b").moe and not M.spec("8b").moe


# --- A tiny random Qwen3-VL, with transformers ----------------------------------------------------------

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
# Qwen3-VL's chat template, cut down to what the judge sends: one user turn of text and images
TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' }}"
    "{%- if message.content is string %}{{- message.content }}{%- else %}{%- for content in message.content %}"
    "{%- if content.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{%- elif content.type == 'text' %}{{- content.text }}{%- endif %}{%- endfor %}{%- endif %}"
    "{{- '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)
# The pinned models' preprocessor_config.json (both are the same)
IMAGE_SETTINGS = dict(
    patch_size=16,
    temporal_patch_size=2,
    merge_size=2,
    size={"shortest_edge": 65536, "longest_edge": 16777216},
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
)


def tiny_folder(folder, moe):
    """A random Qwen3-VL a few layers deep, its byte-level tokenizer and its processor, saved like the real ones."""
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        GenerationConfig,
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLProcessor,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    byte_level = Tokenizer(models.BPE(vocab={char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}, merges=[]))
    byte_level.pre_tokenizer = ByteLevel(add_prefix_space=False)
    byte_level.decoder = decoders.ByteLevel()
    byte_level.add_special_tokens(SPECIALS)
    tokenizer = Qwen2TokenizerFast(tokenizer_object=byte_level, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token=None, bos_token=None)
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIALS}
    # Small images stay small (the real minimum is 65536 pixels)
    settings = {**IMAGE_SETTINGS, "size": {"shortest_edge": 32 * 32, "longest_edge": 512 * 512}}
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(**settings), tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=TEMPLATE
    )
    text = dict(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=8192, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"],
    )
    vision = dict(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32, patch_size=16, spatial_merge_size=2,
        temporal_patch_size=2, num_position_embeddings=64, deepstack_visual_indexes=[0, 1],
    )
    tokens = dict(
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
    )
    torch.manual_seed(0)
    if moe:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, decoder_sparse_step=1, mlp_only_layers=[])
        model = Qwen3VLMoeForConditionalGeneration(Qwen3VLMoeConfig(text_config=text, vision_config=vision, **tokens))
    else:
        model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(text_config=text, vision_config=vision, **tokens))
    # Sampling by default, as Qwen3-VL's own generation_config.json
    model.generation_config = GenerationConfig(
        do_sample=True, top_k=20, top_p=0.8, temperature=0.7, bos_token_id=ids["<|endoftext|>"],
        pad_token_id=ids["<|endoftext|>"], eos_token_id=[ids["<|im_end|>"], ids["<|endoftext|>"]],
    )
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    return ids


@needs_qwen3_vl
@pytest.mark.parametrize("which", ["8b", "30b"])
def test_a_tiny_qwen3_vl_judges_end_to_end_on_the_cpu(tmp_path, which):
    ids = tiny_folder(tmp_path, moe=M.MODELS[which].moe)
    qwen = M.QwenVL(which, path=str(tmp_path), device="cpu", dtype="float32")
    assert type(qwen.model).__name__ == ("Qwen3VLMoeForConditionalGeneration" if which == "30b" else "Qwen3VLForConditionalGeneration")
    picture = Image.new("RGB", (64, 64), (240, 240, 240))
    grids = [Image.new("RGB", (96, 64), colour) for colour in ((200, 30, 30), (30, 200, 30), (30, 30, 200))]

    # The processor takes the judge's chat as it is: every image in, each expanded to its tokens
    from judge_worker import prompt

    chat = prompt.messages(picture, grids, "a lamp", ["K", "L", "M"])
    inputs = qwen.processor.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    per_image = (inputs["image_grid_thw"].prod(-1) // 4).tolist()
    assert per_image == [4, 6, 6, 6]  # 64 x 64 and 96 x 64 pixels at 32 x 32 a token
    assert int((inputs["input_ids"] == ids["<|image_pad|>"]).sum()) == sum(per_image)
    assert int((inputs["input_ids"] == ids["<|vision_start|>"]).sum()) == 4

    # Greedy: the same reply whatever the random state, though the model's config samples
    results = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        results.append(J.judge(picture, grids, "a lamp", model=qwen, max_new_tokens=12))
    first, second = results
    assert first["raw"] == second["raw"]
    assert first["tokens"] == {"prompt": int(inputs["input_ids"].shape[1]), "reply": first["tokens"]["reply"]}
    assert 1 <= first["tokens"]["reply"] <= 12
    # A random model's reply is no answer: the generation's own texture, with the reason
    assert first["best"] == 0 and first["parse_error"] and first["verdicts"] == [None, None, None]
    json.dumps(first)  # plain data, as a Modal call returns it


@needs_qwen3_vl
def test_the_image_token_budget_under_the_pinned_settings():
    from transformers import Qwen2VLImageProcessorFast

    processor = Qwen2VLImageProcessorFast(**IMAGE_SETTINGS)
    picture = J.fit(Image.new("RGB", (1024, 1024)), J.PICTURE_SIDE)
    grid = J.fit(Image.new("RGB", (1152, 768)), J.GRID_SIDE, J.GRID_PIXELS)
    big_grid = J.fit(Image.new("RGB", (1536, 1024)), J.GRID_SIDE, J.GRID_PIXELS)  # the reviewers' grids
    out = processor(images=[picture, grid, big_grid], return_tensors="pt")
    tokens = (out["image_grid_thw"].prod(-1) // 4).tolist()
    assert tokens == [576, 864, 864]
    # The picture and four grids: about 4,000 image tokens and 1,000 of text
    assert tokens[0] + 4 * tokens[1] == 4032


@needs_qwen3_vl
def test_the_mixture_of_experts_runs_only_the_experts_used_and_gets_the_same_answer(tmp_path):
    tiny_folder(tmp_path, moe=True)
    qwen = M.QwenVL("30b", path=str(tmp_path), device="cpu", dtype="float32")
    experts = [module for module in qwen.model.modules() if type(module).__name__ == "Qwen3VLMoeTextExperts"]
    assert len(experts) == 2 and all(module.training for module in experts)
    assert not qwen.model.training and not qwen.model.model.visual.training  # the rest is in eval
    from judge_worker import prompt

    picture, grids = Image.new("RGB", (64, 64), (240, 240, 240)), [Image.new("RGB", (96, 64), (200, 30, 30))] * 2
    chat = prompt.messages(picture, grids, "a lamp", ["K", "L"])
    inputs = qwen.processor.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    with torch.inference_mode():
        used = qwen.model(**inputs).logits
        for module in experts:
            module.eval()  # transformers' default: every expert on every token
        every = qwen.model(**inputs).logits
    torch.testing.assert_close(used, every, atol=1e-5, rtol=1e-4)
    assert M.only_used_experts(qwen.model) == 2
