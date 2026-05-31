from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


class SmokeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        ids = [(byte % 250) + 2 for byte in text.encode("utf-8")]
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [
            int(token)
            for token in token_ids
            if not skip_special_tokens or int(token) > self.eos_token_id
        ]
        return "".join(chr((token - 2) % 250) for token in tokens)


class LlamaAttentionBackend(Enum):
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    FLEX = "flex_attention"


class LlamaSize(Enum):
    SMOKE = "sshleifer/tiny-gpt2"
    TINY1_1B = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    L1B = "meta-llama/Llama-3.2-1B-Instruct"
    L3B = "meta-llama/Llama-3.2-3B"
    L8B = "meta-llama/Meta-Llama-3.1-8B"

    @property
    def model_name(self) -> str:
        if self is LlamaSize.TINY1_1B:
            return "./llama-tiny1_1b/main"
        return self.value

    @property
    def suffix(self) -> str:
        if self.name.startswith("L"):
            return self.name[1:]
        return self.name

    @classmethod
    def from_suffix(cls, suffix: str):
        normalized = suffix.upper().replace(".", "_").replace("-", "_")
        smoke_aliases = {"SMOKE", "TINY_GPT2", "TINYGPT2", "TEST"}
        if normalized in smoke_aliases:
            return cls.SMOKE
        tiny_aliases = {"1B", "1_1B", "TINY1_1B", "TINYLLAMA"}
        if normalized in tiny_aliases:
            return cls.TINY1_1B

        for size in cls:
            if size.suffix.upper() == normalized or size.name.upper() == normalized:
                return size
        raise ValueError(f"Unknown Llama suffix: {suffix}")


class Llama:
    _models = {}
    _tokenizers = {}

    @classmethod
    def clear_model_cache(cls):
        cls._models.clear()

    @classmethod
    def get_model(
        cls,
        size: LlamaSize = LlamaSize.L1B,
        revision: str = "main",
        attention_backend: LlamaAttentionBackend = None,
        torch_dtype=None,
        gradient_checkpointing: bool = False,
    ):
        if attention_backend is None:
            attention_backend = LlamaAttentionBackend.SDPA

        cache_key = (
            size,
            revision,
            attention_backend,
            str(torch_dtype),
            gradient_checkpointing,
        )
        if cache_key not in cls._models:
            if size is LlamaSize.SMOKE:
                config = GPT2Config(
                    vocab_size=256,
                    n_positions=128,
                    n_embd=64,
                    n_layer=2,
                    n_head=4,
                    bos_token_id=1,
                    eos_token_id=1,
                    pad_token_id=0,
                )
                model = GPT2LMHeadModel(config)
                if torch_dtype is not None:
                    model = model.to(dtype=torch_dtype)
                if gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                cls._models[cache_key] = model
                return model

            model_kwargs = {
                "revision": revision,
                "cache_dir": f"./llama-{size.suffix.lower()}/{revision}",
                "attn_implementation": attention_backend.value,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    size.model_name,
                    **model_kwargs,
                )
            except (TypeError, ValueError) as exc:
                if size is not LlamaSize.SMOKE:
                    raise
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(
                    size.model_name,
                    **model_kwargs,
                )
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
            cls._models[cache_key] = model
        return cls._models[cache_key]

    @classmethod
    def get_tokenizer(
        cls,
        size: LlamaSize = LlamaSize.L1B,
        revision: str = "main",
    ):
        cache_key = (size, revision)
        if cache_key not in cls._tokenizers:
            if size is LlamaSize.SMOKE:
                cls._tokenizers[cache_key] = SmokeTokenizer()
                return cls._tokenizers[cache_key]

            tokenizer = AutoTokenizer.from_pretrained(
                size.model_name,
                revision=revision,
                cache_dir=f"./llama-{size.suffix.lower()}/{revision}",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            cls._tokenizers[cache_key] = tokenizer
        return cls._tokenizers[cache_key]
