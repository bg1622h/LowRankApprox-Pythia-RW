from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaAttentionBackend(Enum):
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    FLEX = "flex_attention"


class LlamaSize(Enum):
    L1B = "meta-llama/Llama-3.2-1B"
    L3B = "meta-llama/Llama-3.2-3B"
    L8B = "meta-llama/Meta-Llama-3.1-8B"

    @property
    def model_name(self) -> str:
        return self.value

    @property
    def suffix(self) -> str:
        return self.name[1:]

    @classmethod
    def from_suffix(cls, suffix: str):
        normalized = suffix.upper().replace(".", "_")
        for size in cls:
            if size.suffix.upper() == normalized:
                return size
        raise ValueError(f"Unknown Llama suffix: {suffix}")


class Llama:
    _models = {}
    _tokenizers = {}

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
            model_kwargs = {
                "revision": revision,
                "cache_dir": f"./llama-{size.suffix.lower()}/{revision}",
                "attn_implementation": attention_backend.value,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

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
            tokenizer = AutoTokenizer.from_pretrained(
                size.model_name,
                revision=revision,
                cache_dir=f"./llama-{size.suffix.lower()}/{revision}",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            cls._tokenizers[cache_key] = tokenizer
        return cls._tokenizers[cache_key]
