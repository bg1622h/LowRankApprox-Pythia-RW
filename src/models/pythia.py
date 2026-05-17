from enum import Enum

from transformers import AutoTokenizer, GPTNeoXForCausalLM


class PythiaAttentionBackend(Enum):
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    FLEX = "flex_attention"


class PythiaSize(Enum):
    P14M = ("14M", 14_067_712, 1_189_888)
    P31M = ("31M", 30_494_720, 4_739_072)
    P70M = ("70M", 70_426_624, 18_915_328)
    P160M = ("160M", 162_322_944, 85_056_000)
    P410M = ("410M", 405_334_016, 302_311_424)
    P1B = ("1B", 1_011_781_632, 805_736_448)
    P1_4B = ("1.4B", 1_414_647_808, 1_208_602_624)
    P2_8B = ("2.8B", 2_775_208_960, 2_517_652_480)
    P6_9B = ("6.9B", 6_857_302_016, 6_444_163_072)
    P12B = ("12B", 11_846_072_320, 11_327_027_200)

    def __init__(
        self,
        suffix: str,
        total_params: int,
        non_embedding_params: int,
    ):
        self.suffix = suffix
        self.total_params = total_params
        self.non_embedding_params = non_embedding_params

    @property
    def model_name(self) -> str:
        return f"EleutherAI/pythia-{self.suffix.lower()}"

    @property
    def old_suffix(self) -> str:
        old_map = {
            "70M": "19M",
            "160M": "125M",
            "410M": "350M",
            "1B": "800M",
            "1.4B": "1.3B",
            "2.8B": "2.7B",
            "6.9B": "6.7B",
            "12B": "13B",
        }
        return old_map.get(self.suffix, "—")

    @classmethod
    def from_suffix(cls, suffix: str):
        for size in cls:
            if size.suffix == suffix:
                return size
        raise ValueError(f"Unknown suffix: {suffix}")


class Pythia:
    _models = {}
    _tokenizers = {}

    @classmethod
    def get_model(
        cls,
        size: PythiaSize = PythiaSize.P31M,
        revision: str = "step0",
        attention_backend: PythiaAttentionBackend = None,
        torch_dtype=None,
        gradient_checkpointing: bool = False,
    ):
        if attention_backend is None:
            attention_backend = PythiaAttentionBackend.SDPA

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
                "cache_dir": f"./pythia-{size.suffix.lower()}/{revision}",
                "attn_implementation": attention_backend.value,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            model = GPTNeoXForCausalLM.from_pretrained(
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
        size: PythiaSize = PythiaSize.P31M,
        revision: str = "step0",
    ):
        if size not in cls._tokenizers:
            cls._tokenizers[size] = AutoTokenizer.from_pretrained(
                size.model_name,
                revision=revision,
                cache_dir=f"./pythia-{size.suffix.lower()}/{revision}",
            )
            if cls._tokenizers[size].pad_token is None:
                eos_token = cls._tokenizers[size].eos_token
                cls._tokenizers[size].pad_token = eos_token
        return cls._tokenizers[size]
