import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

DEFAULT_DATASET_NAME = "tiiuae/falcon-refinedweb"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMNS = ("content", "text")
DEFAULT_SHARD_SIZE_BYTES = 512 * 1024 * 1024


@dataclass
class RefinedWebCacheConfig:
    output_dir: str
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = DEFAULT_SPLIT
    text_column: Optional[str] = None
    max_bytes: Optional[int] = None
    max_documents: Optional[int] = None
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES
    streaming: bool = True
    seed: int = 42
    shuffle_buffer_size: Optional[int] = None
    token: Optional[str] = None


@dataclass
class CachedDatasetStats:
    documents: int = 0
    text_bytes: int = 0
    disk_bytes: int = 0
    tokens: Optional[int] = None


def _extract_text(example: dict, text_column: Optional[str] = None) -> str:
    if text_column is not None:
        text = example.get(text_column, "")
        return text if isinstance(text, str) else ""

    for column in DEFAULT_TEXT_COLUMNS:
        text = example.get(column, "")
        if isinstance(text, str) and text:
            return text

    for value in example.values():
        if isinstance(value, str) and value:
            return value

    return ""


def _jsonl_files(data_dir: str | Path) -> list[Path]:
    path = Path(data_dir)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def _iter_jsonl_records(files: Iterable[Path]) -> Iterator[dict]:
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _split_files_for_worker(files: list[Path]) -> list[Path]:
    worker = get_worker_info()
    if worker is None:
        return files
    return [
        file
        for index, file in enumerate(files)
        if index % worker.num_workers == worker.id
    ]


def cache_refinedweb_subset(
    config: RefinedWebCacheConfig,
) -> CachedDatasetStats:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        config.dataset_name,
        split=config.split,
        streaming=config.streaming,
        token=config.token,
    )
    if config.shuffle_buffer_size:
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer_size,
        )

    stats = CachedDatasetStats()
    shard_id = 0
    shard_bytes = 0
    shard = None

    def open_shard():
        path = output_dir / f"{config.split}-{shard_id:05d}.jsonl"
        return path.open("w", encoding="utf-8")

    try:
        shard = open_shard()
        for example in dataset:
            text = _extract_text(example, config.text_column)
            if not text:
                continue

            record = {"text": text}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))
            text_bytes = len(text.encode("utf-8"))

            next_shard_bytes = shard_bytes + line_bytes
            shard_would_overflow = next_shard_bytes > config.shard_size_bytes
            if shard_bytes > 0 and shard_would_overflow:
                shard.close()
                shard_id += 1
                shard_bytes = 0
                shard = open_shard()

            shard.write(line)
            shard_bytes += line_bytes
            stats.documents += 1
            stats.text_bytes += text_bytes
            stats.disk_bytes += line_bytes

            max_documents = config.max_documents
            if max_documents and stats.documents >= max_documents:
                break
            if config.max_bytes and stats.text_bytes >= config.max_bytes:
                break
    finally:
        if shard is not None:
            shard.close()

    metadata = {
        "config": asdict(config),
        "stats": asdict(stats),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return stats


class LocalJsonlTextDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        text_column: Optional[str] = "text",
        shuffle_files: bool = False,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.text_column = text_column
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[str]:
        files = _split_files_for_worker(_jsonl_files(self.data_dir))
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(files)

        for record in _iter_jsonl_records(files):
            text = _extract_text(record, self.text_column)
            if text:
                yield text


class RefinedWebStreamingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_length: int,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        text_column: Optional[str] = None,
        shuffle_buffer_size: Optional[int] = None,
        seed: int = 42,
        token: Optional[str] = None,
        drop_last: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.token = token
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            token=self.token,
        )
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.shuffle_buffer_size,
            )

        def texts():
            for example in dataset:
                yield _extract_text(example, self.text_column)

        yield from _pack_texts(
            texts(),
            self.tokenizer,
            self.seq_length,
            drop_last=self.drop_last,
        )


class PackedCausalLMDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        seq_length: int,
        text_column: Optional[str] = "text",
        shuffle_files: bool = False,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.text_dataset = LocalJsonlTextDataset(
            data_dir=data_dir,
            text_column=text_column,
            shuffle_files=shuffle_files,
            seed=seed,
        )
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from _pack_texts(
            self.text_dataset,
            self.tokenizer,
            self.seq_length,
            drop_last=self.drop_last,
        )


TokenizedCausalLMDataset = PackedCausalLMDataset


def _tokenize_document(
    text: str,
    tokenizer,
    seq_length: int,
) -> list[list[int]]:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    eos_token_id = tokenizer.eos_token_id
    chunk_size = seq_length
    if eos_token_id is not None:
        if seq_length < 2:
            raise ValueError("seq_length must fit at least one token and EOS")
        chunk_size = seq_length - 1

    chunks = []
    for start in range(0, len(token_ids), chunk_size):
        end = start + chunk_size
        chunk = token_ids[start:end]
        if eos_token_id is not None:
            chunk = chunk + [eos_token_id]
        if chunk:
            chunks.append(chunk)
    return chunks


def _make_packed_sample(
    input_ids: list[int],
    position_ids: list[int],
    sequence_ids: list[int],
) -> dict[str, torch.Tensor]:
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    position_tensor = torch.tensor(position_ids, dtype=torch.long)
    sequence_tensor = torch.tensor(sequence_ids, dtype=torch.long)
    labels = input_tensor.clone()
    labels[position_tensor == 0] = -100
    return {
        "input_ids": input_tensor,
        "attention_mask": torch.ones_like(input_tensor),
        "labels": labels,
        "position_ids": position_tensor,
        "sequence_ids": sequence_tensor,
    }


def _pack_texts(
    texts: Iterable[str],
    tokenizer,
    seq_length: int,
    drop_last: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    input_ids: list[int] = []
    position_ids: list[int] = []
    sequence_ids: list[int] = []
    sequence_id = 0

    for text in texts:
        if not text:
            continue
        for chunk in _tokenize_document(text, tokenizer, seq_length):
            if len(input_ids) + len(chunk) > seq_length:
                yield _make_packed_sample(
                    input_ids,
                    position_ids,
                    sequence_ids,
                )
                input_ids = []
                position_ids = []
                sequence_ids = []
                sequence_id = 0

            input_ids.extend(chunk)
            position_ids.extend(range(len(chunk)))
            sequence_ids.extend([sequence_id] * len(chunk))
            sequence_id += 1

            if len(input_ids) == seq_length:
                yield _make_packed_sample(
                    input_ids,
                    position_ids,
                    sequence_ids,
                )
                input_ids = []
                position_ids = []
                sequence_ids = []
                sequence_id = 0

    if input_ids and not drop_last:
        yield _make_packed_sample(input_ids, position_ids, sequence_ids)


def build_packed_attention_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
    _, seq_length = sequence_ids.shape
    token_index = torch.arange(
        seq_length,
        device=sequence_ids.device,
    )
    causal_mask = token_index[:, None] >= token_index[None, :]
    same_sequence = sequence_ids[:, :, None] == sequence_ids[:, None, :]
    valid_tokens = sequence_ids >= 0
    return (
        same_sequence
        & causal_mask
        & valid_tokens[:, :, None]
        & valid_tokens[:, None, :]
    )


def _pad_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(example["input_ids"].numel() for example in batch)
    output = {}
    pad_values = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "labels": -100,
        "position_ids": 0,
        "sequence_ids": -1,
    }
    for key, pad_value in pad_values.items():
        values = []
        for example in batch:
            value = example[key]
            pad_length = max_length - value.numel()
            if pad_length:
                padding = torch.full(
                    (pad_length,),
                    pad_value,
                    dtype=value.dtype,
                )
                value = torch.cat([value, padding])
            values.append(value)
        output[key] = torch.stack(values)
    return output


def _collate_causal_lm(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
    packed_attention: bool,
) -> dict[str, torch.Tensor]:
    output = _pad_batch(batch, pad_token_id)
    if packed_attention:
        output["padding_mask"] = output["attention_mask"]
        output["attention_mask"] = build_packed_attention_mask(
            output["sequence_ids"],
        )
        mask = output["attention_mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        output["attention_mask"] = mask
    return output


def build_refinedweb_dataloader(
    data_dir: str | Path,
    tokenizer,
    seq_length: int,
    batch_size: int,
    text_column: Optional[str] = "text",
    shuffle_files: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    packed_attention: bool = True,
) -> DataLoader:
    dataset = PackedCausalLMDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_length=seq_length,
        text_column=text_column,
        shuffle_files=shuffle_files,
        seed=seed,
        drop_last=drop_last,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: _collate_causal_lm(
            batch,
            pad_token_id,
            packed_attention,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def count_cached_dataset(
    data_dir: str | Path,
    tokenizer=None,
    text_column: Optional[str] = "text",
) -> CachedDatasetStats:
    stats = CachedDatasetStats()
    for record in _iter_jsonl_records(_jsonl_files(data_dir)):
        text = _extract_text(record, text_column)
        if not text:
            continue
        stats.documents += 1
        stats.text_bytes += len(text.encode("utf-8"))
        stats.disk_bytes += len(
            (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        )
        if tokenizer is not None:
            stats.tokens = (stats.tokens or 0) + len(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            if tokenizer.eos_token_id is not None:
                stats.tokens += 1
    return stats
