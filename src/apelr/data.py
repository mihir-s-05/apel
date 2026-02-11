from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    train_split: str
    val_split: str
    text_field: str
    default_streaming: bool = False


DATASET_SPECS: dict[str, DatasetSpec] = {
    "tinystories": DatasetSpec(
        path="roneneldan/TinyStories",
        name=None,
        train_split="train",
        val_split="validation",
        text_field="text",
        default_streaming=False,
    ),
    "wikitext103": DatasetSpec(
        path="Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        train_split="train",
        val_split="validation",
        text_field="text",
        default_streaming=False,
    ),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        train_split="train",
        val_split="train",
        text_field="text",
        default_streaming=True,
    ),
    "c4_en": DatasetSpec(
        path="allenai/c4",
        name="en",
        train_split="train",
        val_split="validation",
        text_field="text",
        default_streaming=True,
    ),
}


def available_sources() -> list[str]:
    return sorted(DATASET_SPECS.keys())


def get_dataset_spec(source: str) -> DatasetSpec:
    if source not in DATASET_SPECS:
        raise ValueError(f"Unknown source '{source}'. Available: {available_sources()}")
    return DATASET_SPECS[source]


def _iter_rows(
    *,
    source: str,
    split: str,
    max_examples: int,
    streaming: bool | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    spec = get_dataset_spec(source)
    use_streaming = spec.default_streaming if streaming is None else streaming
    if max_examples <= 0:
        return [], use_streaming
    ds = load_dataset(
        path=spec.path,
        name=spec.name,
        split=split,
        streaming=use_streaming,
        cache_dir=cache_dir,
    )
    if use_streaming:
        return list(itertools.islice(ds, max_examples)), use_streaming
    upper = min(max_examples, len(ds))
    return [ds[i] for i in range(upper)], use_streaming


def load_corpus_text(
    *,
    source: str,
    split: str,
    max_examples: int,
    min_chars: int = 8,
    streaming: bool | None = None,
    cache_dir: str | None = None,
) -> tuple[str, dict[str, int]]:
    spec = get_dataset_spec(source)
    rows, use_streaming = _iter_rows(
        source=source,
        split=split,
        max_examples=max_examples,
        streaming=streaming,
        cache_dir=cache_dir,
    )
    kept: list[str] = []
    skipped = 0
    for row in rows:
        text = str(row.get(spec.text_field, ""))
        text = text.strip()
        if len(text) < min_chars:
            skipped += 1
            continue
        kept.append(text)
    corpus = "\n\n".join(kept)
    stats = {
        "num_rows_total": len(rows),
        "num_rows_kept": len(kept),
        "num_rows_skipped": skipped,
        "num_chars": len(corpus),
        "streaming": int(use_streaming),
    }
    return corpus, stats


class PackedSequenceDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if len(token_ids) <= seq_len:
            raise ValueError("token_ids shorter than seq_len")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        max_start = len(token_ids) - (seq_len + 1)
        self.starts = list(range(0, max_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[index]
        e = s + self.seq_len + 1
        block = self.tokens[s:e]
        x = block[:-1]
        y = block[1:]
        return x, y
