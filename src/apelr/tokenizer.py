from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
SPECIAL_TOKEN_SET = set(SPECIAL_TOKENS)


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        vocab = SPECIAL_TOKENS + chars
        stoi = {ch: idx for idx, ch in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        pieces: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            tok = self.itos[idx]
            if tok in SPECIAL_TOKEN_SET:
                continue
            pieces.append(tok)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos}
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = list(payload["itos"])
        stoi = {ch: idx for idx, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class TokenizerLike(Protocol):
    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]: ...
    def decode(self, ids: list[int] | tuple[int, ...]) -> str: ...
    def save(self, path: str | Path) -> None: ...


@dataclass
class BPETokenizer:
    tokenizer: object
    special_tokens: list[str]

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        byte_level: bool = True,
        lowercase: bool = False,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        except Exception as exc:
            raise RuntimeError(
                "BPE tokenizer requires the 'tokenizers' package. Install dependencies and retry."
            ) from exc

        special = list(special_tokens or SPECIAL_TOKENS)
        if "<unk>" not in special:
            special.append("<unk>")

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        norm_steps: list[object] = [normalizers.NFKC()]
        if lowercase:
            norm_steps.append(normalizers.Lowercase())
        tok.normalizer = normalizers.Sequence(norm_steps)
        if byte_level:
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
        else:
            tok.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer=tok, special_tokens=special)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except Exception as exc:
            raise RuntimeError(
                "BPE tokenizer requires the 'tokenizers' package. Install dependencies and retry."
            ) from exc
        tok = Tokenizer.from_file(str(path))
        return cls(tokenizer=tok, special_tokens=list(special_tokens or SPECIAL_TOKENS))

    @property
    def pad_id(self) -> int:
        tok_id = self.tokenizer.token_to_id("<pad>")
        if tok_id is None:
            raise ValueError("BPE tokenizer missing <pad> token.")
        return int(tok_id)

    @property
    def bos_id(self) -> int:
        tok_id = self.tokenizer.token_to_id("<bos>")
        if tok_id is None:
            raise ValueError("BPE tokenizer missing <bos> token.")
        return int(tok_id)

    @property
    def eos_id(self) -> int:
        tok_id = self.tokenizer.token_to_id("<eos>")
        if tok_id is None:
            raise ValueError("BPE tokenizer missing <eos> token.")
        return int(tok_id)

    @property
    def unk_id(self) -> int:
        tok_id = self.tokenizer.token_to_id("<unk>")
        if tok_id is None:
            raise ValueError("BPE tokenizer missing <unk> token.")
        return int(tok_id)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        enc = self.tokenizer.encode(text)
        ids = [int(t) for t in enc.ids]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        filtered = [int(i) for i in ids if int(i) not in special_ids and int(i) >= 0]
        return str(self.tokenizer.decode(filtered, skip_special_tokens=True))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


def _detect_tokenizer_type(path: str | Path) -> str:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "bpe"
    if isinstance(payload, dict) and "itos" in payload:
        return "char"
    return "bpe"


def load_tokenizer(
    path: str | Path,
    *,
    tokenizer_type: str | None = None,
    special_tokens: list[str] | None = None,
) -> TokenizerLike:
    tok_type = (tokenizer_type or _detect_tokenizer_type(path)).lower()
    if tok_type == "char":
        return CharTokenizer.load(path)
    if tok_type == "bpe":
        return BPETokenizer.load(path, special_tokens=special_tokens)
    raise ValueError(f"Unsupported tokenizer_type '{tok_type}'.")
