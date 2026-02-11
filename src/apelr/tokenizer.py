from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
