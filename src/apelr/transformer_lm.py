from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerLMConfig:
    vocab_size: int
    max_seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.1
    eps: float = 1e-5


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=float(self.cfg.dropout) if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        out = self.resid_drop(self.out(attn))
        return out


class _MLP(nn.Module):
    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        d_ff = int(cfg.d_ff) if cfg.d_ff is not None else 4 * int(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)


class _Block(nn.Module):
    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.attn = _CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg.vocab_size)
        self.max_seq_len = int(cfg.max_seq_len)
        self.d_model = int(cfg.d_model)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(int(cfg.n_layers))])
        self.ln_f = nn.LayerNorm(self.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [B, T]")
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = self.tok_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def nll(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape != target_ids.shape:
            raise ValueError("input_ids and target_ids must have same shape")
        logits = self.forward(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_ids.reshape(-1), reduction="mean")
        return loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in (0, 1]")

        seq = list(int(x) for x in prompt_ids)

        def banned_tokens_for_no_repeat_ngram(cur_seq: list[int], n: int) -> set[int]:
            if n <= 1:
                return set()
            if len(cur_seq) < n - 1:
                return set()
            prefix = tuple(cur_seq[-(n - 1) :])
            banned: set[int] = set()
            for i in range(0, len(cur_seq) - n + 1):
                ngram = tuple(cur_seq[i : i + n])
                if ngram[:-1] == prefix:
                    banned.add(int(ngram[-1]))
            return banned

        for _ in range(int(max_new_tokens)):
            ctx = seq[-self.max_seq_len :]
            x = torch.tensor(ctx, device=device, dtype=torch.long).unsqueeze(0)
            logits = self.forward(x)[:, -1, :]
            logits = logits.squeeze(0)

            if repetition_penalty > 1.0 and seq:
                seen = torch.tensor(sorted(set(seq)), device=device, dtype=torch.long)
                logits = logits.clone()
                logits[seen] = logits[seen] - math.log(repetition_penalty)

            if temperature != 1.0:
                logits = logits / max(float(temperature), 1e-6)
            probs = F.softmax(logits, dim=-1)

            if no_repeat_ngram_size > 0:
                banned = banned_tokens_for_no_repeat_ngram(seq, int(no_repeat_ngram_size))
                if banned:
                    banned_idx = torch.tensor(sorted(banned), device=device, dtype=torch.long)
                    probs = probs.clone()
                    probs[banned_idx] = 0.0

            if top_k is not None and top_k > 0 and top_k < probs.numel():
                top_vals, top_idx = torch.topk(probs, k=int(top_k))
                probs = torch.zeros_like(probs).scatter(0, top_idx, top_vals)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=0)
                keep = cum <= float(top_p)
                if keep.numel() > 0:
                    keep[0] = True
                filtered = torch.zeros_like(probs)
                filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
                probs = filtered

            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = probs / probs_sum

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            seq.append(next_id)
            if eos_id is not None and next_id == int(eos_id):
                break
        return seq

