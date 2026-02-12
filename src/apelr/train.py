from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PackedSequenceDataset, get_dataset_spec, load_corpus_text
from .model import APELRModel, APELRModelConfig
from .model_v2 import APELRV2Model, APELRV2ModelConfig
from .tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SPECIAL_TOKENS,
    TokenizerLike,
    load_tokenizer,
)

MODEL_VERSION_V1 = "v1_filtered_mixture"
MODEL_VERSION_V2 = "v2_planner_required"
CONFIG_SCHEMA_VERSION = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def sanitize_for_console(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_lr_with_warmup(
    *,
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if max_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    if max_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    scale = float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine
    return base_lr * scale


def lerp(start: float, end: float, frac: float) -> float:
    f = min(max(float(frac), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * f


def _is_auto_batch_size(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"auto", "adaptive"}
    if isinstance(value, (int, float)):
        return int(value) <= 0
    return False


def resolve_batch_size(
    *,
    train_cfg: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    model_version: str,
    train_ds: PackedSequenceDataset,
    use_amp: bool,
    amp_dtype: torch.dtype,
    loss_weights: dict[str, float],
    v2_commitment: str,
    v2_plan_temperature: float,
    v2_rep_unlikelihood_window: int,
) -> int:
    raw = train_cfg.get("batch_size", 1)
    adaptive_cfg = train_cfg.get("adaptive_batch", {})
    adaptive_enabled = bool(adaptive_cfg) if isinstance(adaptive_cfg, bool) else bool(
        isinstance(adaptive_cfg, dict) and adaptive_cfg.get("enabled", True)
    )
    adaptive_cfg = adaptive_cfg if isinstance(adaptive_cfg, dict) else {}

    if not _is_auto_batch_size(raw) and not adaptive_enabled:
        return int(raw)

    if device.type != "cuda":
        cpu_bs = int(train_cfg.get("cpu_batch_size", raw if not _is_auto_batch_size(raw) else 1))
        print(
            "Adaptive batch size enabled but device is CPU. "
            f"Using cpu_batch_size={cpu_bs}."
        )
        return max(cpu_bs, 1)

    base_bs = int(raw) if not _is_auto_batch_size(raw) else int(train_cfg.get("cpu_batch_size", 1))
    min_bs = int(adaptive_cfg.get("min_batch_size", train_cfg.get("batch_size_min", 1)))
    max_bs = int(adaptive_cfg.get("max_batch_size", train_cfg.get("batch_size_cap", 1024)))
    safety = float(adaptive_cfg.get("safety_factor", train_cfg.get("batch_size_safety", 0.85)))
    probe_bs = int(adaptive_cfg.get("probe_batch_size", train_cfg.get("batch_size_probe", max(1, min(base_bs, 4)))))
    probe_bs = max(probe_bs, 1)
    min_bs = max(min_bs, 1)
    max_bs = max(max_bs, min_bs)
    safety = min(max(safety, 0.1), 0.98)

    probe_loader = DataLoader(
        train_ds,
        batch_size=probe_bs,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    x, y = next(iter(probe_loader))

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    model.train()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        if model_version == MODEL_VERSION_V1:
            nll, aux = model.filtered_nll(x, y)
            loss = (
                nll
                + loss_weights["entropy_reg"] * aux["belief_entropy"]
                + loss_weights["usage_balance"] * aux["usage_kl_to_uniform"]
                + loss_weights["chunk_bow"] * aux["chunk_bow_loss"]
                - loss_weights["plan_mi"] * aux["plan_mi"]
                + loss_weights["chunk_post_kl"] * aux["chunk_post_kl"]
            )
        else:
            nll, aux = model.compute_losses(
                x,
                y,
                planner_mode="normal",
                commitment=v2_commitment,
                planner_temperature=v2_plan_temperature,
                rep_unlikelihood_window=v2_rep_unlikelihood_window,
            )
            loss = (
                nll
                + loss_weights["v2_usage"] * aux["usage_kl_to_uniform"]
                + loss_weights["v2_boundary_entropy"] * aux["boundary_entropy"]
                + loss_weights["v2_future"] * aux["future_contrastive_loss"]
                - loss_weights["v2_js"] * aux["plan_js_div_loss"]
                + loss_weights["v2_rep_unlikelihood"] * aux["rep_unlikelihood_loss"]
            )

    loss.backward()
    torch.cuda.synchronize(device_index)
    peak = torch.cuda.max_memory_allocated(device_index)
    model.zero_grad(set_to_none=True)
    del x, y

    per_sample = max(int(peak / probe_bs), 1)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
    usable = int(free_bytes * safety)
    bs = max(min_bs, min(max_bs, int(usable / per_sample)))
    if not _is_auto_batch_size(raw):
        scale_ratio = bs / float(max(probe_bs, 1))
        bs = int(max(min_bs, min(max_bs, round(base_bs * scale_ratio))))
    print(
        "Adaptive batch size: "
        f"base_bs={base_bs}, probe_bs={probe_bs}, per_sample={(per_sample / (1024 ** 2)):.1f} MiB, "
        f"free={(free_bytes / (1024 ** 2)):.0f} MiB, safety={safety:.2f} -> batch_size={bs}"
    )
    return bs


def get_model_version(model_cfg: dict[str, Any], resume_ckpt: dict[str, Any] | None) -> str:
    if "version" in model_cfg:
        return str(model_cfg["version"])
    if "architecture" in model_cfg:
        return str(model_cfg["architecture"])
    if resume_ckpt is not None:
        return str(resume_ckpt.get("model_version", MODEL_VERSION_V1))
    return MODEL_VERSION_V1


def instantiate_model(model_cfg: dict[str, Any], vocab_size: int, model_version: str) -> torch.nn.Module:
    if model_version == MODEL_VERSION_V1:
        return APELRModel(
            APELRModelConfig(
                vocab_size=vocab_size,
                num_plan_states=int(model_cfg["num_plan_states"]),
                chunk_size=int(model_cfg["chunk_size"]),
                token_dim=int(model_cfg["token_dim"]),
                hidden_dim=int(model_cfg["hidden_dim"]),
                fusion_dim=int(model_cfg["fusion_dim"]),
                num_layers=int(model_cfg["num_layers"]),
                dropout=float(model_cfg["dropout"]),
                planner_context_scale=float(model_cfg.get("planner_context_scale", 1.0)),
                planner_self_bias=float(model_cfg.get("planner_self_bias", 1.5)),
            )
        )
    if model_version == MODEL_VERSION_V2:
        return APELRV2Model(
            APELRV2ModelConfig(
                vocab_size=vocab_size,
                num_plan_states=int(model_cfg["num_plan_states"]),
                num_experts=int(model_cfg.get("num_experts", model_cfg["num_plan_states"])),
                chunk_size=int(model_cfg["chunk_size"]),
                token_dim=int(model_cfg["token_dim"]),
                hidden_dim=int(model_cfg["hidden_dim"]),
                num_layers=int(model_cfg["num_layers"]),
                dropout=float(model_cfg["dropout"]),
                planner_self_bias=float(model_cfg.get("planner_self_bias", 1.8)),
                planner_context_scale=float(model_cfg.get("planner_context_scale", 1.0)),
                future_horizon_chunks=int(model_cfg.get("future_horizon_chunks", 2)),
                planner_temperature=float(model_cfg.get("plan_temperature_start", model_cfg.get("plan_temperature", 1.0))),
            )
        )
    raise ValueError(f"Unsupported model version '{model_version}'.")


def evaluate_v1(model: APELRModel, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    entropies: list[float] = []
    usage_kls: list[float] = []
    chunk_bows: list[float] = []
    plan_mis: list[float] = []
    chunk_post_kls: list[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            loss, aux = model.filtered_nll(x, y)
            losses.append(float(loss.item()))
            entropies.append(float(aux["belief_entropy"].item()))
            usage_kls.append(float(aux["usage_kl_to_uniform"].item()))
            chunk_bows.append(float(aux["chunk_bow_loss"].item()))
            plan_mis.append(float(aux["plan_mi"].item()))
            chunk_post_kls.append(float(aux["chunk_post_kl"].item()))
    if not losses:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "belief_entropy": float("nan"),
            "usage_kl_to_uniform": float("nan"),
            "chunk_bow_loss": float("nan"),
            "plan_mi": float("nan"),
            "chunk_post_kl": float("nan"),
        }
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "ppl": float(math.exp(min(mean_loss, 20.0))),
        "belief_entropy": float(np.mean(entropies)),
        "usage_kl_to_uniform": float(np.mean(usage_kls)),
        "chunk_bow_loss": float(np.mean(chunk_bows)),
        "plan_mi": float(np.mean(plan_mis)),
        "chunk_post_kl": float(np.mean(chunk_post_kls)),
    }


def evaluate_v2(
    model: APELRV2Model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    rep_unlikelihood_window: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    usage_kls: list[float] = []
    boundary_ents: list[float] = []
    future_losses: list[float] = []
    js_losses: list[float] = []
    repu_losses: list[float] = []
    mask_deltas: list[float] = []
    force_divs: list[float] = []
    state_persist: list[float] = []
    expert_utils: list[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            nll, aux = model.compute_losses(
                x,
                y,
                planner_mode="normal",
                commitment="soft",
                rep_unlikelihood_window=rep_unlikelihood_window,
            )
            diag = model.planner_usage_metrics(x, y)
            losses.append(float(nll.item()))
            usage_kls.append(float(aux["usage_kl_to_uniform"].item()))
            boundary_ents.append(float(aux["boundary_entropy"].item()))
            future_losses.append(float(aux["future_contrastive_loss"].item()))
            js_losses.append(float(aux["plan_js_div_loss"].item()))
            repu_losses.append(float(aux["rep_unlikelihood_loss"].item()))
            mask_deltas.append(float(diag["planner_mask_delta_loss"].item()))
            force_divs.append(float(diag["forced_state_divergence"].item()))
            state_persist.append(float(aux["state_persistence"].item()))
            expert_utils.append(float(aux["expert_utilization"].item()))
    if not losses:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "usage_kl_to_uniform": float("nan"),
            "boundary_entropy": float("nan"),
            "future_contrastive_loss": float("nan"),
            "plan_js_div_loss": float("nan"),
            "rep_unlikelihood_loss": float("nan"),
            "planner_mask_delta_loss": float("nan"),
            "forced_state_divergence": float("nan"),
            "state_persistence": float("nan"),
            "expert_utilization": float("nan"),
        }
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "ppl": float(math.exp(min(mean_loss, 20.0))),
        "usage_kl_to_uniform": float(np.mean(usage_kls)),
        "boundary_entropy": float(np.mean(boundary_ents)),
        "future_contrastive_loss": float(np.mean(future_losses)),
        "plan_js_div_loss": float(np.mean(js_losses)),
        "rep_unlikelihood_loss": float(np.mean(repu_losses)),
        "planner_mask_delta_loss": float(np.mean(mask_deltas)),
        "forced_state_divergence": float(np.mean(force_divs)),
        "state_persistence": float(np.mean(state_persist)),
        "expert_utilization": float(np.mean(expert_utils)),
    }


def evaluate_model(
    model: torch.nn.Module,
    model_version: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    rep_unlikelihood_window: int = 0,
) -> dict[str, float]:
    if model_version == MODEL_VERSION_V1:
        return evaluate_v1(model, loader, device, max_batches)
    return evaluate_v2(model, loader, device, max_batches, rep_unlikelihood_window)


def prepare_tokens(tokenizer: TokenizerLike, text: str) -> list[int]:
    ids = [tokenizer.bos_id]
    ids.extend(tokenizer.encode(text, add_bos=False, add_eos=True))
    return ids


def build_tokenizer(tokenizer_cfg: dict[str, Any], train_text: str) -> tuple[TokenizerLike, str, dict[str, Any]]:
    tok_type = str(tokenizer_cfg.get("type", "char")).lower()
    if tok_type == "char":
        tok = CharTokenizer.fit(train_text)
        return tok, tok_type, {}
    if tok_type == "bpe":
        vocab_size = int(tokenizer_cfg.get("vocab_size", 4096))
        min_frequency = int(tokenizer_cfg.get("min_frequency", 2))
        byte_level = bool(tokenizer_cfg.get("byte_level", True))
        lowercase = bool(tokenizer_cfg.get("lowercase", False))
        special_tokens = list(tokenizer_cfg.get("special_tokens", SPECIAL_TOKENS))
        tok = BPETokenizer.fit(
            texts=[train_text],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            byte_level=byte_level,
            lowercase=lowercase,
            special_tokens=special_tokens,
        )
        meta = {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "byte_level": byte_level,
            "lowercase": lowercase,
            "special_tokens": special_tokens,
        }
        return tok, tok_type, meta
    raise ValueError(f"Unsupported tokenizer type '{tok_type}'.")


def maybe_sample_preview(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    prompt: str,
    max_new_tokens: int,
) -> str:
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_bos=False, add_eos=False)
    if model_version == MODEL_VERSION_V1:
        sample_ids, _ = model.generate_filtered(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_k=40,
            top_p=0.95,
            eos_id=tokenizer.eos_id,
            lookahead_steps=1,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )
    else:
        sample_ids, _ = model.generate_planned(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_k=40,
            top_p=0.95,
            eos_id=tokenizer.eos_id,
            lookahead_steps=1,
            planner_temperature=1.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(sample_ids)


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    tokenizer_path: Path,
    tokenizer_type: str,
    tokenizer_meta: dict[str, Any],
    model_version: str,
    global_step: int,
) -> None:
    checkpoint = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "model_version": model_version,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "model_config": model.cfg.__dict__,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": file_sha256(tokenizer_path),
        "tokenizer_type": tokenizer_type,
        "tokenizer_config": tokenizer_meta,
        "train_config": train_cfg,
    }
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train APEL-R model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tokenizer_cfg = cfg.get("tokenizer", {"type": "char"})

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    source = data_cfg["source"]
    spec = get_dataset_spec(source)

    print(f"Loading dataset source={source} ({spec.path})...")
    train_text, train_stats = load_corpus_text(
        source=source,
        split=data_cfg.get("train_split", spec.train_split),
        max_examples=int(data_cfg["max_train_examples"]),
        min_chars=int(data_cfg.get("min_chars", 8)),
        max_chars=(int(data_cfg["max_train_chars"]) if data_cfg.get("max_train_chars") is not None else None),
        streaming=data_cfg.get("streaming"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    val_text, val_stats = load_corpus_text(
        source=source,
        split=data_cfg.get("val_split", spec.val_split),
        max_examples=int(data_cfg["max_val_examples"]),
        min_chars=int(data_cfg.get("min_chars", 8)),
        max_chars=(int(data_cfg["max_val_chars"]) if data_cfg.get("max_val_chars") is not None else None),
        streaming=data_cfg.get("streaming"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    print(f"Train stats: {train_stats}")
    print(f"Val stats:   {val_stats}")

    resume_from = train_cfg.get("resume_from")
    resume_ckpt: dict[str, Any] | None = None
    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from path not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_tok_path = resume_ckpt.get("tokenizer_path")
        if ckpt_tok_path:
            tokenizer_type = str(resume_ckpt.get("tokenizer_type", "char"))
            tokenizer_meta = dict(resume_ckpt.get("tokenizer_config", {}))
            tokenizer = load_tokenizer(
                Path(ckpt_tok_path),
                tokenizer_type=tokenizer_type,
                special_tokens=tokenizer_meta.get("special_tokens"),
            )
            tokenizer_path = out_dir / "tokenizer.json"
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer loaded from resume checkpoint: type={tokenizer_type}, vocab size={tokenizer.vocab_size}")
        else:
            tokenizer, tokenizer_type, tokenizer_meta = build_tokenizer(tokenizer_cfg, train_text)
            tokenizer_path = out_dir / "tokenizer.json"
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer type: {tokenizer_type}, vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer, tokenizer_type, tokenizer_meta = build_tokenizer(tokenizer_cfg, train_text)
        tokenizer_path = out_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer type: {tokenizer_type}, vocab size: {tokenizer.vocab_size}")

    model_version = get_model_version(model_cfg, resume_ckpt)
    if resume_ckpt is not None and "model_version" in resume_ckpt:
        ckpt_model_version = str(resume_ckpt["model_version"])
        if ("version" in model_cfg or "architecture" in model_cfg) and ckpt_model_version != model_version:
            raise ValueError(
                f"Config model version '{model_version}' does not match resume checkpoint model version '{ckpt_model_version}'."
            )

    train_ids = prepare_tokens(tokenizer, train_text)
    val_ids = prepare_tokens(tokenizer, val_text)

    seq_len = int(train_cfg["seq_len"])
    stride = int(train_cfg.get("stride", seq_len))
    train_ds = PackedSequenceDataset(train_ids, seq_len=seq_len, stride=stride)
    val_ds = PackedSequenceDataset(val_ids, seq_len=seq_len, stride=seq_len)

    model = instantiate_model(model_cfg=model_cfg, vocab_size=tokenizer.vocab_size, model_version=model_version)

    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(device)

    print(f"Device: {device}")
    print(f"Model version: {model_version}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    base_lr = float(train_cfg["lr"])
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    max_steps = int(train_cfg["max_steps"])
    eval_interval = int(train_cfg.get("eval_interval", 200))
    log_interval = int(train_cfg.get("log_interval", 20))
    val_batches = int(train_cfg.get("val_batches", 20))
    preview_interval = int(train_cfg.get("preview_interval", eval_interval))
    preview_prompt = str(train_cfg.get("preview_prompt", "Once upon a time"))
    preview_tokens = int(train_cfg.get("preview_tokens", 120))
    lr_schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("train.lr_schedule must be one of: constant, cosine")

    entropy_reg_weight = float(train_cfg.get("entropy_reg_weight", 0.0))
    usage_balance_weight = float(train_cfg.get("usage_balance_weight", 0.0))
    chunk_bow_weight = float(train_cfg.get("chunk_bow_weight", 0.0))
    chunk_bow_warmup_steps = int(train_cfg.get("chunk_bow_warmup_steps", 0))
    plan_mi_weight = float(train_cfg.get("plan_mi_weight", 0.0))
    chunk_post_kl_weight = float(train_cfg.get("chunk_post_kl_weight", 0.0))

    v2_w = dict(train_cfg.get("loss_weights", {}))
    v2_future_weight = float(v2_w.get("future_contrastive", 0.0))
    v2_js_weight = float(v2_w.get("plan_js_div", 0.0))
    v2_boundary_entropy_weight = float(v2_w.get("boundary_entropy", 0.0))
    v2_usage_weight = float(v2_w.get("usage_balance", usage_balance_weight))
    v2_rep_unlikelihood_weight = float(v2_w.get("rep_unlikelihood", 0.0))
    v2_rep_unlikelihood_window = int(v2_w.get("rep_window", 0))
    v2_commitment = str(model_cfg.get("plan_commitment", "soft"))
    v2_commitment_warmup_steps = int(model_cfg.get("commitment_warmup_steps", 0))
    v2_plan_temperature_start = float(model_cfg.get("plan_temperature_start", model_cfg.get("plan_temperature", 1.0)))
    v2_plan_temperature_end = float(model_cfg.get("plan_temperature_end", v2_plan_temperature_start))

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    precision = str(train_cfg.get("precision", "fp16" if device.type == "cuda" else "fp32")).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision '{precision}'. Use fp32, fp16, or bf16.")
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))
    print(f"Precision: {precision}, grad_accum_steps: {grad_accum_steps}, use_amp: {use_amp}")

    loss_weights = {
        "entropy_reg": entropy_reg_weight,
        "usage_balance": usage_balance_weight,
        "chunk_bow": chunk_bow_weight,
        "plan_mi": plan_mi_weight,
        "chunk_post_kl": chunk_post_kl_weight,
        "v2_usage": v2_usage_weight,
        "v2_boundary_entropy": v2_boundary_entropy_weight,
        "v2_future": v2_future_weight,
        "v2_js": v2_js_weight,
        "v2_rep_unlikelihood": v2_rep_unlikelihood_weight,
    }

    batch_size = resolve_batch_size(
        train_cfg=train_cfg,
        device=device,
        model=model,
        model_version=model_version,
        train_ds=train_ds,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        loss_weights=loss_weights,
        v2_commitment=v2_commitment,
        v2_plan_temperature=v2_plan_temperature_start,
        v2_rep_unlikelihood_window=v2_rep_unlikelihood_window,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )

    save_interval = int(train_cfg.get("save_interval", 0))

    global_step = 0
    if resume_from:
        if resume_ckpt is None:
            raise RuntimeError("Internal error: resume checkpoint was not loaded.")
        ckpt_model_version = str(resume_ckpt.get("model_version", MODEL_VERSION_V1))
        if ckpt_model_version != model_version:
            raise ValueError(
                f"Refusing to load resume checkpoint model version '{ckpt_model_version}' into model version '{model_version}'."
            )
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        opt_state = resume_ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        global_step = int(resume_ckpt.get("global_step", 0))
        print(f"Resumed from {Path(str(resume_from))} at global_step={global_step}")

    start_time = time.time()

    model.train()
    progress = tqdm(total=max_steps, initial=global_step, desc="train")
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps:
        for x, y in train_loader:
            if global_step >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            current_lr = (
                cosine_lr_with_warmup(
                    step=global_step,
                    max_steps=max_steps,
                    base_lr=base_lr,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                )
                if lr_schedule == "cosine"
                else base_lr
            )
            set_optimizer_lr(optimizer, current_lr)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if model_version == MODEL_VERSION_V1:
                    nll, aux = model.filtered_nll(x, y)
                    if chunk_bow_warmup_steps > 0:
                        chunk_warm = min(1.0, float(global_step + 1) / float(max(chunk_bow_warmup_steps, 1)))
                    else:
                        chunk_warm = 1.0
                    effective_chunk_bow_weight = chunk_bow_weight * chunk_warm
                    loss = (
                        nll
                        + entropy_reg_weight * aux["belief_entropy"]
                        + usage_balance_weight * aux["usage_kl_to_uniform"]
                        + effective_chunk_bow_weight * aux["chunk_bow_loss"]
                        - plan_mi_weight * aux["plan_mi"]
                        + chunk_post_kl_weight * aux["chunk_post_kl"]
                    )
                else:
                    progress_frac = float(global_step) / float(max(max_steps - 1, 1))
                    current_plan_temp = lerp(v2_plan_temperature_start, v2_plan_temperature_end, progress_frac)
                    current_commitment = (
                        "soft"
                        if (v2_commitment == "gumbel_st" and global_step < v2_commitment_warmup_steps)
                        else v2_commitment
                    )
                    nll, aux = model.compute_losses(
                        x,
                        y,
                        planner_mode="normal",
                        commitment=current_commitment,
                        planner_temperature=current_plan_temp,
                        rep_unlikelihood_window=v2_rep_unlikelihood_window,
                    )
                    chunk_warm = 1.0
                    effective_chunk_bow_weight = 0.0
                    loss = (
                        nll
                        + v2_usage_weight * aux["usage_kl_to_uniform"]
                        + v2_boundary_entropy_weight * aux["boundary_entropy"]
                        + v2_future_weight * aux["future_contrastive_loss"]
                        - v2_js_weight * aux["plan_js_div_loss"]
                        + v2_rep_unlikelihood_weight * aux["rep_unlikelihood_loss"]
                    )

                loss_for_backward = loss / float(max(grad_accum_steps, 1))

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            micro_step += 1

            if micro_step % max(grad_accum_steps, 1) != 0:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % log_interval == 0 or global_step == 1:
                elapsed = time.time() - start_time
                tok_per_s = (global_step * x.shape[0] * x.shape[1] * max(grad_accum_steps, 1)) / max(elapsed, 1e-6)
                if model_version == MODEL_VERSION_V1:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        ppl=f"{math.exp(min(nll.item(), 20.0)):.2f}",
                        ent=f"{aux['belief_entropy'].item():.3f}",
                        klu=f"{aux['usage_kl_to_uniform'].item():.3f}",
                        cbow=f"{aux['chunk_bow_loss'].item():.3f}",
                        cbw=f"{effective_chunk_bow_weight:.3f}",
                        mi=f"{aux['plan_mi'].item():.3f}",
                        cpk=f"{aux['chunk_post_kl'].item():.3f}",
                        tok_s=f"{tok_per_s:.0f}",
                    )
                else:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        ppl=f"{math.exp(min(nll.item(), 20.0)):.2f}",
                        klu=f"{aux['usage_kl_to_uniform'].item():.3f}",
                        bent=f"{aux['boundary_entropy'].item():.3f}",
                        fcl=f"{aux['future_contrastive_loss'].item():.3f}",
                        js=f"{aux['plan_js_div_loss'].item():.3f}",
                        repu=f"{aux['rep_unlikelihood_loss'].item():.3f}",
                        sp=f"{aux['state_persistence'].item():.3f}",
                        eu=f"{aux['expert_utilization'].item():.3f}",
                        ptmp=f"{current_plan_temp:.2f}",
                        lr=f"{current_lr:.2e}",
                        tok_s=f"{tok_per_s:.0f}",
                    )

            if global_step % eval_interval == 0 or global_step == max_steps:
                eval_metrics = evaluate_model(
                    model,
                    model_version,
                    val_loader,
                    device,
                    max_batches=val_batches,
                    rep_unlikelihood_window=v2_rep_unlikelihood_window,
                )
                if model_version == MODEL_VERSION_V1:
                    print(
                        f"\n[eval step {global_step}] "
                        f"val_loss={eval_metrics['loss']:.4f} "
                        f"val_ppl={eval_metrics['ppl']:.2f} "
                        f"belief_ent={eval_metrics['belief_entropy']:.3f} "
                        f"usage_kl={eval_metrics['usage_kl_to_uniform']:.3f} "
                        f"chunk_bow={eval_metrics['chunk_bow_loss']:.3f} "
                        f"plan_mi={eval_metrics['plan_mi']:.3f} "
                        f"chunk_post_kl={eval_metrics['chunk_post_kl']:.3f}"
                    )
                else:
                    print(
                        f"\n[eval step {global_step}] "
                        f"val_loss={eval_metrics['loss']:.4f} "
                        f"val_ppl={eval_metrics['ppl']:.2f} "
                        f"usage_kl={eval_metrics['usage_kl_to_uniform']:.3f} "
                        f"bent={eval_metrics['boundary_entropy']:.3f} "
                        f"fcl={eval_metrics['future_contrastive_loss']:.3f} "
                        f"js={eval_metrics['plan_js_div_loss']:.3f} "
                        f"repu={eval_metrics['rep_unlikelihood_loss']:.3f} "
                        f"mask_d={eval_metrics['planner_mask_delta_loss']:.3f} "
                        f"force_js={eval_metrics['forced_state_divergence']:.3f} "
                        f"sp={eval_metrics['state_persistence']:.3f} "
                        f"eu={eval_metrics['expert_utilization']:.3f}"
                    )
                model.train()

            if global_step % preview_interval == 0 or global_step == max_steps:
                preview = maybe_sample_preview(
                    model=model,
                    model_version=model_version,
                    tokenizer=tokenizer,
                    prompt=preview_prompt,
                    max_new_tokens=preview_tokens,
                )
                print(f"\n[sample step {global_step}] {sanitize_for_console(preview[:500])}\n")
                model.train()

            if save_interval > 0 and (global_step % save_interval == 0 or global_step == max_steps):
                save_checkpoint(
                    path=out_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    train_cfg=cfg,
                    tokenizer_path=tokenizer_path,
                    tokenizer_type=tokenizer_type,
                    tokenizer_meta=tokenizer_meta,
                    model_version=model_version,
                    global_step=global_step,
                )
                print(f"\n[checkpoint step {global_step}] saved to {out_dir / 'checkpoint.pt'}")

    progress.close()

    ckpt_path = out_dir / "checkpoint.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        train_cfg=cfg,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        tokenizer_meta=tokenizer_meta,
        model_version=model_version,
        global_step=global_step,
    )

    metrics = evaluate_model(
        model,
        model_version,
        val_loader,
        device,
        max_batches=val_batches,
        rep_unlikelihood_window=v2_rep_unlikelihood_window,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if model_version == MODEL_VERSION_V2:
        planner_eval = {
            "planner_mask_delta_loss": metrics.get("planner_mask_delta_loss"),
            "forced_state_divergence": metrics.get("forced_state_divergence"),
            "state_persistence": metrics.get("state_persistence"),
            "expert_utilization": metrics.get("expert_utilization"),
        }
        (out_dir / "planner_eval.json").write_text(json.dumps(planner_eval, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()

