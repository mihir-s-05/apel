from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .data import (
    PackedMemmapDataset,
    PackedSequenceDataset,
    get_dataset_spec,
    load_corpus_text,
    write_tokenized_corpus,
)
from .model import APELRModel, APELRModelConfig
from .model_v2 import APELRV2Model, APELRV2ModelConfig
from .transformer_lm import TransformerLM, TransformerLMConfig
from .tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SPECIAL_TOKENS,
    TokenizerLike,
    load_tokenizer,
)

MODEL_VERSION_V1 = "v1_filtered_mixture"
MODEL_VERSION_V2 = "v2_planner_required"
MODEL_VERSION_TRANSFORMER = "transformer_lm"
CONFIG_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return (not self.enabled) or self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _parse_distributed_context(train_cfg: dict[str, Any], *, device_type: str) -> DistributedContext:
    dist_cfg = train_cfg.get("distributed", {})
    if isinstance(dist_cfg, bool):
        cfg_enabled = bool(dist_cfg)
        dist_cfg = {}
    elif isinstance(dist_cfg, dict):
        cfg_enabled = bool(dist_cfg.get("enabled", False))
    else:
        cfg_enabled = False
        dist_cfg = {}

    world_size_env = _env_int("WORLD_SIZE", 1)
    enabled = int(world_size_env) > 1
    if not enabled:
        if cfg_enabled:
            print(
                "Warning: train.distributed.enabled=true but WORLD_SIZE=1. "
                "Run with torchrun (or set WORLD_SIZE/RANK/LOCAL_RANK) to enable distributed training."
            )
        return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, backend="")

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    backend = str(dist_cfg.get("backend") or "").strip().lower()
    if not backend:
        want_nccl = device_type == "cuda" and dist.is_nccl_available() and os.name != "nt"
        backend = "nccl" if want_nccl else "gloo"
    return DistributedContext(
        enabled=True,
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size_env),
        backend=backend,
    )


def _maybe_init_process_group(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    if dist.is_initialized():
        return
    dist.init_process_group(backend=ctx.backend, rank=ctx.rank, world_size=ctx.world_size)


def _maybe_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def _broadcast_int(ctx: DistributedContext, value: int, *, device: torch.device) -> int:
    if not ctx.enabled or not dist.is_initialized():
        return int(value)
    t = torch.tensor([int(value) if ctx.is_main else 0], device=device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return int(t.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted-path overrides, e.g. ``train.lr=0.0003``."""
    import ast as _ast

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        # auto-cast to int / float / bool / None when possible
        try:
            value = _ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        node[parts[-1]] = value
    return cfg


def file_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def sanitize_for_console(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _all_finite_tensors(loss: torch.Tensor, aux: dict[str, torch.Tensor]) -> bool:
    if not bool(torch.isfinite(loss.detach()).all().item()):
        return False
    for value in aux.values():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value.detach()).all().item()):
            return False
    return True


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
    train_ds: Dataset,
    use_amp: bool,
    amp_dtype: torch.dtype,
    loss_weights: dict[str, float],
    v2_commitment: str,
    v2_plan_temperature: float,
    v2_rep_unlikelihood_window: int,
) -> int:
    raw = train_cfg.get("batch_size", 1)
    adaptive_cfg = train_cfg.get("adaptive_batch", {})
    if isinstance(adaptive_cfg, bool):
        adaptive_enabled = adaptive_cfg
    elif isinstance(adaptive_cfg, dict):
        adaptive_enabled = bool(adaptive_cfg.get("enabled", _is_auto_batch_size(raw)))
    else:
        adaptive_enabled = _is_auto_batch_size(raw)
    adaptive_cfg = adaptive_cfg if isinstance(adaptive_cfg, dict) else {}

    if not _is_auto_batch_size(raw) and not adaptive_enabled:
        return max(1, int(raw))

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

    if isinstance(train_ds, IterableDataset):
        probe_loader = DataLoader(
            train_ds,
            batch_size=probe_bs,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )
    else:
        probe_loader = DataLoader(
            train_ds,
            batch_size=probe_bs,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )
    try:
        x, y = next(iter(probe_loader))
    except StopIteration:
        fallback = base_bs if base_bs > 0 else min_bs
        return max(min_bs, min(max_bs, fallback))

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    model.train()

    try:
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
            elif model_version == MODEL_VERSION_V2:
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
            elif model_version == MODEL_VERSION_TRANSFORMER:
                nll = model.nll(x, y)
                loss = nll
            else:
                raise ValueError(f"Unsupported model version '{model_version}'.")
        loss.backward()
        torch.cuda.synchronize(device_index)
        peak = torch.cuda.max_memory_allocated(device_index)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            fallback = max(min_bs, min(max_bs, max(1, base_bs // 2)))
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            return fallback
        raise
    finally:
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
    return MODEL_VERSION_V2


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
                token_filtering=bool(model_cfg.get("token_filtering", True)),
                lookahead_horizon=int(model_cfg.get("lookahead_horizon", 2)),
                lookahead_feedback_scale=float(model_cfg.get("lookahead_feedback_scale", 0.25)),
                async_planner=bool(model_cfg.get("async_planner", True)),
            )
        )
    if model_version == MODEL_VERSION_TRANSFORMER:
        return TransformerLM(
            TransformerLMConfig(
                vocab_size=vocab_size,
                max_seq_len=int(model_cfg.get("max_seq_len", 1024)),
                d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_dim", 512))),
                n_layers=int(model_cfg.get("n_layers", model_cfg.get("num_layers", 6))),
                n_heads=int(model_cfg.get("n_heads", 8)),
                d_ff=(int(model_cfg["d_ff"]) if model_cfg.get("d_ff") is not None else None),
                dropout=float(model_cfg.get("dropout", 0.1)),
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
    *,
    commitment: str,
    planner_temperature: float,
    include_planner_diagnostics: bool = True,
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
    feedback_deltas: list[float] = []
    lookahead_steps = int(model.cfg.lookahead_horizon)
    lookahead_feedback_scale = float(model.cfg.lookahead_feedback_scale)
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
                commitment=commitment,
                planner_temperature=planner_temperature,
                lookahead_steps=lookahead_steps,
                lookahead_feedback_scale=lookahead_feedback_scale,
                rep_unlikelihood_window=rep_unlikelihood_window,
            )
            losses.append(float(nll.item()))
            usage_kls.append(float(aux["usage_kl_to_uniform"].item()))
            boundary_ents.append(float(aux["boundary_entropy"].item()))
            future_losses.append(float(aux["future_contrastive_loss"].item()))
            js_losses.append(float(aux["plan_js_div_loss"].item()))
            repu_losses.append(float(aux["rep_unlikelihood_loss"].item()))
            state_persist.append(float(aux["state_persistence"].item()))
            expert_utils.append(float(aux["expert_utilization"].item()))
            if include_planner_diagnostics:
                nll_no_feedback, _ = model.compute_losses(
                    x,
                    y,
                    planner_mode="normal",
                    commitment=commitment,
                    planner_temperature=planner_temperature,
                    lookahead_steps=lookahead_steps,
                    lookahead_feedback_scale=0.0,
                    rep_unlikelihood_window=rep_unlikelihood_window,
                )
                diag = model.planner_usage_metrics(
                    x,
                    y,
                    commitment=commitment,
                    planner_temperature=planner_temperature,
                )
                feedback_deltas.append(float((nll_no_feedback - nll).item()))
                mask_deltas.append(float(diag["planner_mask_delta_loss"].item()))
                force_divs.append(float(diag["forced_state_divergence"].item()))
    if not losses:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "usage_kl_to_uniform": float("nan"),
            "boundary_entropy": float("nan"),
            "future_contrastive_loss": float("nan"),
            "plan_js_div_loss": float("nan"),
            "rep_unlikelihood_loss": float("nan"),
            "feedback_delta_loss": float("nan"),
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
        "feedback_delta_loss": float(np.mean(feedback_deltas)) if feedback_deltas else float("nan"),
        "planner_mask_delta_loss": float(np.mean(mask_deltas)) if mask_deltas else float("nan"),
        "forced_state_divergence": float(np.mean(force_divs)) if force_divs else float("nan"),
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
    *,
    v2_commitment: str = "soft",
    v2_planner_temperature: float = 1.0,
    include_planner_diagnostics: bool = True,
) -> dict[str, float]:
    if model_version == MODEL_VERSION_V1:
        return evaluate_v1(model, loader, device, max_batches)
    if model_version == MODEL_VERSION_V2:
        return evaluate_v2(
            model,
            loader,
            device,
            max_batches,
            rep_unlikelihood_window,
            commitment=v2_commitment,
            planner_temperature=v2_planner_temperature,
            include_planner_diagnostics=include_planner_diagnostics,
        )
    if model_version == MODEL_VERSION_TRANSFORMER:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                loss = model.nll(x, y)
                losses.append(float(loss.item()))
        if not losses:
            return {"loss": float("nan"), "ppl": float("nan")}
        mean_loss = float(np.mean(losses))
        return {"loss": mean_loss, "ppl": float(math.exp(min(mean_loss, 20.0)))}
    raise ValueError(f"Unsupported model version '{model_version}'.")


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
    try:
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
        elif model_version == MODEL_VERSION_V2:
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
        elif model_version == MODEL_VERSION_TRANSFORMER:
            sample_ids = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                eos_id=tokenizer.eos_id,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )
        else:
            raise ValueError(f"Unsupported model version '{model_version}'.")
    except Exception as exc:
        return f"[preview skipped: sampling failed ({exc.__class__.__name__}: {exc})]"
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
    dist_ctx: DistributedContext | None = None,
    save_optimizer_state: bool = True,
) -> None:
    ctx = dist_ctx or DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, backend="")
    core_model = unwrap_model(model)

    opt_state: dict[str, Any] | None = None
    if save_optimizer_state:
        if isinstance(optimizer, ZeroRedundancyOptimizer):
            if ctx.enabled:
                optimizer.consolidate_state_dict(to=0)
                if ctx.is_main:
                    opt_state = optimizer.state_dict()
            else:
                opt_state = optimizer.state_dict()
        else:
            opt_state = optimizer.state_dict()

    if not ctx.is_main:
        return

    checkpoint = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "model_version": model_version,
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": opt_state,
        "global_step": int(global_step),
        "model_config": core_model.cfg.__dict__,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": file_sha256(tokenizer_path),
        "tokenizer_type": tokenizer_type,
        "tokenizer_config": tokenizer_meta,
        "train_config": train_cfg,
        "distributed": {
            "world_size": int(ctx.world_size),
            "backend": str(ctx.backend),
        },
    }
    torch.save(checkpoint, path)


def build_loaders(
    *,
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    train_cfg: dict[str, Any],
    dist_ctx: DistributedContext | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    ctx = dist_ctx or DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, backend="")
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    prefetch_factor_cfg = train_cfg.get("prefetch_factor")
    pin_memory_device = train_cfg.get("pin_memory_device")
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = str(pin_memory_device)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        if prefetch_factor_cfg is not None:
            loader_kwargs["prefetch_factor"] = max(int(prefetch_factor_cfg), 1)
    train_sampler = None
    if isinstance(train_ds, IterableDataset):
        if ctx.enabled and ctx.is_main:
            print("Warning: IterableDataset does not support DistributedSampler; each rank will see the full stream.")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            drop_last=True,
            **loader_kwargs,
        )
    else:
        if ctx.enabled:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=int(ctx.world_size),
                rank=int(ctx.rank),
                shuffle=True,
                seed=int(seed),
                drop_last=True,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train APEL-R model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="Override config values, e.g. --set train.lr=0.0003 model.num_layers=4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = _apply_overrides(cfg, args.overrides)
    seed = int(cfg.get("seed", 42))

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tokenizer_cfg = cfg.get("tokenizer", {"type": "char"})

    requested_device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    requested_device = torch.device(requested_device_name)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: device='cuda' requested but CUDA is not available. Falling back to CPU.")
        requested_device = torch.device("cpu")

    dist_ctx = _parse_distributed_context(train_cfg, device_type=requested_device.type)
    device = requested_device
    if dist_ctx.enabled and requested_device.type == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_ctx.local_rank}")
        torch.cuda.set_device(device)
    _maybe_init_process_group(dist_ctx)

    set_seed(seed)

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    is_main = dist_ctx.is_main

    source = data_cfg["source"]
    spec = get_dataset_spec(source)
    token_cache_enabled = bool(data_cfg.get("token_cache", False))

    if is_main:
        print(f"Loading dataset source={source} ({spec.path})...")
    train_text = ""
    val_text = ""
    train_stats: dict[str, int] = {}
    val_stats: dict[str, int] = {}
    if not token_cache_enabled:
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
        if is_main:
            print(f"Train stats: {train_stats}")
            print(f"Val stats:   {val_stats}")
    else:
        if is_main:
            print("Token-cache mode enabled: using on-disk tokenized corpora for large-scale training.")

    resume_from = args.resume or train_cfg.get("resume_from")
    resume_ckpt: dict[str, Any] | None = None

    def ensure_tokenizer_seed_text() -> str:
        nonlocal train_text
        if train_text:
            return train_text
        fit_max_examples = int(data_cfg.get("tokenizer_fit_max_examples", min(int(data_cfg["max_train_examples"]), 50_000)))
        fit_max_chars_cfg = data_cfg.get("tokenizer_fit_max_chars")
        fit_max_chars = int(fit_max_chars_cfg) if fit_max_chars_cfg is not None else None
        fit_text, fit_stats = load_corpus_text(
            source=source,
            split=data_cfg.get("tokenizer_fit_split", data_cfg.get("train_split", spec.train_split)),
            max_examples=fit_max_examples,
            min_chars=int(data_cfg.get("min_chars", 8)),
            max_chars=fit_max_chars,
            streaming=data_cfg.get("tokenizer_fit_streaming", data_cfg.get("streaming")),
            cache_dir=data_cfg.get("cache_dir"),
        )
        train_text = fit_text
        if is_main:
            print(f"Tokenizer fit stats: {fit_stats}")
        return train_text

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer_type = str(tokenizer_cfg.get("type", "char")).lower()
    tokenizer_meta: dict[str, Any] = {}
    if tokenizer_type == "bpe":
        tokenizer_meta = {
            "vocab_size": int(tokenizer_cfg.get("vocab_size", 4096)),
            "min_frequency": int(tokenizer_cfg.get("min_frequency", 2)),
            "byte_level": bool(tokenizer_cfg.get("byte_level", True)),
            "lowercase": bool(tokenizer_cfg.get("lowercase", False)),
            "special_tokens": list(tokenizer_cfg.get("special_tokens", SPECIAL_TOKENS)),
        }

    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from path not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_tok_path = resume_ckpt.get("tokenizer_path")
        if ckpt_tok_path:
            tokenizer_type = str(resume_ckpt.get("tokenizer_type", tokenizer_type))
            tokenizer_meta = dict(resume_ckpt.get("tokenizer_config", tokenizer_meta))

    if is_main:
        if resume_ckpt is not None and resume_ckpt.get("tokenizer_path"):
            ckpt_tok_path = Path(str(resume_ckpt["tokenizer_path"]))
            tokenizer = load_tokenizer(
                ckpt_tok_path,
                tokenizer_type=tokenizer_type,
                special_tokens=tokenizer_meta.get("special_tokens"),
            )
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer loaded from resume checkpoint: type={tokenizer_type}, vocab size={tokenizer.vocab_size}")
        else:
            tokenizer_seed_text = ensure_tokenizer_seed_text()
            tokenizer, tokenizer_type, tokenizer_meta = build_tokenizer(tokenizer_cfg, tokenizer_seed_text)
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer type: {tokenizer_type}, vocab size: {tokenizer.vocab_size}")

    _maybe_barrier(dist_ctx)
    tokenizer = load_tokenizer(
        tokenizer_path,
        tokenizer_type=tokenizer_type,
        special_tokens=tokenizer_meta.get("special_tokens"),
    )

    model_version = get_model_version(model_cfg, resume_ckpt)
    if resume_ckpt is not None and "model_version" in resume_ckpt:
        ckpt_model_version = str(resume_ckpt["model_version"])
        if ("version" in model_cfg or "architecture" in model_cfg) and ckpt_model_version != model_version:
            raise ValueError(
                f"Config model version '{model_version}' does not match resume checkpoint model version '{ckpt_model_version}'."
            )

    seq_len = int(train_cfg["seq_len"])
    stride = int(train_cfg.get("stride", seq_len))
    if token_cache_enabled:
        token_cache_dir = Path(str(data_cfg.get("token_cache_dir", out_dir / "token_cache")))
        token_cache_dir.mkdir(parents=True, exist_ok=True)
        train_token_path = Path(str(data_cfg.get("train_token_file", token_cache_dir / "train.bin")))
        val_token_path = Path(str(data_cfg.get("val_token_file", token_cache_dir / "val.bin")))
        reuse_token_cache = bool(data_cfg.get("reuse_token_cache", True))

        if is_main:
            if (not reuse_token_cache) or (not train_token_path.exists()):
                train_cache_stats = write_tokenized_corpus(
                    source=source,
                    split=data_cfg.get("train_split", spec.train_split),
                    tokenizer=tokenizer,
                    output_path=train_token_path,
                    max_examples=int(data_cfg["max_train_examples"]),
                    max_tokens=(int(data_cfg["max_train_tokens"]) if data_cfg.get("max_train_tokens") is not None else None),
                    min_chars=int(data_cfg.get("min_chars", 8)),
                    max_chars=(int(data_cfg["max_train_chars"]) if data_cfg.get("max_train_chars") is not None else None),
                    streaming=data_cfg.get("streaming"),
                    cache_dir=data_cfg.get("cache_dir"),
                    add_bos=True,
                    add_eos=True,
                )
                print(f"Train token cache stats: {train_cache_stats}")
            else:
                print(f"Reusing train token cache: {train_token_path}")

            if (not reuse_token_cache) or (not val_token_path.exists()):
                val_cache_stats = write_tokenized_corpus(
                    source=source,
                    split=data_cfg.get("val_split", spec.val_split),
                    tokenizer=tokenizer,
                    output_path=val_token_path,
                    max_examples=int(data_cfg["max_val_examples"]),
                    max_tokens=(int(data_cfg["max_val_tokens"]) if data_cfg.get("max_val_tokens") is not None else None),
                    min_chars=int(data_cfg.get("min_chars", 8)),
                    max_chars=(int(data_cfg["max_val_chars"]) if data_cfg.get("max_val_chars") is not None else None),
                    streaming=data_cfg.get("streaming"),
                    cache_dir=data_cfg.get("cache_dir"),
                    add_bos=True,
                    add_eos=True,
                )
                print(f"Val token cache stats: {val_cache_stats}")
            else:
                print(f"Reusing val token cache: {val_token_path}")

        _maybe_barrier(dist_ctx)

        train_ds = PackedMemmapDataset(train_token_path, seq_len=seq_len, stride=stride)
        val_ds = PackedMemmapDataset(val_token_path, seq_len=seq_len, stride=seq_len)
    else:
        train_ids = prepare_tokens(tokenizer, train_text)
        val_ids = prepare_tokens(tokenizer, val_text)
        train_ds = PackedSequenceDataset(train_ids, seq_len=seq_len, stride=stride)
        val_ds = PackedSequenceDataset(val_ids, seq_len=seq_len, stride=seq_len)

    model = instantiate_model(model_cfg=model_cfg, vocab_size=tokenizer.vocab_size, model_version=model_version)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32" if allow_tf32 else "ieee"
        elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32" if allow_tf32 else "ieee"
        elif hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        matmul_precision = str(train_cfg.get("matmul_precision", "high")).lower()
        if matmul_precision in {"highest", "high", "medium"}:
            torch.set_float32_matmul_precision(matmul_precision)
    model.to(device)

    if is_main:
        if dist_ctx.enabled:
            print(f"Distributed: backend={dist_ctx.backend}, world_size={dist_ctx.world_size}")
        print(f"Device: {device}")
        print(f"Model version: {model_version}")
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    compile_cfg = train_cfg.get("compile", {})
    if isinstance(compile_cfg, bool):
        compile_enabled = compile_cfg
        compile_cfg = {}
    else:
        compile_enabled = bool(compile_cfg.get("enabled", False)) if isinstance(compile_cfg, dict) else False
    if compile_enabled and hasattr(torch, "compile"):
        compile_mode = str(compile_cfg.get("mode", "reduce-overhead"))
        compile_fullgraph = bool(compile_cfg.get("fullgraph", False))
        compile_dynamic = bool(compile_cfg.get("dynamic", False))
        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
            )
            if is_main:
                print(
                    "Enabled torch.compile "
                    f"(mode={compile_mode}, fullgraph={compile_fullgraph}, dynamic={compile_dynamic})."
                )
        except Exception as exc:
            if is_main:
                print(f"Warning: torch.compile failed ({exc!r}); continuing without compile.")

    base_lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    dist_cfg = train_cfg.get("distributed", {})
    if isinstance(dist_cfg, dict):
        use_zero = bool(dist_cfg.get("zero_optimizer", False))
    else:
        use_zero = False
    if use_zero and not dist_ctx.enabled:
        if is_main:
            print("Warning: train.distributed.zero_optimizer=true but distributed training is not enabled; using AdamW.")
        use_zero = False

    use_fused_adamw = bool(train_cfg.get("fused_adamw", device.type == "cuda")) and device.type == "cuda"
    adamw_kwargs: dict[str, Any] = {
        "lr": base_lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
    }
    if use_fused_adamw:
        adamw_kwargs["fused"] = True

    if dist_ctx.enabled and use_zero:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=AdamW,
            **adamw_kwargs,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            **adamw_kwargs,
        )

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    max_steps = int(train_cfg["max_steps"])
    eval_interval = int(train_cfg.get("eval_interval", 200))
    log_interval = int(train_cfg.get("log_interval", 20))
    val_batches = int(train_cfg.get("val_batches", 20))
    eval_planner_diagnostics = bool(train_cfg.get("eval_planner_diagnostics", True))
    final_eval_planner_diagnostics = bool(
        train_cfg.get("final_eval_planner_diagnostics", eval_planner_diagnostics)
    )
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
    v2_boundary_entropy_weight = float(v2_w.get("boundary_entropy", entropy_reg_weight))
    v2_usage_weight = float(v2_w.get("usage_balance", usage_balance_weight))
    v2_rep_unlikelihood_weight = float(v2_w.get("rep_unlikelihood", 0.0))
    v2_rep_unlikelihood_window = int(v2_w.get("rep_window", 0))
    effective_v2_rep_unlikelihood_window = (
        v2_rep_unlikelihood_window if v2_rep_unlikelihood_weight > 0.0 else 0
    )
    v2_commitment = str(model_cfg.get("plan_commitment", "soft"))
    v2_commitment_warmup_steps = int(model_cfg.get("commitment_warmup_steps", 0))
    v2_commitment_ramp_steps = int(model_cfg.get("commitment_ramp_steps", 0))
    v2_plan_temperature_start = float(model_cfg.get("plan_temperature_start", model_cfg.get("plan_temperature", 1.0)))
    v2_plan_temperature_end = float(model_cfg.get("plan_temperature_end", v2_plan_temperature_start))
    if model_version == MODEL_VERSION_V2:
        if v2_commitment not in {"soft", "gumbel_st"}:
            raise ValueError("model.plan_commitment must be 'soft' or 'gumbel_st' for V2.")
        planner_core_weights = {
            "future_contrastive": v2_future_weight,
            "plan_js_div": v2_js_weight,
            "boundary_entropy": v2_boundary_entropy_weight,
            "usage_balance": v2_usage_weight,
        }
        if all(abs(w) <= 1e-12 for w in planner_core_weights.values()):
            if bool(train_cfg.get("allow_zero_v2_planner_losses", False)):
                print(
                    "Warning: all core V2 planner loss weights are zero. "
                    "Planner-required coupling may collapse to weak/degenerate behavior."
                )
            else:
                raise ValueError(
                    "All core V2 planner loss weights are zero. "
                    "Set at least one of train.loss_weights.{future_contrastive, plan_js_div, "
                    "boundary_entropy, usage_balance} > 0, or set "
                    "train.allow_zero_v2_planner_losses=true to override."
                )

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    precision = str(train_cfg.get("precision", "fp16" if device.type == "cuda" else "fp32")).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision '{precision}'. Use fp32, fp16, or bf16.")
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))
    if is_main:
        print(f"Precision: {precision}, grad_accum_steps: {grad_accum_steps}, use_amp: {use_amp}")
        if v2_rep_unlikelihood_weight <= 0.0 and v2_rep_unlikelihood_window > 0:
            print(
                "Rep-unlikelihood disabled by weight=0; "
                f"ignoring rep_window={v2_rep_unlikelihood_window} for faster training/eval."
            )

    max_nonfinite_steps = max(int(train_cfg.get("max_nonfinite_steps", 3)), 1)

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

    if dist_ctx.enabled:
        if is_main:
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
                v2_rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
            )
        else:
            batch_size = 0
        batch_size = _broadcast_int(dist_ctx, batch_size, device=device)
    else:
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
            v2_rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
        )

    train_loader, val_loader = build_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        train_cfg=train_cfg,
        dist_ctx=dist_ctx,
        seed=seed,
    )
    adaptive_batch_cfg = train_cfg.get("adaptive_batch", {})
    batch_size_raw = train_cfg.get("batch_size", 1)
    if isinstance(adaptive_batch_cfg, bool):
        adaptive_batch_enabled = adaptive_batch_cfg
        adaptive_batch_cfg = {}
    elif isinstance(adaptive_batch_cfg, dict):
        adaptive_batch_enabled = bool(adaptive_batch_cfg.get("enabled", _is_auto_batch_size(batch_size_raw)))
    else:
        adaptive_batch_enabled = _is_auto_batch_size(batch_size_raw)
        adaptive_batch_cfg = {}
    reprobe_interval_steps = int(adaptive_batch_cfg.get("reprobe_interval_steps", 0))
    dynamic_reprobe = adaptive_batch_enabled and device.type == "cuda" and reprobe_interval_steps > 0

    save_interval = int(train_cfg.get("save_interval", 0))
    save_optimizer_state = bool(train_cfg.get("save_optimizer_state", True))

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
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as exc:
                if is_main:
                    print(f"Warning: failed to load optimizer state (continuing without it): {exc!r}")
        global_step = int(resume_ckpt.get("global_step", 0))
        if is_main:
            print(f"Resumed from {Path(str(resume_from))} at global_step={global_step}")

    if dist_ctx.enabled:
        ddp_device_ids = [int(device.index)] if device.type == "cuda" and device.index is not None else None
        model = DDP(model, device_ids=ddp_device_ids, broadcast_buffers=False)
    core_model = unwrap_model(model)

    start_time = time.time()

    model.train()
    progress = tqdm(total=max_steps, initial=global_step, desc="train", disable=(not is_main))
    micro_step = 0
    epoch = 0
    tokens_seen = 0
    consecutive_nonfinite_steps = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps:
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch += 1

        reprobe_triggered = False
        for x, y in train_loader:
            if global_step >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tokens_seen += int(x.shape[0]) * int(x.shape[1])
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

            accum_steps = max(int(grad_accum_steps), 1)
            will_step = ((micro_step + 1) % accum_steps) == 0
            sync_ctx = (
                model.no_sync()
                if (dist_ctx.enabled and accum_steps > 1 and not will_step)
                else nullcontext()
            )
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if model_version == MODEL_VERSION_V1:
                        nll, aux = core_model.filtered_nll(x, y)
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
                    elif model_version == MODEL_VERSION_V2:
                        progress_frac = float(global_step) / float(max(max_steps - 1, 1))
                        current_plan_temp = lerp(v2_plan_temperature_start, v2_plan_temperature_end, progress_frac)
                        if v2_commitment == "gumbel_st" and global_step < v2_commitment_warmup_steps:
                            current_commitment = "soft"
                            current_hard_frac = 0.0
                        elif v2_commitment == "gumbel_st" and v2_commitment_ramp_steps > 0:
                            ramp_progress = float(global_step - v2_commitment_warmup_steps) / float(v2_commitment_ramp_steps)
                            current_hard_frac = max(0.0, min(1.0, ramp_progress))
                            current_commitment = v2_commitment
                        else:
                            current_commitment = v2_commitment
                            current_hard_frac = 1.0
                        nll, aux = core_model.compute_losses(
                            x,
                            y,
                            planner_mode="normal",
                            commitment=current_commitment,
                            planner_temperature=current_plan_temp,
                            commitment_hard_fraction=current_hard_frac,
                            rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
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
                    elif model_version == MODEL_VERSION_TRANSFORMER:
                        nll = core_model.nll(x, y)
                        aux = {}
                        current_plan_temp = 1.0
                        chunk_warm = 1.0
                        effective_chunk_bow_weight = 0.0
                        loss = nll
                    else:
                        raise ValueError(f"Unsupported model version '{model_version}'.")

                    if not _all_finite_tensors(loss, aux):
                        consecutive_nonfinite_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        micro_step = 0
                        if is_main:
                            print(
                                f"\n[warn] non-finite training values at step={global_step} "
                                f"(consecutive={consecutive_nonfinite_steps}/{max_nonfinite_steps}); "
                                "skipping update."
                            )
                        if consecutive_nonfinite_steps >= max_nonfinite_steps:
                            raise RuntimeError(
                                "Aborting: encountered repeated non-finite loss/aux values. "
                                "Lower train.lr and/or adaptive_batch.max_batch_size, then resume from a clean checkpoint."
                            )
                        continue

                    loss_for_backward = loss / float(accum_steps)

                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
            micro_step += 1

            if not will_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=grad_clip)
            if not bool(torch.isfinite(grad_norm).all().item()):
                consecutive_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                micro_step = 0
                if is_main:
                    print(
                        f"\n[warn] non-finite grad norm at step={global_step} "
                        f"(consecutive={consecutive_nonfinite_steps}/{max_nonfinite_steps}); "
                        "skipping optimizer step."
                    )
                if consecutive_nonfinite_steps >= max_nonfinite_steps:
                    raise RuntimeError(
                        "Aborting: encountered repeated non-finite gradients. "
                        "Lower train.lr and/or adaptive_batch.max_batch_size, then resume from a clean checkpoint."
                    )
                continue
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            consecutive_nonfinite_steps = 0

            global_step += 1
            if is_main:
                progress.update(1)

            if dynamic_reprobe and (global_step % reprobe_interval_steps == 0) and global_step < max_steps:
                if dist_ctx.enabled:
                    if is_main:
                        with model.no_sync():
                            new_batch_size = resolve_batch_size(
                                train_cfg=train_cfg,
                                device=device,
                                model=core_model,
                                model_version=model_version,
                                train_ds=train_ds,
                                use_amp=use_amp,
                                amp_dtype=amp_dtype,
                                loss_weights=loss_weights,
                                v2_commitment=v2_commitment,
                                v2_plan_temperature=v2_plan_temperature_start,
                                v2_rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
                            )
                    else:
                        new_batch_size = 0
                    new_batch_size = _broadcast_int(dist_ctx, new_batch_size, device=device)
                else:
                    new_batch_size = resolve_batch_size(
                        train_cfg=train_cfg,
                        device=device,
                        model=core_model,
                        model_version=model_version,
                        train_ds=train_ds,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        loss_weights=loss_weights,
                        v2_commitment=v2_commitment,
                        v2_plan_temperature=v2_plan_temperature_start,
                        v2_rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
                    )
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    train_loader, val_loader = build_loaders(
                        train_ds=train_ds,
                        val_ds=val_ds,
                        batch_size=batch_size,
                        train_cfg=train_cfg,
                        dist_ctx=dist_ctx,
                        seed=seed,
                    )
                    if is_main:
                        print(f"\n[adaptive batch] step={global_step} updated batch_size={batch_size}")
                    reprobe_triggered = True
                    break

            if is_main and (global_step % log_interval == 0 or global_step == 1):
                elapsed = time.time() - start_time
                total_tokens = int(tokens_seen) * (int(dist_ctx.world_size) if dist_ctx.enabled else 1)
                tok_per_s = float(total_tokens) / max(elapsed, 1e-6)
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
                elif model_version == MODEL_VERSION_V2:
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
                elif model_version == MODEL_VERSION_TRANSFORMER:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        ppl=f"{math.exp(min(nll.item(), 20.0)):.2f}",
                        lr=f"{current_lr:.2e}",
                        tok_s=f"{tok_per_s:.0f}",
                    )
                else:
                    raise ValueError(f"Unsupported model version '{model_version}'.")

            if global_step % eval_interval == 0 or global_step == max_steps:
                eval_commitment = "soft"
                eval_plan_temp = 1.0
                if model_version == MODEL_VERSION_V2:
                    eval_progress_frac = float(global_step) / float(max(max_steps - 1, 1))
                    eval_plan_temp = lerp(v2_plan_temperature_start, v2_plan_temperature_end, eval_progress_frac)
                    eval_commitment = (
                        "soft"
                        if (v2_commitment == "gumbel_st" and global_step < v2_commitment_warmup_steps)
                        else v2_commitment
                    )
                if is_main:
                    eval_metrics = evaluate_model(
                        core_model,
                        model_version,
                        val_loader,
                        device,
                        max_batches=val_batches,
                        rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
                        v2_commitment=eval_commitment,
                        v2_planner_temperature=eval_plan_temp,
                        include_planner_diagnostics=eval_planner_diagnostics,
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
                    elif model_version == MODEL_VERSION_V2:
                        print(
                            f"\n[eval step {global_step}] "
                            f"val_loss={eval_metrics['loss']:.4f} "
                            f"val_ppl={eval_metrics['ppl']:.2f} "
                            f"usage_kl={eval_metrics['usage_kl_to_uniform']:.3f} "
                            f"bent={eval_metrics['boundary_entropy']:.3f} "
                            f"fcl={eval_metrics['future_contrastive_loss']:.3f} "
                            f"js={eval_metrics['plan_js_div_loss']:.3f} "
                            f"repu={eval_metrics['rep_unlikelihood_loss']:.3f} "
                            f"fb_d={eval_metrics['feedback_delta_loss']:.3f} "
                            f"mask_d={eval_metrics['planner_mask_delta_loss']:.3f} "
                            f"force_js={eval_metrics['forced_state_divergence']:.3f} "
                            f"sp={eval_metrics['state_persistence']:.3f} "
                            f"eu={eval_metrics['expert_utilization']:.3f}"
                        )
                    elif model_version == MODEL_VERSION_TRANSFORMER:
                        print(
                            f"\n[eval step {global_step}] "
                            f"val_loss={eval_metrics['loss']:.4f} "
                            f"val_ppl={eval_metrics['ppl']:.2f}"
                        )
                    else:
                        raise ValueError(f"Unsupported model version '{model_version}'.")
                _maybe_barrier(dist_ctx)
                core_model.train()

            if global_step % preview_interval == 0 or global_step == max_steps:
                if is_main:
                    preview = maybe_sample_preview(
                        model=core_model,
                        model_version=model_version,
                        tokenizer=tokenizer,
                        prompt=preview_prompt,
                        max_new_tokens=preview_tokens,
                    )
                    print(f"\n[sample step {global_step}] {sanitize_for_console(preview[:500])}\n")
                _maybe_barrier(dist_ctx)
                core_model.train()

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
                    dist_ctx=dist_ctx,
                    save_optimizer_state=save_optimizer_state,
                )
                _maybe_barrier(dist_ctx)
                if is_main:
                    print(f"\n[checkpoint step {global_step}] saved to {out_dir / 'checkpoint.pt'}")
        if reprobe_triggered:
            continue

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
        dist_ctx=dist_ctx,
        save_optimizer_state=save_optimizer_state,
    )
    _maybe_barrier(dist_ctx)

    if is_main:
        metrics = evaluate_model(
            core_model,
            model_version,
            val_loader,
            device,
            max_batches=val_batches,
            rep_unlikelihood_window=effective_v2_rep_unlikelihood_window,
            v2_commitment=(
                "soft"
                if (
                    model_version == MODEL_VERSION_V2
                    and v2_commitment == "gumbel_st"
                    and global_step < v2_commitment_warmup_steps
                )
                else v2_commitment
            ),
            v2_planner_temperature=lerp(
                v2_plan_temperature_start,
                v2_plan_temperature_end,
                float(global_step) / float(max(max_steps - 1, 1)),
            ),
            include_planner_diagnostics=final_eval_planner_diagnostics,
        )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if model_version == MODEL_VERSION_V2:
            planner_eval = {
                "planner_mask_delta_loss": metrics.get("planner_mask_delta_loss"),
                "forced_state_divergence": metrics.get("forced_state_divergence"),
                "state_persistence": metrics.get("state_persistence"),
                "expert_utilization": metrics.get("expert_utilization"),
                "future_contrastive_loss": metrics.get("future_contrastive_loss"),
                "plan_js_div_loss": metrics.get("plan_js_div_loss"),
                "feedback_delta_loss": metrics.get("feedback_delta_loss"),
            }
            (out_dir / "planner_eval.json").write_text(json.dumps(planner_eval, indent=2), encoding="utf-8")
        print(f"Saved checkpoint to: {ckpt_path}")
        print(f"Final metrics: {metrics}")

    _maybe_barrier(dist_ctx)
    if dist_ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

