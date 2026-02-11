from __future__ import annotations

import argparse
import json
import math
import random
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
from .tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prepare_tokens(tokenizer: CharTokenizer, text: str) -> list[int]:
    ids = [tokenizer.bos_id]
    ids.extend(tokenizer.encode(text, add_bos=False, add_eos=True))
    return ids


def evaluate(model: APELRModel, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    entropies: list[float] = []
    usage_kls: list[float] = []
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
    if not losses:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "belief_entropy": float("nan"),
            "usage_kl_to_uniform": float("nan"),
        }
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "ppl": float(math.exp(min(mean_loss, 20.0))),
        "belief_entropy": float(np.mean(entropies)),
        "usage_kl_to_uniform": float(np.mean(usage_kls)),
    }


def maybe_sample_preview(
    *,
    model: APELRModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_bos=False, add_eos=False)
    sample_ids, _ = model.generate_filtered(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_k=40,
        eos_id=tokenizer.eos_id,
        lookahead_steps=1,
    )
    return tokenizer.decode(sample_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train APEL-R model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

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
        streaming=data_cfg.get("streaming"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    val_text, val_stats = load_corpus_text(
        source=source,
        split=data_cfg.get("val_split", spec.val_split),
        max_examples=int(data_cfg["max_val_examples"]),
        min_chars=int(data_cfg.get("min_chars", 8)),
        streaming=data_cfg.get("streaming"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    print(f"Train stats: {train_stats}")
    print(f"Val stats:   {val_stats}")

    tokenizer = CharTokenizer.fit(train_text)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    train_ids = prepare_tokens(tokenizer, train_text)
    val_ids = prepare_tokens(tokenizer, val_text)

    seq_len = int(train_cfg["seq_len"])
    stride = int(train_cfg.get("stride", seq_len))
    train_ds = PackedSequenceDataset(train_ids, seq_len=seq_len, stride=stride)
    val_ds = PackedSequenceDataset(val_ids, seq_len=seq_len, stride=seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )

    model = APELRModel(
        APELRModelConfig(
            vocab_size=tokenizer.vocab_size,
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

    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
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
    entropy_reg_weight = float(train_cfg.get("entropy_reg_weight", 0.0))
    usage_balance_weight = float(train_cfg.get("usage_balance_weight", 0.0))

    global_step = 0
    start_time = time.time()

    model.train()
    progress = tqdm(total=max_steps, desc="train")
    while global_step < max_steps:
        for x, y in train_loader:
            if global_step >= max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            nll, aux = model.filtered_nll(x, y)
            loss = nll + entropy_reg_weight * aux["belief_entropy"] + usage_balance_weight * aux["usage_kl_to_uniform"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            global_step += 1
            progress.update(1)

            if global_step % log_interval == 0 or global_step == 1:
                elapsed = time.time() - start_time
                tok_per_s = (global_step * x.shape[0] * x.shape[1]) / max(elapsed, 1e-6)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ppl=f"{math.exp(min(nll.item(), 20.0)):.2f}",
                    ent=f"{aux['belief_entropy'].item():.3f}",
                    klu=f"{aux['usage_kl_to_uniform'].item():.3f}",
                    tok_s=f"{tok_per_s:.0f}",
                )

            if global_step % eval_interval == 0 or global_step == max_steps:
                eval_metrics = evaluate(model, val_loader, device, max_batches=val_batches)
                print(
                    f"\n[eval step {global_step}] "
                    f"val_loss={eval_metrics['loss']:.4f} "
                    f"val_ppl={eval_metrics['ppl']:.2f} "
                    f"belief_ent={eval_metrics['belief_entropy']:.3f} "
                    f"usage_kl={eval_metrics['usage_kl_to_uniform']:.3f}"
                )
                model.train()

            if global_step % preview_interval == 0 or global_step == max_steps:
                preview = maybe_sample_preview(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=preview_prompt,
                    max_new_tokens=preview_tokens,
                )
                print(f"\n[sample step {global_step}] {preview[:500]}\n")
                model.train()

    progress.close()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.cfg.__dict__,
        "tokenizer_path": str(tokenizer_path),
        "train_config": cfg,
    }
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    metrics = evaluate(model, val_loader, device, max_batches=val_batches)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to: {ckpt_path}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
