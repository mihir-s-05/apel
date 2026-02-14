from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from .data import PackedSequenceDataset
from .model import APELRModel, APELRModelConfig
from .model_v2 import APELRV2Model, APELRV2ModelConfig
from .tokenizer import SPECIAL_TOKENS, TokenizerLike, load_tokenizer
from .transformer_lm import TransformerLM, TransformerLMConfig

MODEL_VERSION_V1 = "v1_filtered_mixture"
MODEL_VERSION_V2 = "v2_planner_required"
MODEL_VERSION_TRANSFORMER = "transformer_lm"


@dataclass(frozen=True)
class EvalSettings:
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    max_input_len: int
    max_examples: int
    batch_size: int
    v2_commitment: str
    v2_planner_temperature: float
    v2_lookahead_steps: int | None
    v2_lookahead_feedback_scale: float | None


def _autocast_ctx(settings: EvalSettings):
    return torch.autocast(device_type=settings.device.type, dtype=settings.amp_dtype, enabled=settings.use_amp)


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module, str, TokenizerLike]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_version = str(ckpt.get("model_version", MODEL_VERSION_V1))
    if model_version == MODEL_VERSION_V1:
        model_cfg = APELRModelConfig(**ckpt["model_config"])
        model: torch.nn.Module = APELRModel(model_cfg)
    elif model_version == MODEL_VERSION_V2:
        model_cfg = APELRV2ModelConfig(**ckpt["model_config"])
        model = APELRV2Model(model_cfg)
    elif model_version == MODEL_VERSION_TRANSFORMER:
        model_cfg = TransformerLMConfig(**ckpt["model_config"])
        model = TransformerLM(model_cfg)
    else:
        raise ValueError(f"Unsupported model_version '{model_version}' in checkpoint: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys in checkpoint load: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint load: {unexpected}")

    tok_path = ckpt.get("tokenizer_path")
    if tok_path is None:
        raise ValueError("Checkpoint missing tokenizer_path.")
    tok_type = ckpt.get("tokenizer_type")
    tok_cfg = ckpt.get("tokenizer_config", {})
    tokenizer = load_tokenizer(
        Path(str(tok_path)),
        tokenizer_type=tok_type,
        special_tokens=tok_cfg.get("special_tokens", SPECIAL_TOKENS),
    )

    model.to(device)
    model.eval()
    return ckpt, model, model_version, tokenizer


def _infer_max_input_len(ckpt: dict[str, Any], model: torch.nn.Module, model_version: str, default_len: int = 256) -> int:
    if model_version == MODEL_VERSION_TRANSFORMER:
        return int(getattr(model, "max_seq_len"))
    train_cfg = ckpt.get("train_config") or {}
    seq_len = (train_cfg.get("train") or {}).get("seq_len")
    if seq_len is not None:
        return int(seq_len)
    return int(default_len)


@torch.inference_mode()
def _nll_sum_ids(
    model: torch.nn.Module,
    model_version: str,
    ids: list[int],
    settings: EvalSettings,
) -> float:
    # Sum over t of -log p(ids[t] | ids[:t]) for t >= 1.
    if len(ids) < 2:
        return 0.0
    x = torch.tensor(ids[:-1], device=settings.device, dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], device=settings.device, dtype=torch.long).unsqueeze(0)
    with _autocast_ctx(settings):
        if model_version == MODEL_VERSION_V1:
            nll_mean, _aux = model.filtered_nll(x, y)
        elif model_version == MODEL_VERSION_V2:
            nll_mean, _aux = model.compute_losses(
                x,
                y,
                planner_mode="normal",
                commitment=settings.v2_commitment,
                planner_temperature=settings.v2_planner_temperature,
                lookahead_steps=settings.v2_lookahead_steps,
                lookahead_feedback_scale=settings.v2_lookahead_feedback_scale,
                rep_unlikelihood_window=0,
            )
        elif model_version == MODEL_VERSION_TRANSFORMER:
            nll_mean = model.nll(x, y)
        else:
            raise ValueError(f"Unsupported model_version '{model_version}'.")
    return float(nll_mean.item()) * float(x.shape[1])


@torch.inference_mode()
def _v1_teacher_forced_log_probs(
    model: APELRModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    settings: EvalSettings,
) -> torch.Tensor:
    if input_ids.shape != target_ids.shape:
        raise ValueError("input_ids and target_ids must have same shape")
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    with _autocast_ctx(settings):
        logp_per_plan, ctx_logits, log_probs, _h = model._logp_per_plan(input_ids, target_ids)
        P = model.transition_matrix()
        belief = model.initial_belief(bsz, device)
        mix_steps: list[torch.Tensor] = []
        for t in range(int(seq_len)):
            if t > 0 and t % model.chunk_size == 0:
                belief = model._apply_chunk_boundary_update(belief, P, ctx_logits[:, t, :])
            log_b = model._belief_log(belief)
            mix_t = torch.logsumexp(log_b.unsqueeze(-1) + log_probs[:, t, :, :], dim=1)
            mix_steps.append(mix_t)
            belief = model._apply_posterior_update(belief, logp_per_plan[:, t, :])
        return torch.stack(mix_steps, dim=1)


@torch.inference_mode()
def _v2_teacher_forced_log_probs(
    model: APELRV2Model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    settings: EvalSettings,
) -> torch.Tensor:
    if input_ids.shape != target_ids.shape:
        raise ValueError("input_ids and target_ids must have same shape")
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    effective_lookahead_steps = max(
        0,
        int(model.cfg.lookahead_horizon if settings.v2_lookahead_steps is None else settings.v2_lookahead_steps),
    )
    effective_feedback_scale = min(
        max(
            float(
                model.cfg.lookahead_feedback_scale
                if settings.v2_lookahead_feedback_scale is None
                else settings.v2_lookahead_feedback_scale
            ),
            0.0,
        ),
        1.0,
    )
    planner_tau = max(float(settings.v2_planner_temperature), 1e-4)

    with _autocast_ctx(settings):
        x = model.token_emb(input_ids)
        h, _ = model.backbone(x)
        chunk_summary, _token_chunk_idx_ctx = model._chunk_summaries(h)
        expert_log_probs = model._expert_log_probs(h)
        trans = model.transition_matrix()

        belief = model._initial_belief(bsz, device)
        current_chunk = 0
        mix_steps: list[torch.Tensor] = []

        for t in range(int(seq_len)):
            target_chunk = int(((t + 1) // model.chunk_size))
            if target_chunk != current_chunk:
                if target_chunk != current_chunk + 1:
                    raise RuntimeError(f"Unexpected chunk jump: {current_chunk} -> {target_chunk}")
                if current_chunk >= chunk_summary.shape[1]:
                    raise RuntimeError("Chunk summary index out of range for boundary update.")
                belief = model._apply_boundary_update(belief, trans, chunk_summary[:, current_chunk, :])
                current_chunk = target_chunk

            gate_t, _ = model._planner_gate_with_lookahead(
                belief,
                trans,
                planner_temperature=planner_tau,
                lookahead_steps=effective_lookahead_steps,
                feedback_scale=effective_feedback_scale,
            )
            mix_t = torch.logsumexp(model._belief_log(gate_t).unsqueeze(-1) + expert_log_probs[:, t, :, :], dim=1)
            mix_steps.append(mix_t)

            # Online token filtering update (teacher-forced on target_ids).
            tgt = target_ids[:, t]
            obs_logp = torch.gather(
                expert_log_probs[:, t, :, :],
                dim=-1,
                index=tgt.view(-1, 1, 1).expand(-1, model.num_plan_states, 1),
            ).squeeze(-1)
            belief = F.softmax(model._belief_log(belief) + obs_logp, dim=-1)

        return torch.stack(mix_steps, dim=1)


@torch.inference_mode()
def _teacher_forced_log_probs(
    model: torch.nn.Module,
    model_version: str,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    settings: EvalSettings,
) -> torch.Tensor:
    if model_version == MODEL_VERSION_V1:
        return _v1_teacher_forced_log_probs(model, input_ids, target_ids, settings)
    if model_version == MODEL_VERSION_V2:
        return _v2_teacher_forced_log_probs(model, input_ids, target_ids, settings)
    if model_version == MODEL_VERSION_TRANSFORMER:
        with _autocast_ctx(settings):
            logits = model.forward(input_ids)
        return F.log_softmax(logits.float(), dim=-1)
    raise ValueError(f"Unsupported model_version '{model_version}'.")


@torch.inference_mode()
def _continuation_nll_sum_and_first_log_probs(
    model: torch.nn.Module,
    model_version: str,
    full_ids: list[int],
    *,
    prefix_len: int,
    settings: EvalSettings,
) -> tuple[float, torch.Tensor | None]:
    if len(full_ids) < 2:
        return 0.0, None
    if prefix_len <= 0:
        raise ValueError("prefix_len must be > 0")
    if prefix_len >= len(full_ids):
        return 0.0, None

    x = torch.tensor(full_ids[:-1], device=settings.device, dtype=torch.long).unsqueeze(0)
    y = torch.tensor(full_ids[1:], device=settings.device, dtype=torch.long).unsqueeze(0)
    start_t = max(int(prefix_len) - 1, 0)

    log_probs = _teacher_forced_log_probs(model, model_version, x, y, settings)  # [1, T, V]
    sel = log_probs[:, start_t:, :]
    tgt = y[:, start_t:]
    lp = torch.gather(sel, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    nll_sum = float((-lp).sum().item())
    first = log_probs[0, start_t, :]
    return nll_sum, first

def _truncate_prompt_keep_bos(
    *,
    bos_id: int,
    prompt_no_bos: list[int],
    cont_ids: list[int],
    max_input_len: int,
) -> list[int]:
    # We feed input_ids of length <= max_input_len. Total ids length is max_input_len + 1.
    max_total = int(max_input_len) + 1
    keep_prompt = max_total - 1 - len(cont_ids)
    keep_prompt = max(0, int(keep_prompt))
    tail = prompt_no_bos[-keep_prompt:] if keep_prompt > 0 else []
    return [int(bos_id)] + [int(x) for x in tail]


def _truncate_continuation(cont_ids: list[int], *, max_input_len: int) -> list[int]:
    # Ensure (BOS + prompt_tail + cont_ids) can fit in max_input_len+1 tokens by
    # hard-capping the continuation when necessary (e.g., PIQA can be long under small BPE vocabs).
    max_total = int(max_input_len) + 1
    max_cont = max_total - 1
    if max_cont <= 0:
        return []
    if len(cont_ids) <= max_cont:
        return cont_ids
    return cont_ids[:max_cont]


def _pick_join(ctx: str, ending: str) -> str:
    if not ctx:
        return ending
    if not ending:
        return ending
    if ctx.endswith((" ", "\n", "\t")):
        return ctx + ending
    if ending[:1] in {" ", "\n", "\t", ".", ",", ";", ":", "?", "!", "'", "\""}:
        return ctx + ending
    return ctx + " " + ending


def _preprocess_hellaswag(doc: dict[str, Any]) -> dict[str, Any]:
    # Match EleutherAI lm-evaluation-harness behavior.
    ctx = str(doc["ctx_a"]) + " " + str(doc["ctx_b"]).capitalize()
    endings = [str(e).capitalize() for e in doc["endings"]]
    endings = [e.replace(" [title]", ".") for e in endings]
    endings = [e.replace("  ", " ") for e in endings]
    return {"ctx": ctx, "endings": endings}


@torch.inference_mode()
def _score_choices(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    prompt: str,
    choices: list[str],
    settings: EvalSettings,
) -> tuple[list[float], list[float]]:
    prompt_no_bos = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    prompt_no_bos = [int(x) for x in prompt_no_bos]
    sums: list[float] = []
    means: list[float] = []
    for choice in choices:
        cont = tokenizer.encode(choice, add_bos=False, add_eos=False)
        cont_ids = _truncate_continuation([int(x) for x in cont], max_input_len=settings.max_input_len)
        prompt_ids = _truncate_prompt_keep_bos(
            bos_id=int(tokenizer.bos_id),
            prompt_no_bos=prompt_no_bos,
            cont_ids=cont_ids,
            max_input_len=settings.max_input_len,
        )
        full_ids = prompt_ids + cont_ids
        cond, _first = _continuation_nll_sum_and_first_log_probs(
            model,
            model_version,
            full_ids,
            prefix_len=len(prompt_ids),
            settings=settings,
        )
        denom = max(len(cont_ids), 1)
        sums.append(float(cond))
        means.append(float(cond) / float(denom))
    return sums, means


def _acc_from_scores(scores: list[float]) -> int:
    if not scores:
        return -1
    return int(min(range(len(scores)), key=lambda i: scores[i]))


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return float("nan")
    return float(np.mean(xs))


@torch.inference_mode()
def eval_wikitext_ppl(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    name: str,
    settings: EvalSettings,
    split: str = "test",
    min_chars: int = 8,
) -> dict[str, float]:
    if name not in {"wikitext2", "wikitext103"}:
        raise ValueError("name must be wikitext2 or wikitext103")
    variant = "wikitext-2-raw-v1" if name == "wikitext2" else "wikitext-103-raw-v1"
    ds = load_dataset("Salesforce/wikitext", variant, split=split)

    # Tokenize similarly to training (BOS once, then concatenate docs with double newlines).
    texts: list[str] = []
    for i in range(min(len(ds), max(settings.max_examples, 1_000_000))):
        s = str(ds[i].get("text", "")).strip()
        if len(s) < int(min_chars):
            continue
        texts.append(s)
    corpus = "\n\n".join(texts)
    ids = [int(tokenizer.bos_id)]
    ids.extend(int(x) for x in tokenizer.encode(corpus, add_bos=False, add_eos=True))
    if len(ids) <= (settings.max_input_len + 1):
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": float(len(ids))}

    ds_tok = PackedSequenceDataset(ids, seq_len=settings.max_input_len, stride=settings.max_input_len)
    loader = DataLoader(ds_tok, batch_size=settings.batch_size, shuffle=False, drop_last=False, num_workers=0)
    losses: list[float] = []
    for x, y in loader:
        x = x.to(settings.device)
        y = y.to(settings.device)
        if model_version in {MODEL_VERSION_V1, MODEL_VERSION_V2}:
            log_probs = _teacher_forced_log_probs(model, model_version, x, y, settings)
            nll = F.nll_loss(log_probs.reshape(-1, tokenizer.vocab_size), y.reshape(-1), reduction="mean")
        elif model_version == MODEL_VERSION_TRANSFORMER:
            with _autocast_ctx(settings):
                nll = model.nll(x, y)
        else:
            raise ValueError(f"Unsupported model_version '{model_version}'.")
        losses.append(float(nll.item()))

    mean_loss = _mean(losses)
    return {"loss": mean_loss, "ppl": float(math.exp(min(mean_loss, 20.0))), "tokens": float(len(ids))}


@torch.inference_mode()
def eval_lambada_openai(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    total_nll = 0.0
    total_tokens = 0
    correct_1tok = 0
    total_1tok = 0

    upper = min(len(ds), int(settings.max_examples))
    for i in range(upper):
        text = str(ds[i].get("text", "")).strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        prompt = " ".join(parts[:-1])
        target = " " + str(parts[-1])

        prompt_no_bos = [int(x) for x in tokenizer.encode(prompt, add_bos=False, add_eos=False)]
        cont_ids = _truncate_continuation(
            [int(x) for x in tokenizer.encode(target, add_bos=False, add_eos=False)],
            max_input_len=settings.max_input_len,
        )
        if not cont_ids:
            continue
        prompt_ids = _truncate_prompt_keep_bos(
            bos_id=int(tokenizer.bos_id),
            prompt_no_bos=prompt_no_bos,
            cont_ids=cont_ids,
            max_input_len=settings.max_input_len,
        )
        full_ids = prompt_ids + cont_ids
        nll_sum, first = _continuation_nll_sum_and_first_log_probs(
            model,
            model_version,
            full_ids,
            prefix_len=len(prompt_ids),
            settings=settings,
        )
        total_nll += float(nll_sum)
        total_tokens += int(len(cont_ids))
        if len(cont_ids) == 1 and first is not None:
            pred = int(torch.argmax(first).item())
            correct_1tok += int(pred == int(cont_ids[0]))
            total_1tok += 1

    if total_tokens <= 0:
        return {"nll_per_token": float("nan"), "ppl": float("nan"), "acc_1tok": float("nan"), "examples": float(upper)}
    mean_nll = float(total_nll) / float(total_tokens)
    return {
        "nll_per_token": mean_nll,
        "ppl": float(math.exp(min(mean_nll, 20.0))),
        "acc_1tok": float(correct_1tok) / float(total_1tok) if total_1tok > 0 else float("nan"),
        "examples": float(upper),
        "tokens": float(total_tokens),
        "examples_1tok": float(total_1tok),
    }


@torch.inference_mode()
def eval_piqa(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    # lm-eval-harness piqa.yaml:
    # doc_to_text: "Question: {{goal}}\nAnswer:"
    # doc_to_choice: [sol1, sol2]
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        prompt = f"Question: {ex['goal']}\nAnswer:"
        choices = [str(ex["sol1"]), str(ex["sol2"])]
        gold = int(ex["label"])
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choices,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


@torch.inference_mode()
def eval_social_iqa(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    # lm-eval-harness siqa.yaml:
    # doc_to_text: "Q: {{context}} {{question}}\nA:"
    # doc_to_choice: [answerA, answerB, answerC]
    # doc_to_target: label-1, target_delimiter: " "
    ds = load_dataset("social_i_qa", split="validation", trust_remote_code=True)
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        prompt = f"Q: {ex['context']} {ex['question']}\nA:"
        choices = [f" {ex['answerA']}", f" {ex['answerB']}", f" {ex['answerC']}"]
        gold = int(ex["label"]) - 1
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choices,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


@torch.inference_mode()
def eval_hellaswag(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    ds = load_dataset("hellaswag", split="validation")
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = _preprocess_hellaswag(ds[i])
        prompt = str(ex["ctx"])
        choices = [_pick_join("", str(e)) for e in ex["endings"]]
        # In harness, endings are scored as continuations to ctx (no extra label).
        choices = [_pick_join(" ", c).lstrip("\n") if not c.startswith(" ") else c for c in choices]
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choices,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        label = int(ds[i]["label"])
        correct += int(pred == label)
        correct_norm += int(pred_norm == label)
        gold_nll_sum.append(float(scores_sum[label]))
        gold_nll_mean.append(float(scores_mean[label]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


def _choice_target_index_from_label(choice_labels: list[str], answer_key: str) -> int:
    if answer_key in choice_labels:
        return int(choice_labels.index(answer_key))
    # Some datasets use numeric keys.
    for i, lab in enumerate(choice_labels):
        if str(lab).strip().lower() == str(answer_key).strip().lower():
            return int(i)
    raise ValueError(f"answerKey '{answer_key}' not found in labels {choice_labels}")


@torch.inference_mode()
def eval_arc(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
    subset: str,
) -> dict[str, float]:
    if subset not in {"ARC-Easy", "ARC-Challenge"}:
        raise ValueError("subset must be ARC-Easy or ARC-Challenge")
    ds = load_dataset("ai2_arc", subset, split="validation")
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        prompt = f"{ex['question']}\nAnswer:"
        choice_texts = [str(x) for x in ex["choices"]["text"]]
        choice_labels = [str(x) for x in ex["choices"]["label"]]
        gold = _choice_target_index_from_label(choice_labels, str(ex["answerKey"]))
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choice_texts,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


@torch.inference_mode()
def eval_openbookqa(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    ds = load_dataset("openbookqa", split="validation")
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        prompt = f"Question: {ex['question_stem']}\nAnswer:"
        choice_texts = [str(x) for x in ex["choices"]["text"]]
        choice_labels = [str(x) for x in ex["choices"]["label"]]
        gold = _choice_target_index_from_label(choice_labels, str(ex["answerKey"]))
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choice_texts,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


@torch.inference_mode()
def eval_winogrande_xl(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        sentence = str(ex["sentence"])
        if "_" not in sentence:
            continue
        prefix, suffix = sentence.split("_", 1)
        prompt = prefix
        choices = [str(ex["option1"]) + suffix, str(ex["option2"]) + suffix]
        gold = int(str(ex["answer"]).strip()) - 1
        if gold not in (0, 1):
            continue
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choices,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


@torch.inference_mode()
def eval_boolq(
    *,
    model: torch.nn.Module,
    model_version: str,
    tokenizer: TokenizerLike,
    settings: EvalSettings,
) -> dict[str, float]:
    ds = load_dataset("boolq", split="validation")
    upper = min(len(ds), int(settings.max_examples))
    correct = 0
    correct_norm = 0
    gold_nll_sum: list[float] = []
    gold_nll_mean: list[float] = []

    for i in range(upper):
        ex = ds[i]
        passage = str(ex["passage"])
        question = str(ex["question"]).rstrip("?")
        prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
        choices = [" no", " yes"]
        gold = 1 if bool(ex["answer"]) else 0
        scores_sum, scores_mean = _score_choices(
            model=model,
            model_version=model_version,
            tokenizer=tokenizer,
            prompt=prompt,
            choices=choices,
            settings=settings,
        )
        pred = _acc_from_scores(scores_sum)
        pred_norm = _acc_from_scores(scores_mean)
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        gold_nll_sum.append(float(scores_sum[gold]))
        gold_nll_mean.append(float(scores_mean[gold]))

    return {
        "acc": float(correct) / float(max(upper, 1)),
        "acc_norm": float(correct_norm) / float(max(upper, 1)),
        "gold_nll_sum": _mean(gold_nll_sum),
        "gold_nll_mean": _mean(gold_nll_mean),
        "examples": float(upper),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardized evals (LM-harness style) for APEL-R and baselines.")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="One or more checkpoint.pt paths.")
    p.add_argument("--tasks", type=str, default="titans", help="Comma list or 'titans' or 'all'.")
    p.add_argument("--max-examples", type=int, default=512, help="Max examples per task (except WikiText).")
    p.add_argument("--max-input-len", type=int, default=0, help="Override max input length (tokens). 0 uses checkpoint defaults.")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for perplexity evaluation.")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--v2-planner-temperature", type=float, default=1.0)
    p.add_argument("--v2-commitment", type=str, default="soft", choices=["soft", "gumbel_st"])
    p.add_argument("--no-lookahead", action="store_true", help="Disable V2 lookahead in gating for eval (steps=0, scale=0).")
    p.add_argument("--output", type=str, default="", help="Write JSON results to this path.")
    return p.parse_args()


def _expand_tasks(tasks: str) -> list[str]:
    tasks = tasks.strip().lower()
    if tasks in {"titans", "titan"}:
        return [
            "wikitext2",
            "wikitext103",
            "lambada_openai",
            "piqa",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande_xl",
            "social_iqa",
            "openbookqa",
            "boolq",
        ]
    if tasks in {"all"}:
        return _expand_tasks("titans")
    return [t.strip() for t in tasks.split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    precision = str(args.precision).lower()
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    tasks = _expand_tasks(str(args.tasks))

    results: dict[str, Any] = {"created_at": time.time(), "device": str(device), "precision": precision, "tasks": tasks, "runs": {}}

    for ckpt_path in args.checkpoints:
        ckpt, model, model_version, tokenizer = _load_checkpoint(ckpt_path, device=device)
        max_input_len = int(args.max_input_len) if int(args.max_input_len) > 0 else _infer_max_input_len(ckpt, model, model_version)
        settings = EvalSettings(
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_input_len=max_input_len,
            max_examples=int(args.max_examples),
            batch_size=int(args.batch_size),
            v2_commitment=str(args.v2_commitment),
            v2_planner_temperature=float(args.v2_planner_temperature),
            v2_lookahead_steps=(0 if bool(args.no_lookahead) else None),
            v2_lookahead_feedback_scale=(0.0 if bool(args.no_lookahead) else None),
        )

        run: dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "model_version": model_version,
            "max_input_len": max_input_len,
            "metrics": {},
        }

        for task in tasks:
            t0 = time.time()
            if task == "wikitext2":
                run["metrics"][task] = eval_wikitext_ppl(
                    model=model, model_version=model_version, tokenizer=tokenizer, name="wikitext2", settings=settings, split="test"
                )
            elif task == "wikitext103":
                run["metrics"][task] = eval_wikitext_ppl(
                    model=model, model_version=model_version, tokenizer=tokenizer, name="wikitext103", settings=settings, split="test"
                )
            elif task == "lambada_openai":
                run["metrics"][task] = eval_lambada_openai(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "piqa":
                run["metrics"][task] = eval_piqa(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "hellaswag":
                run["metrics"][task] = eval_hellaswag(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "arc_easy":
                run["metrics"][task] = eval_arc(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings, subset="ARC-Easy"
                )
            elif task == "arc_challenge":
                run["metrics"][task] = eval_arc(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings, subset="ARC-Challenge"
                )
            elif task == "winogrande_xl":
                run["metrics"][task] = eval_winogrande_xl(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "social_iqa":
                run["metrics"][task] = eval_social_iqa(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "openbookqa":
                run["metrics"][task] = eval_openbookqa(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            elif task == "boolq":
                run["metrics"][task] = eval_boolq(
                    model=model, model_version=model_version, tokenizer=tokenizer, settings=settings
                )
            else:
                raise ValueError(f"Unknown task '{task}'.")
            dt = time.time() - t0
            run["metrics"][task]["seconds"] = float(dt)
            print(f"[{Path(str(ckpt_path)).parent.name}] {task}: {json.dumps(run['metrics'][task])}")

        results["runs"][ckpt_path] = run

    if args.output:
        Path(str(args.output)).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
