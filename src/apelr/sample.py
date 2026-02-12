from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import APELRModel, APELRModelConfig
from .model_v2 import APELRV2Model, APELRV2ModelConfig
from .tokenizer import load_tokenizer

MODEL_VERSION_V1 = "v1_filtered_mixture"
MODEL_VERSION_V2 = "v2_planner_required"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from trained APEL-R model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--lookahead-steps", type=int, default=2)
    parser.add_argument("--force-state", type=int, default=None)
    parser.add_argument("--freeze-planner", action="store_true")
    parser.add_argument("--planner-temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_version = str(ckpt.get("model_version", MODEL_VERSION_V1))
    if model_version == MODEL_VERSION_V1:
        model_cfg = APELRModelConfig(**ckpt["model_config"])
        model: torch.nn.Module = APELRModel(model_cfg)
    elif model_version == MODEL_VERSION_V2:
        model_cfg = APELRV2ModelConfig(**ckpt["model_config"])
        model = APELRV2Model(model_cfg)
    else:
        raise ValueError(f"Unsupported model_version '{model_version}' in checkpoint.")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint load: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint load: {unexpected}")
    model.to(args.device)
    model.eval()

    tokenizer_path = ckpt.get("tokenizer_path")
    if tokenizer_path is None:
        raise ValueError("Checkpoint missing tokenizer_path.")
    tokenizer_type = ckpt.get("tokenizer_type")
    tokenizer_cfg = ckpt.get("tokenizer_config", {})
    tokenizer = load_tokenizer(
        Path(tokenizer_path),
        tokenizer_type=tokenizer_type,
        special_tokens=tokenizer_cfg.get("special_tokens"),
    )

    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    if model_version == MODEL_VERSION_V1:
        out_ids, lookahead = model.generate_filtered(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
            lookahead_steps=args.lookahead_steps,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    else:
        out_ids, lookahead = model.generate_planned(
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
            lookahead_steps=args.lookahead_steps,
            force_state=args.force_state,
            freeze_planner=bool(args.freeze_planner),
            planner_temperature=args.planner_temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    text = tokenizer.decode(out_ids)
    print(text)
    if lookahead:
        print("\nPlanner lookahead (first-step belief snapshots):")
        for i, state in enumerate(lookahead[:10]):
            top_idx = max(range(len(state)), key=lambda j: state[j])
            print(f"  t+{i+1}: top_z={top_idx} prob={state[top_idx]:.3f}")


if __name__ == "__main__":
    main()

