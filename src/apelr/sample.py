from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import APELRModel, APELRModelConfig
from .tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from trained APEL-R model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--lookahead-steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_cfg = APELRModelConfig(**ckpt["model_config"])
    model = APELRModel(model_cfg)
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
    tokenizer = CharTokenizer.load(Path(tokenizer_path))

    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    out_ids, lookahead = model.generate_filtered(
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
        lookahead_steps=args.lookahead_steps,
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
