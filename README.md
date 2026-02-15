# APEL-R PyTorch Prototype

APEL-R is a language-model prototype built around a simple idea: while the model is producing text, it also maintains an internal planner state (a small discrete distribution) and can project that state forward.

This repo gives you a runnable end-to-end implementation for training and sampling.

## Architecture in Plain Language

APEL-R has two cooperating parts.

1. Executor:
   It is the token generator. It reads previous tokens and predicts the next token.

2. Planner:
   It keeps a probability distribution over a small set of internal plan states (discrete modes like "intent/style") and updates that state online from the tokens it has actually seen/generated.

Generation is chunked.

1. A chunk is a small block of tokens (`chunk_size` in config).
2. Inside a chunk, the model maintains a belief over the current latent plan state.
3. At chunk boundaries, planner transition dynamics move belief into the next chunk's prior.

What "token filtering" means here (state-space policy LM):

1. For each token, the model computes a mixture over plan states.
2. After the next token is chosen/observed, it updates the planner state using the experts' token log-likelihoods (a filtering-like online update).
3. This is a causal internal state update that keeps the planner state coupled to the realized token sequence; it is not a claim of recovering an exact posterior over a "true" latent variable.

What "asynchronous lookahead" means here:

1. In `v2_planner_required`, while speaking, the model can repeatedly apply planner transition dynamics to predict likely future plan-state beliefs.
2. In V2, those lookahead beliefs can be fed back into the emission gate (configurable blend) so planner forecasts influence current token choices.
3. V2 generation can run a background planner worker so lookahead forecasting and token sampling overlap in wall-clock time.

There are two versioned architectures:

1. `v1_filtered_mixture`:
   Original filtered-mixture planner/executor model.
2. `v2_planner_required` (default for new training runs):
   Planner-required expert-head model where token logits are planner-gated expert mixtures (reducing planner bypass).

See `docs/model_versions.md` for compatibility and version history.

## Repository Layout

1. `src/apelr/model.py`: APEL-R model (planner + executor, filtering, generation).
2. `src/apelr/train.py`: training and evaluation loop.
3. `src/apelr/sample.py`: checkpoint-based text generation CLI.
4. `src/apelr/data.py`: dataset loading and sequence packing.
5. `src/apelr/tokenizer.py`: tokenizer abstraction with char and BPE implementations.
6. `configs/*.yaml`: ready-to-run experiment configs.
7. `runs/*`: checkpoints, tokenizer files, and metrics.
8. `architecture_math_log.txt`: running architecture-change and detailed math log.
9. `docs/model_versions.md`: model versioning and checkpoint compatibility policy.
10. `docs/experiments_v2.md`: CUDA fast-iteration experiment guide for V2.

## Installation

CUDA install (recommended for RTX 4070):

```powershell
uv venv .venv
uv sync
```

For A100/H100, use the CUDA build and set `train.precision: bf16` in your config.

Note: this repo pins a CUDA-enabled PyTorch wheel (`torch==2.6.0+cu124`). For CPU-only installs,
change the `torch` dependency in `pyproject.toml` and re-lock (`uv lock`) before syncing.

## Quick Start

Train:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.train --config configs\tinystories_smoke.yaml
```

Sample:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.sample --checkpoint runs\tinystories_smoke\checkpoint.pt --prompt "Once upon a time"
```

## Tests

```powershell
uv run --python .venv\Scripts\python.exe python -m unittest discover -s tests -p "test_*.py"
```

## Standard Benchmarks

This repo includes a lightweight, dependency-free (beyond `datasets`) subset of the EleutherAI LM Evaluation Harness-style benchmarks, aligned with the benchmark set used in the Titans paper (WikiText + common multiple-choice datasets, including PIQA and SocialIQA).

Run evals for one or more checkpoints:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.eval_standard `
  --checkpoints runs\wikitext2_v2_cuda_4070m_bpe2048_smoke\checkpoint.pt `
  --tasks titans --max-examples 512 --max-input-len 96
```

Compare APEL-R to a decoder-only Transformer baseline:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.eval_standard `
  --checkpoints `
    runs\wikitext2_v2_cuda_4070m_bpe2048_smoke\checkpoint.pt `
    runs\wikitext2_transformer_cuda_4070m_bpe2048_smoke\checkpoint.pt `
  --tasks wikitext2,lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande_xl,openbookqa,boolq `
  --max-examples 256 --max-input-len 96 --output runs\eval_compare.json
```

## Config Guide

Useful configs:

1. `configs/tinystories_smoke.yaml`:
   Fast smoke test, good for verifying setup and code paths.
2. `configs/tinystories_iter.yaml`:
   Longer run with better quality.
3. `configs/tinystories_selfbias_smoke.yaml`:
   Uses stronger planner self-transition bias initialization.
4. `configs/tinystories_small.yaml`:
   Intermediate training run.

Add one-off experiment YAMLs under `configs/local/` (ignored by git).

Important knobs:

1. `model.num_plan_states`: number of latent planner states.
2. `model.chunk_size`: planner update cadence across token stream.
3. `model.planner_context_scale`: boundary context influence on planner belief.
4. `model.planner_self_bias`: initial preference for staying in same plan state.
5. `tokenizer.type`: `char` or `bpe`.
6. `tokenizer.vocab_size`: BPE vocabulary size when `tokenizer.type: bpe`.
7. `train.entropy_reg_weight`: (V1) regularization on belief entropy; (V2) fallback for `train.loss_weights.boundary_entropy` if unset.
8. `train.usage_balance_weight`: (V1) regularization for plan-state usage balance; (V2) fallback for `train.loss_weights.usage_balance` if unset.
9. `train.chunk_bow_weight`: chunk-level plan-predictive bag-of-words loss for stronger plan identifiability.
10. `train.chunk_bow_warmup_steps`: optional warmup for `chunk_bow_weight` to avoid overwhelming NLL early.
11. `train.plan_mi_weight`: weight for mutual-information surrogate that increases plan-state emission separability.
12. `train.chunk_post_kl_weight`: chunk-level state alignment weight (recognizer distribution vs online-filtered distribution).
13. `train.save_interval`: optional periodic checkpoint save frequency (in steps).
14. `train.resume_from`: optional checkpoint path to resume optimizer+model+global_step.
15. `model.version`: architecture version (`v1_filtered_mixture` or `v2_planner_required`, default is V2 when unset).
16. `train.precision`: `fp32`, `fp16`, or `bf16` (AMP used automatically on CUDA).
17. `train.grad_accum_steps`: gradient accumulation factor for VRAM-constrained training.
18. `train.loss_weights.*` (V2): planner-required objectives (`future_contrastive`, `plan_js_div`, `boundary_entropy`, `usage_balance`).
19. `data.token_cache`: if `true`, builds/uses on-disk tokenized corpora (`.bin`) instead of holding full text in RAM.
20. `model.lookahead_horizon` / `model.lookahead_feedback_scale` (V2): planner forecast horizon and planner-to-emission feedback strength.
21. `model.async_planner` (V2): enable asynchronous planner forecasting thread during generation.
22. `model.token_filtering` (V2): when `true`, planner beliefs are updated per token during generation; when `false`, planner updates happen at chunk boundaries only.
23. `data.token_cache_dir`: output directory for token cache files (default under `train.out_dir`).
24. `data.max_train_tokens` / `data.max_val_tokens`: hard token caps for cache construction.
25. `data.tokenizer_fit_max_examples` / `data.tokenizer_fit_max_chars`: bounded sample used to fit tokenizer in token-cache mode.
26. `data.reuse_token_cache`: reuse previously built `.bin` files instead of rebuilding.
27. `train.adaptive_batch.enabled`: enable VRAM-based auto batch probe.
28. `train.adaptive_batch.probe_batch_size`: micro-batch used for memory probing.
29. `train.adaptive_batch.reprobe_interval_steps`: periodically re-probe and update batch size mid-run.
30. `train.num_workers` / `train.prefetch_factor` / `train.persistent_workers`: DataLoader throughput knobs for better GPU feeding.
31. `train.fused_adamw`: enable fused AdamW kernel on CUDA for faster optimizer steps.
32. `train.allow_tf32`: allow TF32 matmul/cuDNN paths on Ampere/Hopper when available.
33. `train.matmul_precision`: sets PyTorch float32 matmul precision (`highest`, `high`, `medium`).
34. `train.compile.enabled`: toggles `torch.compile` for training speedups when supported.
35. `train.eval_planner_diagnostics`: include expensive V2 planner intervention diagnostics during periodic eval.
36. `train.final_eval_planner_diagnostics`: include planner diagnostics in final eval/metrics writeout.

## Large-Scale Training

For GPT-2-scale corpora, use token-cache mode so training reads pretokenized memmap files:

```yaml
data:
  source: fineweb_edu
  streaming: true
  token_cache: true
  token_cache_dir: runs/my_run/token_cache
  reuse_token_cache: true
  max_train_examples: 2000000
  max_val_examples: 50000
  max_train_tokens: 500000000
  max_val_tokens: 20000000
  tokenizer_fit_max_examples: 100000
  tokenizer_fit_max_chars: 20000000

train:
  precision: bf16
  batch_size: auto
  adaptive_batch:
    enabled: true
    probe_batch_size: 2
    min_batch_size: 1
    max_batch_size: 64
    safety_factor: 0.9
    reprobe_interval_steps: 500
```

Multi-GPU (DDP) is supported when you launch via `torchrun` (WORLD_SIZE > 1). Example (Linux):

```bash
torchrun --nproc_per_node=8 -m apelr.train --config configs/fineweb_edu_scale_pilot_65m.yaml
```

Optional distributed knobs:

```yaml
train:
  distributed:
    backend: nccl
    zero_optimizer: true  # shards AdamW optimizer states across ranks
  save_optimizer_state: false  # avoids huge checkpoints; enable for exact resume
```

## Sampling Options

Common flags for `apelr.sample`:

1. `--temperature`: randomness.
2. `--top-k`: truncation for token sampling.
3. `--lookahead-steps`: prints first-step planner lookahead snapshots.
4. `--max-new-tokens`: output length cap.
5. `--force-state`: force a planner state at inference (V2 only).
6. `--freeze-planner`: disable planner updates during generation (V2 only).
7. `--planner-temperature`: planner distribution sharpness during V2 generation.
8. `--lookahead-feedback-scale`: override planner-to-emission feedback strength during sampling.
9. `--no-async-planner`: disable background planner worker and run synchronous lookahead only.

## Data

Supported dataset keys in configs:

1. `tinystories`
2. `wikitext103`
3. `wikitext2`
4. `fineweb_edu`
5. `fineweb`
6. `c4_en`
7. `wiki40b_en`
8. `cc_news`

See `training_data_sources.md` for practical source recommendations and scaling notes.

## Outputs

Each run writes:

1. `checkpoint.pt`
2. `tokenizer.json`
3. `metrics.json`
4. `planner_eval.json` (V2 runs)

under the configured `train.out_dir`.

## Notes and Caveats

1. BPE runs generally produce cleaner lexical units than char-level runs at the same parameter budget, but still require longer training to reach strong coherence.
   Char tokenization remains useful for very fast smoke tests.
2. Loading older checkpoints remains supported in sampling (`strict=False`), but you may see warnings about missing/new keys.
   Legacy checkpoints without model version metadata default to V1 loading.
3. You may see a non-fatal `multiprocess.resource_tracker` warning at shutdown from `datasets`; checkpoints and metrics are still typically written correctly.
