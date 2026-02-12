# APEL-R PyTorch Prototype

APEL-R is a language-model prototype built around a simple idea: while the model is producing text, it also keeps a running belief about a higher-level "plan" for the current chunk of text and can project that plan forward.

This repo gives you a runnable end-to-end implementation for training and sampling.

## Architecture in Plain Language

APEL-R has two cooperating parts.

1. Executor:
   It is the token generator. It reads previous tokens and predicts the next token.

2. Planner:
   It keeps a probability distribution over a small set of latent plan states (discrete states like hidden "intent modes") and updates that belief as tokens are observed.

Generation is chunked.

1. A chunk is a small block of tokens (`chunk_size` in config).
2. Inside a chunk, the model maintains a belief over the current latent plan state.
3. At chunk boundaries, planner transition dynamics move belief into the next chunk's prior.

What "exact filtering" means here:

1. For each token, the model computes a mixture over plan states.
2. After the token is chosen/observed, it performs a Bayes-style posterior update over plan states.
3. This keeps planner belief mathematically consistent with generated text for the discrete-latent setup implemented here.

What "asynchronous lookahead" means here:

1. While speaking, the model can repeatedly apply planner transition dynamics to predict likely future plan-state beliefs.
2. These lookahead beliefs are available for introspection/debugging and future extensions.

There are now two versioned architectures:

1. `v1_filtered_mixture`:
   Original filtered-mixture planner/executor model.
2. `v2_planner_required`:
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
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -e .
```

CPU-only install:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

## Quick Start

Train:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.train --config configs\tinystories_smoke.yaml
```

Sample:

```powershell
uv run --python .venv\Scripts\python.exe python -m apelr.sample --checkpoint runs\tinystories_smoke\checkpoint.pt --prompt "Once upon a time"
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
5. `configs/tinystories_bpe_smoke.yaml`:
   BPE tokenizer run with identifiability loss enabled.
6. `configs/tinystories_bpe_ident_iter.yaml`:
   Longer BPE + plan-identifiability iteration.
7. `configs/tinystories_bpe_ident_cpk_scale.yaml`:
   Larger BPE run adding chunk-posterior KL alignment for stronger plan identifiability.
8. `configs/wikitext2_bpe_ident_cpk.yaml`:
   Additional-source run for cross-domain validation with the same identifiability stack.
9. `configs/cc_news_bpe_ident_cpk.yaml`:
   News-domain run to test whether planner states remain identifiable off story-style corpora.
10. `configs/tinystories_bpe_ident_resume_sharp.yaml`:
   Continuation run from a saved checkpoint with sharper identifiability regularization.
11. `configs/wikitext2_bpe_ident_cpk_smoke.yaml`:
   Short cross-domain smoke run to validate WikiText-2 training dynamics.
12. `configs/v2_tinystories_cuda_fast.yaml`:
   V2 planner-required CUDA fast-iteration config.
13. `configs/v2_wikitext2_cuda_fast.yaml`:
   V2 cross-domain CUDA fast-iteration config.

Important knobs:

1. `model.num_plan_states`: number of latent planner states.
2. `model.chunk_size`: planner update cadence across token stream.
3. `model.planner_context_scale`: boundary context influence on planner belief.
4. `model.planner_self_bias`: initial preference for staying in same plan state.
5. `tokenizer.type`: `char` or `bpe`.
6. `tokenizer.vocab_size`: BPE vocabulary size when `tokenizer.type: bpe`.
7. `train.entropy_reg_weight`: regularization on belief entropy.
8. `train.usage_balance_weight`: regularization for plan-state usage balance.
9. `train.chunk_bow_weight`: chunk-level plan-predictive bag-of-words loss for stronger plan identifiability.
10. `train.chunk_bow_warmup_steps`: optional warmup for `chunk_bow_weight` to avoid overwhelming NLL early.
11. `train.plan_mi_weight`: weight for mutual-information surrogate that increases plan-state emission separability.
12. `train.chunk_post_kl_weight`: chunk-posterior alignment weight (recognizer posterior vs filtered posterior).
13. `train.save_interval`: optional periodic checkpoint save frequency (in steps).
14. `train.resume_from`: optional checkpoint path to resume optimizer+model+global_step.
15. `model.version`: architecture version (`v1_filtered_mixture` or `v2_planner_required`).
16. `train.precision`: `fp32`, `fp16`, or `bf16` (AMP used automatically on CUDA).
17. `train.grad_accum_steps`: gradient accumulation factor for VRAM-constrained training.
18. `train.loss_weights.*` (V2): planner-required objectives (`future_contrastive`, `plan_js_div`, `boundary_entropy`, `usage_balance`).

## Sampling Options

Common flags for `apelr.sample`:

1. `--temperature`: randomness.
2. `--top-k`: truncation for token sampling.
3. `--lookahead-steps`: prints first-step planner lookahead snapshots.
4. `--max-new-tokens`: output length cap.
5. `--force-state`: force a planner state at inference (V2 only).
6. `--freeze-planner`: disable planner updates during generation (V2 only).
7. `--planner-temperature`: planner distribution sharpness during V2 generation.

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
