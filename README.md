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

## Repository Layout

1. `src/apelr/model.py`: APEL-R model (planner + executor, filtering, generation).
2. `src/apelr/train.py`: training and evaluation loop.
3. `src/apelr/sample.py`: checkpoint-based text generation CLI.
4. `src/apelr/data.py`: dataset loading and sequence packing.
5. `src/apelr/tokenizer.py`: lightweight char tokenizer used in current experiments.
6. `configs/*.yaml`: ready-to-run experiment configs.
7. `runs/*`: checkpoints, tokenizer files, and metrics.

## Installation

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

Important knobs:

1. `model.num_plan_states`: number of latent planner states.
2. `model.chunk_size`: planner update cadence across token stream.
3. `model.planner_context_scale`: boundary context influence on planner belief.
4. `model.planner_self_bias`: initial preference for staying in same plan state.
5. `train.entropy_reg_weight`: regularization on belief entropy.
6. `train.usage_balance_weight`: regularization for plan-state usage balance.

## Sampling Options

Common flags for `apelr.sample`:

1. `--temperature`: randomness.
2. `--top-k`: truncation for token sampling.
3. `--lookahead-steps`: prints first-step planner lookahead snapshots.
4. `--max-new-tokens`: output length cap.

## Data

Supported dataset keys in configs:

1. `tinystories`
2. `wikitext103`
3. `fineweb_edu`
4. `c4_en`

See `training_data_sources.md` for practical source recommendations and scaling notes.

## Outputs

Each run writes:

1. `checkpoint.pt`
2. `tokenizer.json`
3. `metrics.json`

under the configured `train.out_dir`.

## Notes and Caveats

1. Current tokenizer is character-level for fast architectural iteration, so generated text quality is lower than a subword-tokenized model at similar scale.
2. Loading older checkpoints remains supported in sampling (`strict=False`), but you may see warnings about missing/new keys.
3. You may see a non-fatal `multiprocess.resource_tracker` warning at shutdown from `datasets`; checkpoints and metrics are still typically written correctly.
