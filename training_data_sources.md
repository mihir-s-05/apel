# Online Training Data Sources for APEL-R (Few-M to Tens-of-M Params)

These are practical sources for training and iterating on small-to-mid language models.

## Recommended progression
1. Start on `TinyStories` for fast architecture debugging and clean generations.
2. Move to `WikiText-103` for broader vocabulary and longer-context structure.
3. Scale with streaming subsets from `FineWeb-Edu` or `C4` for robustness.

## Source list
1. TinyStories
   - URL: https://huggingface.co/datasets/roneneldan/TinyStories
   - Why: high-quality short narratives, easy to overfit/diagnose quickly.
   - Suggested use: 2k-100k examples depending on iteration speed.

2. WikiText-103 (raw)
   - URL: https://huggingface.co/datasets/Salesforce/wikitext
   - Config: `wikitext-103-raw-v1`
   - Why: classic language-model benchmark with richer factual language.
   - Suggested use: full train split for models >=5M params, or sampled subset for fast cycles.

3. FineWeb-Edu
   - URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
   - Example config: `sample-10BT`
   - Why: large modern web corpus with educational filtering; good scaling path.
   - Suggested use: stream first N examples (50k-500k+) for controlled experiments.

4. C4 English
   - URL: https://huggingface.co/datasets/allenai/c4
   - Config: `en`
   - Why: large diverse web corpus used in many LM baselines.
   - Suggested use: streaming subset for small models; full-scale only with significant compute.

## Practical token-budget guidance
1. 3M-8M params:
   - Target 20M-150M tokens for meaningful quality.
   - Use TinyStories + WikiText first, then add streamed FineWeb/C4.

2. 8M-30M params:
   - Target 100M-800M tokens (compute permitting).
   - Mix curated and broad web text; monitor deduplication and quality filtering.

## Data quality notes
1. Keep explicit train/val split to detect overfitting.
2. Deduplicate near-identical samples when scaling beyond TinyStories.
3. Track source mix ratios in logs for reproducibility.
4. Verify dataset terms/licenses directly on each dataset card before production use.

