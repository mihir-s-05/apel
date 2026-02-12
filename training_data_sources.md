# Online Training Data Sources for APEL-R (Few-M to Tens-of-M Params)

These are practical sources for training and iterating on small-to-mid language models.

## Recommended progression
1. Start on `TinyStories` for fast architecture debugging and clean generations.
2. Move to `WikiText-2` for a lightweight mixed-domain benchmark, then `WikiText-103` for broader vocabulary and longer-context structure.
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

3. WikiText-2 (raw)
   - URL: https://huggingface.co/datasets/Salesforce/wikitext
   - Config: `wikitext-2-raw-v1`
   - Why: faster iteration than WikiText-103 while preserving benchmark-style structure.
   - Suggested use: architecture/optimizer sweeps where you want shorter turnaround.

4. FineWeb-Edu
   - URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
   - Example config: `sample-10BT`
   - Why: large modern web corpus with educational filtering; good scaling path.
   - Suggested use: stream first N examples (50k-500k+) for controlled experiments.

5. C4 English
   - URL: https://huggingface.co/datasets/allenai/c4
   - Config: `en`
   - Why: large diverse web corpus used in many LM baselines.
   - Suggested use: streaming subset for small models; full-scale only with significant compute.

6. FineWeb (general)
   - URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb
   - Example config: `sample-10BT`
   - Why: broad web corpus for scaling experiments beyond educational filtering.
   - Suggested use: stream subset first to control token budget.

7. Wiki40B (English)
   - URL: https://huggingface.co/datasets/google/wiki40b
   - Config: `en`
   - Why: larger encyclopedic corpus than WikiText with structured article style.
   - Suggested use: streamable mid-scale generalization run.

8. CC-News
   - URL: https://huggingface.co/datasets/vblagoje/cc_news
   - Why: news-domain diversity and medium-size practical download.
   - Suggested use: domain-mix run after TinyStories/WikiText baselines.

## Loader compatibility note
1. Prefer non-script dataset loaders with current `datasets` versions; some older script-based datasets can fail to load in modern environments.

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
