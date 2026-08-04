# Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs

This repository contains the code and data for the paper "Topic-Specific Classifiers are Better
Relevance Judges than Prompted LLMs".

## Repository Structure

```
├── data/
│   └── processed/           # Data; relevance judgments for different test collections predicted by different models
├── src/
│   └── topic_judges/        # Importable package + `topic-judges` CLI (train/predict/data)
├── scripts/                 # Original standalone scripts + LLM inference / evaluation
└── README.md                # This file
```

## Package: the `topic-judges` CLI

The reusable code lives in `src/topic_judges/` and installs a single console
script, `topic-judges`, with three subcommands. See
[`docs/refactor-design.md`](docs/refactor-design.md) for the design.

> **Note:** this package is a refactor of the standalone `scripts/` (documented
> below). Until it has been verified against those scripts on the GPU system,
> the scripts remain the reference implementation.

```sh
# Build datasets (all emit the shared query_id/doc_id/query/doc/relevance schema)
topic-judges data cdet-train    --topics ... --coll ... --output data/external/train-cdet.parquet
topic-judges data cdet-predict  --topics ... --pool ... --coll ... --output data/external/test-cdet.parquet
topic-judges data robust04      --output data/external/train-robust04.parquet
topic-judges data trecdl        --output data/external/train-trecdl.parquet

# Train a per-query judge adapter (optionally with a held-out validation set)
topic-judges train \
  --encoder-type mono --lora \
  --model-name castorini/monot5-base-msmarco-10k \
  --train-data data/external/train-cdet.parquet \
  --val-data   data/external/val-cdet.parquet \
  --query-id cd2026_001 \
  --output-dir ./outputs

# Predict relevance for one query with a trained adapter
topic-judges predict \
  --encoder-type mono --lora \
  --base-model-name castorini/monot5-base-msmarco-10k \
  --adapter-path ./outputs/mono-monot5-base-msmarco-10k-lora-cd2026_001 \
  --predict-data data/external/test-cdet.parquet \
  --query-id cd2026_001 \
  --output ./predictions/cd2026_001.jsonl
```

Dataset builders come in two kinds: **judged** (Robust04, TREC-DL — positive and
negative labels come from qrels) and **positive-only + sampling** (CDet — only
positive docs are given, negatives are sampled at `--neg-pos-ratio`, default 19).

## Data

### Processed Data (`data/processed/`)

The `data/processed/` directory contains processed relevance judgments from a variety of models:

- **`trecdl-llm-qrels.parquet`**: TREC Deep Learning track relevance judgments produced by GPT-4o; includes different prompting strategies, indicated in the `prompt` columns
- **`trecdl-shallow-qrels-{64,128,192,256}.parquet`**: TREC Deep Learning shallow pooling relevance judgments from topic-specific adapters using different train set sizes (64, 128, 192, 256 docs)
- **`robust04-shallow-qrels-{64,128,192,256}.parquet`**: TREC Robust 2004 shallow pooling relevance judgments from topic-specific adapters using different train set sizes (64, 128, 192, 256 docs)

These files contain query-document pairs with relevance labels, to be used in TREC-style evaluation.

## Standalone scripts

Dataset building, training, and prediction now live in the `topic-judges`
package (above). The remaining standalone scripts cover LLM-as-judge inference
and evaluation, which are not (yet) part of the package.

### LLM inference

- **`data_llm_inference.py`**: Performs large language model inference for relevance scoring. Uses any OpenAI-API-compliant endpoint to generate relevance scores for query-document pairs, with different prompts.

### Evaluation

- **`eval_robust04_shallow.py`**: Evaluates shallow pooling strategies on the TREC Robust 2004 collection. Simulates reduced evaluation pools and measures the impact on retrieval evaluation metrics across different pool sizes and topic subsets.
- **`eval_trecdl_shallow.py`**: Evaluates shallow pooling strategies on the TREC Deep Learning collections. Similar to the Robust 2004 evaluation but adapted for the Deep Learning track datasets and topics.

### Regression check

- **`regression-check.sh`**: Confirms `topic-judges predict` reproduces the original prediction script's output for one query. Retained for future refactors.

