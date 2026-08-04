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

These are the original scripts the package refactors. They remain the reference
implementation until the `topic-judges` package is verified against them, and
they also cover LLM-as-judge inference and evaluation, which are not (yet) part
of the package.

### Data Processing

- **`data_robust04_train.py`**: Processes the TREC Robust 2004 dataset for training. Downloads and formats the dataset from ir_datasets, combining queries, documents, and relevance judgments into a unified training format.
- **`data_trecdl_train.py`**: Processes the TREC Deep Learning (2019 and 2020) datasets for training. Merges data from both years and formats it for model training.
- **`data_llm_inference.py`**: Performs large language model inference for relevance scoring. Uses any OpenAI-API-compliant endpoint to generate relevance scores for query-document pairs, with different prompts.

### Training

- **`train.py`**: Main training script for topic-adapted relevance models. Implements PyTorch Lightning training pipeline with LoRA (Low-Rank Adaptation) for fine-tuning transformer models on relevance prediction tasks.

```sh 
usage: train.py [-h] [--encoder_type {cross,bi,mono}] --data_path DATA_PATH --query_id QUERY_ID [--lora LORA] [--model_name MODEL_NAME] [--batch_size BATCH_SIZE] [--accumulate_batches ACCUMULATE_BATCHES]
                [--max_epochs MAX_EPOCHS] [--learning_rate LEARNING_RATE] [--lora_r LORA_R] [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT] [--max_length MAX_LENGTH] [--output_dir OUTPUT_DIR] [--binary BINARY]
                [--predict PREDICT] [--split SPLIT] [--seed SEED]

Train topic-specific relevance judge models with LoRA

options:
  -h, --help            show this help message and exit
  --encoder_type {cross,bi,mono}  Type of encoder: cross-encoder, bi-encoder, or mono-decoder
  --data_path DATA_PATH           Path to load training data from; has to be formatted as produced by data_* scripts in this directory
  --query_id QUERY_ID             The topic ID to train a judge for
  --lora LORA                     Whether to apply lora or not
  --model_name MODEL_NAME         Pretrained base ranker model to load
  --batch_size BATCH_SIZE         Batch size to train with
  --accumulate_batches ACCUMULATE_BATCHES Gradient accumulation batches
  --max_epochs MAX_EPOCHS         Maximum epochs to train for
  --learning_rate LEARNING_RATE   Learning rate
  --lora_r LORA_R                 r parameter for LoRA
  --lora_alpha LORA_ALPHA         a parameter for LoRA
  --lora_dropout LORA_DROPOUT     Dropout rate for LoRA
  --max_length MAX_LENGTH         Maximum sequence length of input, concatenated query/doc in cross/mono models
  --output_dir OUTPUT_DIR         Directory to save model and prediction data to
  --binary BINARY                 Whether to cast relevance scores to binary labels
  --predict PREDICT               Whether to predict on test data and write to file
  --split SPLIT                   Size of train split per topic
  --seed SEED                     Random seed
```


### Evaluation

- **`eval_robust04_shallow.py`**: Evaluates shallow pooling strategies on the TREC Robust 2004 collection. Simulates reduced evaluation pools and measures the impact on retrieval evaluation metrics across different pool sizes and topic subsets.
- **`eval_trecdl_shallow.py`**: Evaluates shallow pooling strategies on the TREC Deep Learning collections. Similar to the Robust 2004 evaluation but adapted for the Deep Learning track datasets and topics.
