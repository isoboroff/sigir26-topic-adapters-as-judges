#!/usr/bin/env bash

# Run one query through `topic-judges predict`. Paths are read from the
# environment (with defaults below) so this stays reusable under GNU parallel
# without per-run edits:
#
#   parallel -j 8 --line-buffer --joblog predict_jobs.log --resume --progress \
#     ./predict-one.sh {} {%} :::: query_ids.txt
#
# Override any of these by exporting them before invoking parallel, e.g.:
#   export PREDICT_DATA=../data/external/test-cdet.parquet
#   export BASE_MODEL=castorini/monot5-base-msmarco-10k

set -euo pipefail
QUERY_ID="$1"
SLOT="$2"

export CUDA_VISIBLE_DEVICES=$(( SLOT - 1 ))

ENCODER_TYPE="${ENCODER_TYPE:-mono}"
BASE_MODEL="${BASE_MODEL:-castorini/monot5-base-msmarco-10k}"
ADAPTER_DIR="${ADAPTER_DIR:-./outputs}"
ADAPTER_PATH="${ADAPTER_PATH:-${ADAPTER_DIR}/mono-monot5-base-msmarco-10k-lora-${QUERY_ID}}"
PREDICT_DATA="${PREDICT_DATA:-../data/external/test-cdet.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./predictions}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

topic-judges predict \
  --encoder-type "$ENCODER_TYPE" \
  --base-model-name "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --predict-data "$PREDICT_DATA" \
  --query-id "$QUERY_ID" \
  --output "${OUTPUT_DIR}/${QUERY_ID}.jsonl" \
  --batch-size "$BATCH_SIZE" \
  --lora \
  --device "$DEVICE"
