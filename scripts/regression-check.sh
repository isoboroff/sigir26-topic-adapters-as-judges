#!/usr/bin/env bash
#
# Regression check: confirm the refactored `topic-judges predict` produces the
# same predictions as the original scripts/predict.py for one query, before the
# old scripts are deleted (design sec 7, step 7 of the refactor).
#
# Run this on the GPU system, from the scripts/ directory, against an adapter
# you have already trained for QUERY_ID. It runs both implementations and diffs
# the resulting jsonl.
#
# Usage:
#   cd scripts
#   ./regression-check.sh <QUERY_ID> [ADAPTER_PATH]
#
# Env overrides (defaults match predict-one.sh):
#   ENCODER_TYPE   default: mono
#   BASE_MODEL     default: castorini/monot5-base-msmarco-10k
#   PREDICT_DATA   default: ../data/external/test-cdet.parquet
#   BATCH_SIZE     default: 256
#   DEVICE         default: cuda

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <QUERY_ID> [ADAPTER_PATH]" >&2
  exit 2
fi

QUERY_ID="$1"

ENCODER_TYPE="${ENCODER_TYPE:-mono}"
BASE_MODEL="${BASE_MODEL:-castorini/monot5-base-msmarco-10k}"
ADAPTER_PATH="${2:-${ADAPTER_PATH:-./outputs/mono-monot5-base-msmarco-10k-lora-${QUERY_ID}}}"
PREDICT_DATA="${PREDICT_DATA:-../data/external/test-cdet.parquet}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -e "$ADAPTER_PATH" ]]; then
  echo "adapter not found: $ADAPTER_PATH" >&2
  echo "train one first (or pass its path), e.g.:" >&2
  echo "  topic-judges train --encoder-type $ENCODER_TYPE --lora \\" >&2
  echo "    --model-name $BASE_MODEL --train-data ../data/external/train-cdet.parquet \\" >&2
  echo "    --query-id $QUERY_ID --output-dir ./outputs" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OLD_OUT="$WORK/old.jsonl"
NEW_OUT="$WORK/new.jsonl"

echo "== Query:   $QUERY_ID"
echo "== Adapter: $ADAPTER_PATH"
echo

echo "== [1/3] Original scripts/predict.py ..."
python predict.py \
  --encoder_type "$ENCODER_TYPE" \
  --base_model_name "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --predict_data_path "$PREDICT_DATA" \
  --query_id "$QUERY_ID" \
  --output_path "$OLD_OUT" \
  --batch_size "$BATCH_SIZE" \
  --lora True \
  --device "$DEVICE"

echo "== [2/3] Refactored topic-judges predict ..."
topic-judges predict \
  --encoder-type "$ENCODER_TYPE" \
  --base-model-name "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --predict-data "$PREDICT_DATA" \
  --query-id "$QUERY_ID" \
  --output "$NEW_OUT" \
  --batch-size "$BATCH_SIZE" \
  --lora \
  --device "$DEVICE"

echo "== [3/3] Comparing outputs ..."
# Compare as (query_id, doc_id, relevance) records. Both writers emit rows in
# predict-data order for the query, so a plain diff should suffice; we also do a
# tolerant numeric compare in case of float-formatting differences.
python - "$OLD_OUT" "$NEW_OUT" <<'PY'
import json, sys

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

old, new = load(sys.argv[1]), load(sys.argv[2])

if len(old) != len(new):
    print(f"MISMATCH: row counts differ (old={len(old)}, new={len(new)})")
    sys.exit(1)

# Order-sensitive id check (both should be in predict-data order).
id_mismatch = [
    i for i, (o, n) in enumerate(zip(old, new))
    if (o["query_id"], o["doc_id"]) != (n["query_id"], n["doc_id"])
]
if id_mismatch:
    print(f"MISMATCH: {len(id_mismatch)} rows have differing (query_id, doc_id) ordering")
    i = id_mismatch[0]
    print(f"  first at row {i}: old={old[i]['query_id'],old[i]['doc_id']} new={new[i]['query_id'],new[i]['doc_id']}")
    sys.exit(1)

TOL = 1e-6
diffs = [
    (i, o["relevance"], n["relevance"])
    for i, (o, n) in enumerate(zip(old, new))
    if abs(float(o["relevance"]) - float(n["relevance"])) > TOL
]
if diffs:
    print(f"MISMATCH: {len(diffs)}/{len(old)} relevance scores differ by > {TOL}")
    for i, ov, nv in diffs[:5]:
        print(f"  row {i}: old={ov} new={nv} (|d|={abs(float(ov)-float(nv)):.3e})")
    sys.exit(1)

print(f"OK: {len(old)} rows identical (ids match, scores within {TOL})")
PY

echo "== Regression check passed."
