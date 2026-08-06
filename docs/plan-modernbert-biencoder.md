# Plan: ModernBERT long-context bi-encoder judge

## Why (settled from the investigation)

- **H1 confirmed** the score skew is present in the untuned base monoT5 — not a
  training artifact — so the fix isn't in the loss or negatives, it's the input.
- **H4 measured** on `cd2026_001`: query alone is 333 tokens, **97.7% of pairs
  exceed the 512-token cap**, and only **~26% of the average document is
  retained**. Truncation is the concrete defect.
- **Fix:** a long-context **bi-encoder** (encodes query and doc separately, so
  each gets its own length budget — no query+doc contention). The bi-encoder
  archetype is already validated in the paper (all-MiniLM-L6-v2).
- **Checkpoint:** `joe32140/ModernBERT-base-msmarco`, chosen on the prior that a
  ranking-tuned embedding beats a raw MLM. (`large` doesn't fit the V100s.)
- **Not measurable on CDet now** (no labels); verification deferred to the human
  assessments gathered from the prioritized pools.

## Scope

A minimal, additive change: make `BiEncoderLoRA` able to load a
sentence-transformers checkpoint with its trained pooling, and let query/doc
lengths be set independently. No change to the mono/cross paths, the data
schema, the CLI surface (beyond one optional flag), or defaults for existing
runs.

## Changes

### 1. `models.py` — `BiEncoderLoRA` sentence-transformers loading

- Add a constructor flag `sentence_transformer: bool = False`.
  - **False (default):** current behavior exactly — `AutoModel` + mean-pool +
    fresh 128-dim projection. Preserves the paper's all-MiniLM recipe and every
    existing bi-encoder run.
  - **True:** load via `sentence_transformers.SentenceTransformer(model_name)`,
    use its **native pooling + normalization**, and **drop the fresh
    projection** (score cosine on the model's own embedding). This is what makes
    the MS-MARCO ranking-tuning actually count.
- `encode_text` branches on the flag: ST path runs the ST module stack (transformer
  → pooling → normalize) and returns the pooled embedding directly; the AutoModel
  path is unchanged.
- LoRA still applies to the underlying transformer in both paths (`get_peft_model`
  on the base transformer module).
- `forward` (cosine → `(sim+1)/2`), loss, metrics, predict_step: **unchanged**.

### 2. `data.py` — independent query/doc max lengths + explicit front-truncation

- Add `max_length_doc` (optional) to `RelevanceDataModule`; `max_length` stays as
  the query/default cap. When `max_length_doc` is None, fall back to `max_length`
  (current behavior).
- In the bi-encoder branch of `collate_fn`, tokenize docs with `max_length_doc`
  and queries with `max_length`.
- Keep the tokenizer's default **tail truncation (keep the front)** and add a
  comment: news articles are editorially front-loaded, so keeping the lead is
  the right call — do not change to keep-tail.
- Cross/mono branch untouched.

### 3. `models.build_model` + `train.py`/`predict.py` — thread the knobs

- `build_model(...)` gains `sentence_transformer=False` and passes it to
  `BiEncoderLoRA` (ignored by cross/mono).
- `train()` and `predict()` gain `sentence_transformer=False` and
  `max_length_doc=None`, forwarded to `build_model` / `RelevanceDataModule`.

### 4. `cli.py` — expose the new options

- `--sentence-transformer` (store_true) on both `train` and `predict`.
- `--max-length-doc` (int, default None) on both.
- No change to existing flags/defaults.

### 5. Docs

- `score-skew-hypotheses.md`: record the **H4 result** (26% retention @512,
  97.7% over cap) and that it motivated the long-context bi-encoder move.
- `issues.md`: check off H4; update the ModernBERT entry to "in progress" with
  the settled decisions (checkpoint, native ST pooling, max_length ~2048,
  front-truncation, V100/FA2 caveat, verification deferred to gathered
  assessments).
- Add a short note (in `issues.md` or a comment) on the **FlashAttention-2 vs
  Volta** caveat: V100 is sm_70, FA2 needs sm_80+, so ModernBERT falls back to
  SDPA/eager and attention memory is ~O(L²) — hence modest `max_length` and
  small batch + gradient checkpointing.

## Suggested run configuration (not code — for the GPU box)

```sh
topic-judges train \
  --encoder-type bi --sentence-transformer --lora \
  --model-name joe32140/ModernBERT-base-msmarco \
  --train-data data/external/train-cdet-hard-5-95.parquet \
  --query-id cd2026_001 \
  --max-length 384 --max-length-doc 2048 \
  --batch-size 8 --accumulate-batches 4 \
  --output-dir ./outputs --device cuda
```

- `--max-length 384` fits the ~333-token narrative query; `--max-length-doc
  2048` covers ~p90 of docs whole and keeps the lead for the long tail.
- Start batch 8 + grad-accum; tune against `nvidia-smi`. Push doc length toward
  4096 only if memory allows.
- Gradient checkpointing: verify it's enabled for ModernBERT (memory lever for
  L≥2048 under O(L²) attention). May need a small follow-up if not on by default.

## Explicitly out of scope

- Passage-scoring / chunk-and-aggregate for mono (the bi-encoder makes it
  unnecessary).
- Any change to the mono/cross paths, loss, negative sampling, or data schema.
- Measuring bi-encoder quality on CDet (no labels) — deferred to human
  assessments.

## Risks / things to confirm at build time

- **ST + PEFT interplay:** applying `get_peft_model` to the transformer inside a
  `SentenceTransformer` module — confirm the LoRA layers train and save/load via
  the existing `load_adapter` path. If saving the ST wrapper is awkward, we save
  the adapter on the underlying transformer (matches current `model.model`
  convention).
- **Gradient checkpointing** availability on the ModernBERT ST module.
- **`predict.py` alignment** is unaffected (still bi-encoder forward), but worth
  a smoke check that ST-mode predict writes the same jsonl schema.
