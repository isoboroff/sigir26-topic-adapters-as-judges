# Refactor Design: Relevance-Judge Package

**Branch:** `refactor/relevance-judge-package`
**Author:** soboroff
**Date:** 2026-08-04

## 1. Goal

Turn the current pile of standalone scripts into a small, importable Python
package plus thin CLI entry points, so that training a topic-specific relevance
judge and generating predictions on **new** data (starting with
`data/external/test-cdet.parquet`) is a matter of pointing the tools at a
parquet file — not copying and editing a script.

The immediate deliverable that must keep working: **per-query relevance
predictions over `test-cdet.parquet`** (50 queries, ~840K query/doc pairs),
driven by a trained `mono` LoRA adapter per query.

## 2. Current state

### Scripts (all in `scripts/`, no package)

| Script | Role | Problem |
| --- | --- | --- |
| `train.py` | model classes + data module + train CLI | Core classes trapped in a CLI script |
| `train_cdet.py` | **near-duplicate** of `train.py` | Only `RelevanceDataModule.__init__` differs (split train/predict paths) |
| `predict.py` | prediction CLI | `import`s model/data classes **from `train_cdet.py`** — prediction depends on a training script |
| `data_cdet_train.py` | build train parquet from CD topics | Bespoke, but emits the shared schema |
| `data_cdet_test.py` | build predict parquet from a pool | Bespoke, but emits the shared schema |
| `data_robust04_train.py` | build train parquet from ir_datasets | Shared schema |
| `data_trecdl_train.py` | build train parquet from ir_datasets | Shared schema |
| `data_llm_inference.py` | LLM-as-judge baseline | Independent; leave as-is for now |
| `eval_*_shallow.py` | evaluation | Independent; leave as-is for now |
| `predict-one.sh` | one-query predict, for GNU parallel | Hardcoded paths/model; references `predict.py` by bare name |

### Duplication / coupling

- Three model classes (`BiEncoderLoRA`, `CrossEncoderLoRA`, `MonoDecoderLoRA`),
  `RelevanceDataset`, and `RelevanceDataModule` are defined **twice**
  (`train.py`, `train_cdet.py`) and imported by `predict.py`.
- `mono` → `cross` encoder-type remapping logic is repeated in `train_cdet.py`
  and `predict.py`.
- Prediction-ordering/id-alignment logic is repeated in `train_cdet.main` and
  `predict.py`.

### Shared data contract (already consistent — keep it)

Every `data_*` script writes a parquet with exactly:

```
query_id: str   doc_id: str   query: str   doc: str   relevance: int
```

`relevance = -1` marks "unlabeled / to be predicted" in the test file. This
schema is the seam the whole package should be built around.

### Packaging gaps

- `pyproject.toml` declares dependencies but **no package** (no build target,
  no `[project.scripts]`).
- `name = "ecir26-topic-adapters-as-judges"` but the directory (and paper) is
  `sigir26`.

## 3. Target layout

```
src/
  topic_judges/
    __init__.py
    data.py            # RelevanceDataset, RelevanceDataModule (the schema seam)
    models.py          # BiEncoderLoRA, CrossEncoderLoRA, MonoDecoderLoRA + build_model()
    train.py           # train() library fn (no argparse)
    predict.py         # predict() library fn: model + parquet + query_id -> jsonl
    io.py              # load/save parquet + jsonl helpers, id/prediction alignment
    cli.py             # argparse entry points: topic-judges train / predict
    datasets/          # dataset-specific builders, all emit the shared schema
      __init__.py
      common.py        # shared: rows -> parquet; sample_negatives() helper
      cdet.py          # positive-only + negative sampling (was data_cdet_{train,test})
      robust04.py      # judged (explicit pos/neg from qrels)
      trecdl.py        # judged (explicit pos/neg from qrels)
scripts/
  predict-one.sh       # updated: calls `topic-judges predict`, paths via env vars
  data_llm_inference.py  # unchanged for now
  eval_*_shallow.py      # unchanged for now
docs/
  refactor-design.md   # this file
```

`pyproject.toml` gains a `src`-layout build target and console scripts:

```toml
[project.scripts]
topic-judges = "topic_judges.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Also rename `name` to `sigir26-topic-adapters-as-judges` to match the repo.

## 3a. Additional requirements

These three items refine the design in §3/§4 and are load-bearing for the
package's API and data model.

### 3a.1 Optional validation set for training

`train()` (and the `topic-judges train` CLI) must **optionally** accept a
validation set. Today training runs `trainer.fit(model, data_module)` with no
validation loop — the `test_step`/metrics machinery exists but is never wired
into `fit`. The refactor adds:

- `RelevanceDataModule` gains an optional `val_data_path` (default `None`) and a
  `val_dataloader()` that is only registered when a validation set is present.
- `train(..., val_data_path=None)` / CLI `--val-data`; when omitted, behavior is
  identical to today (train-only, no validation loop).
- When a validation set is supplied, the existing `MetricCollection`
  (accuracy/precision/recall/f1) is reported per epoch via a `validation_step`
  (reusing the current `test_step` body), so we can watch for over/under-fitting
  — directly relevant to debugging the loss weighting (§3a.3).

Open sub-question: for `cdet`, where the builder only has positives + sampled
negatives, a held-out validation set must be sampled the **same way** (same
pos/neg ratio) to be meaningful — see §3a.2.

### 3a.2 Two kinds of data builders — make the distinction explicit

The builders fall into two categories that the package should model as such,
not blur together:

| Kind | Sources | Labels come from | Negatives |
| --- | --- | --- | --- |
| **Judged** | Robust04, TREC-DL | explicit qrels (pos *and* neg) | present in the source |
| **Positive-only + sampling** | CDet | only positive `rel_docs` per topic | **synthesized** by random sampling from the collection |

Design implications:

- `datasets/` splits along this line. A shared helper builds the final parquet
  from `(query_id, doc_id, query, doc, relevance)` rows; the two kinds differ
  only in how they produce those rows.
  - **Judged builders** (`robust04.py`, `trecdl.py`) map qrels → rows directly.
  - **Positive-only builders** (`cdet.py`) take positives from the topics file
    and call a `sample_negatives(...)` helper (the current logic: draw random
    docs not already positive for that topic).
- The negative-sampling ratio must be an explicit, named parameter — not the
  hardcoded `* 19` loop in `data_cdet_train.py`. Default preserves today's
  behavior (19 negatives per positive ⇒ 5% positive / 95% negative).
- Because this ratio is a **modeling choice**, not a property of the data, it is
  recorded in the design as coupled to the loss weights (§3a.3) and should be
  surfaced (e.g. logged / stored alongside the parquet) so a prediction run can
  reason about the training prior.

### 3a.3 Loss weighting is coupled to the pos/neg ratio — and may be wrong

**This is flagged for debugging, not just refactoring.** Do not silently "clean
it up" — preserve current numeric behavior behind a parameter and make the
coupling visible so it can be tested.

Current behavior (identical in every model's `training_step`/`test_step`):

```python
weight = (labels * 0.9) + 0.05      # label 1 -> 0.95, label 0 -> 0.05
loss = F.mse_loss(preds, labels, weight=weight)
```

So **positives are weighted 19× more than negatives** (0.95 vs 0.05). The CDet
builder samples **19 negatives per positive** (`* 19`, a 5/95 split). The loss
weight is therefore the *inverse* of the data frequency — a standard
inverse-frequency reweighting that makes the effective positive/negative
contribution to the loss roughly equal.

The chain of assumptions to verify:

1. **Data ratio ↔ loss weight are two encodings of the same number (19).** They
   are currently set independently in two files. If someone changes the sampling
   ratio without changing the weight (or vice versa), the reweighting silently
   stops balancing. The refactor should derive the weight from the ratio (or at
   least validate they agree) rather than hardcoding both.
2. **The weight is tuned to the training mix, but the goal is the prediction
   mix.** Inverse-frequency balancing targets a ~50/50 effective objective. But
   the *expected outcome* on `test-cdet.parquet` is presumably far from 50%
   positive (a pool is mostly non-relevant). If the intent was to match the
   expected positive rate in predictions, the weight (and/or a decision
   threshold at predict time) is likely **mis-set** — this is the part the
   author is unsure about.
3. **Threshold at prediction time.** Predictions currently emit a raw score;
   downstream thresholding (>0.5) assumes calibrated probabilities. Reweighting
   the loss shifts calibration, so the "right" threshold is entangled with the
   weight and the data ratio.

Refactor actions:

- Introduce explicit parameters: `neg_pos_ratio` (data builder) and
  `pos_weight`/`neg_weight` (or a single `class_balance` knob) on the model /
  `train()`. Default values **exactly reproduce** today's `0.95/0.05` and `*19`.
- Add a single documented place (docstring + this section) stating the invariant
  the author *intends*. The **working hypothesis is the current code**: weights
  are the inverse of the training ratio (both encode 19). This is what we
  preserve as the default and measure against.
- Add a debugging aid (out of the strict refactor, but planned): a small script
  / notebook that, for a trained adapter, plots the predicted-score distribution
  on the validation set vs. the known label balance, so the weight/threshold
  choice can be checked empirically. The optional validation set (§3a.1) exists
  partly to enable this.

### 3a.4 Observed symptom & debugging log (to explore *after* the refactor)

**Symptom:** with a trained `mono`+LoRA adapter, predicted scores on the test
data are heavily skewed toward 1.0, peaking around **~0.8**. The pipeline
assumes a **0.5 decision boundary**, which is almost certainly the wrong
operating point given this distribution.

**What's already been tried:**

| Experiment | Result |
| --- | --- |
| Current code: 5/95 split + `0.95/0.05` weighting | skew toward ~0.8 |
| **Balanced** train/test split **+** `0.95/0.05` weighting | **same skew** |

**What this rules in/out:** because balancing the data did **not** move the
peak, the training class *ratio* is not the sole driver. The skew persists
independent of balance, which points away from "inverse-frequency weight is
mismatched to the ratio" and toward one of:

- the reweighting formula / MSE-on-sigmoid objective producing a
  poorly-calibrated score regardless of balance;
- a fixed `0.5` threshold that is simply the wrong operating point for this
  model's score distribution;
- something upstream in how `mono` "true"-token probabilities become a score
  (`MonoDecoderLoRA.forward`).

**Options to try post-refactor (rough order):**

1. **Remove the weighting entirely + balanced split** — the not-yet-tried
   corner of the 2x2 (balanced data, *no* weighting). Simplest thing that could
   work; with `class_balance` defaulting to "off" this is a one-flag run.
2. **Pick the threshold empirically** from the validation-set score
   distribution instead of hardcoding 0.5.
3. **Inspect calibration** — histogram predicted scores split by true label
   (the §3a.3 debugging aid) to see whether the classes are separable at *any*
   threshold or genuinely overlapping.

These are exploration, not part of the refactor's definition of done. The
refactor's job is to make each a small, parameterized experiment (toggle
weighting, swap in a balanced builder, set a threshold) rather than a code edit.

**Validation (extends §7):** the regression check must fix `neg_pos_ratio` and
the weights to their defaults and confirm identical output to the current
scripts *before* any tuning of these values is attempted.

## 4. Public API (the reusable core)

```python
# topic_judges.models
def build_model(encoder_type, model_name, *, lora=True, **lora_kw): ...
    # returns (model, data_encoder_type); folds the mono->cross remap in ONE place

def load_adapter(model, adapter_path, *, lora=True): ...
    # PeftModel.from_pretrained + merge_and_unload, or full-FT load

# topic_judges.data
class RelevanceDataModule(pl.LightningDataModule): ...
    # single definition; train_data_path optional (None when predict-only),
    # val_data_path optional (None => no validation loop, today's behavior)

# topic_judges.train
def train(*, encoder_type, train_data_path, query_id, model_name, output_dir,
          val_data_path=None, lora=True, predict=False, predict_data_path=None,
          neg_pos_ratio=19, class_balance=None, ...) -> Path: ...
    # returns the saved adapter path; class_balance defaults reproduce 0.95/0.05

# topic_judges.predict
def predict(*, encoder_type, base_model_name, adapter_path, predict_data_path,
            query_id, output_path, batch_size=256, ...) -> Path: ...
    # writes jsonl {query_id, doc_id, relevance}; alignment check lives here
```

`predict` no longer imports from a training script — both `train` and `predict`
import the model/data classes from `topic_judges.models` / `topic_judges.data`.

## 5. CLI

Single console script with subcommands, replacing the per-file `cli()` funcs:

```
topic-judges train    --encoder-type mono --model-name ... --train-data ... --query-id ... --output-dir ...
topic-judges predict  --encoder-type mono --base-model-name ... --adapter-path ... --predict-data ... --query-id ... --output ...
topic-judges data cdet-train   --topics ... --coll ... --output ...
topic-judges data cdet-predict --topics ... --pool ... --coll ... --output ...
```

`scripts/predict-one.sh` is updated to call `topic-judges predict` and to read
model / data / output paths from environment variables (with the current values
as defaults), so it stays usable under `parallel` without per-run edits.

## 6. Migration plan (incremental, each step leaves the repo runnable)

1. Create `src/topic_judges/` and move the model classes into `models.py`,
   adding `build_model()` (single home for the mono→cross remap). *(no behavior
   change)*
2. Move `RelevanceDataset` / `RelevanceDataModule` into `data.py`; make
   `train_data_path` optional for predict-only use and add optional
   `val_data_path` + `val_dataloader()` (§3a.1).
3. Extract `train()` and `predict()` library functions into `train.py` /
   `predict.py`, importing from `models`/`data`. Centralize id/prediction
   alignment in `io.py`. Surface `neg_pos_ratio` / `class_balance` as
   parameters with defaults that reproduce today's numbers (§3a.3).
4. Add `cli.py` with `argparse` subcommands (incl. `--val-data`); wire
   `[project.scripts]`.
5. Move the builders into `datasets/`, split by kind — judged
   (`robust04.py`, `trecdl.py`) vs positive-only+sampling (`cdet.py`) over a
   shared `common.py` (§3a.2) — behind `topic-judges data ...`.
6. Update `scripts/predict-one.sh` to the new CLI + env-var paths.
7. Delete `train_cdet.py` and the old `predict.py` / `train.py` script bodies
   once the package reproduces their output.
8. Update `README.md` and `Makefile`; fix the `pyproject.toml` name.

## 7. Validation

- **Regression check:** run the new `topic-judges predict` for one `cdet` query
  against an existing trained adapter and diff the output jsonl against a run of
  the current `predict.py`. They must match (same seed, same schema).
- Confirm `topic-judges data cdet-predict` reproduces the current
  `test-cdet.parquet` byte-for-byte (or row-for-row) from the same inputs.
- Smoke-test a short `topic-judges train` run (few epochs) on one query to
  confirm the adapter saves and loads.

## 8. Explicitly out of scope (this pass)

- Refactoring `data_llm_inference.py` and the `eval_*_shallow.py` scripts.
- Changing model architectures, loss, or hyperparameter defaults.
- Multi-GPU / distributed prediction changes.
- Any change to the parquet schema or the `relevance = -1` sentinel.

## 9. Resolved decisions & open questions

Resolved:

1. **Package name** `topic_judges` — confirmed.
2. **Encoders**: `bi`/`cross` stay first-class (they're in the original paper),
   but `mono`+LoRA is the going-forward path for new predictions, since the
   paper found it best.
3. **Dataset builders** live in the package (`topic-judges data ...`), split by
   kind per §3a.2.

Open (to debug, not just refactor):

4. **Loss-weight intent (§3a.3)** — is the `0.95/0.05` weighting meant to be the
   inverse of the training pos/neg ratio (balance the objective), or to encode
   the expected positive rate at prediction time? These imply different values
   and different predict-time thresholds. Resolve empirically using the
   validation set before changing defaults.
