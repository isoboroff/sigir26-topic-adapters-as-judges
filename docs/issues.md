# Open Issues & Loose Ends

Running tracker for this project. Grouped into **experiments** (research threads
to run), **code gaps** (things the docs assume but the code doesn't do yet), and
**process** (repo hygiene / workflow). Check items off or delete them as they
close. See `refactor-design.md` and `score-skew-hypotheses.md` for background.

_Last updated: 2026-08-05._

---

## Experiments (queued research)

- [ ] **H1 — base-model score histogram.** Tooling is ready
  (`topic-judges predict` with `--lora` omitted → base checkpoint, then
  `topic-judges score-dist`). Run on `cd2026_001` on the GPU box; recipe in
  `score-skew-hypotheses.md` (H1). Tells us if the skew predates training.
- [ ] **H5 — hard-negative training runs.** Two sets already built locally
  (`train-cdet-hard-5-95.parquet`, `train-cdet-hard-50-50.parquet`; regenerate
  commands in `score-skew-hypotheses.md`). Train + predict + `score-dist`,
  compare against the random-negative baseline. Note the balance-vs-purity
  confound between the two sets (documented under H5-fix).
- [ ] **H2 — the not-yet-tried 2×2 corner:** no weighting + balanced split.
  Now a one-flag run: `topic-judges train --class-balance off` on the 50/50
  training set.
- [ ] **H3 — separation / threshold check.** Needs labels: a held-out labeled
  slice of `train-cdet.parquet` (via `--val-data`) or Robust04/TREC-DL. Report
  AUC/AP + histogram-by-label. Reframe: for annotator selection this is a
  *ranking* question, not a threshold one.
- [ ] **H6 — swap weighted MSE for (weighted) BCE** and compare score
  distribution. Blocked on / entangled with the loss-function work.
- [ ] **H1 follow-up — MonoT5 prompt template.** Wrap inputs in
  `Query: {q} Document: {d} Relevant:` and re-check. Needs a data-module hook;
  not built.
- [ ] **ModernBERT long-context model** (`joe32140/ModernBERT-large-msmarco`).
  It's a *bi-encoder* sentence-transformer (8192 ctx), so it maps to the
  `BiEncoderLoRA` path — but that path mean-pools + adds an untrained 128-dim
  projection, which discards the checkpoint's trained embedding. Loading it
  properly (sentence-transformers pooling, drop/rethink the projection) is a
  real `models.py` change, not a flag. Attacks H4 (truncation).
- [ ] **H4 — truncation impact.** Measure the fraction of pairs hitting
  `max_length=512` (label-free); compare larger context / query-truncation.
- [ ] **H5 caveat — inspect sampled negatives.** Manually eyeball ~20 random
  and ~20 hard negatives per a couple queries for actual relevance (false-negative
  check).

## Code gaps (docs assume, code doesn't do yet)

- [x] **`class_balance` toggle** — done. `topic-judges train --class-balance
  {inverse-ratio,off}`; `inverse-ratio` (default) reproduces the original
  0.95/0.05, `off` is plain MSE. Threaded `build_model` → model → `_weighted_mse`.
- [x] **`--binary` flag** — removed (was a no-op that couldn't be turned off).
- [ ] **Full-fine-tune path is untested.** `load_adapter(lora=False)` reloads a
  model directory via `from_pretrained`; the base-model and full-FT cases share
  this path. Works by design, but no regression coverage like predict has.
- [ ] **MonoT5 template hook** (see H1 follow-up) — no way to apply an
  input template in the data module.

## Process / repo hygiene

- [ ] **`feature/hard-negatives` is unpushed / no PR.** 4 commits ahead of
  `main` (hard-neg builder, window default, training-set docs, score-dist +
  --lora fix). Push and open a PR when ready.
- [ ] **`refactor/relevance-judge-package` branch still exists locally** after
  its PR merged (#1). Safe to delete.
- [ ] **`.DS_Store` not in `.gitignore`** — shows up as untracked in every
  status. Add it.
- [ ] **Data files are not in the repo by design.** `train-cdet*.parquet`,
  `test-cdet.parquet`, and `cdet-bm25-results.trec` are regenerated from
  documented commands (`score-skew-hypotheses.md`, builder CLIs). Keep the
  regenerate commands current when builders change.
