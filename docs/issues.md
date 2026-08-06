# Open Issues & Loose Ends

Running tracker for this project. Grouped into **experiments** (research threads
to run), **code gaps** (things the docs assume but the code doesn't do yet), and
**process** (repo hygiene / workflow). Check items off or delete them as they
close. See `refactor-design.md` and `score-skew-hypotheses.md` for background.

_Last updated: 2026-08-05._

---

## Experiments (queued research)

- [x] **H1 — base-model score histogram.** DONE (2026-08-05, `cd2026_001`):
  **H1 confirmed** — the untuned base model already skews high (mean 0.77,
  97.8% >0.5, peak ~0.82), so the skew is *not* a training artifact. Demotes
  H2/H5/H6 to secondary. Full result in `score-skew-hypotheses.md` (H1).
- [ ] **H5 — hard-negative training runs.** Two sets already built locally
  (`train-cdet-hard-5-95.parquet`, `train-cdet-hard-50-50.parquet`; regenerate
  commands in `score-skew-hypotheses.md`). Train + predict + `score-dist`,
  compare against the random-negative baseline. Note the balance-vs-purity
  confound between the two sets (documented under H5-fix). **Lower priority now
  that H1 shows the skew predates training** — hard negatives won't fix the
  base-model prior.
- [ ] **H2 — the not-yet-tried 2×2 corner:** no weighting + balanced split.
  Now a one-flag run: `topic-judges train --class-balance off` on the 50/50
  training set. **Lower priority for the same reason as H5.**
- [ ] **H3 — separation / threshold check.** Needs labels: a held-out labeled
  slice of `train-cdet.parquet` (via `--val-data`) or Robust04/TREC-DL. Report
  AUC/AP + histogram-by-label. Reframe: for annotator selection this is a
  *ranking* question, not a threshold one. **Most useful next diagnostic** — the
  base scores span 0.16–0.99, so they may rank well despite the skew.
- [ ] **H6 — swap weighted MSE for (weighted) BCE** and compare score
  distribution. Blocked on / entangled with the loss-function work.
- [ ] **H1 follow-up — MonoT5 prompt template. BLOCKED on context length.**
  Wrapping inputs in `Query: {q} Document: {d} Relevant:` would test whether the
  missing template causes the skew, but the CDet narrative + document already
  overflows MonoT5's 512-token window, so the template can't be applied without
  first solving length. Converges with H4 / ModernBERT below.
- [~] **ModernBERT long-context model** — IMPLEMENTED (branch
  `feature/modernbert-biencoder`), not yet run on GPU. `joe32140/
  ModernBERT-base-msmarco` (the `base`, not `large` — large doesn't fit the
  V100s) via the bi-encoder path with a new `--sentence-transformer` mode that
  keeps the checkpoint's trained pooling, plus `--max-length-doc` for an
  independent document budget. Chosen on the prior that a ranking-tuned
  embedding beats a raw MLM. **To run:** see the config in
  `docs/plan-modernbert-biencoder.md` (start `--max-length-doc 2048`, small
  batch + grad accum, tune vs. nvidia-smi). **Verification deferred** to the
  human assessments from the prioritized pools (no CDet labels now).
- [x] **H4 — truncation impact.** Measured (2026-08-05, `cd2026_001`): query is
  333 tokens, 97.7% of pairs exceed 512, only ~26% of the doc retained. Confirms
  truncation is severe; motivates the long-context bi-encoder above. Full
  numbers in `score-skew-hypotheses.md` (H4).
- [ ] **H5 caveat — inspect sampled negatives.** Manually eyeball ~20 random
  and ~20 hard negatives per a couple queries for actual relevance (false-negative
  check).

## GPU-box V100 caveat

FlashAttention-2 requires Ampere (sm_80+); the V100 is Volta (sm_70), so
ModernBERT falls back to SDPA/eager attention whose memory grows ~O(length²).
Keep `--max-length-doc` moderate (≈2048 covers ~p90 of docs) and use a small
batch + gradient accumulation; confirm gradient checkpointing is on if pushing
length higher.

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
- [x] **`.DS_Store` in `.gitignore`** — added.
- [ ] **Data files are not in the repo by design.** `train-cdet*.parquet`,
  `test-cdet.parquet`, and `cdet-bm25-results.trec` are regenerated from
  documented commands (`score-skew-hypotheses.md`, builder CLIs). Keep the
  regenerate commands current when builders change.
