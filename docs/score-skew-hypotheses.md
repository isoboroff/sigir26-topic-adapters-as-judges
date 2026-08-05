# Prediction-Score Skew: Hypotheses

**Context:** A trained `mono`+LoRA CDet adapter produces predicted scores heavily
skewed toward 1.0, peaking around ~0.8. The original pipeline assumes a 0.5
decision boundary. See `refactor-design.md` §3a.3–3a.4 for the background and the
experiments already tried.

**Important constraint on testing.** The actual deliverable — predictions on
`data/external/test-cdet.parquet` — has **no true labels** (`relevance == -1`).
The goal is to *rank* pooled docs so the most-likely-relevant ones can be sent to
human annotators. So:

- Hypotheses that only need to explain the *shape* of the score distribution can
  be tested on the unlabeled prediction set.
- Hypotheses about whether the model actually *separates* relevant from
  non-relevant need labels — either a held-out labeled split of
  `train-cdet.parquet` (via `--val-data`), or the Robust04 / TREC-DL sets.
- For the real goal (annotator selection), threshold-free **ranking** quality
  (AUC / AP) matters more than where a 0.5 cut lands. A high, clustered score
  distribution is only a *problem* if the scores also fail to rank well.

**Data facts** (from `train-cdet.parquet`, checked 2026-08-04):

- Exactly **5% positive** per query (4,148 pos / 78,812 neg), i.e. the 19:1
  negative sampling ratio.
- Documents are long: p50 ≈ 2,500 chars, p95 ≈ 7,900 chars — most are truncated
  at `max_length=512` tokens.
- CDet queries are multi-sentence narratives + questions, fed to the model with
  **no MonoT5 prompt template**.

---

## Ranked hypotheses

### H1 — Wrong output head / missing MonoT5 template for narrative queries (highest prior)

`MonoDecoderLoRA.forward` keeps only the `▁true`/`▁false` logits and softmaxes
them. MonoT5 was trained on the template `Query: {q} Document: {d} Relevant:` →
true/false, but the CDet builder feeds a raw multi-sentence narrative with no
template. The base model's true/false prior is therefore uncalibrated for this
input, and LoRA on 5% positives can push it further toward "true."

- **Testable without labels, on existing prediction data.** Run the **base model
  (no adapter)** over `test-cdet.parquet` and histogram the raw scores. If the
  base model already peaks high on CDet-style input, the head/template mismatch
  is real and independent of training.
- Follow-up: wrap inputs in the actual MonoT5 template and re-check.

### H2 — Loss weighting sets the target mean, independent of data balance

With `weight = label*0.9 + 0.05`, positives count 19× more than negatives. MSE on
a probability output pushes predictions toward the *weighted* mean, not the data
mean. This is consistent with the observation that a **balanced split + weighting
still skewed** — because the weight itself (not the data ratio) sets the target.

- **Explains the shape; needs labels to judge correctness.** On the unlabeled
  prediction set you can compare score histograms across the 2×2 (weighting
  on/off × balanced/5-95 split) and see *where the mass moves* — but not whether
  the result is any good.
- Key experiment (the not-yet-tried 2×2 corner): `class_balance` **off** (no
  weighting) + balanced split. The refactor exposes this as a one-flag run.

### H3 — 0.5 is the wrong threshold; the model may separate fine, just not at 0.5

Reweighting + MSE decalibrates the output, so the operating point moves. A peak
at 0.8 with a fixed 0.5 cut would call almost everything relevant even if the
classes are perfectly separable at, e.g., 0.85.

- **NOT testable on the unlabeled prediction set** (no labels to split by).
  Requires a labeled held-out slice of `train-cdet.parquet` (`--val-data`) or the
  Robust04 / TREC-DL sets.
- Diagnostic: histogram held-out val scores split by true label; report
  threshold-free AUC / AP. If classes separate (both distributions just shifted
  up), pick the threshold from val (Youden's J / F1-max) — or, for annotator
  selection, ignore the threshold and select by rank.
- **Reframe for the real goal:** annotator selection is a ranking task. If scores
  rank relevant docs above the rest, the clustering at 0.8 does not matter —
  sort the pool and take the top-k (or a score-stratified sample). The skew is
  only fatal if the scores fail to *rank*, which is exactly what the val
  separation check measures.

### H4 — Truncation destroys the discriminative signal

p95 doc ≈ 8k chars ≫ 512 tokens, and the long narrative query also consumes the
token budget. If docs are truncated to mostly-query, the model has little to
discriminate on and falls back to its prior.

- **Test:** measure the fraction of pairs that hit `max_length`; compare 512 vs a
  larger budget (or query truncation) and check whether separation improves
  (needs labels for the "separation" part; truncation rate itself is label-free).

### H5 — Sampled negatives are noisy (actually-relevant) positives

CDet negatives are random collection docs *assumed* irrelevant. For broad
narratives ("institutions funding AI degrees"), random docs may genuinely be
on-topic, giving the model contradictory labels and pushing it to hedge high.

More fundamentally, **random negatives are trivially easy** — almost always
off-topic — so the judge learns "is this about the topic" rather than the hard
boundary "on-topic but *not relevant* to this need," which is exactly what
annotators care about.

- **Test:** manually inspect ~20 sampled negatives per a couple queries for real
  relevance; or train with harder/verified negatives and recheck.

#### H5-fix — BM25 hard negatives

Replace (some) random negatives with **BM25 top-ranked non-positives**:
lexically on-topic docs that are (mostly) not relevant — the hard boundary we
want the judge to learn. Standard hard-negative mining.

**Available data:** `data/external/cdet-bm25-results.trec` — TREC run format,
top-3000 per topic, 50 topics, query = the same string used in
`train-cdet.parquet`.

**Overlap analysis** (vs. `train-cdet.parquet` positives):

- **80.4%** of known positives (3,335 / 4,148) appear in the BM25 top-3000
  (was 71.3% at top-1000; extending the run to 3000 recovers the tail — 100% of
  *retrieved* positives are within 3000, i.e. no positive this run finds ranks
  deeper than 3000).
- Positives rank **high**: median rank 94, p25 rank 30, p75 rank 353.
- Positives within top-k (as % of retrieved positives): top-25 → 22%,
  top-50 → 37%, top-100 → 52%, top-500 → 81%, top-1000 → 89%, top-2000 → 97%,
  top-3000 → 100%.

**Two consequences the numbers force:**

1. **False negatives are a real risk.** BM25 hits enrich for relevance, so
   BM25 top docs that are not *known* positives include many *unjudged*
   positives — concentrated at the top (37% of retrieved positives fall in the
   top 50). Excluding known positives is mandatory but not sufficient. Mitigate
   with a **rank window** (skip the top) and a **hard/random mix** (keep clean
   easy negatives as an anchor).
2. **BM25 ↔ human judgments diverge** for these narrative queries (~20% of
   positives not retrieved even at depth 3000). This *strengthens* the
   hard-negative case (BM25 surfaces lexically-similar-but-not-judged-relevant
   docs) but also means BM25 relevance is a noisy proxy — another reason to mix
   rather than go 100% hard.

**Supply vs. demand** (rank window 50–2000, `neg_pos_ratio=19`): every topic can
fill its hard quota at `hard_frac <= 0.5` with **zero** random top-up; one topic
falls short at 0.75 and two at 1.0 (topics with many positives need
`19 × n_pos × hard_frac` negatives, which can exceed the window's non-positive
supply). Extending the window to 50–3000 raises supply further but pulls in the
noisiest tail (weakest BM25 hits, more unjudged positives), so 50–2000 is the
default.

**Builder design** (`datasets/cdet.py`, exposed on `topic-judges data cdet-train`):

- `sample_hard_negatives(...)` alongside the existing `sample_negatives(...)`,
  sharing the "avoid known positives, emit relevance=0 rows" contract.
- Knobs (all exposed as CLI flags so they can be swept):
  - `--negatives {random,hard,mixed}` — negative source (default `random`,
    preserving current behavior).
  - `--bm25-run PATH` — the TREC run file (required for hard/mixed).
  - `--hard-frac F` — fraction of each topic's negatives drawn hard (mixed
    only); remainder from the random sampler.
  - `--rank-window LO HI` — inclusive BM25 rank range to draw hard negatives
    from (default `50 2000`: skips the top ~37%-positive ranks, and supplies
    enough for `hard_frac <= 0.5` on every topic without random top-up).
  - `--neg-pos-ratio` — unchanged (default 19); only the *source* of negatives
    changes, so runs stay comparable.
- If a topic's rank window can't supply enough hard negatives, top up from
  random and `log()` the shortfall (no silent caps).

**Training sets built for the experiment** (data is not in the repo; regenerate
with these commands — deterministic given `--seed`). Both use the top-3000 BM25
run and the default `--rank-window 50 2000`:

```sh
# 5/95 balance, hard negatives (production class balance)
topic-judges data cdet-train \
  --negatives hard --bm25-run data/external/cdet-bm25-results.trec \
  --neg-pos-ratio 19 --seed 42 \
  --output data/external/train-cdet-hard-5-95.parquet

# 50/50 balance, hard negatives (ratio 1)
topic-judges data cdet-train \
  --negatives hard --bm25-run data/external/cdet-bm25-results.trec \
  --neg-pos-ratio 1 --seed 42 \
  --output data/external/train-cdet-hard-50-50.parquet
```

Resulting composition (validated: exact per-query balance, no positive leaked
into negatives, no intra-topic duplicates):

| Set | Rows | Balance | Hard negatives |
| --- | --- | --- | --- |
| `train-cdet-hard-5-95` | 82,960 | 4,148 pos / 78,812 neg | **89.2%** BM25-window (11% random top-up) |
| `train-cdet-hard-50-50` | 8,296 | 4,148 / 4,148 | **100%** BM25-window (no top-up) |

**Confound to keep in mind.** The two sets differ in *both* class balance and
hard-negative purity: 5/95 needs `19 × n_pos` negatives, which exceeds the
window on high-positive topics (e.g. `cd2026_005`, 244 positives), so ~11% fall
back to random; 50/50 needs only `1 × n_pos`, fully suppliable as hard. So a
difference between the two runs cannot be attributed to balance alone. 50/50 is
the clean all-hard set; 5/95 is the production-balance ("mostly hard") set. Both
are compared against the original random-negative baseline
(`train-cdet.parquet`).

### H6 — MSE-on-probability is the wrong objective

Regressing a sigmoid/softmax output toward {0, 1} with MSE has vanishing
gradients near the saturated ends and does not calibrate the way cross-entropy
would.

- **Test:** swap `_weighted_mse` for (weighted) BCE and compare the score
  histogram (label-free for the shape; labels needed to confirm it helps
  separation).

---

## Suggested first experiment

**H1's base-model score histogram** is the cheapest high-information test and the
only one runnable directly on the unlabeled prediction set: it tells you whether
the skew exists *before any training*, which would immediately implicate the
head/template rather than the loss weighting or data. If the base model looks
fine, the next fork is a labeled held-out CDet split (H3 diagnostic: AUC/AP +
histogram-by-label) to decide whether this is a ranking-quality problem or just a
threshold/calibration one.
