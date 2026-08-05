"""Score-distribution diagnostic for prediction files.

Reads a predictions JSON Lines file (as written by ``topic-judges predict``:
one ``{"query_id", "doc_id", "relevance"}`` per line) and prints summary
statistics plus a text histogram of the relevance scores. No plotting or
pandas dependency, and no GPU -- pure post-processing over an existing file.

Used to inspect the score skew (docs/score-skew-hypotheses.md): e.g. score
``test-cdet.parquet`` with the untuned base model and check whether the
distribution already peaks high before any training (H1).
"""

import json
from pathlib import Path


def read_scores(path):
    """Read the ``relevance`` score from each line of a predictions jsonl."""
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(float(json.loads(line)["relevance"]))
    return scores


def _quantile(sorted_scores, q):
    """Linear-interpolation quantile of an already-sorted list (q in [0, 1])."""
    if not sorted_scores:
        return float("nan")
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    pos = q * (len(sorted_scores) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_scores):
        return sorted_scores[lo] * (1 - frac) + sorted_scores[lo + 1] * frac
    return sorted_scores[lo]


def summarize(scores, *, bins=20, width=50):
    """Return a printable summary (stats + text histogram) for a score list.

    ``bins`` histogram buckets span [0, 1]; ``width`` is the max bar length.
    Scores outside [0, 1] are clamped into the edge bins for display but counted
    faithfully in the stats.
    """
    n = len(scores)
    if n == 0:
        return "no scores"

    s = sorted(scores)
    mean = sum(s) / n
    stats = [
        f"count   {n}",
        f"mean    {mean:.4f}",
        f"min     {s[0]:.4f}",
        f"p10     {_quantile(s, 0.10):.4f}",
        f"p25     {_quantile(s, 0.25):.4f}",
        f"median  {_quantile(s, 0.50):.4f}",
        f"p75     {_quantile(s, 0.75):.4f}",
        f"p90     {_quantile(s, 0.90):.4f}",
        f"max     {s[-1]:.4f}",
        f"%>0.5   {100 * sum(1 for x in s if x > 0.5) / n:.1f}%",
    ]

    # Histogram over [0, 1]. Values are clamped to the edge bins for display.
    counts = [0] * bins
    for x in scores:
        idx = int(min(max(x, 0.0), 1.0) * bins)
        if idx == bins:  # x == 1.0 lands in the last bin
            idx = bins - 1
        counts[idx] += 1
    peak = max(counts) or 1

    lines = ["", "score distribution:"]
    for i, c in enumerate(counts):
        lo, hi = i / bins, (i + 1) / bins
        bar = "#" * round(width * c / peak)
        lines.append(f"  [{lo:.2f},{hi:.2f}) {c:>7d} {bar}")

    return "\n".join(stats) + "\n" + "\n".join(lines)


def run(path, *, bins=20):
    """Read ``path`` and print its score summary. Returns the score list."""
    scores = read_scores(path)
    print(f"{Path(path)}")
    print(summarize(scores, bins=bins))
    return scores
