"""I/O helpers shared by training and prediction.

Predictions are written as JSON Lines, one object per query/doc pair::

    {"query_id": ..., "doc_id": ..., "relevance": <score>}

The single source of truth for aligning model outputs back to their ids lives
here (``align_predictions``) so train and predict cannot drift apart.
"""

import json
from pathlib import Path


def flatten_predictions(batched):
    """Flatten Lightning's per-batch prediction tensors into a flat score list."""
    return [x.cpu().tolist() for batch in batched for x in batch]


def align_predictions(ids, scores):
    """Zip ``[[query_id, doc_id], ...]`` with model scores, checking alignment.

    Trainer must run single-device so prediction order matches ``ids`` order.
    Raises if the counts differ, since a mismatch means the zip would silently
    misalign scores to the wrong pairs.
    """
    if len(scores) != len(ids):
        raise RuntimeError(
            f"Prediction count ({len(scores)}) != id count ({len(ids)}); "
            "ordering/alignment cannot be guaranteed."
        )
    return [
        {"query_id": q, "doc_id": d, "relevance": s}
        for (q, d), s in zip(ids, scores)
    ]


def write_predictions(records, output_path):
    """Write prediction records as JSON Lines, creating parent dirs. Returns path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return output_path
