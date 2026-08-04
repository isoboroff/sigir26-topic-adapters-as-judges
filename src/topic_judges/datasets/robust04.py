"""Judged builder: TREC Robust 2004 (via ir_datasets).

Labels come from explicit qrels (positive and negative), so no negative
sampling is needed.
"""

import pandas as pd
from ir_datasets import load

from .common import rows_to_parquet  # noqa: F401  (kept for symmetry / future use)


def build_train(output_path="data/external/train-robust04.parquet"):
    """Build the Robust04 training parquet from ir_datasets qrels."""
    dataset = load("disks45/nocr/trec-robust-2004")

    qrels = pd.DataFrame(dataset.qrels_iter())
    docs = pd.DataFrame(dataset.docs_iter())
    queries = pd.DataFrame(dataset.queries_iter())

    (
        pd.merge(
            qrels,
            queries.rename(columns={"title": "query"}),
            on="query_id",
            how="left",
        )
        .merge(
            docs.assign(doc=lambda df: df["title"] + "\n\n" + df["body"]).loc[:, ["doc_id", "doc"]],
            on="doc_id",
            how="left",
        )
        .loc[:, ["query_id", "doc_id", "query", "doc", "relevance"]]
        .to_parquet(output_path)
    )
    return output_path
