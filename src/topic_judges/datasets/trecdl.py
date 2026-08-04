"""Judged builder: TREC Deep Learning 2019+2020 (via ir_datasets).

Labels come from explicit qrels (positive and negative), so no negative
sampling is needed. The two years are concatenated.
"""

import pandas as pd
from ir_datasets import load


def build_train(output_path="data/external/train-trecdl.parquet"):
    """Build the TREC-DL training parquet by merging 2019 and 2020 judged sets."""
    dataset_19 = load("msmarco-passage/trec-dl-2019/judged")
    dataset_20 = load("msmarco-passage/trec-dl-2020/judged")

    qrels = (
        pd.concat([pd.DataFrame(dataset_19.qrels_iter()), pd.DataFrame(dataset_20.qrels_iter())])
        .loc[:, ["query_id", "doc_id", "relevance"]]
    )
    docs = (
        pd.concat([pd.DataFrame(dataset_19.docs_iter()), pd.DataFrame(dataset_20.docs_iter())])
        .rename(columns={"text": "doc"})
        .loc[:, ["doc_id", "doc"]]
    )
    queries = (
        pd.concat([pd.DataFrame(dataset_19.queries_iter()), pd.DataFrame(dataset_20.queries_iter())])
        .rename(columns={"text": "query"})
    )

    (
        pd.merge(qrels, queries, on="query_id", how="left")
        .merge(docs, on="doc_id", how="left")
        .loc[:, ["query_id", "doc_id", "query", "doc", "relevance"]]
        .to_parquet(output_path)
    )
    return output_path
