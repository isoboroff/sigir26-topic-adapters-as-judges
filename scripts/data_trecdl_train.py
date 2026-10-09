import argparse

import pandas as pd
from ir_datasets import load


def main(output_path: str):
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
        pd.merge(
            qrels,
            queries,
            on="query_id",
            how="left"
        )
        .merge(
            docs,
            on="doc_id",
            how="left"
        )
        .loc[:, ["query_id", "doc_id", "query", "doc", "relevance"]]
        .to_parquet(output_path)
    )

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_path", default="../data/external/train-trecdl.parquet")
    args = arg_parser.parse_args()
    main(args.output_path)