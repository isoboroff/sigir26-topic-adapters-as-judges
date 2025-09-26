import argparse

import pandas as pd
from ir_datasets import load


def main(output_path: str):
    dataset = load("disks45/nocr/trec-robust-2004")

    qrels = pd.DataFrame(dataset.qrels_iter())
    docs = pd.DataFrame(dataset.docs_iter())
    queries = pd.DataFrame(dataset.queries_iter())

    (
        pd.merge(
            qrels,
            queries.rename(columns={"title": "query"}),
            on="query_id",
            how="left"
        )
        .merge(
            docs.assign(doc=lambda df: df["title"] + "\n\n" + df["body"]).loc[:, ["doc_id", "doc"]],
            on="doc_id",
            how="left"
        )
        .loc[:, ["query_id", "doc_id", "query", "doc", "relevance"]]
        .to_parquet(output_path)
    )

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_path", default="../data/external/train-robust04.parquet")
    args = arg_parser.parse_args()
    main(args.output_path)