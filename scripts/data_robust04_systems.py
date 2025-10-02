import argparse

import pandas as pd
import trectools
import ir_datasets
import glob
from tqdm.notebook import tqdm

TOPICS = ["301", "304", "306", "307", "311", "318", "319", "321", "324", "343", "352", "353", "354", "357", "360", "367", "370", "374", "383", "389", "390", "391", "392", "395", "398", "399", "400", "404", "408", "412", "415", "418", "422", "424", "425", "431", "434", "435", "436", "438", "439", "443", "450", "631", "656", "683", "688", "700"]


def main(run_path: str = "../data/external/robust04-runs/*.gz", output_path: str = "../data/interim/run_evals.parquet"):
    qrels = trectools.TrecQrel()
    qrels.qrels_data = (
        pd.DataFrame(
            ir_datasets
            .load("disks45/nocr/trec-robust-2004")
            .qrels_iter()
        )
        .rename(columns={
            "query_id": "query",
            "doc_id": "docid",
            "relevance": "rel"
        })
        .assign(q0="Q0")
        .drop(columns=["iteration"])
    )

    eval_data = []
    for runfile in tqdm(glob.glob(run_path)):
        run = trectools.TrecRun(runfile)
        eval = trectools.TrecEval(run, qrels)
        eval_data.append(
            eval.get_ndcg(depth=10, per_query=True)
            .reset_index()
            .assign(system=run.get_runid())
            .query("query in @TOPICS")
        )

    df = pd.concat(eval_data)
    df.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", default="../data/external/robust04-runs/*.gz")
    parser.add_argument("--output_path", default="../data/interim/run_evals.parquet")
    args = parser.parse_args()
    main(args.run_path, args.output_path)