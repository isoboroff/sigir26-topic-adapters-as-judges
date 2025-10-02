import trectools
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from glob import glob
import ir_datasets
import random
from tqdm import tqdm

TOPICS = ['104861', '1063750', '1106007', '1112341', '1113437', '1114646', '1114819', '1117099', '1124210', '1133167', '131843', '148538', '156493', '168216', '182539', '183378', '264014', '359349', '443396', '451602', '47923', '489204', '490595', '527433', '573724', '833860', '87181', '87452', '915593', '1037496', '1051399', '1106979', '1108651', '1109707', '1110678', '1121353', '1122767', '1132532', '1133579', '1136962', '118440', '121171', '141630', '174463', '23849', '330975', '332593', '405163', '47210', '555530', '583468', '640502', '67316', '877809', '914916', '938400', '940547']
RATIOS = [0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_ITER = 10
DEPTHS = [5, 10, 20, 50, 100, 1000]


def eval_subsampled(iteration, runs, original_qrels, original_pool, ratios=RATIOS, depths=DEPTHS, task="dl19"):
    eval_data = []
    eval_data_unjudged = []

    pbar = tqdm(total=len(ratios) * 6 * len(runs), desc=f"Iteration {iteration}/{NUM_ITER}")

    for rate in ratios:
        # Simulate reduced pool
        run_sample = random.sample(runs, int(len(runs) * rate))
        pool = trectools.trec_pool_maker.TrecPoolMaker().make_pool(run_sample, "topX", topX=100)
        judged = pd.DataFrame([{"query": q, "docid": d} for q, v in pool.pool.items() for d in v])
        unjudged = pd.concat([original_pool.loc[:, ["query", "docid"]], judged]).drop_duplicates(keep=False)
        for source in [64, 128, 192, 256]:
            qrels_predicted = (
                pd.read_parquet(f"data/processed/trecdl-shallow-qrels-{source}.parquet")
                .rename(columns={"query_id": "query", "doc_id": "docid", "relevance": "rel"})
                .assign(q0="Q0", rel=lambda df: df["rel"] >= 0.5)
                .astype({"query": str, "docid": str})
                .query("query in @TOPICS")
            )
            qrels = trectools.TrecQrel()
            qrels.qrels_data = (
                pd.concat([
                    judged.merge(original_qrels, on=["query", "docid"]),
                    unjudged.merge(qrels_predicted, on=["query", "docid"])
                ])
                .set_index(["query", "docid"])
                .fillna(original_qrels.set_index(["query", "docid"]))
                .reset_index()
            )

            for r in runs:
                if r in run_sample:
                    continue
                # Prediction qrels
                eval = trectools.TrecEval(r, qrels)
                eval_data.append(
                    pd.concat([
                        pd.concat([eval.get_ndcg(depth=k, per_query=True) for k in
                                   depths], axis=1).reset_index().melt(
                            id_vars=["query"]).rename(columns={"variable": "metric"}),
                        pd.concat([eval.get_unjudged(depth=k, per_query=True) for k in
                                   depths], axis=1).reset_index().melt(
                            id_vars=["query"]).rename(columns={"variable": "metric"})
                    ])
                    .reset_index()
                    .assign(
                        system=r.get_runid(),
                        iteration=iteration,
                        run_sample_rate=rate,
                        unjudged_ratio=len(unjudged) / (len(unjudged) + len(judged)),
                        source=f"predicted_{source}",
                        in_pool=r in run_sample
                    )
                    .query("query in @TOPICS")
                )
                pbar.update(1)

        # Filling qrels
        for fill_value in [0, 1]:
            qrels = trectools.TrecQrel()
            qrels.qrels_data = pd.concat([
                judged.merge(original_qrels, on=["query", "docid"]),
                unjudged.assign(rel=fill_value, q0="Q0")
            ])

            for r in runs:
                if r in run_sample:
                    continue
                eval = trectools.TrecEval(r, qrels)
                eval_data_unjudged.append(
                    pd.concat([
                        pd.concat([eval.get_ndcg(depth=k, per_query=True) for k in
                                   depths], axis=1).reset_index().melt(
                            id_vars=["query"]).rename(columns={"variable": "metric"}),
                        pd.concat([eval.get_unjudged(depth=k, per_query=True) for k in
                                   depths], axis=1).reset_index().melt(
                            id_vars=["query"]).rename(columns={"variable": "metric"})
                    ])
                    .reset_index()
                    .assign(
                        system=r.get_runid(),
                        iteration=iteration,
                        run_sample_rate=rate,
                        unjudged_ratio=len(unjudged) / (len(unjudged) + len(judged)),
                        source=f"filled_{fill_value}",
                        in_pool=r in run_sample
                    )
                    .query("query in @TOPICS")
                )
                pbar.update(1)

    df_pool = pd.concat(eval_data).astype({
        'query': str,
        'metric': str,
        'value': float,
        'system': str,
        'iteration': int,
        'run_sample_rate': float,
        'unjudged_ratio': float,
        'source': str,
        'in_pool': bool
    })
    df_pool.to_parquet(f"data/processed/dl{task}-eval-subsampled-predicted_fillna_{iteration}.parquet")
    df_unjudged = pd.concat(eval_data_unjudged).astype({
        'query': str,
        'metric': str,
        'value': float,
        'system': str,
        'iteration': int,
        'run_sample_rate': float,
        'unjudged_ratio': float,
        'source': str,
        'in_pool': bool
    })
    df_unjudged.to_parquet(f"data/processed/dl{task}-eval-subsampled-default_fillna_{iteration}.parquet")


def eval_original(runs, original_qrels, depths=DEPTHS, task="dl19"):
    df_original = []

    qrels = trectools.TrecQrel()
    qrels.qrels_data = original_qrels

    for r in tqdm(runs):
        eval = trectools.TrecEval(r, qrels)
        df_original.append(
            pd.concat([
                pd.concat([eval.get_ndcg(depth=k, per_query=True) for k in depths],
                    axis=1).reset_index().melt(id_vars=["query"]).rename(columns={"variable": "metric"}),
                pd.concat(
                    [eval.get_unjudged(depth=k, per_query=True) for k in depths],
                    axis=1).reset_index().melt(id_vars=["query"]).rename(columns={"variable": "metric"})
            ])
            .reset_index()
            .assign(
                system=r.get_runid(),
                iteration=0,
                run_sample_rate=1.0,
                unjudged_ratio=1.0,
                source=f"original",
                in_pool=True
            )
            .query("query in @TOPICS")
        )

    df_original = pd.concat(df_original).astype({
        'query': str,
        'metric': str,
        'value': float,
        'system': str,
        'iteration': int,
        'run_sample_rate': float,
        'unjudged_ratio': float,
        'source': str,
        'in_pool': bool
    })
    df_original.to_parquet(f"data/processed/dl{task}-eval-original.parquet")


def main():
    for task in [20, 19]:
        original_qrels = (
            pd.DataFrame(
                ir_datasets
                .load(f"msmarco-passage/trec-dl-20{task}/judged")
                .qrels_iter()
            )
            .assign(relevance=lambda df: df["relevance"] > 0)
            .rename(columns={
                "query_id": "query",
                "doc_id": "docid",
                "relevance": "rel"
            })
            .assign(q0="Q0")
            .drop(columns=["iteration"])
            .query("query in @TOPICS")
        )

        runs = [trectools.TrecRun(f) for f in glob(f"data/external/dl{task}-passages-runs/*.gz")]

        original_pool = (
            pd.DataFrame([{"query": q, "docid": d} for q, v in
                          trectools.trec_pool_maker.TrecPoolMaker().make_pool(runs, "topX", topX=100).pool.items() for d in
                          v])
        )
        #for iteration in range(NUM_ITER):
        #    eval_subsampled(iteration=iteration + 4, runs=runs, original_qrels=original_qrels, original_pool=original_pool, task=task)
        eval_original(runs=runs, original_qrels=original_qrels, task=task)


if __name__ == "__main__":
    main()