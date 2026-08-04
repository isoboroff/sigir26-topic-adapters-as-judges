"""`topic-judges data <builder>` subcommands.

Registered lazily by topic_judges.cli. Each builder emits the shared parquet
schema (see topic_judges.datasets).
"""

import os

_HOME = os.environ.get("HOME", "/home/ubuntu")


def add_data_parser(sub):
    p = sub.add_parser("data", help="Build datasets")
    dsub = p.add_subparsers(dest="dataset", required=True)

    # --- CDet (positive-only + sampling) ---
    ct = dsub.add_parser("cdet-train",
                         help="CDet training parquet (positives + sampled negatives)")
    ct.add_argument("--topics", dest="topics_path",
                    default=f"{_HOME}/trec/trec35/change/topics/test-topics-all-docs.jsonl")
    ct.add_argument("--coll", dest="coll_path",
                    default=f"{_HOME}/ragtime1/eng-docs.jsonl.gz")
    ct.add_argument("--output", dest="output_path",
                    default="data/external/train-cdet.parquet")
    ct.add_argument("--neg-pos-ratio", dest="neg_pos_ratio", type=int, default=19,
                    help="Negatives sampled per positive (default 19 => 5/95)")
    ct.add_argument("--seed", type=int, default=None,
                    help="Seed for reproducible negative sampling")
    ct.set_defaults(func=_run_cdet_train)

    cp = dsub.add_parser("cdet-predict",
                         help="CDet prediction parquet from a retrieval pool")
    cp.add_argument("--topics", dest="topics_path",
                    default=f"{_HOME}/trec/trec35/change/topics/test-topics-all-docs.jsonl")
    cp.add_argument("--pool", dest="pool_path",
                    default=f"{_HOME}/trec/trec35/change/pooling/pool.d3")
    cp.add_argument("--coll", dest="coll_path",
                    default=f"{_HOME}/ragtime1/eng-docs.jsonl.gz")
    cp.add_argument("--output", dest="output_path",
                    default="data/external/test-cdet.parquet")
    cp.set_defaults(func=_run_cdet_predict)

    # --- Judged builders ---
    rb = dsub.add_parser("robust04", help="Robust04 training parquet (judged)")
    rb.add_argument("--output", dest="output_path",
                    default="data/external/train-robust04.parquet")
    rb.set_defaults(func=_run_robust04)

    dl = dsub.add_parser("trecdl", help="TREC-DL 2019+2020 training parquet (judged)")
    dl.add_argument("--output", dest="output_path",
                    default="data/external/train-trecdl.parquet")
    dl.set_defaults(func=_run_trecdl)


def _run_cdet_train(args):
    from .cdet import build_train
    build_train(topics_path=args.topics_path, coll_path=args.coll_path,
                output_path=args.output_path, neg_pos_ratio=args.neg_pos_ratio,
                seed=args.seed)


def _run_cdet_predict(args):
    from .cdet import build_predict
    build_predict(topics_path=args.topics_path, pool_path=args.pool_path,
                  coll_path=args.coll_path, output_path=args.output_path)


def _run_robust04(args):
    from .robust04 import build_train
    build_train(output_path=args.output_path)


def _run_trecdl(args):
    from .trecdl import build_train
    build_train(output_path=args.output_path)
