"""Command-line entry point: ``topic-judges <command> ...``.

Subcommands:

- ``train``   -- fit and save a judge adapter for one query
- ``predict`` -- run a saved adapter over a parquet and write jsonl
- ``data``    -- build datasets (see topic_judges.datasets)

Preserves the flags of the original train.py/predict.py scripts, with hyphenated
long options; ``--val-data`` is new (design sec 3a.1).
"""

import argparse


def _add_train_parser(sub):
    p = sub.add_parser("train", help="Train a judge adapter for one query")
    p.add_argument("--encoder-type", choices=["cross", "bi", "mono"], default="mono")
    p.add_argument("--train-data", dest="train_data_path", required=True)
    p.add_argument("--val-data", dest="val_data_path", default=None,
                   help="Optional held-out parquet; enables the validation loop")
    p.add_argument("--predict-data", dest="predict_data_path", default=None,
                   help="Parquet to predict on when --predict is set (default: train data)")
    p.add_argument("--query-id", dest="query_id", required=True)
    p.add_argument("--model-name", dest="model_name", default="castorini/monot5-base-msmarco-10k")
    p.add_argument("--output-dir", dest="output_dir", default="./outputs")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--accumulate-batches", dest="accumulate_batches", type=int, default=1)
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=10)
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=64)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=128)
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.1)
    p.add_argument("--max-length", dest="max_length", type=int, default=512)
    p.add_argument("--binary", action="store_true", default=True)
    p.add_argument("--predict", action="store_true",
                   help="After training, predict and write predictions.json")
    p.add_argument("--seed", type=int, default=159410123)
    p.add_argument("--device", default="mps")
    p.set_defaults(func=_run_train)


def _add_predict_parser(sub):
    p = sub.add_parser("predict", help="Run a saved adapter over a parquet")
    p.add_argument("--encoder-type", choices=["cross", "bi", "mono"], required=True)
    p.add_argument("--base-model-name", dest="base_model_name", required=True,
                   help="Original base model name used during training")
    p.add_argument("--adapter-path", dest="adapter_path", required=True,
                   help="Path to the saved model/adapter directory")
    p.add_argument("--predict-data", dest="predict_data_path", required=True)
    p.add_argument("--query-id", dest="query_id", required=True)
    p.add_argument("--output", dest="output_path", required=True)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--max-length", dest="max_length", type=int, default=512)
    p.add_argument("--binary", action="store_true", default=True)
    p.add_argument("--lora", action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_run_predict)


def _run_train(args):
    from .train import train
    train(
        encoder_type=args.encoder_type,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        predict_data_path=args.predict_data_path,
        query_id=args.query_id,
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora=args.lora,
        batch_size=args.batch_size,
        accumulate_batches=args.accumulate_batches,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        binary=args.binary,
        predict=args.predict,
        seed=args.seed,
        device=args.device,
    )


def _run_predict(args):
    from .predict import predict
    predict(
        encoder_type=args.encoder_type,
        base_model_name=args.base_model_name,
        adapter_path=args.adapter_path,
        predict_data_path=args.predict_data_path,
        query_id=args.query_id,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        binary=args.binary,
        lora=args.lora,
        device=args.device,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="topic-judges",
                                     description="Topic-specific relevance judges")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(sub)
    _add_predict_parser(sub)

    # `data` subcommands are registered by the datasets package (step 5).
    try:
        from .datasets.cli import add_data_parser
        add_data_parser(sub)
    except ImportError:
        pass

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
