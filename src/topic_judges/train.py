"""Train a topic-specific relevance judge (LoRA adapter) for one query.

``train()`` fits a model on a training parquet, optionally validates against a
held-out parquet (design sec 3a.1), saves the adapter, and optionally runs a
prediction pass. It does not parse arguments -- see ``topic_judges.cli``.
"""

from pathlib import Path

import torch
import lightning.pytorch as pl

from .models import build_model
from .data import RelevanceDataModule
from .io import flatten_predictions, align_predictions, write_predictions


def train(*, encoder_type, train_data_path, query_id, model_name, output_dir,
          val_data_path=None, predict_data_path=None, lora=False,
          batch_size=64, accumulate_batches=1, max_epochs=10,
          learning_rate=1e-4, lora_r=64, lora_alpha=128, lora_dropout=0.1,
          max_length=512, max_length_doc=None, class_balance=None,
          sentence_transformer=False, predict=False, seed=159410123,
          device="mps"):
    """Fit and save a judge adapter. Returns the saved adapter path.

    When ``val_data_path`` is given, a validation loop reports per-epoch metrics.
    When ``predict`` is True, runs a prediction pass over ``predict_data_path``
    (defaults to the training data if not given) and writes predictions.json
    beside the adapter.

    ``class_balance`` selects the loss weighting (default reproduces the
    original 0.95/0.05; ``"off"`` = plain MSE) -- see topic_judges.models.
    ``sentence_transformer`` + ``max_length_doc`` apply to the ``bi`` encoder
    (load via sentence-transformers; give docs their own length budget) -- see
    docs/understanding-bi-encoders.md.
    """
    pl.seed_everything(seed)
    run = f"{encoder_type}-{model_name.split('/')[-1]}-{'lora' if lora else 'native'}-{query_id}"
    logger = pl.loggers.CSVLogger("logs_shallow", name=run)

    model, data_encoder_type = build_model(
        encoder_type, model_name, lora=lora, learning_rate=learning_rate,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        class_balance=class_balance, sentence_transformer=sentence_transformer,
    )

    data_module = RelevanceDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        predict_data_path=predict_data_path or train_data_path,
        query_id=query_id,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        max_length_doc=max_length_doc,
        encoder_type=data_encoder_type,
        num_workers=0 if device == 'mps' else 4,
    )

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator='mps' if device == 'mps' else 'auto',
        devices=1 if device == 'mps' else 'auto',
        strategy='auto' if device == 'mps' else 'ddp',
        precision='16-mixed' if device != 'mps' else '32-true',
        accumulate_grad_batches=accumulate_batches,
        callbacks=[],
        enable_checkpointing=False,
    )

    trainer.fit(model, data_module)

    # Save the final model / adapter
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    model_save_path = output_dir / run
    model.model.save_pretrained(str(model_save_path))
    print(f"Model saved to {model_save_path}")

    if predict:
        # Single-device trainer for clean, ordered prediction
        predict_trainer = pl.Trainer(
            accelerator='mps' if device == 'mps' else 'gpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        model_save_path.mkdir(exist_ok=True)
        ids = data_module.predict_data.loc[:, ["query_id", "doc_id"]].values.tolist()
        scores = flatten_predictions(predict_trainer.predict(model, data_module))
        records = align_predictions(ids, scores)
        write_predictions(records, model_save_path / "predictions.json")
        print(f"Predictions saved to {model_save_path}/predictions.json")

    return model_save_path
