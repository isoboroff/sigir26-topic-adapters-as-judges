"""Run a trained judge over a parquet of query/doc pairs.

Loads a base model + saved (LoRA) adapter, predicts relevance for a single
``query_id``, and writes JSON Lines. This module does not depend on the training
code -- both import model/data classes from the package.
"""

from pathlib import Path

import lightning.pytorch as pl

from .models import build_model, load_adapter
from .data import RelevanceDataModule
from .io import flatten_predictions, align_predictions, write_predictions


def predict(*, encoder_type, base_model_name, adapter_path, predict_data_path,
            query_id, output_path, batch_size=128, max_length=512,
            lora=True, device="cuda"):
    """Predict relevance for ``query_id`` and write JSON Lines to ``output_path``.

    Returns the output path. ``base_model_name`` is the original base model used
    at training time; ``adapter_path`` is the saved adapter/model directory.
    """
    model, data_encoder_type = build_model(encoder_type, base_model_name, lora=lora)
    model = load_adapter(model, adapter_path, lora=lora)

    data_module = RelevanceDataModule(
        predict_data_path=predict_data_path,
        query_id=query_id,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        encoder_type=data_encoder_type,
        num_workers=0 if device == "mps" else 4,
    )

    # Single-device trainer => ordered predictions, valid zip() in align_predictions
    trainer = pl.Trainer(
        accelerator="mps" if device == "mps" else "gpu",
        devices=1,
        precision='16-mixed' if device == "cuda" else '32-true',
        logger=False,
        enable_checkpointing=False,
    )

    data_module.setup()
    ids = data_module.predict_data.loc[
        data_module.predict_data.query_id == query_id, ["query_id", "doc_id"]
    ].values.tolist()

    scores = flatten_predictions(trainer.predict(model, data_module))
    records = align_predictions(ids, scores)
    output_path = write_predictions(records, output_path)
    print(f"Predictions saved to {output_path}")
    return output_path
