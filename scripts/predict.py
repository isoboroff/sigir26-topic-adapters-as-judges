import json
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl

# Reuse the classes you already have
from train_cdet import (
    BiEncoderLoRA,
    CrossEncoderLoRA,
    MonoDecoderLoRA,
    RelevanceDataModule,
)


def load_model(encoder_type, base_model_name, adapter_path, lora):
    # Recreate the model architecture, then load the saved (LoRA) weights
    if encoder_type == "cross":
        model = CrossEncoderLoRA(model_name=base_model_name, lora=lora)
        data_encoder_type = "cross"
    elif encoder_type == "bi":
        model = BiEncoderLoRA(model_name=base_model_name, lora=lora)
        data_encoder_type = "bi"
    elif encoder_type == "mono":
        model = MonoDecoderLoRA(model_name=base_model_name, lora=lora)
        data_encoder_type = "cross"  # same remap as training
    else:
        raise ValueError("Unknown encoder type")

    # Load the saved weights into the underlying HF model.
    # If you used LoRA + save_pretrained, this loads the adapter.
    from peft import PeftModel
    if lora:
        model.model = PeftModel.from_pretrained(model.model, adapter_path)
        model.model = model.model.merge_and_unload()
    else:
        # full fine-tune: load the saved model weights
        model.model = model.model.from_pretrained(adapter_path)

    model.eval()
    return model, data_encoder_type


def main(
    encoder_type,
    base_model_name,
    adapter_path,
    predict_data_path,
    query_id,
    output_path,
    batch_size,
    max_length,
    binary,
    lora,
    device,
):
    model, data_encoder_type = load_model(
        encoder_type, base_model_name, adapter_path, lora
    )

    data_module = RelevanceDataModule(
        train_data_path=predict_data_path,   # unused for predict, but required by __init__
        predict_data_path=predict_data_path,
        query_id=query_id,
        tokenizer=model.tokenizer,
        batch_size=batch_size,               # can be much larger for inference
        max_length=max_length,
        encoder_type=data_encoder_type,
        binary=binary,
        num_workers=0 if device == "mps" else 4,
    )

    # Single-device trainer => ordered predictions, valid zip()
    trainer = pl.Trainer(
        accelerator="mps" if device == "mps" else "gpu",
        devices=1,
        precision='16-mixed' if device == "cuda" else '32-true',
        logger=False,
        enable_checkpointing=False,
    )

    data_module.setup()
    ids = data_module.predict_data.loc[data_module.predict_data.query_id == query_id, ["query_id", "doc_id"]].values.tolist()
    predictions = trainer.predict(model, data_module)
    predictions = [x.cpu().tolist() for y in predictions for x in y]

    if len(predictions) != len(ids):
        raise RuntimeError(
            f"Prediction count ({len(predictions)}) != id count ({len(ids)}); "
            "ordering/alignment cannot be guaranteed."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for (q, d), s in zip(ids, predictions):
            f.write(json.dumps({"query_id": q, "doc_id": d, "relevance": s}) + "\n")
    print(f"Predictions saved to {output_path}")


def cli():
    parser = argparse.ArgumentParser(description="Run predictions with a trained LoRA model")
    parser.add_argument("--encoder_type", type=str, choices=["cross", "bi", "mono"], required=True)
    parser.add_argument("--base_model_name", type=str, required=True,
                        help="The original base model name used during training")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the saved model/adapter directory")
    parser.add_argument("--predict_data_path", type=str, required=True)
    parser.add_argument("--query_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size (can be large; no gradients stored)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--binary", type=bool, default=True)
    parser.add_argument("--lora", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
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


if __name__ == "__main__":
    cli()