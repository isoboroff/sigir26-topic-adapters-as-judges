import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM
import argparse


class RelevanceDataset(Dataset):
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'query': str(row['query']),
            'doc': str(row['doc']),
            'relevance': float(row['relevance'])
        }


class BiEncoderLoRA(pl.LightningModule):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Configure LoRA
        if lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules='all-linear',
            )
            self.model = get_peft_model(self.model, lora_config)

        # Projection head for similarity scoring
        self.projection = nn.Linear(self.model.base_model.config.hidden_size, 128)
        self.criterion = nn.MSELoss()

        # Metrics (assuming binary relevance > 0.5 is relevant)
        self.metrics = MetricCollection({
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
        })

    def encode_text(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return self.projection(mean_pooled)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        query_emb = self.encode_text(query_input_ids, query_attention_mask)
        doc_emb = self.encode_text(doc_input_ids, doc_attention_mask)
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(query_emb, doc_emb)
        return (similarity + 1)/2

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(
            batch['query_input_ids'], batch['query_attention_mask'],
            batch['doc_input_ids'], batch['doc_attention_mask']
        )
        # Normalize cosine sim from -1/1 to 0/1
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(
            batch['query_input_ids'], batch['query_attention_mask'],
            batch['doc_input_ids'], batch['doc_attention_mask']
        )
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_test', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class CrossEncoderLoRA(pl.LightningModule):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L12-v2", learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

        # Configure LoRA
        if lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules='all-linear',
            )
            self.model = get_peft_model(self.model, lora_config)

        self.criterion = nn.MSELoss()

        # Metrics (assuming binary relevance > 0.5 is relevant)
        self.metrics = MetricCollection({
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
        })


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits.squeeze(-1))

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_test', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class MonoDecoderLoRA(pl.LightningModule):
    def __init__(self, model_name="castorini/monot5-base-msmarco-10k", prediction_token_false='▁false', prediction_token_true='▁true', learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.token_false_id = self.tokenizer.get_vocab()[prediction_token_false]
        self.token_true_id = self.tokenizer.get_vocab()[prediction_token_true]
        # Configure LoRA
        if lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules='all-linear',
            )
            self.model = get_peft_model(self.model, lora_config)

        # Metrics (assuming binary relevance > 0.5 is relevant)
        self.metrics = MetricCollection({
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
        })

    def logits(self, input_ids, attention_mask):
        decode_ids = torch.full((input_ids.size(0), 1), self.model.config.decoder_start_token_id, dtype=torch.long).to(input_ids.device)
        encoder_outputs = self.model.get_encoder()(input_ids, attention_mask=attention_mask)
        model_inputs = self.model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )
        outputs = self.model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        return next_token_logits

    def forward(self, input_ids, attention_mask):
        logits = self.logits(input_ids, attention_mask)
        logits = logits[:, [self.token_false_id, self.token_true_id]]
        logits = torch.nn.functional.log_softmax(logits, dim=1)
        return torch.exp(logits[:, 1]) # Only 'true' logit

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        weight = (labels * 0.9) + 0.05
        loss = torch.nn.functional.mse_loss(preds, labels, weight=weight)
        self.log('loss_test', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class RelevanceDataModule(pl.LightningDataModule):
    def __init__(self, data_path="../data/external/robust04.parquet", query_id="301",
                 tokenizer=None, batch_size=16, max_length=512, train_split=0.8,
                 num_workers=4, encoder_type="cross", binary: bool = True):
        super().__init__()
        self.data_path = data_path
        self.query_id = query_id
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_split = train_split
        self.num_workers = num_workers
        self.encoder_type = encoder_type
        self.binary = binary

    def setup(self, stage=None):
        # Load and split data
        data = pd.read_parquet(self.data_path)

        # Shuffle data
        topic_sample = data[data.query_id == self.query_id].sample(frac=1.0)
        if self.binary:
            topic_sample["relevance"] = (topic_sample["relevance"] > 0.5).astype(int)

        # Split into train/test, stratified by relevance
        pos = topic_sample[topic_sample["relevance"] > 0]
        neg = topic_sample[topic_sample["relevance"] == 0]

        if isinstance(self.train_split, float):
            train_end_pos = int(len(pos) * self.train_split)
            train_end_neg = int(len(neg) * self.train_split)
        elif isinstance(self.train_split, int):
            train_end_pos = int(self.train_split * 0.125)
            train_end_neg = self.train_split - train_end_pos
        else:
            raise NotImplementedError

        self.train_data = pd.concat([
            pos.iloc[:train_end_pos],
            neg.iloc[:train_end_neg]
        ]).loc[:, ["query_id", "query", "doc_id", "doc", "relevance"]]
        self.test_data = pd.concat([
            pos.iloc[train_end_pos:],
            neg.iloc[train_end_neg:]
        ]).loc[:, ["query_id", "query", "doc_id", "doc", "relevance"]]

        # Create datasets
        self.train_dataset = RelevanceDataset(self.train_data.loc[:, ["query", "doc", "relevance"]])
        self.test_dataset = RelevanceDataset(self.test_data.loc[:, ["query", "doc", "relevance"]])

    def collate_fn(self, batch):
        queries = [item['query'] for item in batch]
        docs = [item['doc'] for item in batch]
        relevances = [item['relevance'] for item in batch]

        if self.encoder_type == "cross":
            # Cross-encoder: tokenize query-document pairs together
            encoding = self.tokenizer(
                queries,
                docs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': torch.tensor(relevances, dtype=torch.float)
            }
        else:
            # Bi-encoder: tokenize queries and documents separately
            query_encoding = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            doc_encoding = self.tokenizer(
                docs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'query_input_ids': query_encoding['input_ids'],
                'query_attention_mask': query_encoding['attention_mask'],
                'doc_input_ids': doc_encoding['input_ids'],
                'doc_attention_mask': doc_encoding['attention_mask'],
                'labels': torch.tensor(relevances, dtype=torch.float)
            }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

def main(
    encoder_type,
    data_path,
    query_id,
    lora,
    model_name,
    batch_size,
    accumulate_batches,
    max_epochs,
    learning_rate,
    lora_r,
    lora_alpha,
    lora_dropout,
    max_length,
    output_dir,
    binary,
    predict,
    split,
    seed
):

    pl.seed_everything(seed)
    run = f"{encoder_type}-{model_name.split('/')[-1]}-{'lora' if lora else 'native'}-{query_id}"
    logger = pl.loggers.CSVLogger("logs_shallow", name=run)

    if encoder_type == "cross":
        model = CrossEncoderLoRA(
            model_name=model_name,
            learning_rate=learning_rate,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    elif encoder_type == "bi":
        model = BiEncoderLoRA(
            model_name=model_name,
            learning_rate=learning_rate,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    elif encoder_type == "mono":
        model = MonoDecoderLoRA(
            model_name=model_name,
            learning_rate=learning_rate,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        encoder_type = "cross" # Remap for data module instantiation
    else:
        raise ValueError("Unknown encoder type")

    data_module = RelevanceDataModule(
        data_path=data_path,
        query_id=query_id,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        train_split=split,
        encoder_type=encoder_type,
        binary=binary,
    )
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator="auto",         devices="auto",         accumulate_grad_batches=accumulate_batches,
        callbacks=[],
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module)
    metrics = trainer.test(model, data_module)


    # Save the final model
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    model_save_path = output_dir / run
    model.model.save_pretrained(str(model_save_path))
    #model.tokenizer.save_pretrained(str(model_save_path))
    print(f"Model saved to {model_save_path}")

    if predict:
        model_save_path.mkdir(exist_ok=True)
        ids = data_module.test_data.loc[:, ["query_id", "doc_id"]].values.tolist()
        predictions = trainer.predict(model, data_module)
        predictions = [x.cpu().tolist() for y in predictions for x in y]
        predictions = [(q, d, s) for ((q, d), s) in zip(ids, predictions)]
        with open(f"{model_save_path}/predictions.json", "w") as f:
            for (q, d, s) in predictions:
                f.write(json.dumps({"query_id": q, "doc_id": d, "relevance": s}) + "\n")
        print(f"Predictions saved to {output_dir}/{run}/predictions.json")

    return metrics

def cli():
    parser = argparse.ArgumentParser(description="Train Encoder with LoRA")
    parser.add_argument("--encoder_type", type=str, choices=["cross", "bi", "mono"], default="mono",                         help="Type of encoder: cross-encoder or bi-encoder")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--query_id", type=str, required=True)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="castorini/monot5-base-msmarco-10k",                         help="Model name for cross-encoder, or sentence transformer model for bi-encoder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulate_batches", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--binary", type=bool, default=True, help="Whether to cast relevance scores to binary labels")
    parser.add_argument("--predict", type=bool, default=False, help="Whether to predict on test data and write to file")
    parser.add_argument("--split", type=float, default=0.8, help="Size of train split per topic")
    parser.add_argument("--seed", type=int, default=159410123, help="Random seed")
    args = parser.parse_args()

    main(
        encoder_type=args.encoder_type,
        data_path=args.data_path,
        query_id=args.query_id,
        lora=args.lora,
        model_name=args.model_name,
        batch_size=args.batch_size,
        accumulate_batches=args.accumulate_batches,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        output_dir=args.output_dir,
        binary=args.binary,
        predict=args.predict,
        split=args.split,
        seed=args.seed
    )

if __name__ == "__main__":
    cli()
