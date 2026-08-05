"""Model definitions for topic-specific relevance judges.

Three PyTorch Lightning modules wrap a base ranker with optional LoRA adapters:

- ``BiEncoderLoRA``    -- dual-encoder, cosine similarity of pooled embeddings
- ``CrossEncoderLoRA`` -- single-tower sequence classifier over [query, doc]
- ``MonoDecoderLoRA``  -- monoT5-style seq2seq, score = P("true")

All three share the same training objective: MSE on the (0, 1) relevance label
with a per-example weight ``(label * 0.9) + 0.05`` (positives weighted 19x over
negatives). This weighting is the *inverse of the CDet training pos/neg ratio*
(19 sampled negatives per positive); see docs/refactor-design.md sec 3a.3 for
why it is under investigation. Behavior here is preserved verbatim from the
original scripts -- do not retune without updating that section.

``build_model()`` is the single place that maps an ``encoder_type`` string to a
model instance and reports the ``data_encoder_type`` the data module should use
(``mono`` reuses the ``cross`` tokenization path).
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from peft import get_peft_model, PeftModel, LoraConfig, TaskType
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)


def _relevance_metrics():
    """Binary classification metrics keyed for logging (threshold 0.5)."""
    return MetricCollection({
        "accuracy": Accuracy(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        "f1": F1Score(task="binary"),
    })


def _weighted_mse(preds, labels, pos_weight=0.95, neg_weight=0.05):
    """MSE with per-example weights interpolated by label.

    ``weight = label * (pos_weight - neg_weight) + neg_weight``. The defaults
    (0.95 / 0.05) reproduce the original ``(label*0.9)+0.05`` -- positives
    weighted 19x negatives, the inverse of the 5/95 training ratio (design
    sec 3a.3). Passing ``pos_weight == neg_weight`` (e.g. class_balance="off")
    makes the weighting uniform, i.e. plain MSE.
    """
    weight = labels * (pos_weight - neg_weight) + neg_weight
    return torch.nn.functional.mse_loss(preds, labels, weight=weight)


# Named class-balance presets -> (pos_weight, neg_weight) for the loss.
# "inverse-ratio" is the original behavior; "off" is uniform (plain MSE).
CLASS_BALANCE = {
    "inverse-ratio": (0.95, 0.05),
    "off": (1.0, 1.0),
}


def resolve_class_balance(class_balance):
    """Map a class_balance name (or None -> default) to (pos_weight, neg_weight)."""
    if class_balance is None:
        class_balance = "inverse-ratio"
    try:
        return CLASS_BALANCE[class_balance]
    except KeyError:
        raise ValueError(
            f"Unknown class_balance {class_balance!r}; "
            f"choose from {sorted(CLASS_BALANCE)}"
        ) from None


class BiEncoderLoRA(pl.LightningModule):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1, pos_weight=0.95, neg_weight=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

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

        self.metrics = _relevance_metrics()

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
        return (similarity + 1) / 2

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(
            batch['query_input_ids'], batch['query_attention_mask'],
            batch['doc_input_ids'], batch['doc_attention_mask']
        )
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch, log_key):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(
            batch['query_input_ids'], batch['query_attention_mask'],
            batch['doc_input_ids'], batch['doc_attention_mask']
        )
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_test')

    def predict_step(self, batch, batch_idx):
        return self.forward(
            batch['query_input_ids'], batch['query_attention_mask'],
            batch['doc_input_ids'], batch['doc_attention_mask']
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class CrossEncoderLoRA(pl.LightningModule):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L12-v2", learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1, pos_weight=0.95, neg_weight=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

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
        self.metrics = _relevance_metrics()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits.squeeze(-1))

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch, log_key):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_test')

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


class MonoDecoderLoRA(pl.LightningModule):
    def __init__(self, model_name="castorini/monot5-base-msmarco-10k", prediction_token_false='▁false', prediction_token_true='▁true', learning_rate=2e-5, lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1, pos_weight=0.95, neg_weight=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

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

        self.metrics = _relevance_metrics()

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
        return torch.exp(logits[:, 1])  # Only 'true' logit

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch, log_key):
        labels = batch['labels'].to(dtype=torch.float)
        preds = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = _weighted_mse(preds, labels, self.pos_weight, self.neg_weight)
        self.log(log_key, loss, on_step=False, on_epoch=True, prog_bar=False)
        self.metrics((preds > 0.5).int(), (labels > 0.5).int())
        self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, 'loss_test')

    def predict_step(self, batch, batch_idx):
        return self.forward(batch['input_ids'], batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


# Map encoder_type -> (model class, data-module encoder_type). `mono` reuses the
# `cross` tokenization path (query/doc concatenated into one sequence).
_ENCODERS = {
    "bi": (BiEncoderLoRA, "bi"),
    "cross": (CrossEncoderLoRA, "cross"),
    "mono": (MonoDecoderLoRA, "cross"),
}


def build_model(encoder_type, model_name, *, lora=False, learning_rate=2e-5,
                lora_r=16, lora_alpha=32, lora_dropout=0.1, class_balance=None):
    """Instantiate a judge model.

    Returns ``(model, data_encoder_type)`` where ``data_encoder_type`` is what
    :class:`~topic_judges.data.RelevanceDataModule` should tokenize for. This is
    the single home for the ``mono`` -> ``cross`` remap.

    ``class_balance`` selects the loss weighting: ``None``/``"inverse-ratio"``
    reproduces the original 0.95/0.05 (positives 19x); ``"off"`` is uniform
    (plain MSE). See :func:`resolve_class_balance`.
    """
    try:
        model_cls, data_encoder_type = _ENCODERS[encoder_type]
    except KeyError:
        raise ValueError(f"Unknown encoder type: {encoder_type!r}") from None

    pos_weight, neg_weight = resolve_class_balance(class_balance)
    model = model_cls(
        model_name=model_name,
        learning_rate=learning_rate,
        lora=lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
    )
    return model, data_encoder_type


def load_adapter(model, adapter_path, *, lora=True):
    """Load saved weights into ``model`` in place and return it (eval mode).

    With ``lora=True`` this loads a PEFT adapter and merges it into the base
    weights; otherwise it reloads a full fine-tuned model.
    """
    if lora:
        model.model = PeftModel.from_pretrained(model.model, adapter_path)
        model.model = model.model.merge_and_unload()
    else:
        model.model = model.model.from_pretrained(adapter_path)
    model.eval()
    return model
