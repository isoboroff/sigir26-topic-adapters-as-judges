"""Data loading for topic-specific relevance judges.

Everything here is built around the shared parquet schema every ``data_*``
builder emits::

    query_id: str   doc_id: str   query: str   doc: str   relevance: int

``relevance == -1`` marks an unlabeled pair to be predicted. The data module
selects a single ``query_id`` for prediction (one adapter per topic).

``RelevanceDataModule`` supports three independent roles, any subset of which
may be active depending on which paths are supplied:

- training      -- requires ``train_data_path``
- validation    -- optional, requires ``val_data_path`` (design sec 3a.1)
- prediction    -- requires ``predict_data_path`` (+ ``query_id``)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


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
            'relevance': float(row['relevance']),
        }


class RelevanceDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data_path=None,
                 val_data_path=None,
                 predict_data_path=None,
                 query_id="301",
                 tokenizer=None, batch_size=16, max_length=512,
                 num_workers=4, encoder_type="cross", binary: bool = True):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.predict_data_path = predict_data_path
        self.query_id = query_id
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.encoder_type = encoder_type
        self.binary = binary

        self.train_data = self.val_data = self.predict_data = None
        self.train_dataset = self.val_dataset = self.predict_dataset = None

    def setup(self, stage=None):
        cols = ["query", "doc", "relevance"]

        if self.train_data_path is not None:
            self.train_data = pd.read_parquet(self.train_data_path)
            self.train_dataset = RelevanceDataset(self.train_data.loc[:, cols])

        if self.val_data_path is not None:
            self.val_data = pd.read_parquet(self.val_data_path)
            self.val_dataset = RelevanceDataset(self.val_data.loc[:, cols])

        if self.predict_data_path is not None:
            self.predict_data = pd.read_parquet(self.predict_data_path)
            self.predict_dataset = RelevanceDataset(
                self.predict_data.loc[self.predict_data.query_id == self.query_id, cols]
            )

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
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': torch.tensor(relevances, dtype=torch.float),
            }
        else:
            # Bi-encoder: tokenize queries and documents separately
            query_encoding = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            doc_encoding = self.tokenizer(
                docs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            return {
                'query_input_ids': query_encoding['input_ids'],
                'query_attention_mask': query_encoding['attention_mask'],
                'doc_input_ids': doc_encoding['input_ids'],
                'doc_attention_mask': doc_encoding['attention_mask'],
                'labels': torch.tensor(relevances, dtype=torch.float),
            }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        # Only meaningful when a validation set was supplied; Lightning skips the
        # validation loop entirely if this returns None.
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
