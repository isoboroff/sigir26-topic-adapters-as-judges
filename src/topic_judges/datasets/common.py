"""Shared building blocks for dataset builders.

The final parquet always has these columns, in this order:
"""

import random

import pandas as pd

COLUMNS = ["query_id", "doc_id", "query", "doc", "relevance"]

# relevance sentinel for unlabeled pairs (prediction sets)
UNLABELED = -1

# Default negatives sampled per positive for positive-only datasets. 19 gives a
# 5% positive / 95% negative split, and is the inverse of the loss weighting in
# topic_judges.models (see design sec 3a.3 -- the two encode the same number).
DEFAULT_NEG_POS_RATIO = 19


def rows_to_parquet(rows, output_path):
    """Write a list of row dicts to ``output_path`` as parquet in COLUMNS order."""
    pd.DataFrame(rows).loc[:, COLUMNS].to_parquet(output_path)
    return output_path


def sample_negatives(*, positives, all_doc_ids, topic_docs, doc_topics, docs,
                     narratives, neg_pos_ratio=DEFAULT_NEG_POS_RATIO, rng=None):
    """Yield sampled negative rows for positive-only datasets.

    For each topic, draw ``neg_pos_ratio`` random documents per positive that
    are neither already selected for the topic nor relevant to it in any
    question, labeling them ``relevance = 0``.

    Args:
        positives: set of already-selected (relevant) doc_ids to avoid.
        all_doc_ids: list of candidate doc_ids to sample from.
        topic_docs: {topic_id: set(rel doc_ids)}.
        doc_topics: {doc_id: set(topic_ids it is relevant to)}.
        docs: {doc_id: text}.
        narratives: {topic_id: query text}.
        neg_pos_ratio: negatives to draw per positive.
        rng: optional random.Random for reproducibility.
    """
    rng = rng or random
    for topic in topic_docs:
        already_have = set(positives)
        already_have.update(topic_docs[topic])
        n_neg = len(topic_docs[topic]) * neg_pos_ratio
        for _ in range(n_neg):
            while True:
                docid = rng.choice(all_doc_ids)
                if docid not in already_have and topic not in doc_topics[docid]:
                    already_have.add(docid)
                    yield {
                        "query_id": topic,
                        "doc_id": docid,
                        "query": narratives[topic],
                        "doc": docs[docid],
                        "relevance": 0,
                    }
                    break
