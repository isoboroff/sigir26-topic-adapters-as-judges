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


def read_trec_run(run_path):
    """Read a TREC run file into ``{query_id: [doc_id, ...]}`` ordered by rank.

    Expects the standard 6-field format::

        query_id  Q0  doc_id  rank  score  tag

    Docs are returned in ascending rank order (as they appear in the file).
    """
    run = {}
    with open(run_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank = parts[0], parts[1], parts[2], parts[3]
            run.setdefault(qid, []).append((int(rank), docid))
    return {qid: [d for _, d in sorted(docs)] for qid, docs in run.items()}



def rows_to_parquet(rows, output_path):
    """Write a list of row dicts to ``output_path`` as parquet in COLUMNS order."""
    pd.DataFrame(rows).loc[:, COLUMNS].to_parquet(output_path)
    return output_path


def _neg_row(topic, docid, docs, narratives):
    return {
        "query_id": topic,
        "doc_id": docid,
        "query": narratives[topic],
        "doc": docs[docid],
        "relevance": 0,
    }


def sample_negatives(*, positives, all_doc_ids, topic_docs, doc_topics, docs,
                     narratives, neg_pos_ratio=DEFAULT_NEG_POS_RATIO, rng=None):
    """Yield randomly-sampled negative rows for positive-only datasets.

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
        yield from _sample_random_for_topic(
            topic, n_neg, already_have, all_doc_ids, doc_topics, docs,
            narratives, rng,
        )


def _sample_random_for_topic(topic, n_neg, already_have, all_doc_ids,
                             doc_topics, docs, narratives, rng):
    """Yield ``n_neg`` random negative rows for one topic, avoiding conflicts.

    Mutates ``already_have`` so callers can chain random on top of hard
    negatives without redrawing the same doc.
    """
    for _ in range(n_neg):
        while True:
            docid = rng.choice(all_doc_ids)
            if docid not in already_have and topic not in doc_topics.get(docid, ()):
                already_have.add(docid)
                yield _neg_row(topic, docid, docs, narratives)
                break


def sample_hard_negatives(*, positives, topic_docs, doc_topics, docs,
                          narratives, bm25_run, all_doc_ids=None,
                          neg_pos_ratio=DEFAULT_NEG_POS_RATIO, hard_frac=1.0,
                          rank_window=(50, 2000), rng=None, log=None):
    """Yield negative rows drawn from BM25 top-ranked non-positives.

    For each topic, take ``hard_frac`` of its ``neg_pos_ratio`` * n_pos
    negatives from ``bm25_run`` within ``rank_window`` (1-indexed, inclusive),
    excluding any known positive for that topic. The remaining ``1 - hard_frac``
    come from random sampling. If the rank window can't supply enough hard
    negatives, the shortfall is topped up from random and reported via ``log``.

    Args:
        positives: set of doc_ids relevant to *any* topic (excluded globally,
            matching the random sampler's contract).
        topic_docs: {topic_id: set(rel doc_ids)}.
        doc_topics: {doc_id: set(topic_ids it is relevant to)}.
        docs: {doc_id: text}.
        narratives: {topic_id: query text}.
        bm25_run: {topic_id: [doc_id, ...]} in ascending rank order.
        all_doc_ids: candidate pool for the random remainder / top-up. Required
            when ``hard_frac < 1`` or top-up may be needed.
        neg_pos_ratio: total negatives to draw per positive.
        hard_frac: fraction of each topic's negatives drawn from BM25 (0..1).
        rank_window: (lo, hi) inclusive 1-indexed BM25 rank range for hard negs.
        rng: optional random.Random for reproducibility.
        log: optional callable(str) for shortfall / diagnostic messages.
    """
    rng = rng or random
    lo, hi = rank_window
    for topic in topic_docs:
        n_pos = len(topic_docs[topic])
        n_neg = n_pos * neg_pos_ratio
        n_hard = round(n_neg * hard_frac)
        n_rand = n_neg - n_hard

        already_have = set(positives)
        already_have.update(topic_docs[topic])

        # Hard negatives: BM25 docs in the rank window, not a known positive
        # anywhere, and present in the loaded collection.
        ranked = bm25_run.get(topic, [])
        window = ranked[lo - 1:hi]  # 1-indexed inclusive
        n_hard_emitted = 0
        for docid in window:
            if n_hard_emitted >= n_hard:
                break
            if docid in already_have:
                continue
            if topic in doc_topics.get(docid, ()):  # positive for this topic
                continue
            if docid not in docs:  # not in the loaded collection
                continue
            already_have.add(docid)
            n_hard_emitted += 1
            yield _neg_row(topic, docid, docs, narratives)

        # Top up the hard shortfall from random, plus the intended random share.
        shortfall = n_hard - n_hard_emitted
        n_rand_total = n_rand + shortfall
        if shortfall and log:
            log(f"{topic}: BM25 window supplied {n_hard_emitted}/{n_hard} hard "
                f"negatives; topping up {shortfall} from random")
        if n_rand_total:
            if all_doc_ids is None:
                raise ValueError(
                    "all_doc_ids is required for the random negative share/top-up"
                )
            yield from _sample_random_for_topic(
                topic, n_rand_total, already_have, all_doc_ids, doc_topics,
                docs, narratives, rng,
            )

