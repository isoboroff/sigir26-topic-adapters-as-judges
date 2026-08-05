"""Positive-only builder: TREC Change Detection (CDet) topics.

The CDet topics file gives only *positive* relevant docs per topic question, so
the training set synthesizes negatives by random sampling from the collection
(design sec 3a.2). The prediction set is built from a retrieval pool instead:
every pooled doc becomes an unlabeled (relevance = -1) pair to be scored.

Both readers share the same topics format: JSON lines with ``tid``,
``narrative``, and ``questions`` (each question has a ``question`` string and,
for training, ``rel_docs``). The query text is the narrative followed by all of
the topic's question strings.
"""

import collections
import gzip
import json
import random

from tqdm import tqdm

from .common import (
    UNLABELED,
    DEFAULT_NEG_POS_RATIO,
    rows_to_parquet,
    sample_negatives,
    sample_hard_negatives,
    read_trec_run,
)


def _read_topics(topics_path):
    """Read topics jsonl and return {topic_id: query_text}, plus the raw topics."""
    with open(topics_path, "r") as f:
        topics = [json.loads(line) for line in f]
    return topics


def _narrative_for(topic):
    """Query text = narrative followed by each question string."""
    narrative = topic["narrative"]
    for question in topic["questions"]:
        narrative += "\n" + question["question"]
    return narrative


def _load_collection(coll_path, docs):
    """Fill ``docs`` (dict keyed by doc_id) with text from a gzipped jsonl coll.

    If ``docs`` starts non-empty, only those ids are kept (predict path); if it
    starts empty, every doc is loaded (train path, which needs the full pool to
    sample negatives from).
    """
    keep_only = len(docs) > 0
    with gzip.open(coll_path, "rt") as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line)
            if not keep_only or doc["id"] in docs:
                docs[doc["id"]] = doc["text"]
    return docs


def build_train(*, topics_path, coll_path, output_path,
                neg_pos_ratio=DEFAULT_NEG_POS_RATIO, seed=None,
                negatives="random", bm25_run=None, hard_frac=1.0,
                rank_window=(50, 2000), log=print):
    """Build the CDet training parquet: positives from topics + sampled negatives.

    Negatives are drawn at ``neg_pos_ratio`` per positive (default 19 => 5/95).

    ``negatives`` selects the negative source:

    - ``"random"`` (default): random docs from the collection (original behavior).
    - ``"hard"``: BM25 top-ranked non-positives from ``bm25_run`` within
      ``rank_window`` (``hard_frac`` forced to 1.0).
    - ``"mixed"``: ``hard_frac`` from BM25, the rest random.

    ``bm25_run`` is the path to a TREC run file, required for hard/mixed.
    ``rank_window`` is an inclusive 1-indexed (lo, hi) BM25 rank range. See
    docs/score-skew-hypotheses.md (H5-fix) for why hard negatives and these
    defaults (50-2000: enough supply for hard_frac<=0.5 across all topics while
    excluding the noisiest rank tail).
    """
    if negatives not in ("random", "hard", "mixed"):
        raise ValueError(f"Unknown negatives mode: {negatives!r}")
    if negatives in ("hard", "mixed") and not bm25_run:
        raise ValueError(f"negatives={negatives!r} requires bm25_run")

    topics = _read_topics(topics_path)

    doc_topics = collections.defaultdict(set)
    topic_docs = collections.defaultdict(set)
    all_rel = set()
    narratives = {}

    for topic in topics:
        tid = topic["tid"]
        for question in topic["questions"]:
            for docid in question["rel_docs"]:
                topic_docs[tid].add(docid)
                doc_topics[docid].add(tid)
                all_rel.add(docid)
        narratives[tid] = _narrative_for(topic)

    # Train needs the full collection so negatives can be sampled from any doc.
    docs = {}
    _load_collection(coll_path, docs)
    all_docs = list(docs.keys())

    rows = []
    for topic in tqdm(topic_docs, desc="Building train data (positives)"):
        for docid in topic_docs[topic]:
            rows.append({
                "query_id": topic,
                "doc_id": docid,
                "query": narratives[topic],
                "doc": docs[docid],
                "relevance": 1,
            })

    rng = random.Random(seed) if seed is not None else random

    if negatives == "random":
        neg_iter = sample_negatives(
            positives=all_rel,
            all_doc_ids=all_docs,
            topic_docs=topic_docs,
            doc_topics=doc_topics,
            docs=docs,
            narratives=narratives,
            neg_pos_ratio=neg_pos_ratio,
            rng=rng,
        )
    else:
        run = read_trec_run(bm25_run)
        neg_iter = sample_hard_negatives(
            positives=all_rel,
            topic_docs=topic_docs,
            doc_topics=doc_topics,
            docs=docs,
            narratives=narratives,
            bm25_run=run,
            all_doc_ids=all_docs,
            neg_pos_ratio=neg_pos_ratio,
            hard_frac=1.0 if negatives == "hard" else hard_frac,
            rank_window=rank_window,
            rng=rng,
            log=log,
        )

    rows.extend(tqdm(neg_iter, desc=f"Adding negatives ({negatives})"))

    return rows_to_parquet(rows, output_path)


def build_predict(*, topics_path, pool_path, coll_path, output_path):
    """Build the CDet prediction parquet from a retrieval pool.

    Every doc in the pool becomes an unlabeled (relevance = -1) pair for its
    topic. Pool lines are whitespace-separated:
    ``topic date docid qid rank sim runtag``.
    """
    topics = _read_topics(topics_path)
    narratives = {topic["tid"]: _narrative_for(topic) for topic in topics}

    docs = {}
    with open(pool_path, "r") as f:
        for line in tqdm(f, desc="Loading pool docids"):
            topic, date, docid, qid, rank, sim, runtag = line.strip().split()
            docs[docid] = None

    _load_collection(coll_path, docs)

    rows = []
    with open(pool_path, "r") as f:
        for line in tqdm(f, desc="Building predict data"):
            topic, date, docid, qid, rank, sim, runtag = line.strip().split()
            rows.append({
                "query_id": topic,
                "doc_id": docid,
                "query": narratives[topic],
                "doc": docs[docid],
                "relevance": UNLABELED,
            })

    return rows_to_parquet(rows, output_path)
