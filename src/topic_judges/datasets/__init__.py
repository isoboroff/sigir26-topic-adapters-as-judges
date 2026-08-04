"""Dataset builders.

Every builder emits the same parquet schema::

    query_id: str   doc_id: str   query: str   doc: str   relevance: int

Builders come in two kinds (design sec 3a.2):

- **judged** (``robust04``, ``trecdl``): labels come from explicit qrels, which
  contain both positive and negative examples.
- **positive-only + sampling** (``cdet``): only positive rel_docs are given per
  topic, so negatives are synthesized by random sampling from the collection.

``common`` holds the pieces shared across both kinds (schema constants, the
rows->parquet writer, and the negative sampler).
"""
