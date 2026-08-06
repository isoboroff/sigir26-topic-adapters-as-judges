# Understanding the Bi-Encoder / SentenceTransformer Change

A layered explainer for the ModernBERT bi-encoder work. **Layer 1** is
conceptual (no code); **Layer 2** anchors those concepts to the exact mechanics
we're adding to `models.py`; **Layer 3** points to deeper topics if you want to
follow the threads further.

You do not need Layer 2/3 to follow the design discussion — read Layer 1, and
dip into the rest when a specific line of code puzzles you.

---

## Layer 1 — Concepts

### What a relevance judge model does here

Every model in this project turns a (query, document) pair into a number in
[0, 1]: "how relevant is this doc to this query." The three archetypes differ
in *how* they combine query and doc to get that number.

### Cross-encoder / mono-decoder vs. bi-encoder

- **Cross-encoder & mono-decoder** glue the query and document into **one**
  sequence (`[query] [doc]`) and feed it through the model together, so every
  query word can attend to every doc word. Powerful, but the query and doc
  **share one length budget** — monoT5's is 512 tokens. That's the root of our
  problem: a 333-token narrative query + a long news article blows past 512, and
  ~74% of the document gets truncated away (see H4 in
  `score-skew-hypotheses.md`).

- **Bi-encoder** encodes the query and the document **separately** into two
  fixed-length vectors ("embeddings"), then scores them by how close those
  vectors are (cosine similarity). Because they're encoded independently, **each
  side gets its own length budget** — the whole query fits *and* the whole
  document fits. That's exactly why we're switching: it structurally removes the
  truncation problem.

The bi-encoder is not a new/untested idea here — the paper evaluates it as one
of its three archetypes (using `all-MiniLM-L6-v2`).

### What "embedding" and "pooling" mean

A transformer reads a sequence of tokens and emits **one vector per token**
(the "last hidden state"). But a bi-encoder needs **one vector for the whole
text**. **Pooling** is the step that collapses the per-token vectors into a
single sentence vector. The simplest pooling is **mean pooling**: average all
the token vectors (ignoring padding). There are others (take the `[CLS]` token,
take the max), but mean is the common default.

### What a "sentence-transformer" is (and why it matters)

A plain Hugging Face model (loaded with `AutoModel`) gives you the per-token
vectors and **stops there** — it has no opinion about how to pool them into a
sentence vector, because it was trained for masked-language-modeling, not for
"are these two texts similar."

A **sentence-transformer** is that same transformer **plus a trained recipe** for
producing a *good* sentence embedding: a specific pooling step and often a
normalization step, and — crucially — the whole thing was **fine-tuned so that
similar texts land close together**. `joe32140/ModernBERT-base-msmarco` was
fine-tuned this way on MS MARCO (a ranking dataset), so its embeddings are
already tuned for query-document relevance.

### The design choice this creates

Our existing bi-encoder code does the *plain* thing: raw `AutoModel` +
mean-pooling + a **fresh, untrained** little projection layer. That's fine for
`all-MiniLM` (the paper's setup). But if we point it at
`ModernBERT-base-msmarco` that way, we'd **throw away the ranking-tuning** —
we'd ignore its trained pooling and bolt an untrained layer in front of its
good embedding.

So the change adds a second mode that loads the model **as a
sentence-transformer**, keeping its trained pooling/normalization and scoring
cosine on its native embedding. Same class, a flag picks the mode; the old mode
stays the default so nothing else changes.

### Where LoRA fits

We still want to *adapt* the model per topic. **LoRA** injects small trainable
weight-deltas into the transformer while freezing the big pretrained weights —
cheap, and each topic gets its own adapter. It attaches to the transformer part
in **both** modes; the sentence-transformer wrapper around it doesn't change
that.

---

## Layer 2 — How that maps to the code

### The SentenceTransformer object is a pipeline of modules

Loading `SentenceTransformer("...msmarco")` gives you a small `nn.Sequential`
of modules. For these models it is literally:

```
0  Transformer   # the HF model; produces per-token vectors
1  Pooling       # collapses per-token -> one sentence vector (the trained recipe)
2  Normalize     # scales the vector to unit length
```

Calling the whole object on a batch runs `0 -> 1 -> 2` and hands back a
`sentence_embedding`. This is the "trained pooling + normalization" from Layer 1,
made concrete: modules 1 and 2 are exactly what the plain-`AutoModel` path
lacks.

### Reaching the inner transformer

- `st_model[0]` is the `Transformer` module.
- `st_model[0].auto_model` is the underlying Hugging Face transformer — the same
  kind of object the old path gets from `AutoModel.from_pretrained(...)`.

That `auto_model` is what we (a) wrap with LoRA and (b) save/load as the
adapter, so the rest of the codebase's `model.model` convention still holds.

### Feeding our own tokenized inputs

The data module already tokenizes queries and docs into `input_ids` /
`attention_mask`. The ST modules consume a **features dict** and add to it:

```
{"input_ids", "attention_mask"}  --run through the pipeline-->  {..., "sentence_embedding"}
```

So `encode_text` in the ST mode just builds that dict, calls the ST model, and
reads back `["sentence_embedding"]` — instead of doing manual mean-pooling. The
rest (cosine of query vs. doc embedding, mapped to [0, 1]) is unchanged.

### Why LoRA still works through the wrapper

`get_peft_model(auto_model, ...)` returns a LoRA-wrapped transformer; we set it
back as `st_model[0].auto_model` so the pipeline uses the adapted transformer.
Training updates only the LoRA deltas; pooling/normalize have no trained params
to disturb. Saving/loading uses the same adapter path as the other archetypes.

### The one-line summary of the diff

The default (`sentence_transformer=False`) is byte-for-byte the old bi-encoder.
The new branch swaps "AutoModel + hand-rolled mean pool + fresh projection" for
"SentenceTransformer's own transformer + trained pooling + normalize," and LoRA
attaches to the inner transformer either way.

---

## Layer 3 — If you want to go deeper

- **The three archetypes & the mono true/false head** — see `models.py`
  (`BiEncoderLoRA`, `CrossEncoderLoRA`, `MonoDecoderLoRA`) and the paper
  (`docs/sigir26-paper.pdf`, Table 1 / §3–4).
- **Why the skew led us here** — `docs/score-skew-hypotheses.md` (H1 confirmed
  the skew predates training; H4 measured the truncation that this change fixes).
- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*.
  In this repo: the `LoraConfig` / `get_peft_model` calls in `models.py`.
- **sentence-transformers internals** — the `Transformer`, `Pooling`,
  `Normalize` module classes in the `sentence_transformers.models` package; the
  library's docs on "Creating custom models."
- **Bi- vs. cross-encoder tradeoffs** — bi-encoders are cheaper (you can embed
  every doc once and reuse it) but generally less precise than cross-encoders,
  which see query and doc jointly. The paper measures this tradeoff across
  archetypes.
- **Why the V100 caps our length** — FlashAttention-2 needs Ampere (sm_80+); the
  V100 is Volta (sm_70), so ModernBERT falls back to standard attention whose
  memory grows ~O(length²). Hence a moderate `max_length` + small batch. Noted
  in `docs/issues.md`.
