---
noteId: "db20733023d111f190113db5f1ba83e8"
tags: []

---

# Module 3 — Retrieval

## Overview

Module 3 receives a raw natural language query, preprocesses it, and executes it in parallel against both indexes built in M2. It returns two independent ranked lists — one from BM25 (lexical) and one from FAISS HNSW (semantic) — which are passed to M4 for fusion.

---

## Location

```
src/retrieval/
├── __init__.py
├── __main__.py
├── query_processor.py
├── bm25_retriever.py
├── faiss_retriever.py
└── pipeline.py
```

---

## Pipeline

```
raw query (str)
        │
        ▼
┌────────────────────┐
│ query_processor.py │  clean text + lemmatize (same pipeline as M1)
└─────────┬──────────┘
          │
          ├──────────────────────────────┐
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│ bm25_retriever   │          │  faiss_retriever      │
│ BM25 top-20      │          │  HNSW top-20          │
│ (lexical)        │          │  (semantic)           │
└────────┬─────────┘          └──────────┬────────────┘
         │                               │
         ▼                               ▼
  list[(chunk_id, score)]      list[(chunk_id, score)]
         │                               │
         └───────────────────────────────┘
                         │
                         ▼
                  RetrievalResult
              (passed to M4 fusion)
```

---

## Files

### `query_processor.py`

Preprocesses the raw user query before it is sent to either retriever.

The query must go through **the exact same cleaning and lemmatization pipeline as the corpus documents in M1**. If query tokens and document tokens are processed differently, BM25 term matching breaks — a query token like `"running"` would never match a document token `"run"` unless both are lemmatized.

**Key function:**

```python
process_query(query: str, model_name: str = "en_core_web_sm") -> ProcessedQuery
```

**`ProcessedQuery` dataclass:**

| Field | Type | Description |
|---|---|---|
| `raw` | str | Original query as typed by the user |
| `clean` | str | Cleaned query text — used for embedding generation |
| `tokens` | list[str] | Lemmatized tokens — used for BM25 scoring |

**Example:**

```
input  : "What are the symptoms of high blood pressure?"
clean  : "What are the symptoms of high blood pressure?"
tokens : ["symptom", "high", "blood", "pressure"]
```

Internally reuses `clean_text()` from `src/ingestion/cleaner.py` and `lemmatize()` from `src/ingestion/lemmatizer.py` — no duplication.

---

### `bm25_retriever.py`

Executes a lemmatized query against the BM25 index.

**`BM25Retriever` class:**

```python
retriever = BM25Retriever()
results   = retriever.search(tokens=["symptom", "blood", "pressure"], top_k=20)
```

**`search()` method:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tokens` | list[str] | required | Lemmatized query tokens from `query_processor` |
| `top_k` | int | 20 | Number of results to return |

Internally calls `bm25.get_scores(tokens)`, ranks all chunks by score descending, and discards chunks with score ≤ 0 (no token overlap with the query).

**Result dict schema:**

```python
{
    "chunk_id": "wiki_00001_c003",
    "score":    18.42,            # BM25 raw score (not normalized)
    "rank":     1,                # 1-based position
    "title":    "Hypertension",
    "url":      "https://en.wikipedia.org/wiki/Hypertension",
    "category": "Cardiology",
    "text":     "The kidneys are responsible for...",
    "doc_id":   "wiki_00001"
}
```

**Note on scores:** BM25 scores are raw floating-point values whose magnitude depends on corpus size and query length. They are meaningful for within-run ranking but cannot be compared directly against FAISS cosine similarity scores. M4 uses only rank positions, not raw scores.

---

### `faiss_retriever.py`

Encodes the cleaned query with the same sentence-transformer model used in M2 and searches the HNSW index.

**`FAISSRetriever` class:**

```python
retriever = FAISSRetriever()
results   = retriever.search(query_text="symptoms of hypertension", top_k=20)
```

**`search()` method:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | str | required | Cleaned query text from `query_processor` |
| `top_k` | int | 20 | Number of results to return |

Internally encodes the query with `generate_embeddings()` from `src/indexing/embedder.py` (same model, L2-normalized), then calls `search_faiss()` which returns L2 distances converted to cosine similarity scores in `[0, 1]`.

**Result dict schema:** same structure as BM25, with `score` representing cosine similarity (`1.0` = identical, `0.0` = orthogonal).

---

### `pipeline.py`

Orchestrates both retrievers and exposes a single `search()` method.

**`RetrievalPipeline` class:**

```python
pipeline = RetrievalPipeline(top_k=20)
result   = pipeline.search("symptoms of hypertension")
```

Loads both retrievers once at initialization — subsequent calls to `search()` reuse the loaded indexes without reloading from disk.

**`RetrievalResult` dataclass:**

| Field | Type | Description |
|---|---|---|
| `query` | ProcessedQuery | Preprocessed query object |
| `bm25_results` | list[dict] | Top-K BM25 results |
| `faiss_results` | list[dict] | Top-K FAISS results |

**Interactive CLI:**

```bash
python -m src.retrieval
```

Launches a prompt where you can type queries and see the top-5 BM25 and FAISS results side by side.

---

## Output schema

Both `bm25_results` and `faiss_results` are lists of dicts with this structure:

```json
{
  "chunk_id": "wiki_00001_c003",
  "score":    18.42,
  "rank":     1,
  "title":    "Hypertension",
  "url":      "https://en.wikipedia.org/wiki/Hypertension",
  "category": "Cardiology",
  "text":     "The kidneys are responsible for long-term blood pressure...",
  "doc_id":   "wiki_00001"
}
```

The only difference between the two lists is the meaning of `score`:
- BM25: raw BM25 score (higher = more lexically relevant)
- FAISS: cosine similarity in `[0, 1]` (higher = more semantically similar)

---

## How to run

```bash
# Interactive CLI
python -m src.retrieval

# In code
from src.retrieval.pipeline import RetrievalPipeline

pipeline = RetrievalPipeline(top_k=20)
result   = pipeline.search("causes of type 2 diabetes")

print(result.bm25_results[0])
print(result.faiss_results[0])
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `rank-bm25` | BM25 index loading and scoring |
| `faiss-cpu` | HNSW index loading and search |
| `sentence-transformers` | Query embedding generation |
| `spacy` | Query lemmatization (via M1 modules) |

All artifacts loaded by M3 (`bm25.pkl`, `faiss.index`, `faiss_ids.json`, `chunk_map.json`) are produced by M2.

---

## Design decisions

**Why reuse M1's cleaning and lemmatization for the query?**
Consistency between query and document processing is a hard requirement for BM25. If documents are lemmatized but queries are not, a query for `"running"` would never match a document token `"run"`. By importing `clean_text()` and `lemmatize()` directly from `src/ingestion/`, the query is guaranteed to follow the identical transformation path.

**Why use the cleaned text for FAISS and the token list for BM25?**
The sentence-transformer model handles its own internal tokenization — it expects raw text, not a pre-tokenized list. Passing lemmatized tokens to the embedding model would produce degraded representations since the model was trained on natural language. BM25, on the other hand, requires discrete tokens to perform term matching.

**Why load both retrievers at pipeline initialization?**
Loading the BM25 index (pickle) and the FAISS HNSW index takes a few seconds. Doing this once at startup avoids penalizing every query call. The `RetrievalPipeline` object is designed to be long-lived — instantiate it once and call `search()` repeatedly.

---

## Related modules

| Module | Relation | Description |
|---|---|---|
| M1 | M3 imports from M1 | `cleaner.py` and `lemmatizer.py` used for query preprocessing |
| M2 | M3 depends on M2 | Loads all index artifacts from `indexes/` |
| M4 | M4 depends on M3 | Receives `RetrievalResult` as input to fusion |