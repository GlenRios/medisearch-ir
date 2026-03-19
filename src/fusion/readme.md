# Module 4 — Fusion & Re-ranking

## Overview

Module 4 takes the two independent ranked lists produced by M3 and combines them into a single, higher-quality ranking. It operates in two sequential steps: **Reciprocal Rank Fusion (RRF)** merges the BM25 and FAISS rankings using only their rank positions, and a **cross-encoder reranker** then re-scores the top candidates by evaluating the query and each passage together.

---

## Location

```
src/fusion/
├── __init__.py
├── __main__.py
├── rrf.py
├── reranker.py
└── pipeline.py
```

---

## Pipeline

```
RetrievalResult (from M3)
        │
        ├── bm25_results  (list of 20 ranked dicts)
        └── faiss_results (list of 20 ranked dicts)
                │
                ▼
┌───────────────────────────┐
│          rrf.py           │
│  Reciprocal Rank Fusion   │
│  k = 60                   │
│  RRF score = Σ 1/(k+rank) │
└─────────────┬─────────────┘
              │  ~30-40 unique candidates, sorted by RRF score
              ▼
┌───────────────────────────┐
│        reranker.py        │
│  Cross-encoder scoring    │
│  top-20 candidates        │
│  ms-marco-MiniLM-L-6-v2  │
└─────────────┬─────────────┘
              │  top-10 final results, sorted by CE score
              ▼
         FusionResult
```

---

## Files

### `rrf.py`

Implements Reciprocal Rank Fusion to combine two ranked lists into one without requiring score normalization.

**Algorithm:**

For each document `d` that appears in at least one ranking:

```
RRF_score(d) = 1 / (k + rank_BM25(d))  +  1 / (k + rank_FAISS(d))
```

Where:
- `k = 60` is a smoothing constant that prevents very high-ranked documents from dominating
- `rank_i(d)` is the 1-based position of document `d` in ranking `i`
- Documents absent from a ranking receive `rank = len(ranking) + 1` (worst possible rank)

**Why RRF works without score normalization:**
BM25 scores (e.g. `18.4`) and FAISS cosine similarities (e.g. `0.89`) are on completely different scales and cannot be combined by simple addition or weighting. RRF discards the raw scores entirely and uses only the rank positions — an ordinal scale that is consistent across both systems.

**Key function:**

```python
reciprocal_rank_fusion(
    bm25_results:  list[dict],
    faiss_results: list[dict],
    k: int = 60
) -> list[FusedResult]
```

**`FusedResult` dataclass:**

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique chunk identifier |
| `rrf_score` | float | Combined RRF score (higher = more relevant) |
| `bm25_rank` | int | Rank in BM25 list (0 if absent) |
| `faiss_rank` | int | Rank in FAISS list (0 if absent) |
| `title` | str | Article title |
| `url` | str | Wikipedia URL |
| `category` | str | Wikipedia category |
| `text` | str | Raw chunk text |
| `doc_id` | str | Parent article ID |

**Example score calculation** (k=60):

| Document | BM25 rank | FAISS rank | RRF score |
|---|---|---|---|
| Hypertension c003 | 1 | 2 | 1/61 + 1/62 = 0.03233 |
| Hypertension c007 | 3 | 1 | 1/63 + 1/61 = 0.03222 |
| Blood pressure c001 | 2 | 5 | 1/62 + 1/65 = 0.03150 |
| Kidney disease c002 | — | 3 | 1/21 + 1/63 = 0.06349 ← penalty rank |

---

### `reranker.py`

Re-ranks the top RRF candidates using a cross-encoder model that scores each `(query, passage)` pair jointly.

**Bi-encoder vs cross-encoder:**

| | Bi-encoder (M2/M3) | Cross-encoder (M4) |
|---|---|---|
| How it works | Encodes query and document independently into vectors | Concatenates query + document and scores jointly |
| Speed | Very fast (dot product) | Slow (full inference per pair) |
| Accuracy | Good | Better — sees both texts at once |
| Use case | Full corpus retrieval | Re-ranking a small candidate set |

The cross-encoder is applied only to the top-20 RRF candidates, keeping inference time bounded regardless of corpus size.

**Default model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained on MS MARCO passage ranking benchmark
- ~70 MB, fast inference on CPU
- Outputs raw logit scores (higher = more relevant)
- No API key required — runs fully locally

**Key functions:**

```python
load_reranker(model_name: str) -> CrossEncoder
```

Loads and caches the cross-encoder model for the process lifetime.

```python
rerank(
    query:      str,
    candidates: list[FusedResult],
    model:      CrossEncoder,
    top_k:      int = 20
) -> list[RerankedResult]
```

**`RerankedResult` dataclass:**

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | |
| `ce_score` | float | Raw cross-encoder logit (higher = more relevant) |
| `rrf_score` | float | Original RRF score (kept for analysis) |
| `bm25_rank` | int | Original BM25 rank (0 if absent) |
| `faiss_rank` | int | Original FAISS rank (0 if absent) |
| `final_rank` | int | 1-based final position after re-ranking |
| `title` | str | |
| `url` | str | |
| `category` | str | |
| `text` | str | |
| `doc_id` | str | |

---

### `pipeline.py`

Orchestrates RRF fusion followed by cross-encoder re-ranking.

**`FusionPipeline` class:**

```python
fusion = FusionPipeline()
result = fusion.fuse(retrieval_result)
```

Loads the cross-encoder once at initialization. Subsequent calls to `fuse()` reuse the loaded model.

**Configuration:**

| Constant | Default | Description |
|---|---|---|
| `RRF_K_VALUE` | `60` | RRF smoothing constant |
| `RERANK_TOP_K` | `20` | Candidates passed to cross-encoder |
| `FINAL_TOP_K` | `10` | Final results returned |

**`FusionResult` dataclass:**

| Field | Type | Description |
|---|---|---|
| `query_raw` | str | Original raw query |
| `rrf_results` | list[FusedResult] | All RRF-fused candidates |
| `final_results` | list[RerankedResult] | Top-10 after cross-encoder |

**Interactive CLI:**

```bash
python -m src.fusion
```

Runs the full hybrid pipeline end-to-end and prints final results with full traceability: BM25 rank, FAISS rank, RRF score, and CE score for each result.

```
── Final results for: 'symptoms of hypertension' ──────────
  [1] Hypertension
       CE score : 8.4231  |  RRF: 0.031746  |  BM25#1  FAISS#2
       High blood pressure symptoms include headaches...

  [2] Blood pressure
       CE score : 6.1823  |  RRF: 0.028571  |  BM25#3  FAISS#1
       Elevated systolic pressure is associated with...
```

---

## How to run

```bash
# Interactive CLI (full hybrid pipeline)
python -m src.fusion

# In code
from src.retrieval.pipeline import RetrievalPipeline
from src.fusion.pipeline    import FusionPipeline

retrieval = RetrievalPipeline()
fusion    = FusionPipeline()

ret    = retrieval.search("causes of type 2 diabetes")
result = fusion.fuse(ret)

for r in result.final_results:
    print(r.final_rank, r.title, r.ce_score)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Cross-encoder model loading and inference |
| `rank-bm25` | Indirectly (via M3 results) |

---

## Design decisions

**Why k=60 in RRF?**
The value k=60 was empirically shown to be optimal across many IR benchmarks in the original RRF paper (Cormack et al., SIGIR 2009). It provides enough smoothing so that a single very high-ranked document in one list does not dominate the fused score.

**Why apply the cross-encoder only to top-20 and not all candidates?**
Cross-encoder inference requires a full forward pass for each `(query, passage)` pair — it cannot be batched efficiently for large candidate sets. Applying it to all 40 unique RRF candidates would take several seconds on CPU. Limiting to top-20 keeps latency acceptable while still covering the candidates most likely to be relevant.

**Why keep `rrf_score` and `bm25_rank`/`faiss_rank` in the final result?**
Traceability. Knowing that a result ranked #1 in both BM25 and FAISS but was re-ranked to #3 by the cross-encoder is useful information for analysis and debugging. The Streamlit interface in the `app/` module uses these fields to display the full ranking comparison table.

**Why not normalize BM25 and FAISS scores before combining?**
Score normalization (e.g. min-max) is sensitive to the score distribution of each query — the same document gets a different normalized score depending on what other documents were retrieved. RRF's rank-based approach is robust to this issue and consistently outperforms score-combination methods in practice.

---

## References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*. SIGIR 2009.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.

---

## Related modules

| Module | Relation | Description |
|---|---|---|
| M3 | M4 depends on M3 | Receives `RetrievalResult` as input |
| app/ | Depends on M4 | Calls `FusionPipeline.fuse()` for every search query |