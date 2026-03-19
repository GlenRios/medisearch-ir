"""
rrf.py
------
Implements Reciprocal Rank Fusion (RRF) to combine two ranked lists
into a single unified ranking.

Algorithm
---------
For each document d across all input rankings:

    RRF_score(d) = Σ  1 / (k + rank_i(d))
                   i

Where:
  - rank_i(d) is the 1-based position of document d in ranking i
  - k is a smoothing constant (default 60, from the original paper)
  - Documents absent from a ranking receive rank = len(ranking) + 1

Key properties
--------------
- Requires only rank positions, not raw scores.
  BM25 scores and FAISS cosine similarities are on different scales
  and cannot be combined directly. RRF sidesteps this entirely.
- Robust to outliers — a single very high score in one list cannot
  dominate the fusion result.
- Parameter k=60 was shown to be optimal across many IR benchmarks
  in the original paper (Cormack et al., SIGIR 2009).

Reference
---------
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
Reciprocal rank fusion outperforms condorcet and individual rank
learning methods. SIGIR 2009.
"""

from dataclasses import dataclass

RRF_K = 60   # smoothing constant from the original paper


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FusedResult:
    """
    A single result after RRF fusion.

    Attributes
    ----------
    chunk_id   : str    unique chunk identifier
    rrf_score  : float  combined RRF score (higher = more relevant)
    bm25_rank  : int    rank in BM25 list (0 if absent)
    faiss_rank : int    rank in FAISS list (0 if absent)
    title      : str
    url        : str
    category   : str
    text       : str
    doc_id     : str
    """
    chunk_id:   str
    rrf_score:  float
    bm25_rank:  int
    faiss_rank: int
    title:      str
    url:        str
    category:   str
    text:       str
    doc_id:     str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(bm25_results: list[dict],
                           faiss_results: list[dict],
                           k: int = RRF_K) -> list[FusedResult]:
    """
    Fuse two ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    bm25_results  : list[dict]   ranked BM25 results from M3
                                 each dict must have: chunk_id, rank, title,
                                 url, category, text, doc_id
    faiss_results : list[dict]   ranked FAISS results from M3
    k             : int          smoothing constant (default 60)

    Returns
    -------
    list[FusedResult]
        Combined ranking sorted by RRF score descending.
        Includes all documents that appeared in at least one list.
    """
    # Build rank lookup dicts: chunk_id → rank (1-based)
    bm25_ranks  = {r["chunk_id"]: r["rank"] for r in bm25_results}
    faiss_ranks = {r["chunk_id"]: r["rank"] for r in faiss_results}

    # Build metadata lookup from both lists
    meta = {}
    for r in bm25_results + faiss_results:
        if r["chunk_id"] not in meta:
            meta[r["chunk_id"]] = {
                "title":    r.get("title",    ""),
                "url":      r.get("url",      ""),
                "category": r.get("category", ""),
                "text":     r.get("text",     ""),
                "doc_id":   r.get("doc_id",   ""),
            }

    # Penalty rank for documents absent from a list
    worst_bm25  = len(bm25_results)  + 1
    worst_faiss = len(faiss_results) + 1

    # Compute RRF score for every unique chunk_id
    all_ids = set(bm25_ranks) | set(faiss_ranks)
    fused   = []

    for chunk_id in all_ids:
        r_bm25  = bm25_ranks.get(chunk_id,  worst_bm25)
        r_faiss = faiss_ranks.get(chunk_id, worst_faiss)

        rrf_score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_faiss))

        fused.append(FusedResult(
            chunk_id   = chunk_id,
            rrf_score  = rrf_score,
            bm25_rank  = bm25_ranks.get(chunk_id,  0),
            faiss_rank = faiss_ranks.get(chunk_id, 0),
            **meta[chunk_id],
        ))

    # Sort by RRF score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    return fused