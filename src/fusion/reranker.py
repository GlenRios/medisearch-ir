"""
reranker.py
-----------
Re-ranks a list of candidate chunks using a cross-encoder model.

Bi-encoder vs cross-encoder
----------------------------
The retrievers in M3 use a bi-encoder architecture: query and document
are encoded independently into vectors, and similarity is a fast dot
product. This is efficient for large corpora but misses fine-grained
query-document interactions.

A cross-encoder takes the query and document *concatenated* as input
and produces a single relevance score. It sees both texts at once,
allowing the model to capture subtle interactions. This is much more
accurate but also much slower — making it suitable only for re-ranking
a small set of candidates (top-20 to top-50), not the full corpus.

Typical pipeline
----------------
BM25 + FAISS → RRF top-20 → cross-encoder re-rank → final top-10

Model
-----
'cross-encoder/ms-marco-MiniLM-L-6-v2' is used by default:
  - Trained on MS MARCO passage ranking
  - Fast and lightweight (~70 MB)
  - Outputs a raw logit score (higher = more relevant)
  - No normalization needed — used only for ranking

No API key or internet connection is needed after the first download.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .rrf import FusedResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_K = 20    # number of RRF candidates to re-rank

# Module-level model cache
_MODEL_CACHE: dict[str, CrossEncoder] = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RerankedResult:
    """
    A single result after cross-encoder re-ranking.

    Attributes
    ----------
    chunk_id      : str
    ce_score      : float  raw cross-encoder logit (higher = more relevant)
    rrf_score     : float  original RRF score (kept for analysis)
    bm25_rank     : int    original BM25 rank (0 if absent)
    faiss_rank    : int    original FAISS rank (0 if absent)
    final_rank    : int    1-based position in the final re-ranked list
    title         : str
    url           : str
    category      : str
    text          : str
    doc_id        : str
    """
    chunk_id:   str
    ce_score:   float
    rrf_score:  float
    bm25_rank:  int
    faiss_rank: int
    final_rank: int
    title:      str
    url:        str
    category:   str
    text:       str
    doc_id:     str


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_reranker(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """
    Load a CrossEncoder model and cache it for the process lifetime.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ms-marco-MiniLM-L-6-v2).

    Returns
    -------
    CrossEncoder
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    logger.info(f"Loading cross-encoder model '{model_name}'...")
    model = CrossEncoder(model_name)
    _MODEL_CACHE[model_name] = model
    logger.info(f"Cross-encoder '{model_name}' loaded.")
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str,
           candidates: list[FusedResult],
           model: CrossEncoder,
           top_k: int = DEFAULT_RERANK_K) -> list[RerankedResult]:
    """
    Re-rank a list of RRF candidates using a cross-encoder model.

    Only the top `top_k` candidates are scored — the rest are discarded.
    This keeps inference time bounded regardless of how many candidates
    RRF produces.

    Parameters
    ----------
    query      : str              raw or cleaned query text
    candidates : list[FusedResult] RRF-fused candidates, sorted by rrf_score
    model      : CrossEncoder     loaded cross-encoder model
    top_k      : int              max candidates to re-rank (default 20)

    Returns
    -------
    list[RerankedResult]
        Candidates re-ranked by cross-encoder score, best first.
        Length = min(top_k, len(candidates)).
    """
    if not candidates:
        logger.warning("rerank() called with empty candidates list.")
        return []

    pool = candidates[:top_k]

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c.text) for c in pool]

    logger.info(f"Cross-encoder scoring {len(pairs)} candidates...")
    scores = model.predict(pairs, show_progress_bar=False)   # shape (N,)
    logger.info("Cross-encoder scoring complete.")

    # Attach scores to candidates and sort
    scored = sorted(
        zip(pool, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for final_rank, (candidate, ce_score) in enumerate(scored, start=1):
        results.append(RerankedResult(
            chunk_id   = candidate.chunk_id,
            ce_score   = float(ce_score),
            rrf_score  = candidate.rrf_score,
            bm25_rank  = candidate.bm25_rank,
            faiss_rank = candidate.faiss_rank,
            final_rank = final_rank,
            title      = candidate.title,
            url        = candidate.url,
            category   = candidate.category,
            text       = candidate.text,
            doc_id     = candidate.doc_id,
        ))

    return results