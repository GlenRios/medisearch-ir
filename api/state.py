"""
state.py
--------
Manages Streamlit session state and lazy-loads the retrieval
and fusion pipelines so they are initialized only once per session.
"""

import json
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent
CHUNKS_FILE  = _ROOT / "data" / "processed" / "chunks.json"
STATS_FILE   = _ROOT / "data" / "processed" / "ingestion_stats.json"
INDEX_STATS  = _ROOT / "indexes" / "index_stats.json"


# ---------------------------------------------------------------------------
# Pipeline loaders (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retrieval pipeline...")
def get_retrieval_pipeline():
    from src.retrieval.pipeline import RetrievalPipeline
    return RetrievalPipeline(top_k=20)


@st.cache_resource(show_spinner="Loading fusion pipeline...")
def get_fusion_pipeline():
    from src.fusion.pipeline import FusionPipeline
    return FusionPipeline()


# ---------------------------------------------------------------------------
# Corpus stats (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_corpus_stats() -> dict:
    stats = {}

    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            stats.update(json.load(f))

    if INDEX_STATS.exists():
        with open(INDEX_STATS, "r", encoding="utf-8") as f:
            stats.update(json.load(f))

    return stats


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

def run_search(query: str) -> dict | None:
    """
    Execute the full hybrid pipeline for a query.
    Returns a dict with bm25, faiss, and hybrid results, or None on error.
    """
    try:
        retrieval = get_retrieval_pipeline()
        fusion    = get_fusion_pipeline()

        ret_result    = retrieval.search(query)
        fusion_result = fusion.fuse(ret_result)

        return {
            "query":          query,
            "bm25_results":   ret_result.bm25_results,
            "faiss_results":  ret_result.faiss_results,
            "hybrid_results": [
                {
                    "chunk_id":   r.chunk_id,
                    "title":      r.title,
                    "url":        r.url,
                    "category":   r.category,
                    "text":       r.text,
                    "ce_score":   r.ce_score,
                    "rrf_score":  r.rrf_score,
                    "bm25_rank":  r.bm25_rank,
                    "faiss_rank": r.faiss_rank,
                    "final_rank": r.final_rank,
                }
                for r in fusion_result.final_results
            ],
        }

    except ValueError as e:
        st.error(f"Query error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.exception(e)
        return None