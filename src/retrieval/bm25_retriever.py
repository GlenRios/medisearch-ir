"""
bm25_retriever.py
-----------------
Executes a lemmatized query against the BM25 index and returns
a ranked list of (chunk_id, score) pairs.

BM25 scores are not normalized — they are raw floating point values
whose magnitude depends on corpus size and query length. They are
meaningful for ranking within a single retrieval run but should not
be compared directly against FAISS similarity scores.

The fusion module (M4) uses only the rank positions, not the raw
scores, so normalization is not required here.
"""

import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.bm25_indexer import load_bm25, load_chunk_map

logger = logging.getLogger(__name__)

# Default artifact paths (relative to project root)
_ROOT          = Path(__file__).parent.parent.parent
BM25_PATH      = _ROOT / "indexes" / "bm25.pkl"
CHUNK_MAP_PATH = _ROOT / "indexes" / "chunk_map.json"


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Wraps a loaded BM25 index and exposes a simple search interface.

    Attributes
    ----------
    index      : BM25Okapi   the fitted BM25 index
    chunk_ids  : list[str]   ordered chunk IDs matching index positions
    chunk_map  : dict        chunk_id → {text, title, url, category, doc_id}
    """

    def __init__(self,
                 bm25_path: Path = BM25_PATH,
                 chunk_map_path: Path = CHUNK_MAP_PATH) -> None:
        """
        Load the BM25 index and chunk map from disk.

        Parameters
        ----------
        bm25_path      : Path — path to bm25.pkl
        chunk_map_path : Path — path to chunk_map.json
        """
        self.index     = load_bm25(bm25_path)
        self.chunk_map = load_chunk_map(chunk_map_path)
        self.chunk_ids = list(self.chunk_map.keys())
        logger.info(f"BM25Retriever ready. Corpus size: {len(self.chunk_ids)} chunks.")

    def search(self,
               tokens: list[str],
               top_k: int = 20) -> list[dict]:
        """
        Query the BM25 index with a list of lemmatized tokens.

        Parameters
        ----------
        tokens : list[str]   lemmatized query tokens from query_processor
        top_k  : int         number of results to return (default 20)

        Returns
        -------
        list[dict]
            Ranked list of result dicts, best score first. Each dict:
            {
                "chunk_id" : str,
                "score"    : float,   BM25 raw score
                "rank"     : int,     1-based rank position
                "title"    : str,
                "url"      : str,
                "category" : str,
                "text"     : str,
                "doc_id"   : str,
            }
        """
        if not tokens:
            logger.warning("BM25 search called with empty token list.")
            return []

        scores = self.index.get_scores(tokens)            # shape (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]   # descending

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk_id = self.chunk_ids[idx]
            score    = float(scores[idx])

            if score <= 0:
                break                   # remaining scores are zero — no match

            meta = self.chunk_map.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "score":    score,
                "rank":     rank,
                "title":    meta.get("title",    ""),
                "url":      meta.get("url",      ""),
                "category": meta.get("category", ""),
                "text":     meta.get("text",     ""),
                "doc_id":   meta.get("doc_id",   ""),
            })

        logger.debug(f"BM25 returned {len(results)} results for tokens={tokens[:5]}...")
        return results