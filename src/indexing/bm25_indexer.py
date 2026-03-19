"""
bm25_indexer.py
---------------
Builds a BM25 index from the tokenized corpus and persists it to disk.

BM25 (Best Match 25) is a probabilistic ranking function that scores
documents based on term frequency (TF) and inverse document frequency
(IDF), with length normalization.

Key parameters
--------------
k1 : float (default 1.5)
    Controls term frequency saturation. Higher values increase the
    influence of repeated terms. Typical range: 1.2 – 2.0.

b : float (default 0.75)
    Controls document length normalization. 0 = no normalization,
    1 = full normalization. Default 0.75 is the standard setting.

What is stored on disk
----------------------
indexes/bm25.pkl   — pickled BM25Okapi object
indexes/chunk_map.json — mapping chunk_id → {text, title, url, category}
                         used at retrieval time to return human-readable results
"""

import json
import pickle
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_bm25(tokenized_corpus: list[list[str]],
               k1: float = 1.5,
               b: float = 0.75) -> BM25Okapi:
    """
    Build a BM25Okapi index from a list of token lists.

    Parameters
    ----------
    tokenized_corpus : list[list[str]]
        One token list per chunk, in the same order as the chunks list.
        These are the lemmatized tokens produced by M1.
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalization parameter (default 0.75).

    Returns
    -------
    BM25Okapi
        Fitted BM25 index ready for querying.
    """
    if not tokenized_corpus:
        raise ValueError("tokenized_corpus is empty — cannot build BM25 index.")

    logger.info(f"Building BM25 index over {len(tokenized_corpus)} chunks "
                f"(k1={k1}, b={b})...")

    index = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    logger.info("BM25 index built successfully.")
    return index


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save_bm25(index: BM25Okapi, output_path: Path) -> None:
    """
    Serialize the BM25 index to disk using pickle.

    Parameters
    ----------
    index       : BM25Okapi — fitted index
    output_path : Path      — destination file (e.g. indexes/bm25.pkl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)
    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info(f"BM25 index saved → {output_path} ({size_mb:.2f} MB)")


def load_bm25(index_path: Path) -> BM25Okapi:
    """
    Load a previously saved BM25 index from disk.

    Parameters
    ----------
    index_path : Path — path to the .pkl file

    Returns
    -------
    BM25Okapi
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {index_path}. "
            "Run the M2 pipeline first."
        )
    logger.info(f"Loading BM25 index from {index_path}...")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    logger.info("BM25 index loaded.")
    return index


# ---------------------------------------------------------------------------
# Chunk map
# ---------------------------------------------------------------------------

def build_chunk_map(chunks: list[dict]) -> dict:
    """
    Build a mapping from chunk_id to its display metadata.

    This map is used at retrieval time to convert a BM25 or FAISS
    result index back into human-readable information.

    Parameters
    ----------
    chunks : list[dict] — all chunks as loaded from chunks.json

    Returns
    -------
    dict
        { chunk_id: { text, title, url, category, doc_id } }
    """
    return {
        chunk["chunk_id"]: {
            "text":     chunk["text"],
            "title":    chunk["title"],
            "url":      chunk["url"],
            "category": chunk["category"],
            "doc_id":   chunk["doc_id"],
        }
        for chunk in chunks
    }


def save_chunk_map(chunk_map: dict, output_path: Path) -> None:
    """Save the chunk map as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunk map saved → {output_path} ({len(chunk_map)} entries)")


def load_chunk_map(map_path: Path) -> dict:
    """Load the chunk map from a JSON file."""
    if not map_path.exists():
        raise FileNotFoundError(
            f"Chunk map not found at {map_path}. "
            "Run the M2 pipeline first."
        )
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)