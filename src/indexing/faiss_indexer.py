"""
faiss_indexer.py
----------------
Builds a FAISS HNSW dense vector index from chunk embeddings and saves it
to disk.

Index type — IndexHNSWFlat
--------------------------
HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm:
  - No training step required (unlike IVF variants)
  - Sub-linear query time — much faster than IndexFlatIP at query time
  - Slightly higher memory footprint (stores the graph structure)
  - Approximate search: recall is controlled by the `ef_search` parameter

Key parameters
--------------
M : int (default 32)
    Number of neighbors each node connects to during graph construction.
    Higher M → better recall, slower build, more memory.
    Recommended range: 16–64. 32 is a solid default.

ef_construction : int (default 200)
    Size of the dynamic candidate list during index build.
    Higher → better graph quality, slower build.
    Recommended: 100–400.

ef_search : int (default 128)
    Size of the candidate list during query time.
    Higher → better recall, slower queries.
    Can be tuned at runtime without rebuilding the index.

Note on similarity
------------------
IndexHNSWFlat uses L2 distance internally. Since embeddings are
L2-normalized in embedder.py, L2 distance and cosine similarity are
monotonically related: minimizing L2 distance is equivalent to maximizing
cosine similarity. Results are returned as L2 distances (lower = more
similar) and converted to a similarity score as: score = 1 - (dist / 2).

What is stored on disk
----------------------
indexes/faiss.index      — the HNSW index binary file
indexes/faiss_ids.json   — ordered list of chunk_ids matching index positions
                           (FAISS uses integer positions, not string ids)
"""

import json
import logging
from pathlib import Path

import numpy as np
import faiss

logger = logging.getLogger(__name__)

# HNSW parameters
HNSW_M               = 32    # graph connectivity
HNSW_EF_CONSTRUCTION = 200   # build-time search depth
HNSW_EF_SEARCH       = 128   # query-time search depth


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_faiss(embeddings: np.ndarray,
                m: int = HNSW_M,
                ef_construction: int = HNSW_EF_CONSTRUCTION,
                ef_search: int = HNSW_EF_SEARCH) -> faiss.IndexHNSWFlat:
    """
    Build a FAISS IndexHNSWFlat from a matrix of L2-normalized embeddings.

    Parameters
    ----------
    embeddings      : np.ndarray — shape (N, D) float32, L2-normalized
    m               : int        — HNSW graph connectivity (default 32)
    ef_construction : int        — build-time search depth (default 200)
    ef_search       : int        — query-time search depth (default 128)

    Returns
    -------
    faiss.IndexHNSWFlat
        Populated HNSW index ready for querying.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (N, D), got shape {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n_vectors, dim = embeddings.shape
    logger.info(
        f"Building FAISS IndexHNSWFlat — {n_vectors} vectors, dim={dim}, "
        f"M={m}, ef_construction={ef_construction}..."
    )

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch        = ef_search
    index.add(embeddings)

    logger.info(f"HNSW index built. Total vectors: {index.ntotal}")
    return index


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save_faiss(index: faiss.IndexHNSWFlat,
               chunk_ids: list[str],
               index_path: Path,
               ids_path: Path) -> None:
    """
    Write the HNSW index and its chunk ID list to disk.

    Parameters
    ----------
    index      : faiss.IndexHNSWFlat — populated index
    chunk_ids  : list[str]           — chunk_id per vector position
    index_path : Path                — destination .index file
    ids_path   : Path                — destination chunk_ids JSON
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / (1024 ** 2)
    logger.info(f"HNSW index saved → {index_path} ({size_mb:.2f} MB)")

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    logger.info(f"FAISS chunk IDs saved → {ids_path} ({len(chunk_ids)} entries)")


def load_faiss(index_path: Path,
               ids_path: Path) -> tuple[faiss.IndexHNSWFlat, list[str]]:
    """
    Load a HNSW index and its chunk ID list from disk.

    Parameters
    ----------
    index_path : Path — path to the .index binary file
    ids_path   : Path — path to the chunk_ids JSON

    Returns
    -------
    tuple[faiss.IndexHNSWFlat, list[str]]
        (loaded index, ordered list of chunk_ids)
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the M2 pipeline first."
        )
    if not ids_path.exists():
        raise FileNotFoundError(
            f"FAISS chunk IDs not found at {ids_path}. "
            "Run the M2 pipeline first."
        )

    logger.info(f"Loading HNSW index from {index_path}...")
    index = faiss.read_index(str(index_path))

    with open(ids_path, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    logger.info(f"HNSW index loaded. Vectors: {index.ntotal}")
    return index, chunk_ids


# ---------------------------------------------------------------------------
# Search helper (used in M3)
# ---------------------------------------------------------------------------

def search_faiss(index: faiss.IndexHNSWFlat,
                 query_vector: np.ndarray,
                 chunk_ids: list[str],
                 top_k: int = 10) -> list[tuple[str, float]]:
    """
    Query the HNSW index and return ranked (chunk_id, score) pairs.

    HNSW returns L2 distances. Since embeddings are L2-normalized,
    cosine similarity = 1 - (l2_distance² / 2), which is monotonically
    equivalent to ranking by distance. The score returned here is
    converted to [0, 1] range: score = 1 - (dist / 2).

    Parameters
    ----------
    index        : faiss.IndexHNSWFlat
    query_vector : np.ndarray — shape (D,) or (1, D), float32, L2-normalized
    chunk_ids    : list[str]  — ordered chunk IDs matching index positions
    top_k        : int        — number of results to return

    Returns
    -------
    list[tuple[str, float]]
        Ranked list of (chunk_id, similarity_score), best first.
        Score is in [0, 1]: 1.0 = identical, 0.0 = orthogonal.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if query_vector.dtype != np.float32:
        query_vector = query_vector.astype(np.float32)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:              # FAISS returns -1 for unfilled slots
            continue
        similarity = float(1.0 - dist / 2.0)   # L2 dist → cosine similarity
        results.append((chunk_ids[idx], similarity))

    return results