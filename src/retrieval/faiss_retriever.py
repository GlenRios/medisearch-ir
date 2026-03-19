"""
faiss_retriever.py
------------------
Executes a semantic query against the HNSW FAISS index and returns
a ranked list of (chunk_id, score) pairs.

The query text is encoded with the same sentence-transformers model
used during indexing (M2). The resulting vector is L2-normalized and
searched against the HNSW index using L2 distance, which is converted
to a cosine similarity score in [0, 1].
"""

import logging
from pathlib import Path

import numpy as np

from src.indexing.faiss_indexer import load_faiss, search_faiss
from src.indexing.bm25_indexer  import load_chunk_map
from src.indexing.embedder      import load_model, generate_embeddings

logger = logging.getLogger(__name__)

# Default artifact paths
_ROOT          = Path(__file__).parent.parent.parent
FAISS_PATH     = _ROOT / "indexes" / "faiss.index"
FAISS_IDS_PATH = _ROOT / "indexes" / "faiss_ids.json"
CHUNK_MAP_PATH = _ROOT / "indexes" / "chunk_map.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Wraps a loaded HNSW FAISS index and exposes a simple search interface.

    Attributes
    ----------
    index      : faiss.IndexHNSWFlat
    chunk_ids  : list[str]   ordered chunk IDs matching index positions
    chunk_map  : dict        chunk_id → {text, title, url, category, doc_id}
    model      : SentenceTransformer
    """

    def __init__(self,
                 faiss_path: Path = FAISS_PATH,
                 faiss_ids_path: Path = FAISS_IDS_PATH,
                 chunk_map_path: Path = CHUNK_MAP_PATH,
                 model_name: str = EMBEDDING_MODEL) -> None:
        """
        Load the FAISS index, chunk IDs, chunk map, and embedding model.

        Parameters
        ----------
        faiss_path     : Path — path to faiss.index
        faiss_ids_path : Path — path to faiss_ids.json
        chunk_map_path : Path — path to chunk_map.json
        model_name     : str  — sentence-transformers model identifier
        """
        self.index, self.chunk_ids = load_faiss(faiss_path, faiss_ids_path)
        self.chunk_map             = load_chunk_map(chunk_map_path)
        self.model                 = load_model(model_name)
        logger.info(f"FAISSRetriever ready. Index vectors: {self.index.ntotal}")

    def search(self,
               query_text: str,
               top_k: int = 20) -> list[dict]:
        """
        Encode the query and search the HNSW index.

        Parameters
        ----------
        query_text : str   cleaned query text from query_processor
        top_k      : int   number of results to return (default 20)

        Returns
        -------
        list[dict]
            Ranked list of result dicts, best score first. Each dict:
            {
                "chunk_id" : str,
                "score"    : float,  cosine similarity in [0, 1]
                "rank"     : int,    1-based rank position
                "title"    : str,
                "url"      : str,
                "category" : str,
                "text"     : str,
                "doc_id"   : str,
            }
        """
        if not query_text or not query_text.strip():
            logger.warning("FAISS search called with empty query text.")
            return []

        # Encode query with the same model used during indexing
        query_vector = generate_embeddings(
            [query_text],
            self.model,
            batch_size=1,
            show_progress=False,
        )[0]                              # shape (D,)

        raw_results = search_faiss(
            self.index,
            query_vector,
            self.chunk_ids,
            top_k=top_k,
        )                                 # list[(chunk_id, similarity_score)]

        results = []
        for rank, (chunk_id, score) in enumerate(raw_results, start=1):
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

        logger.debug(f"FAISS returned {len(results)} results for '{query_text[:50]}...'")
        return results