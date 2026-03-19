"""
embedder.py
-----------
Handles loading the sentence-transformers model and generating
dense vector embeddings for a list of text chunks.

The embeddings produced here are consumed by faiss_indexer.py
to build the dense vector index.

Model choice
------------
'sentence-transformers/all-MiniLM-L6-v2' is used by default:
  - 384-dimensional embeddings
  - Fast inference, small footprint (~80 MB)
  - Strong performance on semantic similarity tasks
  - Fully compatible with FAISS IndexFlatIP (inner product)

To switch models, change EMBEDDING_MODEL in pipeline.py.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level model cache (loaded once per process)
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a SentenceTransformer model and cache it for the process lifetime.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local path.

    Returns
    -------
    SentenceTransformer
        The loaded embedding model.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    logger.info(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    logger.info(f"Model '{model_name}' loaded. "
                f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str],
                        model: SentenceTransformer,
                        batch_size: int = 64,
                        show_progress: bool = True) -> np.ndarray:
    """
    Generate normalized L2 embeddings for a list of texts.

    Embeddings are L2-normalized so that inner product search
    in FAISS is equivalent to cosine similarity.

    Parameters
    ----------
    texts        : list[str]   — raw chunk texts (not tokens)
    model        : SentenceTransformer
    batch_size   : int         — texts per inference batch (default 64)
    show_progress: bool        — show tqdm progress bar

    Returns
    -------
    np.ndarray
        Shape (N, D) float32 array where N = len(texts), D = embedding dim.
    """
    if not texts:
        raise ValueError("texts list is empty — nothing to embed.")

    logger.info(f"Generating embeddings for {len(texts)} chunks "
                f"(batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2 normalize → cosine via inner product
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    logger.info(f"Embeddings generated. Shape: {embeddings.shape}")
    return embeddings


def get_embedding_dim(model: SentenceTransformer) -> int:
    """Return the embedding dimensionality of the loaded model."""
    return model.get_sentence_embedding_dimension()