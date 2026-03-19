"""
pipeline.py
-----------
Orchestrates the full M2 indexing pipeline.

Execution order
---------------
1. Load processed chunks from data/processed/chunks.json
2. Build BM25 index from lemmatized tokens     → indexes/bm25.pkl
3. Generate embeddings for raw chunk texts      → (in memory)
4. Build FAISS index from embeddings            → indexes/faiss.index
5. Save chunk ID list for FAISS lookup          → indexes/faiss_ids.json
6. Save chunk map for retrieval display         → indexes/chunk_map.json
7. Save indexing stats                          → indexes/index_stats.json

After this module runs, the indexes/ directory contains everything
M3 (retrieval) needs to answer queries.
"""

import json
import logging
import time
from pathlib import Path

from .bm25_indexer  import build_bm25, save_bm25, build_chunk_map, save_chunk_map
from .faiss_indexer import build_faiss, save_faiss
from .embedder      import load_model, generate_embeddings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).parent.parent.parent
CHUNKS_FILE   = _ROOT / "data"    / "processed" / "chunks.json"
INDEXES_DIR   = _ROOT / "indexes"

BM25_PATH     = INDEXES_DIR / "bm25.pkl"
FAISS_PATH    = INDEXES_DIR / "faiss.index"
FAISS_IDS     = INDEXES_DIR / "faiss_ids.json"
CHUNK_MAP     = INDEXES_DIR / "chunk_map.json"
STATS_FILE    = INDEXES_DIR / "index_stats.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 64     # chunks per embedding batch
BM25_K1         = 1.5
BM25_B          = 0.75


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Entry point for the M2 indexing pipeline.
    Reads chunks.json, builds both indexes, and writes all artifacts.
    """
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # -- Load chunks ---------------------------------------------------------
    if not CHUNKS_FILE.exists():
        raise SystemExit(
            f"\n[ERROR] Chunks file not found at {CHUNKS_FILE}\n"
            "Run the M1 ingestion pipeline first.\n"
        )

    logger.info(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"{len(chunks)} chunks loaded.")

    if not chunks:
        raise SystemExit("[ERROR] chunks.json is empty — nothing to index.")

    # -- Extract fields ------------------------------------------------------
    chunk_ids         = [c["chunk_id"] for c in chunks]
    tokenized_corpus  = [c["tokens"]   for c in chunks]   # list[list[str]]
    texts             = [c["text"]     for c in chunks]   # list[str]

    # ── BM25 Index ──────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 1/3 — Building BM25 index...")
    t0 = time.time()

    bm25_index = build_bm25(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    save_bm25(bm25_index, BM25_PATH)

    bm25_time = time.time() - t0
    logger.info(f"BM25 done in {bm25_time:.1f}s")

    # ── Embeddings + FAISS Index ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 2/3 — Generating embeddings...")
    t0 = time.time()

    embed_model  = load_model(EMBEDDING_MODEL)
    embeddings   = generate_embeddings(texts, embed_model,
                                       batch_size=BATCH_SIZE)

    embed_time = time.time() - t0
    logger.info(f"Embeddings done in {embed_time:.1f}s")

    logger.info("Step 3/3 — Building FAISS index...")
    t0 = time.time()

    faiss_index = build_faiss(embeddings)
    save_faiss(faiss_index, chunk_ids, FAISS_PATH, FAISS_IDS)

    faiss_time = time.time() - t0
    logger.info(f"FAISS done in {faiss_time:.1f}s")

    # ── Chunk map ────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Saving chunk map...")
    chunk_map = build_chunk_map(chunks)
    save_chunk_map(chunk_map, CHUNK_MAP)

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = {
        "total_chunks":     len(chunks),
        "embedding_model":  EMBEDDING_MODEL,
        "embedding_dim":    int(embeddings.shape[1]),
        "bm25_k1":          BM25_K1,
        "bm25_b":           BM25_B,
        "bm25_build_secs":  round(bm25_time, 2),
        "embed_build_secs": round(embed_time, 2),
        "faiss_build_secs": round(faiss_time, 2),
        "faiss_index_type": "IndexHNSWFlat",
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 50)
    logger.info("M2 complete. Artifacts saved:")
    logger.info(f"  {BM25_PATH}")
    logger.info(f"  {FAISS_PATH}")
    logger.info(f"  {FAISS_IDS}")
    logger.info(f"  {CHUNK_MAP}")
    logger.info(f"  {STATS_FILE}")