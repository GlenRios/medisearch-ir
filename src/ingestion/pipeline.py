"""
pipeline.py
-----------
Orchestrates the full M1 ingestion pipeline.

Execution order
---------------
1. Load raw corpus from data/raw/corpus.json
2. For each article:
   a. Clean text              (cleaner.py)
   b. Split into sentences    (lemmatizer.py → spaCy senter)
   c. Chunk with sliding win  (chunker.py)
   d. Lemmatize each chunk    (lemmatizer.py → spaCy lemmatizer)
   e. Build Chunk dataclass
3. Save all chunks to data/processed/chunks.json
4. Save ingestion stats to data/processed/ingestion_stats.json

Output schema (one entry in chunks.json)
-----------------------------------------
{
  "chunk_id":    "wiki_00001_c003",
  "doc_id":      "wiki_00001",
  "title":       "Hypertension",
  "url":         "https://en.wikipedia.org/wiki/Hypertension",
  "category":    "Cardiology",
  "chunk_index": 3,
  "text":        "The kidneys are responsible for long-term...",
  "tokens":      ["kidney", "responsible", "blood", "pressure", ...],
  "n_tokens":    47
}
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from tqdm import tqdm

from .cleaner    import clean_text, is_valid_text
from .chunker    import sliding_window, split_into_sentences
from .lemmatizer import load_model, get_sentences, lemmatize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT        = Path(__file__).parent.parent.parent
INPUT_FILE   = _ROOT / "data" / "raw"       / "corpus.json"
OUTPUT_DIR   = _ROOT / "data" / "processed"
CHUNKS_FILE  = OUTPUT_DIR / "chunks.json"
STATS_FILE   = OUTPUT_DIR / "ingestion_stats.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPACY_MODEL      = "en_core_web_sm"
CHUNK_SIZE       = 5      # sentences per chunk
CHUNK_OVERLAP    = 1      # shared sentences between consecutive chunks
MIN_CHUNK_TOKENS = 30     # discard lemmatized chunks shorter than this


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:    str
    doc_id:      str
    title:       str
    url:         str
    category:    str
    chunk_index: int
    text:        str        # raw text  → used for embedding (M2)
    tokens:      list       # lemmas    → used for BM25 (M2)
    n_tokens:    int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Entry point for the M1 ingestion pipeline.
    Reads the raw corpus, processes every article, and writes chunks.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # -- Load spaCy ----------------------------------------------------------
    nlp = load_model(SPACY_MODEL)

    # -- Load raw corpus -----------------------------------------------------
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"\n[ERROR] Corpus not found at {INPUT_FILE}\n"
            "Run module0_corpus.py first.\n"
        )

    logger.info(f"Loading corpus from {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"{len(articles)} articles loaded.")

    # -- Process articles ----------------------------------------------------
    all_chunks: list[dict] = []
    skipped = 0

    for article in tqdm(articles, desc="Ingesting articles"):
        raw_text = article.get("text", "")

        # Step 1 — clean
        clean = clean_text(raw_text)
        if not is_valid_text(clean):
            skipped += 1
            continue

        # Step 2 — sentence splitting
        sentences = get_sentences(clean, nlp)
        if not sentences:
            skipped += 1
            continue

        # Step 3 & 4 — chunk + lemmatize
        for raw_chunk in sliding_window(sentences, CHUNK_SIZE, CHUNK_OVERLAP):
            tokens = lemmatize(raw_chunk.text, nlp)

            if len(tokens) < MIN_CHUNK_TOKENS:
                continue                    # skip stub chunks

            chunk = Chunk(
                chunk_id    = f"{article['doc_id']}_c{raw_chunk.chunk_index:03d}",
                doc_id      = article["doc_id"],
                title       = article["title"],
                url         = article["url"],
                category    = article["category"],
                chunk_index = raw_chunk.chunk_index,
                text        = raw_chunk.text,
                tokens      = tokens,
                n_tokens    = len(tokens),
            )
            all_chunks.append(asdict(chunk))

    # -- Save chunks ---------------------------------------------------------
    logger.info(f"Saving {len(all_chunks)} chunks → {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # -- Save stats ----------------------------------------------------------
    _save_stats(articles, all_chunks, skipped)

    logger.info("=" * 50)
    logger.info(f"Done. {len(all_chunks)} chunks saved.")
    logger.info(f"Articles processed : {len(articles) - skipped}")
    logger.info(f"Articles skipped   : {skipped}")
    if all_chunks:
        avg = sum(c["n_tokens"] for c in all_chunks) // len(all_chunks)
        logger.info(f"Avg tokens/chunk   : {avg}")


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _save_stats(articles: list,
                chunks: list,
                skipped: int) -> None:
    cat_counts: dict = {}
    for c in chunks:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1

    stats = {
        "articles_loaded":      len(articles),
        "articles_processed":   len(articles) - skipped,
        "articles_skipped":     skipped,
        "total_chunks":         len(chunks),
        "avg_tokens_per_chunk": (
            sum(c["n_tokens"] for c in chunks) // len(chunks) if chunks else 0
        ),
        "chunk_size_sentences": CHUNK_SIZE,
        "chunk_overlap":        CHUNK_OVERLAP,
        "min_chunk_tokens":     MIN_CHUNK_TOKENS,
        "chunks_per_category":  cat_counts,
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)