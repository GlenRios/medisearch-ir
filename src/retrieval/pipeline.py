"""
pipeline.py
-----------
Orchestrates the M3 retrieval pipeline.

Given a raw user query, this module:
1. Preprocesses the query (clean + lemmatize)
2. Executes BM25 retrieval  → top-K lexical results
3. Executes FAISS retrieval → top-K semantic results
4. Returns both result lists for fusion in M4

The pipeline can also be run standalone for testing individual queries
from the command line.

Output schema (per result in either list)
------------------------------------------
{
    "chunk_id" : "wiki_00012_c005",
    "score"    : 14.32,          # BM25 raw score or FAISS cosine similarity
    "rank"     : 1,              # 1-based position in this retriever's list
    "title"    : "Hypertension",
    "url"      : "https://en.wikipedia.org/wiki/Hypertension",
    "category" : "Cardiology",
    "text"     : "The kidneys are responsible for...",
    "doc_id"   : "wiki_00012"
}
"""

import json
import logging
from dataclasses import dataclass, field

from .query_processor import process_query, ProcessedQuery
from .bm25_retriever  import BM25Retriever
from .faiss_retriever import FAISSRetriever

logger = logging.getLogger(__name__)

TOP_K = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Container for the output of a single retrieval run.

    Attributes
    ----------
    query       : ProcessedQuery   preprocessed query object
    bm25_results  : list[dict]     top-K results from BM25
    faiss_results : list[dict]     top-K results from FAISS
    """
    query:         ProcessedQuery
    bm25_results:  list[dict] = field(default_factory=list)
    faiss_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    Loads both retrievers once and exposes a search() method for repeated
    queries without reloading indexes on every call.

    Usage
    -----
        pipeline = RetrievalPipeline()
        result   = pipeline.search("symptoms of hypertension")

        print(result.bm25_results[0])
        print(result.faiss_results[0])
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        """
        Initialize both retrievers (loads indexes into memory).

        Parameters
        ----------
        top_k : int   number of results to retrieve from each index
        """
        self.top_k = top_k
        logger.info("Initializing RetrievalPipeline...")
        self.bm25_retriever  = BM25Retriever()
        self.faiss_retriever = FAISSRetriever()
        logger.info("RetrievalPipeline ready.")

    def search(self, raw_query: str) -> RetrievalResult:
        """
        Run the full retrieval pipeline for a single query.

        Parameters
        ----------
        raw_query : str   natural language query from the user

        Returns
        -------
        RetrievalResult
            Contains the processed query and both ranked result lists.
        """
        # Step 1 — preprocess query
        query = process_query(raw_query)
        logger.info(f"Query: '{query.raw}' → tokens: {query.tokens}")

        # Step 2 — BM25 retrieval (lexical)
        logger.info("Running BM25 retrieval...")
        bm25_results = self.bm25_retriever.search(query.tokens, top_k=self.top_k)
        logger.info(f"BM25: {len(bm25_results)} results.")

        # Step 3 — FAISS retrieval (semantic)
        logger.info("Running FAISS retrieval...")
        faiss_results = self.faiss_retriever.search(query.clean, top_k=self.top_k)
        logger.info(f"FAISS: {len(faiss_results)} results.")

        return RetrievalResult(
            query         = query,
            bm25_results  = bm25_results,
            faiss_results = faiss_results,
        )


# ---------------------------------------------------------------------------
# CLI — run a test query from the terminal
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Interactive CLI for testing the retrieval pipeline.
    Prints BM25 and FAISS top-5 results side by side.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = RetrievalPipeline(top_k=TOP_K)

    print("\nMediSearch-IR — Retrieval Test")
    print("Type a query and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            raw_query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw_query:
            continue
        if raw_query.lower() == "exit":
            break

        try:
            result = pipeline.search(raw_query)
        except ValueError as e:
            print(f"[ERROR] {e}\n")
            continue

        _print_results("BM25",  result.bm25_results[:5])
        _print_results("FAISS", result.faiss_results[:5])


def _print_results(label: str, results: list[dict]) -> None:
    print(f"\n── {label} top-5 ──────────────────────────────")
    if not results:
        print("  No results.")
        return
    for r in results:
        print(f"  [{r['rank']}] {r['title']} (score={r['score']:.4f})")
        print(f"       {r['text'][:120].strip()}...")
        print(f"       {r['url']}")
    print()