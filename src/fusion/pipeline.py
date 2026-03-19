"""
pipeline.py
-----------
Orchestrates the full M4 fusion pipeline.

Given a RetrievalResult from M3, this module:
1. Fuses BM25 and FAISS rankings with RRF
2. Re-ranks the top-N RRF candidates with a cross-encoder
3. Returns a FusionResult containing all intermediate and final rankings

This module is designed to be used together with M3:

    from src.retrieval.pipeline import RetrievalPipeline
    from src.fusion.pipeline    import FusionPipeline

    retrieval = RetrievalPipeline()
    fusion    = FusionPipeline()

    retrieval_result = retrieval.search("symptoms of hypertension")
    fusion_result    = fusion.fuse(retrieval_result)

    for r in fusion_result.final_results:
        print(r.final_rank, r.title, r.ce_score)
"""

import logging
from dataclasses import dataclass, field

from src.retrieval.pipeline import RetrievalResult

from .rrf      import reciprocal_rank_fusion, FusedResult, RRF_K
from .reranker import load_reranker, rerank, RerankedResult, DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RRF_K_VALUE      = RRF_K    # 60 — smoothing constant
RERANK_TOP_K     = 20       # candidates passed to the cross-encoder
FINAL_TOP_K      = 10       # results returned in the final list


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """
    Full output of the M4 fusion pipeline for a single query.

    Attributes
    ----------
    query_raw      : str                 original raw query
    rrf_results    : list[FusedResult]   all RRF-fused candidates
    final_results  : list[RerankedResult] top results after cross-encoder
    """
    query_raw:     str
    rrf_results:   list[FusedResult]    = field(default_factory=list)
    final_results: list[RerankedResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FusionPipeline:
    """
    Loads the cross-encoder once and exposes a fuse() method for
    repeated queries without reloading the model on every call.

    Usage
    -----
        fusion = FusionPipeline()
        result = fusion.fuse(retrieval_result)
    """

    def __init__(self,
                 reranker_model: str = DEFAULT_MODEL,
                 rrf_k: int = RRF_K_VALUE,
                 rerank_top_k: int = RERANK_TOP_K,
                 final_top_k: int = FINAL_TOP_K) -> None:
        """
        Initialize the fusion pipeline.

        Parameters
        ----------
        reranker_model : str   HuggingFace cross-encoder model identifier
        rrf_k          : int   RRF smoothing constant (default 60)
        rerank_top_k   : int   candidates to pass to the cross-encoder
        final_top_k    : int   results to return after re-ranking
        """
        self.rrf_k        = rrf_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k  = final_top_k
        self.reranker     = load_reranker(reranker_model)
        logger.info("FusionPipeline ready.")

    def fuse(self, retrieval_result: RetrievalResult) -> FusionResult:
        """
        Run RRF fusion followed by cross-encoder re-ranking.

        Parameters
        ----------
        retrieval_result : RetrievalResult
            Output from M3 RetrievalPipeline.search().

        Returns
        -------
        FusionResult
            Contains the full RRF ranking and the final re-ranked results.
        """
        query_raw = retrieval_result.query.raw
        logger.info(f"Fusing results for query: '{query_raw}'")

        # Step 1 — RRF
        rrf_results = reciprocal_rank_fusion(
            bm25_results  = retrieval_result.bm25_results,
            faiss_results = retrieval_result.faiss_results,
            k             = self.rrf_k,
        )
        logger.info(f"RRF produced {len(rrf_results)} unique candidates.")

        # Step 2 — Cross-encoder re-ranking
        final_results = rerank(
            query      = retrieval_result.query.clean,
            candidates = rrf_results,
            model      = self.reranker,
            top_k      = self.rerank_top_k,
        )

        # Trim to final_top_k
        final_results = final_results[:self.final_top_k]
        logger.info(f"Final re-ranked results: {len(final_results)}")

        return FusionResult(
            query_raw     = query_raw,
            rrf_results   = rrf_results,
            final_results = final_results,
        )


# ---------------------------------------------------------------------------
# CLI — run a test query from the terminal
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Interactive CLI for testing the full hybrid pipeline end-to-end:
    query → BM25 + FAISS → RRF → cross-encoder → final results.
    """
    import logging
    from src.retrieval.pipeline import RetrievalPipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    retrieval = RetrievalPipeline(top_k=20)
    fusion    = FusionPipeline()

    print("\nMediSearch-IR — Hybrid Search (RRF + Cross-Encoder)")
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
            retrieval_result = retrieval.search(raw_query)
            fusion_result    = fusion.fuse(retrieval_result)
        except ValueError as e:
            print(f"[ERROR] {e}\n")
            continue

        _print_results(fusion_result)


def _print_results(result: FusionResult) -> None:
    print(f"\n── Final results for: '{result.query_raw}' ────────────────")
    if not result.final_results:
        print("  No results.")
        return
    for r in result.final_results:
        bm25_tag  = f"BM25#{r.bm25_rank}"  if r.bm25_rank  else "BM25:—"
        faiss_tag = f"FAISS#{r.faiss_rank}" if r.faiss_rank else "FAISS:—"
        print(f"\n  [{r.final_rank}] {r.title}")
        print(f"       CE score : {r.ce_score:.4f}  |  "
              f"RRF: {r.rrf_score:.6f}  |  {bm25_tag}  {faiss_tag}")
        print(f"       {r.text[:140].strip()}...")
        print(f"       {r.url}")
    print()