"""
src/fusion/
-----------
M4 — Fusion & Re-ranking package.

Modules
-------
rrf.py       — Reciprocal Rank Fusion algorithm
reranker.py  — cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
pipeline.py  — orchestrates RRF → reranker + interactive CLI

Usage
-----
    # Interactive CLI (full hybrid pipeline)
    python -m src.fusion

    # In code
    from src.retrieval.pipeline import RetrievalPipeline
    from src.fusion.pipeline    import FusionPipeline

    retrieval = RetrievalPipeline()
    fusion    = FusionPipeline()

    r = retrieval.search("causes of type 2 diabetes")
    f = fusion.fuse(r)

    for result in f.final_results:
        print(result.final_rank, result.title, result.ce_score)
"""