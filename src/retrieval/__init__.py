"""
src/retrieval/
--------------
M3 — Retrieval package.

Modules
-------
query_processor.py  — clean + lemmatize the user query
bm25_retriever.py   — lexical search over BM25 index
faiss_retriever.py  — semantic search over HNSW FAISS index
pipeline.py         — orchestrates both retrievers + CLI

Usage
-----
    # Interactive CLI
    python -m src.retrieval

    # In code
    from src.retrieval.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline()
    result   = pipeline.search("symptoms of hypertension")
"""