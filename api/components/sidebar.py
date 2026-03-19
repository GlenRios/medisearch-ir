"""
components/sidebar.py
---------------------
Renders the sidebar with corpus and index statistics.
"""

import streamlit as st
from state import get_corpus_stats


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## MediSearch-IR")
        st.caption("Hybrid Information Retrieval System")
        st.divider()

        stats = get_corpus_stats()

        # Corpus stats
        st.markdown("### Corpus")
        col1, col2 = st.columns(2)
        col1.metric("Articles",  stats.get("articles_processed", "—"))
        col2.metric("Chunks",    stats.get("total_chunks",       "—"))

        avg = stats.get("avg_tokens_per_chunk", "—")
        st.metric("Avg tokens / chunk", avg)

        # Index stats
        st.markdown("### Index")
        col3, col4 = st.columns(2)
        col3.metric("Index type", "HNSW")
        col4.metric("Emb. dim",   stats.get("embedding_dim", 384))

        model = stats.get("embedding_model", "all-MiniLM-L6-v2")
        st.caption(f"Model: `{model.split('/')[-1]}`")

        # Category breakdown
        cat_counts = stats.get("chunks_per_category", {})
        if cat_counts:
            st.markdown("### Categories")
            sorted_cats = sorted(cat_counts.items(),
                                 key=lambda x: x[1], reverse=True)
            for cat, count in sorted_cats[:8]:
                pct = int(count / sum(cat_counts.values()) * 100)
                st.progress(pct / 100, text=f"{cat} ({count})")

        st.divider()
        st.caption("BM25 · FAISS HNSW · RRF · Cross-Encoder")