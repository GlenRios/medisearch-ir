"""
components/columns.py
---------------------
Renders the three-column layout showing BM25, FAISS, and Hybrid results
side by side.
"""

import streamlit as st
from .result_card import render_result_card

TOP_N = 5   # results shown per column by default


def render_results_columns(search_results: dict) -> None:
    """
    Render BM25, FAISS, and Hybrid results in three side-by-side columns.

    Parameters
    ----------
    search_results : dict
        Output from state.run_search() containing bm25_results,
        faiss_results, and hybrid_results.
    """
    st.markdown("### Results")

    col_bm25, col_faiss, col_hybrid = st.columns(3)

    # ── BM25 column ─────────────────────────────────────────────────────────
    with col_bm25:
        st.markdown("#### BM25")
        st.caption("Lexical retrieval")
        results = search_results.get("bm25_results", [])[:TOP_N]
        if results:
            for r in results:
                render_result_card(
                    result      = r,
                    score_label = "BM25 score",
                    score_value = r.get("score", 0.0),
                    rank        = r.get("rank",  0),
                )
        else:
            st.info("No results.")

    # ── FAISS column ─────────────────────────────────────────────────────────
    with col_faiss:
        st.markdown("#### FAISS")
        st.caption("Semantic retrieval")
        results = search_results.get("faiss_results", [])[:TOP_N]
        if results:
            for r in results:
                render_result_card(
                    result      = r,
                    score_label = "Cosine sim",
                    score_value = r.get("score", 0.0),
                    rank        = r.get("rank",  0),
                )
        else:
            st.info("No results.")

    # ── Hybrid column ─────────────────────────────────────────────────────────
    with col_hybrid:
        st.markdown("#### Hybrid")
        st.caption("RRF + Cross-Encoder")
        results = search_results.get("hybrid_results", [])[:TOP_N]
        if results:
            for r in results:
                render_result_card(
                    result      = r,
                    score_label = "CE score",
                    score_value = r.get("ce_score", 0.0),
                    rank        = r.get("final_rank", 0),
                )
        else:
            st.info("No results.")