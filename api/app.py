"""
app.py
------
MediSearch-IR — Streamlit web interface.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from state import run_search
from components.sidebar    import render_sidebar
from components.search_bar import render_search_bar
from components.columns    import render_results_columns
from components.comparison import render_comparison


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "MediSearch-IR",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    # Header
    st.title("🔬 MediSearch-IR")
    st.caption(
        "Hybrid Information Retrieval over a Medical Wikipedia Corpus — "
        "BM25 · FAISS HNSW · Reciprocal Rank Fusion · Cross-Encoder"
    )
    st.divider()

    # Search bar
    query = render_search_bar()

    if not query:
        st.info("Enter a query above to search the medical corpus.")
        return

    # Run search
    with st.spinner(f'Searching for "{query}"...'):
        results = run_search(query)

    if results is None:
        return

    # Results count summary
    n_hybrid = len(results.get("hybrid_results", []))
    n_bm25   = len(results.get("bm25_results",   []))
    n_faiss  = len(results.get("faiss_results",  []))
    st.success(
        f"Found **{n_hybrid}** hybrid results "
        f"(BM25: {n_bm25} · FAISS: {n_faiss})"
    )

    st.divider()

    # Three-column results
    render_results_columns(results)

    st.divider()

    # Ranking comparison table
    render_comparison(results)


if __name__ == "__main__":
    main()