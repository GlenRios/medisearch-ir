"""
components/comparison.py
------------------------
Renders a ranking comparison table showing where each document
appeared across BM25, FAISS, and Hybrid rankings.
"""

import streamlit as st


def render_comparison(search_results: dict) -> None:
    """
    Render a unified comparison table of all three system rankings.

    Shows each unique chunk that appeared in any ranking, alongside
    its rank position in each system. Helps visualize how RRF fusion
    combines the two individual rankings.

    Parameters
    ----------
    search_results : dict
        Output from state.run_search().
    """
    st.markdown("### Ranking Comparison")
    st.caption(
        "Rank position of each document across the three systems. "
        "— means the document did not appear in that system's top results."
    )

    bm25_results   = search_results.get("bm25_results",   [])
    faiss_results  = search_results.get("faiss_results",  [])
    hybrid_results = search_results.get("hybrid_results", [])

    # Build lookup dicts: chunk_id → rank/score
    bm25_map   = {r["chunk_id"]: r for r in bm25_results}
    faiss_map  = {r["chunk_id"]: r for r in faiss_results}
    hybrid_map = {r["chunk_id"]: r for r in hybrid_results}

    # Collect all unique chunk_ids that appear in any ranking
    all_ids = (
        {r["chunk_id"] for r in hybrid_results}   # hybrid first (final ranking)
        | {r["chunk_id"] for r in bm25_results[:10]}
        | {r["chunk_id"] for r in faiss_results[:10]}
    )

    if not all_ids:
        st.info("No results to compare.")
        return

    # Build table rows
    rows = []
    for chunk_id in all_ids:
        b = bm25_map.get(chunk_id)
        f = faiss_map.get(chunk_id)
        h = hybrid_map.get(chunk_id)

        # Title from whichever system has it
        title = (
            (h or b or f or {}).get("title", chunk_id)
        )

        rows.append({
            "Title":       title[:45] + "..." if len(title) > 45 else title,
            "Hybrid rank": f"#{h['final_rank']}" if h else "—",
            "CE score":    f"{h['ce_score']:.3f}" if h else "—",
            "BM25 rank":   f"#{b['rank']}"        if b else "—",
            "BM25 score":  f"{b['score']:.2f}"    if b else "—",
            "FAISS rank":  f"#{f['rank']}"         if f else "—",
            "FAISS sim":   f"{f['score']:.3f}"     if f else "—",
        })

    # Sort by hybrid rank first, then by BM25 rank
    def sort_key(row):
        h = row["Hybrid rank"]
        b = row["BM25 rank"]
        return (
            int(h[1:]) if h != "—" else 999,
            int(b[1:]) if b != "—" else 999,
        )

    rows.sort(key=sort_key)

    st.dataframe(
        rows,
        use_container_width = True,
        hide_index          = True,
        column_config       = {
            "Title":       st.column_config.TextColumn(width="large"),
            "Hybrid rank": st.column_config.TextColumn(width="small"),
            "CE score":    st.column_config.TextColumn(width="small"),
            "BM25 rank":   st.column_config.TextColumn(width="small"),
            "BM25 score":  st.column_config.TextColumn(width="small"),
            "FAISS rank":  st.column_config.TextColumn(width="small"),
            "FAISS sim":   st.column_config.TextColumn(width="small"),
        },
    )