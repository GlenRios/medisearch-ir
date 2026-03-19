"""
components/result_card.py
-------------------------
Renders a single result as an expandable card.
"""

import streamlit as st


def render_result_card(result: dict,
                       score_label: str,
                       score_value: float,
                       rank: int) -> None:
    """
    Render one search result as a styled expandable card.

    Parameters
    ----------
    result      : dict   result dict with title, text, url, category, etc.
    score_label : str    label for the score (e.g. 'BM25 score', 'CE score')
    score_value : float  the score to display
    rank        : int    1-based rank position
    """
    title    = result.get("title",    "Untitled")
    text     = result.get("text",     "")
    url      = result.get("url",      "#")
    category = result.get("category", "")

    preview = text[:200].strip() + "..." if len(text) > 200 else text

    with st.expander(f"**{rank}.** {title}", expanded=rank <= 3):
        # Metadata row
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.caption(f"Category: **{category}**")
        col2.caption(f"{score_label}: **{score_value:.4f}**")
        col3.caption(f"Rank #{rank}")

        # Text preview
        st.markdown(f"> {preview}")

        # Source link
        st.markdown(f"[Read full article →]({url})")