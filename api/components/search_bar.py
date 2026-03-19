"""
components/search_bar.py
------------------------
Renders the main search bar and returns the submitted query.
"""

import streamlit as st

EXAMPLE_QUERIES = [
    "symptoms of hypertension",
    "how does the immune system work",
    "causes of type 2 diabetes",
    "treatment options for depression",
    "what is insulin resistance",
]


def render_search_bar() -> str | None:
    """
    Renders the search input and example query buttons.

    Returns
    -------
    str | None
        The submitted query string, or None if no query was submitted.
    """
    st.markdown("## Search the Medical Corpus")

    query = st.text_input(
        label       = "query",
        placeholder = "e.g. symptoms of hypertension...",
        label_visibility = "collapsed",
    )

    # Example query buttons
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, example in zip(cols, EXAMPLE_QUERIES):
        if col.button(example, use_container_width=True):
            query = example

    return query.strip() if query and query.strip() else None