""" Shared helpers for dashboard tables, columns, and Matplotlib figures. """

import matplotlib.pyplot as plt
import streamlit as st


def first_value(mapping, *keys, default=None):
    """Return the first non-None value found for the supplied mapping keys."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_column_name(value):
    """Normalize a column label for case- and spacing-insensitive lookup."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, candidates):
    """Find the first candidate column present in a DataFrame."""
    lookup = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        if normalize_column_name(candidate) in lookup:
            return lookup[normalize_column_name(candidate)]
    return None


def styled_figure(fig):
    """Render a Matplotlib figure in Streamlit and close it afterwards."""
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
