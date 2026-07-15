"""
ui/theme.py — light/dark theming for the dashboard
==================================================
A single injected-CSS dark theme, extended to cover the elements
the first pass missed: dataframes, code/LaTeX blocks, download buttons, tab
headers, expanders, inputs and the progress bar. ``inject_theme(dark)`` is a
no-op in light mode (Streamlit's default light styling is fine).
"""

from __future__ import annotations

import streamlit as st

_DARK_CSS = """
<style>
:root {
    --qfs-bg:    #0e1117;
    --qfs-bg2:   #161b22;
    --qfs-card:  #161b22;
    --qfs-text:  #e6edf3;
    --qfs-muted: #9aa7b4;
    --qfs-accent:#60a5fa;
    --qfs-border:#30363d;
}

/* app shell + sidebar */
.stApp { background-color: var(--qfs-bg); color: var(--qfs-text); }
section[data-testid="stSidebar"] { background-color: var(--qfs-bg2); }

/* headings, text, captions */
h1, h2, h3, h4, h5, p, li, label, .stMarkdown, .stCaption, .stText {
    color: var(--qfs-text) !important;
}
[data-testid="stCaptionContainer"], .stCaption p { color: var(--qfs-muted) !important; }

/* metric cards */
[data-testid="stMetric"] {
    background-color: var(--qfs-card); border: 1px solid var(--qfs-border);
    border-radius: 10px; padding: 12px 16px;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--qfs-text) !important; }

/* buttons + download buttons */
.stButton>button, .stDownloadButton>button {
    background-color: var(--qfs-accent); color: #0e1117; border: none; font-weight: 600;
}
.stButton>button:hover, .stDownloadButton>button:hover { filter: brightness(1.1); }

/* tabs */
button[data-baseweb="tab"] { color: var(--qfs-muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--qfs-text) !important; border-bottom-color: var(--qfs-accent) !important;
}

/* dataframes / tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: var(--qfs-card); border: 1px solid var(--qfs-border); border-radius: 8px;
}
[data-testid="stDataFrame"] * { color: var(--qfs-text) !important; }

/* code + LaTeX blocks */
.stCode, pre, code { background-color: #0b0f14 !important; color: #e6edf3 !important; }
[data-testid="stMarkdownContainer"] .katex, .katex { color: var(--qfs-text) !important; }

/* inputs / selects / sliders */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #0b0f14 !important; color: var(--qfs-text) !important;
}
.stMultiSelect div[data-baseweb="select"] { background-color: #0b0f14 !important; }

/* expander */
.streamlit-expanderHeader, details summary { color: var(--qfs-text) !important; }
details { border: 1px solid var(--qfs-border); border-radius: 8px; }

/* progress + alerts keep accent */
.stProgress > div > div > div { background-color: var(--qfs-accent) !important; }
</style>
"""


def inject_theme(dark: bool) -> None:
    """Inject the dark CSS when ``dark`` is True; light mode uses defaults."""
    if dark:
        st.markdown(_DARK_CSS, unsafe_allow_html=True)
