import re
from pathlib import Path
import pandas as pd
import streamlit as st
import os

FEEDBACK_FILE = "data/raw/feedback.csv"

# Ensure CSV exists
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=["Reviews", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

def stars_from_bubbles(text: str) -> str:
    m = re.search(r"(\d(?:\.\d)?)\s*of\s*5", str(text))
    if not m:
        return "☆☆☆☆☆"
    try:
        score = float(m.group(1))
    except:
        score = 0
    full = max(0, min(int(score), 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_grid_with_delete(max_rows: int = 10):
    """Show last N feedback in a compact 2-column grid with delete checkboxes."""
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"⚠️ Could not load feedback: {e}")
        return

    if df.empty:
        st.caption("No feedback yet.")
        return

    # Keep absolute indices so we can delete the exact rows
    last = df.tail(max_rows)
    abs_indices = list(last.index)

    cols = st.columns(2)  # 2-column layout
    to_delete = set()

    for i, (abs_idx, row) in enumerate(last.iterrows()):
        col = cols[i % 2]
        with col.container(border=True):
            stars = stars_from_bubbles(row.get("Reviews", ""))
            comment = str(row.get("Comments", "")).strip() or "— (no comment) —"
            st.markdown(f"<div style='font-size:1.05rem'>{stars}</div>", unsafe_allow_html=True)
            st.write(comment)
            # Per-card checkbox to mark for deletion
            if st.checkbox("Delete", key=f"del_{abs_idx}"):
                to_delete.add(abs_idx)

    # One button to delete selected rows
    if to_delete:
        if st.button(f"Delete {len(to_delete)} selected", type="primary"):
            df = df.drop(index=list(to_delete))
            df.to_csv(FEEDBACK_FILE, index=False)
            st.success("Deleted. Refreshing…")
            st.rerun()
