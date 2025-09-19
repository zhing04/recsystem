import os
import re
import html
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# -------------------- Paths --------------------
DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"
ICON_PATH = "data/App_icon.png"
COVER_IMG = "data/restaurant.jpg"
FOOTER_IMG = "data/food_2.jpg"
FEEDBACK_FILE = "data/raw/feedback.csv"

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Restaurant Supervised Recommender",
                   layout="centered",
                   initial_sidebar_state="expanded")

# Sidebar icon (optional)
if os.path.isfile(ICON_PATH):
    st.sidebar.image(ICON_PATH, use_container_width=True)

# -------------------- Ensure feedback CSV exists --------------------
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=["Reviews", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

# -------------------- Helpers --------------------
def stars_from_bubbles(text: str) -> str:
    """Render star icons from strings like '4.5 of 5 bubbles'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*of\s*5", str(text))
    try:
        score = float(m.group(1)) if m else 0.0
    except Exception:
        score = 0.0
    full = max(0, min(int(round(score)), 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_grid(max_rows: int = 10) -> None:
    """Show the last N feedback entries in a compact grid."""
    try:
        df_fb = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"Could not load feedback: {e}")
        return
    if df_fb.empty:
        st.caption("No feedback yet.")
        return

    df_fb["Comments"] = df_fb["Comments"].astype(str)
    df_fb = df_fb[df_fb["Comments"].str.strip().ne("")]
    if df_fb.empty:
        st.caption("No feedback yet.")
        return

    last = df_fb.tail(max_rows).reset_index(drop=True)
    cols = st.columns(2)
    for i, row in last.iterrows():
        col = cols[i % 2]
        stars = stars_from_bubbles(row.get("Reviews", ""))
        safe_comment = html.escape(str(row.get("Comments", "")).strip())
        col.markdown(
            (
                '<div style="border:1px solid #ccc;border-radius:8px;'
                'padding:8px;margin-bottom:8px;">'
                f'<div style="font-size:1.1rem;margin-bottom:6px;">{stars}</div>'
                f'<div style="font-size:0.98rem;line-height:1.35;">{safe_comment}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

# -------------------- Load dataset --------------------
try:
    df = pd.read_csv(DATA_TRIP)
except Exception as e:
    st.error(f"Could not load dataset at {DATA_TRIP}. Error: {e}")
    st.stop()

# Minimal cleaning to match prior prep
if {"Street Address", "Location"}.issubset(df.columns):
    df["Location"] = df["Street Address"].astype(str) + ", " + df["Location"].astype(str)
    df = df.drop(columns=["Street Address"], errors="ignore")

# Keep unique restaurants
if "Name" in df.columns:
    df = df.drop_duplicates(subset="Name").reset_index(drop=True)

# -------------------- Header --------------------
st.markdown("<h1 style='text-align:center;'>Restaurant Supervised Recommender</h1>", unsafe_allow_html=True)
st.markdown(
    "This app uses **Supervised Learning (Gradient Boosting)** to rank restaurants. "
    "Pick a **Top-N** list, or choose **Similar to a Restaurant** to get targeted suggestions."
)
if os.path.isfile(COVER_IMG):
    st.image(Image.open(COVER_IMG), use_container_width=True)

# -------------------- Sidebar controls --------------------
st.sidebar.header("Controls")
top_n = st.sidebar.slider("How many results to show (Top-N):", min_value=5, max_value=30, value=10, step=1)
q = st.sidebar.slider(
    "Positive class quantile (Top % threshold)",
    min_value=0.50, max_value=0.90, value=0.70, step=0.05,
    help="Top (1−q) fraction becomes positive class. q=0.70 → top 30% are positives."
)

# -------------------- Feature selection --------------------
# Prefer sentiment columns that contain the word "Sentiment"
sentiment_cols = [c for c in df.columns if "Sentiment" in c]

if not sentiment_cols:
    st.error(
        "No sentiment columns found. Expected columns containing 'Sentiment' "
        "(e.g., 'Average Food Sentiment', 'Average Service Sentiment')."
    )
    st.stop()

# Drop rows with missing sentiments
df_use = df.dropna(subset=sentiment_cols).copy()
if df_use.empty:
    st.error("All rows have missing sentiment values. Please check your dataset.")
    st.stop()

X = df_use[sentiment_cols].astype(float).values
composite = df_use[sentiment_cols].astype(float).mean(axis=1)
threshold = float(np.quantile(composite, q))
y = (composite >= threshold).astype(int).values

# Guard against degenerate labels
if y.sum() == 0 or y.sum() == len(y):
    st.warning("Labels degenerated (all same). Falling back to simple sort by composite score.")
    df_use["Composite"] = composite
    top_fallback = df_use.sort_values("Composite", ascending=False).head(top_n)
    st.subheader("Top Restaurants (Fallback: Composite Score)")
    show_cols = ["Name", "Composite"] + sentiment_cols
    st.dataframe(top_fallback[show_cols])
    # Feedback + footer
    st.markdown("## Rate Your Experience")
    rating = st.slider("Rate this restaurant (1-5)", 1, 5)
    feedback_comment = st.text_area("Your Feedback")
    if st.button("Submit Feedback"):
        df_fb = pd.read_csv(FEEDBACK_FILE) if os.path.isfile(FEEDBACK_FILE) else pd.DataFrame(columns=["Reviews", "Comments"])
        comment_clean = str(feedback_comment).strip()
        if comment_clean and comment_clean.lower() != "nan":
            new_feedback = pd.DataFrame([{"Reviews": f"{rating} of 5 bubbles", "Comments": comment_clean}])
            df_fb = pd.concat([df_fb, new_feedback], ignore_index=True)
            df_fb.to_csv(FEEDBACK_FILE, index=False)
            st.success("Thanks for your feedback!")
        else:
            st.warning("Please enter a real comment.")
    st.subheader("Recent Feedback")
    render_feedback_grid(max_rows=10)
    if os.path.isfile(FOOTER_IMG):
        st.image(Image.open(FOOTER_IMG), use_container_width=True)
    st.stop()

# -------------------- Train/Test split & model --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
clf.fit(X_train, y_train)

# -------------------- Mode switch --------------------
mode = st.radio("Recommendation Mode:", options=["Top-N Ranking", "Similar to a Restaurant"])

if mode == "Top-N Ranking":
    probs_all = clf.predict_proba(X)[:, 1]
    df_rank = df_use.copy()
    df_rank["Match Probability"] = probs_all
    top = df_rank.sort_values("Match Probability", ascending=False).head(top_n)
    st.subheader("Top Recommended Restaurants")
    show_cols = ["Name", "Match Probability"] + sentiment_cols
    st.dataframe(top[show_cols].assign(**{"Match Probability": top["Match Probability"].round(3)}), use_container_width=True)

else:
    st.markdown("### Pick a restaurant to find similar ones")
    selectable_names = df_use["Name"].dropna().unique().tolist() if "Name" in df_use.columns else []
    if not selectable_names:
        st.warning("No 'Name' column available to select a restaurant.")
    selected = st.selectbox("Restaurant:", options=selectable_names)
    if selected:
        probs_all = clf.predict_proba(X)[:, 1]
        out = df_use.copy()
        out["Match Probability"] = probs_all
        out = out[out["Name"] != selected] if "Name" in out.columns else out
        top_sim = out.sort_values("Match Probability", ascending=False).head(top_n)
        st.subheader(f"Restaurants similar to '{selected}'")
        show_cols = ["Name", "Match Probability"] + sentiment_cols
        st.dataframe(top_sim[show_cols].assign(**{"Match Probability": top_sim["Match Probability"].round(3)}),
                     use_container_width=True)

# -------------------- (Optional) Algorithm tab --------------------
with st.expander("Show algorithm evaluation (test set)"):
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    st.markdown("**Classification Report**")
    st.code(classification_report(y_test, y_pred, digits=3), language="text")

    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    st.markdown(f"**ROC-AUC:** {auc:.3f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    st.pyplot(fig)

# -------------------- Feedback --------------------
st.markdown("## Rate Your Experience")
rating = st.slider("Rate this restaurant (1-5)", 1, 5)
feedback_comment = st.text_area("Your Feedback")

if st.button("Submit Feedback"):
    df_fb = pd.read_csv(FEEDBACK_FILE) if os.path.isfile(FEEDBACK_FILE) else pd.DataFrame(columns=["Reviews", "Comments"])
    comment_clean = str(feedback_comment).strip()
    if comment_clean and comment_clean.lower() != "nan":
        new_feedback = pd.DataFrame([{"Reviews": f"{rating} of 5 bubbles", "Comments": comment_clean}])
        df_fb = pd.concat([df_fb, new_feedback], ignore_index=True)
        df_fb.to_csv(FEEDBACK_FILE, index=False)
        st.success("Thanks for your feedback!")
    else:
        st.warning("Please enter a real comment.")

st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)

if os.path.isfile(FOOTER_IMG):
    st.image(Image.open(FOOTER_IMG), use_container_width=True)
