import os
from pathlib import Path
import re
import html
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ---------- paths ----------
DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"

icon_path = "data/App_icon.png"
COVER_IMG = "data/restaurant.jpg"
FOOTER_IMG = "data/food_2.jpg"
FEEDBACK_FILE = "data/raw/feedback.csv"

# ---------- ensure feedback CSV exists ----------
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

if os.path.isfile(icon_path):
    st.sidebar.image(icon_path, use_container_width=True)

# ---------- helpers ----------
def _stars_from_bubbles(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*of\s*5", str(text))
    try:
        score = float(m.group(1)) if m else 0
    except:
        score = 0
    full = max(0, min(int(round(score)), 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_grid(max_rows: int = 10):
    try:
        df_fb = pd.read_csv(FEEDBACK_FILE)
    except Exception:
        st.caption("⚠️ Could not load feedback")
        return
    if df_fb.empty:
        st.caption("No feedback yet.")
        return

    df_fb['Comments'] = df_fb['Comments'].astype(str)
    df_fb = df_fb[df_fb['Comments'].str.strip().ne('')]
    last = df_fb.tail(max_rows).reset_index(drop=True)
    cols = st.columns(2)

    for i, row in last.iterrows():
        col = cols[i % 2]
        stars = _stars_from_bubbles(row.get("Reviews", ""))
        safe_comment = html.escape(str(row.get("Comments", "")).strip())
        col.markdown(
            f"""
            <div class="feedback-card" style="border:1px solid #ccc;border-radius:8px;padding:8px;margin-bottom:8px;">
              <div class="feedback-stars">{stars}</div>
              <div class="feedback-text">{safe_comment}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- load dataset ----------
try:
    df = pd.read_csv(DATA_TRIP)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if {'Street Address', 'Location'}.issubset(df.columns):
    df["Location"] = df["Street Address"].astype(str) + ', ' + df["Location"].astype(str)
    df = df.drop(['Street Address'], axis=1)

df = df.drop_duplicates(subset='Name').reset_index(drop=True)

# ---------- header ----------
st.markdown("<h1 style='text-align: center;'>Restaurant Supervised Recommender</h1>", unsafe_allow_html=True)

st.markdown("""
This app uses **Supervised Learning (Gradient Boosting)** to rank restaurants.

- **Top-N Ranking** → shows the top restaurants overall.  
- **Similar to a Restaurant** → you pick a restaurant, and it shows the most similar ones.
""")

if os.path.isfile(COVER_IMG):
    st.image(Image.open(COVER_IMG), use_container_width=True)

# ---------- sidebar controls ----------
st.sidebar.header("Controls")
top_n = st.sidebar.slider("How many results to show (Top-N):", 5, 30, 10, 1)
q = st.sidebar.slider("Positive class quantile (Top % threshold)", 0.50, 0.90, 0.70, 0.05,
                      help="Top (1−q) fraction becomes positive class. q=0.70 → top 30% are positives.")

# ---------- features ----------
sentiment_cols = [c for c in df.columns if "Sentiment" in c]
if not sentiment_cols:
    st.error("No sentiment columns found (e.g., 'Average Food Sentiment').")
    st.stop()

X = df[sentiment_cols].astype(float).values
composite = df[sentiment_cols].mean(axis=1)
threshold = np.quantile(composite, q)
y = (composite >= threshold).astype(int).values

if y.sum() == 0 or y.sum() == len(y):
    st.warning("Labels degenerated (all same). Falling back to simple sort.")
    df["Composite"] = composite
    top = df.sort_values("Composite", ascending=False).head(top_n)
    st.dataframe(top[["Name", "Composite", *sentiment_cols]])
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ---------- mode switch ----------
mode = st.radio("Recommendation Mode:", ["Top-N Ranking", "Similar to a Restaurant"])

if mode == "Top-N Ranking":
    probs_all = clf.predict_proba(X)[:, 1]
    df["Match Probability"] = probs_all
    top = df.sort_values("Match Probability", ascending=False).head(top_n)
    st.subheader("Top Recommended Restaurants")
    st.dataframe(top[["Name", "Match Probability", *sentiment_cols]])

else:
    selected = st.selectbox("Pick a restaurant:", df["Name"].unique())
    if selected:
        probs_all = clf.predict_proba(X)[:, 1]
        out = df.copy()
        out["Match Probability"] = probs_all
        out = out[out["Name"] != selected]
        top_sim = out.sort_values("Match Probability", ascending=False).head(top_n)
        st.subheader(f"Restaurants similar to **{selected}**")
        st.dataframe(top_sim[["Name", "Match Probability", *sentiment_cols]])

# ---------- feedback ----------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    if not os.path.isfile(FEEDBACK_FILE):
        pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

    df_fb = pd.read_csv(FEEDBACK_FILE)
    comment_clean = str(feedback_comment).strip()
    if comment_clean and comment_clean.lower() != 'nan':
        new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': comment_clean}])
        df_fb = pd.concat([df_fb, new_feedback], ignore_index=True)
        df_fb.to_csv(FEEDBACK_FILE, index=False)
        st.success('Thanks for your feedback!')
    else:
        st.warning("Please enter a real comment.")

st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)

if os.path.isfile(FOOTER_IMG):
    st.image(Image.open(FOOTER
