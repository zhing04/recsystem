import os
from pathlib import Path
import re
import html  # for safe comment rendering

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
RATING_IMG_45 = "data/Ratings/Img4.5.png"
RATING_IMG_40 = "data/Ratings/Img4.0.png"
RATING_IMG_50 = "data/Ratings/Img5.0.png"
FEEDBACK_FILE = "data/raw/feedback.csv"

# ---------- ensure feedback CSV exists ----------
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

# Sidebar icon
if not os.path.isfile(icon_path):
    st.warning(f"⚠️ Sidebar icon not found at {icon_path}")
else:
    st.sidebar.image(icon_path, use_container_width=True)

# Global styles (feedback cards)
st.markdown("""
<style>
  .feedback-card{
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:12px 14px;
    margin-bottom:12px;
    background:#ffffff;
  }
  .feedback-stars{
    font-size:1.1rem;
    margin-bottom:6px;
  }
  .feedback-text{
    font-size:0.98rem;
    color:#111827;
    line-height:1.35;
    word-wrap:break-word;
    white-space:pre-wrap;
  }
  @media (prefers-color-scheme: dark){
    .feedback-card{ border-color:#374151; background:#111827; }
    .feedback-text{ color:#e5e7eb; }
  }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def _stars_from_bubbles(text: str) -> str:
    m = re.search(r"(\d(?:\.\d)?)\s*of\s*5", str(text))
    if not m:
        return "☆☆☆☆☆"
    try:
        score = float(m.group(1))
    except:
        score = 0
    full = max(0, min(int(round(score)), 5))
    return "⭐" * full + "☆" * (5 - full)

def _parse_rating_from_reviews(text: str) -> float:
    """Extract numeric rating from strings like '4.5 of 5 bubbles'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*of\s*5", str(text))
    try:
        return float(m.group(1)) if m else np.nan
    except:
        return np.nan

def render_feedback_grid(max_rows: int = 10):
    """Compact two-column feedback with consistent padding & clear text color."""
    try:
        df_fb = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"⚠️ Could not load feedback: {e}")
        return
    if df_fb.empty:
        st.caption("No feedback yet.")
        return

    # Hide empty/'nan' comments
    df_fb['Comments'] = df_fb['Comments'].astype(str)
    mask_valid = df_fb['Comments'].str.strip().ne('') & df_fb['Comments'].str.strip().str.lower().ne('nan')
    df_fb = df_fb[mask_valid]
    if df_fb.empty:
        st.caption("No feedback yet.")
        return

    last = df_fb.tail(max_rows).reset_index(drop=True)
    cols = st.columns(2)

    for i, row in last.iterrows():
        col = cols[i % 2]
        stars = _stars_from_bubbles(row.get("Reviews", ""))
        raw_comment = str(row.get("Comments", "")).strip() or "— (no comment) —"
        safe_comment = html.escape(raw_comment)
        col.markdown(
            f"""
            <div class="feedback-card">
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
    st.error(f"Could not load dataset at {DATA_TRIP}. Error: {e}")
    st.stop()

# minimal cleaning to match your original prep
if {'Street Address', 'Location'}.issubset(df.columns):
    df["Location"] = df["Street Address"].astype(str) + ', ' + df["Location"].astype(str)
    df = df.drop(['Street Address'], axis=1)

# drop rows with missing type only if column exists (not needed for supervised)
if 'Type' in df.columns:
    df = df[df['Type'].notna()]

df = df.drop_duplicates(subset='Name').reset_index(drop=True)

# ---------- header & intro ----------
st.markdown("<h1 style='text-align: center;'>Restaurant Based Recommendation</h1>", unsafe_allow_html=True)

st.markdown("""
### Welcome to the Supervised Restaurant Recommender!

This version uses **Supervised Learning (Gradient Boosting)** to rank restaurants.
It learns from numeric features (e.g., sentiment scores or ratings) and predicts the **probability** that a restaurant belongs to the **top class**.

**Pipeline:**
1) Build **features (X)** from sentiment columns (preferred) or numeric ratings (fallback).  
2) Create **labels (y)** by marking the **top X%** as positive class.  
3) Train a **Gradient Boosting** classifier.  
4) Rank all restaurants by predicted probability and show the **Top-N**.
""")

if os.path.isfile(COVER_IMG):
    st.image(Image.open(COVER_IMG), use_container_width=True)

# ---------- sidebar controls ----------
st.sidebar.header("Supervised Settings")
top_n = st.sidebar.slider("Top-N restaurants to display", 5, 30, 10, 1)
q = st.sidebar.slider("Positive class quantile (Top % threshold)", 0.50, 0.90, 0.70, 0.05,
                      help="Top (1−q) fraction becomes positive class. q=0.70 → top 30% are positives.")

# ---------- feature assembly ----------
# Prefer sentiment columns if present
sentiment_cols = [c for c in df.columns if "Sentiment" in c]
numeric_cols = []

if sentiment_cols:
    X_source_cols = sentiment_cols
else:
    # Fallback: try to derive numeric rating(s)
    if 'Reviews' in df.columns:
        df['Rating_num'] = df['Reviews'].apply(_parse_rating_from_reviews)  # 0..5
        numeric_cols.append('Rating_num')
    if 'Ratings' in df.columns and pd.api.types.is_numeric_dtype(df['Ratings']):
        numeric_cols.append('Ratings')

    # keep only numeric, non-null columns
    X_source_cols = [c for c in numeric_cols if c in df.columns]
    df = df.dropna(subset=X_source_cols)

if not X_source_cols:
    st.error("No suitable numeric features found. Expected sentiment columns (e.g., 'Average Food Sentiment') "
             "or numeric ratings (parsed from 'Reviews' or a numeric 'Ratings' column).")
    st.stop()

# ---------- build X, y ----------
X = df[X_source_cols].astype(float).values

# Label rule:
# If using sentiment columns → composite = mean across sentiments.
# If using ratings → composite = mean across available numeric rating features.
composite = df[X_source_cols].astype(float).mean(axis=1)
threshold = np.quantile(composite, q)
y = (composite >= threshold).astype(int).values

# Degenerate guard
if y.sum() == 0 or y.sum() == len(y):
    st.warning("Labels degenerated (all the same). Showing a simple sort by composite score instead.")
    out = df.copy()
    out["Composite"] = composite
    top = out.sort_values("Composite", ascending=False).head(top_n)
    show_cols = ["Name"] + (["url"] if "url" in top.columns else []) + (["Composite"] if "Composite" in top.columns else []) + X_source_cols
    st.dataframe(top[show_cols])
    # Footer image
    if os.path.isfile(FOOTER_IMG):
        st.image(Image.open(FOOTER_IMG), use_container_width=True)
    # Feedback section continues below
else:
    # ---------- train/test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ---------- train classifier ----------
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    clf.fit(X_train, y_train)

    # ---------- evaluate on test ----------
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    algo_tab, res_tab = st.tabs(["🔬 Algorithm", "🏁 Results"])

    with algo_tab:
        st.subheader("Model Evaluation (Test Set)")
        st.markdown(f"**Features:** {', '.join(X_source_cols)}  \n"
                    f"**Positive class:** Top **{int((1-q)*100)}%** by composite score")

        st.markdown("**Classification Report**")
        st.code(classification_report(y_test, y_pred, digits=3), language="text")

        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        st.markdown(f"**ROC-AUC:** {auc:.3f}")

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0,1], [0,1], linestyle="--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(True)
        st.pyplot(fig)

        # Feature importance (tree models)
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            order = np.argsort(imp)[::-1]
            imp_df = pd.DataFrame({
                "Feature": np.array(X_source_cols)[order],
                "Importance": imp[order]
            })
            st.markdown("**Feature Importance**")
            st.bar_chart(imp_df.set_index("Feature"))
        else:
            st.caption("Model does not expose feature importances.")

    with res_tab:
        st.subheader("Top Recommendations (Ranked by Predicted Probability)")
        probs_all = clf.predict_proba(X)[:, 1]
        out = df.copy()
        out["Match Probability"] = probs_all
        top = out.sort_values("Match Probability", ascending=False).head(top_n)

        display_cols = ["Name", "Match Probability"] + X_source_cols
        if "url" in top.columns:
            display_cols.insert(1, "url")

        to_show = top[display_cols].copy()
        to_show["Match Probability"] = to_show["Match Probability"].round(3)
        st.dataframe(to_show, use_container_width=True)

# ---------- feedback ----------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # (re)ensure file exists
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
        st.warning("Please enter a real comment (not empty).")

# ---------- recent feedback ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)

st.text("")
if os.path.isfile(FOOTER_IMG):
    st.image(Image.open(FOOTER_IMG), use_container_width=True)
