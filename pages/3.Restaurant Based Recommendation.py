import os
from pathlib import Path
import re
import html  # for safe comment rendering

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ---------- paths ----------
DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"

icon_path = "data/App_icon.png"
if not os.path.isfile(icon_path):
    st.warning(f"⚠️ Sidebar icon not found at {icon_path}")
else:
    st.sidebar.image(icon_path, use_container_width=True)

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

# Global styles
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
    full = max(0, min(int(score), 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_grid(max_rows: int = 10):
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"⚠️ Could not load feedback: {e}")
        return
    if df.empty:
        st.caption("No feedback yet.")
        return

    df['Comments'] = df['Comments'].astype(str)
    mask_valid = df['Comments'].str.strip().ne('') & df['Comments'].str.strip().str.lower().ne('nan')
    df = df[mask_valid]
    if df.empty:
        st.caption("No feedback yet.")
        return

    last = df.tail(max_rows).reset_index(drop=True)
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
df = pd.read_csv(DATA_TRIP)
df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Type'].notna()]
df = df.drop_duplicates(subset='Name').reset_index(drop=True)

# ---------- header ----------
st.markdown("<h1 style='text-align: center;'>Restaurant Based Recommendation</h1>", unsafe_allow_html=True)

st.markdown("""
### Welcome to Restaurant Recommender!

Looking for the perfect place to dine? Look no further! Our Restaurant Recommender is here to help you discover the finest dining experiences tailored to your taste.

### Recommendation Modes:

- **Content-Based Filtering:**  
  Finds restaurants similar to the one you like (based on category/type).
  
- **Supervised Learning (Gradient Boosting):**  
  Learns from ratings/sentiments and predicts which restaurants are most likely to be in the "top class".

↓ Choose a mode below to start!
""")

st.image(Image.open(COVER_IMG), use_container_width=True)

# ---------- mode selection ----------
mode = st.radio("Select Recommendation Mode:", ["Content-Based", "Supervised Learning"])

# ---------- user input ----------
if mode == "Content-Based":
    st.markdown("### Select Restaurant")
    name = st.selectbox('Select the Restaurant you like', list(df['Name'].unique()))

# ---------- content-based recommender ----------
def recom(dataframe, name):
    dfw = dataframe.copy()
    for col in ["Trip_advisor Url", "Menu"]:
        if col in dfw.columns:
            dfw = dfw.drop(columns=[col])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dfw['Type'].fillna('').astype(str))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(dfw.index, index=dfw['Name']).drop_duplicates()
    if name not in indices:
        st.warning("The selected restaurant isn’t available. Please pick another.")
        return

    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    if tfidf_matrix.shape[0] < 2:
        st.info("Not enough data to compute similar restaurants.")
        return

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    restaurant_indices = [i for i, _ in sim_scores]

    cols = ['Name'] + (['Ratings'] if 'Ratings' in dfw.columns else [])
    recommended = dfw.iloc[restaurant_indices][cols].copy()
    if 'Ratings' in recommended.columns:
        recommended = recommended.sort_values(by='Ratings', ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")
    st.dataframe(recommended)

# ---------- supervised recommender ----------
def supervised_recom(dataframe, top_n=10):
    # Ensure sentiment columns exist
    cols = [c for c in dataframe.columns if "Sentiment" in c]
    if not cols:
        st.error("No sentiment columns found in dataset.")
        return

    dfw = dataframe.dropna(subset=cols).copy()
    comp = dfw[cols].mean(axis=1)
    y = (comp >= np.quantile(comp, 0.70)).astype(int).values
    X = dfw[cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(dfw[cols].values)[:, 1]
    dfw["Match Probability"] = probs

    top = dfw.sort_values("Match Probability", ascending=False).head(top_n)

    st.markdown("## Top Recommended Restaurants (Supervised Learning)")
    st.dataframe(top[["Name", "Match Probability", *cols]])

# ---------- run recommendation ----------
if mode == "Content-Based":
    recom(df, name)
else:
    supervised_recom(df, top_n=10)

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
        st.warning("Please enter a real comment (not empty).")

# ---------- recent feedback ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)

st.text("")
st.image(Image.open(FOOTER_IMG), use_container_width=True)
