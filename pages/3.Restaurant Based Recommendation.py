import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------- paths (match your repo layout) ----------
DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"
APP_ICON = "data/App_icon.png"
COVER_IMG = "data/food_cover.jpg"
FOOTER_IMG = "data/food_2.jpg"
RATING_IMG_45 = "data/Ratings/Img4.5.png"
RATING_IMG_40 = "data/Ratings/Img4.0.png"
RATING_IMG_50 = "data/Ratings/Img5.0.png"
FEEDBACK_FILE = "data/raw/feedback.csv"

# ---------- ensure feedback CSV exists ----------
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

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
    """Compact two-column feedback with consistent padding & clear text color."""
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"⚠️ Could not load feedback: {e}")
        return
    if df.empty:
        st.caption("No feedback yet.")
        return

    last = df.tail(max_rows).reset_index(drop=True)
    cols = st.columns(2)

    for i, row in last.iterrows():
        col = cols[i % 2]
        stars = _stars_from_bubbles(row.get("Reviews", ""))
        raw_comment = str(row.get("Comments", "")).strip() or "— (no comment) —"
        safe_comment = html.escape(raw_comment)  # ensure safe HTML
        col.markdown(
            f"""
            <div class="feedback-card">
              <div class="feedback-stars">{stars}</div>
              <div class="feedback-text">{safe_comment}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')
st.sidebar.image(APP_ICON, use_container_width=True)

# ---------- load dataset (same logic as original) ----------
df = pd.read_csv(DATA_TRIP)
df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Type'].notna()]
df = df.drop_duplicates(subset='Name').reset_index(drop=True)

# ---------- header & intro (original copy) ----------
st.markdown("<h1 style='text-align: center;'>Restaurant Based</h1>", unsafe_allow_html=True)

st.markdown("""
### Welcome to Restaurant Recommender!

Looking for the perfect place to dine? Look no further! Our Restaurant Recommender is here to help you discover the finest dining experiences tailored to your taste.

### How It Works:

1. **Select Your Favorite Restaurant:**
   Choose from a list of renowned restaurants that pique your interest.

2. **Explore Similar Gems:**
   Our advanced recommendation system analyzes customer reviews and ratings to suggest similar restaurants you might love.

3. **Discover Your Next Culinary Adventure:**
   Dive into detailed information about each recommended restaurant, including ratings, reviews, cuisine types, locations, and contact details.

4. **Enjoy Your Meal:**
   With our recommendations in hand, savor a delightful dining experience at your chosen restaurant!

### Start Your Culinary Journey Now!

Begin exploring the diverse culinary landscape and uncover hidden gastronomic treasures with Restaurant Recommender.
↓
""")

st.image(Image.open(COVER_IMG), use_container_width=True)

st.markdown("### Select Restaurant")

# ---------- user selection ----------
name = st.selectbox('Select the Restaurant you like', list(df['Name'].unique()))

# ---------- recommender (fixed to avoid KeyError) ----------
def recom(dataframe, name):
    dfw = dataframe.copy()
    for col in ["Trip_advisor Url", "Menu"]:
        if col in dfw.columns:
            dfw = dfw.drop(columns=[col])

    # TF-IDF on 'Type' (no comment filtering here)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dfw['Type'].fillna('').astype(str))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Map name -> index
    indices = pd.Series(dfw.index, index=dfw['Name']).drop_duplicates()
    if name not in indices:
        st.warning("The selected restaurant isn’t available for similarity right now. Please pick another.")
        return

    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    if tfidf_matrix.shape[0] < 2:
        st.info("Not enough data to compute similar restaurants.")
        return

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # top-10, skip self
    restaurant_indices = [i for i, _ in sim_scores]

    # Top 10 list (include Ratings when available)
    cols = ['Name'] + (['Ratings'] if 'Ratings' in dfw.columns else [])
    recommended = dfw.iloc[restaurant_indices][cols].copy()
    if 'Ratings' in recommended.columns:
        recommended = recommended.sort_values(by='Ratings', ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")
    title = st.selectbox(
        'Restaurants most similar [Based on user ratings(collaborative)]',
        list(recommended['Name'])
    )

    if title in dfw['Name'].values:
        details = dfw.loc[dfw['Name'] == title].iloc[0]

        # Rating bubbles → images
        if 'Reviews' in details:
            st.markdown("### Restaurant Rating:")
            rv = str(details['Reviews'])
            if rv == '4.5 of 5 bubbles':
                st.image(Image.open(RATING_IMG_45), use_container_width=True)
            elif rv == '4 of 5 bubbles':
                st.image(Image.open(RATING_IMG_40), use_container_width=True)
            elif rv == '5 of 5 bubbles':
                st.image(Image.open(RATING_IMG_50), use_container_width=True)

        # Comments (if present and meaningful)
        if 'Comments' in dfw.columns:
            cmt = details.get('Comments', None)
            if pd.notna(cmt) and cmt != "No Comments":
                st.markdown("### Comments:")
                st.warning(str(cmt))

        # Type / Category
        if 'Type' in details:
            st.markdown("### Restaurant Category:")
            st.error(str(details['Type']))

        # Address
        if 'Location' in details:
            st.markdown("### The Address:")
            st.success(str(details['Location']))

        # Contact
        if 'Contact Number' in details and str(details['Contact Number']) != "Not Available":
            st.markdown("### Contact Details:")
            st.info("Phone: " + str(details['Contact Number']))

    st.text("")
    st.image(Image.open(FOOTER_IMG), use_container_width=True)

# ---------- run recommender ----------
recom(df, name)

# ---------- feedback (original behavior, updated path) ----------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # (re)ensure file exists
    if not os.path.isfile(FEEDBACK_FILE):
        pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

    # append
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': feedback_comment}])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

    st.success('Thanks for your feedback!')

# ---------- last 10 feedback (compact 2-column, boxed) ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)
