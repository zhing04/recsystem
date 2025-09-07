import pandas as pd
import streamlit as st
import os
from pathlib import Path
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

# ---------- paths (match your repo layout) ----------
DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"
APP_ICON = "data/App_icon.png"
COVER_IMG = "data/food_cover.jpg"
FOOTER_IMG = "data/food_2.jpg"
RATING_IMG_45 = "data/Ratings/Img4.5.png"
RATING_IMG_40 = "data/Ratings/Img4.0.png"
RATING_IMG_50 = "data/Ratings/Img5.0.png"
FEEDBACK_FILE = "data/raw/feedback.csv"

# Ensure feedback CSV exists with correct columns
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

# ---------- small helpers ----------
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
    """Compact two-column feedback (no delete)."""
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
        with col.container(border=True):
            stars = _stars_from_bubbles(row.get("Reviews", ""))
            comment = str(row.get("Comments", "")).strip() or "— (no comment) —"
            col.markdown(f"<div style='font-size:1.05rem'>{stars}</div>", unsafe_allow_html=True)
            col.write(comment)

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')
st.sidebar.image(APP_ICON, use_container_width=True)

# ---------- load dataset (same logic as original) ----------
df = pd.read_csv(DATA_TRIP)
df["Location"] = df["Street Address"] + ', ' + df["Location"]
df = df.drop(['Street Address'], axis=1)
df = df[df['Type'].notna()]
df = df.drop_duplicates(subset='Name')
df = df.reset_index(drop=True)

# ---------- header & intro (original copy) ----------
st.markdown("<h1 style='text-align: center;'>Recommended</h1>", unsafe_allow_html=True)

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

image = Image.open(COVER_IMG)
st.image(image, use_container_width=True)

st.markdown("### Select Restaurant")

# ---------- user selection ----------
name = st.selectbox('Select the Restaurant you like', list(df['Name'].unique()))

def recom(dataframe, name):
    dataframe = dataframe.drop(["Trip_advisor Url", "Menu"], axis=1)

    # Filter out restaurants without comments
    dataframe = dataframe[dataframe['Comments'].notna() & (dataframe['Comments'] != "No Comments")]

    # Creating recommendations based on 'Type'
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['Type'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Mapping restaurant names to their indices
    indices = pd.Series(dataframe.index, index=dataframe.Name).drop_duplicates()

    # Index of the anchor
    idx = indices[name]
    if isinstance(idx, pd.Series):
        idx = idx[0]

    # Similarities
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top-10 (skip self)
    restaurant_indices = [i[0] for i in sim_scores]

    # Top 10
    recommended = dataframe.iloc[restaurant_indices]
    recommended = recommended[['Name', 'Ratings']]
    recommended = recommended.sort_values(by='Ratings', ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")

    # Pick from recommended
    title = st.selectbox('Restaurants most similar [Based on user ratings(collaborative)]', recommended['Name'])
    if title in dataframe['Name'].values:
        details = dataframe[dataframe['Name'] == title].iloc[0]
        reviews = details['Reviews']

        st.markdown("### Restaurant Rating:")

        # Review bubbles → images
        if reviews == '4.5 of 5 bubbles':
            st.image(Image.open(RATING_IMG_45), use_container_width=True)
        elif reviews == '4 of 5 bubbles':
            st.image(Image.open(RATING_IMG_40), use_container_width=True)
        elif reviews == '5 of 5 bubbles':
            st.image(Image.open(RATING_IMG_50), use_container_width=True)

        # Comments
        if 'Comments' in dataframe.columns:
            comment = details['Comments']
            if comment != "No Comments":
                st.markdown("### Comments:")
                st.warning(comment)

        # Type
        rest_type = details['Type']
        st.markdown("### Restaurant Category:")
        st.error(rest_type)

        # Location
        location = details['Location']
        st.markdown("### The Address:")
        st.success(location)

        # Contact
        contact_no = details['Contact Number']
        if contact_no != "Not Available":
            st.markdown("### Contact Details:")
            st.info('Phone: ' + contact_no)

    st.text("")
    st.image(Image.open(FOOTER_IMG), use_container_width=True)

# ---------- run recommender ----------
recom(df, name)

# ---------- feedback (same behavior as original, path updated) ----------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # Create the CSV file if it doesn't exist (already ensured above; safe to keep)
    if not os.path.isfile(FEEDBACK_FILE):
        feedback_df = pd.DataFrame(columns=['Reviews', 'Comments'])
        feedback_df.to_csv(FEEDBACK_FILE, index=False)

    # Load existing, append, save
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': feedback_comment}])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

    st.success('Thanks for your feedback!')

# ---------- last 10 feedback (compact 2-column) ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)
