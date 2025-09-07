import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"
ICON_PATH = "data/App_icon.png"           # keep logo in sidebar (same as other pages)
COVER_IMG = "data/food_cover.jpg"
FOOTER_IMG = "data/food_2.jpg"
RATINGS_IMG = {
    "4.5 of 5 bubbles": "data/Ratings/Img4.5.png",
    "4 of 5 bubbles":   "data/Ratings/Img4.0.png",
    "5 of 5 bubbles":   "data/Ratings/Img5.0.png",
}
FEEDBACK_FILE = "data/raw/feedback.csv"   # unified feedback CSV

# ensure feedback CSV exists (Reviews, Comments)
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=["Reviews", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

# -----------------------------
# Small helpers
# -----------------------------
def safe_image(place, path: str):
    p = Path(path)
    if p.exists():
        place.image(str(p), use_container_width=True)
    else:
        place.caption(f"⚠️ Image not found: {p}")

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

def render_feedback_cards(df: pd.DataFrame, max_rows: int = 10):
    if df.empty:
        st.caption("No feedback yet.")
        return
    for _, row in df.tail(max_rows).iterrows():
        stars = stars_from_bubbles(row.get("Reviews", ""))
        comment = str(row.get("Comments", "")).strip()
        with st.container(border=True):
            st.markdown(f"<div style='font-size:1.1rem'>{stars}</div>", unsafe_allow_html=True)
            st.write(comment or "— (no comment) —")

# -----------------------------
# Sidebar logo (keep position)
# -----------------------------
safe_image(st.sidebar, ICON_PATH)

# -----------------------------
# Load & prep data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Street Address" in df.columns and "Location" in df.columns:
        df["Location"] = df["Street Address"].fillna("").astype(str) + ", " + df["Location"].fillna("").astype(str)
        df = df.drop(columns=["Street Address"])
    if "Type" in df.columns:
        df = df[df["Type"].notna()]
    df = df.drop_duplicates(subset="Name").reset_index(drop=True)
    return df

df = load_data(DATA_TRIP)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align: center;'>Recommended</h1>", unsafe_allow_html=True)
st.markdown("""
### Welcome to Restaurant Recommender!

Looking for the perfect place to dine? Look no further! Our Restaurant Recommender is here to help you discover the finest dining experiences tailored to your taste.

**How It Works**
1. **Select Your Favorite Restaurant** – Choose a place you like.
2. **Explore Similar Gems** – We suggest similar restaurants.
3. **Discover Details** – Ratings, reviews, cuisines, locations, contacts.
4. **Enjoy Your Meal!**
""")
safe_image(st, COVER_IMG)

# -----------------------------
# Select anchor restaurant
# -----------------------------
st.markdown("### Select Restaurant")
name = st.selectbox('Select the Restaurant you like', sorted(df['Name'].dropna().unique()))

# -----------------------------
# Recommender
# -----------------------------
def recom(dataframe: pd.DataFrame, anchor_name: str):
    for col in ["Trip_advisor Url", "Menu"]:
        if col in dataframe.columns:
            dataframe = dataframe.drop(columns=[col])

    if "Comments" in dataframe.columns:
        mask = dataframe["Comments"].notna() & (dataframe["Comments"] != "No Comments")
        dataframe = dataframe[mask] if mask.any() else dataframe

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['Type'].fillna("").astype(str))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(dataframe.index, index=dataframe["Name"]).drop_duplicates()
    if anchor_name not in indices:
        st.warning("Selected restaurant not found.")
        return

    idx = indices[anchor_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    restaurant_indices = [i for i, _ in sim_scores]

    cols = ["Name"]
    if "Ratings" in dataframe.columns:
        cols.append("Ratings")
    recommended = dataframe.iloc[restaurant_indices][cols].copy()
    if "Ratings" in recommended.columns:
        recommended = recommended.sort_values(by="Ratings", ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")
    title = st.selectbox('Restaurants most similar [Based on user ratings (collaborative)]', list(recommended["Name"]))

    if title in dataframe["Name"].values:
        details = dataframe.loc[dataframe["Name"] == title].iloc[0]

        if "Reviews" in details:
            st.markdown("### Restaurant Rating:")
            img_path = RATINGS_IMG.get(str(details["Reviews"]), None)
            if img_path:
                safe_image(st, img_path)

        if "Comments" in dataframe.columns:
            comment = details["Comments"]
            if pd.notna(comment) and comment != "No Comments":
                st.markdown("### Comments:")
                st.warning(str(comment))

        if "Type" in details:
            st.markdown("### Restaurant Category:")
            st.error(str(details["Type"]))

        if "Location" in details:
            st.markdown("### The Address:")
            st.success(str(details["Location"]))

        if "Contact Number" in details and str(details["Contact Number"]) != "Not Available":
            st.markdown("### Contact Details:")
            st.info("Phone: " + str(details["Contact Number"]))

    st.text("")
    safe_image(st, FOOTER_IMG)

recom(df, name)

# -----------------------------
# Feedback (shared CSV)
# -----------------------------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    if feedback_comment.strip():
        new_feedback = pd.DataFrame([[f'{rating} of 5 bubbles', feedback_comment]],
                                    columns=['Reviews', 'Comments'])
        new_feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        st.success('✅ Thanks for your feedback!')
    else:
        st.warning("Please enter a comment before submitting.")

# Recent feedback (last 10) as themed cards
try:
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    if not feedback_df.empty:
        st.subheader("Recent Feedback")
        render_feedback_cards(feedback_df, max_rows=10)
except Exception as e:
    st.caption(f"⚠️ Could not load feedback: {e}")
