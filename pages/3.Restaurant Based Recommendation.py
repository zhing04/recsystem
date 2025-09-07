import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =========================
# Config & constants
# =========================
st.set_page_config(layout="centered", initial_sidebar_state="expanded")

DATA_TRIP = "data/raw/TripAdvisor_RestauarantRecommendation1.csv"
ICON_PATH = "data/App_icon.png"             # <- keep logo in the SIDEBAR (same as your other pages)
COVER_IMG = "data/food_cover.jpg"
FOOTER_IMG = "data/food_2.jpg"
RATINGS_IMG = {
    "4.5 of 5 bubbles": "data/Ratings/Img4.5.png",
    "4 of 5 bubbles":   "data/Ratings/Img4.0.png",
    "5 of 5 bubbles":   "data/Ratings/Img5.0.png",
}
FEEDBACK_FILE = "data/raw/feedback.csv"     # CSV columns: Reviews, Comments

# Ensure feedback CSV exists with correct columns
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=["Reviews", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

# =========================
# Helpers
# =========================
def safe_image_to(place, path: str, **kwargs):
    """Render image to 'place' (st or st.sidebar) only if file exists."""
    p = Path(path)
    if p.exists():
        place.image(str(p), **kwargs)
    else:
        place.caption(f"⚠️ Image not found: {p}")

def stars_from_bubbles(text: str) -> str:
    """
    Convert '4 of 5 bubbles' -> '⭐⭐⭐⭐☆'
    """
    m = re.search(r"(\d(?:\.\d)?)\s*of\s*5", str(text))
    if not m:
        return "☆☆☆☆☆"
    val = m.group(1)
    try:
        score = float(val)
    except:
        return "☆☆☆☆☆"
    full = int(score)  # show integer stars (consistent with bubbles)
    full = max(0, min(full, 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_cards(df: pd.DataFrame, max_rows: int = 10):
    """
    Nicely styled feedback list (cards) that fits Streamlit theme.
    Uses only Reviews (bubbles text) & Comments (your schema).
    """
    if df.empty:
        st.caption("No feedback yet.")
        return

    show = df.tail(max_rows)
    for _, row in show.iterrows():
        stars = stars_from_bubbles(row.get("Reviews", ""))
        comment = str(row.get("Comments", "")).strip()
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:1.1rem; line-height:1.3'>{stars}</div>",
                unsafe_allow_html=True,
            )
            if comment:
                st.write(comment)
            else:
                st.caption("— (no comment) —")

# =========================
# Sidebar icon (KEEP POSITION)
# =========================
safe_image_to(st.sidebar, ICON_PATH, use_container_width=True)

# =========================
# Load & prep dataset
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    # Combine 'Street Address' + 'Location'
    if "Street Address" in df_.columns and "Location" in df_.columns:
        df_["Location"] = df_["Street Address"].fillna("").astype(str) + ", " + df_["Location"].fillna("").astype(str)
        df_ = df_.drop(columns=[c for c in ["Street Address"] if c in df_.columns], errors="ignore")
    # clean
    if "Type" in df_.columns:
        df_ = df_[df_["Type"].notna()]
    df_ = df_.drop_duplicates(subset="Name").reset_index(drop=True)
    return df_

df = load_data(DATA_TRIP)

# =========================
# Header
# =========================
st.markdown("<h1 style='text-align: center;'>Recommended</h1>", unsafe_allow_html=True)

st.markdown(
    """
### Welcome to Restaurant Recommender!

Looking for the perfect place to dine? Look no further! Our Restaurant Recommender is here to help you discover the finest dining experiences tailored to your taste.

**How It Works**
1. **Select Your Favorite Restaurant** – Choose a place you like.
2. **Explore Similar Gems** – We suggest similar restaurants.
3. **Discover Details** – Ratings, reviews, cuisines, locations, contacts.
4. **Enjoy Your Meal!**
"""
)
safe_image_to(st, COVER_IMG, use_container_width=True)

# =========================
# Select anchor restaurant
# =========================
st.markdown("### Select Restaurant")
name = st.selectbox("Select the Restaurant you like", sorted(df["Name"].dropna().unique()))

# =========================
# Recommender
# =========================
def recom(dataframe: pd.DataFrame, anchor_name: str):
    # Drop unused columns if present
    for col in ["Trip_advisor Url", "Menu"]:
        if col in dataframe.columns:
            dataframe = dataframe.drop([col], axis=1)

    # Filter where Comments are valid (if column exists)
    if "Comments" in dataframe.columns:
        mask = dataframe["Comments"].notna() & (dataframe["Comments"] != "No Comments")
        dataframe = dataframe[mask] if mask.any() else dataframe

    # TF-IDF on 'Type'
    types = dataframe["Type"].fillna("").astype(str)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(types)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Map name -> index
    indices = pd.Series(dataframe.index, index=dataframe["Name"]).drop_duplicates()

    if anchor_name not in indices:
        st.warning("Selected restaurant not found in the filtered dataset.")
        return

    idx = indices[anchor_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Similarities
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # top 10 (skip self)
    restaurant_indices = [i for i, _ in sim_scores]

    # Top 10 names + ratings (if column exists)
    cols = ["Name"]
    if "Ratings" in dataframe.columns:
        cols.append("Ratings")
    recommended = dataframe.iloc[restaurant_indices][cols].copy()
    if "Ratings" in recommended.columns:
        recommended = recommended.sort_values(by="Ratings", ascending=False)

    st.markdown("## Top 10 Restaurants you might like:")
    title = st.selectbox(
        "Restaurants most similar [Based on user ratings (collaborative)]",
        list(recommended["Name"]),
    )

    # Show details
    if title in dataframe["Name"].values:
        details = dataframe.loc[dataframe["Name"] == title].iloc[0]

        # Rating image by "Reviews" text, if present
        if "Reviews" in details:
            st.markdown("### Restaurant Rating:")
            reviews_text = details["Reviews"]
            img_path = RATINGS_IMG.get(reviews_text, None)
            if img_path:
                safe_image_to(st, img_path, use_container_width=True)

        # Comments
        if "Comments" in dataframe.columns:
            comment = details["Comments"]
            if pd.notna(comment) and comment != "No Comments":
                st.markdown("### Comments:")
                st.warning(str(comment))

        # Category / Type
        if "Type" in details:
            st.markdown("### Restaurant Category:")
            st.error(str(details["Type"]))

        # Address
        if "Location" in details:
            st.markdown("### The Address:")
            st.success(str(details["Location"]))

        # Contact
        if "Contact Number" in details and str(details["Contact Number"]) != "Not Available":
            st.markdown("### Contact Details:")
            st.info("Phone: " + str(details["Contact Number"]))

    st.text("")
    safe_image_to(st, FOOTER_IMG, use_container_width=True)

# Run recommender
recom(df, name)

# =========================
# Feedback (keeps your CSV schema: Reviews, Comments)
# =========================
st.markdown("## Rate Your Experience")
rating = st.slider("Rate this restaurant (1-5)", 1, 5, 3)
feedback_comment = st.text_area("Your Feedback")

if st.button("Submit Feedback"):
    if feedback_comment.strip():
        # Append a new row without rewriting the header
        new_row = pd.DataFrame([[f"{rating} of 5 bubbles", feedback_comment]], columns=["Reviews", "Comments"])
        new_row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        st.success("✅ Thanks for your feedback!")
        # Show immediately
        try:
            df_tmp = pd.read_csv(FEEDBACK_FILE)
            st.subheader("Recent Feedback")
            render_feedback_cards(df_tmp, max_rows=10)
        except Exception as e:
            st.caption(f"⚠️ Could not load feedback file: {e}")
    else:
        st.warning("Please enter a comment before submitting.")

# Always show the latest feedback list (nice cards)
try:
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    if not feedback_df.empty:
        st.subheader("Recent Feedback")
        render_feedback_cards(feedback_df, max_rows=10)
except Exception as e:
    st.caption(f"⚠️ Could not load feedback file: {e}")
