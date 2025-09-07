import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

ICON_PATH = 'data/App_icon.png'
FOOTER_IMG = 'data/food_2.jpg'
RATING_MAP = {
    '4.5': 'data/Ratings/Img4.5.png',
    '4':   'data/Ratings/Img4.0.png',
    '5':   'data/Ratings/Img5.0.png',
}
FEEDBACK_FILE = "data/raw/feedback.csv"

# Ensure feedback CSV exists
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=["Reviews", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

# -----------------------------
# Helpers
# -----------------------------
def safe_image(place, path: str):
    p = Path(path)
    if p.exists():
        place.image(str(p), use_container_width=True)

def stars_from_bubbles(text: str) -> str:
    m = re.search(r"(\d(?:\.\d)?)", str(text))
    if not m:
        return "☆☆☆☆☆"
    try:
        score = float(m.group(1))
    except:
        score = 0
    full = max(0, min(int(score), 5))
    return "⭐" * full + "☆" * (5 - full)

def render_feedback_grid_with_delete(max_rows: int = 10):
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except Exception as e:
        st.caption(f"⚠️ Could not load feedback: {e}")
        return

    if df.empty:
        st.caption("No feedback yet.")
        return

    last = df.tail(max_rows)
    cols = st.columns(2)
    to_delete = set()

    for i, (abs_idx, row) in enumerate(last.iterrows()):
        col = cols[i % 2]
        with col.container(border=True):
            stars = stars_from_bubbles(row.get("Reviews", ""))
            comment = str(row.get("Comments", "")).strip() or "— (no comment) —"
            st.markdown(f"<div style='font-size:1.05rem'>{stars}</div>", unsafe_allow_html=True)
            st.write(comment)
            if st.checkbox("Delete", key=f"del_state_{abs_idx}"):
                to_delete.add(abs_idx)

    if to_delete and st.button(f"Delete {len(to_delete)} selected", type="primary"):
        df = df.drop(index=list(to_delete))
        df.to_csv(FEEDBACK_FILE, index=False)
        st.success("Deleted. Refreshing…")
        st.rerun()

# -----------------------------
# Sidebar logo
# -----------------------------
safe_image(st.sidebar, ICON_PATH)

st.markdown("<h1 style='text-align: center;'>Restaurants</h1>", unsafe_allow_html=True)

# -----------------------------
# Load per-state data
# -----------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Street Address" in df.columns and "Location" in df.columns:
        df["Location"] = df["Street Address"].fillna("").astype(str) + ", " + df["Location"].fillna("").astype(str)
        df = df.drop(columns=["Street Address"])
    return df

California  = load_and_clean('data/California/California.csv')
New_York    = load_and_clean('data/New York/New_York.csv')
New_Jersey  = load_and_clean('data/New Jersey/New_Jersey.csv')
Texas       = load_and_clean('data/Texas/Texas.csv')
Washington  = load_and_clean('data/Washington/Washington.csv')

# -----------------------------
# UI
# -----------------------------
option = st.selectbox('Select Your State', ('New York', 'New Jersey', 'California', 'Texas', 'Washington'))

def details(dataframe: pd.DataFrame):
    unique_restaurants = dataframe['Name'].drop_duplicates().head(20)
    title = st.selectbox('Select Your Restaurant (Top 20)', unique_restaurants)

    if title in dataframe['Name'].values:
        idx = dataframe['Name'].eq(title).idxmax()

        Reviews = str(dataframe.at[idx, 'Reviews'])
        st.subheader("Restaurant Rating:-")
        img_path = RATING_MAP.get(Reviews)
        if img_path:
            safe_image(st, img_path)

        if 'Comments' in dataframe.columns:
            comment = dataframe.at[idx, 'Comments']
            if pd.notna(comment) and comment != "No Comments":
                st.subheader("Comments:-")
                st.warning(str(comment))

        Type = dataframe.at[idx, 'Type']
        st.subheader("Restaurant Category:-")
        st.error(str(Type))

        Location = dataframe.at[idx, 'Location']
        st.subheader("The Address:-")
        st.success(str(Location))

        contact_no = dataframe.at[idx, 'Contact Number']
        if str(contact_no) != "Not Available":
            st.subheader("Contact Details:-")
            st.info('Phone:- ' + str(contact_no))

    st.text("")
    safe_image(st, FOOTER_IMG)

if option == 'New Jersey':
    details(New_Jersey)
elif option == 'New York':
    details(New_York)
elif option == 'California':
    details(California)
elif option == 'Texas':
    details(Texas)
elif option == 'Washington':
    details(Washington)

# -----------------------------
# Feedback
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

st.subheader("Recent Feedback")
render_feedback_grid_with_delete(max_rows=10)
