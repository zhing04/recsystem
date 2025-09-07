import os
from pathlib import Path
import re
import html  # for safe comment rendering

import pandas as pd
import streamlit as st
from PIL import Image

# ---------- paths (match your repo layout) ----------
APP_ICON = 'data/App_icon.png'
FOOTER_IMG = 'data/food_2.jpg'
RATING_IMG_45 = 'data/Ratings/Img4.5.png'
RATING_IMG_40 = 'data/Ratings/Img4.0.png'
RATING_IMG_50 = 'data/Ratings/Img5.0.png'
FEEDBACK_FILE = "data/raw/feedback.csv"

# Ensure feedback CSV exists with correct columns
Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(FEEDBACK_FILE):
    pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')
# Global styles for feedback cards
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

st.sidebar.image(APP_ICON, use_container_width=True)
st.markdown("<h1 style='text-align: center;'>State Based Recommendation</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: justify;'>Embark on a gastronomic journey with our curated selection of restaurants across various states. Whether you're craving the bold flavors of Texas barbecue, the diverse cuisine of California, or the iconic dishes of New York, we've got you covered.</p>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def _stars_from_bubbles(text: str) -> str:
    m = re.search(r"(\d(?:\.\d)?)", str(text))
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

    # Hide empty/'nan' comments
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
        # Stars from 'Reviews'
        txt_reviews = str(row.get("Reviews", ""))
        m = re.search(r"(\d(?:\.\d)?)\s*of\s*5", txt_reviews)
        try:
            sc = float(m.group(1)) if m else 0
        except:
            sc = 0
        sc = max(0, min(int(sc), 5))
        stars = "⭐" * sc + "☆" * (5 - sc)

        # Comment
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

def rating_to_image_path(value) -> str | None:
    """
    Accepts '4', '4.0', '4.5', '5', or '4.5 of 5 bubbles' and maps to the PNG.
    """
    s = str(value).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        x = float(m.group(1))
    except:
        return None

    if 4.75 <= x <= 5.1:
        return RATING_IMG_50
    if 4.25 <= x < 4.75:
        return RATING_IMG_45
    if 3.75 <= x < 4.25:
        return RATING_IMG_40
    return None

# ---------- load per-state data ----------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Street Address" in df.columns and "Location" in df.columns:
        df["Location"] = df["Street Address"] + ", " + df["Location"]
        df = df.drop(columns=["Street Address"])
    return df

California  = load_and_clean('data/California/California.csv')
New_York    = load_and_clean('data/New York/New_York.csv')
New_Jersey  = load_and_clean('data/New Jersey/New_Jersey.csv')
Texas       = load_and_clean('data/Texas/Texas.csv')
Washington  = load_and_clean('data/Washington/Washington.csv')

# ---------- pick state ----------
option = st.selectbox('Select Your State', ('New York', 'New Jersey', 'California', 'Texas', 'Washington'))

# ---------- details renderer ----------
def details(dataframe):
    unique_restaurants = dataframe['Name'].drop_duplicates().head(20)
    title = st.selectbox('Select Your Restaurant (Top 20)', unique_restaurants)

    if title in dataframe['Name'].values:
        idx = dataframe['Name'].eq(title).idxmax()

        Reviews = dataframe.at[idx, 'Reviews']
        st.subheader("Restaurant Rating:-")
        img_path = rating_to_image_path(Reviews)
        if img_path:
            st.image(img_path, use_container_width=True)

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
    st.image(Image.open(FOOTER_IMG), use_container_width=True)

# ---------- route by state ----------
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

# ---------- last 10 feedback (compact 2-column) ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)
