import pandas as pd
import streamlit as st
import os
from pathlib import Path
from PIL import Image
import re
import html

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

# ---------- small helpers ----------
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

# ---------- Streamlit config ----------
st.set_page_config(layout='centered', initial_sidebar_state='expanded')
# Global styles for feedback cards
st.markdown("""
<style>
  .feedback-card{
    border:1px solid #e5e7eb;          /* light gray */
    border-radius:12px;
    padding:12px 14px;                  /* fixed, consistent padding */
    margin-bottom:12px;
    background:#ffffff;                 /* white card */
  }
  .feedback-stars{
    font-size:1.1rem;
    margin-bottom:6px;
  }
  .feedback-text{
    font-size:0.98rem;
    color:#111827;                      /* gray-900 (much darker than gray) */
    line-height:1.35;
    word-wrap:break-word;
    white-space:pre-wrap;               /* keep user line breaks */
  }

  /* Dark mode adjustments (if user’s device is in dark mode) */
  @media (prefers-color-scheme: dark){
    .feedback-card{
      border-color:#374151;             /* gray-700 */
      background:#111827;               /* near black */
    }
    .feedback-text{
      color:#e5e7eb;                    /* gray-200 */
    }
  }
</style>
""", unsafe_allow_html=True)

st.sidebar.image(APP_ICON, use_container_width=True)
st.markdown("<h1 style='text-align: center;'>State Based Recommendation</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: justify;'>Embark on a gastronomic journey with our curated selection of restaurants across various states. Whether you're craving the bold flavors of Texas barbecue, the diverse cuisine of California, or the iconic dishes of New York, we've got you covered. Our app is your passport to culinary exploration, delivering personalized recommendations based on real user reviews and ratings.</p>

<p style='text-align: justify;'>Discover hidden gems, indulge in mouthwatering dishes, and immerse yourself in the vibrant food culture of your chosen destination. From cozy cafes to upscale fine dining establishments, there's something for every palate and occasion.</p>
""", unsafe_allow_html=True)

# ---------- load per-state data (paths adjusted to your repo) ----------
California = pd.read_csv('data/California/California.csv', sep=',')
California["Location"] = California["Street Address"] + ', ' + California["Location"]
California = California.drop(['Street Address'], axis=1)

New_York = pd.read_csv('data/New York/New_York.csv', sep=',')
New_York["Location"] = New_York["Street Address"] + ', ' + New_York["Location"]
New_York = New_York.drop(['Street Address'], axis=1)

New_Jersey = pd.read_csv('data/New Jersey/New_Jersey.csv', sep=',')
New_Jersey["Location"] = New_Jersey["Street Address"] + ', ' + New_Jersey["Location"]
New_Jersey = New_Jersey.drop(['Street Address'], axis=1)

Texas = pd.read_csv('data/Texas/Texas.csv', sep=',')
Texas["Location"] = Texas["Street Address"] + ', ' + Texas["Location"]
Texas = Texas.drop(['Street Address'], axis=1)

Washington = pd.read_csv('data/Washington/Washington.csv', sep=',')
Washington["Location"] = Washington["Street Address"] + ', ' + Washington["Location"]
Washington = Washington.drop(['Street Address'], axis=1)

# ---------- pick state ----------
option = st.selectbox('Select Your State', ('New York', 'New Jersey', 'California', 'Texas', 'Washington'))

# ---------- details renderer (original logic, minor cleanup) ----------
def details(dataframe):
    unique_restaurants = dataframe['Name'].drop_duplicates().head(20)
    title = st.selectbox('Select Your Restaurant (Top 20)', unique_restaurants)

    if title in dataframe['Name'].values:
        idx = dataframe['Name'].eq(title).idxmax()

        Reviews = str(dataframe.at[idx, 'Reviews'])
        st.subheader("Restaurant Rating:-")

        if Reviews == '4.5':
            st.image(Image.open(RATING_IMG_45), use_container_width=True)
        elif Reviews == '4':
            st.image(Image.open(RATING_IMG_40), use_container_width=True)
        elif Reviews == '5':
            st.image(Image.open(RATING_IMG_50), use_container_width=True)

        if 'Comments' in dataframe.columns:
            comment = dataframe.at[idx, 'Comments']
            if comment != "No Comments":
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

# ---------- feedback (same behavior as original, path updated) ----------
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    if not os.path.isfile(FEEDBACK_FILE):
        pd.DataFrame(columns=['Reviews', 'Comments']).to_csv(FEEDBACK_FILE, index=False)

    feedback_df = pd.read_csv(FEEDBACK_FILE)
    new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': feedback_comment}])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

    st.success('Thanks for your feedback!')

# ---------- last 10 feedback (compact 2-column) ----------
st.subheader("Recent Feedback")
render_feedback_grid(max_rows=10)
