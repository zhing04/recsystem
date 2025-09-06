import streamlit as st
import pandas as pd

# Add title and description for the app
image_url = "https://images.unsplash.com/photo-1525648199074-cee30ba79a4a?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Custom CSS for the app
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
    }}
    .table-container {{
        background-color: rgba(50, 50, 50, 0.9); 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); 
        overflow-x: auto; 
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: auto; 
    th, td {{
        border: 1px solid #555; 
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: #333; 
        color: #ffffff; 
    }}
    td {{
        color: #ffffff; 
    }}
    td.url {{
        color: #1a0dab; 
    }}
    .stButton button {{
        background-color: #ffd54b;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }}
    .stButton button:hover {{
        background-color: #ffd54b;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown(
    """
    <h3 style='font-family:Forte; font-size:36px; text-align:center;'>
    Restaurant Recommendation System
    </h3>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='font-family:"Gill Sans MT", sans-serif; font-size:24px;'>
    Get the best restaurant recommendations based on customer experience!
    </h4>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='font-family:"Gill Sans MT", sans-serif; font-size:18px;'>
    Choose the elements (Food, Price, Service, Ambiance) that are important to you and we will suggest the best restaurants based on their overall rankings.
    </p>
    """, 
    unsafe_allow_html=True
)

# Load the DataFrame with caching to avoid repeated loading
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('./Data/final_sentiment_df.xlsx')  # Assuming the data file is named 'final_sentiment_df.xlsx'
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data once
final_sentiment_df = load_data()

# Create multiple selection widgets
aspect_options = ['Food', 'Price', 'Service', 'Ambiance']
st.markdown(
    """
    <label style='font-family:"Comic Sans MS", cursive; font-size:18px;'>
    Select Aspects to Sort By:
    </label>
    """, 
    unsafe_allow_html=True
)

# Multi-select input for aspects
selected_aspects = st.multiselect('', aspect_options)

# Add a slider for users to choose how many restaurants to display
st.markdown(
    """
    <label style='font-family:"Comic Sans MS", cursive; font-size:18px;'>
    Select Number of Top Restaurants to Display:
    </label>
    """, 
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stSlider > div > div > div > div {
        background: none;
    }
    .stSlider > div > div > div > div::after {
        content: '❤️';
        font-size: 24px;
        position: relative;
        left: 2px;
        top: 5px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Slider component with default value set to 10
top_n = st.slider('', min_value=5, max_value=20, value=10)

# Create a placeholder for the table
table_placeholder = st.empty()

# Recommendation function
def recommend_restaurants(aspects, top_n):
    if not aspects:
        st.warning("Please select at least one aspect.")
        return None

    # Filter the DataFrame based on selected aspects
    filtered_df = final_sentiment_df.copy()

    # Sort by selected aspects and keep only the top N
    sort_columns = [f'Average {aspect} Sentiment' for aspect in aspects]
    filtered_df = filtered_df.sort_values(by=sort_columns, ascending=False).head(top_n)

    # Display results
    if filtered_df.empty:
        st.info("No restaurants match the selected criteria.")
    else:
        # Display the sentence directly above the table
        st.markdown(f"<p style='font-family:\"Gill Sans MT\", sans-serif; font-size:18px;'>**Displaying Top {top_n} restaurants based on your selected criteria:**</p>", unsafe_allow_html=True)

        # Create a new DataFrame to hold clickable URLs with shortened display
        display_df = filtered_df[['name', *sort_columns]].copy()
        display_df['url'] = filtered_df['url'].apply(lambda x: f'<a href="{x}" target="_blank">Visit</a>')

        # Display DataFrame with clickable links wrapped in a div container
        table_placeholder.markdown(
            '<div class="table-container">' + display_df.to_html(escape=False, index=False) + '</div>', 
            unsafe_allow_html=True
        )

# Add a button to trigger the recommendation
if st.button('Recommend'):
    recommend_restaurants(selected_aspects, top_n)
