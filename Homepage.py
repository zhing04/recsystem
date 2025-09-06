import streamlit as st
from PIL import Image



st.set_page_config(layout='centered', initial_sidebar_state='expanded')

st. sidebar.image('Data/App_icon.png')

image = Image.open('Data/Food.jpg')
st.image(image, use_column_width=True)

st.title("Discover the best places to eat in your town.")

st.markdown("Powered by data from TripAdvisor, our app curates recommendations from 20 cities across New York, New Jersey, California, Texas, and Washington. Find the top 10 restaurants similar to your favorites.")
st.markdown("Leveraging Natural Language Processing and Content-Based Recommender Systems, we prioritize user comments to deliver personalized suggestions.")
st.success("Satisfy your cravings with ease! :fork_and_knife:" ":yum:")
