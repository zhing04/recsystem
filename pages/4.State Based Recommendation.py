import pandas as pd
import streamlit as st
import os
from bokeh.models.widgets import Div
from PIL import Image

st.set_page_config(layout='centered', initial_sidebar_state='expanded')
st.sidebar.image('Data/App_icon.png')
st.markdown("<h1 style='text-align: center;'>Restaurants</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: justify;'>Embark on a gastronomic journey with our curated selection of restaurants across various states. Whether you're craving the bold flavors of Texas barbecue, the diverse cuisine of California, or the iconic dishes of New York, we've got you covered. Our app is your passport to culinary exploration, delivering personalized recommendations based on real user reviews and ratings.</p>

<p style='text-align: justify;'>Discover hidden gems, indulge in mouthwatering dishes, and immerse yourself in the vibrant food culture of your chosen destination. From cozy cafes to upscale fine dining establishments, there's something for every palate and occasion.</p>
""", unsafe_allow_html=True)

# Load data
California = pd.read_csv('Data/California/California.csv', sep=',')
California["Location"] = California["Street Address"] +', '+ California["Location"]
California = California.drop(['Street Address',], axis=1)

New_York = pd.read_csv('Data/New York/New_York.csv', sep=',')
New_York["Location"] = New_York["Street Address"] +', '+ New_York["Location"]
New_York = New_York.drop(['Street Address', ], axis=1)

New_Jersey = pd.read_csv('Data/New Jersey/New_Jersey.csv', sep=',')
New_Jersey["Location"] = New_Jersey["Street Address"] +', '+ New_Jersey["Location"]
New_Jersey  = New_Jersey.drop(['Street Address', ], axis=1)

Texas = pd.read_csv('Data/Texas/Texas.csv', sep=',')
Texas["Location"] = Texas["Street Address"] +', '+ Texas["Location"]
Texas = Texas.drop(['Street Address', ],axis=1)

Washington = pd.read_csv('Data/Washington/Washington.csv', sep=',')
Washington["Location"] = Washington["Street Address"] +', '+ Washington["Location"]
Washington = Washington.drop(['Street Address', ], axis=1)

# Select state
option = st.selectbox('Select Your State', ('New York', 'New Jersey', 'California', 'Texas', 'Washington'))

# Details of every restaurant
def details(dataframe):
    # Filter for unique restaurant names and limit to top 20
    unique_restaurants = dataframe['Name'].drop_duplicates().head(20)
    
    # Use the unique restaurant names in the selectbox
    title = st.selectbox('Select Your Restaurant (Top 20)', unique_restaurants)

    # Check if the selected restaurant exists in the dataframe
    if title in dataframe['Name'].values:
        Reviews = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Reviews']
        st.subheader("Restaurant Rating:-")

        # REVIEWS
        if Reviews == '4.5':
            image = Image.open('Data/Ratings/Img4.5.png')
            st.image(image, use_column_width=True)
        elif Reviews == '4':
            image = Image.open('Data/Ratings/Img4.0.png')
            st.image(image, use_column_width=True)
        elif Reviews == '5':
            image = Image.open('Data/Ratings/Img5.0.png')
            st.image(image, use_column_width=True)
        else:
            pass

        # Comments section
        if 'Comments' in dataframe.columns:
            comment = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Comments']
            if comment != "No Comments":
                st.subheader("Comments:-")
                st.warning(comment)

        # Type of restaurant
        Type = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Type']
        st.subheader("Restaurant Category:-")
        st.error(Type)

        # Location
        Location = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Location']
        st.subheader("The Address:-")
        st.success(Location)

        # Contact details
        contact_no = dataframe.at[dataframe['Name'].eq(title).idxmax(), 'Contact Number']
        if contact_no != "Not Available":
            st.subheader("Contact Details:-")
            st.info('Phone:- ' + contact_no)

    st.text("")
    image = Image.open('Data/food_2.jpg')
    st.image(image, use_column_width=True)

# Call the details function based on the selected state
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

# Collect User Feedback
st.markdown("## Rate Your Experience")
rating = st.slider('Rate this restaurant (1-5)', 1, 5)
feedback_comment = st.text_area('Your Feedback')

if st.button('Submit Feedback'):
    # Save the feedback to a CSV file
    feedback_file = 'Data/feedback.csv'
    
    # Create the CSV file if it doesn't exist
    if not os.path.isfile(feedback_file):
        feedback_df = pd.DataFrame(columns=['Reviews', 'Comments'])
        feedback_df.to_csv(feedback_file, index=False)
    
    # Load existing feedback data
    feedback_df = pd.read_csv(feedback_file)

    # Append new feedback
    new_feedback = pd.DataFrame([{'Reviews': f'{rating} of 5 bubbles', 'Comments': feedback_comment}])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(feedback_file, index=False)
    
    # Clear the fields after submission
    st.success('Thanks for your feedback!')

