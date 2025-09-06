import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium

# Initialize the Google Maps client with your API key
API_KEY = 'AIzaSyDK7boLSVOjAK2lPx6NoOrBYPaXLpCAUoA'
gmaps = googlemaps.Client(key=API_KEY)

def get_autocomplete_suggestions(input_text):
    try:
        suggestions = gmaps.places_autocomplete(input_text)
        return [suggestion['description'] for suggestion in suggestions]
    except Exception as e:
        st.error(f"Error fetching autocomplete suggestions: {e}")
        return []

def get_lat_long(address):
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            return (lat, lng)
    except Exception as e:
        st.error(f"Error fetching latitude and longitude: {e}")
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon1 - lon2)

    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def find_nearby_restaurants(lat, lng, df, max_distance_km=5):
    df['distance'] = df.apply(lambda row: haversine(lat, lng, row['latitude'], row['longitude']), axis=1)
    nearby_restaurants = df[df['distance'] <= max_distance_km].sort_values(by='distance').head(5)
    return nearby_restaurants

# Load your dataset with restaurant name, address, latitude, and longitude
df_with_lat_lon = pd.read_excel('df_with_lat_lon.xlsx')

# Remove duplicates based on the 'name' column
df_with_lat_lon = df_with_lat_lon.drop_duplicates(subset=['Name'])

# Add a placeholder URL column if it's missing
if 'URL' not in df_with_lat_lon.columns:
    df_with_lat_lon['URL'] = df_with_lat_lon['Name'].apply(lambda name: f"https://www.google.com/search?q={name.replace(' ', '+')}")

# Streamlit application interface

image_url = "https://www.restolacuisine.com/restaurants/restaurant-la-cuisine/website/images/Lacuisine_resto.jpg"
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='font-family:Forte; color:#white; font-size:35px; text-align:center;'>
    Nearby Restaurant Finder with Map
    </h1>
    """, 
    unsafe_allow_html=True
)

# Step 1: Enter a location
st.markdown(
    """
    <label style='font-family:"Comic Sans MS", cursive; color:#white; font-size:25px;'>
    Enter a location:
    </label>
    <p style='font-size:20px;'> Example: Harborside Financial Center - Plaza 5, Jersey City, NJ 07311
    </p>
    """, 
    unsafe_allow_html=True
)
user_input = st.text_input('', key='location_input')

if user_input:
    suggestions = get_autocomplete_suggestions(user_input)
    
    if suggestions:
        selected_address = suggestions[0] 
        
        if selected_address:
            coordinates = get_lat_long(selected_address)
            
            if coordinates:
                st.markdown(
                    f"""
                    <p style='font-family:"Comic Sans MS", cursive; font-size:20px;'>
                    Selected Location: {selected_address}
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <p style='font-family:"Comic Sans MS", cursive; font-size:20px;'>
                    Latitude: {coordinates[0]}, Longitude: {coordinates[1]}
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Step 3: Find nearby restaurants
                nearby_restaurants = find_nearby_restaurants(coordinates[0], coordinates[1], df_with_lat_lon)
                
                if not nearby_restaurants.empty:
                    st.markdown(
                        """
                        <p style='font-family:"Comic Sans MS", cursive; font-size:20px;'>
                        Top 5 Restaurants within 5 km:
                        </p>
                        """, 
                        unsafe_allow_html=True
                    )         
                    
                    # Convert the URLs to clickable links in the dataframe
                    nearby_restaurants['URL'] = nearby_restaurants.apply(lambda row: f'<a href="{row["URL"]}" class="url" target="_blank">{row["Name"]}</a>', axis=1)
                    
                    # Create a placeholder for the table
                    table_placeholder = st.empty()
                    table_placeholder.markdown('<div class="table-container">' + nearby_restaurants[['Name', 'distance', 'URL']].to_html(escape=False, index=False) + '</div>', unsafe_allow_html=True)
                    
                    # Displaying map using folium
                    m = folium.Map(location=[coordinates[0], coordinates[1]], zoom_start=12)
                    
                    # Add a marker for the selected location
                    folium.Marker([coordinates[0], coordinates[1]], tooltip='Selected Location').add_to(m)
                    
                    # Add markers for nearby restaurants
                    for idx, row in nearby_restaurants.iterrows():
                        folium.Marker(
                            [row['latitude'], row['longitude']], 
                            popup=row['Name'],
                            tooltip=row['Name']
                        ).add_to(m)
                    
                    # Display map using streamlit_folium
                    st_folium(m, width=705, height=600)
                else:
                    st.write("No restaurants found within 5 km.")
            else:
                st.error("Could not fetch coordinates for the selected location.")
    else:
        st.error("No suggestions found.")
