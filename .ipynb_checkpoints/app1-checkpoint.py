import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import fuzz

# Load the data
electronics = pickle.load(open('data.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def find_closest_product(search_text, df):
    """Find the closest matching product name for a given search text"""
    max_ratio = 0
    closest_match = None
    
    # Convert search text to lowercase for better matching
    search_text = search_text.lower()
    
    for product_name in df['name']:
        # Calculate similarity ratio between search text and product name
        ratio = fuzz.partial_ratio(search_text.lower(), product_name.lower())
        if ratio > max_ratio:
            max_ratio = ratio
            closest_match = product_name
    
    return closest_match if max_ratio > 50 else None

def get_recommendations(product):
    """Get recommendations for a given product"""
    electronics_index = electronics[electronics['name']==product].index[0]
    similarity_list = list(enumerate(similarity[electronics_index]))
    top_10_similar_product = sorted(similarity_list, key=lambda x: x[1], reverse=True)[1:11]
    
    similar_products = []
    for idx, similar_score in top_10_similar_product:
        product_info = {
            'name': electronics.loc[idx]['name'],
            'image': electronics.loc[idx]['image'],
            'ratings': electronics.loc[idx]['ratings'],
            'actual_price': electronics.loc[idx]['actual_price']
        }
        similar_products.append(product_info)
    return similar_products

# Streamlit UI
st.title('Smart Product Search Engine')

# Text input for search
search_query = st.text_input('Search for any product:', '', placeholder='Enter product name or description...')

if st.button('Search', key='search_button'):
    if search_query:
        # Add a spinner while searching
        with st.spinner('Searching for products...'):
            # Find the closest matching product
            matched_product = find_closest_product(search_query, electronics)
            
            if matched_product:
                st.success(f"Best match found: {matched_product}")
                
                # Get recommendations
                search_products = get_recommendations(matched_product)
                
                # Display recommendations
                st.write("### Similar Products:")
                
                # Create columns for better layout
                cols = st.columns(2)
                for idx, rec in enumerate(search_products):
                    with cols[idx % 2]:
                        st.image(rec['image'], width=200)
                        st.write(f"**{rec['name']}**")
                        st.write(f"⭐ Ratings: {rec['ratings']}")
                        
                        st.divider()
            else:
                st.error(f"No products found matching '{search_query}'. Please try a different search term.")
    else:
        st.warning("Please enter a search term.")

# Add some styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        background-color: #000000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Python")