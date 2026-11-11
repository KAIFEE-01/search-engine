import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

# Page configuration
st.set_page_config(
    page_title="Smart Product Search & Recommendation Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling that supports both light and dark modes
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .product-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background: rgba(125, 125, 125, 0.08);
        border: 1px solid rgba(125, 125, 125, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .badge-matched {
        background-color: #2e7d32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .similarity-pill {
        background-color: #1976d2;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .price-discount {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e53935;
    }
    .price-actual {
        font-size: 0.95rem;
        text-decoration: line-through;
        color: #9e9e9e;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading product database & AI model...")
def load_data():
    """Load cached dataset and similarity matrix. Auto-generate if missing."""
    data_path = 'data.pkl'
    similarity_path = 'similarity.pkl'

    if not os.path.exists(data_path) or not os.path.exists(similarity_path):
        from preprocess import process_data
        st.info("Pickle files not found. Generating them from Speakers.csv for the first run...")
        electronics, similarity = process_data(csv_path="Speakers.csv")
    else:
        with open(data_path, 'rb') as f:
            electronics = pickle.load(f)
        with open(similarity_path, 'rb') as f:
            similarity = pickle.load(f)

    return electronics, similarity


def find_closest_product(search_text, df):
    """
    Find the closest matching product name using token set & partial ratio.
    Returns (best_match_name, best_match_row, score) or (None, None, 0)
    """
    search_text = search_text.strip().lower()
    if not search_text:
        return None, None, 0

    max_score = 0
    best_match_idx = None

    for idx, product_name in enumerate(df['name']):
        name_lower = str(product_name).lower()
        # Combine token_set_ratio (handles multi-word order) and partial_ratio
        token_score = fuzz.token_set_ratio(search_text, name_lower)
        partial_score = fuzz.partial_ratio(search_text, name_lower)
        combined_score = (token_score * 0.7) + (partial_score * 0.3)

        if combined_score > max_score:
            max_score = combined_score
            best_match_idx = idx

    if max_score >= 50 and best_match_idx is not None:
        best_row = df.iloc[best_match_idx]
        return best_row['name'], best_row, max_score

    return None, None, max_score


def get_recommendations(product_name, df, similarity, top_n=10):
    """Get content-based recommendations for a given product."""
    matched_indices = df[df['name'] == product_name].index
    if len(matched_indices) == 0:
        return []

    product_idx = matched_indices[0]
    similarity_list = list(enumerate(similarity[product_idx]))
    # Sort by descending cosine similarity
    sorted_similar = sorted(similarity_list, key=lambda x: x[1], reverse=True)

    similar_products = []
    seen_names = {product_name.strip().lower()}

    for idx, score in sorted_similar:
        candidate = df.iloc[idx]
        norm_name = str(candidate['name']).strip().lower()

        # Avoid exact duplicate product names
        if norm_name in seen_names:
            continue
        seen_names.add(norm_name)

        similar_products.append({
            'name': candidate['name'],
            'image': candidate.get('image', ''),
            'ratings': candidate.get('ratings', None),
            'no_of_ratings': candidate.get('no_of_ratings', None),
            'discount_price': candidate.get('discount_price', None),
            'actual_price': candidate.get('actual_price', None),
            'link': candidate.get('link', None),
            'similarity_pct': round(float(score) * 100, 1)
        })

        if len(similar_products) >= top_n:
            break

    return similar_products


def format_price(discount, actual):
    """Format price display with discount and original prices."""
    has_discount = pd.notna(discount) and str(discount).strip() != ''
    has_actual = pd.notna(actual) and str(actual).strip() != ''

    if has_discount and has_actual:
        return f'<span class="price-discount">{discount}</span> <span class="price-actual">{actual}</span>'
    elif has_discount:
        return f'<span class="price-discount">{discount}</span>'
    elif has_actual:
        return f'<span class="price-discount">{actual}</span>'
    return '<span style="color: #9e9e9e;">Price not available</span>'


def format_ratings(rating, count):
    """Format ratings nicely with star badge."""
    if pd.notna(rating) and str(rating).strip() not in ['', 'nan', 'None']:
        count_str = f" ({count})" if pd.notna(count) and str(count).strip() not in ['', 'nan'] else ""
        return f"⭐ **{rating}** / 5.0{count_str}"
    return "⭐ *No ratings yet*"


# Main App Execution
def main():
    electronics, similarity = load_data()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Overview")
        st.metric("Total Products", f"{len(electronics):,}")
        st.write("Specialized in audio devices, portable speakers, soundbars, and home theaters.")

        st.divider()
        st.subheader("💡 Try Searching For:")
        sample_queries = [
            "boAt Stone",
            "JBL Flip Bluetooth Speaker",
            "Echo Dot smart speaker",
            "Sony Soundbar 5.1",
            "Zebronics wireless speaker"
        ]
        for query in sample_queries:
            if st.button(f"🔍 {query}", key=f"btn_{query}"):
                st.session_state['search_input'] = query
                st.rerun()

        st.divider()
        if st.button("🔄 Rebuild Model / Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading...")
            st.rerun()

    # Main Area
    st.markdown('<div class="main-title">🔍 Smart Product Search & Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Find any audio device and instantly discover AI-powered similar recommendations using NLP and Cosine Similarity.</div>', unsafe_allow_html=True)

    # Search Form for immediate submission on Enter
    default_search = st.session_state.get('search_input', '')
    with st.form(key='search_form'):
        col_search, col_btn = st.columns([5, 1])
        with col_search:
            search_query = st.text_input(
                label="Search product",
                value=default_search,
                placeholder="Enter speaker name, brand, or feature (e.g. JBL waterproof, Sony soundbar)...",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button(label="Search", use_container_width=True, type="primary")

    active_query = search_query if submitted else default_search

    if active_query:
        with st.spinner("Finding best match and generating recommendations..."):
            matched_name, matched_row, match_score = find_closest_product(active_query, electronics)

        if matched_name is not None and matched_row is not None:
            # Display Primary Match
            st.markdown('<div class="badge-matched">🎯 BEST MATCH FOUND</div>', unsafe_allow_html=True)

            with st.container():
                match_cols = st.columns([1, 3])
                with match_cols[0]:
                    if pd.notna(matched_row.get('image')):
                        st.image(matched_row['image'], width=220)
                with match_cols[1]:
                    st.subheader(matched_row['name'])
                    st.write(format_ratings(matched_row.get('ratings'), matched_row.get('no_of_ratings')))
                    st.markdown(format_price(matched_row.get('discount_price'), matched_row.get('actual_price')), unsafe_allow_html=True)
                    st.caption(f"Fuzzy Match Confidence: **{match_score:.1f}%**")

                    if pd.notna(matched_row.get('link')) and str(matched_row['link']).startswith('http'):
                        st.link_button("View on Amazon ↗", matched_row['link'], use_container_width=False)

            st.markdown("---")

            # Display Recommendations
            recommendations = get_recommendations(matched_name, electronics, similarity, top_n=9)

            if recommendations:
                st.subheader("✨ Recommended Similar Products:")

                # 3-column responsive grid
                cols = st.columns(3)
                for idx, rec in enumerate(recommendations):
                    with cols[idx % 3]:
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)
                        
                        # Image
                        if rec['image'] and str(rec['image']).startswith('http'):
                            st.image(rec['image'], use_container_width=True)

                        # Match similarity pill
                        st.markdown(f'<span class="similarity-pill">{rec["similarity_pct"]}% Similarity</span>', unsafe_allow_html=True)

                        # Product Title
                        st.write(f"**{rec['name']}**")

                        # Ratings
                        st.write(format_ratings(rec['ratings'], rec['no_of_ratings']))

                        # Price
                        st.markdown(format_price(rec['discount_price'], rec['actual_price']), unsafe_allow_html=True)

                        # Product Link
                        if rec['link'] and str(rec['link']).startswith('http'):
                            st.link_button("View Details ↗", rec['link'], use_container_width=True)

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write("")
            else:
                st.info("No close recommendations found for this product.")

        else:
            st.error(f"No products found matching '{active_query}'. Try searching for a brand like **boAt**, **JBL**, **Sony**, or **Zebronics**.")

    # Footer
    st.markdown("---")
    st.caption("Powered by Streamlit, Scikit-Learn (CountVectorizer & Cosine Similarity), and NLTK.")


if __name__ == '__main__':
    main()
