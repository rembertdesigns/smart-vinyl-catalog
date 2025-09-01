import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config.bigquery_config import config

st.set_page_config(page_title="Smart Vinyl Catalog", page_icon="🎵", layout="wide")

@st.cache_data
def load_collection_data():
    client = config.get_client()
    
    query = """
    SELECT 
        pc.collection_id,
        dr.title,
        dr.artist,
        dr.year,
        dr.label,
        dr.genre,
        dr.style,
        pc.purchase_date,
        pc.purchase_price,
        pc.condition,
        pc.personal_rating,
        pc.times_played,
        ar.rating as critic_rating,
        ar.review_text
    FROM `vinyl_catalog.personal_collection` pc
    JOIN `vinyl_catalog.discogs_releases` dr ON pc.release_id = dr.release_id
    JOIN `vinyl_catalog.album_reviews` ar ON pc.release_id = ar.album_id
    ORDER BY pc.purchase_date
    """
    
    return client.query(query).to_dataframe()

# Main Dashboard
st.title("🎵 Smart Vinyl Catalog Dashboard")
st.markdown("AI-powered vinyl collection management and analytics")

# Load data
try:
    df = load_collection_data()
    
    # Sidebar filters
    st.sidebar.header("Collection Filters")
    
    # Artist filter
    selected_artists = st.sidebar.multiselect(
        "Select Artists",
        options=df['artist'].unique(),
        default=df['artist'].unique()
    )
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['artist'].isin(selected_artists)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Albums", len(filtered_df))
    
    with col2:
        st.metric("Total Value", f"${filtered_df['purchase_price'].sum()}")
    
    with col3:
        st.metric("Avg Personal Rating", f"{filtered_df['personal_rating'].mean():.1f}/10")
    
    with col4:
        st.metric("Avg Critic Rating", f"{filtered_df['critic_rating'].mean():.1f}/5")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Collection Timeline")
        fig_timeline = px.scatter(filtered_df, 
                                 x='purchase_date', 
                                 y='purchase_price',
                                 size='personal_rating',
                                 color='artist',
                                 hover_data=['title', 'condition'],
                                 title="Purchase History")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.subheader("Rating Comparison")
        fig_ratings = px.scatter(filtered_df,
                               x='critic_rating',
                               y='personal_rating', 
                               size='purchase_price',
                               color='label',
                               hover_data=['title', 'artist'],
                               title="Personal vs Critic Ratings")
        st.plotly_chart(fig_ratings, use_container_width=True)
    
    # Collection table
    st.subheader("Collection Details")
    st.dataframe(
        filtered_df[['title', 'artist', 'year', 'label', 'purchase_price', 'condition', 'personal_rating']],
        use_container_width=True
    )
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure BigQuery connection is working and data is uploaded.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and BigQuery for the Kaggle AI Challenge")