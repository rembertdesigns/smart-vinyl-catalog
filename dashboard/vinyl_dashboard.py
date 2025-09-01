import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config.bigquery_config import config

st.set_page_config(page_title="Smart Vinyl Catalog", page_icon="🎵", layout="wide")

# Initialize BigQuery client
@st.cache_resource
def init_bigquery():
    return config.get_client()

client = init_bigquery()

# Load scaled data
@st.cache_data
def load_scaled_catalog():
    query = """
    SELECT 
        dr.release_id,
        dr.title,
        dr.artist,
        dr.year,
        dr.genre,
        dr.label,
        dr.country,
        ar.rating,
        ar.review_text,
        ar.review_source
    FROM `vinyl_catalog.discogs_releases` dr
    JOIN `vinyl_catalog.album_reviews` ar ON dr.release_id = ar.album_id
    ORDER BY ar.rating DESC
    """
    return client.query(query).to_dataframe()

@st.cache_data
def load_personal_collection():
    query = """
    SELECT 
        pc.*,
        dr.title,
        dr.artist,
        dr.genre,
        dr.label
    FROM `vinyl_catalog.personal_collection` pc
    JOIN `vinyl_catalog.discogs_releases` dr ON pc.release_id = dr.release_id
    """
    try:
        return client.query(query).to_dataframe()
    except:
        return pd.DataFrame()  # Return empty if personal collection doesn't exist

# Natural language query function
def process_natural_language_query(query_text):
    """Convert natural language to SQL and execute"""
    base_sql = """
    SELECT dr.title, dr.artist, dr.year, dr.genre, dr.label, ar.rating
    FROM `vinyl_catalog.discogs_releases` dr
    JOIN `vinyl_catalog.album_reviews` ar ON dr.release_id = ar.album_id
    WHERE {conditions}
    ORDER BY ar.rating DESC
    LIMIT 20
    """
    
    conditions = []
    query_lower = query_text.lower()
    
    # Genre detection
    if 'jazz' in query_lower:
        conditions.append("dr.genre = 'Jazz'")
    if 'rock' in query_lower:
        conditions.append("dr.genre = 'Rock'")
    if 'electronic' in query_lower:
        conditions.append("dr.genre = 'Electronic'")
    
    # Time period detection
    if '60s' in query_lower or 'sixties' in query_lower:
        conditions.append("dr.year BETWEEN 1960 AND 1969")
    if '70s' in query_lower or 'seventies' in query_lower:
        conditions.append("dr.year BETWEEN 1970 AND 1979")
    if '80s' in query_lower or 'eighties' in query_lower:
        conditions.append("dr.year BETWEEN 1980 AND 1989")
    
    # Rating detection
    if 'high' in query_lower or 'best' in query_lower:
        conditions.append("ar.rating >= 4.0")
    
    # Label detection
    if 'blue note' in query_lower:
        conditions.append("dr.label LIKE '%Blue Note%'")
    if 'columbia' in query_lower:
        conditions.append("dr.label LIKE '%Columbia%'")
    
    if not conditions:
        conditions.append("1=1")
    
    final_query = base_sql.format(conditions=" AND ".join(conditions))
    
    try:
        return client.query(final_query).to_dataframe()
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()

# Main app
st.title("🎵 Smart Vinyl Catalog - Production Dashboard")
st.markdown("AI-powered vinyl collection management with natural language querying")

# Load data
try:
    catalog_df = load_scaled_catalog()
    personal_df = load_personal_collection()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Data overview
    st.sidebar.metric("Total Catalog", f"{len(catalog_df):,}")
    st.sidebar.metric("Personal Collection", f"{len(personal_df):,}")
    st.sidebar.metric("Average Rating", f"{catalog_df['rating'].mean():.1f}/5.0")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Natural Language Search", "Collection Analytics", "Recommendations", "Data Explorer"])
    
    with tab1:
        st.header("Natural Language Query Interface")
        st.markdown("Ask questions about your vinyl catalog in plain English")
        
        # Query examples
        example_queries = [
            "Show me jazz from the 60s",
            "Find the highest rated albums",
            "What electronic music do you have?",
            "Show me Blue Note releases"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_query = st.text_input(
                "Ask about your collection:",
                placeholder="e.g., Show me jazz albums from the 60s"
            )
        
        with col2:
            st.selectbox("Quick Examples", example_queries, key="examples")
            if st.button("Use Example"):
                user_query = st.session_state.examples
        
        if user_query:
            with st.spinner("Searching catalog..."):
                results = process_natural_language_query(user_query)
                
                if len(results) > 0:
                    st.success(f"Found {len(results)} albums matching your query")
                    
                    # Display results
                    for idx, row in results.head(10).iterrows():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.write(f"**{row['title']}**")
                            st.write(f"by {row['artist']}")
                        
                        with col2:
                            st.write(f"{row['year']} • {row['genre']}")
                            st.write(f"Label: {row['label']}")
                        
                        with col3:
                            st.metric("Rating", f"{row['rating']:.1f}/5.0")
                else:
                    st.warning("No results found. Try a different query.")
    
    with tab2:
        st.header("Collection Analytics")
        
        # Genre distribution
        col1, col2 = st.columns(2)
        
        with col1:
            genre_counts = catalog_df['genre'].value_counts()
            fig_genre = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Collection by Genre"
            )
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig_rating = px.histogram(
                catalog_df,
                x='rating',
                nbins=20,
                title="Rating Distribution"
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        # Timeline analysis
        decade_analysis = catalog_df.copy()
        decade_analysis['decade'] = (decade_analysis['year'] // 10) * 10
        decade_stats = decade_analysis.groupby('decade').agg({
            'title': 'count',
            'rating': 'mean'
        }).reset_index()
        decade_stats.columns = ['decade', 'album_count', 'avg_rating']
        
        fig_timeline = px.scatter(
            decade_stats,
            x='decade',
            y='avg_rating',
            size='album_count',
            title="Albums by Decade (size = count, y = avg rating)"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab3:
        st.header("Smart Recommendations")
        
        # Recommendation filters
        selected_genre = st.selectbox(
            "Get recommendations for genre:",
            options=['All'] + list(catalog_df['genre'].unique())
        )
        
        min_rating = st.slider("Minimum rating", 2.0, 5.0, 4.0, 0.1)
        
        # Generate recommendations
        if st.button("Generate Recommendations"):
            genre_filter = "" if selected_genre == "All" else f"AND dr.genre = '{selected_genre}'"
            
            rec_query = f"""
            SELECT 
                dr.title,
                dr.artist,
                dr.year,
                dr.genre,
                dr.label,
                ar.rating,
                SUBSTR(ar.review_text, 1, 200) as review_snippet
            FROM `vinyl_catalog.discogs_releases` dr
            JOIN `vinyl_catalog.album_reviews` ar ON dr.release_id = ar.album_id
            WHERE ar.rating >= {min_rating}
            {genre_filter}
            ORDER BY ar.rating DESC
            LIMIT 15
            """
            
            recommendations = client.query(rec_query).to_dataframe()
            
            st.subheader(f"Top Recommendations: {selected_genre} (Rating >= {min_rating})")
            
            for idx, rec in recommendations.iterrows():
                with st.expander(f"{rec['title']} by {rec['artist']} - {rec['rating']}/5.0"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Year:** {rec['year']}")
                        st.write(f"**Label:** {rec['label']}")
                        st.write(f"**Review:** {rec['review_snippet']}...")
                    
                    with col2:
                        st.metric("Rating", f"{rec['rating']}/5.0")
                        st.write(f"Genre: {rec['genre']}")
    
    with tab4:
        st.header("Data Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            genre_filter = st.multiselect(
                "Filter by Genre",
                options=catalog_df['genre'].unique(),
                default=catalog_df['genre'].unique()
            )
        
        with col2:
            year_range = st.slider(
                "Year Range",
                int(catalog_df['year'].min()),
                int(catalog_df['year'].max()),
                (int(catalog_df['year'].min()), int(catalog_df['year'].max()))
            )
        
        with col3:
            rating_filter = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
        
        # Apply filters
        filtered_data = catalog_df[
            (catalog_df['genre'].isin(genre_filter)) &
            (catalog_df['year'] >= year_range[0]) &
            (catalog_df['year'] <= year_range[1]) &
            (catalog_df['rating'] >= rating_filter)
        ]
        
        st.write(f"Showing {len(filtered_data)} albums")
        
        # Display data
        st.dataframe(
            filtered_data[['title', 'artist', 'year', 'genre', 'label', 'rating']],
            use_container_width=True
        )

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure your BigQuery connection is working and scaled data is uploaded.")

# Footer
st.markdown("---")
st.markdown("**Smart Vinyl Catalog** - Built with Streamlit, BigQuery AI, and Python")