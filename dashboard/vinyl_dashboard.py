import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, timedelta
import difflib
from typing import Dict, List
import sys
import os
import warnings
from pathlib import Path

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

st.set_page_config(
    page_title="Smart Vinyl Catalog Pro", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions for data cleaning
def safe_string_operation(value, operation='lower', default='unknown'):
    """Safely perform string operations on potentially NaN values"""
    if pd.isna(value) or value is None:
        return default
    try:
        return getattr(str(value), operation)()
    except:
        return default

def clean_dataframe_types(df):
    """Clean data types to prevent errors"""
    df = df.copy()
    
    # Ensure string columns are strings
    string_columns = ['title', 'artist', 'genre', 'label', 'source', 'review_text']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    # Ensure numeric columns are numeric
    numeric_columns = ['rating', 'year', 'duration', 'plays', 'favorites', 'purchase_price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# Enhanced data loading with integrated catalog
@st.cache_data
def load_combined_dataset():
    """Load the real combined FMA + Discogs catalog"""
    
    # First try to load the combined catalog
    combined_path = Path("data/processed/combined_catalog_with_discogs.csv")
    
    if combined_path.exists():
        try:
            catalog_df = pd.read_csv(combined_path)
            catalog_df = clean_dataframe_types(catalog_df)
            
            # Count records by source
            source_counts = catalog_df['source'].value_counts() if 'source' in catalog_df.columns else {}
            fma_count = source_counts.get('fma_data', 0)
            discogs_count = source_counts.get('discogs_sample', 0)
            
            st.sidebar.success(f"✅ Loaded combined catalog: {len(catalog_df):,} tracks")
            st.sidebar.info(f"📊 FMA: {fma_count:,} | Discogs: {discogs_count:,}")
            
            return catalog_df, discogs_count
            
        except Exception as e:
            st.sidebar.error(f"Error loading combined catalog: {e}")
    
    # Fallback: try individual files
    fma_path = Path("data/processed/fma_integrated.csv")
    if fma_path.exists():
        try:
            catalog_df = pd.read_csv(fma_path)
            catalog_df = clean_dataframe_types(catalog_df)
            
            # Ensure source column exists
            if 'source' not in catalog_df.columns:
                catalog_df['source'] = 'fma_data'
            
            st.sidebar.warning("⚠️ Using FMA data only. Run Discogs integration for full catalog.")
            return catalog_df, 0
            
        except Exception as e:
            st.sidebar.error(f"Error loading FMA catalog: {e}")
    
    # Final fallback: create sample data
    st.sidebar.warning("⚠️ No real data found. Creating sample data.")
    
    # Enhanced sample data with both FMA-style and Discogs-style records
    catalog_data = []
    
    # FMA-style electronic/experimental tracks
    fma_artists = ['Burial', 'Flying Lotus', 'Boards of Canada', 'Autechre', 'Squarepusher', 'Aphex Twin']
    fma_genres = ['Electronic', 'Experimental', 'Ambient', 'IDM', 'Drum & Bass']
    
    for i in range(500):
        catalog_data.append({
            'release_id': f'FMA_{i:03d}',
            'title': f'Creative Track {i+1}',
            'artist': fma_artists[i % len(fma_artists)],
            'year': 2000 + (i % 24),
            'genre': fma_genres[i % len(fma_genres)],
            'label': 'Independent',
            'rating': round(np.random.normal(3.8, 0.5), 1),
            'review_text': f'Experimental {fma_genres[i % len(fma_genres)]} track with innovative sound design',
            'source': 'fma_sample',
            'duration': np.random.randint(120, 600),
            'plays': np.random.randint(50, 500),
            'favorites': np.random.randint(10, 100)
        })
    
    # Discogs-style classic vinyl
    classic_artists = ['Miles Davis', 'John Coltrane', 'The Beatles', 'Pink Floyd', 'Led Zeppelin']
    classic_labels = ['Blue Note', 'Columbia', 'Atlantic', 'Impulse!', 'Verve']
    classic_albums = ['Kind of Blue', 'A Love Supreme', 'Abbey Road', 'Dark Side of the Moon', 'IV']
    classic_genres = ['Jazz', 'Rock', 'Progressive Rock', 'Blues', 'Soul']
    
    for i in range(500):
        catalog_data.append({
            'release_id': f'CLASSIC_{i:03d}',
            'title': classic_albums[i % len(classic_albums)] if i < len(classic_albums) else f'Classic Album {i+1}',
            'artist': classic_artists[i % len(classic_artists)],
            'year': 1950 + (i % 50),
            'genre': classic_genres[i % len(classic_genres)],
            'label': classic_labels[i % len(classic_labels)],
            'rating': round(np.random.normal(4.2, 0.4), 1),
            'review_text': f'Classic {classic_genres[i % len(classic_genres)]} album from the golden era',
            'source': 'discogs_sample',
            'duration': np.random.randint(1800, 3600),  # LP length
            'plays': np.random.randint(100, 1000),
            'favorites': np.random.randint(50, 200)
        })
    
    sample_df = pd.DataFrame(catalog_data)
    sample_df = clean_dataframe_types(sample_df)
    
    return sample_df, 500

# Load personal collection data
@st.cache_data
def load_personal_collection():
    """Load personal collection with enhanced data"""
    personal_data = [
        {
            'collection_id': 'PC_001', 'release_id': 'CLASSIC_001', 
            'purchase_date': '2020-03-15', 'purchase_price': 28.0, 
            'condition': 'VG+', 'personal_rating': 9, 'times_played': 25,
            'listening_notes': 'Perfect for late night sessions. Incredible trumpet work.'
        },
        {
            'collection_id': 'PC_002', 'release_id': 'CLASSIC_002', 
            'purchase_date': '2020-07-20', 'purchase_price': 45.0, 
            'condition': 'Mint', 'personal_rating': 10, 'times_played': 30,
            'listening_notes': 'Spiritual masterpiece. Life-changing album.'
        },
        {
            'collection_id': 'PC_003', 'release_id': 'CLASSIC_003', 
            'purchase_date': '2021-01-10', 'purchase_price': 35.0, 
            'condition': 'VG', 'personal_rating': 9, 'times_played': 22,
            'listening_notes': 'Hard bop at its finest. Great rhythm section.'
        },
        {
            'collection_id': 'PC_004', 'release_id': 'FMA_010', 
            'purchase_date': '2022-05-12', 'purchase_price': 15.0, 
            'condition': 'Mint', 'personal_rating': 8, 'times_played': 18,
            'listening_notes': 'Discovered through Creative Commons. Amazing electronic soundscapes.'
        },
        {
            'collection_id': 'PC_005', 'release_id': 'CLASSIC_005', 
            'purchase_date': '2021-09-08', 'purchase_price': 25.0, 
            'condition': 'Good+', 'personal_rating': 8, 'times_played': 15,
            'listening_notes': 'Classic rock masterpiece with incredible production.'
        }
    ]
    return pd.DataFrame(personal_data)

# Advanced search engine
class AdvancedSearchEngine:
    def __init__(self, catalog_df):
        self.catalog = clean_dataframe_types(catalog_df)
    
    def fuzzy_search(self, query: str, threshold: float = 0.3):
        """Enhanced fuzzy search across all catalog fields"""
        query_lower = str(query).lower()
        results = []
        
        for _, album in self.catalog.iterrows():
            # Create comprehensive searchable text with safe string operations
            searchable_parts = [
                safe_string_operation(album.get('title', ''), 'lower', ''),
                safe_string_operation(album.get('artist', ''), 'lower', ''),
                safe_string_operation(album.get('genre', ''), 'lower', ''),
                safe_string_operation(album.get('label', ''), 'lower', ''),
                safe_string_operation(album.get('review_text', ''), 'lower', '')
            ]
            searchable_text = ' '.join(filter(None, searchable_parts))
            
            if len(searchable_text.strip()) > 0:
                similarity = difflib.SequenceMatcher(None, query_lower, searchable_text).ratio()
                
                if similarity >= threshold:
                    result = album.copy()
                    result['similarity_score'] = similarity
                    results.append(result)
        
        if results:
            results_df = pd.DataFrame(results).sort_values('similarity_score', ascending=False)
            return clean_dataframe_types(results_df)
        return pd.DataFrame()
    
    def semantic_search(self, concept: str):
        """Enhanced semantic search with music concepts"""
        concept_mappings = {
            'spiritual': ['coltrane', 'love supreme', 'spiritual', 'transcendent', 'meditation'],
            'cool': ['miles davis', 'kind of blue', 'cool', 'laid back', 'smooth'],
            'experimental': ['free', 'avant', 'experimental', 'boundary', 'innovative', 'electronic'],
            'electronic': ['synthesizer', 'digital', 'electronic', 'techno', 'ambient', 'idm'],
            'folk': ['acoustic', 'traditional', 'storytelling', 'folk', 'roots'],
            'classic': ['classic', 'vintage', 'legendary', 'masterpiece', 'essential'],
            'jazz': ['jazz', 'bebop', 'hard bop', 'fusion', 'swing'],
            'rock': ['rock', 'guitar', 'blues', 'progressive', 'psychedelic'],
            'underground': ['independent', 'underground', 'alternative', 'creative commons']
        }
        
        concept_lower = safe_string_operation(concept, 'lower', concept)
        if concept_lower in concept_mappings:
            search_terms = ' '.join(concept_mappings[concept_lower])
            return self.fuzzy_search(search_terms, threshold=0.2)
        else:
            return self.fuzzy_search(concept, threshold=0.25)

# Enhanced recommendation engine
class EnhancedRecommendationEngine:
    def __init__(self, catalog_df):
        self.catalog = clean_dataframe_types(catalog_df)
        self.fma_data = self.catalog[self.catalog.get('source', '').str.contains('fma', na=False)] if 'source' in self.catalog.columns else pd.DataFrame()
        self.discogs_data = self.catalog[self.catalog.get('source', '').str.contains('discogs', na=False)] if 'source' in self.catalog.columns else pd.DataFrame()
    
    def genre_based_recommendations(self, target_genre, min_rating=3.5, max_results=10):
        """Enhanced genre-based recommendations"""
        target_genre = str(target_genre) if target_genre is not None else ''
        
        if len(target_genre) == 0:
            return []
        
        # Safe string comparison
        genre_matches = self.catalog[
            (self.catalog['genre'].astype(str).str.contains(target_genre, case=False, na=False)) &
            (self.catalog['rating'] >= min_rating)
        ].sort_values(['rating', 'plays'], ascending=False)
        
        recommendations = []
        for _, album in genre_matches.head(max_results).iterrows():
            source = safe_string_operation(album.get('source', 'unknown'))
            confidence = 0.85 if 'fma' in source else 0.9
            
            recommendations.append({
                'title': safe_string_operation(album.get('title', 'Unknown'), 'title', 'Unknown'),
                'artist': safe_string_operation(album.get('artist', 'Unknown'), 'title', 'Unknown'),
                'genre': safe_string_operation(album.get('genre', 'Unknown'), 'title', 'Unknown'),
                'rating': float(album.get('rating', 0)),
                'year': album.get('year', 'Unknown'),
                'source': source,
                'reason': f"High-rated {target_genre} music",
                'confidence': confidence,
                'plays': int(album.get('plays', 0))
            })
        
        return recommendations
    
    def cross_source_recommendations(self, max_results=12):
        """Generate recommendations comparing FMA vs Discogs sources"""
        recommendations = []
        
        # Get top tracks from each source
        if len(self.fma_data) > 0:
            fma_top = self.fma_data.nlargest(6, 'rating')
            for _, track in fma_top.iterrows():
                recommendations.append({
                    'title': safe_string_operation(track.get('title', 'Unknown'), 'title', 'Unknown'),
                    'artist': safe_string_operation(track.get('artist', 'Unknown'), 'title', 'Unknown'),
                    'genre': safe_string_operation(track.get('genre', 'Unknown'), 'title', 'Unknown'),
                    'rating': float(track.get('rating', 0)),
                    'year': track.get('year', 'Unknown'),
                    'source': track.get('source', 'fma'),
                    'reason': "Top-rated from Creative Commons catalog",
                    'confidence': 0.8,
                    'plays': int(track.get('plays', 0))
                })
        
        if len(self.discogs_data) > 0:
            discogs_top = self.discogs_data.nlargest(6, 'rating')
            for _, track in discogs_top.iterrows():
                recommendations.append({
                    'title': safe_string_operation(track.get('title', 'Unknown'), 'title', 'Unknown'),
                    'artist': safe_string_operation(track.get('artist', 'Unknown'), 'title', 'Unknown'),
                    'genre': safe_string_operation(track.get('genre', 'Unknown'), 'title', 'Unknown'),
                    'rating': float(track.get('rating', 0)),
                    'year': track.get('year', 'Unknown'),
                    'source': track.get('source', 'discogs'),
                    'reason': "Classic vinyl collection highlight",
                    'confidence': 0.9,
                    'plays': int(track.get('plays', 0))
                })
        
        # Sort by rating and return
        recommendations.sort(key=lambda x: x['rating'], reverse=True)
        return recommendations[:max_results]

# Load data
catalog_df, fma_count = load_combined_dataset()
personal_df = load_personal_collection()
search_engine = AdvancedSearchEngine(catalog_df)
rec_engine = EnhancedRecommendationEngine(catalog_df)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-box {
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .source-badge-fma {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .source-badge-discogs {
        background-color: #17a2b8;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown('<h1 class="main-header">🎵 Smart Vinyl Catalog Pro</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Music Discovery Platform with FMA + Discogs Integration**")

# Enhanced sidebar
st.sidebar.header("🎛️ Dashboard Controls")

# Collection metrics with safe operations
total_catalog = len(catalog_df)
total_personal = len(personal_df)
avg_rating = catalog_df['rating'].mean() if 'rating' in catalog_df.columns else 0
collection_value = personal_df['purchase_price'].sum() if 'purchase_price' in personal_df.columns else 0

st.sidebar.markdown("### 📊 Collection Overview")
st.sidebar.metric("Total Catalog", f"{total_catalog:,}")
st.sidebar.metric("Personal Collection", f"{total_personal}")
st.sidebar.metric("Collection Value", f"${collection_value:.0f}")
st.sidebar.metric("Avg Catalog Rating", f"{avg_rating:.1f}/5.0")

# Data source breakdown
st.sidebar.markdown("### 🎼 Data Sources")
if 'source' in catalog_df.columns:
    source_counts = catalog_df['source'].value_counts()
    for source, count in source_counts.items():
        if 'fma' in source.lower():
            st.sidebar.success(f"✅ FMA Data: {count:,} tracks")
        elif 'discogs' in source.lower():
            st.sidebar.info(f"📀 Discogs Data: {count:,} releases")
        else:
            st.sidebar.metric(source.replace('_', ' ').title(), f"{count:,}")

# Enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 AI Search", "📊 Analytics", "🎯 Recommendations", 
    "🎼 Discovery", "👥 Social Insights", "📈 Market Intel"
])

with tab1:
    st.header("🔍 Advanced AI-Powered Search")
    st.markdown("Search across your integrated music catalog with intelligent matching")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search your music catalog:",
            placeholder="e.g., spiritual jazz, electronic ambient, Miles Davis cool, experimental"
        )
    
    with col2:
        search_type = st.selectbox("Search Algorithm", ["Fuzzy Search", "Semantic Search"])
    
    if search_query and len(str(search_query).strip()) > 0:
        with st.spinner("Searching across all music data..."):
            try:
                if search_type == "Fuzzy Search":
                    results = search_engine.fuzzy_search(search_query)
                else:
                    results = search_engine.semantic_search(search_query)
                
                if len(results) > 0:
                    st.success(f"Found {len(results)} matches across {total_catalog:,} tracks")
                    
                    # Enhanced results display
                    for idx, (_, row) in enumerate(results.head(12).iterrows()):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                            
                            with col1:
                                title = safe_string_operation(row.get('title', 'Unknown'), 'title', 'Unknown')
                                artist = safe_string_operation(row.get('artist', 'Unknown'), 'title', 'Unknown')
                                st.markdown(f"**{title}**")
                                st.markdown(f"*{artist}*")
                                
                                # Enhanced source badges
                                if 'source' in row:
                                    source = safe_string_operation(row.get('source', ''))
                                    if 'fma' in source:
                                        st.markdown('<span class="source-badge-fma">🎵 Creative Commons</span>', unsafe_allow_html=True)
                                    elif 'discogs' in source:
                                        st.markdown('<span class="source-badge-discogs">💿 Vinyl Classic</span>', unsafe_allow_html=True)
                            
                            with col2:
                                year = row.get('year', 'Unknown')
                                genre = safe_string_operation(row.get('genre', 'Unknown'), 'title', 'Unknown')
                                label = safe_string_operation(row.get('label', ''), 'title', '')
                                
                                st.write(f"📅 {year}")
                                st.write(f"🎵 {genre}")
                                if label and label != 'Unknown':
                                    st.write(f"🏷️ {label}")
                            
                            with col3:
                                rating = float(row.get('rating', 0))
                                plays = int(row.get('plays', 0))
                                st.metric("Rating", f"{rating:.1f}/5")
                                if plays > 0:
                                    st.caption(f"{plays} plays")
                            
                            with col4:
                                if 'similarity_score' in row:
                                    match_pct = float(row.get('similarity_score', 0)) * 100
                                    st.metric("Match", f"{match_pct:.0f}%")
                                    
                                    # Color-coded match quality
                                    if match_pct >= 80:
                                        st.success("Excellent match")
                                    elif match_pct >= 60:
                                        st.info("Good match")
                                    else:
                                        st.warning("Fair match")
                            
                            st.divider()
                else:
                    st.warning("No matches found. Try different keywords or use semantic search.")
            except Exception as e:
                st.error(f"Search error: {str(e)}")

with tab2:
    st.header("📊 Advanced Music Analytics")
    
    # Enhanced key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Tracks</h3><h2>{total_catalog:,}</h2><p>Multi-Source</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Avg Rating</h3><h2>{avg_rating:.1f}/5.0</h2><p>Quality Score</p></div>', unsafe_allow_html=True)
    
    with col3:
        unique_artists = catalog_df['artist'].nunique() if 'artist' in catalog_df.columns else 0
        st.markdown(f'<div class="metric-card"><h3>Artists</h3><h2>{unique_artists}</h2><p>Unique Musicians</p></div>', unsafe_allow_html=True)
    
    with col4:
        unique_genres = catalog_df['genre'].nunique() if 'genre' in catalog_df.columns else 0
        st.markdown(f'<div class="metric-card"><h3>Genres</h3><h2>{unique_genres}</h2><p>Musical Styles</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced visualizations with cross-source analysis
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Source distribution
            if 'source' in catalog_df.columns:
                source_counts = catalog_df['source'].value_counts()
                if len(source_counts) > 0:
                    fig_source = px.pie(
                        values=source_counts.values,
                        names=[s.replace('_', ' ').title() for s in source_counts.index],
                        title="Music Catalog by Source",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hole=0.4
                    )
                    fig_source.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
                    fig_source.update_layout(showlegend=True, height=400)
                    st.plotly_chart(fig_source, use_container_width=True)
                else:
                    st.info("No source data available for visualization")
            else:
                st.info("Source data not available")
        except Exception as e:
            st.error(f"Error creating source chart: {str(e)}")
    
    with col2:
        try:
            # Rating distribution by source
            if 'rating' in catalog_df.columns and 'source' in catalog_df.columns:
                fig_ratings = px.histogram(
                    catalog_df,
                    x='rating',
                    color='source',
                    title="Rating Distribution by Source",
                    nbins=20,
                    labels={'rating': 'Rating (1-5)', 'count': 'Number of Tracks'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_ratings.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig_ratings, use_container_width=True)
            else:
                st.info("Rating/source data not available")
        except Exception as e:
            st.error(f"Error creating rating chart: {str(e)}")
    
    # Genre analysis
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Top genres
            if 'genre' in catalog_df.columns:
                genre_counts = catalog_df['genre'].value_counts().head(10)
                if len(genre_counts) > 0:
                    fig_genre = px.bar(
                        x=genre_counts.values,
                        y=genre_counts.index,
                        orientation='h',
                        title="Top 10 Genres",
                        labels={'x': 'Number of Tracks', 'y': 'Genre'},
                        color=genre_counts.values,
                        color_continuous_scale='viridis'
                    )
                    fig_genre.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_genre, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating genre chart: {str(e)}")
    
    with col2:
        try:
            # Year distribution
            if 'year' in catalog_df.columns:
                year_data = pd.to_numeric(catalog_df['year'], errors='coerce').dropna()
                if len(year_data) > 0:
                    fig_years = px.histogram(
                        x=year_data,
                        nbins=30,
                        title="Release Years Distribution",
                        labels={'x': 'Year', 'count': 'Number of Releases'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_years.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_years, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating year chart: {str(e)}")

with tab3:
    st.header("🎯 Smart Music Recommendation Engine")
    st.markdown("AI-powered recommendations leveraging your integrated catalog")
    
    # Enhanced recommendation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        rec_genre = st.selectbox(
            "Target Genre for Recommendations:",
            options=['All Genres'] + sorted(catalog_df['genre'].unique().tolist()) if 'genre' in catalog_df.columns else ['All Genres']
        )
    
    with col2:
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.5, 0.1)
    
    with col3:
        max_recs = st.selectbox("Number of Recommendations", [5, 10, 15, 20], index=1)
    
    # Recommendation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎵 Genre-Based Recommendations", use_container_width=True):
            st.session_state['show_genre_recs'] = True
            st.session_state['show_cross_recs'] = False
            st.session_state['show_discovery_recs'] = False
    
    with col2:
        if st.button("🌈 Cross-Source Discovery", use_container_width=True):
            st.session_state['show_cross_recs'] = True
            st.session_state['show_genre_recs'] = False
            st.session_state['show_discovery_recs'] = False
    
    with col3:
        if st.button("🔍 Smart Discovery", use_container_width=True):
            st.session_state['show_discovery_recs'] = True
            st.session_state['show_genre_recs'] = False
            st.session_state['show_cross_recs'] = False
    
    # Display recommendations based on selection
    if st.session_state.get('show_genre_recs', False):
        with st.spinner("Generating genre-based recommendations..."):
            try:
                if rec_genre != 'All Genres':
                    recommendations = rec_engine.genre_based_recommendations(rec_genre, min_rating, max_recs)
                else:
                    # Get diverse recommendations across top genres
                    recommendations = []
                    top_genres = catalog_df['genre'].value_counts().head(3).index.tolist()
                    for genre in top_genres:
                        genre_recs = rec_engine.genre_based_recommendations(genre, min_rating, 3)
                        recommendations.extend(genre_recs)
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations")
                    
                    for i, rec in enumerate(recommendations):
                        with st.container():
                            st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{rec['title']}**")
                                st.markdown(f"*by {rec['artist']}*")
                                st.caption(f"🎵 {rec['genre']} • 📅 {rec['year']}")
                                st.caption(f"💡 {rec['reason']}")
                            
                            with col2:
                                st.metric("Rating", f"{rec['rating']:.1f}/5")
                                if rec['plays'] > 0:
                                    st.caption(f"{rec['plays']} plays")
                            
                            with col3:
                                confidence_pct = rec['confidence'] * 100
                                st.metric("Confidence", f"{confidence_pct:.0f}%")
                                
                                # Source badge
                                source = rec['source']
                                if 'fma' in source:
                                    st.markdown('<span class="source-badge-fma">Creative Commons</span>', unsafe_allow_html=True)
                                elif 'discogs' in source:
                                    st.markdown('<span class="source-badge-discogs">Vinyl Classic</span>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if i < len(recommendations) - 1:
                                st.divider()
                else:
                    st.warning("No recommendations found for the selected criteria.")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    if st.session_state.get('show_cross_recs', False):
        with st.spinner("Generating cross-source recommendations..."):
            try:
                recommendations = rec_engine.cross_source_recommendations()
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} cross-source recommendations")
                    
                    # Group by source for comparison
                    fma_recs = [r for r in recommendations if 'fma' in r['source']]
                    discogs_recs = [r for r in recommendations if 'discogs' in r['source']]
                    
                    if fma_recs:
                        st.subheader("🎵 Creative Commons Highlights")
                        for rec in fma_recs[:6]:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.markdown(f"**{rec['title']}** *by {rec['artist']}*")
                                st.caption(f"🎵 {rec['genre']} • 📅 {rec['year']}")
                            with col2:
                                st.metric("Rating", f"{rec['rating']:.1f}/5")
                            with col3:
                                st.markdown('<span class="source-badge-fma">CC Licensed</span>', unsafe_allow_html=True)
                            st.divider()
                    
                    if discogs_recs:
                        st.subheader("💿 Vinyl Classics")
                        for rec in discogs_recs[:6]:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.markdown(f"**{rec['title']}** *by {rec['artist']}*")
                                st.caption(f"🎵 {rec['genre']} • 📅 {rec['year']}")
                            with col2:
                                st.metric("Rating", f"{rec['rating']:.1f}/5")
                            with col3:
                                st.markdown('<span class="source-badge-discogs">Vinyl</span>', unsafe_allow_html=True)
                            st.divider()
                else:
                    st.warning("No cross-source recommendations available.")
            except Exception as e:
                st.error(f"Error generating cross-source recommendations: {str(e)}")

with tab4:
    st.header("🎼 Music Discovery Hub")
    st.markdown("Explore new music across different sources and genres")
    
    # Discovery metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'source' in catalog_df.columns:
            fma_count = len(catalog_df[catalog_df['source'].str.contains('fma', na=False)])
            st.metric("Creative Commons Tracks", f"{fma_count:,}")
    
    with col2:
        if 'source' in catalog_df.columns:
            discogs_count = len(catalog_df[catalog_df['source'].str.contains('discogs', na=False)])
            st.metric("Vinyl Classics", f"{discogs_count:,}")
    
    with col3:
        high_rated_count = len(catalog_df[pd.to_numeric(catalog_df['rating'], errors='coerce') >= 4.5])
        st.metric("Highly Rated (4.5+)", f"{high_rated_count:,}")
    
    # Genre explorer
    st.subheader("🎵 Genre Explorer")
    
    selected_genre = st.selectbox(
        "Explore a genre:",
        options=sorted(catalog_df['genre'].unique()) if 'genre' in catalog_df.columns else []
    )
    
    if selected_genre:
        genre_data = catalog_df[catalog_df['genre'] == selected_genre]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{selected_genre} Statistics:**")
            st.write(f"📊 Total tracks: {len(genre_data):,}")
            
            if 'rating' in genre_data.columns:
                avg_rating = pd.to_numeric(genre_data['rating'], errors='coerce').mean()
                st.write(f"⭐ Average rating: {avg_rating:.2f}")
            
            if 'year' in genre_data.columns:
                years = pd.to_numeric(genre_data['year'], errors='coerce').dropna()
                if len(years) > 0:
                    st.write(f"📅 Year range: {int(years.min())} - {int(years.max())}")
            
            # Top artists in genre
            if 'artist' in genre_data.columns:
                top_artists = genre_data['artist'].value_counts().head(5)
                st.write("🎤 Top artists:")
                for artist, count in top_artists.items():
                    st.write(f"  • {artist} ({count} tracks)")
        
        with col2:
            # Show top-rated tracks in this genre
            if 'rating' in genre_data.columns:
                top_rated = genre_data.nlargest(5, 'rating')
                st.markdown(f"**Top Rated {selected_genre} Tracks:**")
                
                for _, track in top_rated.iterrows():
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{track['title']}** by {track['artist']}")
                            if 'year' in track:
                                st.caption(f"📅 {track['year']}")
                        with col_b:
                            st.metric("", f"{track['rating']:.1f}⭐")

with tab5:
    st.header("👥 Social Music Insights")
    st.markdown("Community trends and music discovery patterns")
    
    # Trending analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Trending Genres")
        
        if 'genre' in catalog_df.columns:
            genre_popularity = catalog_df['genre'].value_counts().head(8)
            
            fig_trending = px.bar(
                x=genre_popularity.values,
                y=genre_popularity.index,
                orientation='h',
                title="Most Popular Genres",
                color=genre_popularity.values,
                color_continuous_scale='viridis'
            )
            fig_trending.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_trending, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Quality Distribution")
        
        if 'rating' in catalog_df.columns:
            ratings = pd.to_numeric(catalog_df['rating'], errors='coerce').dropna()
            
            if len(ratings) > 0:
                fig_quality = px.histogram(
                    x=ratings,
                    nbins=25,
                    title="Rating Distribution",
                    color_discrete_sequence=['#ff6b6b']
                )
                fig_quality.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_quality, use_container_width=True)
    
    # Community features
    st.subheader("🎵 Community Highlights")
    
    # High-rated discoveries
    if 'rating' in catalog_df.columns:
        high_rated = catalog_df[pd.to_numeric(catalog_df['rating'], errors='coerce') >= 4.5]
        
        if len(high_rated) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Highest Rated Discoveries:**")
                top_discoveries = high_rated.nlargest(5, 'rating')
                
                for _, track in top_discoveries.iterrows():
                    source_type = "Creative Commons" if 'fma' in str(track.get('source', '')) else "Vinyl Classic"
                    st.write(f"⭐ **{track['title']}** by {track['artist']}")
                    st.caption(f"Rating: {track['rating']:.1f}/5 • {source_type}")
            
            with col2:
                st.markdown("**📊 Discovery Statistics:**")
                st.metric("Highly Rated Tracks", len(high_rated))
                
                if 'source' in high_rated.columns:
                    fma_high = len(high_rated[high_rated['source'].str.contains('fma', na=False)])
                    discogs_high = len(high_rated[high_rated['source'].str.contains('discogs', na=False)])
                    
                    st.write(f"🎵 CC High-Rated: {fma_high}")
                    st.write(f"💿 Vinyl High-Rated: {discogs_high}")

with tab6:
    st.header("📈 Market Intelligence & Analytics")
    st.markdown("Data-driven insights for music collection and discovery")
    
    # Market overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average rating by decade
        if 'year' in catalog_df.columns and 'rating' in catalog_df.columns:
            years = pd.to_numeric(catalog_df['year'], errors='coerce')
            ratings = pd.to_numeric(catalog_df['rating'], errors='coerce')
            
            valid_data = catalog_df[years.notna() & ratings.notna()].copy()
            if len(valid_data) > 0:
                valid_data['decade'] = (pd.to_numeric(valid_data['year']) // 10) * 10
                decade_ratings = valid_data.groupby('decade')['rating'].mean().reset_index()
                
                current_decade_rating = decade_ratings['rating'].iloc[-1] if len(decade_ratings) > 0 else 0
                st.metric("Current Decade Avg Rating", f"{current_decade_rating:.2f}/5")
    
    with col2:
        # Genre diversity index
        if 'genre' in catalog_df.columns:
            unique_genres = catalog_df['genre'].nunique()
            total_tracks = len(catalog_df)
            diversity_index = unique_genres / total_tracks * 100
            st.metric("Genre Diversity Index", f"{diversity_index:.1f}%")
    
    with col3:
        # Source balance
        if 'source' in catalog_df.columns:
            source_counts = catalog_df['source'].value_counts()
            if len(source_counts) > 1:
                balance_score = 1 - (source_counts.max() - source_counts.min()) / len(catalog_df)
                st.metric("Source Balance Score", f"{balance_score:.2f}")
    
    # Detailed analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Temporal Analysis")
        
        if 'year' in catalog_df.columns:
            years = pd.to_numeric(catalog_df['year'], errors='coerce').dropna()
            
            if len(years) > 0:
                # Tracks by decade
                decades = (years // 10) * 10
                decade_counts = decades.value_counts().sort_index()
                
                fig_decades = px.bar(
                    x=decade_counts.index.astype(str) + 's',
                    y=decade_counts.values,
                    title="Music Catalog by Decade",
                    color=decade_counts.values,
                    color_continuous_scale='plasma'
                )
                fig_decades.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_decades, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Insights")
        
        if 'rating' in catalog_df.columns and 'source' in catalog_df.columns:
            # Rating comparison by source
            source_ratings = []
            
            for source in catalog_df['source'].unique():
                if pd.notna(source):
                    source_data = catalog_df[catalog_df['source'] == source]
                    ratings = pd.to_numeric(source_data['rating'], errors='coerce').dropna()
                    
                    if len(ratings) > 0:
                        source_ratings.append({
                            'Source': source.replace('_', ' ').title(),
                            'Average Rating': ratings.mean(),
                            'Track Count': len(ratings)
                        })
            
            if source_ratings:
                ratings_df = pd.DataFrame(source_ratings)
                
                fig_source_quality = px.bar(
                    ratings_df,
                    x='Source',
                    y='Average Rating',
                    title="Average Rating by Source",
                    color='Average Rating',
                    color_continuous_scale='viridis'
                )
                fig_source_quality.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_source_quality, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎵 Smart Vinyl Catalog Pro</h4>
    <p>AI-Powered Music Discovery with FMA + Discogs Integration</p>
    <p>Total Catalog: {total_catalog:,} tracks • Personal Collection: {total_personal} items</p>
    <p>Built with Streamlit • Enhanced with Real Music Data</p>
</div>
""", unsafe_allow_html=True)