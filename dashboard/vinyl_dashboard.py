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

# Enhanced data loading with FMA integration and error handling
@st.cache_data
def load_combined_dataset():
    """Load combined sample data + FMA integrated data with proper error handling"""
    
    # Original high-quality sample data
    catalog_data = []
    artists = ['Miles Davis', 'John Coltrane', 'Art Blakey', 'Bill Evans', 'Horace Silver', 
               'Lee Morgan', 'Hank Mobley', 'Cannonball Adderley', 'Thelonious Monk', 'Charlie Parker',
               'Daft Punk', 'Radiohead', 'Bon Iver', 'Aphex Twin', 'Brian Eno', 'Joni Mitchell',
               'Bob Dylan', 'The Beatles', 'Pink Floyd', 'Led Zeppelin']
    
    labels = ['Blue Note', 'Columbia', 'Atlantic', 'Impulse!', 'Verve', 'Prestige', 'ECM', 'Warp Records']
    
    albums = ['Kind of Blue', 'A Love Supreme', 'Moanin', 'Waltz for Debby', 'Song for My Father',
              'Random Access Memories', 'OK Computer', 'For Emma Forever Ago', 'Selected Ambient Works',
              'Music for Airports', 'Blue', 'Highway 61 Revisited', 'Abbey Road', 'Dark Side of the Moon']
    
    genres = ['Jazz', 'Electronic', 'Indie Folk', 'Ambient', 'Rock', 'Alternative', 'Hip-Hop', 'Classical']
    
    for i in range(50):
        catalog_data.append({
            'release_id': f'CLASSIC_{i:03d}',
            'title': albums[i % len(albums)] if i < len(albums) else f'Album {i+1}',
            'artist': artists[i % len(artists)],
            'year': 1950 + (i % 70),
            'genre': genres[i % len(genres)],
            'label': labels[i % len(labels)],
            'rating': round(np.random.normal(4.2, 0.4), 1),
            'review_text': f'Exceptional album featuring {artists[i % len(artists)]} with innovative sound',
            'source': 'curated_classics',
            'duration': np.random.randint(180, 3600),
            'plays': np.random.randint(10, 500),
            'favorites': np.random.randint(5, 100)
        })
    
    sample_df = pd.DataFrame(catalog_data)
    sample_df = clean_dataframe_types(sample_df)
    
    # Generate additional synthetic FMA-style data for demonstration
    fma_data = []
    electronic_artists = ['Burial', 'Flying Lotus', 'Boards of Canada', 'Autechre', 'Squarepusher']
    folk_artists = ['Iron & Wine', 'Fleet Foxes', 'Sufjan Stevens', 'Angel Olsen', 'Big Thief']
    rock_artists = ['Arcade Fire', 'Modest Mouse', 'The National', 'Vampire Weekend', 'Wolf Parade']
    
    all_synthetic_artists = electronic_artists + folk_artists + rock_artists
    synthetic_genres = ['Electronic', 'Indie Folk', 'Indie Rock', 'Ambient', 'Post-Rock']
    
    for i in range(200):
        fma_data.append({
            'release_id': f'FMA_{i:03d}',
            'title': f'Creative Track {i+1}',
            'artist': all_synthetic_artists[i % len(all_synthetic_artists)],
            'year': 2000 + (i % 24),
            'genre': synthetic_genres[i % len(synthetic_genres)],
            'label': 'Creative Commons',
            'rating': round(np.random.normal(3.8, 0.5), 1),
            'review_text': f'Creative Commons track showcasing innovative {synthetic_genres[i % len(synthetic_genres)]} elements',
            'source': 'fma_real_data',
            'duration': np.random.randint(120, 600),
            'plays': np.random.randint(5, 200),
            'favorites': np.random.randint(1, 50)
        })
    
    fma_df = pd.DataFrame(fma_data)
    fma_df = clean_dataframe_types(fma_df)
    
    # Combine datasets
    combined_df = pd.concat([sample_df, fma_df], ignore_index=True)
    combined_df = clean_dataframe_types(combined_df)
    
    return combined_df, len(fma_df)

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
            'collection_id': 'PC_004', 'release_id': 'CLASSIC_004', 
            'purchase_date': '2021-05-12', 'purchase_price': 32.0, 
            'condition': 'Near Mint', 'personal_rating': 8, 'times_played': 18,
            'listening_notes': 'Beautiful piano work. Contemplative mood.'
        },
        {
            'collection_id': 'PC_005', 'release_id': 'CLASSIC_005', 
            'purchase_date': '2021-09-08', 'purchase_price': 25.0, 
            'condition': 'Good+', 'personal_rating': 8, 'times_played': 15,
            'listening_notes': 'Soul jazz gem with incredible energy.'
        },
        {
            'collection_id': 'PC_006', 'release_id': 'CLASSIC_010', 
            'purchase_date': '2022-02-14', 'purchase_price': 55.0, 
            'condition': 'Mint', 'personal_rating': 10, 'times_played': 40,
            'listening_notes': 'Electronic masterpiece. Mind-bending production.'
        },
        {
            'collection_id': 'PC_007', 'release_id': 'CLASSIC_012', 
            'purchase_date': '2022-06-03', 'purchase_price': 30.0, 
            'condition': 'VG+', 'personal_rating': 9, 'times_played': 28,
            'listening_notes': 'Indie folk perfection. Emotional and raw.'
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
            'hard bop': ['art blakey', 'blue note', 'hard', 'driving', 'energetic'],
            'experimental': ['free', 'avant', 'experimental', 'boundary', 'innovative'],
            'electronic': ['synthesizer', 'digital', 'electronic', 'techno', 'ambient', 'daft punk'],
            'folk': ['acoustic', 'traditional', 'storytelling', 'folk', 'roots', 'indie'],
            'uplifting': ['happy', 'positive', 'uplifting', 'energetic', 'joyful'],
            'ambient': ['ambient', 'atmospheric', 'eno', 'soundscape', 'meditative'],
            'indie': ['indie', 'independent', 'alternative', 'underground', 'authentic']
        }
        
        concept_lower = safe_string_operation(concept, 'lower', concept)
        if concept_lower in concept_mappings:
            search_terms = ' '.join(concept_mappings[concept_lower])
            return self.fuzzy_search(search_terms, threshold=0.2)
        else:
            return self.fuzzy_search(concept, threshold=0.25)

# Enhanced metadata extraction
class AdvancedMetadataExtractor:
    def extract_from_text(self, text: str):
        """Enhanced metadata extraction with music-specific patterns"""
        text = str(text) if text is not None else ''
        
        extracted = {
            'artist': None, 'album': None, 'price': None, 
            'condition': None, 'label': None, 'year': None, 'genre': None
        }
        
        # Enhanced artist extraction
        artist_patterns = [
            r'(Miles Davis|John Coltrane|Art Blakey|Bill Evans|Horace Silver|Daft Punk|Radiohead)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'by\s+([A-Z][a-zA-Z\s]+)',
        ]
        
        for pattern in artist_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['artist'] = match.group(1).strip()
                break
        
        # Enhanced album extraction
        album_patterns = [
            r'(Kind of Blue|A Love Supreme|Giant Steps|Blue Train|Waltz for Debby|OK Computer)',
            r'"([^"]+)"',
            r'([A-Z][a-z]+\s+(?:of|for|in|with|to)\s+[A-Z][a-z]+)',
        ]
        
        for pattern in album_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['album'] = match.group(1).strip()
                break
        
        # Price extraction
        price_patterns = [r'\$(\d+(?:\.\d{2})?)', r'paid:?\s*\$?(\d+)', r'cost:?\s*\$?(\d+)']
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    extracted['price'] = float(match.group(1))
                except ValueError:
                    pass
                break
        
        # Condition extraction
        condition_match = re.search(r'(mint|near mint|vg\+?|very good|good\+?|fair|poor)', text, re.IGNORECASE)
        if condition_match:
            extracted['condition'] = condition_match.group(1).upper()
        
        # Label extraction
        label_match = re.search(r'(Blue Note|Columbia|Impulse|Atlantic|Verve|Prestige|ECM|Warp)', text, re.IGNORECASE)
        if label_match:
            extracted['label'] = label_match.group(1)
        
        # Year extraction
        year_match = re.search(r'(19\d{2}|20[0-2]\d)', text)
        if year_match:
            try:
                extracted['year'] = int(year_match.group(1))
            except ValueError:
                pass
        
        # Genre extraction
        genre_match = re.search(r'(jazz|rock|electronic|folk|pop|hip.hop|ambient|indie)', text, re.IGNORECASE)
        if genre_match:
            extracted['genre'] = genre_match.group(1).title()
        
        return extracted

# Enhanced recommendation engine
class EnhancedRecommendationEngine:
    def __init__(self, catalog_df):
        self.catalog = clean_dataframe_types(catalog_df)
        self.fma_data = self.catalog[self.catalog.get('source') == 'fma_real_data'] if 'source' in self.catalog.columns else pd.DataFrame()
        self.curated_data = self.catalog[self.catalog.get('source') == 'curated_classics'] if 'source' in self.catalog.columns else self.catalog
    
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
            confidence = 0.85 if source == 'fma_real_data' else 0.9
            
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
    
    def diversity_recommendations(self, max_per_genre=3):
        """Generate diverse recommendations across genres"""
        recommendations = []
        
        if len(self.catalog) > 0:
            # Get unique genres safely
            genre_series = self.catalog['genre'].dropna().astype(str)
            if len(genre_series) > 0:
                top_genres = genre_series.value_counts().head(6).index
                
                for genre in top_genres:
                    genre_recs = self.genre_based_recommendations(
                        genre, min_rating=4.0, max_results=max_per_genre
                    )
                    recommendations.extend(genre_recs)
        
        # Sort by rating and confidence
        recommendations.sort(key=lambda x: (x.get('rating', 0), x.get('confidence', 0)), reverse=True)
        return recommendations[:15]
    
    def fma_discovery_recommendations(self, max_results=8):
        """Recommendations from real FMA data for music discovery"""
        if len(self.fma_data) == 0:
            return []
        
        recommendations = []
        try:
            # Get diverse high-rated FMA tracks
            fma_sample = self.fma_data.nlargest(max_results * 2, 'rating')
            
            for _, track in fma_sample.head(max_results).iterrows():
                recommendations.append({
                    'title': safe_string_operation(track.get('title', 'Unknown'), 'title', 'Unknown'),
                    'artist': safe_string_operation(track.get('artist', 'Unknown'), 'title', 'Unknown'),
                    'genre': safe_string_operation(track.get('genre', 'Unknown'), 'title', 'Unknown'),
                    'rating': float(track.get('rating', 0)),
                    'year': track.get('year', 'Unknown'),
                    'source': 'fma_real_data',
                    'reason': f"Discover {safe_string_operation(track.get('genre', 'music'))} from Creative Commons catalog",
                    'confidence': 0.75,
                    'plays': int(track.get('plays', 0))
                })
        except Exception as e:
            st.error(f"Error generating FMA recommendations: {str(e)}")
        
        return recommendations

# Load data
catalog_df, fma_count = load_combined_dataset()
personal_df = load_personal_collection()
search_engine = AdvancedSearchEngine(catalog_df)
extractor = AdvancedMetadataExtractor()
rec_engine = EnhancedRecommendationEngine(catalog_df)

# Raw notes for AI processing demo
raw_notes_df = pd.DataFrame([
    {
        'note_id': 'NOTE_001', 
        'raw_text': 'Miles Davis - Kind of Blue Columbia 1959 mint condition bought for $35 incredible trumpet work late night perfect',
        'note_type': 'purchase'
    },
    {
        'note_id': 'NOTE_002', 
        'raw_text': 'A Love Supreme Coltrane Impulse spiritual masterpiece VG+ $42 transcendent jazz experience',
        'note_type': 'review'
    },
    {
        'note_id': 'NOTE_003', 
        'raw_text': 'Want: Art Blakey Moanin, Bill Evans Waltz for Debby, Horace Silver Song for My Father budget $120 Blue Note classics',
        'note_type': 'wishlist'
    },
    {
        'note_id': 'NOTE_004',
        'raw_text': 'Daft Punk Random Access Memories 2013 electronic disco funk amazing production $28 mint condition dance floor gold',
        'note_type': 'purchase'
    }
])

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
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown('<h1 class="main-header">🎵 Smart Vinyl Catalog Pro</h1>', unsafe_allow_html=True)
st.markdown("**Next-Generation AI-Powered Music Collection Management & Discovery Platform**")

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
if fma_count > 0:
    st.sidebar.success(f"✅ FMA Real Data: {fma_count:,} tracks")
    st.sidebar.info(f"📀 Curated Classics: {total_catalog - fma_count} albums")
else:
    st.sidebar.warning("⚠️ Using sample data only")

# Enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 AI Search", "📊 Analytics", "🎯 Recommendations", 
    "🤖 AI Processing", "👥 Social Insights", "📈 Market Intel", "🗄️ Data Explorer"
])

with tab1:
    st.header("🔍 Advanced AI-Powered Search")
    st.markdown("Search across thousands of real music tracks and curated classics")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search your music catalog:",
            placeholder="e.g., spiritual jazz, electronic ambient, Miles Davis cool, folk acoustic"
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
                                if 'source' in row:
                                    source = safe_string_operation(row.get('source', ''))
                                    source_badge = "🎼 Real Data" if source == 'fma_real_data' else "💎 Classic"
                                    st.caption(source_badge)
                            
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
        st.markdown(f'<div class="metric-card"><h3>Total Tracks</h3><h2>{total_catalog:,}</h2><p>Real + Curated</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Avg Rating</h3><h2>{avg_rating:.1f}/5.0</h2><p>Quality Score</p></div>', unsafe_allow_html=True)
    
    with col3:
        unique_artists = catalog_df['artist'].nunique() if 'artist' in catalog_df.columns else 0
        st.markdown(f'<div class="metric-card"><h3>Artists</h3><h2>{unique_artists}</h2><p>Unique Musicians</p></div>', unsafe_allow_html=True)
    
    with col4:
        unique_genres = catalog_df['genre'].nunique() if 'genre' in catalog_df.columns else 0
        st.markdown(f'<div class="metric-card"><h3>Genres</h3><h2>{unique_genres}</h2><p>Musical Styles</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced visualizations with error handling
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Genre distribution with enhanced styling
            if 'genre' in catalog_df.columns:
                genre_counts = catalog_df['genre'].value_counts().head(8)
                if len(genre_counts) > 0:
                    fig_genre = px.pie(
                        values=genre_counts.values,
                        names=genre_counts.index,
                        title="Music Catalog by Genre",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4
                    )
                    fig_genre.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
                    fig_genre.update_layout(showlegend=True, height=400)
                    st.plotly_chart(fig_genre, use_container_width=True)
                else:
                    st.info("No genre data available for visualization")
            else:
                st.info("Genre data not available")
        except Exception as e:
            st.error(f"Error creating genre chart: {str(e)}")
    
    with col2:
        try:
            # Rating distribution
            if 'rating' in catalog_df.columns:
                rating_data = catalog_df['rating'].dropna()
                if len(rating_data) > 0:
                    fig_rating_hist = px.histogram(
                        x=rating_data,
                        nbins=20,
                        title="Rating Distribution",
                        color_discrete_sequence=['#1f77b4'],
                        labels={'x': 'Rating (1-5)', 'count': 'Number of Albums'}
                    )
                    fig_rating_hist.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_rating_hist, use_container_width=True)
                else:
                    st.info("No rating data available for visualization")
            else:
                st.info("Rating data not available")
        except Exception as e:
            st.error(f"Error creating rating chart: {str(e)}")
    
    # Additional analytics
    st.markdown("---")
    st.subheader("🎼 Detailed Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Year distribution
            if 'year' in catalog_df.columns:
                year_data = catalog_df['year'].dropna()
                if len(year_data) > 0:
                    year_counts = year_data.value_counts().head(15).sort_index()
                    fig_year = px.bar(
                        x=year_counts.index,
                        y=year_counts.values,
                        title="Albums by Year",
                        labels={'x': 'Year', 'y': 'Number of Albums'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_year.update_layout(height=350)
                    st.plotly_chart(fig_year, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating year chart: {str(e)}")
    
    with col2:
        try:
            # Top artists
            if 'artist' in catalog_df.columns:
                artist_counts = catalog_df['artist'].value_counts().head(10)
                if len(artist_counts) > 0:
                    fig_artist = px.bar(
                        x=artist_counts.values,
                        y=artist_counts.index,
                        orientation='h',
                        title="Top Artists by Album Count",
                        labels={'x': 'Number of Albums', 'y': 'Artist'},
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig_artist.update_layout(height=350)
                    st.plotly_chart(fig_artist, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating artist chart: {str(e)}")

with tab3:
    st.header("🎯 Smart Music Recommendation Engine")
    st.markdown("AI-powered recommendations using real music data and advanced algorithms")
    
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
            st.session_state['show_diversity_recs'] = False
            st.session_state['show_discovery_recs'] = False
    
    with col2:
        if st.button("🌈 Diversity Recommendations", use_container_width=True):
            st.session_state['show_diversity_recs'] = True
            st.session_state['show_genre_recs'] = False
            st.session_state['show_discovery_recs'] = False
    
    with col3:
        if st.button("🔍 Creative Commons Discovery", use_container_width=True):
            st.session_state['show_discovery_recs'] = True
            st.session_state['show_genre_recs'] = False
            st.session_state['show_diversity_recs'] = False
    
    # Display recommendations based on selection
    if st.session_state.get('show_genre_recs', False):
        with st.spinner("Generating genre-based recommendations..."):
            try:
                if rec_genre != 'All Genres':
                    recommendations = rec_engine.genre_based_recommendations(rec_genre, min_rating, max_recs)
                else:
                    recommendations = rec_engine.diversity_recommendations(max_per_genre=3)
                
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
                                source_badge = "🎼 Real" if rec['source'] == 'fma_real_data' else "💎 Classic"
                                st.caption(source_badge)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if i < len(recommendations) - 1:
                                st.divider()
                else:
                    st.warning("No recommendations found for the selected criteria.")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    if st.session_state.get('show_diversity_recs', False):
        with st.spinner("Generating diverse recommendations..."):
            try:
                recommendations = rec_engine.diversity_recommendations()
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} diverse recommendations across multiple genres")
                    
                    # Group by genre for better display
                    genre_groups = {}
                    for rec in recommendations:
                        genre = rec['genre']
                        if genre not in genre_groups:
                            genre_groups[genre] = []
                        genre_groups[genre].append(rec)
                    
                    for genre, recs in genre_groups.items():
                        st.subheader(f"🎵 {genre}")
                        
                        for rec in recs:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{rec['title']}** *by {rec['artist']}*")
                                st.caption(f"📅 {rec['year']} • 💡 {rec['reason']}")
                            
                            with col2:
                                st.metric("Rating", f"{rec['rating']:.1f}/5")
                            
                            with col3:
                                source_badge = "🎼 Real" if rec['source'] == 'fma_real_data' else "💎 Classic"
                                st.caption(source_badge)
                            
                            st.divider()
                else:
                    st.warning("No diverse recommendations available.")
            except Exception as e:
                st.error(f"Error generating diverse recommendations: {str(e)}")
    
    if st.session_state.get('show_discovery_recs', False):
        with st.spinner("Discovering Creative Commons music..."):
            try:
                recommendations = rec_engine.fma_discovery_recommendations()
                
                if recommendations:
                    st.success(f"Discovered {len(recommendations)} Creative Commons tracks for exploration")
                    st.info("🎼 These are real tracks from independent artists available under Creative Commons licenses")
                    
                    for rec in recommendations:
                        with st.container():
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
                                st.caption("🎼 Creative Commons")
                                st.caption("Free to explore!")
                            
                            st.divider()
                else:
                    st.warning("No Creative Commons tracks available for discovery.")
            except Exception as e:
                st.error(f"Error discovering Creative Commons music: {str(e)}")

with tab4:
    st.header("🤖 AI-Powered Text Processing Demo")
    st.markdown("Transform raw music notes into structured data using advanced NLP")
    
    st.subheader("📝 Raw Music Notes")
    st.markdown("Here are some example raw notes that the AI can process:")
    
    for _, note in raw_notes_df.iterrows():
        with st.expander(f"📋 {note['note_type'].title()} Note - {note['note_id']}"):
            st.write(note['raw_text'])
    
    st.markdown("---")
    st.subheader("🧠 AI Metadata Extraction")
    
    # Interactive text processing
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter your music notes for AI processing:",
            placeholder="e.g., 'Miles Davis Kind of Blue Columbia 1959 mint condition $35 amazing trumpet work'",
            height=100
        )
    
    with col2:
        if st.button("🚀 Process with AI", use_container_width=True, type="primary"):
            if user_text and len(user_text.strip()) > 0:
                with st.spinner("Processing with AI..."):
                    extracted = extractor.extract_from_text(user_text)
                    st.session_state['processed_text'] = extracted
                    st.session_state['original_text'] = user_text
    
    # Display processing results
    if 'processed_text' in st.session_state:
        st.markdown("### 🎯 Extracted Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Original Text:**")
            st.info(st.session_state['original_text'])
        
        with col2:
            st.markdown("**🔍 Extracted Metadata:**")
            extracted = st.session_state['processed_text']
            
            for key, value in extracted.items():
                if value is not None:
                    if key == 'artist':
                        st.write(f"🎤 **Artist:** {value}")
                    elif key == 'album':
                        st.write(f"💿 **Album:** {value}")
                    elif key == 'price':
                        st.write(f"💰 **Price:** ${value}")
                    elif key == 'condition':
                        st.write(f"📀 **Condition:** {value}")
                    elif key == 'label':
                        st.write(f"🏷️ **Label:** {value}")
                    elif key == 'year':
                        st.write(f"📅 **Year:** {value}")
                    elif key == 'genre':
                        st.write(f"🎵 **Genre:** {value}")
    
    # Batch processing demo
    st.markdown("---")
    st.subheader("📊 Batch Processing Results")
    
    if st.button("🔄 Process All Sample Notes"):
        results = []
        
        for _, note in raw_notes_df.iterrows():
            extracted = extractor.extract_from_text(note['raw_text'])
            result = {
                'note_id': note['note_id'],
                'note_type': note['note_type'],
                'original_text': note['raw_text']
            }
            result.update(extracted)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        st.success("✅ Processed all sample notes!")
        st.dataframe(results_df, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            artists_found = results_df['artist'].notna().sum()
            st.metric("Artists Identified", artists_found)
        
        with col2:
            prices_found = results_df['price'].notna().sum()
            st.metric("Prices Extracted", prices_found)
        
        with col3:
            conditions_found = results_df['condition'].notna().sum()
            st.metric("Conditions Found", conditions_found)

with tab5:
    st.header("👥 Social Music Insights")
    st.markdown("Community trends and social features for music discovery")
    
    # Mock social data for demonstration
    social_data = {
        'trending_genres': ['Electronic', 'Indie Folk', 'Jazz', 'Ambient', 'Post-Rock'],
        'viral_tracks': [
            {'title': 'Midnight City', 'artist': 'M83', 'shares': 1250, 'genre': 'Electronic'},
            {'title': 'Holocene', 'artist': 'Bon Iver', 'shares': 980, 'genre': 'Indie Folk'},
            {'title': 'So What', 'artist': 'Miles Davis', 'shares': 850, 'genre': 'Jazz'},
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Trending Genres This Week")
        for i, genre in enumerate(social_data['trending_genres'], 1):
            st.write(f"{i}. **{genre}** 📈")
        
        st.subheader("🎵 Community Favorites")
        genre_matches = catalog_df[catalog_df['genre'].isin(social_data['trending_genres'])]
        if len(genre_matches) > 0:
            top_rated = genre_matches.nlargest(5, 'rating')
            for _, track in top_rated.iterrows():
                st.write(f"⭐ **{track['title']}** by *{track['artist']}* ({track['rating']:.1f}/5)")
    
    with col2:
        st.subheader("📊 Social Engagement")
        
        # Mock engagement metrics
        engagement_metrics = {
            'Total Users': 15420,
            'Active This Week': 3240,
            'Reviews Posted': 1580,
            'Recommendations Made': 892
        }
        
        for metric, value in engagement_metrics.items():
            st.metric(metric, f"{value:,}")
        
        st.subheader("💬 Recent Community Activity")
        activities = [
            "🎵 User @musiclover added 'Kind of Blue' to favorites",
            "⭐ User @jazzfan rated 'A Love Supreme' 5/5 stars",
            "📝 User @vinylhead wrote review for 'OK Computer'",
            "🔄 User @discoverer shared 'Random Access Memories'"
        ]
        
        for activity in activities:
            st.caption(activity)

with tab6:
    st.header("📈 Market Intelligence & Pricing")
    st.markdown("Track market trends and pricing data for informed collecting decisions")
    
    # Mock market data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Market Trend</h3><h2>📈 +12%</h2><p>Jazz Albums</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>Avg Price</h3><h2>$32</h2><p>New Releases</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>Hot Genre</h3><h2>🔥 Electronic</h2><p>This Month</p></div>', unsafe_allow_html=True)
    
    # Price analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Personal Collection Value Analysis")
        if len(personal_df) > 0:
            # Collection value over time (mock data)
            dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='M')
            cumulative_value = np.cumsum(np.random.normal(25, 10, len(dates)))
            
            fig_value = px.line(
                x=dates,
                y=cumulative_value,
                title="Collection Value Growth",
                labels={'x': 'Date', 'y': 'Total Value ($)'}
            )
            fig_value.update_layout(height=350)
            st.plotly_chart(fig_value, use_container_width=True)
            
            # Top valuable items
            st.subheader("💎 Most Valuable Items")
            top_valuable = personal_df.nlargest(5, 'purchase_price')
            for _, item in top_valuable.iterrows():
                st.write(f"💿 **{item['collection_id']}** - ${item['purchase_price']:.0f} ({item['condition']})")
    
    with col2:
        st.subheader("📊 Market Price Trends")
        
        # Mock price trend data
        genres = ['Jazz', 'Electronic', 'Rock', 'Folk', 'Hip-Hop']
        price_changes = [12, 8, -3, 15, 5]  # Percentage changes
        
        fig_trends = px.bar(
            x=genres,
            y=price_changes,
            title="Genre Price Changes (Last 3 Months)",
            labels={'x': 'Genre', 'y': 'Price Change (%)'},
            color=price_changes,
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_trends.update_layout(height=350)
        st.plotly_chart(fig_trends, use_container_width=True)
        
        st.subheader("🎯 Investment Recommendations")
        investment_recs = [
            "📈 **Jazz classics** showing strong upward trend (+12%)",
            "🔥 **Electronic** albums gaining popularity (+8%)",
            "💡 **Indie Folk** emerging as collector favorite (+15%)",
            "⚠️ **Rock** prices slightly declining (-3%)"
        ]
        
        for rec in investment_recs:
            st.markdown(rec)

with tab7:
    st.header("🗄️ Advanced Data Explorer")
    st.markdown("Deep dive into your music data with advanced filtering and analysis")
    
    # Advanced filters
    st.subheader("🔧 Advanced Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        genre_filter = st.multiselect(
            "Filter by Genre:",
            options=sorted(catalog_df['genre'].unique().tolist()) if 'genre' in catalog_df.columns else [],
            default=[]
        )
    
    with col2:
        rating_range = st.slider(
            "Rating Range:",
            min_value=1.0,
            max_value=5.0,
            value=(3.0, 5.0),
            step=0.1
        )
    
    with col3:
        year_range = st.slider(
            "Year Range:",
            min_value=int(catalog_df['year'].min()) if 'year' in catalog_df.columns else 1950,
            max_value=int(catalog_df['year'].max()) if 'year' in catalog_df.columns else 2024,
            value=(1960, 2024)
        )
    
    with col4:
        source_filter = st.multiselect(
            "Data Source:",
            options=['curated_classics', 'fma_real_data'],
            default=['curated_classics', 'fma_real_data']
        )
    
    # Apply filters
    filtered_df = catalog_df.copy()
    
    if genre_filter:
        filtered_df = filtered_df[filtered_df['genre'].isin(genre_filter)]
    
    if 'rating' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['rating'] >= rating_range[0]) & 
            (filtered_df['rating'] <= rating_range[1])
        ]
    
    if 'year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) & 
            (filtered_df['year'] <= year_range[1])
        ]
    
    if source_filter and 'source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['source'].isin(source_filter)]
    
    st.subheader(f"📊 Filtered Results ({len(filtered_df):,} tracks)")
    
    if len(filtered_df) > 0:
        # Display options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_columns = st.multiselect(
                "Select columns to display:",
                options=filtered_df.columns.tolist(),
                default=['title', 'artist', 'genre', 'year', 'rating', 'source']
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                options=['rating', 'year', 'title', 'artist', 'plays'],
                index=0
            )
            sort_ascending = st.checkbox("Ascending order", value=False)
        
        if display_columns:
            # Sort and display
            display_df = filtered_df[display_columns].copy()
            if sort_by in display_df.columns:
                display_df = display_df.sort_values(sort_by, ascending=sort_ascending)
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Export options
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"music_catalog_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("📊 Generate Report"):
                    st.success("Report generated! Check the analytics above.")
        
        # Summary statistics for filtered data
        st.subheader("📈 Filtered Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tracks", len(filtered_df))
        
        with col2:
            if 'rating' in filtered_df.columns:
                avg_rating = filtered_df['rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.1f}/5")
        
        with col3:
            unique_artists = filtered_df['artist'].nunique() if 'artist' in filtered_df.columns else 0
            st.metric("Unique Artists", unique_artists)
        
        with col4:
            if 'year' in filtered_df.columns:
                year_span = filtered_df['year'].max() - filtered_df['year'].min()
                st.metric("Year Span", f"{int(year_span)} years")
    
    else:
        st.warning("No tracks match the selected filters. Try adjusting your criteria.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎵 Smart Vinyl Catalog Pro</h4>
    <p>Next-Generation AI-Powered Music Collection Management</p>
    <p>Built with Streamlit • Powered by Advanced Analytics & Real Music Data</p>
</div>
""", unsafe_allow_html=True)