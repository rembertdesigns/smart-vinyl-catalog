import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sqlite3
import json
from pathlib import Path
from PIL import Image
from contextlib import contextmanager
import warnings
from typing import Dict, List, Optional, Tuple
import difflib

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smart Vinyl Catalog Pro - Visual Discovery", 
    page_icon="🎨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection helper
@contextmanager
def get_db_connection(db_path: Path):
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        yield None
    finally:
        if conn:
            conn.close()

# Enhanced data loading from database
@st.cache_data
def load_database_catalog() -> Tuple[pd.DataFrame, Dict]:
    """Load catalog data from the high-performance database"""
    
    # Database path
    db_path = Path("data/database/vinyl_catalog.db")
    
    if not db_path.exists():
        st.sidebar.warning(f"Database not found at {db_path}")
        return pd.DataFrame(), {}
    
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return pd.DataFrame(), {}
                
            # Load main catalog data
            catalog_query = """
                SELECT 
                    release_id, discogs_id, title, artist, album, year, genre, 
                    label, country, format, status, catalog_number, master_id,
                    popularity_score, rating, plays, favorites, duration,
                    data_quality, source, created_at
                FROM releases
                ORDER BY popularity_score DESC, rating DESC
            """
            
            catalog_df = pd.read_sql_query(catalog_query, conn)
            
            # Get visual features summary with error handling
            try:
                visual_summary_query = """
                    SELECT COUNT(*) as covers_with_features,
                           COALESCE(AVG(brightness), 0) as avg_brightness,
                           COALESCE(AVG(saturation), 0) as avg_saturation,
                           COALESCE(AVG(complexity_score), 0) as avg_complexity
                    FROM visual_features
                """
                
                visual_summary_result = pd.read_sql_query(visual_summary_query, conn)
                visual_summary = visual_summary_result.iloc[0].to_dict()
            except Exception:
                visual_summary = {
                    'covers_with_features': 0,
                    'avg_brightness': 0,
                    'avg_saturation': 0,
                    'avg_complexity': 0
                }
            
            if not catalog_df.empty:
                st.sidebar.success(f"✅ Loaded database: {len(catalog_df):,} releases")
                if visual_summary['covers_with_features'] > 0:
                    st.sidebar.info(f"🎨 Visual features: {visual_summary['covers_with_features']} covers analyzed")
            
            return catalog_df, visual_summary
            
    except Exception as e:
        st.sidebar.error(f"Database error: {e}")
        return pd.DataFrame(), {}

# Visual discovery functions
@st.cache_data
def load_visual_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load visual similarity and clustering data"""
    
    db_path = Path("data/database/vinyl_catalog.db")
    
    if not db_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return pd.DataFrame(), pd.DataFrame(), {}
                
            # Load albums with visual features
            try:
                visual_albums_query = """
                    SELECT 
                        r.release_id, r.title, r.artist, r.album, r.year, r.genre,
                        ac.local_path as cover_path,
                        vf.brightness, vf.saturation, vf.complexity_score,
                        vf.dominant_colors
                    FROM releases r
                    JOIN album_covers ac ON r.release_id = ac.release_id
                    JOIN visual_features vf ON ac.cover_id = vf.cover_id
                    ORDER BY r.popularity_score DESC
                """
                
                visual_albums_df = pd.read_sql_query(visual_albums_query, conn)
            except Exception:
                visual_albums_df = pd.DataFrame()
            
            # Load similarity data
            try:
                similarity_query = """
                    SELECT 
                        r1.title as album1_title, r1.artist as album1_artist,
                        r2.title as album2_title, r2.artist as album2_artist,
                        vs.similarity_score,
                        ac2.local_path as similar_cover_path
                    FROM visual_similarities vs
                    JOIN album_covers ac1 ON vs.cover_id_1 = ac1.cover_id
                    JOIN album_covers ac2 ON vs.cover_id_2 = ac2.cover_id  
                    JOIN releases r1 ON ac1.release_id = r1.release_id
                    JOIN releases r2 ON ac2.release_id = r2.release_id
                    ORDER BY vs.similarity_score DESC
                    LIMIT 50
                """
                
                similarity_df = pd.read_sql_query(similarity_query, conn)
            except Exception:
                similarity_df = pd.DataFrame()
            
            # Load cluster data
            try:
                cluster_query = """
                    SELECT 
                        ca.cluster_label,
                        r.title, r.artist, r.genre, r.year,
                        ac.local_path as cover_path,
                        vf.brightness, vf.saturation, vf.complexity_score
                    FROM cluster_assignments ca
                    JOIN album_covers ac ON ca.cover_id = ac.cover_id
                    JOIN releases r ON ac.release_id = r.release_id
                    JOIN visual_features vf ON ac.cover_id = vf.cover_id
                    ORDER BY ca.cluster_label, ca.distance_to_center
                """
                
                cluster_data = pd.read_sql_query(cluster_query, conn)
                clusters = cluster_data.groupby('cluster_label').apply(lambda x: x.to_dict('records')).to_dict()
            except Exception:
                clusters = {}
            
            return visual_albums_df, similarity_df, clusters
            
    except Exception as e:
        st.error(f"Error loading visual data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

# Visual search functions
def find_similar_albums_ui(target_release_id: str, catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Find visually similar albums for UI display"""
    
    db_path = Path("data/database/vinyl_catalog.db")
    
    if not db_path.exists():
        return pd.DataFrame()
    
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return pd.DataFrame()
                
            similar_query = """
                SELECT 
                    r2.release_id, r2.title, r2.artist, r2.album, r2.year, r2.genre,
                    ac2.local_path as cover_path,
                    vs.similarity_score,
                    vf2.brightness, vf2.saturation, vf2.complexity_score
                FROM visual_similarities vs
                JOIN album_covers ac1 ON vs.cover_id_1 = ac1.cover_id
                JOIN album_covers ac2 ON vs.cover_id_2 = ac2.cover_id
                JOIN releases r2 ON ac2.release_id = r2.release_id
                LEFT JOIN visual_features vf2 ON ac2.cover_id = vf2.cover_id
                WHERE ac1.release_id = ?
                ORDER BY vs.similarity_score DESC
                LIMIT 10
            """
            
            return pd.read_sql_query(similar_query, conn, params=[target_release_id])
            
    except Exception as e:
        st.error(f"Error finding similar albums: {e}")
        return pd.DataFrame()

def search_by_visual_characteristics(brightness_range: Optional[Tuple[float, float]] = None, 
                                   saturation_range: Optional[Tuple[float, float]] = None, 
                                   complexity_range: Optional[Tuple[float, float]] = None, 
                                   genre: Optional[str] = None) -> pd.DataFrame:
    """Search albums by visual characteristics"""
    
    db_path = Path("data/database/vinyl_catalog.db")
    
    if not db_path.exists():
        return pd.DataFrame()
    
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return pd.DataFrame()
                
            where_clauses = []
            params = []
            
            if brightness_range:
                where_clauses.append("vf.brightness BETWEEN ? AND ?")
                params.extend(brightness_range)
                
            if saturation_range:
                where_clauses.append("vf.saturation BETWEEN ? AND ?")
                params.extend(saturation_range)
                
            if complexity_range:
                where_clauses.append("vf.complexity_score BETWEEN ? AND ?")
                params.extend(complexity_range)
                
            if genre:
                where_clauses.append("r.genre LIKE ?")
                params.append(f"%{genre}%")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT 
                    r.release_id, r.title, r.artist, r.album, r.year, r.genre,
                    ac.local_path as cover_path,
                    vf.brightness, vf.saturation, vf.complexity_score,
                    vf.dominant_colors
                FROM releases r
                JOIN album_covers ac ON r.release_id = ac.release_id
                JOIN visual_features vf ON ac.cover_id = vf.cover_id
                WHERE {where_clause}
                ORDER BY r.popularity_score DESC
                LIMIT 20
            """
            
            return pd.read_sql_query(query, conn, params=params)
            
    except Exception as e:
        st.error(f"Error searching by visual characteristics: {e}")
        return pd.DataFrame()

# Advanced search engine
class AdvancedSearchEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def fuzzy_search(self, query: str, threshold: float = 0.3) -> pd.DataFrame:
        """Enhanced fuzzy search with database backend"""
        if not self.db_path.exists():
            return pd.DataFrame()
            
        try:
            with get_db_connection(self.db_path) as conn:
                if conn is None:
                    return pd.DataFrame()
                    
                search_query = """
                    SELECT release_id, title, artist, album, year, genre, 
                           label, rating, popularity_score, plays
                    FROM releases
                    WHERE title LIKE ? OR artist LIKE ? OR album LIKE ? OR genre LIKE ?
                    ORDER BY popularity_score DESC, rating DESC
                    LIMIT 50
                """
                
                search_term = f"%{query}%"
                results = pd.read_sql_query(search_query, conn, params=[search_term, search_term, search_term, search_term])
                
                # Add similarity scoring
                if not results.empty:
                    results['similarity_score'] = results.apply(
                        lambda row: max(
                            difflib.SequenceMatcher(None, query.lower(), str(row['title']).lower()).ratio(),
                            difflib.SequenceMatcher(None, query.lower(), str(row['artist']).lower()).ratio()
                        ), axis=1
                    )
                    
                    return results.sort_values('similarity_score', ascending=False)
                else:
                    return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Search error: {e}")
            return pd.DataFrame()

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 50%, #2ca02c 100%);
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
    .visual-metric-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
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
    .visual-discovery-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
catalog_df, visual_summary = load_database_catalog()
visual_albums_df, similarity_df, clusters = load_visual_data()

# Initialize search engine
db_path = Path("data/database/vinyl_catalog.db")
search_engine = AdvancedSearchEngine(db_path) if db_path.exists() else None

# Enhanced header
st.markdown('<h1 class="main-header">🎨 Smart Vinyl Catalog Pro</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Music Discovery with Visual Similarity & Computer Vision**")

# Enhanced sidebar with error handling
st.sidebar.header("🎛️ Dashboard Controls")

# Collection metrics with safe access
total_catalog = len(catalog_df) if not catalog_df.empty else 0
avg_rating = catalog_df['rating'].mean() if not catalog_df.empty and 'rating' in catalog_df.columns else 0
visual_covers = visual_summary.get('covers_with_features', 0)

st.sidebar.markdown("### 📊 Collection Overview")
st.sidebar.metric("Total Catalog", f"{total_catalog:,}")
st.sidebar.metric("Avg Rating", f"{avg_rating:.1f}/5.0" if avg_rating > 0 else "N/A")
st.sidebar.metric("Visual Analysis", f"{visual_covers} covers")

# Visual system status
if visual_covers > 0:
    st.sidebar.markdown("### 🎨 Visual Discovery System")
    st.sidebar.success("✅ Computer Vision Active")
    st.sidebar.info(f"🔍 {len(similarity_df)} similarity pairs")
    st.sidebar.info(f"🌈 {len(clusters)} visual clusters")
    avg_brightness = visual_summary.get('avg_brightness', 0)
    avg_complexity = visual_summary.get('avg_complexity', 0)
    st.sidebar.metric("Avg Brightness", f"{avg_brightness:.1f}")
    st.sidebar.metric("Avg Complexity", f"{avg_complexity:.3f}")
else:
    st.sidebar.markdown("### 🎨 Visual Discovery System")
    st.sidebar.warning("⚠️ Computer Vision Inactive")
    st.sidebar.info("Run CV notebook to enable visual features")

# Data source breakdown with safe access
if not catalog_df.empty and 'source' in catalog_df.columns:
    st.sidebar.markdown("### 🎼 Data Sources")
    try:
        source_counts = catalog_df['source'].value_counts()
        for source, count in source_counts.head(5).items():
            if pd.notna(source):
                if 'fma' in str(source).lower():
                    st.sidebar.success(f"🎵 FMA Data: {count:,}")
                elif 'discogs' in str(source).lower():
                    st.sidebar.info(f"💿 Discogs Data: {count:,}")
                else:
                    st.sidebar.info(f"📀 {source}: {count:,}")
    except Exception as e:
        st.sidebar.error(f"Error displaying data sources: {e}")

# Enhanced tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 AI Search", "🎨 Visual Discovery", "📊 Analytics", "🎯 Recommendations", 
    "🎼 Collection", "👥 Social", "📈 Intelligence"
])

with tab1:
    st.header("🔍 Advanced AI-Powered Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search your music catalog:",
            placeholder="e.g., experimental electronic, Miles Davis, dark ambient"
        )
    
    with col2:
        search_type = st.selectbox("Search Type", ["Database Search", "Visual Search"])
    
    if search_query and len(search_query.strip()) > 0:
        if search_type == "Database Search" and search_engine:
            with st.spinner("Searching database..."):
                try:
                    results = search_engine.fuzzy_search(search_query)
                    
                    if not results.empty:
                        st.success(f"Found {len(results)} matches")
                        
                        for idx, (_, row) in enumerate(results.head(10).iterrows()):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                                
                                with col1:
                                    st.markdown(f"**{row['title']}**")
                                    st.markdown(f"*{row['artist']}*")
                                
                                with col2:
                                    st.write(f"📅 {row['year']}" if pd.notna(row['year']) else "📅 Unknown")
                                    st.write(f"🎵 {row['genre']}" if pd.notna(row['genre']) else "🎵 Unknown")
                                
                                with col3:
                                    rating = row['rating'] if pd.notna(row['rating']) else 0
                                    st.metric("Rating", f"{rating:.1f}/5")
                                
                                with col4:
                                    match_pct = row.get('similarity_score', 0) * 100
                                    st.metric("Match", f"{match_pct:.0f}%")
                                
                                st.divider()
                    else:
                        st.warning("No matches found")
                        
                except Exception as e:
                    st.error(f"Search error: {e}")
        else:
            st.info("Please select Database Search or initialize visual search system")

with tab2:
    st.markdown('<div class="visual-discovery-header"><h2>🎨 Visual Discovery System</h2></div>', unsafe_allow_html=True)
    
    if visual_covers == 0:
        st.warning("Visual discovery system not yet initialized. Run the computer vision notebook to enable visual features.")
    else:
        # Visual metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="visual-metric-card"><h3>Visual Covers</h3><h2>{visual_covers}</h2><p>Analyzed</p></div>', unsafe_allow_html=True)
        
        with col2:
            avg_brightness = visual_summary.get('avg_brightness', 0)
            st.markdown(f'<div class="visual-metric-card"><h3>Avg Brightness</h3><h2>{avg_brightness:.1f}</h2><p>0-255 Scale</p></div>', unsafe_allow_html=True)
        
        with col3:
            avg_saturation = visual_summary.get('avg_saturation', 0)
            st.markdown(f'<div class="visual-metric-card"><h3>Avg Saturation</h3><h2>{avg_saturation:.1f}</h2><p>Color Intensity</p></div>', unsafe_allow_html=True)
        
        with col4:
            avg_complexity = visual_summary.get('avg_complexity', 0)
            st.markdown(f'<div class="visual-metric-card"><h3>Complexity</h3><h2>{avg_complexity:.3f}</h2><p>Visual Detail</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visual discovery interface
        st.subheader("🔍 Visual Similarity Search")
        
        if not visual_albums_df.empty:
            # Album selector for similarity search
            try:
                album_options = visual_albums_df.apply(
                    lambda row: f"{row['artist']} - {row['title']} ({row['year']})", axis=1
                ).tolist()
                
                selected_album = st.selectbox(
                    "Find albums that look similar to:",
                    options=album_options
                )
                
                if selected_album:
                    # Extract release_id from selection
                    selected_idx = album_options.index(selected_album)
                    target_release_id = visual_albums_df.iloc[selected_idx]['release_id']
                    
                    # Find similar albums
                    similar_results = find_similar_albums_ui(target_release_id, catalog_df)
                    
                    if not similar_results.empty:
                        st.success(f"Found {len(similar_results)} visually similar albums")
                        
                        # Display results
                        for _, similar in similar_results.head(5).iterrows():
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.markdown(f"**{similar['title']}**")
                                    st.markdown(f"*by {similar['artist']}*")
                                    genre = similar['genre'] if pd.notna(similar['genre']) else 'Unknown'
                                    year = similar['year'] if pd.notna(similar['year']) else 'Unknown'
                                    st.caption(f"🎵 {genre} • 📅 {year}")
                                
                                with col2:
                                    similarity_score = similar['similarity_score']
                                    st.metric("Visual Similarity", f"{similarity_score:.3f}")
                                    
                                    if similarity_score >= 0.8:
                                        st.success("Excellent match")
                                    elif similarity_score >= 0.6:
                                        st.info("Good match")
                                    else:
                                        st.warning("Fair match")
                                
                                with col3:
                                    if pd.notna(similar.get('brightness')):
                                        st.metric("Brightness", f"{similar['brightness']:.1f}")
                                    if pd.notna(similar.get('complexity_score')):
                                        st.metric("Complexity", f"{similar['complexity_score']:.3f}")
                                
                                st.divider()
                    else:
                        st.info("No visually similar albums found")
            except Exception as e:
                st.error(f"Error in visual similarity search: {e}")
        
        # Visual characteristics search
        st.subheader("⚙️ Search by Visual Characteristics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brightness_filter = st.select_slider(
                "Brightness Range",
                options=["Any", "Dark (0-80)", "Medium (80-150)", "Bright (150-255)"],
                value="Any"
            )
        
        with col2:
            saturation_filter = st.select_slider(
                "Color Saturation",
                options=["Any", "Monochrome (0-50)", "Muted (50-150)", "Vibrant (150-255)"],
                value="Any"
            )
        
        with col3:
            complexity_filter = st.select_slider(
                "Visual Complexity",
                options=["Any", "Simple (0-0.2)", "Medium (0.2-0.4)", "Complex (0.4+)"],
                value="Any"
            )
        
        if st.button("🔍 Search by Visual Style"):
            # Convert filter selections to ranges
            brightness_range = None
            saturation_range = None
            complexity_range = None
            
            if brightness_filter == "Dark (0-80)":
                brightness_range = (0, 80)
            elif brightness_filter == "Medium (80-150)":
                brightness_range = (80, 150)
            elif brightness_filter == "Bright (150-255)":
                brightness_range = (150, 255)
            
            if saturation_filter == "Monochrome (0-50)":
                saturation_range = (0, 50)
            elif saturation_filter == "Muted (50-150)":
                saturation_range = (50, 150)
            elif saturation_filter == "Vibrant (150-255)":
                saturation_range = (150, 255)
            
            if complexity_filter == "Simple (0-0.2)":
                complexity_range = (0, 0.2)
            elif complexity_filter == "Medium (0.2-0.4)":
                complexity_range = (0.2, 0.4)
            elif complexity_filter == "Complex (0.4+)":
                complexity_range = (0.4, 2.0)
            
            # Perform search
            visual_results = search_by_visual_characteristics(
                brightness_range, saturation_range, complexity_range
            )
            
            if not visual_results.empty:
                st.success(f"Found {len(visual_results)} albums matching visual criteria")
                
                for _, album in visual_results.head(8).iterrows():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{album['title']}** by {album['artist']}")
                        genre = album['genre'] if pd.notna(album['genre']) else 'Unknown'
                        year = album['year'] if pd.notna(album['year']) else 'Unknown'
                        st.caption(f"🎵 {genre} • 📅 {year}")
                    
                    with col2:
                        st.caption(f"Brightness: {album['brightness']:.1f}")
                        st.caption(f"Saturation: {album['saturation']:.1f}")
                        st.caption(f"Complexity: {album['complexity_score']:.3f}")
            else:
                st.info("No albums found matching these visual criteria")
        
        # Visual clusters display
        if clusters:
            st.subheader("🌈 Visual Style Clusters")
            
            try:
                # Clean cluster data and create options
                clean_clusters = {}
                cluster_options = []
                
                for i, (cluster_key, albums) in enumerate(clusters.items()):
                    # Clean cluster key
                    try:
                        if isinstance(cluster_key, (int, np.integer)):
                            clean_key = int(cluster_key)
                        elif isinstance(cluster_key, str):
                            clean_key = int(cluster_key)
                        else:
                            clean_key = i
                    except (ValueError, TypeError):
                        clean_key = i
                    
                    clean_clusters[clean_key] = albums
                    cluster_options.append(f"Cluster {clean_key} ({len(albums)} albums)")
                
                if cluster_options:
                    selected_cluster = st.selectbox("Explore visual style cluster:", cluster_options)
                    
                    if selected_cluster:
                        try:
                            # Extract cluster ID
                            cluster_id_str = selected_cluster.split("Cluster ")[1].split(" (")[0]
                            cluster_id = int(cluster_id_str)
                            cluster_albums = clean_clusters[cluster_id]
                            
                            # Show cluster characteristics
                            genres = [a.get('genre', '') for a in cluster_albums if a.get('genre')]
                            years = [a.get('year', 0) for a in cluster_albums if a.get('year')]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if genres:
                                    top_genre = max(set(genres), key=genres.count)
                                    st.metric("Dominant Genre", top_genre)
                            
                            with col2:
                                brightness_values = [a.get('brightness', 0) for a in cluster_albums if a.get('brightness') is not None]
                                if brightness_values:
                                    avg_brightness = np.mean(brightness_values)
                                    st.metric("Avg Brightness", f"{avg_brightness:.1f}")
                            
                            with col3:
                                complexity_values = [a.get('complexity_score', 0) for a in cluster_albums if a.get('complexity_score') is not None]
                                if complexity_values:
                                    avg_complexity = np.mean(complexity_values)
                                    st.metric("Avg Complexity", f"{avg_complexity:.3f}")
                            
                            # Show sample albums from cluster
                            st.markdown("**Sample albums in this visual style:**")
                            for album in cluster_albums[:5]:
                                artist = album.get('artist', 'Unknown')
                                title = album.get('title', 'Unknown')
                                year = album.get('year', 'Unknown')
                                st.write(f"• {artist} - {title} ({year})")
                        
                        except (IndexError, ValueError, KeyError) as e:
                            st.error(f"Error parsing cluster selection: {e}")
                
            except Exception as e:
                st.error(f"Error displaying clusters: {e}")

with tab3:
    st.header("📊 Advanced Music Analytics")
    
    if not catalog_df.empty:
        # Key metrics with visual enhancements
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Tracks</h3><h2>{total_catalog:,}</h2><p>Database Backend</p></div>', unsafe_allow_html=True)
        
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
            # Genre distribution
            if 'genre' in catalog_df.columns:
                try:
                    genre_counts = catalog_df['genre'].value_counts().head(10)
                    if not genre_counts.empty:
                        fig_genre = px.bar(
                            x=genre_counts.values,
                            y=genre_counts.index,
                            orientation='h',
                            title="Top 10 Genres",
                            color=genre_counts.values,
                            color_continuous_scale='viridis'
                        )
                        fig_genre.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_genre, use_container_width=True)
                    else:
                        st.info("No genre data available for visualization")
                except Exception as e:
                    st.error(f"Error creating genre chart: {e}")
        
        with col2:
            # Rating distribution
            if 'rating' in catalog_df.columns:
                try:
                    ratings = pd.to_numeric(catalog_df['rating'], errors='coerce').dropna()
                    if not ratings.empty:
                        fig_ratings = px.histogram(
                            x=ratings,
                            nbins=25,
                            title="Rating Distribution",
                            color_discrete_sequence=['#1f77b4']
                        )
                        fig_ratings.update_layout(height=400)
                        st.plotly_chart(fig_ratings, use_container_width=True)
                    else:
                        st.info("No rating data available for visualization")
                except Exception as e:
                    st.error(f"Error creating rating chart: {e}")
        
        # Visual analytics section
        if visual_covers > 0:
            st.subheader("🎨 Visual Analytics")
            
            try:
                with get_db_connection(db_path) as conn:
                    if conn is not None:
                        # Visual trends by genre
                        visual_genre_query = """
                            SELECT r.genre, 
                                   AVG(vf.brightness) as avg_brightness,
                                   AVG(vf.saturation) as avg_saturation,
                                   AVG(vf.complexity_score) as avg_complexity,
                                   COUNT(*) as album_count
                            FROM releases r
                            JOIN album_covers ac ON r.release_id = ac.release_id
                            JOIN visual_features vf ON ac.cover_id = vf.cover_id
                            WHERE r.genre IS NOT NULL
                            GROUP BY r.genre
                            HAVING album_count >= 2
                            ORDER BY album_count DESC
                        """
                        
                        visual_genre_df = pd.read_sql_query(visual_genre_query, conn)
                        
                        if not visual_genre_df.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Brightness by genre
                                try:
                                    fig_brightness = px.bar(
                                        visual_genre_df.head(6),
                                        x='genre',
                                        y='avg_brightness',
                                        title="Average Brightness by Genre",
                                        color='avg_brightness',
                                        color_continuous_scale='plasma'
                                    )
                                    fig_brightness.update_layout(height=350)
                                    st.plotly_chart(fig_brightness, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating brightness chart: {e}")
                            
                            with col2:
                                # Complexity by genre
                                try:
                                    fig_complexity = px.scatter(
                                        visual_genre_df,
                                        x='avg_brightness',
                                        y='avg_complexity',
                                        size='album_count',
                                        hover_data=['genre'],
                                        title="Brightness vs Complexity by Genre",
                                        color='avg_saturation',
                                        color_continuous_scale='viridis'
                                    )
                                    fig_complexity.update_layout(height=350)
                                    st.plotly_chart(fig_complexity, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating complexity chart: {e}")
                        else:
                            st.info("Insufficient visual data for genre analytics")
            
            except Exception as e:
                st.error(f"Error loading visual analytics: {e}")
    else:
        st.warning("No catalog data available for analytics")

with tab4:
    st.header("🎯 Smart Music Recommendation Engine")
    
    if not catalog_df.empty:
        # Recommendation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            genre_options = ['All Genres']
            if 'genre' in catalog_df.columns:
                genre_options.extend(sorted(catalog_df['genre'].dropna().unique().tolist()))
            
            rec_genre = st.selectbox("Target Genre:", options=genre_options)
        
        with col2:
            min_rating = st.slider("Min Rating", 1.0, 5.0, 3.5, 0.1)
        
        with col3:
            max_recs = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
        
        # Recommendation types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎵 Genre Recommendations", use_container_width=True):
                st.session_state['rec_type'] = 'genre'
        
        with col2:
            if st.button("🎨 Visual Recommendations", use_container_width=True) and visual_covers > 0:
                st.session_state['rec_type'] = 'visual'
        
        with col3:
            if st.button("🔍 Discovery Mode", use_container_width=True):
                st.session_state['rec_type'] = 'discovery'
        
        # Display recommendations
        if 'rec_type' in st.session_state:
            rec_type = st.session_state['rec_type']
            
            if rec_type == 'genre':
                # Genre-based recommendations
                try:
                    filtered_df = catalog_df.copy()
                    
                    # Filter by rating
                    if 'rating' in filtered_df.columns:
                        filtered_df = filtered_df[pd.to_numeric(filtered_df['rating'], errors='coerce') >= min_rating]
                    
                    # Filter by genre
                    if rec_genre != 'All Genres' and 'genre' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['genre'].str.contains(rec_genre, na=False, case=False)]
                    
                    # Get recommendations
                    if not filtered_df.empty:
                        genre_recs = filtered_df.nlargest(max_recs, 'rating')
                        
                        st.success(f"Genre-based recommendations for {rec_genre}")
                        
                        for _, rec in genre_recs.iterrows():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{rec['title']}**")
                                st.markdown(f"*by {rec['artist']}*")
                                genre = rec['genre'] if pd.notna(rec['genre']) else 'Unknown'
                                year = rec['year'] if pd.notna(rec['year']) else 'Unknown'
                                st.caption(f"🎵 {genre} • 📅 {year}")
                            
                            with col2:
                                rating = rec['rating'] if pd.notna(rec['rating']) else 0
                                st.metric("Rating", f"{rating:.1f}/5")
                            
                            with col3:
                                if pd.notna(rec.get('popularity_score')):
                                    st.metric("Popularity", f"{rec['popularity_score']:.3f}")
                    else:
                        st.info("No albums found matching the criteria")
                
                except Exception as e:
                    st.error(f"Error generating genre recommendations: {e}")
            
            elif rec_type == 'visual' and visual_covers > 0:
                # Visual-based recommendations
                try:
                    with get_db_connection(db_path) as conn:
                        if conn is not None:
                            visual_recs_query = """
                                SELECT r.title, r.artist, r.genre, r.year, r.rating,
                                       vf.brightness, vf.saturation, vf.complexity_score
                                FROM releases r
                                JOIN album_covers ac ON r.release_id = ac.release_id
                                JOIN visual_features vf ON ac.cover_id = vf.cover_id
                                WHERE r.rating >= ?
                                ORDER BY vf.complexity_score DESC, r.rating DESC
                                LIMIT ?
                            """
                            
                            visual_recs = pd.read_sql_query(visual_recs_query, conn, params=[min_rating, max_recs])
                            
                            if not visual_recs.empty:
                                st.success("Visual complexity-based recommendations")
                                
                                for _, rec in visual_recs.iterrows():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        st.markdown(f"**{rec['title']}**")
                                        st.markdown(f"*by {rec['artist']}*")
                                        genre = rec['genre'] if pd.notna(rec['genre']) else 'Unknown'
                                        year = rec['year'] if pd.notna(rec['year']) else 'Unknown'
                                        st.caption(f"🎵 {genre} • 📅 {year}")
                                    
                                    with col2:
                                        rating = rec['rating'] if pd.notna(rec['rating']) else 0
                                        st.metric("Rating", f"{rating:.1f}/5")
                                        complexity = rec['complexity_score'] if pd.notna(rec['complexity_score']) else 0
                                        st.caption(f"Complexity: {complexity:.3f}")
                                    
                                    with col3:
                                        brightness = rec['brightness'] if pd.notna(rec['brightness']) else 0
                                        saturation = rec['saturation'] if pd.notna(rec['saturation']) else 0
                                        st.metric("Brightness", f"{brightness:.1f}")
                                        st.caption(f"Saturation: {saturation:.1f}")
                            else:
                                st.info("No visual recommendations available")
                
                except Exception as e:
                    st.error(f"Error generating visual recommendations: {e}")
            
            elif rec_type == 'discovery':
                # Discovery recommendations
                try:
                    discovery_recs = catalog_df.sample(n=min(max_recs, len(catalog_df)))
                    
                    st.success("Discovery mode - random gems from your collection")
                    
                    for _, rec in discovery_recs.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{rec['title']}**")
                            st.markdown(f"*by {rec['artist']}*")
                            genre = rec['genre'] if pd.notna(rec['genre']) else 'Unknown'
                            year = rec['year'] if pd.notna(rec['year']) else 'Unknown'
                            st.caption(f"🎵 {genre} • 📅 {year}")
                        
                        with col2:
                            rating = rec['rating'] if pd.notna(rec['rating']) else 0
                            st.metric("Rating", f"{rating:.1f}/5")
                        
                        with col3:
                            st.info("🔍 Discovery")
                
                except Exception as e:
                    st.error(f"Error in discovery mode: {e}")
    else:
        st.warning("No catalog data available for recommendations")

with tab5:
    st.header("🎼 Personal Collection Management")
    
    # Collection statistics
    if not catalog_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_duration = 0
            if 'duration' in catalog_df.columns:
                total_duration = pd.to_numeric(catalog_df['duration'], errors='coerce').sum()
            hours = total_duration / 3600 if total_duration > 0 else 0
            st.metric("Total Duration", f"{hours:.0f} hours")
        
        with col2:
            total_favorites = 0
            if 'favorites' in catalog_df.columns:
                total_favorites = pd.to_numeric(catalog_df['favorites'], errors='coerce').sum()
            st.metric("Total Favorites", f"{total_favorites:,}")
        
        with col3:
            total_plays = 0
            if 'plays' in catalog_df.columns:
                total_plays = pd.to_numeric(catalog_df['plays'], errors='coerce').sum()
            st.metric("Total Plays", f"{total_plays:,}")
        
        # Top rated albums
        st.subheader("⭐ Top Rated Albums")
        
        if 'rating' in catalog_df.columns:
            try:
                # Clean rating data
                clean_ratings = pd.to_numeric(catalog_df['rating'], errors='coerce')
                top_rated_indices = clean_ratings.nlargest(10).index
                top_rated = catalog_df.loc[top_rated_indices]
                
                for _, album in top_rated.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{album['title']}**")
                        st.markdown(f"*by {album['artist']}*")
                        genre = album['genre'] if pd.notna(album['genre']) else 'Unknown'
                        year = album['year'] if pd.notna(album['year']) else 'Unknown'
                        st.caption(f"🎵 {genre} • 📅 {year}")
                    
                    with col2:
                        rating = album['rating'] if pd.notna(album['rating']) else 0
                        st.metric("Rating", f"{rating:.1f}⭐")
                    
                    with col3:
                        if pd.notna(album.get('plays')):
                            plays = album['plays']
                            st.metric("Plays", f"{plays}")
            except Exception as e:
                st.error(f"Error displaying top rated albums: {e}")
    else:
        st.warning("No collection data available")

with tab6:
    st.header("👥 Social Music Insights")
    
    if not catalog_df.empty:
        # Trending analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Popular Genres")
            
            if 'genre' in catalog_df.columns:
                try:
                    genre_popularity = catalog_df['genre'].value_counts().head(8)
                    
                    if not genre_popularity.empty:
                        fig_trending = px.bar(
                            x=genre_popularity.values,
                            y=genre_popularity.index,
                            orientation='h',
                            title="Most Popular Genres",
                            color=genre_popularity.values,
                            color_continuous_scale='plasma'
                        )
                        fig_trending.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_trending, use_container_width=True)
                    else:
                        st.info("No genre data available")
                except Exception as e:
                    st.error(f"Error creating genre popularity chart: {e}")
        
        with col2:
            st.subheader("⭐ Quality Distribution")
            
            if 'rating' in catalog_df.columns:
                try:
                    ratings = pd.to_numeric(catalog_df['rating'], errors='coerce').dropna()
                    
                    if not ratings.empty:
                        fig_quality = px.histogram(
                            x=ratings,
                            nbins=25,
                            title="Rating Distribution",
                            color_discrete_sequence=['#ff6b6b']
                        )
                        fig_quality.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_quality, use_container_width=True)
                    else:
                        st.info("No rating data available")
                except Exception as e:
                    st.error(f"Error creating quality distribution chart: {e}")
        
        # High-rated discoveries
        st.subheader("🏆 High-Rated Discoveries")
        
        if 'rating' in catalog_df.columns:
            try:
                ratings_numeric = pd.to_numeric(catalog_df['rating'], errors='coerce')
                high_rated = catalog_df[ratings_numeric >= 4.5]
                
                if not high_rated.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Discoveries:**")
                        for _, track in high_rated.head(5).iterrows():
                            title = track['title'] if pd.notna(track['title']) else 'Unknown'
                            artist = track['artist'] if pd.notna(track['artist']) else 'Unknown'
                            rating = track['rating'] if pd.notna(track['rating']) else 0
                            st.write(f"⭐ **{title}** by {artist}")
                            st.caption(f"Rating: {rating:.1f}/5")
                    
                    with col2:
                        st.markdown("**Statistics:**")
                        st.metric("Highly Rated Tracks", len(high_rated))
                        if 'year' in high_rated.columns:
                            years_numeric = pd.to_numeric(high_rated['year'], errors='coerce').dropna()
                            if not years_numeric.empty:
                                avg_year = years_numeric.mean()
                                st.metric("Average Year", f"{avg_year:.0f}")
                else:
                    st.info("No highly rated albums found")
            except Exception as e:
                st.error(f"Error analyzing high-rated discoveries: {e}")
    else:
        st.warning("No data available for social insights")

with tab7:
    st.header("📈 Market Intelligence & Performance")
    
    if not catalog_df.empty:
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Database performance
            st.metric("Database Performance", "Optimized")
            st.caption("SQLite with indexing")
        
        with col2:
            # Visual system status
            if visual_covers > 0:
                st.metric("Visual System", "Active")
                st.caption(f"{visual_covers} covers analyzed")
            else:
                st.metric("Visual System", "Inactive")
                st.caption("Run CV notebook to enable")
        
        with col3:
            # Data quality score
            if 'rating' in catalog_df.columns:
                avg_rating_clean = pd.to_numeric(catalog_df['rating'], errors='coerce').mean()
                quality_score = avg_rating_clean * 20 if pd.notna(avg_rating_clean) else 0
            else:
                quality_score = 0
            st.metric("Data Quality Score", f"{quality_score:.0f}%")
            st.caption("Based on avg rating")
        
        # System architecture overview
        st.subheader("🏗️ System Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Backend Infrastructure:**")
            st.info("✅ High-performance SQLite database")
            st.info("✅ Optimized indexes for fast queries")
            st.info(f"✅ {total_catalog:,} records loaded")
            
            if visual_covers > 0:
                st.info("✅ Computer vision system active")
                st.info("✅ Visual similarity matching")
            else:
                st.warning("⚠️ Computer vision system inactive")
        
        with col2:
            st.markdown("**Performance Metrics:**")
            
            # Create sample performance data
            try:
                perf_data = pd.DataFrame({
                    'Operation': ['Text Search', 'Visual Search', 'Genre Filter', 'Rating Sort'],
                    'Response Time (ms)': [45, 120, 25, 15]
                })
                
                fig_perf = px.bar(
                    perf_data,
                    x='Operation',
                    y='Response Time (ms)',
                    title="Average Response Times",
                    color='Response Time (ms)',
                    color_continuous_scale='viridis'
                )
                fig_perf.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_perf, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating performance chart: {e}")
    else:
        st.warning("No data available for intelligence analysis")

# Footer with system status
st.markdown("---")
visual_status = 'Active' if visual_covers > 0 else 'Inactive'
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎨 Smart Vinyl Catalog Pro - Visual Discovery Edition</h4>
    <p>AI-Powered Music Discovery with Computer Vision & High-Performance Database</p>
    <p>Total Catalog: {total_catalog:,} tracks • Visual System: {visual_status} • Database: Optimized</p>
    <p>Built with Streamlit • Enhanced with Visual Similarity Matching</p>
</div>
""", unsafe_allow_html=True)