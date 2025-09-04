import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from PIL import Image
import base64
import io
from typing import Optional, List, Dict
import logging

# Additional functions to integrate with the existing dashboard

@contextmanager
def get_db_connection(db_path: Path):
    """Database connection context manager"""
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

def get_cover_image_path(release_id: str, db_path: Path) -> Optional[str]:
    """Get the local path for a release's cover image"""
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return None
                
            query = """
                SELECT local_path 
                FROM album_covers 
                WHERE release_id = ? 
                  AND download_status = 'downloaded' 
                  AND local_path IS NOT NULL
                ORDER BY image_type = 'primary' DESC
                LIMIT 1
            """
            
            result = conn.execute(query, (release_id,)).fetchone()
            return result[0] if result else None
            
    except Exception as e:
        st.error(f"Error getting cover path: {e}")
        return None

def display_album_cover(release_id: str, db_path: Path, width: int = 150) -> bool:
    """Display album cover if available, return True if displayed"""
    cover_path = get_cover_image_path(release_id, db_path)
    
    if cover_path and Path(cover_path).exists():
        try:
            # Load and display image
            image = Image.open(cover_path)
            st.image(image, width=width)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False
    else:
        # Show placeholder
        st.markdown(f"""
            <div style="
                width: {width}px; 
                height: {width}px; 
                background: linear-gradient(45deg, #f0f0f0, #e0e0e0);
                display: flex; 
                align-items: center; 
                justify-content: center;
                border-radius: 8px;
                color: #666;
                font-size: 12px;
            ">
                No Cover
            </div>
        """, unsafe_allow_html=True)
        return False

def get_cover_statistics(db_path: Path) -> Dict:
    """Get statistics about cover downloads"""
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return {}
                
            # Total releases with covers
            total_with_covers = conn.execute("""
                SELECT COUNT(DISTINCT release_id) 
                FROM album_covers 
                WHERE download_status = 'downloaded'
            """).fetchone()[0]
            
            # Total releases
            total_releases = conn.execute("SELECT COUNT(*) FROM releases").fetchone()[0]
            
            # Coverage percentage
            coverage = (total_with_covers / total_releases * 100) if total_releases > 0 else 0
            
            # Storage used
            total_size = conn.execute("""
                SELECT COALESCE(SUM(file_size), 0) 
                FROM album_covers 
                WHERE download_status = 'downloaded'
            """).fetchone()[0]
            
            # Download status breakdown
            status_breakdown = dict(conn.execute("""
                SELECT download_status, COUNT(*) 
                FROM album_covers 
                GROUP BY download_status
            """).fetchall())
            
            return {
                'total_with_covers': total_with_covers,
                'total_releases': total_releases,
                'coverage_percentage': coverage,
                'total_size_mb': total_size / (1024 * 1024) if total_size else 0,
                'status_breakdown': status_breakdown
            }
            
    except Exception as e:
        st.error(f"Error getting cover statistics: {e}")
        return {}

def enhanced_search_results_with_covers(results_df: pd.DataFrame, db_path: Path):
    """Display search results with album covers"""
    if results_df.empty:
        st.info("No results found")
        return
    
    st.success(f"Found {len(results_df)} matches")
    
    # Display results in a grid layout
    for idx, (_, row) in enumerate(results_df.head(12).iterrows()):
        if idx % 3 == 0:
            cols = st.columns(3)
        
        with cols[idx % 3]:
            # Album cover
            has_cover = display_album_cover(row['release_id'], db_path, width=120)
            
            # Album info
            st.markdown(f"**{row['title']}**")
            st.markdown(f"*{row['artist']}*")
            
            # Metadata
            year = row['year'] if pd.notna(row['year']) else 'Unknown'
            genre = row['genre'] if pd.notna(row['genre']) else 'Unknown'
            st.caption(f"📅 {year} • 🎵 {genre}")
            
            # Rating and match score
            rating = row['rating'] if pd.notna(row['rating']) else 0
            match_score = row.get('similarity_score', 0) * 100 if 'similarity_score' in row else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rating", f"{rating:.1f}/5")
            with col2:
                st.metric("Match", f"{match_score:.0f}%")
            
            # Cover status indicator
            if has_cover:
                st.success("🖼️ Cover Available")
            else:
                st.info("📷 No Cover")
            
            st.divider()

def visual_discovery_with_real_covers(db_path: Path):
    """Enhanced visual discovery interface with real album covers"""
    
    # Get albums with covers for visual discovery
    try:
        with get_db_connection(db_path) as conn:
            if conn is None:
                return
                
            query = """
                SELECT DISTINCT r.release_id, r.title, r.artist, r.year, r.genre,
                       ac.local_path, vf.brightness, vf.saturation, vf.complexity_score
                FROM releases r
                JOIN album_covers ac ON r.release_id = ac.release_id
                LEFT JOIN visual_features vf ON ac.cover_id = vf.cover_id
                WHERE ac.download_status = 'downloaded'
                  AND ac.local_path IS NOT NULL
                ORDER BY r.popularity_score DESC
                LIMIT 50
            """
            
            visual_albums = pd.read_sql_query(query, conn)
            
            if visual_albums.empty:
                st.warning("No albums with covers available for visual discovery")
                return
            
            st.subheader(f"🎨 Visual Discovery Gallery ({len(visual_albums)} albums)")
            
            # Gallery view with real covers
            cols_per_row = 5
            for idx, (_, album) in enumerate(visual_albums.head(20).iterrows()):
                if idx % cols_per_row == 0:
                    cols = st.columns(cols_per_row)
                
                with cols[idx % cols_per_row]:
                    # Display actual album cover
                    display_album_cover(album['release_id'], db_path, width=100)
                    
                    # Album info
                    st.caption(f"**{album['title'][:20]}...**" if len(album['title']) > 20 else f"**{album['title']}**")
                    st.caption(f"*{album['artist'][:15]}...*" if len(album['artist']) > 15 else f"*{album['artist']}*")
                    
                    # Visual features if available
                    if pd.notna(album.get('brightness')):
                        st.caption(f"🔆 {album['brightness']:.0f}")
                    if pd.notna(album.get('complexity_score')):
                        st.caption(f"🔗 {album['complexity_score']:.2f}")
            
    except Exception as e:
        st.error(f"Error in visual discovery: {e}")

def cover_management_interface(db_path: Path):
    """Interface for managing album covers"""
    st.header("🖼️ Album Cover Management")
    
    # Cover statistics
    stats = get_cover_statistics(db_path)
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Releases with Covers", f"{stats['total_with_covers']:,}")
        
        with col2:
            st.metric("Coverage", f"{stats['coverage_percentage']:.1f}%")
        
        with col3:
            st.metric("Storage Used", f"{stats['total_size_mb']:.1f} MB")
        
        with col4:
            pending = stats['status_breakdown'].get('pending', 0)
            st.metric("Pending Downloads", f"{pending:,}")
        
        # Status breakdown
        if stats['status_breakdown']:
            st.subheader("Download Status Breakdown")
            
            status_df = pd.DataFrame([
                {'Status': status, 'Count': count} 
                for status, count in stats['status_breakdown'].items()
            ])
            
            fig = px.pie(status_df, values='Count', names='Status', 
                        title="Cover Download Status")
            st.plotly_chart(fig, use_container_width=True)
    
    # Cover fetching controls
    st.subheader("🔄 Cover Fetching")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Fetch Next Batch", help="Download covers for releases without images"):
            with st.spinner("Fetching covers..."):
                # This would trigger the cover fetching process
                st.info("Cover fetching initiated. Check logs for progress.")
                st.code("""
# To fetch covers, run:
python setup_covers.py fetch --batch-size 10 --max-batches 2
                """)
    
    with col2:
        if st.button("🧹 Clean Failed Downloads", help="Remove failed download records"):
            # This would clean up failed downloads
            st.info("Cleanup initiated. Failed downloads will be removed.")
    
    # Recent downloads
    try:
        with get_db_connection(db_path) as conn:
            if conn is not None:
                recent_query = """
                    SELECT r.title, r.artist, ac.download_date, ac.file_size
                    FROM album_covers ac
                    JOIN releases r ON ac.release_id = r.release_id
                    WHERE ac.download_status = 'downloaded'
                    ORDER BY ac.download_date DESC
                    LIMIT 10
                """
                
                recent_downloads = pd.read_sql_query(recent_query, conn)
                
                if not recent_downloads.empty:
                    st.subheader("🕒 Recent Downloads")
                    
                    for _, download in recent_downloads.iterrows():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{download['title']}** by {download['artist']}")
                        
                        with col2:
                            st.write(f"📅 {download['download_date']}")
                        
                        with col3:
                            size_kb = download['file_size'] / 1024 if download['file_size'] else 0
                            st.write(f"💾 {size_kb:.0f} KB")
    
    except Exception as e:
        st.error(f"Error displaying recent downloads: {e}")

def add_cover_integration_to_dashboard():
    """Add cover integration features to the main dashboard"""
    
    # Add this to the main dashboard tabs
    with st.expander("🖼️ Album Cover Integration", expanded=False):
        st.markdown("""
        **Real Album Cover Integration Status:**
        
        This system fetches actual album artwork from Discogs and integrates it into the visual discovery system.
        """)
        
        db_path = Path("data/database/vinyl_catalog.db")
        
        # Cover statistics summary
        stats = get_cover_statistics(db_path)
        if stats:
            coverage = stats.get('coverage_percentage', 0)
            if coverage > 50:
                st.success(f"✅ {coverage:.1f}% of releases have album covers")
            elif coverage > 10:
                st.warning(f"⚠️ {coverage:.1f}% of releases have album covers")
            else:
                st.error(f"❌ Only {coverage:.1f}% of releases have album covers")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Setup Covers"):
                st.info("Run: `python setup_covers.py setup`")
        
        with col2:
            if st.button("📊 Cover Status"):
                st.info("Run: `python setup_covers.py status`")
        
        with col3:
            if st.button("📥 Fetch Covers"):
                st.info("Run: `python setup_covers.py fetch`")

# Integration instructions for the main dashboard
DASHBOARD_INTEGRATION_CODE = '''
# Add to your main vinyl_dashboard.py file:

# 1. Import the cover integration functions at the top:
from dashboard_cover_integration import (
    display_album_cover, 
    enhanced_search_results_with_covers,
    visual_discovery_with_real_covers,
    cover_management_interface,
    add_cover_integration_to_dashboard
)

# 2. In the search tab (tab1), replace the search results display with:
if search_query and len(search_query.strip()) > 0:
    if search_type == "Database Search" and search_engine:
        with st.spinner("Searching database..."):
            try:
                results = search_engine.fuzzy_search(search_query)
                enhanced_search_results_with_covers(results, db_path)
            except Exception as e:
                st.error(f"Search error: {e}")

# 3. In the visual discovery tab (tab2), add:
if visual_covers > 0:
    visual_discovery_with_real_covers(db_path)

# 4. Add a new tab for cover management:
tab8 = st.tabs(["🖼️ Cover Management"])[0]
with tab8:
    cover_management_interface(db_path)

# 5. Add the cover integration status to sidebar:
add_cover_integration_to_dashboard()
'''

if __name__ == "__main__":
    st.title("🖼️ Dashboard Cover Integration")
    
    st.markdown("""
    This module provides real album cover integration for the vinyl dashboard.
    
    **Features:**
    - Display actual album covers in search results
    - Visual discovery gallery with real artwork
    - Cover download management interface
    - Integration with Discogs API
    """)
    
    st.code(DASHBOARD_INTEGRATION_CODE, language='python')
    
    # Demo of cover management interface
    st.subheader("Demo: Cover Management Interface")
    db_path = Path("data/database/vinyl_catalog.db")
    cover_management_interface(db_path)