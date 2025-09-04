import requests
import sqlite3
import json
import time
import os
from pathlib import Path
from PIL import Image
import hashlib
from urllib.parse import urlparse
from contextlib import contextmanager
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import configparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discogs_cover_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CoverImage:
    """Data class for cover image information"""
    release_id: str
    discogs_id: str
    image_url: str
    image_type: str  # 'primary', 'secondary'
    width: int
    height: int
    local_path: Optional[str] = None
    file_size: Optional[int] = None
    download_status: str = 'pending'  # 'pending', 'downloaded', 'failed'

class DiscogsAPIClient:
    """Enhanced Discogs API client with rate limiting and error handling"""
    
    def __init__(self, user_token: str):
        self.user_token = user_token
        self.base_url = "https://api.discogs.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VinylCatalogPro/1.0',
            'Authorization': f'Discogs token={user_token}'
        })
        
        # Rate limiting (60 requests per minute for authenticated users)
        self.request_queue = Queue()
        self.rate_limit_delay = 1.1  # Slightly over 1 second to stay safe
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_release(self, discogs_id: str) -> Optional[Dict]:
        """Get release information from Discogs API"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/releases/{discogs_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Release {discogs_id} not found")
                return None
            elif response.status_code == 429:
                # Rate limited - wait longer
                logger.warning("Rate limited - waiting 60 seconds")
                time.sleep(60)
                return self.get_release(discogs_id)
            else:
                logger.error(f"API error for release {discogs_id}: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for release {discogs_id}: {e}")
            return None
    
    def get_master_release(self, master_id: str) -> Optional[Dict]:
        """Get master release information from Discogs API"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/masters/{master_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error for master {master_id}: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for master {master_id}: {e}")
            return None

class ImageDownloader:
    """Async image downloader with optimization"""
    
    def __init__(self, download_dir: Path, max_concurrent: int = 5):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def download_image(self, session: aiohttp.ClientSession, cover: CoverImage) -> CoverImage:
        """Download a single image with error handling"""
        async with self.semaphore:
            try:
                # Generate filename
                url_hash = hashlib.md5(cover.image_url.encode()).hexdigest()
                extension = self._get_extension(cover.image_url)
                filename = f"{cover.release_id}_{cover.discogs_id}_{url_hash}{extension}"
                local_path = self.download_dir / filename
                
                # Skip if already exists
                if local_path.exists():
                    cover.local_path = str(local_path)
                    cover.file_size = local_path.stat().st_size
                    cover.download_status = 'downloaded'
                    return cover
                
                # Download image
                async with session.get(cover.image_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Save original
                        async with aiofiles.open(local_path, 'wb') as f:
                            await f.write(content)
                        
                        # Optimize image
                        optimized_path = await self._optimize_image(local_path)
                        
                        cover.local_path = str(optimized_path)
                        cover.file_size = optimized_path.stat().st_size
                        cover.download_status = 'downloaded'
                        
                        logger.info(f"Downloaded cover for {cover.release_id}")
                        return cover
                    else:
                        logger.error(f"Failed to download {cover.image_url}: {response.status}")
                        cover.download_status = 'failed'
                        return cover
                        
            except Exception as e:
                logger.error(f"Error downloading {cover.image_url}: {e}")
                cover.download_status = 'failed'
                return cover
    
    async def _optimize_image(self, image_path: Path) -> Path:
        """Optimize image size and quality"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                optimized_path = await loop.run_in_executor(
                    executor, self._optimize_image_sync, image_path
                )
            return optimized_path
        except Exception as e:
            logger.error(f"Error optimizing image {image_path}: {e}")
            return image_path
    
    def _optimize_image_sync(self, image_path: Path) -> Path:
        """Synchronous image optimization"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large (max 800x800 for covers)
                if img.width > 800 or img.height > 800:
                    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                # Save optimized version
                optimized_path = image_path.with_suffix('.jpg')
                img.save(optimized_path, 'JPEG', quality=85, optimize=True)
                
                # Remove original if different
                if optimized_path != image_path:
                    image_path.unlink()
                
                return optimized_path
                
        except Exception as e:
            logger.error(f"Error in image optimization: {e}")
            return image_path
    
    def _get_extension(self, url: str) -> str:
        """Extract file extension from URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            return '.jpg'
        elif path.endswith('.png'):
            return '.png'
        elif path.endswith('.gif'):
            return '.gif'
        else:
            return '.jpg'  # Default to jpg

@contextmanager
def get_db_connection(db_path: Path):
    """Database connection context manager"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class CoverDatabase:
    """Database manager for album covers"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_cover_tables()
    
    def _init_cover_tables(self):
        """Initialize album cover tables if they don't exist"""
        with get_db_connection(self.db_path) as conn:
            # Enhanced album_covers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS album_covers (
                    cover_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    discogs_id TEXT,
                    image_url TEXT NOT NULL,
                    image_type TEXT DEFAULT 'primary',
                    width INTEGER,
                    height INTEGER,
                    local_path TEXT,
                    file_size INTEGER,
                    download_status TEXT DEFAULT 'pending',
                    download_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (release_id) REFERENCES releases (release_id)
                )
            """)
            
            # Cover processing status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cover_processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id TEXT,
                    discogs_id TEXT,
                    status TEXT, -- 'pending', 'processing', 'completed', 'failed'
                    error_message TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_covers_release_id ON album_covers(release_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_covers_discogs_id ON album_covers(discogs_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_covers_status ON album_covers(download_status)")
            
            conn.commit()
    
    def get_releases_without_covers(self, limit: int = 100) -> List[Tuple[str, str]]:
        """Get releases that don't have covers yet"""
        with get_db_connection(self.db_path) as conn:
            query = """
                SELECT r.release_id, r.discogs_id 
                FROM releases r
                LEFT JOIN album_covers ac ON r.release_id = ac.release_id
                WHERE ac.release_id IS NULL 
                  AND r.discogs_id IS NOT NULL 
                  AND r.discogs_id != ''
                ORDER BY r.popularity_score DESC
                LIMIT ?
            """
            return [(row[0], row[1]) for row in conn.execute(query, (limit,))]
    
    def save_cover(self, cover: CoverImage):
        """Save cover information to database"""
        with get_db_connection(self.db_path) as conn:
            cover_id = f"{cover.release_id}_{hashlib.md5(cover.image_url.encode()).hexdigest()[:8]}"
            
            conn.execute("""
                INSERT OR REPLACE INTO album_covers 
                (cover_id, release_id, discogs_id, image_url, image_type, 
                 width, height, local_path, file_size, download_status, download_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                cover_id, cover.release_id, cover.discogs_id, cover.image_url,
                cover.image_type, cover.width, cover.height, cover.local_path,
                cover.file_size, cover.download_status
            ))
            conn.commit()
    
    def log_processing_status(self, release_id: str, discogs_id: str, status: str, error: str = None):
        """Log processing status"""
        with get_db_connection(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cover_processing_log 
                (release_id, discogs_id, status, error_message)
                VALUES (?, ?, ?, ?)
            """, (release_id, discogs_id, status, error))
            conn.commit()
    
    def get_cover_stats(self) -> Dict:
        """Get statistics about cover downloads"""
        with get_db_connection(self.db_path) as conn:
            stats = {}
            
            # Total covers
            result = conn.execute("SELECT COUNT(*) FROM album_covers").fetchone()
            stats['total_covers'] = result[0]
            
            # By status
            status_query = """
                SELECT download_status, COUNT(*) 
                FROM album_covers 
                GROUP BY download_status
            """
            stats['by_status'] = dict(conn.execute(status_query).fetchall())
            
            # Total file size
            size_query = "SELECT SUM(file_size) FROM album_covers WHERE file_size IS NOT NULL"
            result = conn.execute(size_query).fetchone()
            stats['total_size_mb'] = (result[0] or 0) / (1024 * 1024)
            
            return stats

class DiscogsCoverFetcher:
    """Main class for fetching album covers from Discogs"""
    
    def __init__(self, db_path: Path, download_dir: Path, user_token: str):
        self.db = CoverDatabase(db_path)
        self.api = DiscogsAPIClient(user_token)
        self.downloader = ImageDownloader(download_dir)
        
    def extract_images_from_release(self, release_data: Dict, release_id: str, discogs_id: str) -> List[CoverImage]:
        """Extract image information from Discogs release data"""
        covers = []
        
        if 'images' in release_data:
            for i, image in enumerate(release_data['images']):
                # Prefer larger images
                if image.get('width', 0) >= 300:  # Only get decent resolution images
                    cover = CoverImage(
                        release_id=release_id,
                        discogs_id=discogs_id,
                        image_url=image['uri'],
                        image_type='primary' if i == 0 else 'secondary',
                        width=image.get('width', 0),
                        height=image.get('height', 0)
                    )
                    covers.append(cover)
        
        return covers
    
    async def process_batch(self, releases: List[Tuple[str, str]], batch_size: int = 20):
        """Process a batch of releases"""
        logger.info(f"Processing batch of {len(releases)} releases")
        
        # Get release data from API
        covers_to_download = []
        
        for release_id, discogs_id in releases:
            try:
                self.db.log_processing_status(release_id, discogs_id, 'processing')
                
                # Get release data
                release_data = self.api.get_release(discogs_id)
                
                if release_data:
                    # Extract cover images
                    covers = self.extract_images_from_release(release_data, release_id, discogs_id)
                    covers_to_download.extend(covers)
                    
                    if covers:
                        self.db.log_processing_status(release_id, discogs_id, 'found_images')
                    else:
                        self.db.log_processing_status(release_id, discogs_id, 'no_images')
                else:
                    self.db.log_processing_status(release_id, discogs_id, 'api_failed')
                    
            except Exception as e:
                logger.error(f"Error processing {release_id}: {e}")
                self.db.log_processing_status(release_id, discogs_id, 'failed', str(e))
        
        # Download images
        if covers_to_download:
            await self._download_covers(covers_to_download)
    
    async def _download_covers(self, covers: List[CoverImage]):
        """Download all covers in the list"""
        logger.info(f"Downloading {len(covers)} covers")
        
        # Create download batches
        async with aiohttp.ClientSession() as session:
            tasks = []
            for cover in covers:
                task = self.downloader.download_image(session, cover)
                tasks.append(task)
            
            # Execute downloads with progress tracking
            completed_covers = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                cover = await task
                completed_covers.append(cover)
                
                # Save to database
                self.db.save_cover(cover)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Downloaded {i + 1}/{len(covers)} covers")
        
        # Log final results
        successful = sum(1 for c in completed_covers if c.download_status == 'downloaded')
        logger.info(f"Successfully downloaded {successful}/{len(covers)} covers")
    
    async def fetch_all_covers(self, batch_size: int = 50, max_batches: int = None):
        """Fetch covers for all releases without covers"""
        logger.info("Starting cover fetching process")
        
        batch_count = 0
        while True:
            # Get next batch of releases
            releases = self.db.get_releases_without_covers(batch_size)
            
            if not releases:
                logger.info("No more releases to process")
                break
            
            if max_batches and batch_count >= max_batches:
                logger.info(f"Reached maximum batch limit: {max_batches}")
                break
            
            # Process batch
            await self.process_batch(releases, batch_size)
            
            batch_count += 1
            logger.info(f"Completed batch {batch_count}")
            
            # Rate limiting between batches
            await asyncio.sleep(2)
        
        # Print final statistics
        stats = self.db.get_cover_stats()
        logger.info(f"Cover fetching complete. Stats: {stats}")

def load_config() -> Dict[str, str]:
    """Load configuration from file"""
    config = configparser.ConfigParser()
    config_path = Path('config.ini')
    
    if config_path.exists():
        config.read(config_path)
        return dict(config['discogs'])
    else:
        # Create sample config
        config['discogs'] = {
            'user_token': 'YOUR_DISCOGS_TOKEN_HERE',
            'download_directory': 'data/covers',
            'database_path': 'data/database/vinyl_catalog.db'
        }
        
        with open(config_path, 'w') as f:
            config.write(f)
        
        logger.warning(f"Created sample config at {config_path}. Please update with your Discogs token.")
        return dict(config['discogs'])

async def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    
    user_token = config.get('user_token')
    if not user_token or user_token == 'YOUR_DISCOGS_TOKEN_HERE':
        logger.error("Please set your Discogs user token in config.ini")
        return
    
    # Setup paths
    db_path = Path(config.get('database_path', 'data/database/vinyl_catalog.db'))
    download_dir = Path(config.get('download_directory', 'data/covers'))
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return
    
    # Initialize fetcher
    fetcher = DiscogsCoverFetcher(db_path, download_dir, user_token)
    
    # Start fetching
    await fetcher.fetch_all_covers(batch_size=20, max_batches=10)  # Limit for initial run

if __name__ == "__main__":
    # Run the cover fetching process
    asyncio.run(main())