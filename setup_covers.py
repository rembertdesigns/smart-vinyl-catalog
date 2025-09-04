#!/usr/bin/env python3
"""
Setup and management script for Discogs cover integration
"""

import asyncio
import argparse
import sys
from pathlib import Path
import sqlite3
from discogs_cover_fetcher import DiscogsCoverFetcher, CoverDatabase, load_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are met"""
    try:
        import aiohttp
        import aiofiles
        from PIL import Image
        import requests
        logger.info("✅ All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.error("Install with: pip install aiohttp aiofiles pillow requests")
        return False

def setup_database_schema(db_path: Path):
    """Setup database schema for covers"""
    logger.info("Setting up database schema for covers...")
    
    try:
        db = CoverDatabase(db_path)
        logger.info("✅ Database schema setup complete")
        return True
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def validate_config():
    """Validate configuration"""
    logger.info("Validating configuration...")
    
    try:
        config = load_config()
        
        # Check token
        token = config.get('user_token')
        if not token or token == 'YOUR_DISCOGS_TOKEN_HERE':
            logger.error("❌ Discogs token not configured")
            logger.error("Get your token from: https://www.discogs.com/settings/developers")
            logger.error("Update config.ini with your token")
            return False
        
        # Check paths
        db_path = Path(config.get('database_path', 'data/database/vinyl_catalog.db'))
        if not db_path.exists():
            logger.error(f"❌ Database not found: {db_path}")
            return False
        
        # Create download directory
        download_dir = Path(config.get('download_directory', 'data/covers'))
        download_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def get_status_report(db_path: Path):
    """Generate status report"""
    logger.info("Generating status report...")
    
    try:
        db = CoverDatabase(db_path)
        stats = db.get_cover_stats()
        
        print("\n" + "="*50)
        print("DISCOGS COVER INTEGRATION STATUS")
        print("="*50)
        
        print(f"Total covers in database: {stats.get('total_covers', 0)}")
        print(f"Total storage used: {stats.get('total_size_mb', 0):.1f} MB")
        
        print("\nBreakdown by status:")
        for status, count in stats.get('by_status', {}).items():
            print(f"  {status}: {count}")
        
        # Get releases without covers
        releases_without_covers = db.get_releases_without_covers(limit=1000)
        print(f"\nReleases without covers: {len(releases_without_covers)}")
        
        if len(releases_without_covers) > 0:
            print(f"Next 5 to process:")
            for i, (release_id, discogs_id) in enumerate(releases_without_covers[:5]):
                print(f"  {i+1}. Release {release_id} (Discogs: {discogs_id})")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error generating status report: {e}")

async def fetch_covers_batch(db_path: Path, download_dir: Path, user_token: str, 
                           batch_size: int = 20, max_batches: int = 5):
    """Fetch a limited batch of covers"""
    logger.info(f"Fetching covers - batch size: {batch_size}, max batches: {max_batches}")
    
    try:
        fetcher = DiscogsCoverFetcher(db_path, download_dir, user_token)
        await fetcher.fetch_all_covers(batch_size=batch_size, max_batches=max_batches)
        logger.info("✅ Cover fetching completed")
    except Exception as e:
        logger.error(f"❌ Cover fetching failed: {e}")

async def fetch_single_release(db_path: Path, download_dir: Path, user_token: str, 
                             release_id: str, discogs_id: str):
    """Fetch cover for a single release"""
    logger.info(f"Fetching cover for release {release_id} (Discogs: {discogs_id})")
    
    try:
        fetcher = DiscogsCoverFetcher(db_path, download_dir, user_token)
        await fetcher.process_batch([(release_id, discogs_id)])
        logger.info("✅ Single release cover fetching completed")
    except Exception as e:
        logger.error(f"❌ Single release fetching failed: {e}")

def clean_failed_downloads(db_path: Path):
    """Clean up failed downloads from database"""
    logger.info("Cleaning up failed downloads...")
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Remove failed download records
            cursor = conn.execute("DELETE FROM album_covers WHERE download_status = 'failed'")
            deleted = cursor.rowcount
            
            # Reset pending downloads older than 24 hours
            cursor = conn.execute("""
                UPDATE album_covers 
                SET download_status = 'pending' 
                WHERE download_status = 'processing' 
                  AND created_at < datetime('now', '-1 day')
            """)
            reset = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"✅ Cleanup complete - deleted {deleted} failed, reset {reset} stale")
            
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Discogs Cover Integration Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup cover integration')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show status report')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch covers')
    fetch_parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    fetch_parser.add_argument('--max-batches', type=int, default=5, help='Maximum batches')
    
    # Single release command
    single_parser = subparsers.add_parser('single', help='Fetch single release')
    single_parser.add_argument('release_id', help='Release ID')
    single_parser.add_argument('discogs_id', help='Discogs ID')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean failed downloads')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load config
    try:
        config = load_config()
        db_path = Path(config.get('database_path', 'data/database/vinyl_catalog.db'))
        download_dir = Path(config.get('download_directory', 'data/covers'))
        user_token = config.get('user_token')
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    if args.command == 'setup':
        if not check_requirements():
            return
        if not validate_config():
            return
        if not setup_database_schema(db_path):
            return
        logger.info("✅ Setup completed successfully!")
        
    elif args.command == 'status':
        get_status_report(db_path)
        
    elif args.command == 'fetch':
        if not validate_config():
            return
        asyncio.run(fetch_covers_batch(
            db_path, download_dir, user_token, 
            args.batch_size, args.max_batches
        ))
        
    elif args.command == 'single':
        if not validate_config():
            return
        asyncio.run(fetch_single_release(
            db_path, download_dir, user_token,
            args.release_id, args.discogs_id
        ))
        
    elif args.command == 'clean':
        clean_failed_downloads(db_path)
        
    elif args.command == 'test':
        if check_requirements() and validate_config():
            logger.info("✅ All systems ready!")
        else:
            logger.error("❌ Configuration issues found")

if __name__ == "__main__":
    main()