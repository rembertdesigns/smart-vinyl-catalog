"""Environment setup script for Smart Vinyl Catalog."""
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

def check_credentials():
    """Verify Google Cloud credentials are set."""
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set!")
        return False
    
    if not os.path.exists(cred_path):
        print(f"❌ Credentials file not found: {cred_path}")
        return False
    
    print(f"✅ Credentials found: {cred_path}")
    return True

def setup_bigquery_dataset():
    """Create BigQuery dataset if it doesn't exist."""
    sys.path.append('./src')
    from google.cloud import bigquery
    from config.bigquery_config import config
    
    client = config.get_client()
    dataset_id = f"{config.project_id}.{config.dataset_id}"
    
    try:
        client.get_dataset(dataset_id)
        print(f"✅ Dataset {config.dataset_id} already exists")
    except:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"✅ Created dataset {config.dataset_id}")

if __name__ == "__main__":
    print("🎵 Smart Vinyl Catalog - Environment Setup")
    print("=" * 50)
    
    if not check_credentials():
        print("\n🔑 Please set up Google Cloud credentials first.")
        sys.exit(1)
    
    setup_bigquery_dataset()
    print("\n🚀 Environment setup complete!")