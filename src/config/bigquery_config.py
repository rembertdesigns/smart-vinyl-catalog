"""BigQuery configuration for Smart Vinyl Catalog."""
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

class BigQueryConfig:
    def __init__(self):
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'smart-vinyl-catalog')
        self.dataset_id = os.getenv('BQ_DATASET', 'vinyl_catalog')
        
    def get_client(self):
        """Initialize BigQuery client with credentials."""
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            return bigquery.Client(credentials=credentials, project=self.project_id)
        else:
            return bigquery.Client(project=self.project_id)
    
    def get_table_id(self, table_name):
        """Get full table ID in format project.dataset.table"""
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

config = BigQueryConfig()