"""Collection analysis and recommendation logic using traditional methods."""
import pandas as pd
from typing import List, Dict
import numpy as np

class CollectionAnalyzer:
    def __init__(self, client):
        self.client = client
    
    def get_collection_data(self):
        """Fetch complete collection data."""
        query = """
        SELECT 
            pc.*,
            dr.title, dr.artist, dr.year, dr.genre, dr.style, dr.label,
            ar.rating as critic_rating, ar.review_text
        FROM `vinyl_catalog.personal_collection` pc
        JOIN `vinyl_catalog.discogs_releases` dr ON pc.release_id = dr.release_id
        JOIN `vinyl_catalog.album_reviews` ar ON pc.release_id = ar.album_id
        """
        return self.client.query(query).to_dataframe()
    
    def analyze_listening_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze listening patterns and preferences."""
        patterns = {
            'most_played': data.loc[data['times_played'].idxmax()],
            'highest_rated': data.loc[data['personal_rating'].idxmax()],
            'best_value': data.loc[(data['personal_rating'] / data['purchase_price']).idxmax()],
            'avg_rating_by_era': data.groupby(data['year'] // 10 * 10)['personal_rating'].mean().to_dict(),
            'preferred_labels': data.groupby('label')['personal_rating'].mean().sort_values(ascending=False).to_dict(),
            'price_vs_satisfaction': np.corrcoef(data['purchase_price'], data['personal_rating'])[0,1]
        }
        return patterns
    
    def generate_collection_gaps(self, data: pd.DataFrame) -> List[str]:
        """Identify potential gaps in collection based on existing preferences."""
        gaps = []
        
        # Era gaps
        years = data['year'].tolist()
        if not any(y < 1950 for y in years):
            gaps.append("Pre-1950s jazz (swing era, early bebop)")
        if not any(1970 <= y < 1980 for y in years):
            gaps.append("1970s fusion and post-bop")
        
        # Style gaps based on high-rated albums
        high_rated = data[data['personal_rating'] >= 9]
        if len(high_rated[high_rated['style'].str.contains('Cool', na=False)]) > 0:
            gaps.append("West Coast Cool Jazz (Chet Baker, Gerry Mulligan)")
        
        if 'Blue Note' in data['label'].values:
            gaps.append("More Blue Note classics (Lee Morgan, Art Blakey)")
        
        return gaps
    
    def recommend_similar_albums(self, data: pd.DataFrame, target_album: str) -> List[str]:
        """Recommend albums similar to a high-rated album in collection."""
        target = data[data['title'].str.contains(target_album, case=False, na=False)]
        if target.empty:
            return []
        
        target_info = target.iloc[0]
        recommendations = []
        
        # Same artist recommendations
        if target_info['artist'] == 'John Coltrane':
            recommendations.extend([
                "Ballads (John Coltrane)",
                "Duke Ellington & John Coltrane",
                "My Favorite Things (John Coltrane)"
            ])
        elif target_info['artist'] == 'Miles Davis':
            recommendations.extend([
                "Sketches of Spain (Miles Davis)",
                "Birth of the Cool (Miles Davis)",
                "Round About Midnight (Miles Davis)"
            ])
        
        # Same label recommendations
        if target_info['label'] == 'Blue Note':
            recommendations.extend([
                "The Sidewinder (Lee Morgan)",
                "Song for My Father (Horace Silver)",
                "Moanin' (Art Blakey & The Jazz Messengers)"
            ])
        
        return recommendations[:5]  # Top 5 recommendations