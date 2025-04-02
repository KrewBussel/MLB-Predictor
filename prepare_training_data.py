import pandas as pd
import numpy as np
from data_processor import DataProcessor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import os
import joblib
from datetime import datetime

class TrainingDataPreparer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_batter_features(self, df: pd.DataFrame, recent_games: int = 10) -> Dict:
        """Prepare features for a batter based on their recent performance."""
        if df is None or df.empty:
            return None
            
        # Get the most recent games
        recent_data = df.head(recent_games)
        all_data = df
        
        features = {
            'recent_games': len(recent_data),
            'recent_pa': recent_data['PA'].sum(),
            'recent_ab': recent_data['AB'].sum(),
            'recent_hits': recent_data['H'].sum(),
            'recent_2b': recent_data['2B'].sum(),
            'recent_3b': recent_data['3B'].sum(),
            'recent_hr': recent_data['HR'].sum(),
            'recent_rbi': recent_data['RBI'].sum(),
            'recent_bb': recent_data['BB'].sum(),
            'recent_so': recent_data['SO'].sum(),
            'recent_sb': recent_data['SB'].sum(),
            'recent_avg': recent_data['BA'].mean(),
            'recent_obp': recent_data['OBP'].mean(),
            'recent_slg': recent_data['SLG'].mean(),
            'recent_ops': recent_data['OPS'].mean(),
            'recent_wpa': recent_data['WPA'].sum(),
            'recent_re24': recent_data['RE24'].sum(),
            
            'season_games': len(all_data),
            'season_pa': all_data['PA'].sum(),
            'season_ab': all_data['AB'].sum(),
            'season_hits': all_data['H'].sum(),
            'season_2b': all_data['2B'].sum(),
            'season_3b': all_data['3B'].sum(),
            'season_hr': all_data['HR'].sum(),
            'season_rbi': all_data['RBI'].sum(),
            'season_bb': all_data['BB'].sum(),
            'season_so': all_data['SO'].sum(),
            'season_sb': all_data['SB'].sum(),
            'season_avg': all_data['BA'].mean(),
            'season_obp': all_data['OBP'].mean(),
            'season_slg': all_data['SLG'].mean(),
            'season_ops': all_data['OPS'].mean(),
            'season_wpa': all_data['WPA'].sum(),
            'season_re24': all_data['RE24'].sum()
        }
        
        return features
    
    def prepare_training_data(self, seasons: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from specified seasons."""
        if seasons is None:
            seasons = self.data_processor.get_available_seasons()
        
        all_features = []
        all_labels = []
        feature_names = []
        
        print(f"Preparing training data for seasons: {seasons}")
        
        for season in seasons:
            print(f"\nProcessing season {season}...")
            
            # Get all teams
            teams = self.data_processor.get_teams_for_season(season)
            print(f"Found {len(teams)} teams")
            
            for team in teams:
                print(f"\nProcessing team: {team}")
                
                # Get all batters for this team
                batters = self.data_processor.get_team_batters(season, team)
                print(f"Found {len(batters)} batters")
                
                for batter_name in batters:
                    try:
                        # Load batter data
                        batter_data = self.data_processor.load_batter_data(season, team, batter_name)
                        
                        if batter_data is None or batter_data.empty:
                            continue
                        
                        # Process each game as a training example
                        for idx in range(10, len(batter_data)):  # Start after 10 games for sufficient history
                            # Get data up to this game
                            historical_data = batter_data.iloc[idx:]
                            
                            # Get features based on historical data
                            features = self.prepare_batter_features(historical_data)
                            
                            if features:
                                # Get actual hits for this game (label)
                                current_game = batter_data.iloc[idx-1]
                                actual_hits = current_game['H']
                                
                                # Store feature names if not already stored
                                if not feature_names:
                                    feature_names = list(features.keys())
                                    print("\nFeatures being collected:", feature_names)
                                
                                # Add to training data
                                all_features.append(list(features.values()))
                                all_labels.append(actual_hits)
                                
                    except Exception as e:
                        print(f"Error processing batter {batter_name}: {e}")
                        continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Save feature names and scaler
        self.feature_names = feature_names
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/feature_scaler.joblib')
        
        # Save feature names
        with open('models/feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        
        print(f"\nTraining data preparation complete:")
        print(f"- Total examples: {len(X)}")
        print(f"- Number of features: {len(feature_names)}")
        print(f"- Feature statistics:")
        print(f"  - Mean number of hits: {y.mean():.2f}")
        print(f"  - Max number of hits: {y.max()}")
        print(f"  - Min number of hits: {y.min()}")
        print("\nFeatures:", feature_names)
        
        return X_scaled, y
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features in the order they appear in the feature vector."""
        return self.feature_names

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize data preparer
    preparer = TrainingDataPreparer()
    
    # Prepare training data
    try:
        X, y = preparer.prepare_training_data()
        print(f"\nTraining data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Save training data
        np.save('models/training_features.npy', X)
        np.save('models/training_labels.npy', y)
        print("\nSaved training data to models/training_features.npy and models/training_labels.npy")
        
    except Exception as e:
        print(f"Error preparing training data: {e}")

if __name__ == "__main__":
    main() 