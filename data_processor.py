import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

class DataProcessor:
    # MLB Teams dictionary for standardization
    MLB_TEAMS = {
        'AL_EAST': ['Baltimore Orioles', 'Boston Red Sox', 'New York Yankees', 'Tampa Bay Rays', 'Toronto Blue Jays'],
        'AL_CENTRAL': ['Chicago White Sox', 'Cleveland Guardians', 'Detroit Tigers', 'Kansas City Royals', 'Minnesota Twins'],
        'AL_WEST': ['Houston Astros', 'Los Angeles Angels', 'Oakland Athletics', 'Seattle Mariners', 'Texas Rangers'],
        'NL_EAST': ['Atlanta Braves', 'Miami Marlins', 'New York Mets', 'Philadelphia Phillies', 'Washington Nationals'],
        'NL_CENTRAL': ['Chicago Cubs', 'Cincinnati Reds', 'Milwaukee Brewers', 'Pittsburgh Pirates', 'St. Louis Cardinals'],
        'NL_WEST': ['Arizona Diamondbacks', 'Colorado Rockies', 'Los Angeles Dodgers', 'San Diego Padres', 'San Francisco Giants']
    }

    def __init__(self):
        self.base_dir = 'data'
        self.seasons_dir = os.path.join(self.base_dir, 'seasons')
        self.matchups_dir = os.path.join(self.base_dir, 'matchups')
        self._ensure_directories()
        self.scaler = StandardScaler()
        self.batter_stats = {}
        self.pitcher_stats = {}
        self.matchup_stats = {}
    
    def _ensure_directories(self):
        """Create necessary directories for season-based organization."""
        # Create base directories
        os.makedirs(self.seasons_dir, exist_ok=True)
        os.makedirs(self.matchups_dir, exist_ok=True)
        
        # Create directories for each season
        current_year = datetime.now().year
        for year in range(current_year - 1, current_year + 1):
            season_dir = os.path.join(self.seasons_dir, str(year))
            os.makedirs(os.path.join(season_dir, 'teams'), exist_ok=True)
    
    def get_available_seasons(self) -> List[int]:
        """Get list of available seasons in the data."""
        seasons = []
        for item in os.listdir(self.seasons_dir):
            if os.path.isdir(os.path.join(self.seasons_dir, item)):
                try:
                    seasons.append(int(item))
                except ValueError:
                    continue
        return sorted(seasons)
    
    def get_teams_for_season(self, season: int) -> List[str]:
        """Get list of available teams for a specific season."""
        teams_dir = os.path.join(self.seasons_dir, str(season), 'teams')
        if not os.path.exists(teams_dir):
            return []
        return [d for d in os.listdir(teams_dir) if os.path.isdir(os.path.join(teams_dir, d))]
    
    def get_team_batters(self, season: int, team: str) -> List[str]:
        """Get list of available batters for a specific team and season."""
        batters_dir = os.path.join(self.seasons_dir, str(season), 'teams', team, 'batters')
        if not os.path.exists(batters_dir):
            return []
        return [f.replace('.csv', '') for f in os.listdir(batters_dir) if f.endswith('.csv')]
    
    def get_team_pitchers(self, season: int, team: str) -> List[str]:
        """Get list of available pitchers for a specific team and season."""
        pitchers_dir = os.path.join(self.seasons_dir, str(season), 'teams', team, 'pitchers')
        if not os.path.exists(pitchers_dir):
            return []
        return [f.replace('.csv', '') for f in os.listdir(pitchers_dir) if f.endswith('.csv')]
    
    def _parse_date(self, date_str: str, season: int) -> pd.Timestamp:
        """Parse date string with season information."""
        try:
            # Convert month abbreviation and day to datetime
            return pd.to_datetime(f"{date_str}, {season}", format="%b %d, %Y")
        except:
            # If parsing fails, return None
            return None

    def load_batter_data(self, season: int, team: str, batter_name: str) -> pd.DataFrame:
        """Load batter statistics for a specific season and team."""
        file_path = os.path.join(self.seasons_dir, str(season), 'teams', team, 'batters', f"{batter_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Batter data file not found for {season} {team}: {file_path}")
            
        df = pd.read_csv(file_path)
        # Parse dates with season information
        df['Date'] = df['Date'].apply(lambda x: self._parse_date(x, season))
        return df.sort_values('Date', ascending=False)
    
    def load_pitcher_data(self, season: int, team: str, pitcher_name: str) -> pd.DataFrame:
        """Load pitcher statistics for a specific season and team."""
        file_path = os.path.join(self.seasons_dir, str(season), 'teams', team, 'pitchers', f"{pitcher_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pitcher data file not found for {season} {team}: {file_path}")
            
        df = pd.read_csv(file_path)
        # Parse dates with season information
        df['Date'] = df['Date'].apply(lambda x: self._parse_date(x, season))
        return df.sort_values('Date', ascending=False)
    
    def load_matchup_data(self, batter_name: str, pitcher_name: str) -> pd.DataFrame:
        """Load head-to-head matchup data between a batter and pitcher."""
        matchup_file = os.path.join(self.matchups_dir, f"{batter_name}_vs_{pitcher_name}.csv")
        if os.path.exists(matchup_file):
            return pd.read_csv(matchup_file)
        return None

    def process_batter_data(self, df: pd.DataFrame) -> Dict:
        """Process batter data into features."""
        if df is None or df.empty:
            return None
            
        # Calculate recent performance metrics
        recent_games = df.head(10)  # Last 10 games
        
        features = {
            'recent_games': len(recent_games),
            'recent_hits': recent_games['H'].sum(),
            'recent_hr': recent_games['HR'].sum(),
            'recent_bb': recent_games['BB'].sum(),
            'recent_so': recent_games['SO'].sum(),
            'recent_avg': recent_games['BA'].mean(),
            'recent_ops': recent_games['OPS'].mean(),
            'recent_wpa': recent_games['WPA'].sum(),
            'recent_re24': recent_games['RE24'].sum(),
            'season_games': len(df),
            'season_hits': df['H'].sum(),
            'season_hr': df['HR'].sum(),
            'season_bb': df['BB'].sum(),
            'season_so': df['SO'].sum(),
            'season_avg': df['BA'].mean(),
            'season_ops': df['OPS'].mean(),
            'season_wpa': df['WPA'].sum(),
            'season_re24': df['RE24'].sum()
        }
        return features

    def process_pitcher_data(self, df: pd.DataFrame) -> Dict:
        """Process pitcher data into features."""
        if df is None or df.empty:
            return None
            
        # Calculate recent performance metrics
        recent_games = df.head(5)  # Last 5 games
        
        features = {
            'recent_games': len(recent_games),
            'recent_ip': recent_games['IP'].sum(),
            'recent_hits': recent_games['H'].sum(),
            'recent_er': recent_games['ER'].sum(),
            'recent_bb': recent_games['BB'].sum(),
            'recent_so': recent_games['SO'].sum(),
            'recent_era': recent_games['ERA'].mean(),
            'recent_fip': recent_games['FIP'].mean(),
            'recent_wpa': recent_games['WPA'].sum(),
            'recent_re24': recent_games['RE24'].sum(),
            'season_games': len(df),
            'season_ip': df['IP'].sum(),
            'season_hits': df['H'].sum(),
            'season_er': df['ER'].sum(),
            'season_bb': df['BB'].sum(),
            'season_so': df['SO'].sum(),
            'season_era': df['ERA'].mean(),
            'season_fip': df['FIP'].mean(),
            'season_wpa': df['WPA'].sum(),
            'season_re24': df['RE24'].sum()
        }
        return features

    def process_matchup_data(self, matchup_data: pd.DataFrame) -> Dict:
        """Process matchup data into features."""
        if matchup_data is None or matchup_data.empty:
            return None
            
        features = {
            'matchup_games': matchup_data['G'].iloc[0],
            'matchup_ab': matchup_data['AB'].iloc[0],
            'matchup_hits': matchup_data['H'].iloc[0],
            'matchup_hr': matchup_data['HR'].iloc[0],
            'matchup_bb': matchup_data['BB'].iloc[0],
            'matchup_so': matchup_data['SO'].iloc[0],
            'matchup_avg': matchup_data['AVG'].iloc[0],
            'matchup_ops': matchup_data['OPS'].iloc[0]
        }
        return features

    def prepare_features(self, season: int, batter_team: str, batter_name: str, 
                        pitcher_team: str, pitcher_name: str) -> Dict:
        """Prepare features including matchup data if available."""
        # Load individual stats
        batter_data = self.load_batter_data(season, batter_team, batter_name)
        pitcher_data = self.load_pitcher_data(season, pitcher_team, pitcher_name)
        
        # Load matchup data
        matchup_data = self.load_matchup_data(batter_name, pitcher_name)
        matchup_features = self.process_matchup_data(matchup_data)
        
        # Combine all features
        features = {}
        
        # Add individual stats
        if batter_data is not None:
            features.update(self.process_batter_data(batter_data))
        if pitcher_data is not None:
            features.update(self.process_pitcher_data(pitcher_data))
            
        # Add matchup stats if available
        if matchup_features is not None:
            features.update(matchup_features)
            
        return features

    def prepare_matchup_features(self, season: int, batter_team: str, batter_name: str, 
                               pitcher_team: str, pitcher_name: str) -> np.ndarray:
        """Prepare features for a specific batter-pitcher matchup."""
        try:
            # Get all features
            features = self.prepare_features(season, batter_team, batter_name, pitcher_team, pitcher_name)
            
            # Convert to numpy array
            feature_array = np.array(list(features.values()))
            
            # Reshape for model input
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None

    def _calculate_batter_features(self, df: pd.DataFrame, opposing_team: str) -> np.ndarray:
        """Calculate batter features including performance against specific team."""
        try:
            recent_games = df.head(10)
            vs_team_games = df[df['Opp'] == opposing_team]
            
            # Basic stats (recent performance)
            features = []
            
            # Safe way to calculate means, handling NaN values
            for col in ['H', 'SO', 'BA', 'OBP', 'SLG', 'PA', 'OPS', 'WPA', 'RE24', 'DFS(DK)']:
                try:
                    val = recent_games[col].mean()
                    features.append(0 if pd.isna(val) else val)
                except Exception:
                    features.append(0)
            
            # Performance against specific team
            if not vs_team_games.empty:
                for col in ['H', 'BA', 'OPS']:
                    try:
                        val = vs_team_games[col].mean()
                        features.append(0 if pd.isna(val) else val)
                    except Exception:
                        features.append(0)
            else:
                features.extend([0, 0, 0])  # No history against team
            
            # Trends
            try:
                last_5 = df.head(5)
                prev_5 = df.iloc[5:10]
                
                # Calculate trends safely
                for col in ['H', 'BA', 'OPS']:
                    try:
                        last_mean = last_5[col].mean()
                        prev_mean = prev_5[col].mean()
                        trend = last_mean - prev_mean
                        features.append(0 if pd.isna(trend) else trend)
                    except Exception:
                        features.append(0)
            except Exception:
                features.extend([0, 0, 0])  # Default trend values
            
            return np.array(features)
        
        except Exception as e:
            print(f"Error calculating batter features: {e}")
            return np.zeros(16)  # Return zeros if any error occurs
    
    def _calculate_pitcher_features(self, df: pd.DataFrame, opposing_team: str) -> np.ndarray:
        """Calculate pitcher features including performance against specific team."""
        try:
            recent_games = df.head(10)
            vs_team_games = df[df['Opp'] == opposing_team]
            
            # Basic stats (recent performance)
            features = []
            
            # Safe way to calculate means, handling NaN values
            for col in ['ERA', 'SO', 'HR', 'BB', 'H', 'IP', 'FIP', 'WPA', 'RE24', 'DFS(DK)']:
                try:
                    val = recent_games[col].mean()
                    features.append(0 if pd.isna(val) else val)
                except Exception:
                    features.append(0)
            
            # Performance against specific team
            if not vs_team_games.empty:
                for col in ['ERA', 'SO', 'H']:
                    try:
                        val = vs_team_games[col].mean()
                        features.append(0 if pd.isna(val) else val)
                    except Exception:
                        features.append(0)
            else:
                features.extend([0, 0, 0])  # No history against team
            
            # Trends
            try:
                last_5 = df.head(5)
                prev_5 = df.iloc[5:10]
                
                # Calculate trends safely
                for col in ['ERA', 'SO', 'H']:
                    try:
                        last_mean = last_5[col].mean()
                        prev_mean = prev_5[col].mean()
                        trend = last_mean - prev_mean
                        features.append(0 if pd.isna(trend) else trend)
                    except Exception:
                        features.append(0)
            except Exception:
                features.extend([0, 0, 0])  # Default trend values
            
            return np.array(features)
        
        except Exception as e:
            print(f"Error calculating pitcher features: {e}")
            return np.zeros(16)  # Return zeros if any error occurs
    
    def _calculate_matchup_features(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> np.ndarray:
        """Calculate features specific to the batter-pitcher matchup history."""
        # Find common games (direct matchups)
        batter_games = set(batter_df['Date'].dt.date)
        pitcher_games = set(pitcher_df['Date'].dt.date)
        common_dates = batter_games.intersection(pitcher_games)
        
        if common_dates:
            try:
                matchup_df = pd.merge(
                    batter_df[batter_df['Date'].dt.date.isin(common_dates)],
                    pitcher_df[pitcher_df['Date'].dt.date.isin(common_dates)],
                    on='Date',
                    suffixes=('_batter', '_pitcher')
                )
                
                # Safely access columns with error handling
                features = []
                
                # Hits in matchups
                if 'H_batter' in matchup_df.columns:
                    features.append(matchup_df['H_batter'].mean())
                else:
                    features.append(0)
                
                # Batting average in matchups
                if 'BA_batter' in matchup_df.columns:
                    features.append(matchup_df['BA_batter'].mean())
                elif 'BA' in matchup_df.columns:
                    features.append(matchup_df['BA'].mean())
                else:
                    features.append(0)
                
                # Strikeouts in matchups
                if 'SO_batter' in matchup_df.columns:
                    features.append(matchup_df['SO_batter'].mean())
                elif 'SO' in matchup_df.columns:
                    features.append(matchup_df['SO'].mean())
                else:
                    features.append(0)
                
                # Number of matchups
                features.append(len(common_dates))
                
            except Exception as e:
                print(f"Error calculating matchup features: {e}")
                features = [0, 0, 0, 0]  # Default values on error
        else:
            features = [0, 0, 0, 0]  # No direct matchup history
        
        return np.array(features) 