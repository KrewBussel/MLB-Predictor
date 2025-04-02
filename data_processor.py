import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from datetime import datetime

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
        self.teams_dir = os.path.join(self.base_dir, 'teams')
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for team-based organization."""
        os.makedirs(self.teams_dir, exist_ok=True)
        
        # Create directories for each team
        for division in self.MLB_TEAMS.values():
            for team in division:
                team_dir = os.path.join(self.teams_dir, self._sanitize_team_name(team))
                batters_dir = os.path.join(team_dir, 'batters')
                pitchers_dir = os.path.join(team_dir, 'pitchers')
                os.makedirs(batters_dir, exist_ok=True)
                os.makedirs(pitchers_dir, exist_ok=True)
    
    def _sanitize_team_name(self, team_name: str) -> str:
        """Convert team name to filesystem-friendly format."""
        return team_name.lower().replace(' ', '_')
    
    def get_all_teams(self) -> List[str]:
        """Get list of all MLB teams."""
        return [team for division in self.MLB_TEAMS.values() for team in division]
    
    def get_team_batters(self, team: str) -> List[str]:
        """Get list of available batters for a specific team."""
        team_batters_dir = os.path.join(self.teams_dir, self._sanitize_team_name(team), 'batters')
        if not os.path.exists(team_batters_dir):
            return []
        return [f.replace('.csv', '') for f in os.listdir(team_batters_dir) if f.endswith('.csv')]
    
    def get_team_pitchers(self, team: str) -> List[str]:
        """Get list of available pitchers for a specific team."""
        team_pitchers_dir = os.path.join(self.teams_dir, self._sanitize_team_name(team), 'pitchers')
        if not os.path.exists(team_pitchers_dir):
            return []
        return [f.replace('.csv', '') for f in os.listdir(team_pitchers_dir) if f.endswith('.csv')]
    
    def load_batter_data(self, team: str, batter_name: str) -> pd.DataFrame:
        """Load batter statistics from team-specific CSV file."""
        file_path = os.path.join(self.teams_dir, self._sanitize_team_name(team), 'batters', f"{batter_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Batter data file not found: {file_path}")
        df = pd.read_csv(file_path)
        
        # Handle date conversion with error handling
        try:
            # Try to parse dates, adding current year if missing
            def parse_date(date_str):
                try:
                    # First try standard format
                    return pd.to_datetime(date_str)
                except:
                    try:
                        # Try with current year if only month day is provided
                        current_year = datetime.now().year
                        if len(date_str.split('-')) < 3 and len(date_str.split('/')) < 3:
                            return pd.to_datetime(f"{current_year} {date_str}")
                        return pd.NaT
                    except:
                        return pd.NaT  # Not a Time for invalid dates
            
            df['Date'] = df['Date'].apply(parse_date)
            # Remove rows with invalid dates
            df = df[~df['Date'].isna()]
        except Exception as e:
            print(f"Warning: Could not parse dates for {batter_name}: {e}")
            # Create a default date if parsing fails
            df['Date'] = pd.to_datetime('2024-01-01')
        
        return df.sort_values('Date', ascending=False)
    
    def load_pitcher_data(self, team: str, pitcher_name: str) -> pd.DataFrame:
        """Load pitcher statistics from team-specific CSV file."""
        file_path = os.path.join(self.teams_dir, self._sanitize_team_name(team), 'pitchers', f"{pitcher_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pitcher data file not found: {file_path}")
        df = pd.read_csv(file_path)
        
        # Handle date conversion with error handling
        try:
            # Try to parse dates, adding current year if missing
            def parse_date(date_str):
                try:
                    # First try standard format
                    return pd.to_datetime(date_str)
                except:
                    try:
                        # Try with current year if only month day is provided
                        current_year = datetime.now().year
                        if len(str(date_str).split('-')) < 3 and len(str(date_str).split('/')) < 3:
                            return pd.to_datetime(f"{current_year} {date_str}")
                        return pd.NaT
                    except:
                        return pd.NaT  # Not a Time for invalid dates
            
            df['Date'] = df['Date'].apply(parse_date)
            # Remove rows with invalid dates
            df = df[~df['Date'].isna()]
        except Exception as e:
            print(f"Warning: Could not parse dates for {pitcher_name}: {e}")
            # Create a default date if parsing fails
            df['Date'] = pd.to_datetime('2024-01-01')
        
        return df.sort_values('Date', ascending=False)
    
    def prepare_matchup_features(self, batter_team: str, batter_name: str, pitcher_team: str, pitcher_name: str) -> np.ndarray:
        """Prepare features for a specific batter-pitcher matchup."""
        try:
            # Load data
            batter_df = self.load_batter_data(batter_team, batter_name)
            pitcher_df = self.load_pitcher_data(pitcher_team, pitcher_name)
            
            # Ensure required columns exist in batter dataframe
            required_batter_cols = ['H', 'SO', 'BA', 'OBP', 'SLG', 'PA', 'OPS', 'WPA', 'RE24', 'DFS(DK)']
            for col in required_batter_cols:
                if col not in batter_df.columns:
                    print(f"Warning: Column {col} missing from batter data. Using zeros.")
                    batter_df[col] = 0
            
            # Ensure required columns exist in pitcher dataframe
            required_pitcher_cols = ['ERA', 'SO', 'HR', 'BB', 'H', 'IP', 'FIP', 'WPA', 'RE24', 'DFS(DK)']
            for col in required_pitcher_cols:
                if col not in pitcher_df.columns:
                    print(f"Warning: Column {col} missing from pitcher data. Using zeros.")
                    pitcher_df[col] = 0
            
            # Calculate features
            batter_features = self._calculate_batter_features(batter_df, pitcher_team)
            pitcher_features = self._calculate_pitcher_features(pitcher_df, batter_team)
            matchup_features = self._calculate_matchup_features(batter_df, pitcher_df)
            
            # Combine features
            features = np.concatenate([batter_features, pitcher_features, matchup_features])
            return features
        
        except Exception as e:
            print(f"Error preparing matchup features: {e}")
            # Return default features array of appropriate length (33 = 16 batter + 16 pitcher + 4 matchup)
            return np.zeros(36)
    
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