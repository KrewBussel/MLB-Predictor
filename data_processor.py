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
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date', ascending=False)
    
    def load_pitcher_data(self, team: str, pitcher_name: str) -> pd.DataFrame:
        """Load pitcher statistics from team-specific CSV file."""
        file_path = os.path.join(self.teams_dir, self._sanitize_team_name(team), 'pitchers', f"{pitcher_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pitcher data file not found: {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date', ascending=False)
    
    def prepare_matchup_features(self, batter_team: str, batter_name: str, pitcher_team: str, pitcher_name: str) -> np.ndarray:
        """Prepare features for a specific batter-pitcher matchup."""
        # Load data
        batter_df = self.load_batter_data(batter_team, batter_name)
        pitcher_df = self.load_pitcher_data(pitcher_team, pitcher_name)
        
        # Calculate features
        batter_features = self._calculate_batter_features(batter_df, pitcher_team)
        pitcher_features = self._calculate_pitcher_features(pitcher_df, batter_team)
        matchup_features = self._calculate_matchup_features(batter_df, pitcher_df)
        
        # Combine features
        features = np.concatenate([batter_features, pitcher_features, matchup_features])
        return features
    
    def _calculate_batter_features(self, df: pd.DataFrame, opposing_team: str) -> np.ndarray:
        """Calculate batter features including performance against specific team."""
        recent_games = df.head(10)
        vs_team_games = df[df['Opp'] == opposing_team]
        
        # Basic stats (recent performance)
        features = [
            recent_games['H'].mean(),  # Average hits
            recent_games['SO'].mean(),  # Average strikeouts
            recent_games['BA'].mean(),  # Average batting average
            recent_games['OBP'].mean(),  # Average on-base percentage
            recent_games['SLG'].mean(),  # Average slugging percentage
            recent_games['PA'].mean(),  # Average plate appearances
            recent_games['OPS'].mean(),  # Average OPS
            recent_games['WPA'].mean(),  # Average Win Probability Added
            recent_games['RE24'].mean(),  # Average Run Expectancy
            recent_games['DFS(DK)'].mean(),  # Average DraftKings points
        ]
        
        # Performance against specific team
        if not vs_team_games.empty:
            features.extend([
                vs_team_games['H'].mean(),
                vs_team_games['BA'].mean(),
                vs_team_games['OPS'].mean(),
            ])
        else:
            features.extend([0, 0, 0])  # No history against team
        
        # Trends
        last_5 = df.head(5)
        prev_5 = df.iloc[5:10]
        features.extend([
            last_5['H'].mean() - prev_5['H'].mean(),  # Hit trend
            last_5['BA'].mean() - prev_5['BA'].mean(),  # BA trend
            last_5['OPS'].mean() - prev_5['OPS'].mean(),  # OPS trend
        ])
        
        return np.array(features)
    
    def _calculate_pitcher_features(self, df: pd.DataFrame, opposing_team: str) -> np.ndarray:
        """Calculate pitcher features including performance against specific team."""
        recent_games = df.head(10)
        vs_team_games = df[df['Opp'] == opposing_team]
        
        # Basic stats (recent performance)
        features = [
            recent_games['ERA'].mean(),  # Average ERA
            recent_games['SO'].mean(),  # Average strikeouts
            recent_games['HR'].mean(),  # Average home runs allowed
            recent_games['BB'].mean(),  # Average walks
            recent_games['H'].mean(),  # Average hits allowed
            recent_games['IP'].mean(),  # Average innings pitched
            recent_games['FIP'].mean(),  # Fielding Independent Pitching
            recent_games['WPA'].mean(),  # Win Probability Added
            recent_games['RE24'].mean(),  # Run Expectancy
            recent_games['DFS(DK)'].mean(),  # DraftKings points
        ]
        
        # Performance against specific team
        if not vs_team_games.empty:
            features.extend([
                vs_team_games['ERA'].mean(),
                vs_team_games['SO'].mean(),
                vs_team_games['H'].mean(),
            ])
        else:
            features.extend([0, 0, 0])  # No history against team
        
        # Trends
        last_5 = df.head(5)
        prev_5 = df.iloc[5:10]
        features.extend([
            last_5['ERA'].mean() - prev_5['ERA'].mean(),  # ERA trend
            last_5['SO'].mean() - prev_5['SO'].mean(),  # Strikeout trend
            last_5['H'].mean() - prev_5['H'].mean(),  # Hits allowed trend
        ])
        
        return np.array(features)
    
    def _calculate_matchup_features(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> np.ndarray:
        """Calculate features specific to the batter-pitcher matchup history."""
        # Find common games (direct matchups)
        batter_games = set(batter_df['Date'].dt.date)
        pitcher_games = set(pitcher_df['Date'].dt.date)
        common_dates = batter_games.intersection(pitcher_games)
        
        if common_dates:
            matchup_df = pd.merge(
                batter_df[batter_df['Date'].dt.date.isin(common_dates)],
                pitcher_df[pitcher_df['Date'].dt.date.isin(common_dates)],
                on='Date',
                suffixes=('_batter', '_pitcher')
            )
            
            features = [
                matchup_df['H_batter'].mean(),  # Average hits in matchups
                matchup_df['BA_batter'].mean(),  # Batting average in matchups
                matchup_df['SO_batter'].mean(),  # Strikeouts in matchups
                len(common_dates),  # Number of matchups
            ]
        else:
            features = [0, 0, 0, 0]  # No direct matchup history
        
        return np.array(features) 