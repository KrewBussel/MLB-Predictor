import numpy as np
import pandas as pd
from model import MLBModel
from data_processor import DataProcessor
import os

def get_player_stats(player_type, player_name):
    """Get player statistics from CSV file."""
    if player_type == 'batter':
        file_path = f'data/batters/{player_name}.csv'
    else:
        file_path = f'data/pitchers/{player_name}.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data found for {player_name}")
    
    df = pd.read_csv(file_path)
    # Sort by date in descending order to get most recent games first
    df = df.sort_values('Date', ascending=False)
    
    if player_type == 'batter':
        # Calculate recent stats (last 5 games)
        recent_games = df.head(5)
        print("\nMost recent games:")
        for _, game in recent_games.iterrows():
            print(f"Date: {game['Date']}, H: {game['H']}, AB: {game['AB']}, BA: {game['BA']:.3f}")
            
        stats = {
            'recent_hits': recent_games['H'].mean(),
            'strikeouts': recent_games['SO'].mean(),
            'batting_avg': recent_games['BA'].mean(),
            'obp': recent_games['OBP'].mean(),
            'slg': recent_games['SLG'].mean(),
            'plate_appearances': recent_games['PA'].mean(),
            'ops': recent_games['OPS'].mean(),
            'wpa': recent_games['WPA'].mean(),
            're24': recent_games['RE24'].mean(),
            'dfs_points': recent_games['DFS(DK)'].mean(),
            'hits_vs_team': recent_games['H'].mean(),  # Using overall recent hits as proxy
            'ba_vs_team': recent_games['BA'].mean(),   # Using overall recent BA as proxy
            'ops_vs_team': recent_games['OPS'].mean(), # Using overall recent OPS as proxy
            'hit_trend': (recent_games['H'].iloc[0] - recent_games['H'].iloc[-1]) / len(recent_games),
            'ba_trend': (recent_games['BA'].iloc[0] - recent_games['BA'].iloc[-1]) / len(recent_games),
            'ops_trend': (recent_games['OPS'].iloc[0] - recent_games['OPS'].iloc[-1]) / len(recent_games),
        }
    else:
        # Calculate recent stats (last 5 games for pitchers)
        recent_games = df.head(5)
        print("\nMost recent games:")
        for _, game in recent_games.iterrows():
            print(f"Date: {game['Date']}, IP: {game['IP']}, H: {game['H']}, ERA: {game['ERA']:.2f}")
            
        stats = {
            'era': recent_games['ERA'].mean(),
            'strikeouts': recent_games['SO'].mean(),
            'hr_allowed': recent_games['HR'].mean(),
            'walks': recent_games['BB'].mean(),
            'hits_allowed': recent_games['H'].mean(),
            'innings_pitched': recent_games['IP'].mean(),
            'fip': recent_games['FIP'].mean(),
            'wpa': recent_games['WPA'].mean(),
            're24': recent_games['RE24'].mean(),
            'dfs_points': recent_games['DFS(DK)'].mean(),
            'era_vs_team': recent_games['ERA'].mean(),  # Using overall recent ERA as proxy
            'so_vs_team': recent_games['SO'].mean(),    # Using overall recent SO as proxy
            'hits_vs_team': recent_games['H'].mean(),   # Using overall recent hits allowed as proxy
            'era_trend': (recent_games['ERA'].iloc[0] - recent_games['ERA'].iloc[-1]) / len(recent_games),
            'so_trend': (recent_games['SO'].iloc[0] - recent_games['SO'].iloc[-1]) / len(recent_games),
            'hits_allowed_trend': (recent_games['H'].iloc[0] - recent_games['H'].iloc[-1]) / len(recent_games),
        }
    
    return stats

def get_matchup_stats(batter_name, pitcher_name):
    """Get historical matchup statistics."""
    # For now, we'll use default values since we don't have specific matchup data
    # In a real implementation, you would query a database of historical matchups
    return {
        'hits_in_matchups': 1.2,  # Default value
        'ba_in_matchups': 0.300,  # Default value
        'so_in_matchups': 1.0,    # Default value
        'num_matchups': 5,        # Default value
    }

def create_feature_vector(batter_stats, pitcher_stats, matchup_stats):
    """Create a feature vector from the input statistics."""
    features = np.zeros(36)
    
    # Fill batter features
    features[0] = batter_stats['recent_hits']
    features[1] = batter_stats['strikeouts']
    features[2] = batter_stats['batting_avg']
    features[3] = batter_stats['obp']
    features[4] = batter_stats['slg']
    features[5] = batter_stats['plate_appearances']
    features[6] = batter_stats['ops']
    features[7] = batter_stats['wpa']
    features[8] = batter_stats['re24']
    features[9] = batter_stats['dfs_points']
    features[10] = batter_stats['hits_vs_team']
    features[11] = batter_stats['ba_vs_team']
    features[12] = batter_stats['ops_vs_team']
    features[13] = batter_stats['hit_trend']
    features[14] = batter_stats['ba_trend']
    features[15] = batter_stats['ops_trend']
    
    # Fill pitcher features
    features[16] = pitcher_stats['era']
    features[17] = pitcher_stats['strikeouts']
    features[18] = pitcher_stats['hr_allowed']
    features[19] = pitcher_stats['walks']
    features[20] = pitcher_stats['hits_allowed']
    features[21] = pitcher_stats['innings_pitched']
    features[22] = pitcher_stats['fip']
    features[23] = pitcher_stats['wpa']
    features[24] = pitcher_stats['re24']
    features[25] = pitcher_stats['dfs_points']
    features[26] = pitcher_stats['era_vs_team']
    features[27] = pitcher_stats['so_vs_team']
    features[28] = pitcher_stats['hits_vs_team']
    features[29] = pitcher_stats['era_trend']
    features[30] = pitcher_stats['so_trend']
    features[31] = pitcher_stats['hits_allowed_trend']
    
    # Fill matchup features
    features[32] = matchup_stats['hits_in_matchups']
    features[33] = matchup_stats['ba_in_matchups']
    features[34] = matchup_stats['so_in_matchups']
    features[35] = matchup_stats['num_matchups']
    
    return features.reshape(1, -1)

def test_specific_matchup():
    # Initialize model
    model = MLBModel()
    
    print("Welcome to the MLB Hit Predictor!")
    print("This tool will help predict how many hits a batter will get against a specific pitcher.")
    print("\nAvailable batters:")
    print("- Aaron_Judge")
    print("- Cody_Bellinger")
    print("\nAvailable pitchers:")
    print("- Zack_Wheeler")
    print("- Tarik_Skubal")
    
    # Get input from user
    batter_name = input("\nEnter batter name (e.g., Aaron_Judge): ")
    pitcher_name = input("Enter pitcher name (e.g., Zack_Wheeler): ")
    
    try:
        # Get statistics from CSV files
        batter_stats = get_player_stats('batter', batter_name)
        pitcher_stats = get_player_stats('pitcher', pitcher_name)
        matchup_stats = get_matchup_stats(batter_name, pitcher_name)
        
        # Create feature vector
        features = create_feature_vector(batter_stats, pitcher_stats, matchup_stats)
        
        # Make prediction
        prediction, confidence = model.predict_team_matchups([features])[0]
        
        # Display results
        print(f"\nPrediction for {batter_name} vs {pitcher_name}:")
        print(f"Predicted hits: {prediction:.2f}")
        print(f"Confidence: {confidence:.1f}%")
        
        # Display batter's recent performance
        print(f"\nBatter's Recent Performance (Last 5 Games):")
        print(f"Average Hits per Game: {batter_stats['recent_hits']:.2f}")
        print(f"Batting Average: {batter_stats['batting_avg']:.3f}")
        print(f"On-base Percentage: {batter_stats['obp']:.3f}")
        print(f"Slugging: {batter_stats['slg']:.3f}")
        print(f"OPS: {batter_stats['ops']:.3f}")
        
        # Display pitcher's recent performance
        print(f"\nPitcher's Recent Performance (Last 5 Games):")
        print(f"ERA: {pitcher_stats['era']:.2f}")
        print(f"Strikeouts per Game: {pitcher_stats['strikeouts']:.1f}")
        print(f"Hits Allowed per Game: {pitcher_stats['hits_allowed']:.1f}")
        print(f"Innings Pitched per Game: {pitcher_stats['innings_pitched']:.1f}")
        print(f"FIP: {pitcher_stats['fip']:.2f}")
        
        # Display key factors
        print("\nKey Factors Influencing Prediction:")
        feature_names = [
            "Recent Hits", "Strikeouts", "Batting Avg", "OBP", "SLG", "PA",
            "OPS", "WPA", "RE24", "DFS Points", "Hits vs Team", "BA vs Team",
            "OPS vs Team", "Hit Trend", "BA Trend", "OPS Trend",
            "ERA", "SO", "HR Allowed", "BB", "Hits Allowed", "IP",
            "FIP", "Pitcher WPA", "Pitcher RE24", "Pitcher DFS",
            "ERA vs Team", "SO vs Team", "Hits vs Team",
            "ERA Trend", "SO Trend", "Hits Allowed Trend",
            "Matchup Hits", "Matchup BA", "Matchup SO", "Num Matchups"
        ]
        
        importance_scores = model.get_feature_importance(feature_names)
        print("\nTop 5 most influential factors:")
        for name, score in sorted(importance_scores, key=lambda x: x[1], reverse=True)[:5]:
            print(f"{name}: {score:.4f}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_specific_matchup() 