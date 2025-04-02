from data_processor import DataProcessor
import pandas as pd
import numpy as np

def test_data_processor():
    # Initialize the data processor
    processor = DataProcessor()
    
    # Test getting available seasons
    seasons = processor.get_available_seasons()
    print("\nAvailable seasons:", seasons)
    
    # Test getting teams for 2024
    teams_2024 = processor.get_teams_for_season(2024)
    print("\nTeams available for 2024:", teams_2024)
    
    # Test getting batters for Yankees
    yankees_batters = processor.get_team_batters(2024, "new_york_yankees")
    print("\nYankees batters:", yankees_batters)
    
    # Test getting pitchers for Yankees
    yankees_pitchers = processor.get_team_pitchers(2024, "new_york_yankees")
    print("\nYankees pitchers:", yankees_pitchers)
    
    # Test loading specific player data
    try:
        # Load Aaron Judge's data
        judge_data = processor.load_batter_data(2024, "new_york_yankees", "aaron_judge_2024")
        print("\nAaron Judge's data shape:", judge_data.shape)
        print("\nFirst few rows of Judge's data:")
        print(judge_data.head())
        
        # Load Gerrit Cole's data
        cole_data = processor.load_pitcher_data(2024, "new_york_yankees", "garrit_cole_2024")
        print("\nGerrit Cole's data shape:", cole_data.shape)
        print("\nFirst few rows of Cole's data:")
        print(cole_data.head())
        
        # Test feature preparation
        features = processor.prepare_features(
            2024,
            "new_york_yankees",
            "aaron_judge_2024",
            "new_york_yankees",
            "garrit_cole_2024"
        )
        print("\nPrepared features:")
        for key, value in features.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"\nError during testing: {e}")

if __name__ == "__main__":
    test_data_processor() 