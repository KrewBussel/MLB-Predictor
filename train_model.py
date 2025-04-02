import numpy as np
import pandas as pd
import os
import itertools
from model import MLBModel
from data_processor import DataProcessor
from typing import List, Tuple, Dict

def create_synthetic_data_from_real_stats(data_processor: DataProcessor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on real player statistics.
    
    This function:
    1. Gets all available batters and pitchers
    2. Creates feature vectors for each possible matchup
    3. Generates realistic target values (hits) based on player stats
    
    Returns:
        X: Feature matrix
        y: Target values (number of hits)
    """
    print("Generating training data from real player statistics...")
    
    # Get all teams with available data
    teams = data_processor.get_all_teams()
    available_teams = []
    
    # Find teams that have batters or pitchers
    for team in teams:
        batters = data_processor.get_team_batters(team)
        pitchers = data_processor.get_team_pitchers(team)
        if batters or pitchers:
            available_teams.append(team)
    
    print(f"Found {len(available_teams)} teams with data")
    
    # Collect all batters and pitchers
    all_batters = []
    all_pitchers = []
    
    for team in available_teams:
        team_batters = data_processor.get_team_batters(team)
        for batter in team_batters:
            all_batters.append((team, batter))
        
        team_pitchers = data_processor.get_team_pitchers(team)
        for pitcher in team_pitchers:
            all_pitchers.append((team, pitcher))
    
    print(f"Found {len(all_batters)} batters and {len(all_pitchers)} pitchers")
    
    # Generate features for all matchups
    features_list = []
    hits_values = []
    
    # Create all possible matchups
    matchups = list(itertools.product(all_batters, all_pitchers))
    print(f"Generating {len(matchups)} possible matchups...")
    
    for (batter_team, batter_name), (pitcher_team, pitcher_name) in matchups:
        try:
            # Skip if batter and pitcher are on the same team (unlikely to face each other)
            if batter_team == pitcher_team:
                continue
                
            # Get features
            features = data_processor.prepare_matchup_features(
                batter_team, batter_name, pitcher_team, pitcher_name
            )
            
            # Skip if features contain only zeros (indicates error or missing data)
            if np.sum(features) == 0:
                continue
                
            # Calculate a realistic number of hits based on the features
            # The formula is weighted towards batting average and against pitcher ERA
            # This is a simplified model based on statistical relations
            
            # Extract key features for calculating hits
            batting_avg = features[2]  # Batting average
            obp = features[3]          # On-base percentage
            recent_hits = features[0]  # Recent hits
            pitcher_era = features[16] # Pitcher ERA
            
            # Calculate expected hits (more realistic formula than default 0.5)
            # Higher batting average and OBP increase expected hits
            # Higher pitcher ERA increases expected hits
            expected_hits = (
                batting_avg * 4.0 +       # Batting average is very important
                obp * 1.0 +               # On-base percentage adds some contribution
                recent_hits * 0.3 -       # Recent performance matters
                max(0, (3.5 - pitcher_era)) * 0.2  # Better pitchers (lower ERA) reduce hits
            )
            
            # Ensure hits are within reasonable range (usually 0-4 in a game)
            hits = max(0, min(4, expected_hits))
            
            # Add some noise to avoid perfectly correlated data
            hits += np.random.normal(0, 0.2)
            hits = max(0, hits)  # Ensure non-negative
            
            # Add to our training set
            features_list.append(features)
            hits_values.append(hits)
            
        except Exception as e:
            # Just skip this matchup if there's an error
            continue
    
    # Convert to numpy arrays
    if not features_list:
        raise ValueError("No valid matchups found to train the model")
        
    X = np.array(features_list)
    y = np.array(hits_values)
    
    print(f"Created {len(X)} training samples")
    return X, y

def enhance_training_data(X: np.ndarray, y: np.ndarray, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance training data with additional synthetic samples to improve model generalization.
    
    Args:
        X: Original feature matrix
        y: Original target values
        n_samples: Number of synthetic samples to generate
        
    Returns:
        X_enhanced: Enhanced feature matrix
        y_enhanced: Enhanced target values
    """
    from test_model import generate_sample_data
    
    # Generate additional realistic samples
    X_synthetic, y_synthetic = generate_sample_data(n_samples)
    
    # Combine real data with synthetic data
    X_enhanced = np.vstack([X, X_synthetic])
    y_enhanced = np.concatenate([y, y_synthetic])
    
    print(f"Enhanced dataset with {n_samples} synthetic samples")
    print(f"Total training samples: {len(X_enhanced)}")
    
    return X_enhanced, y_enhanced

def train_model_with_real_data():
    """Train the model with real player data"""
    data_processor = DataProcessor()
    model = MLBModel()
    
    try:
        # Generate training data from real player statistics
        X, y = create_synthetic_data_from_real_stats(data_processor)
        
        # Enhance the training data with additional synthetic samples
        X_enhanced, y_enhanced = enhance_training_data(X, y)
        
        # Train the model
        print("\nTraining model with real and synthetic data...")
        model.train(X_enhanced, y_enhanced)
        
        print("\nModel training complete!")
        print("The model should now make realistic predictions based on player statistics.")
        
        # Test a few predictions if we have data
        if X.shape[0] > 0:
            print("\nTesting predictions with a few real matchups...")
            for i in range(min(3, X.shape[0])):
                test_features = X[i]
                actual_value = y[i]
                pred, conf = model.predict_matchup(test_features)
                print(f"Matchup {i+1}:")
                print(f"  Predicted hits: {pred:.2f}")
                print(f"  Expected hits: {actual_value:.2f}")
                print(f"  Confidence: {conf:.1f}%")
        
    except Exception as e:
        print(f"Error training model: {e}")
        
if __name__ == "__main__":
    train_model_with_real_data()
