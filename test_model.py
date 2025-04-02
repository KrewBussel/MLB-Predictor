import numpy as np
from model import MLBModel
from data_processor import DataProcessor

def generate_sample_data(n_samples=1000):
    """Generate realistic sample data for testing."""
    # Generate features (16 batter + 16 pitcher + 4 matchup = 36 total features)
    X = np.zeros((n_samples, 36))
    
    # Batter features (first 16)
    X[:, 0] = np.random.normal(1.5, 0.5, n_samples)  # Average hits
    X[:, 1] = np.random.normal(1.2, 0.4, n_samples)  # Average strikeouts
    X[:, 2] = np.random.normal(0.25, 0.05, n_samples)  # Batting average
    X[:, 3] = np.random.normal(0.32, 0.05, n_samples)  # On-base percentage
    X[:, 4] = np.random.normal(0.42, 0.08, n_samples)  # Slugging percentage
    X[:, 5] = np.random.normal(4.0, 0.5, n_samples)  # Plate appearances
    X[:, 6] = np.random.normal(0.75, 0.1, n_samples)  # OPS
    X[:, 7] = np.random.normal(0.05, 0.02, n_samples)  # WPA
    X[:, 8] = np.random.normal(0.15, 0.05, n_samples)  # RE24
    X[:, 9] = np.random.normal(8.5, 2.0, n_samples)  # DraftKings points
    X[:, 10] = np.random.normal(1.2, 0.4, n_samples)  # Hits vs team
    X[:, 11] = np.random.normal(0.26, 0.05, n_samples)  # BA vs team
    X[:, 12] = np.random.normal(0.72, 0.1, n_samples)  # OPS vs team
    X[:, 13] = np.random.normal(0.1, 0.05, n_samples)  # Hit trend
    X[:, 14] = np.random.normal(0.02, 0.01, n_samples)  # BA trend
    X[:, 15] = np.random.normal(0.05, 0.02, n_samples)  # OPS trend
    
    # Pitcher features (next 16)
    X[:, 16] = np.random.normal(3.5, 0.5, n_samples)  # ERA
    X[:, 17] = np.random.normal(6.0, 1.0, n_samples)  # Strikeouts
    X[:, 18] = np.random.normal(1.0, 0.3, n_samples)  # Home runs allowed
    X[:, 19] = np.random.normal(2.0, 0.5, n_samples)  # Walks
    X[:, 20] = np.random.normal(6.0, 1.0, n_samples)  # Hits allowed
    X[:, 21] = np.random.normal(6.0, 1.0, n_samples)  # Innings pitched
    X[:, 22] = np.random.normal(3.8, 0.5, n_samples)  # FIP
    X[:, 23] = np.random.normal(-0.05, 0.02, n_samples)  # WPA
    X[:, 24] = np.random.normal(-0.15, 0.05, n_samples)  # RE24
    X[:, 25] = np.random.normal(12.5, 3.0, n_samples)  # DraftKings points
    X[:, 26] = np.random.normal(3.8, 0.6, n_samples)  # ERA vs team
    X[:, 27] = np.random.normal(5.5, 1.0, n_samples)  # SO vs team
    X[:, 28] = np.random.normal(5.8, 1.0, n_samples)  # Hits vs team
    X[:, 29] = np.random.normal(-0.02, 0.01, n_samples)  # ERA trend
    X[:, 30] = np.random.normal(0.1, 0.05, n_samples)  # Strikeout trend
    X[:, 31] = np.random.normal(-0.1, 0.05, n_samples)  # Hits allowed trend
    
    # Matchup features (last 4)
    X[:, 32] = np.random.normal(1.2, 0.4, n_samples)  # Hits in matchups
    X[:, 33] = np.random.normal(0.25, 0.05, n_samples)  # BA in matchups
    X[:, 34] = np.random.normal(1.0, 0.3, n_samples)  # SO in matchups
    X[:, 35] = np.random.randint(0, 10, n_samples)  # Number of matchups
    
    # Generate target values (number of hits) with some noise
    y = (X[:, 0] * 0.3 +  # Recent hits
         X[:, 2] * 1.5 +  # Batting average
         X[:, 3] * 1.2 +  # On-base percentage
         X[:, 10] * 0.2 +  # Hits vs team
         X[:, 13] * 0.1 +  # Hit trend
         X[:, 16] * -0.15 +  # Pitcher ERA
         X[:, 20] * -0.1 +  # Pitcher hits allowed
         X[:, 26] * -0.15 +  # ERA vs team
         X[:, 32] * 0.25 +  # Hits in matchups
         np.random.normal(0, 0.2, n_samples))  # Random noise
    
    # Ensure hits are non-negative
    y = np.maximum(y, 0)
    
    return X, y

def test_model():
    # Initialize components
    model = MLBModel()
    data_processor = DataProcessor()
    
    # Generate realistic sample data
    print("Generating sample data...")
    X, y = generate_sample_data(1000)
    
    # Train model
    print("\nTraining model...")
    model.train(X, y)
    
    # Test team matchup predictions
    print("\nTesting team matchup predictions...")
    n_pitchers = 5
    features_list = [generate_sample_data(1)[0] for _ in range(n_pitchers)]
    
    predictions = model.predict_team_matchups(features_list)
    
    print("\nPredictions for batter against different pitchers:")
    for i, (pred, conf) in enumerate(predictions, 1):
        print(f"Pitcher {i}:")
        print(f"  Predicted hits: {pred:.2f}")
        print(f"  Confidence: {conf:.1f}%")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X, y)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_names = [
        # Batter features
        "Recent Hits", "Strikeouts", "Batting Avg", "OBP", "SLG", "PA",
        "OPS", "WPA", "RE24", "DFS Points", "Hits vs Team", "BA vs Team",
        "OPS vs Team", "Hit Trend", "BA Trend", "OPS Trend",
        # Pitcher features
        "ERA", "SO", "HR Allowed", "BB", "Hits Allowed", "IP",
        "FIP", "Pitcher WPA", "Pitcher RE24", "Pitcher DFS",
        "ERA vs Team", "SO vs Team", "Hits vs Team",
        "ERA Trend", "SO Trend", "Hits Allowed Trend",
        # Matchup features
        "Matchup Hits", "Matchup BA", "Matchup SO", "Num Matchups"
    ]
    
    importance_scores = model.get_feature_importance(feature_names)
    for name, score in importance_scores:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    test_model() 