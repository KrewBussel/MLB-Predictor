import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from typing import Tuple, Dict, List

class MLBModel:
    def __init__(self):
        # Initialize multiple models
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.model_path = 'models/mlb_model.joblib'
        self.scaler_path = 'models/scaler.joblib'
        self._ensure_model_directory()
        self._load_or_initialize_model()
    
    def _ensure_model_directory(self):
        """Create models directory if it doesn't exist."""
        os.makedirs('models', exist_ok=True)
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize new one."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                saved_data = joblib.load(self.model_path)
                self.rf_model = saved_data['rf_model']
                self.gb_model = saved_data['gb_model']
                self.scaler = joblib.load(self.scaler_path)
                print("Loaded existing model and scaler")
        except Exception as e:
            print(f"Could not load existing model: {e}")
            print("Initializing new model")
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with given data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        print("Training Random Forest model...")
        self.rf_model.fit(X_scaled, y)
        print("Training Gradient Boosting model...")
        self.gb_model.fit(X_scaled, y)
        
        # Perform cross-validation
        print("\nPerforming cross-validation...")
        rf_scores = cross_val_score(self.rf_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        gb_scores = cross_val_score(self.gb_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        
        print(f"Random Forest CV RMSE: {np.sqrt(-rf_scores.mean()):.4f}")
        print(f"Gradient Boosting CV RMSE: {np.sqrt(-gb_scores.mean()):.4f}")
        
        # Save models and scaler
        saved_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model
        }
        joblib.dump(saved_data, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("\nSaved trained models and scaler")
    
    def predict_matchup(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction for a single batter-pitcher matchup.
        
        Args:
            features: Array of features for prediction
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(features_scaled)[0]
        gb_pred = self.gb_model.predict(features_scaled)[0]
        
        # Ensemble prediction (weighted average based on feature importance)
        rf_importance = np.mean(self.rf_model.feature_importances_)
        gb_importance = np.mean(self.gb_model.feature_importances_)
        
        total_importance = rf_importance + gb_importance
        rf_weight = rf_importance / total_importance
        gb_weight = gb_importance / total_importance
        
        prediction = rf_pred * rf_weight + gb_pred * gb_weight
        
        # Calculate confidence based on model agreement and feature importance
        model_agreement = 1 - abs(rf_pred - gb_pred) / max(rf_pred, gb_pred)
        feature_importance = (rf_importance + gb_importance) / 2
        confidence = (model_agreement * 0.6 + feature_importance * 0.4) * 100
        
        return prediction, confidence
    
    def predict_team_matchups(self, features_list: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Make predictions for multiple batter-pitcher matchups.
        
        Args:
            features_list: List of feature arrays for each matchup
            
        Returns:
            List of (prediction, confidence) tuples
        """
        predictions = []
        for features in features_list:
            pred, conf = self.predict_matchup(features)
            predictions.append((pred, conf))
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Ensemble predictions
        rf_importance = np.mean(self.rf_model.feature_importances_)
        gb_importance = np.mean(self.gb_model.feature_importances_)
        total_importance = rf_importance + gb_importance
        rf_weight = rf_importance / total_importance
        gb_weight = gb_importance / total_importance
        
        predictions = rf_pred * rf_weight + gb_pred * gb_weight
        
        # Calculate metrics
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> List[Tuple[str, float]]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance
        """
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # Calculate average importance between both models
        importance_scores = [(name, (rf_imp + gb_imp) / 2) 
                           for name, rf_imp, gb_imp in zip(feature_names, rf_importance, gb_importance)]
        
        # Sort by importance score
        return sorted(importance_scores, key=lambda x: x[1], reverse=True) 