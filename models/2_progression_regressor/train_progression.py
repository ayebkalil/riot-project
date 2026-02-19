"""
PROGRESSION REGRESSOR - Training Script
Predict player win rate delta (progression): continuous regression

Team Member: [YOUR NAME HERE]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.shared.mlflow_utils import MLflowTracker
from models.shared.data_loader import DataLoader
from models.shared.visualization import ModelVisualizations


class ProgressionRegressor:
    """Train and evaluate progression (delta_winrate) prediction models"""
    
    EXPERIMENT_NAME = "progression-regression"
    MODEL_DIR = Path(__file__).parent / "models"
    
    def __init__(self):
        """Initialize regressor"""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
    
    def load_data(self):
        """Load and split data"""
        print("\n" + "="*70)
        print("LOADING DATA...")
        print("="*70)
        
        X, y, feature_names = DataLoader.prepare_progression_features()
        self.feature_names = feature_names
        
        print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Target range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        print(f"✓ Features scaled with StandardScaler")
    
    def train_model(self, model_type: str = "random_forest", run_name: str = None, **kwargs):
        """
        Train regression model
        
        Args:
            model_type: "linear", "ridge", or "random_forest"
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        if run_name is None:
            run_name = f"{model_type.upper()}-v1"
        
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Initialize model based on type
        if model_type == "linear":
            self.model = LinearRegression()
            params = {}
        
        elif model_type == "ridge":
            params = {'alpha': kwargs.get('alpha', 1.0), 'random_state': 42}
            params.update(kwargs)
            self.model = Ridge(**params)
        
        elif model_type == "random_forest":
            params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': 1  # Fix psutil compatibility issue
            }
            params.update(kwargs)
            self.model = RandomForestRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params if params else {'model': model_type})
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_model()
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "progression_regressor", flavor="sklearn")
        
        # Log feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            tracker.log_dict(feature_importance.to_dict('list'), 'feature_importance.json')
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def _evaluate_model(self) -> dict:
        """Evaluate model and return metrics"""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        metrics = {
            'mse_train': mean_squared_error(self.y_train, y_pred_train),
            'mse_test': mean_squared_error(self.y_test, y_pred_test),
            'rmse_train': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'rmse_test': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'mae_train': mean_absolute_error(self.y_train, y_pred_train),
            'mae_test': mean_absolute_error(self.y_test, y_pred_test),
            'r2_train': r2_score(self.y_train, y_pred_train),
            'r2_test': r2_score(self.y_test, y_pred_test),
        }
        
        print(f"\n  Train MSE:      {metrics['mse_train']:.6f}")
        print(f"  Test MSE:       {metrics['mse_test']:.6f}")
        print(f"  Train RMSE:     {metrics['rmse_train']:.6f}")
        print(f"  Test RMSE:      {metrics['rmse_test']:.6f}")
        print(f"  Train MAE:      {metrics['mae_train']:.6f}")
        print(f"  Test MAE:       {metrics['mae_test']:.6f}")
        print(f"  Train R²:       {metrics['r2_train']:.4f}")
        print(f"  Test R²:        {metrics['r2_test']:.4f}")
        
        return metrics
    
    def save_model(self, model_name: str = "progression_model.pkl"):
        """Save trained model and metadata"""
        if self.model is None:
            print("✗ No model trained yet")
            return
        
        model_path = self.MODEL_DIR / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = self.MODEL_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'scaler_path': str(scaler_path),
            'model_type': type(self.model).__name__,
            'target': 'delta_winrate'
        }
        
        import json
        metadata_path = self.MODEL_DIR / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Metadata saved: {metadata_path}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("PROGRESSION REGRESSOR - TRAINING PIPELINE")
    print("="*70)
    
    # Initialize regressor
    regressor = ProgressionRegressor()
    regressor.load_data()
    
    # Train Linear Regression (baseline)
    regressor.train_model(
        model_type="linear",
        run_name="LinearRegression-v1"
    )
    
    # Train Ridge Regression v1 (low regularization)
    regressor.train_model(
        model_type="ridge",
        run_name="RidgeRegression-v1",
        alpha=0.1
    )
    
    # Train Ridge Regression v2 (high regularization)
    regressor.train_model(
        model_type="ridge",
        run_name="RidgeRegression-v2",
        alpha=1.0
    )
    
    # Train Random Forest
    regressor.train_model(
        model_type="random_forest",
        run_name="RandomForest-v1",
        n_estimators=100,
        max_depth=15
    )
    
    # Save best model
    regressor.save_model()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print("\nAll runs logged to MLflow:")
    print("  Experiment: progression-regression")
    print("\nView results at: http://localhost:5000")


if __name__ == "__main__":
    main()
