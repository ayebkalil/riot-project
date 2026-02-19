"""
SMURF ANOMALY DETECTOR - Training Script
Unsupervised anomaly detection for smurf/suspicious accounts

Team Member: [YOUR NAME HERE]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.shared.mlflow_utils import MLflowTracker
from models.shared.data_loader import DataLoader
from models.shared.visualization import ModelVisualizations


class SmurfAnomalyDetector:
    """Train and evaluate anomaly detection models for smurf detection"""
    
    EXPERIMENT_NAME = "smurf-anomaly-detection"
    MODEL_DIR = Path(__file__).parent / "models"
    
    def __init__(self):
        """Initialize detector"""
        self.X = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.anomaly_scores = None
    
    def load_data(self):
        """Load data"""
        print("\n" + "="*70)
        print("LOADING DATA...")
        print("="*70)
        
        X, feature_names = DataLoader.prepare_smurf_features()
        self.feature_names = feature_names
        
        print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Scale features
        self.X = self.scaler.fit_transform(X)
        
        print(f"✓ Features scaled with StandardScaler")
    
    def train_isolation_forest(self, run_name: str = "IsolationForest-v1", **kwargs):
        """
        Train Isolation Forest model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'contamination': kwargs.get('contamination', 0.1),
            'n_estimators': kwargs.get('n_estimators', 100),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params)
        
        # Train model
        self.model = IsolationForest(**params)
        predictions = self.model.fit_predict(self.X)
        self.anomaly_scores = self.model.score_samples(self.X)
        
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_anomaly_model(predictions)
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "smurf_anomaly_detector", flavor="sklearn")
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def train_elliptic_envelope(self, run_name: str = "EllipticEnvelope-v1", **kwargs):
        """
        Train Elliptic Envelope model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'contamination': kwargs.get('contamination', 0.1),
            'random_state': 42
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params)
        
        # Train model
        self.model = EllipticEnvelope(**params)
        predictions = self.model.fit_predict(self.X)
        self.anomaly_scores = self.model.score_samples(self.X)
        
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_anomaly_model(predictions)
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "smurf_anomaly_detector", flavor="sklearn")
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def train_local_outlier_factor(self, run_name: str = "LocalOutlierFactor-v1", **kwargs):
        """
        Train Local Outlier Factor model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'n_neighbors': kwargs.get('n_neighbors', 20),
            'contamination': kwargs.get('contamination', 0.1),
            'novelty': False
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params)
        
        # Train model
        self.model = LocalOutlierFactor(**params)
        predictions = self.model.fit_predict(self.X)
        self.anomaly_scores = self.model.negative_outlier_factor_
        
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_anomaly_model(predictions)
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "smurf_anomaly_detector", flavor="sklearn")
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def _evaluate_anomaly_model(self, predictions: np.ndarray) -> dict:
        """
        Evaluate unsupervised anomaly model
        
        Args:
            predictions: Predictions from model (-1 for anomalies, 1 for normal)
        
        Returns:
            Dict of metrics
        """
        n_anomalies = (predictions == -1).sum()
        n_normal = (predictions == 1).sum()
        anomaly_ratio = n_anomalies / len(predictions)
        
        metrics = {
            'n_anomalies': int(n_anomalies),
            'n_normal': int(n_normal),
            'anomaly_ratio': float(anomaly_ratio),
            'mean_anomaly_score': float(np.mean(self.anomaly_scores)),
            'std_anomaly_score': float(np.std(self.anomaly_scores)),
            'min_anomaly_score': float(np.min(self.anomaly_scores)),
            'max_anomaly_score': float(np.max(self.anomaly_scores)),
        }
        
        print(f"\n  Anomalies detected: {n_anomalies} ({anomaly_ratio*100:.2f}%)")
        print(f"  Normal samples: {n_normal} ({(1-anomaly_ratio)*100:.2f}%)")
        print(f"  Mean anomaly score: {metrics['mean_anomaly_score']:.4f}")
        print(f"  Std anomaly score: {metrics['std_anomaly_score']:.4f}")
        print(f"  Score range: [{metrics['min_anomaly_score']:.4f}, {metrics['max_anomaly_score']:.4f}]")
        
        return metrics
    
    def save_model(self, model_name: str = "smurf_anomaly_model.pkl", run_name: str | None = None):
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
            'task': 'anomaly_detection',
            'n_features': len(self.feature_names)
        }
        if run_name:
            metadata['run_name'] = run_name
        
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
    print("SMURF ANOMALY DETECTOR - TRAINING PIPELINE")
    print("="*70)
    
    # Initialize detector
    detector = SmurfAnomalyDetector()
    detector.load_data()
    
    # Train Isolation Forest v1 (low contamination)
    detector.train_isolation_forest(
        run_name="IsolationForest-v1",
        contamination=0.08,
        n_estimators=100
    )
    
    # Train Isolation Forest v2 (higher contamination)
    detector.train_isolation_forest(
        run_name="IsolationForest-v2",
        contamination=0.12,
        n_estimators=150
    )
    best_model = detector.model
    best_run_name = "IsolationForest-v2"
    
    # Train Elliptic Envelope
    detector.train_elliptic_envelope(
        run_name="EllipticEnvelope-v1",
        contamination=0.10
    )
    
    # Train Local Outlier Factor
    detector.train_local_outlier_factor(
        run_name="LocalOutlierFactor-v1",
        n_neighbors=20,
        contamination=0.10
    )
    
    # Save best model
    detector.model = best_model
    detector.save_model(run_name=best_run_name)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print("\nAll runs logged to MLflow:")
    print("  Experiment: smurf-anomaly-detection")
    print("\nView results at: http://localhost:5000")


if __name__ == "__main__":
    main()
