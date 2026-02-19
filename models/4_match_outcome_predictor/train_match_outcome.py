"""
MATCH OUTCOME PREDICTOR - Training Script
Binary classification: Predict match winner from team differentials

Team Member: [YOUR NAME HERE]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.shared.mlflow_utils import MLflowTracker
from models.shared.data_loader import DataLoader
from models.shared.visualization import ModelVisualizations


class MatchOutcomePredictor:
    """Train and evaluate match outcome prediction models"""
    
    EXPERIMENT_NAME = "match-outcome-prediction"
    MODEL_DIR = Path(__file__).parent / "models"
    
    def __init__(self):
        """Initialize predictor"""
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
        
        X, y, feature_names = DataLoader.prepare_match_features()
        self.feature_names = feature_names
        
        print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Class distribution: {np.bincount(y)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        print(f"✓ Features scaled with StandardScaler")
    
    def train_logistic_regression(self, run_name: str = "LogisticRegression-v1", **kwargs):
        """
        Train Logistic Regression model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'C': kwargs.get('C', 1.0),
            'max_iter': kwargs.get('max_iter', 1000),
            'random_state': 42
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params)
        
        # Train model
        self.model = LogisticRegression(**params)
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_model()
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "match_outcome_predictor", flavor="sklearn")
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def train_random_forest(self, run_name: str = "RandomForest-v1", **kwargs):
        """
        Train Random Forest model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 15),
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params(params)
        
        # Train model
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_model()
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "match_outcome_predictor", flavor="sklearn")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        tracker.log_dict(feature_importance.to_dict('list'), 'feature_importance.json')
        
        print(f"✓ Metrics logged to MLflow")
        tracker.end_run()
        
        return metrics
    
    def train_xgboost(self, run_name: str = "XGBoost-v1", **kwargs):
        """
        Train XGBoost model
        
        Args:
            run_name: Name for the MLflow run
            **kwargs: Model hyperparameters
        """
        print("\n" + "="*70)
        print(f"TRAINING: {run_name}")
        print("="*70)
        
        # Default hyperparameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        tracker.log_params({k: v for k, v in params.items() if k not in ['eval_metric']})
        
        # Train model
        self.model = XGBClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model trained")
        
        # Evaluate
        metrics = self._evaluate_model()
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(self.model, "match_outcome_predictor", flavor="xgboost")
        
        # Log feature importance
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
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
        }
        
        print(f"\n  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
        print(f"    FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")
        
        return metrics
    
    def save_model(self, model_name: str = "match_outcome_model.pkl"):
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
            'target': 'team_won',
            'n_features': len(self.feature_names)
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
    print("MATCH OUTCOME PREDICTOR - TRAINING PIPELINE")
    print("="*70)
    
    # Initialize predictor
    predictor = MatchOutcomePredictor()
    predictor.load_data()
    
    # Train Logistic Regression (baseline)
    predictor.train_logistic_regression(
        run_name="LogisticRegression-v1",
        C=1.0
    )
    
    # Train Random Forest v1
    predictor.train_random_forest(
        run_name="RandomForest-v1",
        n_estimators=100,
        max_depth=15
    )
    
    # Train Random Forest v2 (deeper)
    predictor.train_random_forest(
        run_name="RandomForest-v2",
        n_estimators=150,
        max_depth=20
    )
    
    # Train XGBoost v1
    predictor.train_xgboost(
        run_name="XGBoost-v1",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Train XGBoost v2 (higher learning rate)
    predictor.train_xgboost(
        run_name="XGBoost-v2",
        n_estimators=150,
        max_depth=7,
        learning_rate=0.15
    )
    
    # Save best model
    predictor.save_model()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print("\nAll runs logged to MLflow:")
    print("  Experiment: match-outcome-prediction")
    print("\nView results at: http://localhost:5000")


if __name__ == "__main__":
    main()
