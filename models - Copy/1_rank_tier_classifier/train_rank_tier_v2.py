"""
Train Rank Tier Classifier with enriched features (temporal, mastery, team dynamics).

This version uses the new normalized per-minute metrics and champion mastery features
that should improve accuracy from 53% to 60-65%.

Features added:
- goldPerMinute, damagePerMinute, visionScorePerMinute (normalized)
- skillshotAccuracy, mechanicalSkill metrics
- visionControl, wardTakedowns, controlWardsPlaced
- championPoolSize, roleConsistency (team dynamics)
- earlyCS, soloKills, epicMonsterSteals (individual skill)
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.data_loader import DataLoader

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, classification_report, confusion_matrix)
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class RankTierTrainer:
    """Train and evaluate rank tier classification models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.class_names = ['Low', 'Mid', 'High', 'Elite']
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)
    
    def load_data(self, add_interactions=False):
        """Load and prepare data for training"""
        print("\nLoading enriched data...")
        
        # Load enriched features
        df = pd.read_csv('data/processed/rank_features_enriched_v2.csv')
        
        # Separate features and target
        y = pd.Categorical(df['tier']).codes  # Convert tier to numeric (0-9)
        
        # Remap to 4-class
        y = DataLoader.remap_tiers_to_4class(y)
        
        # Drop categorical and metadata columns
        X = df.drop(['tier', 'puuid', 'matches_used', 'role'], axis=1, errors='ignore')
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Add interaction features if requested
        if add_interactions:
            if 'goldPerMinute' in X.columns and 'avg_kda' in X.columns:
                X['gpm_x_kda'] = X['goldPerMinute'] * X['avg_kda']
            if 'goldPerMinute' in X.columns and 'damagePerMinute' in X.columns:
                X['gpm_x_dpm'] = X['goldPerMinute'] * X['damagePerMinute']
            if 'skillshotAccuracy' in X.columns and 'soloKills' in X.columns:
                X['mechanics_x_soloKills'] = X['skillshotAccuracy'] * X['soloKills']
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Loaded data: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"✓ Class distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y == i)
            pct = count / len(y) * 100
            print(f"  {class_name}: {count:,} ({pct:.1f}%)")
        
        return X, y
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, **kwargs):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'n_jobs': 1,  # Fix psutil compatibility
            'random_state': 42
        }
        params.update(kwargs)
        
        self.model = lgb.LGBMClassifier(**params)
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train)
        weights = {i: len(y_train) / (len(np.unique(y_train)) * count) 
                   for i, count in enumerate(class_counts)}
        sample_weights = np.array([weights[y] for y in y_train])
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
    
    def evaluate_model(self, y_true, y_pred, split_name='Test'):
        """Evaluate and print model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{split_name} Set Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\n{split_name} Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names,
                                   zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("RANK TIER CLASSIFIER V2 - WITH ENRICHED FEATURES")
    print("=" * 80)
    print("\nExpected improvement: 53% → 60-65% accuracy")
    print("New features: goldPerMinute, damagePerMinute, visionScorePerMinute,")
    print("              skillshotAccuracy, championPoolSize, roleConsistency")
    
    if not LIGHTGBM_AVAILABLE:
        print("\n❌ LightGBM not available. Install with: pip install lightgbm")
        return
    
    # Initialize trainer
    trainer = RankTierTrainer()
    
    # Load data
    X, y = trainer.load_data(add_interactions=False)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    trainer.scaler = StandardScaler()
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Log to MLflow (optional - skip for now)
    # log_experiment('rank-tier-classification-v2')
    
    # Train LightGBM
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM WITH ENRICHED FEATURES")
    print("=" * 80)
    
    results = trainer.train_lightgbm(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        num_leaves=50,
        learning_rate=0.05,
        n_estimators=300
    )
    
    # Evaluate
    test_metrics = trainer.evaluate_model(results['y_test'], results['y_pred_test'], 'Test')
    
    # Log metrics to MLflow
    hyperparams = {
        'model_type': 'LightGBM',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'features': len(trainer.feature_names),
        'version': 'v2_enriched'
    }
    
    metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'train_accuracy': results['train_accuracy']
    }
    
    # Log metrics to MLflow (optional - skip for now)
    # log_model_metrics('LightGBM-v2-enriched', hyperparams, metrics)
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL & ARTIFACTS")
    print("=" * 80)
    
    model_path = trainer.models_dir / 'rank_tier_model_v2_enriched.pkl'
    scaler_path = trainer.models_dir / 'scaler_v2_enriched.pkl'
    metadata_path = trainer.models_dir / 'metadata_v2_enriched.json'
    
    joblib.dump(trainer.model, model_path)
    joblib.dump(trainer.scaler, scaler_path)
    
    metadata = {
        'model_type': 'LightGBM',
        'features': trainer.feature_names,
        'class_names': trainer.class_names,
        'version': 'v2_enriched',
        'accuracy': float(test_metrics['accuracy']),
        'class_counts': {
            'Low': int(np.sum(y == 0)),
            'Mid': int(np.sum(y == 1)),
            'High': int(np.sum(y == 2)),
            'Elite': int(np.sum(y == 3))
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print("\nComparison:")
    print("  V1 (Original): 53.11% accuracy")
    print(f"  V2 (Enriched): {test_metrics['accuracy']*100:.2f}% accuracy")
    print(f"  Improvement: {(test_metrics['accuracy'] - 0.5311)*100:+.2f} percentage points")

if __name__ == '__main__':
    main()
