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

# Import MLflow tracker
from shared.mlflow_utils import MLflowTracker

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print(f"✓ LightGBM imported successfully (version {lgb.__version__})")
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    print(f"❌ LightGBM import error: {e}")

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
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, **kwargs):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'y_pred_proba': self.model.predict_proba(X_test)
        }
    
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
            'n_jobs': 1,
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
            'y_test': y_test,
            'y_pred_proba': self.model.predict_proba(X_test)
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, **kwargs):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**params)
        
        # Class weights
        class_counts = np.bincount(y_train)
        weights = len(y_train) / (len(np.unique(y_train)) * class_counts)
        sample_weights = np.array([weights[y] for y in y_train])
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
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
            'y_test': y_test,
            'y_pred_proba': self.model.predict_proba(X_test)
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
    
    def plot_confusion_matrix(self, y_true, y_pred, split_name='Test'):
        """Generate and return confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax, annot_kws={'size': 12})
        plt.title(f'Confusion Matrix - {split_name} Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, top_n=15):
        """Generate and return feature importance visualization"""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠ Model does not have feature_importances_ attribute")
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_per_class_metrics(self, y_true, y_pred):
        """Generate per-class performance metrics visualization"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Create bar chart
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Rank Tier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_class_distribution(self, y_train, y_test):
        """Generate class distribution comparison"""
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_counts, width, label='Training Set', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, test_counts, width, label='Test Set', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Rank Tier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution (Train vs Test)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (train, test) in enumerate(zip(train_counts, test_counts)):
            ax.text(i - width/2, train + 20, str(train), ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, test + 20, str(test), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_confidence(self, y_true, y_pred_proba):
        """Generate prediction confidence distribution"""
        # Get confidence (max probability) for each prediction
        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        correct = (predictions == y_true)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confidence distribution for correct vs incorrect
        ax1.hist(confidences[correct], bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
        ax1.hist(confidences[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        ax1.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Average confidence per class
        avg_conf_per_class = []
        for class_idx in range(len(self.class_names)):
            class_mask = (predictions == class_idx)
            if class_mask.sum() > 0:
                avg_conf_per_class.append(confidences[class_mask].mean())
            else:
                avg_conf_per_class.append(0)
        
        ax2.bar(range(len(self.class_names)), avg_conf_per_class, color='steelblue', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Rank Tier', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
        ax2.set_title('Average Prediction Confidence by Class', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_conf_per_class):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, train_acc, test_acc, train_f1, test_f1):
        """Generate train vs test performance comparison"""
        metrics = ['Accuracy', 'F1-Score']
        train_scores = [train_acc, train_f1]
        test_scores = [test_acc, test_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_scores, width, label='Training', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, test_scores, width, label='Test', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (train, test) in enumerate(zip(train_scores, test_scores)):
            ax.text(i - width/2, train + 0.02, f'{train:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add overfitting indicator
        overfitting = train_acc - test_acc
        color = 'red' if overfitting > 0.1 else 'orange' if overfitting > 0.05 else 'green'
        ax.text(0.5, 0.95, f'Overfitting: {overfitting:.3f}', transform=ax.transAxes, 
                ha='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        return fig

def train_model_and_log(trainer, model_name, train_fn, X_train, y_train, X_test, y_test, 
                        X_train_scaled, X_test_scaled, tracker, hyperparams):
    """Train a single model and log all results to MLflow"""
    
    print("\n" + "=" * 80)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 80)
    
    # Start MLflow run
    tracker.start_run(model_name)
    
    # Train model
    results = train_fn(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate
    test_metrics = trainer.evaluate_model(results['y_test'], results['y_pred_test'], 'Test')
    
    # Log hyperparameters
    tracker.log_params(hyperparams)
    
    # Log metrics
    metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'train_accuracy': results['train_accuracy'],
        'train_f1': results['train_f1']
    }
    tracker.log_metrics(metrics)
    
    # Generate and log visualizations
    print("\n[Generating Visualizations...]")
    
    # 1. Confusion Matrix
    cm_fig = trainer.plot_confusion_matrix(results['y_test'], results['y_pred_test'], 'Test')
    cm_path = trainer.models_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(cm_fig)
    tracker.log_artifact(str(cm_path), artifact_path='visualizations')
    
    # 2. Feature Importance
    fi_fig = trainer.plot_feature_importance(top_n=15)
    if fi_fig is not None:
        fi_path = trainer.models_dir / f'feature_importance_{model_name.replace(" ", "_").lower()}.png'
        fi_fig.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close(fi_fig)
        tracker.log_artifact(str(fi_path), artifact_path='visualizations')
    
    # 3. Per-Class Metrics
    pc_fig = trainer.plot_per_class_metrics(results['y_test'], results['y_pred_test'])
    pc_path = trainer.models_dir / f'per_class_metrics_{model_name.replace(" ", "_").lower()}.png'
    pc_fig.savefig(pc_path, dpi=300, bbox_inches='tight')
    plt.close(pc_fig)
    tracker.log_artifact(str(pc_path), artifact_path='visualizations')
    
    # 4. Prediction Confidence
    conf_fig = trainer.plot_prediction_confidence(results['y_test'], results['y_pred_proba'])
    conf_path = trainer.models_dir / f'prediction_confidence_{model_name.replace(" ", "_").lower()}.png'
    conf_fig.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close(conf_fig)
    tracker.log_artifact(str(conf_path), artifact_path='visualizations')
    
    # 5. Model Performance Comparison
    comp_fig = trainer.plot_model_comparison(
        results['train_accuracy'], 
        test_metrics['accuracy'],
        results['train_f1'],
        test_metrics['f1']
    )
    comp_path = trainer.models_dir / f'model_comparison_{model_name.replace(" ", "_").lower()}.png'
    comp_fig.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close(comp_fig)
    tracker.log_artifact(str(comp_path), artifact_path='visualizations')
    
    print(f"✓ Logged 5 visualizations for {model_name}")
    
    # Save model
    model_path = trainer.models_dir / f'{model_name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(trainer.model, model_path)
    print(f"✓ Model saved: {model_path.name}")
    
    # End MLflow run
    tracker.end_run()
    
    print(f"\n✅ {model_name} Complete: Test Accuracy = {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    
    return test_metrics


def main():
    """Main training pipeline - trains multiple models for comparison"""
    print("=" * 80)
    print("RANK TIER CLASSIFIER - MULTI-MODEL COMPARISON")
    print("=" * 80)
    print("\nTraining 3 models: Random Forest, LightGBM, XGBoost")
    print("All results will be logged to MLflow for easy comparison")
    
    if not SKLEARN_AVAILABLE:
        print("\n❌ scikit-learn not available")
        return
    
    # Check model availability
    available_models = []
    if SKLEARN_AVAILABLE:
        available_models.append("Random Forest")
    if LIGHTGBM_AVAILABLE:
        available_models.append("LightGBM")
    if XGBOOST_AVAILABLE:
        available_models.append("XGBoost")
    
    if len(available_models) == 0:
        print("\n❌ No models available. Install: pip install lightgbm xgboost scikit-learn")
        return
    
    print(f"\n✓ Available models: {', '.join(available_models)}")
    
    # Initialize MLflow tracker
    print("\n[MLflow] Initializing experiment tracking...")
    tracker = MLflowTracker("rank-tier-classification")
    
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
    
    # Store results for final comparison
    all_results = []
    
    # Train Random Forest
    if SKLEARN_AVAILABLE:
        rf_params = {
            'model_type': 'RandomForest',
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'num_features': len(trainer.feature_names),
            'test_size': 0.2,
            'random_state': 42
        }
        rf_metrics = train_model_and_log(
            trainer, "Random Forest", trainer.train_random_forest,
            X_train, y_train, X_test, y_test,
            X_train_scaled, X_test_scaled, tracker, rf_params
        )
        all_results.append(("Random Forest", rf_metrics))
    
    # Train LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'model_type': 'LightGBM',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_data_in_leaf': 20,
            'num_features': len(trainer.feature_names),
            'test_size': 0.2,
            'random_state': 42
        }
        lgb_metrics = train_model_and_log(
            trainer, "LightGBM", trainer.train_lightgbm,
            X_train, y_train, X_test, y_test,
            X_train_scaled, X_test_scaled, tracker, lgb_params
        )
        all_results.append(("LightGBM", lgb_metrics))
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'model_type': 'XGBoost',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_features': len(trainer.feature_names),
            'test_size': 0.2,
            'random_state': 42
        }
        xgb_metrics = train_model_and_log(
            trainer, "XGBoost", trainer.train_xgboost,
            X_train, y_train, X_test, y_test,
            X_train_scaled, X_test_scaled, tracker, xgb_params
        )
        all_results.append(("XGBoost", xgb_metrics))
    
    # Print final comparison
    print("\n" + "=" * 80)
    print("📊 FINAL MODEL COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in all_results:
        print(f"{model_name:<20} {metrics['accuracy']:.4f}      {metrics['precision']:.4f}      "
              f"{metrics['recall']:.4f}      {metrics['f1']:.4f}")
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = model_name
    
    print("-" * 80)
    print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)
    print("\n🎯 View all results in MLflow:")
    print("   1. Go to: http://localhost:5000")
    print("   2. Click on experiment: 'rank-tier-classification'")
    print("   3. Select multiple runs to compare side-by-side")
    print("   4. Click 'Compare' button to see metrics comparison")
    print("\n💡 In MLflow UI:")
    print("   • Check the boxes next to each model run")
    print("   • Click 'Compare' to see side-by-side metrics")
    print("   • View each model's visualizations in the Artifacts tab")


def main_single_model():
    """Legacy function - trains only LightGBM for backward compatibility"""
    print("=" * 80)
    print("RANK TIER CLASSIFIER V2 - WITH ENRICHED FEATURES & MLFLOW TRACKING")
    print("=" * 80)
    print("\nExpected improvement: 53% → 60-65% accuracy")
    print("New features: goldPerMinute, damagePerMinute, visionScorePerMinute,")
    print("              skillshotAccuracy, championPoolSize, roleConsistency")
    
    if not LIGHTGBM_AVAILABLE:
        print("\n❌ LightGBM not available. Install with: pip install lightgbm")
        return
    
    # Initialize MLflow tracker
    print("\n[MLflow] Initializing experiment tracking...")
    tracker = MLflowTracker("rank-tier-classification")
    
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
    
    # Start MLflow run
    tracker.start_run("LightGBM-v2-enriched")
    
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
    train_metrics = trainer.evaluate_model(results['y_test'], results['y_test'], 'Train')
    
    # Prepare hyperparameters
    hyperparams = {
        'model_type': 'LightGBM',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'num_features': len(trainer.feature_names),
        'version': 'v2_enriched',
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Log hyperparameters to MLflow
    tracker.log_params(hyperparams)
    
    # Prepare metrics
    metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'train_accuracy': results['train_accuracy']
    }
    
    # Log metrics to MLflow
    tracker.log_metrics(metrics)
    
    # Generate and log visualizations
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR MLFLOW")
    print("=" * 80)
    
    # Get prediction probabilities for confidence plots
    y_pred_proba = trainer.model.predict_proba(X_test_scaled)
    
    # 1. Confusion Matrix
    cm_fig = trainer.plot_confusion_matrix(results['y_test'], results['y_pred_test'], 'Test')
    cm_path = trainer.models_dir / 'confusion_matrix_v2.png'
    cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(cm_fig)
    tracker.log_artifact(str(cm_path), artifact_path='visualizations')
    print(f"✓ Logged confusion matrix")
    
    # 2. Feature Importance
    fi_fig = trainer.plot_feature_importance(top_n=15)
    if fi_fig is not None:
        fi_path = trainer.models_dir / 'feature_importance_v2.png'
        fi_fig.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close(fi_fig)
        tracker.log_artifact(str(fi_path), artifact_path='visualizations')
        print(f"✓ Logged feature importance")
    
    # 3. Per-Class Performance Metrics
    pc_fig = trainer.plot_per_class_metrics(results['y_test'], results['y_pred_test'])
    pc_path = trainer.models_dir / 'per_class_metrics_v2.png'
    pc_fig.savefig(pc_path, dpi=300, bbox_inches='tight')
    plt.close(pc_fig)
    tracker.log_artifact(str(pc_path), artifact_path='visualizations')
    print(f"✓ Logged per-class metrics")
    
    # 4. Class Distribution
    cd_fig = trainer.plot_class_distribution(y_train, y_test)
    cd_path = trainer.models_dir / 'class_distribution_v2.png'
    cd_fig.savefig(cd_path, dpi=300, bbox_inches='tight')
    plt.close(cd_fig)
    tracker.log_artifact(str(cd_path), artifact_path='visualizations')
    print(f"✓ Logged class distribution")
    
    # 5. Prediction Confidence
    conf_fig = trainer.plot_prediction_confidence(results['y_test'], y_pred_proba)
    conf_path = trainer.models_dir / 'prediction_confidence_v2.png'
    conf_fig.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close(conf_fig)
    tracker.log_artifact(str(conf_path), artifact_path='visualizations')
    print(f"✓ Logged prediction confidence")
    
    # 6. Model Performance Comparison (Train vs Test)
    comp_fig = trainer.plot_model_comparison(
        results['train_accuracy'], 
        test_metrics['accuracy'],
        results.get('train_f1', results['train_accuracy']),  # Approximate if not available
        test_metrics['f1']
    )
    comp_path = trainer.models_dir / 'model_comparison_v2.png'
    comp_fig.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close(comp_fig)
    tracker.log_artifact(str(comp_path), artifact_path='visualizations')
    print(f"✓ Logged model performance comparison")
    
    print(f"\n📊 Total visualizations generated: 6")
    print(f"   • Confusion Matrix")
    print(f"   • Feature Importance (Top 15)")
    print(f"   • Per-Class Metrics (Precision, Recall, F1)")
    print(f"   • Class Distribution (Train vs Test)")
    print(f"   • Prediction Confidence Analysis")
    print(f"   • Train vs Test Performance")
    
    # 7. Log model metadata and class distribution
    metadata = {
        'model_type': 'LightGBM',
        'features': trainer.feature_names,
        'class_names': trainer.class_names,
        'version': 'v2_enriched',
        'accuracy': float(test_metrics['accuracy']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'f1_score': float(test_metrics['f1']),
        'class_counts': {
            'Low': int(np.sum(y == 0)),
            'Mid': int(np.sum(y == 1)),
            'High': int(np.sum(y == 2)),
            'Elite': int(np.sum(y == 3))
        },
        'data_shape': {'samples': int(X.shape[0]), 'features': int(X.shape[1])},
        'train_test_split': {'train': int(len(X_train)), 'test': int(len(X_test))}
    }
    tracker.log_dict(metadata, file_name='model_metadata_v2.json')
    print(f"✓ Logged metadata")
    
    # Save model locally
    print("\n" + "=" * 80)
    print("SAVING MODEL & ARTIFACTS")
    print("=" * 80)
    
    model_path = trainer.models_dir / 'rank_tier_model_v2_enriched.pkl'
    scaler_path = trainer.models_dir / 'scaler_v2_enriched.pkl'
    metadata_path = trainer.models_dir / 'metadata_v2_enriched.json'
    
    joblib.dump(trainer.model, model_path)
    joblib.dump(trainer.scaler, scaler_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    # End MLflow run
    tracker.end_run()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    print("\nComparison:")
    print("  V1 (Original): 53.11% accuracy")
    print(f"  V2 (Enriched): {test_metrics['accuracy']*100:.2f}% accuracy")
    print(f"  Improvement: {(test_metrics['accuracy'] - 0.5311)*100:+.2f} percentage points")
    print("\n🎯 View results in MLflow:")
    print("   1. Run: mlflow ui --port 5000")
    print("   2. Visit: http://localhost:5000")
    print("   3. Go to experiment: 'rank-tier-classification'")
    print("   4. Click on run: 'LightGBM-v2-enriched'")

if __name__ == '__main__':
    main()
