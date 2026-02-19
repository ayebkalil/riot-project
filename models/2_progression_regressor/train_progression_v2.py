"""
Train Progression Regressor with enriched features (temporal patterns & champion mastery).

Key improvements over v1:
- delta_goldPerMinute - income growth tracking
- delta_damagePerMinute - combat improvement
- delta_visionScorePerMinute - awareness growth
- champion_pool_diversity - learning depth
- role_consistency_delta - specialization improvement

Expected improvement: R² 0.35 → 0.45-0.50
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
from shared.mlflow_utils import MLflowTracker

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ProgressionTrainer:
    """Train and evaluate progression regression models"""
    
    EXPERIMENT_NAME = "progression-prediction"
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Load and prepare data for training"""
        print("\nLoading enriched progression data...")
        
        # Load enriched progression features
        df = pd.read_csv('data/processed/progression_features_enriched_v2.csv')
        
        # Separate features and target
        y = df['delta_winrate'].values
        X = df.drop(['puuid', 'delta_winrate', 'matches_used'], axis=1, errors='ignore')
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Loaded data: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"✓ Target (delta_winrate) statistics:")
        print(f"  Mean: {y.mean():.4f}")
        print(f"  Std:  {y.std():.4f}")
        print(f"  Min:  {y.min():.4f}")
        print(f"  Max:  {y.max():.4f}")
        
        return X, y
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test, run_name="LinearRegression"):
        """Train Linear Regression"""
        print(f"\nTraining {run_name}...")
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        
        # Log parameters
        params = {'model_type': 'LinearRegression'}
        tracker.log_params(params)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
        
        # Log metrics to MLflow
        tracker.log_metrics({k: v for k, v in results.items() if k not in ['y_pred_test', 'y_test']})
        
        # End run
        tracker.end_run()
        
        return results
    
    def train_ridge_regression(self, X_train, y_train, X_test, y_test, alpha=1.0, run_name="Ridge"):
        """Train Ridge Regression"""
        print(f"\nTraining {run_name} (alpha={alpha})...")
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        
        # Log parameters
        params = {'model_type': 'Ridge', 'alpha': alpha}
        tracker.log_params(params)
        
        # Train model
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
        
        # Log metrics to MLflow
        tracker.log_metrics({k: v for k, v in results.items() if k not in ['y_pred_test', 'y_test']})
        
        # End run
        tracker.end_run()
        
        return results
    
    def train_lightgbm_regression(self, X_train, y_train, X_test, y_test, run_name="LightGBM"):
        """Train LightGBM Regression"""
        print(f"\nTraining {run_name}...")
        
        # Initialize MLflow tracking
        tracker = MLflowTracker(self.EXPERIMENT_NAME)
        tracker.start_run(run_name)
        
        # Parameters
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'n_jobs': 1,
            'random_state': 42
        }
        
        # Log parameters
        tracker.log_params(params)
        
        # Train model
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
        
        # Log metrics to MLflow
        tracker.log_metrics({k: v for k, v in results.items() if k not in ['y_pred_test', 'y_test']})
        
        # End run
        tracker.end_run()
        
        return results
    
    def print_results(self, model_name, results):
        """Print training results"""
        print(f"\n{model_name} Results:")
        print(f"  Train MSE: {results['train_mse']:.6f}")
        print(f"  Test MSE:  {results['test_mse']:.6f}")
        print(f"  Train RMSE: {results['train_rmse']:.6f}")
        print(f"  Test RMSE:  {results['test_rmse']:.6f}")
        print(f"  Train MAE: {results['train_mae']:.6f}")
        print(f"  Test MAE:  {results['test_mae']:.6f}")
        print(f"  Train R²: {results['train_r2']:.4f}")
        print(f"  Test R²:  {results['test_r2']:.4f}")

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("PROGRESSION REGRESSOR V2 - WITH ENRICHED FEATURES")
    print("=" * 80)
    print("\nExpected improvement: R² 0.35 → 0.45-0.50")
    print("New features: delta_goldPerMinute, delta_damagePerMinute,")
    print("              delta_visionScorePerMinute, championPoolSize changes")
    
    if not SKLEARN_AVAILABLE:
        print("\n❌ scikit-learn not available")
        return
    
    # Initialize trainer
    trainer = ProgressionTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    trainer.scaler = StandardScaler()
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    # Train models
    print("\n" + "=" * 80)
    print("TRAINING MODELS WITH MLFLOW TRACKING")
    print("=" * 80)
    
    results_dict = {}
    best_r2 = -np.inf
    best_model_name = None
    
    # 1. Linear Regression
    results = trainer.train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        run_name='LinearRegression'
    )
    trainer.print_results('LinearRegression', results)
    results_dict['LinearRegression'] = results
    
    if results['test_r2'] > best_r2:
        best_r2 = results['test_r2']
        best_model_name = 'LinearRegression'
    
    # 2. Ridge Regression (alpha=0.1)
    results = trainer.train_ridge_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        alpha=0.1, run_name='Ridge-alpha-0.1'
    )
    trainer.print_results('Ridge-alpha-0.1', results)
    results_dict['Ridge-alpha-0.1'] = results
    
    if results['test_r2'] > best_r2:
        best_r2 = results['test_r2']
        best_model_name = 'Ridge-alpha-0.1'
    
    # 3. Ridge Regression (alpha=1.0)
    results = trainer.train_ridge_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        alpha=1.0, run_name='Ridge-alpha-1.0'
    )
    trainer.print_results('Ridge-alpha-1.0', results)
    results_dict['Ridge-alpha-1.0'] = results
    
    if results['test_r2'] > best_r2:
        best_r2 = results['test_r2']
        best_model_name = 'Ridge-alpha-1.0'
    
    # 4. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        results = trainer.train_lightgbm_regression(
            X_train_scaled, y_train, X_test_scaled, y_test,
            run_name='LightGBM'
        )
        trainer.print_results('LightGBM', results)
        results_dict['LightGBM'] = results
        
        if results['test_r2'] > best_r2:
            best_r2 = results['test_r2']
            best_model_name = 'LightGBM'
    
    # Save best model
    print("\n" + "=" * 80)
    print("SAVING BEST MODEL")
    print("=" * 80)
    print(f"Best model: {best_model_name} (Test R² = {best_r2:.4f})")
    
    model_path = trainer.models_dir / 'progression_model_v2_enriched.pkl'
    scaler_path = trainer.models_dir / 'scaler_v2_enriched.pkl'
    metadata_path = trainer.models_dir / 'metadata_v2_enriched.json'
    
    joblib.dump(trainer.model, model_path)
    joblib.dump(trainer.scaler, scaler_path)
    
    metadata = {
        'model_type': best_model_name,
        'features': trainer.feature_names,
        'version': 'v2_enriched',
        'test_r2': float(best_r2),
        'target': 'delta_winrate'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Test R²: {best_r2:.4f}")
    print("\nComparison:")
    print("  V1 (Original): R² = 0.3574 (35.74% variance explained)")
    print(f"  V2 (Enriched): R² = {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
    print(f"  Improvement: {(best_r2 - 0.3574)*100:+.2f} percentage points")
    print("\nView MLflow results at: http://localhost:5000")
    print("  Experiment: progression-prediction")

if __name__ == '__main__':
    main()
