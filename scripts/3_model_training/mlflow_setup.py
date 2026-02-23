"""
MLflow Setup Script
Run this ONCE to initialize MLflow experiments for all 4 models
"""

import sys
from pathlib import Path

# Add models to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.shared.mlflow_utils import MLflowTracker


def setup_mlflow():
    """Create all experiments in MLflow"""
    
    experiments = [
        {
            "name": "rank-tier-classification",
            "description": "9-class tier classification using player rank features"
        },
        {
            "name": "progression-regression",
            "description": "Regression task - predict player win rate delta (progression)"
        },
        {
            "name": "smurf-anomaly-detection",
            "description": "Anomaly detection using IsolationForest for smurf detection"
        },
        {
            "name": "match-outcome-prediction",
            "description": "Binary classification - predict match winner from team differentials"
        }
    ]
    
    print("\n" + "="*70)
    print("MLFLOW SETUP - INITIALIZING EXPERIMENTS")
    print("="*70 + "\n")
    
    for exp in experiments:
        try:
            tracker = MLflowTracker(exp["name"])
            print(f"✓ Experiment initialized: {exp['name']}")
            print(f"  Description: {exp['description']}\n")
        except Exception as e:
            print(f"✗ Error initializing {exp['name']}: {e}\n")
    
    print("="*70)
    print("✓ MLFLOW SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Each team member can now run their training script independently")
    print("2. All runs will be logged to the same MLflow server")
    print("3. View results at: http://localhost:5000")
    print("\nExample usage in training scripts:")
    print("  from models.shared import MLflowTracker")
    print("  tracker = MLflowTracker('rank-tier-classification')")
    print("  tracker.start_run('RandomForest-v1')")
    print("  tracker.log_params(params)")
    print("  tracker.log_metrics(metrics)")
    print("  tracker.end_run()")


if __name__ == "__main__":
    setup_mlflow()
