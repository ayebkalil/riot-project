"""
MASTER TRAINING SCRIPT - Train All 6 Models for Professor Demo

This script trains all 6 models with MLflow tracking:
1. Rank Tier Classifier (4-class classification)
2. Progression Predictor (regression)
3. Smurf Anomaly Detector (unsupervised)
4. Match Outcome Predictor - Post-Game Features (binary classification)
5. Match Outcome Predictor - Early-Game Features (binary classification)
6. Match Outcome Predictor - Cascade (Early -> Post stacked model)

Each model has its own MLflow experiment for independent comparison.

Usage:
    python train_all_models.py

View results:
    mlflow ui --port 5000
    Open: http://localhost:5000
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))


def print_header(title: str, width: int = 80):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def train_model_1_rank_tier():
    """Train Rank Tier Classifier"""
    print_header("MODEL 1: RANK TIER CLASSIFIER")
    print("Task: Multi-class classification (Low/Mid/High/Elite tiers)")
    print("Models: Random Forest, LightGBM, XGBoost")
    print("Experiment: rank-tier-classification\n")
    
    start = time.time()
    
    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '1_rank_tier_classifier'))
        from train_rank_tier_v2 import main as train_rank_tier
        train_rank_tier()
        
        elapsed = time.time() - start
        print(f"\n✓ Model 1 completed in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"\n✗ Model 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model_2_progression():
    """Train Progression Predictor"""
    print_header("MODEL 2: PROGRESSION PREDICTOR")
    print("Task: Regression (predict delta_winrate)")
    print("Models: LinearRegression, Ridge, LightGBM")
    print("Experiment: progression-prediction\n")
    
    start = time.time()
    
    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '2_progression_regressor'))
        from train_progression_v2 import main as train_progression
        train_progression()
        
        elapsed = time.time() - start
        print(f"\n✓ Model 2 completed in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"\n✗ Model 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model_3_smurf():
    """Train Smurf Anomaly Detector"""
    print_header("MODEL 3: SMURF ANOMALY DETECTOR")
    print("Task: Unsupervised anomaly detection")
    print("Models: IsolationForest, EllipticEnvelope, LocalOutlierFactor")
    print("Experiment: smurf-anomaly-detection\n")
    
    start = time.time()
    
    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '3_smurf_anomaly_detector'))
        from train_smurf_anomaly import main as train_smurf
        train_smurf()
        
        elapsed = time.time() - start
        print(f"\n✓ Model 3 completed in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"\n✗ Model 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model_4_match_outcome_post_game():
    """Train Match Outcome Predictor (Post-Game Features)"""
    print_header("MODEL 4: MATCH OUTCOME PREDICTOR (POST-GAME)")
    print("Task: Binary classification (win/loss)")
    print("Feature mode: full (post-game aggregates)")
    print("Models: LogisticRegression, RandomForest, XGBoost")
    print("Experiment: match-outcome-prediction\n")
    
    start = time.time()
    
    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '4_match_outcome_predictor'))
        from train_match_outcome import main as train_match_outcome
        train_match_outcome(['--mode', 'full'])
        
        elapsed = time.time() - start
        print(f"\n✓ Model 4 completed in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"\n✗ Model 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model_5_match_outcome_early(minute: int = 15):
    """Train Match Outcome Predictor (Early-Game Features)"""
    print_header(f"MODEL 5: MATCH OUTCOME PREDICTOR (EARLY-{minute}M)")
    print("Task: Binary classification (win/loss)")
    print(f"Feature mode: early-game timeline snapshot at {minute}m")
    print("Models: LogisticRegression, RandomForest, XGBoost")
    print("Experiment: match-outcome-prediction\n")

    start = time.time()

    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '4_match_outcome_predictor'))
        from train_match_outcome import main as train_match_outcome
        train_match_outcome(['--mode', 'early', '--minute', str(minute)])

        elapsed = time.time() - start
        print(f"\n✓ Model 5 completed in {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"\n✗ Model 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model_6_match_outcome_cascade():
    """Train Match Outcome Predictor (Cascade: Early -> Post)"""
    print_header("MODEL 6: MATCH OUTCOME PREDICTOR (CASCADE)")
    print("Task: Two-stage stacked classification (early -> post)")
    print("Feature mode: cascade (stage1 early probability passed to stage2)")
    print("Models: LogisticRegression (stage1), RandomForest (stage2)")
    print("Experiment: match-outcome-prediction\n")

    start = time.time()

    try:
        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / 'models' / '4_match_outcome_predictor'))
        from train_match_outcome import main as train_match_outcome
        train_match_outcome(['--mode', 'cascade'])

        elapsed = time.time() - start
        print(f"\n✓ Model 6 completed in {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"\n✗ Model 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: dict, total_time: float):
    """Print training summary"""
    print_header("TRAINING SUMMARY")
    
    print("Model Results:")
    print("-" * 80)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name:<50} {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    print("-" * 80)
    print(f"\nTotal: {successful}/{total} models trained successfully")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    print("\n" + "=" * 80)
    print("VIEW RESULTS IN MLFLOW")
    print("=" * 80)
    print("\n1. Start MLflow UI (if not already running):")
    print("   mlflow ui --port 5000")
    print("\n2. Open in browser:")
    print("   http://localhost:5000")
    print("\n3. View experiments:")
    print("   • rank-tier-classification")
    print("   • progression-prediction")
    print("   • smurf-anomaly-detection")
    print("   • match-outcome-prediction")
    print("\n   (match-outcome-prediction now includes post-game and early-game run groups)")
    print("\n4. Compare models within each experiment:")
    print("   • Select multiple runs")
    print("   • Click 'Compare' button")
    print("   • View metrics, parameters, and artifacts")
    print("\n" + "=" * 80)


def main():
    """Main training pipeline"""
    
    print_header("RIOT GAMES ML PROJECT - COMPLETE TRAINING PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training all 6 models with MLflow tracking...\n")
    
    overall_start = time.time()
    results = {}
    
    # Train Model 1: Rank Tier Classifier
    results["Model 1: Rank Tier Classifier"] = train_model_1_rank_tier()
    
    # Train Model 2: Progression Predictor
    results["Model 2: Progression Predictor"] = train_model_2_progression()
    
    # Train Model 3: Smurf Anomaly Detector
    results["Model 3: Smurf Anomaly Detector"] = train_model_3_smurf()
    
    # Train Model 4: Match Outcome Predictor (Post-Game)
    results["Model 4: Match Outcome Predictor (Post-Game)"] = train_model_4_match_outcome_post_game()

    # Train Model 5: Match Outcome Predictor (Early-Game)
    results["Model 5: Match Outcome Predictor (Early-Game 15m)"] = train_model_5_match_outcome_early(minute=15)

    # Train Model 6: Match Outcome Predictor (Cascade)
    results["Model 6: Match Outcome Predictor (Cascade)"] = train_model_6_match_outcome_cascade()
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print_summary(results, overall_time)
    
    # Exit with error code if any model failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
