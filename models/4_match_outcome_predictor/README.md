# Match Outcome Predictor

**Task**: Binary classification (predict match winner)

**Dataset**: `match_features.csv` (306,312 samples, 15 features)

**Target Variable**: `team_win` (0=loss, 1=win)

## Quick Start

```bash
# Run training script
python train_match_outcome.py
```

## What This Does

1. **Loads Data**: Fetches `match_features.csv` from `data/processed/`
2. **Splits Data**: 80/20 train-test split with stratification
3. **Scales Features**: StandardScaler normalization
4. **Trains Models**: LogisticRegression, RandomForest, XGBoost
5. **Logs to MLflow**: All runs, metrics, and artifacts
6. **Saves Model**: `models/match_outcome_model.pkl` + metadata

## Features

All are TEAM DIFFERENTIALS (Team A stat - Team B stat):
- gold_diff, damage_diff, vision_diff
- kda_diff (kills - deaths - assists), cs_diff
- *_per_min_diff for rate-based features
- first_blood_diff, first_tower_diff, first_dragon_diff
- (15 total features)

## Models Included

### LogisticRegression (baseline)
- Fast, interpretable
- Good baseline for comparison

### RandomForest
- Non-linear relationships
- Feature importance
- Robust to outliers

### XGBoost (recommended)
- Gradient boosting
- Often best performance
- Fast training with large datasets

## MLflow Tracking

All runs logged to experiment: **`match-outcome-prediction`**

View results:
```bash
mlflow ui --port 5000
# Then visit: http://localhost:5000
```

## Add More Models

Edit `train_match_outcome.py` to add new algorithms:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Add to main():
predictor.train_model(
    model_type="gradient_boosting",
    run_name="GradientBoosting-v1"
)
```

## Output Files

```
models/
├── match_outcome_model.pkl  # Trained model
├── scaler.pkl               # Feature scaler
└── metadata.json            # Feature names & config
```

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **F1 Score**: Balance between precision & recall
- **Precision**: False positive rate
- **Recall**: False negative rate
- **ROC-AUC**: Model discrimination ability

## Class Balance

Dataset is **perfectly balanced** (50% win, 50% loss):
- This is ideal for binary classification
- No need for class weights adjustment

## Next Steps

1. Run this script to train the model (largest dataset, may take time)
2. Track metrics in MLflow UI (focus on ROC-AUC and F1)
3. Use `model_comparison.py` to compare across all 4 models
4. Export best model to FastAPI backend
