# Progression Regressor

**Task**: Regression (predict player win rate delta)

**Dataset**: `progression_features.csv` (4,128 samples, 14 features)

**Target Variable**: `delta_winrate` (change in win rate from early → late career)

## Quick Start

```bash
# Run training script
python train_progression.py
```

## What This Does

1. **Loads Data**: Fetches `progression_features.csv` from `data/processed/`
2. **Splits Data**: 80/20 train-test split
3. **Scales Features**: StandardScaler normalization
4. **Trains Models**: Linear, Ridge, and RandomForest regressors
5. **Logs to MLflow**: All runs, metrics, and artifacts
6. **Saves Model**: `models/progression_model.pkl` + metadata

## Features

- Performance deltas (change in KDA, CS, gold, damage)
- Streak indicators (win streaks, LP trends)
- Consistency metrics (Z-scores, entropy)
- (14 total features)

## MLflow Tracking

All runs logged to experiment: **`progression-regression`**

View results:
```bash
mlflow ui --port 5000
# Then visit: http://localhost:5000
```

## Adjust Model Training

Edit `train_progression.py` to add new model variations:

```python
# Try different algorithms
regressor.train_model(
    model_type="ridge",
    run_name="RidgeRegression-v3",
    alpha=0.5
)
```

## Output Files

```
models/
├── progression_model.pkl    # Trained model
├── scaler.pkl               # Feature scaler
└── metadata.json            # Feature names & config
```

## Next Steps

1. Run this script to train the model
2. Track metrics in MLflow UI (focus on R² for model quality)
3. Use `model_comparison.py` to compare across all 4 models
4. Export best model to FastAPI backend
