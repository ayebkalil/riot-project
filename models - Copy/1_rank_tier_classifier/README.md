# Rank Tier Classifier

**Task**: 9-class classification (Iron → Challenger)

**Dataset**: `rank_features.csv` (4,340 samples, 17 features)

**Target Variable**: `tier` (player's competitive rank)

## Quick Start

```bash
# Run training script
python train_rank_tier.py
```

## What This Does

1. **Loads Data**: Fetches `rank_features.csv` from `data/processed/`
2. **Splits Data**: 80/20 train-test split with stratification
3. **Scales Features**: StandardScaler normalization
4. **Trains Models**: Multiple RandomForest variations
5. **Logs to MLflow**: All runs, metrics, and artifacts
6. **Saves Model**: `models/rank_tier_model.pkl` + metadata

## Features

- Per-match player statistics (KDA, CS/min, Gold/min, etc.)
- Performance indicators (win rate, kill participation, etc.)
- (17 total features)

## MLflow Tracking

All runs logged to experiment: **`rank-tier-classification`**

View results:
```bash
mlflow ui --port 5000
# Then visit: http://localhost:5000
```

## Modify Hyperparameters

Edit `train_rank_tier.py` in the `train_random_forest()` calls:

```python
classifier.train_random_forest(
    run_name="RandomForest-v4",
    n_estimators=200,      # More trees
    max_depth=20,          # Deeper trees
    min_samples_split=10   # More conservative splits
)
```

## Output Files

```
models/
├── rank_tier_model.pkl      # Trained model
├── scaler.pkl               # Feature scaler
└── metadata.json            # Feature names & config
```

## Next Steps

1. Run this script to train the model
2. Track metrics in MLflow UI
3. Use `model_comparison.py` to compare across all 4 models
4. Export best model to FastAPI backend
