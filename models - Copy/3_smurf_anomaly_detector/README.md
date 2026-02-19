# Smurf Anomaly Detector

**Task**: Unsupervised anomaly detection (identify suspicious accounts)

**Dataset**: `smurf_features.csv` (4,340 samples, 16 features)

**Target Variable**: None (unsupervised - anomaly scores)

## Quick Start

```bash
# Run training script
python train_smurf_anomaly.py
```

## What This Does

1. **Loads Data**: Fetches `smurf_features.csv` from `data/processed/`
2. **Scales Features**: StandardScaler normalization
3. **Trains Models**: IsolationForest, EllipticEnvelope, LocalOutlierFactor
4. **Logs to MLflow**: All runs, anomaly statistics, and artifacts
5. **Saves Model**: `models/smurf_anomaly_model.pkl` + metadata

## Features

- Z-score normalized statistics (winrate_zscore, kda_zscore)
- Playstyle metrics (damage share, gold share)
- Diversity metrics (champion mastery entropy)
- (16 total features)

## Anomaly Detection Methods

### IsolationForest (recommended)
- Fast and efficient
- Works well with high-dimensional data
- Contamination parameter controls sensitivity

### EllipticEnvelope
- Assumes Gaussian distribution
- Good for multivariate outlier detection

### LocalOutlierFactor
- Density-based approach
- Detects local outliers

## MLflow Tracking

All runs logged to experiment: **`smurf-anomaly-detection`**

View results:
```bash
mlflow ui --port 5000
# Then visit: http://localhost:5000
```

## Adjust Contamination Rate

Modify the contamination parameter to control how many samples are marked as anomalies:

```python
detector.train_isolation_forest(
    run_name="IsolationForest-v3",
    contamination=0.15  # 15% of samples marked as anomalies
)
```

## Output Files

```
models/
├── smurf_anomaly_model.pkl  # Trained model
├── scaler.pkl               # Feature scaler
└── metadata.json            # Feature names & config
```

## Interpreting Results

- **n_anomalies**: Number of suspicious accounts detected
- **anomaly_ratio**: Percentage of population flagged
- **anomaly_scores**: Lower scores = more anomalous

## Next Steps

1. Run this script to train the model
2. Track anomaly detection statistics in MLflow UI
3. Use `model_comparison.py` to compare anomaly detection methods
4. Export best model to FastAPI backend
