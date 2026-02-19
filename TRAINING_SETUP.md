# MLflow & Model Training Setup - Quick Guide

## Overview

This project trains 4 separate ML models:
1. **Rank Tier Classifier** - 9-class classification
2. **Progression Regressor** - Continuous regression
3. **Smurf Anomaly Detector** - Unsupervised anomaly detection
4. **Match Outcome Predictor** - Binary classification

Each model is trained independently with **MLflow tracking**.

## One-Time Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize MLflow Experiments
```bash
python mlflow_setup.py
```

### Step 3: Start MLflow Server
```bash
mlflow ui --port 5000
```

Then visit: **http://localhost:5000**

## Training Models (Parallel)

Each team member can train independently:

### Person 1: Rank Tier Classifier
```bash
python models/1_rank_tier_classifier/train_rank_tier.py
```

### Person 2: Progression Regressor
```bash
python models/2_progression_regressor/train_progression.py
```

### Person 3: Smurf Anomaly Detector
```bash
python models/3_smurf_anomaly_detector/train_smurf_anomaly.py
```

### Person 4: Match Outcome Predictor
```bash
python models/4_match_outcome_predictor/train_match_outcome.py
```

**All runs will appear in MLflow UI automatically!**

## After Training: Compare Models

```bash
python model_comparison.py
```

This generates:
- Terminal output with comparison stats
- `model_comparison.png` - visualization
- `MODEL_COMPARISON_REPORT.txt` - detailed report

## Output Structure

Each trained model folder contains:

```
models/X_*/models/
├── model_name.pkl           # Trained model
├── scaler.pkl               # Feature scaler
└── metadata.json            # Configuration
```

These files are ready for FastAPI backend integration.

## Check Training Progress

Open **http://localhost:5000** in browser to see:
- All 4 experiments
- Individual runs for each model
- Metrics and hyperparameters
- Training artifacts

## Common Commands

```bash
# View logs from one model
tail -f logs/model_training.log

# Test one model
python models/1_rank_tier_classifier/train_rank_tier.py --test

# Compare specific metric
mlflow search-runs --experiment-name "rank-tier-classification"
```

## Next Steps

1. Install dependencies (Step 1 above)
2. Init MLflow (Step 2)
3. Start server (Step 3)
4. Split work and run training scripts
5. Compare models
6. Export best models to FastAPI backend

## File Structure

```
Riot Games Project/
├── mlflow_setup.py              # Initialize experiments
├── model_comparison.py          # Compare all models
├── requirements.txt             # Dependencies
│
├── models/
│   ├── README.md               # Full documentation
│   ├── shared/                 # Shared utilities
│   │   ├── mlflow_utils.py     # MLflow helpers
│   │   ├── data_loader.py      # Data loading
│   │   └── visualization.py    # Plotting
│   │
│   ├── 1_rank_tier_classifier/
│   │   ├── train_rank_tier.py
│   │   ├── README.md
│   │   └── models/             # Saved models
│   │
│   ├── 2_progression_regressor/
│   │   ├── train_progression.py
│   │   ├── README.md
│   │   └── models/
│   │
│   ├── 3_smurf_anomaly_detector/
│   │   ├── train_smurf_anomaly.py
│   │   ├── README.md
│   │   └── models/
│   │
│   └── 4_match_outcome_predictor/
│       ├── train_match_outcome.py
│       ├── README.md
│       └── models/
│
└── data/processed/
    ├── rank_features.csv
    ├── progression_features.csv
    ├── smurf_features.csv
    └── match_features.csv
```

## Troubleshooting

**Q: Models folder doesn't exist?**
```bash
mkdir models  # Already created by setup
```

**Q: MLflow UI shows no experiments?**
```bash
# Run setup again to create experiments
python mlflow_setup.py

# Then start server
mlflow ui --port 5000
```

**Q: Import errors in training scripts?**
```bash
# Reinstall dependencies
pip install -r requirements.txt -U
```

**Q: Port 5000 already in use?**
```bash
mlflow ui --port 5001  # Use different port
```

## Example: Adding Custom Hyperparameters

Edit `models/1_rank_tier_classifier/train_rank_tier.py`:

```python
# In main():
classifier.train_random_forest(
    run_name="MyCustomModel-v1",
    n_estimators=250,           # Add new variation
    max_depth=18,
    min_samples_split=8
)
```

Then run: `python models/1_rank_tier_classifier/train_rank_tier.py`

New run will appear in MLflow UI!

---

**Ready to start?** Follow the "One-Time Setup" section above! 🚀
