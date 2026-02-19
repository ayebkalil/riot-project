# ML Models - Training Pipeline

This directory contains all model training scripts for the League of Legends ML project.

## Project Structure

```
models/
├── shared/                          # Shared utilities
│   ├── mlflow_utils.py             # MLflow logging helpers
│   ├── data_loader.py              # Data loading utilities
│   ├── visualization.py            # Plotting utilities
│   └── __init__.py
│
├── 1_rank_tier_classifier/         # Model 1: Tier Classification
│   ├── train_rank_tier.py          # Training script
│   ├── README.md                   # Documentation
│   └── models/                     # Saved models (auto-created)
│
├── 2_progression_regressor/        # Model 2: Win Rate Regression
│   ├── train_progression.py        # Training script
│   ├── README.md                   # Documentation
│   └── models/                     # Saved models (auto-created)
│
├── 3_smurf_anomaly_detector/       # Model 3: Anomaly Detection
│   ├── train_smurf_anomaly.py      # Training script
│   ├── README.md                   # Documentation
│   └── models/                     # Saved models (auto-created)
│
└── 4_match_outcome_predictor/      # Model 4: Match Winner Prediction
    ├── train_match_outcome.py      # Training script
    ├── README.md                   # Documentation
    └── models/                     # Saved models (auto-created)
```

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Setup MLflow

```bash
# Initialize experiments (run ONCE)
python mlflow_setup.py

# Start MLflow server
mlflow ui --port 5000
```

Visit `http://localhost:5000` to see the MLflow dashboard.

### 3. Train Individual Models

Each model can be trained independently by different team members:

```bash
# Model 1: Rank Tier Classifier (9-class)
python models/1_rank_tier_classifier/train_rank_tier.py

# Model 2: Progression Regressor (continuous)
python models/2_progression_regressor/train_progression.py

# Model 3: Smurf Anomaly Detector (unsupervised)
python models/3_smurf_anomaly_detector/train_smurf_anomaly.py

# Model 4: Match Outcome Predictor (binary)
python models/4_match_outcome_predictor/train_match_outcome.py
```

### 4. Compare Models

After training, compare all models:

```bash
python model_comparison.py
```

This generates:
- Comparison statistics
- Visualization plots
- Text report (`MODEL_COMPARISON_REPORT.txt`)

## Team Collaboration

Each model folder is **independent** - team members can work in parallel:

| Team Member | Working On | Script |
|---|---|---|
| Person 1 | Tier Classification | `1_rank_tier_classifier/train_rank_tier.py` |
| Person 2 | Progression Regression | `2_progression_regressor/train_progression.py` |
| Person 3 | Smurf Anomaly Detection | `3_smurf_anomaly_detector/train_smurf_anomaly.py` |
| Person 4 | Match Outcome Prediction | `4_match_outcome_predictor/train_match_outcome.py` |

All runs automatically log to the **same MLflow server** for centralized tracking.

## MLflow Experiments

Each model has its own experiment:

| Experiment Name | Task | Models |
|---|---|---|
| `rank-tier-classification` | 9-class classification | RandomForest (v1, v2, v3) |
| `progression-regression` | Continuous regression | Linear, Ridge, RandomForest |
| `smurf-anomaly-detection` | Unsupervised anomaly detection | IsolationForest, EllipticEnvelope, LOF |
| `match-outcome-prediction` | Binary classification | LogisticRegression, RandomForest, XGBoost (v1, v2) |

## Output Structure

After training, each model folder contains:

```
models/X_*/models/
├── rank_tier_model.pkl (or equivalent)  # Trained model
├── scaler.pkl                          # Feature scaler
└── metadata.json                       # Model metadata
```

## Customization

### Modify Hyperparameters

Edit the training script in each folder:

```python
# Example: train_rank_tier.py
classifier.train_random_forest(
    run_name="RandomForest-v4",
    n_estimators=200,      # Change these
    max_depth=20,
    min_samples_split=5
)
```

### Add New Algorithms

Each training script is modular. Add methods like:

```python
def train_gradient_boosting(self, run_name: str, **kwargs):
    # Implement training logic
    pass
```

### Adjust Data Split

Modify in `train_*.py`:

```python
train_test_split(X, y, test_size=0.15)  # Change from 0.2 to 0.15
```

## Troubleshooting

### MLflow Server Not Running
```bash
mlflow ui --port 5000
```

### Data Files Not Found
Ensure `data/processed/` contains:
- `rank_features.csv`
- `progression_features.csv`
- `smurf_features.csv`
- `match_features.csv`

### Import Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Permission Denied (Windows)
```
python -m train_rank_tier  # Or use: py train_rank_tier.py
```

## Next Phase: FastAPI Backend

After training, use the saved models in FastAPI:

```python
# FastAPI will load models from:
rank_model = joblib.load('models/1_rank_tier_classifier/models/rank_tier_model.pkl')
progression_model = joblib.load('models/2_progression_regressor/models/progression_model.pkl')
smurf_model = joblib.load('models/3_smurf_anomaly_detector/models/smurf_anomaly_model.pkl')
match_model = joblib.load('models/4_match_outcome_predictor/models/match_outcome_model.pkl')
```

## Timeline

- **Day 1-2**: Team members train their assigned models
- **Day 3**: Run `model_comparison.py` to evaluate
- **Day 4+**: Select best models, prepare FastAPI integration

## Questions?

Check each model folder's `README.md` for detailed documentation.
