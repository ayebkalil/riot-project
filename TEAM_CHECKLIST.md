# Team Checklist: Model Training & MLflow Setup

## ✅ Setup Phase (Do ONCE, One Person)

- [ ] **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```
  
- [ ] **Initialize MLflow Experiments**
  ```bash
  python mlflow_setup.py
  ```
  
- [ ] **Start MLflow Server**
  ```bash
  mlflow ui --port 5000
  ```
  Then visit: http://localhost:5000

---

## ✅ Training Phase (Parallel - Each Person)

### Person 1: Rank Tier Classifier

- [ ] Review instructions
  ```bash
  cat models/1_rank_tier_classifier/README.md
  ```

- [ ] Run training
  ```bash
  python models/1_rank_tier_classifier/train_rank_tier.py
  ```

- [ ] Check MLflow results
  - Navigate to http://localhost:5000
  - Experiment: `rank-tier-classification`
  - Look for 3 runs (RandomForest v1, v2, v3)

- [ ] Verify output files
  ```bash
  ls models/1_rank_tier_classifier/models/
  # Should have: rank_tier_model.pkl, scaler.pkl, metadata.json
  ```

### Person 2: Progression Regressor

- [ ] Review instructions
  ```bash
  cat models/2_progression_regressor/README.md
  ```

- [ ] Run training
  ```bash
  python models/2_progression_regressor/train_progression.py
  ```

- [ ] Check MLflow results
  - Experiment: `progression-regression`
  - Look for 4 runs (Linear, Ridge v1, v2, RandomForest)

- [ ] Verify output files
  ```bash
  ls models/2_progression_regressor/models/
  # Should have: progression_model.pkl, scaler.pkl, metadata.json
  ```

### Person 3: Smurf Anomaly Detector

- [ ] Review instructions
  ```bash
  cat models/3_smurf_anomaly_detector/README.md
  ```

- [ ] Run training
  ```bash
  python models/3_smurf_anomaly_detector/train_smurf_anomaly.py
  ```

- [ ] Check MLflow results
  - Experiment: `smurf-anomaly-detection`
  - Look for 4 runs (IsolationForest v1, v2, EllipticEnvelope, LOF)

- [ ] Verify output files
  ```bash
  ls models/3_smurf_anomaly_detector/models/
  # Should have: smurf_anomaly_model.pkl, scaler.pkl, metadata.json
  ```

### Person 4: Match Outcome Predictor

- [ ] Review instructions
  ```bash
  cat models/4_match_outcome_predictor/README.md
  ```

- [ ] Run training
  ```bash
  python models/4_match_outcome_predictor/train_match_outcome.py
  ```

- [ ] Check MLflow results
  - Experiment: `match-outcome-prediction`
  - Look for 5 runs (LogisticRegression, RandomForest v1, v2, XGBoost v1, v2)

- [ ] Verify output files
  ```bash
  ls models/4_match_outcome_predictor/models/
  # Should have: match_outcome_model.pkl, scaler.pkl, metadata.json
  ```

---

## ✅ Comparison Phase (After All Training Complete)

- [ ] One person runs comparison
  ```bash
  python model_comparison.py
  ```

- [ ] Check output files generated
  - Terminal output with stats
  - `MODEL_COMPARISON_REPORT.txt` - detailed report
  - `model_comparison.png` - visualization

- [ ] Review MLflow dashboard
  - All 4 experiments visible
  - Total runs: 16+ across all experiments
  - All metrics logged

---

## ✅ Customization (Optional)

### Add More Hyperparameter Variations

Example: Train another RandomForest variation in rank_tier model

Edit: `models/1_rank_tier_classifier/train_rank_tier.py`

Add to `main()`:
```python
# Additional hyperparameter search
classifier.train_random_forest(
    run_name="RandomForest-v4",
    n_estimators=300,
    max_depth=25,
    min_samples_split=3
)
```

Then run again - new run appears in MLflow!

---

## ✅ Troubleshooting

### MLflow experiments not showing?
```bash
# Re-run setup
python mlflow_setup.py

# Restart server
mlflow ui --port 5000
```

### Training script won't run?
```bash
# Check Python path
python -c "from models.shared import DataLoader; print('OK')"

# Install dependencies again
pip install -r requirements.txt -U
```

### Missing data files?
```bash
# Check data directory
ls data/processed/
# Should have 4 CSV files
```

### Port 5000 in use?
```bash
# Use different port
mlflow ui --port 5001
```

---

## ✅ Expected Results Summary

| Model | Experiment | Runs | Models |
|-------|-----------|------|--------|
| **Rank Tier** | rank-tier-classification | 3+ | RandomForest v1, v2, v3 |
| **Progression** | progression-regression | 4+ | Linear, Ridge, RandomForest |
| **Smurf** | smurf-anomaly-detection | 4+ | IsolationForest, Elliptic, LOF |
| **Match** | match-outcome-prediction | 5+ | LogisticRegression, RF, XGBoost |

Each model produces:
- ✅ `.pkl` file (trained model)
- ✅ `scaler.pkl` (for preprocessing)
- ✅ `metadata.json` (configuration)

---

## ✅ Next Steps After Training

1. Run `python model_comparison.py`
2. Review comparison report
3. Identify best performing model from each category
4. Export models to FastAPI backend
5. Create REST API endpoints

---

## ✅ File Checklist

After everything is complete, verify:

```
✅ models/1_rank_tier_classifier/models/
   ├── rank_tier_model.pkl
   ├── scaler.pkl
   └── metadata.json

✅ models/2_progression_regressor/models/
   ├── progression_model.pkl
   ├── scaler.pkl
   └── metadata.json

✅ models/3_smurf_anomaly_detector/models/
   ├── smurf_anomaly_model.pkl
   ├── scaler.pkl
   └── metadata.json

✅ models/4_match_outcome_predictor/models/
   ├── match_outcome_model.pkl
   ├── scaler.pkl
   └── metadata.json

✅ Model comparison outputs:
   ├── MODEL_COMPARISON_REPORT.txt
   └── model_comparison.png
```

---

## ✅ Communication Tips

When working in parallel:

- **Slack/Email update**: "Completed rank tier training, 3 runs logged"
- **Check MLflow first**: See if someone already trained a model
- **Share findings**: "XGBoost v2 achieved 95% accuracy on match prediction"
- **Coordinate timing**: Run comparison after all models complete

---

**STATUS: All setup complete! Ready to train models! 🚀**

Start with Setup Phase → Then Training Phase (parallel) → Comparison Phase
