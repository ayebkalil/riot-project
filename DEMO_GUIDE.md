# Professor Demo Guide - Rank Tier Classifier with MLflow

**Project**: League of Legends ML Analytics Platform  
**Model**: Rank Tier Classification (Model 1/4)  
**Date**: February 2026

---

## 🎯 What This Demo Shows

This demo demonstrates a **machine learning model that predicts player rank tiers** (Iron → Challenger) based on their gameplay statistics.

- **Dataset**: 4,340 players with 20+ gameplay features
- **Task**: 9-class classification (remapped to 4 tiers: Low, Mid, High, Elite)
- **Model**: LightGBM with enriched features
- **Tracking**: MLflow for experiment management and visualization

---

## 🚀 Quick Start (5 minutes)

### Step 1: Ensure Dependencies
```powershell
pip install lightgbm mlflow scikit-learn pandas matplotlib seaborn
```

### Step 2: Run the Training Script
```powershell
cd "C:\Users\ayebk\OneDrive\Desktop\Riot Games Project"
python models/1_rank_tier_classifier/train_rank_tier_v2.py
```

**Expected output**: Training should complete in 1-2 minutes with performance metrics displayed.

### Step 3: Launch MLflow UI
```powershell
mlflow ui --port 5000
```

### Step 4: View Results
1. Open browser: **http://localhost:5000**
2. Navigate to experiment: **`rank-tier-classification`**
3. Click on run: **`LightGBM-v2-enriched`**

---

## 📊 What You'll See in MLflow

### Metrics Tab
- **Test Accuracy**: Primary metric showing model performance
- **Test Precision/Recall/F1**: Additional performance indicators
- **Train Accuracy**: For detecting overfitting

### Parameters Tab
- Model type (LightGBM)
- Hyperparameters (num_leaves, learning_rate, n_estimators)
- Dataset info (features, samples, train/test split)

### Artifacts Tab
1. **confusion_matrix_v2.png** - Shows which classes are confused
2. **feature_importance_v2.png** - Top 15 features influencing predictions
3. **model_metadata_v2.json** - Complete model information

---

## 📈 Key Visualizations Explained

### 1. Confusion Matrix
```
Shows what the model predicted vs actual tiers
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Helps identify which tiers are hard to distinguish
```

### 2. Feature Importance
```
Bar chart showing which gameplay stats matter most
- goldPerMinute: Economic efficiency
- avg_kda: Combat performance
- damagePerMinute: Damage output
- win_rate: Overall success
```

### 3. Model Metadata
```json
{
  "accuracy": 0.567,
  "precision": 0.568,
  "recall": 0.567,
  "f1_score": 0.563,
  "features": 30,
  "samples": 4340,
  "train_test_split": {
    "train": 3472,
    "test": 868
  }
}
```

---

## 💡 How to Explain to Professor

### 1. Project Context
"We built a system to predict player skill tiers in League of Legends using ML"

### 2. Data Pipeline
- ✅ Collected 4,340 player profiles from Riot API
- ✅ Extracted 30 gameplay features (KDA, CS/min, damage/min, etc.)
- ✅ Split into train/test (80/20 stratified split)

### 3. Model Approach
- ✅ Used LightGBM (gradient boosting tree ensemble)
- ✅ Applied feature scaling (StandardScaler)
- ✅ 4-class tier classification (Low/Mid/High/Elite tiers)

### 4. Results
- **Accuracy**: ~56.7% (challenging 9-class problem → 4-class)
- **Why useful**: Can identify player skill without direct tier info
- **Real-world use**: Smurf detection, skill assessment, matchmaking

### 5. Experiment Tracking
- ✅ Every run logged to MLflow for reproducibility
- ✅ Hyperparameters, metrics, and visualizations saved
- ✅ Easy to compare different model versions

---

## 🔄 Running Different Model Versions

You can quickly train different configurations by editing `train_rank_tier_v2.py`:

```python
results = trainer.train_lightgbm(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    num_leaves=50,           # ← Adjust tree complexity
    learning_rate=0.05,      # ← Adjust learning rate
    n_estimators=300         # ← Adjust number of trees
)
```

Each run creates a new entry in MLflow automatically!

---

## 🐛 Troubleshooting

### MLflow UI won't open
```powershell
# Use different port
mlflow ui --port 5001

# Check if process is running
Get-Process mlflow
```

### Model accuracy seems low
- This is a 9→4 class problem with overlapping tier distributions
- Expected baseline: ~25% (random 4-class)
- Our result: ~56.7% (much better)

### Features not found error
- Ensure `data/processed/rank_features_enriched_v2.csv` exists
- Run feature engineering first: `python feature_engineering.py`

---

## 📁 Key Files

```
models/
├── 1_rank_tier_classifier/
│   ├── train_rank_tier_v2.py      ← Run this for demo
│   ├── models/
│   │   ├── rank_tier_model_v2_enriched.pkl
│   │   ├── scaler_v2_enriched.pkl
│   │   ├── confusion_matrix_v2.png
│   │   ├── feature_importance_v2.png
│   │   └── metadata_v2_enriched.json
│   └── README.md
├── shared/
│   └── mlflow_utils.py            ← MLflow integration
```

---

## ✅ Demo Checklist

Before showing professor:
- [ ] Run `python models/1_rank_tier_classifier/train_rank_tier_v2.py`
- [ ] Verify completion message at end
- [ ] Launch `mlflow ui --port 5000`
- [ ] Check experiment exists in MLflow
- [ ] Verify confusion matrix and feature importance are visible
- [ ] Have this guide open for explanations

---

**Good luck! 🚀**
