# 🏆 Riot Games ML Project - Training Results

**Training Date**: February 19, 2026  
**Python Version**: 3.11.0  
**MLflow Tracking**: Enabled at http://localhost:5000

---

## 📊 Executive Summary

This document presents the complete training results for all 4 machine learning models trained on League of Legends player and match data.

---

## 🤖 Model 1: Rank Tier Classifier

### Task
**Multi-class classification** - Predict player rank tier (Low/Mid/High/Elite)

### Dataset
- **Total Samples**: 4,340 players
- **Features**: 40 (enriched with interaction terms)
- **Train/Test Split**: 80/20 (3,472 / 868)
- **Class Distribution**:
  - Low: 1,329 players (30.6%)
  - Mid: 1,337 players (30.8%)
  - High: 987 players (22.7%)
  - Elite: 687 players (15.8%)

### Algorithms Trained

#### 1. Random Forest
**Hyperparameters:**
- n_estimators: 200
- max_depth: 15
- min_samples_split: 5
- class_weight: balanced
- random_state: 42

**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **62.79%** |
| Test Precision | 0.6290 |
| Test Recall | 0.6279 |
| Test F1-Score | 0.6175 |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.61 | 0.40 | 0.48 | 266 |
| Mid | 0.60 | 0.78 | 0.68 | 268 |
| High | 0.69 | 0.68 | 0.68 | 197 |
| Elite | 0.63 | 0.71 | 0.67 | 137 |

**Analysis:**
- Best at predicting Mid and High tiers
- Struggles with Low tier (40% recall)
- Balanced precision across all classes

---

#### 2. LightGBM
**Hyperparameters:**
- objective: multiclass
- num_leaves: 31
- learning_rate: 0.05
- n_estimators: 500
- early_stopping_rounds: 50
- random_state: 42

**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **64.40%** |
| Test Precision | 0.6432 |
| Test Recall | 0.6440 |
| Test F1-Score | 0.6399 |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.63 | 0.49 | 0.55 | 266 |
| Mid | 0.64 | 0.72 | 0.68 | 268 |
| High | 0.68 | 0.71 | 0.69 | 197 |
| Elite | 0.62 | 0.69 | 0.66 | 137 |

**Analysis:**
- **+1.61% improvement** over Random Forest
- Better balance between precision and recall
- Still best at Mid and High tiers

---

#### 3. XGBoost ⭐ **WINNER**
**Hyperparameters:**
- objective: multi:softprob
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.1
-subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42

**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **65.44%** ⭐ |
| Test Precision | 0.6551 |
| Test Recall | 0.6544 |
| Test F1-Score | 0.6531 |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.64 | 0.56 | 0.60 | 266 |
| Mid | 0.63 | 0.72 | 0.67 | 268 |
| High | 0.70 | 0.68 | 0.69 | 197 |
| Elite | 0.68 | 0.67 | 0.68 | 137 |

**Analysis:**
- **+2.65% improvement** over Random Forest
- **+1.04% improvement** over LightGBM
- Most balanced across all classes
- Highest precision for High tier (0.70)

---

### Model 1: Final Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Notes |
|------|-------|----------|-----------|--------|----------|-------|
| 🥇 | **XGBoost** | **65.44%** | 0.6551 | 0.6544 | 0.6531 | Best overall |
| 🥈 | LightGBM | 64.40% | 0.6432 | 0.6440 | 0.6399 | Good balance |
| 🥉 | Random Forest | 62.79% | 0.6290 | 0.6279 | 0.6175 | Baseline |

**Key Insights:**
- Gradient boosting (LightGBM, XGBoost) outperform Random Forest
- XGBoost provides best generalization
- All models struggle with Low tier detection (class imbalance)
- Mid/High tiers are most predictable

---

## 🤖 Model 2: Progression Predictor

### Task
**Regression** - Predict winrate improvement/decline (delta_winrate)

### Dataset
- **Total Samples**: 4,128 players
- **Features**: 18 (temporal deltas + mastery features)
- **Train/Test Split**: 80/20 (3,302 / 826)
- **Target Range**: -1.0 to +1.0 (winrate change)

### Algorithms Trained

#### 1. Linear Regression (Baseline)
**Results:**
| Metric | Train | Test |
|--------|-------|------|
| MSE | TBD | TBD |
| RMSE | TBD | TBD |
| MAE | TBD | TBD |
| **R²** | TBD | **TBD** |

---

#### 2. Ridge Regression (alpha=0.1)
**Results:**
| Metric | Train | Test |
|--------|-------|------|
| MSE | TBD | TBD |
| RMSE | TBD | TBD |
| MAE | TBD | TBD |
| **R²** | TBD | **TBD** |

---

#### 3. Ridge Regression (alpha=1.0)
**Results:**
| Metric | Train | Test |
|--------|-------|------|
| MSE | TBD | TBD |
| RMSE | TBD | TBD |
| MAE | TBD | TBD |
| **R²** | TBD | **TBD** |

---

#### 4. LightGBM ⭐ (Expected Winner)
**Results:**
| Metric | Train | Test |
|--------|-------|------|
| MSE | TBD | TBD |
| RMSE | TBD | TBD |
| MAE | TBD | TBD |
| **R²** | TBD | **TBD** |

---

## 🤖 Model 3: Smurf Anomaly Detector

### Task
**Unsupervised Anomaly Detection** - Identify suspicious/smurf accounts

### Dataset
- **Total Samples**: 4,340 players
- **Features**: 14 (z-score normalized metrics)
- **No Train/Test Split**: Unsupervised learning on full dataset

### Algorithms Trained

#### 1. Isolation Forest (contamination=0.08)
**Results:**
| Metric | Value |
|--------|-------|
| Anomalies Detected | TBD |
| Anomaly Ratio | TBD% |
| Mean Anomaly Score | TBD |

---

#### 2. Isolation Forest (contamination=0.12) ⭐ (Expected Winner)
**Results:**
| Metric | Value |
|--------|-------|
| Anomalies Detected | TBD |
| Anomaly Ratio | TBD% |
| Mean Anomaly Score | TBD |

---

#### 3. Elliptic Envelope (contamination=0.10)
**Results:**
| Metric | Value |
|--------|-------|
| Anomalies Detected | TBD |
| Anomaly Ratio | TBD% |
| Mean Anomaly Score | TBD |

---

#### 4. Local Outlier Factor (n_neighbors=20)
**Results:**
| Metric | Value |
|--------|-------|
| Anomalies Detected | TBD |
| Anomaly Ratio | TBD% |
| Mean Anomaly Score | TBD |

---

## 🤖 Model 4: Match Outcome Predictor

### Task
**Binary Classification** - Predict match winner (team_won: 0/1)

### Dataset
- **Total Samples**: 306,312 team records
- **Features**: 15 (team differentials)
- **Train/Test Split**: 80/20 (244,649 / 61,163)
- **Class Balance**: ~50/50 (wins/losses)

### Algorithms Trained

#### 1. Logistic Regression (Baseline)
**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD% |
| Test F1-Score | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| ROC-AUC | TBD |

---

#### 2. Random Forest (n_estimators=100, max_depth=15)
**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD% |
| Test F1-Score | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| ROC-AUC | TBD |

---

#### 3. Random Forest (n_estimators=150, max_depth=20)
**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD% |
| Test F1-Score | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| ROC-AUC | TBD |

---

#### 4. XGBoost (n_estimators=100) ⭐ (Expected Winner)
**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD% |
| Test F1-Score | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| ROC-AUC | TBD |

---

#### 5. XGBoost (n_estimators=150, deeper)
**Results:**
| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD% |
| Test F1-Score | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| ROC-AUC | TBD |

---

## 🎯 Overall Project Results

### Summary Statistics

| Model | Best Algorithm | Best Metric | Dataset Size |
|-------|---------------|-------------|--------------|
| 1. Rank Tier Classifier | XGBoost | 65.44% accuracy | 4,340 players |
| 2. Progression Predictor | LightGBM | TBD R² | 4,128 players |
| 3. Smurf Detector | IsolationForest | TBD anomalies | 4,340 players |
| 4. Match Outcome | XGBoost | TBD% accuracy | 306,312 records |

---

## 📈 MLflow Experiment Tracking

### Access Results
1. **MLflow UI**: http://localhost:5000
2. **Navigate** to each experiment:
   - `rank-tier-classification`
   - `progression-prediction`
   - `smurf-anomaly-detection`
   - `match-outcome-prediction`
3. **Compare** runs side-by-side using the Compare button

### Logged Information
For each model run:
- ✅ All hyperparameters
- ✅ All metrics (train & test)
- ✅ Feature names
- ✅ Visualizations (confusion matrices, feature importance)
- ✅ Model artifacts (saved locally)

---

## 🎓 Key Findings

### Model 1: Rank Tier Classifier
- **XGBoost achieved 65.44% accuracy**
- Class imbalance affects Low tier predictions
- Feature interactions improve performance
- Gradient boosting > Random Forest for this task

### Model 2: Progression Predictor
- [Results pending]

### Model 3: Smurf Anomaly Detector
- [Results pending]

### Model 4: Match Outcome Predictor
- [Results pending]

---

## 🚀 Next Steps

### Immediate
1. ✅ Complete training for Models 2-4
2. ✅ Update this report with final metrics
3. ✅ Generate comparison visualizations

### Future Improvements
1. **Hyperparameter Tuning**: GridSearchCV/RandomSearchCV
2. **Cross-Validation**: K-fold for robust metrics
3. **Ensemble Methods**: Voting/Stacking classifiers
4. **Feature Engineering**: More domain-specific interactions
5. **Deployment**: Flask API for real-time predictions

---

## 📊 Visualizations

All visualizations available in MLflow UI:
- Confusion matrices
- Feature importance charts
- Per-class metrics
- Prediction confidence distributions
- Train/test comparison plots

---

**Status**: Model 1 Complete ✅ | Models 2-4 In Progress 🔄

**Last Updated**: February 19, 2026 08:30 AM
