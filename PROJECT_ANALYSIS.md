# 🎮 Riot Games ML Project - Comprehensive Analysis

**Date**: February 19, 2026  
**Python Version**: 3.11.0  
**MLflow**: Installed and configured

---

## 📊 Project Overview

This project builds **4 machine learning models** to analyze League of Legends player and match data:

1. **Rank Tier Classifier** - Predict player rank (Low/Mid/High/Elite)
2. **Progression Predictor** - Predict winrate improvement
3. **Smurf Anomaly Detector** - Detect suspicious accounts
4. **Match Outcome Predictor** - Predict match winner

---

## 📁 Data Files (Processed)

### 1. rank_features_enriched_v2.csv
- **Samples**: 4,340 players
- **Features**: 40 (enriched with interaction terms)
- **Target**: tier (Low/Mid/High/Elite - 4 classes)
- **Key Features**:
  - Performance metrics (KDA, CS, Gold, Damage, Vision)
  - Early game impact (First Blood, Tower, Dragon rates)
  - Champion pool diversity
  - Win rate statistics

### 2. progression_features_enriched_v2.csv
- **Samples**: 4,128 players
- **Features**: 18 (temporal deltas + mastery features)
- **Target**: delta_winrate (continuous regression)
- **Key Features**:
  - delta_goldPerMinute - income growth
  - delta_damagePerMinute - combat improvement
  - delta_visionScorePerMinute - awareness growth
  - championPoolSize changes
  - role_consistency_delta

### 3. smurf_features.csv
- **Samples**: 4,340 players
- **Features**: 14 (z-score normalized)
- **Target**: None (unsupervised anomaly detection)
- **Key Features**:
  - winrate_zscore, kda_zscore (tier-normalized)
  - dmg_share, gold_share (team contribution)
  - champ_mastery_entropy (diversity)
  - avg_kill_participation

### 4. match_features.csv
- **Samples**: 306,312 team records (from matches)
- **Features**: 15 (team differentials)
- **Target**: team_won (binary: 0/1)
- **Key Features**:
  - rank_diff, kda_diff, cs_diff
  - gold_diff, damage_diff, vision_diff
  - first_blood_diff, first_tower_diff, first_dragon_diff

---

## 🤖 Model 1: Rank Tier Classifier

### Task
Multi-class classification (4 classes: Low/Mid/High/Elite)

### Algorithms Trained
1. **Random Forest**
   - n_estimators: 200
   - max_depth: 15
   - class_weight: balanced
   
2. **LightGBM** ⭐ (Best)
   - num_leaves: 31
   - learning_rate: 0.05
   - n_estimators: 500
   - early_stopping: 50 rounds
   
3. **XGBoost**
   - n_estimators: 300
   - max_depth: 6
   - learning_rate: 0.1

### Data Split
- Train: 3,472 samples (80%)
- Test: 868 samples (20%)
- Stratified by class to maintain distribution

### MLflow Experiment
- Name: `rank-tier-classification`
- Metrics: accuracy, precision, recall, f1_score, support
- Artifacts: 5 visualizations per model

---

## 🤖 Model 2: Progression Predictor

### Task
Regression - Predict delta_winrate (improvement/decline)

### Algorithms Trained
1. **LinearRegression**
   - Baseline model
   
2. **Ridge** (alpha=0.1)
   - L2 regularization
   
3. **Ridge** (alpha=1.0)
   - Stronger regularization
   
4. **LightGBM** ⭐ (Best)
   - Gradient boosting for regression
   - Early stopping enabled

### Data Split
- Train: 3,302 samples (80%)
- Test: 826 samples (20%)

### MLflow Experiment
- Name: `progression-prediction`
- Metrics: MSE, RMSE, MAE, R²
- Target: Continuous (-1.0 to +1.0)

---

## 🤖 Model 3: Smurf Anomaly Detector

### Task
Unsupervised anomaly detection

### Algorithms Trained
1. **IsolationForest** (contamination=0.08)
   - n_estimators: 100
   
2. **IsolationForest** (contamination=0.12) ⭐ (Best)
   - n_estimators: 150
   
3. **EllipticEnvelope** (contamination=0.10)
   - Gaussian distribution assumption
   
4. **LocalOutlierFactor** (n_neighbors=20)
   - Density-based detection

### Data
- Full dataset: 4,340 samples (no split)
- Scaled with StandardScaler

### MLflow Experiment
- Name: `smurf-anomaly-detection`
- Metrics: n_anomalies, anomaly_ratio, mean/std/min/max_anomaly_score
- No ground truth labels

---

## 🤖 Model 4: Match Outcome Predictor

### Task
Binary classification - Predict team win/loss

### Algorithms Trained
1. **LogisticRegression**
   - C: 1.0
   - max_iter: 1000
   
2. **RandomForest** (n_estimators=100)
   - max_depth: 15
   
3. **RandomForest** (n_estimators=150)
   - max_depth: 20
   
4. **XGBoost** (n_estimators=100) ⭐ (Best expected)
   - max_depth: 6
   - learning_rate: 0.1
   
5. **XGBoost** (n_estimators=150)
   - max_depth: 7
   - learning_rate: 0.15

### Data Split
- Train: 244,649 records (80%)
- Test: 61,163 records (20%)
- Stratified by team_won

### MLflow Experiment
- Name: `match-outcome-prediction`
- Metrics: accuracy, f1, precision, recall, roc_auc
- Largest dataset (306K+ records)

---

## 🔧 Technical Stack

### Core Libraries
- **Python**: 3.11.0
- **scikit-learn**: Classification, Regression, Preprocessing
- **LightGBM**: 4.6.0 (Gradient Boosting)
- **XGBoost**: Latest (Gradient Boosting)
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### ML Operations
- **MLflow**: 2.10.0
  - Experiment tracking
  - Parameter logging
  - Metric logging
  - Artifact management
- **Matplotlib/Seaborn**: Visualizations

### Data Processing
- **StandardScaler**: Feature normalization
- **Train-test split**: 80/20
- **Stratification**: For classification tasks
- **Missing value handling**: Median imputation

---

## 📈 MLflow Integration

### Experiments Structure
```
mlflow/
├── rank-tier-classification/
│   ├── Random Forest
│   ├── LightGBM
│   └── XGBoost
├── progression-prediction/
│   ├── LinearRegression
│   ├── Ridge-alpha-0.1
│   ├── Ridge-alpha-1.0
│   └── LightGBM
├── smurf-anomaly-detection/
│   ├── IsolationForest-v1
│   ├── IsolationForest-v2
│   ├── EllipticEnvelope-v1
│   └── LocalOutlierFactor-v1
└── match-outcome-prediction/
    ├── LogisticRegression-v1
    ├── RandomForest-v1
    ├── RandomForest-v2
    ├── XGBoost-v1
    └── XGBoost-v2
```

### Tracked Information
**For each run:**
- Model hyperparameters
- Training/test metrics
- Feature names
- Model artifacts (saved locally)
- Visualizations (confusion matrices, feature importance, etc.)

### Access MLflow UI
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

---

## 🎯 Key Features Across Models

### Shared Features (Common patterns)
- **avg_kda**: Kill-Death-Assist ratio
- **avg_gold_per_min**: Gold farming efficiency
- **avg_damage_per_min**: Combat output
- **avg_vision_per_min**: Map awareness
- **team_first_blood_rate**: Early game aggression
- **team_first_tower_rate**: Objective control
- **win_rate**: Historical performance

### Model-Specific Features

**Rank Classifier:**
- skillshotAccuracy, soloKills (mechanics)
- champion_pool, role_diversity (versatility)
- recent_form_10, recent_form_30 (momentum)

**Progression:**
- delta_* features (change over time)
- championPoolSize growth
- role_consistency_delta

**Smurf Detection:**
- Z-score normalized stats (tier-relative)
- champ_mastery_entropy (specialist vs generalist)
- dmg_share, gold_share (team carry)

**Match Outcome:**
- Team differentials (relative advantages)
- Early game leads (first blood/tower/dragon)

---

## 📊 Expected Performance

### Model 1: Rank Tier Classifier
- **Baseline (Random Forest)**: ~63% accuracy
- **Best (LightGBM)**: ~65% accuracy
- **Challenge**: Class imbalance (Elite: 15.8%, Mid: 30.8%)

### Model 2: Progression Predictor
- **Baseline (Linear)**: R² ~0.36
- **Best (LightGBM)**: R² ~0.45-0.50
- **Challenge**: High variance in player improvement

### Model 3: Smurf Detector
- **Anomaly Detection Rate**: 8-12% of players
- **No ground truth** - unsupervised learning
- **Use case**: Flag suspicious accounts for review

### Model 4: Match Outcome Predictor
- **Baseline (Logistic)**: ~70% accuracy
- **Best (XGBoost)**: ~85-90% accuracy
- **Large dataset**: 306K records allows deep learning

---

## 🚀 Training Pipeline

### Master Script: `train_all_models.py`

Trains all 4 models sequentially with MLflow tracking:

```bash
python train_all_models.py
```

**Execution Flow:**
1. Model 1: Rank Tier Classifier (~2-3 min)
2. Model 2: Progression Predictor (~1-2 min)
3. Model 3: Smurf Anomaly Detector (~1-2 min)
4. Model 4: Match Outcome Predictor (~5-8 min, largest dataset)

**Total Time**: ~10-15 minutes

**Output:**
- Trained models saved to `models/X_*/models/*.pkl`
- Scalers saved for deployment
- Metadata JSON files
- MLflow runs logged to each experiment

---

## 📁 Output Structure

```
models/
├── 1_rank_tier_classifier/models/
│   ├── random_forest_model.pkl
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── metadata.json
│   └── visualizations/ (15+ PNG files)
├── 2_progression_regressor/models/
│   ├── progression_model_v2_enriched.pkl
│   ├── scaler_v2_enriched.pkl
│   └── metadata_v2_enriched.json
├── 3_smurf_anomaly_detector/models/
│   ├── smurf_anomaly_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
└── 4_match_outcome_predictor/models/
    ├── match_outcome_model.pkl
    ├── scaler.pkl
    └── metadata.json
```

---

## 🎓 For Professor Demo

### Key Points to Highlight

**1. Comprehensive ML Pipeline**
- 4 different ML tasks (classification, regression, anomaly detection)
- Multiple algorithms per task
- Systematic comparison via MLflow

**2. Production-Ready Code**
- Centralized data loader (`shared/data_loader.py`)
- Reusable MLflow utilities (`shared/mlflow_utils.py`)
- Proper train/test splits
- Feature scaling
- Model persistence

**3. Large-Scale Dataset**
- 4,340 players analyzed
- 306,312 match records
- Real data from Riot API

**4. Advanced Feature Engineering**
- Interaction terms (goldPerMinute × KDA)
- Temporal deltas (progression over time)
- Z-score normalization (tier-relative stats)
- Team differentials (match predictions)

**5. MLflow Experiment Tracking**
- All hyperparameters logged
- All metrics tracked
- Reproducible experiments
- Easy model comparison

### Demo Flow
1. Show processed datasets
2. Run `train_all_models.py`
3. Open MLflow UI
4. Navigate through experiments
5. Compare models side-by-side
6. Highlight best performers
7. Show visualizations (confusion matrices, feature importance)

---

## ✅ Project Strengths

1. **Clean Architecture**: Modular code, shared utilities
2. **Scalability**: Handles 300K+ records efficiently
3. **Reproducibility**: MLflow tracking + random seeds
4. **Best Practices**: Stratified splits, scaling, early stopping
5. **Comprehensive**: 4 models × multiple algorithms = 16 total experiments

---

## 🎯 Next Steps (Future Work)

1. **Hyperparameter Tuning**: GridSearch/RandomSearch
2. **Cross-Validation**: K-fold for robust metrics
3. **Ensemble Methods**: Voting/Stacking classifiers
4. **Deep Learning**: Neural networks for match prediction
5. **Deployment**: Flask API for real-time predictions
6. **Monitoring**: Drift detection, performance tracking

---

**Status**: Ready for training and demonstration ✅
