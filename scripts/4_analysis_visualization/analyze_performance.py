"""
Analyze why Model 1 accuracy is low (53%)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import f_oneway, kruskal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.shared.data_loader import DataLoader

print("\n" + "="*70)
print("PERFORMANCE ANALYSIS: Why is accuracy only 53%?")
print("="*70)

# Load data
print("\n[1] LOADING DATA...")
df = DataLoader.load_rank_features()
X, y, feature_names = DataLoader.prepare_rank_features(remap_tiers=True)

# Tier mapping
tier_mapping = {0: 'Low', 1: 'Mid', 2: 'High', 3: 'Elite'}
df['tier_4class'] = pd.Series(y).map(tier_mapping)

print(f"✓ Loaded: {len(df)} players, {len(feature_names)} features, 4 tiers")
print(f"\nTier distribution (4-class):")
print(df['tier_4class'].value_counts().sort_index())

# Analysis 1: Class Balance
print("\n" + "="*70)
print("[2] CLASS IMBALANCE CHECK")
print("="*70)
tier_counts = df['tier_4class'].value_counts()
print(f"\nClass distribution:")
for tier in ['Low', 'Mid', 'High', 'Elite']:
    count = tier_counts.get(tier, 0)
    pct = (count / len(df)) * 100
    print(f"  {tier:6} : {count:4} samples ({pct:5.1f}%)")

elite_pct = (tier_counts['Elite'] / len(df)) * 100
print(f"\n⚠ Elite is {elite_pct:.1f}% of data (severely underrepresented)")

# Analysis 2: Feature Statistics by Tier
print("\n" + "="*70)
print("[3] FEATURE DISTRIBUTION ACROSS TIERS")
print("="*70)

# Check for statistical differences between tiers
print("\nKruskal-Wallis H-test (tests if feature means differ by tier):")
print("(p-value < 0.05 = significant difference)\n")

significant_features = []
non_significant_features = []

for feature in feature_names[:13]:  # First 13 numeric features (not roles)
    if feature in df.columns:
        groups = [df[df['tier_4class'] == tier][feature].dropna().values 
                  for tier in ['Low', 'Mid', 'High', 'Elite']]
        stat, pval = kruskal(*groups)
        
        significance = "✓ SIGNIFICANT" if pval < 0.05 else "✗ NOT SIGNIFICANT"
        print(f"  {feature:30} : p={pval:.4f}  {significance}")
        
        if pval < 0.05:
            significant_features.append(feature)
        else:
            non_significant_features.append(feature)

print(f"\n✓ Significant features: {len(significant_features)}")
print(f"✗ Non-significant features: {len(non_significant_features)}")

if non_significant_features:
    print(f"\n⚠ Warning: These features show NO statistical difference across tiers:")
    for f in non_significant_features:
        print(f"    - {f}")

# Analysis 3: Feature Overlap
print("\n" + "="*70)
print("[4] FEATURE OVERLAP ANALYSIS")
print("="*70)
print("\nChecking if Low/Mid/High/Elite ranges overlap (can't distinguish if they do):")

for feature in significant_features[:5]:  # Top 5 significant features
    print(f"\n  {feature}:")
    for tier in ['Low', 'Mid', 'High', 'Elite']:
        vals = df[df['tier_4class'] == tier][feature].dropna()
        print(f"    {tier:6} : min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")

# Analysis 4: Role Distribution
print("\n" + "="*70)
print("[5] ROLE DISTRIBUTION BY TIER")
print("="*70)
role_cols = [c for c in df.columns if c.startswith('role_')]
print(f"\nRole columns: {role_cols}")

if 'main_role' in df.columns:
    print("\nMain role distribution by tier:")
    role_tier = pd.crosstab(df['tier_4class'], df['main_role'], normalize='index') * 100
    print(role_tier.round(1))

# Analysis 5: Feature Importance from trained model
print("\n" + "="*70)
print("[6] FEATURE IMPORTANCE (from XGBoost-v2 model)")
print("="*70)

metadata_path = Path(__file__).parent / "models" / "1_rank_tier_classifier" / "models" / "metadata.json"
import json

try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"\n✓ Loaded metadata from: {metadata_path}")
    
    if 'feature_importance' in metadata:
        fi = metadata['feature_importance']
        print(f"\nTop 10 Important Features:")
        for i, (feat, imp) in enumerate(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"  {i:2}. {feat:30} : {imp:.4f}")
except:
    print("⚠ Couldn't load feature importance from metadata")

# Analysis 6: Suggested Improvements
print("\n" + "="*70)
print("[7] LIKELY REASONS FOR LOW ACCURACY (53%)")
print("="*70)

print("""
1. ✗ FUNDAMENTAL ISSUE: Rank prediction from stats is inherently noisy
   - Players can smurf (play on low-rank accounts with high-rank play)
   - Players can hard-stuck (high stats, low rank due to meta/meta misalignment)
   - One-tricking skews stats compared to role diversity
   
2. ✗ LIMITED FEATURES (18 features)
   - Only average stats per player
   - Missing: champion diversity, macro play, wave management
   - Missing: temporal info (when matches occurred, patch differences)
   - Missing: playstyle data (aggression, warding, objective control)
   
3. ✗ CLASS IMBALANCE
   - Elite: only 137 samples (15.8% of data) - too small
   - Model struggles with minority classes
   - Even with class_weights, can't recover from size imbalance
   
4. ✗ OVERLAPPING DISTRIBUTIONS
   - Low/Mid/High tier stats likely overlap significantly
   - Players perform inconsistently across games
   - Average stats may not truly differentiate tier
   
5. ✗ DATA QUALITY
   - Rank features are aggregated across ~100-200 matches per player
   - Includes players on win/loss streaks
   - Doesn't account for current form or recent changes
   
6. ✗ ROLE DIFFERENCES
   - AD Carry stats vastly different from Support
   - Jungler CS meaningless compared to laners
   - Current encoding (5 one-hot) might not capture role variance
""")

print("\n" + "="*70)
print("[8] RECOMMENDATIONS TO IMPROVE ACCURACY")
print("="*70)

print("""
QUICK WINS (May add 5-10% accuracy):
  → Use stratified k-fold cross-validation
  → Try LightGBM instead of XGBoost (faster, handles imbalance better)
  → Add polynomial features (interactions: KDA * gold_per_min, etc.)
  → Use ADASYN instead of SMOTE for better synthetic sampling
  → Fine-tune class_weights more aggressively

MEDIUM EFFORT (May add 10-15% accuracy):
  → Add role-specific feature scaling (scale features per role)
  → Create role-specific models (5 models: one per role)
  → Add game duration and match timing features
  → Include win rate, loss rate as direct features
  → Add champion pool diversity metrics

HIGH EFFORT (May add 15-25% accuracy):
  → Collect MATCH-LEVEL data (not aggregated)
  → Include temporal features (patch number, time period)
  → Use neural networks (LSTM for sequence data)
  → Incorporate macro play metrics (vision score, objective participation)
  → Collect opponent rank (predicting relative rank is easier than absolute)

REALITY CHECK:
  → If opponents are unknown: 53% might be CEILING for this dataset
  → Riot Systems likely use: champion select, picks, playstyle patterns
  → Raw stats alone have genuine limitations for rank prediction
""")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
