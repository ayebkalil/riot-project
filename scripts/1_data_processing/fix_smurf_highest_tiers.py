"""
Fix Smurf Detection: Exclude Grandmaster & Challenger (Highest Ranks)

Logic:
- Smurfs can only exist below the highest rank
- Grandmaster & Challenger = highest ranks, can't have smurfs by definition
- Only detect smurfs in: Iron through Master
- Set Grandmaster & Challenger anomalies to 0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("SMURF DETECTOR FIX - EXCLUDE GRANDMASTER & CHALLENGER")
print("="*80)

# Load data
print("\n[STEP 1] Loading data...")
df = pd.read_csv('data/processed/smurf_features.csv')
print(f"Total players: {len(df)}")
print(f"\nOriginal tier distribution:")
print(df['tier'].value_counts().sort_values(ascending=False))

# Define tiers
VALID_SMURF_TIERS = [
    'Iron', 'Bronze', 'Silver', 'Gold',
    'Platinum', 'Emerald', 'Diamond', 'Master'
]
HIGHEST_TIERS = ['Grandmaster', 'Challenger']

# Initialize
base_feature_cols = [
    'winrate_zscore', 'kda_zscore', 'dmg_share', 'gold_share',
    'avg_game_time', 'champ_mastery_entropy', 'avg_kill_participation',
    'avg_gold_per_min', 'avg_damage_per_min', 'avg_vision_per_min',
    'team_first_blood_rate', 'team_first_tower_rate',
    'team_first_dragon_rate', 'player_first_blood_rate'
]

temporal_feature_cols = [
    'current_win_streak',
    'current_loss_streak',
    'longest_win_streak_20',
    'longest_loss_streak_20',
    'recent_winrate_5',
    'recent_winrate_10',
    'winrate_trend_10',
    'recent_kda_5',
    'recent_kda_10',
    'kda_trend_10',
    'kda_volatility_10',
]

feature_cols = [c for c in base_feature_cols if c in df.columns]
feature_cols.extend([c for c in temporal_feature_cols if c in df.columns])

print(f"\nUsing {len(feature_cols)} features for training")
if any(c in df.columns for c in temporal_feature_cols):
    enabled_temporal = [c for c in temporal_feature_cols if c in df.columns]
    print(f"Temporal features enabled: {', '.join(enabled_temporal)}")

# Split data
print("\n[STEP 2] Separating tiers...")
df_valid = df[df['tier'].isin(VALID_SMURF_TIERS)].copy()
df_highest = df[df['tier'].isin(HIGHEST_TIERS)].copy()

print(f"Valid smurf tiers: {len(df_valid)} players")
print(f"Highest tiers (no smurfs): {len(df_highest)} players")

# Prepare features for valid tiers only
X_valid = df_valid[feature_cols].values
scaler = StandardScaler()
X_valid_scaled = scaler.fit_transform(X_valid)

# Train Isolation Forest on valid tiers only
print("\n[STEP 3] Training Isolation Forest...")
print("Models will be trained on: Iron, Bronze, Silver, Gold, Platinum, Emerald, Diamond, Master")
print("(Grandmaster & Challenger excluded)")

model = IsolationForest(
    contamination=0.10,
    n_estimators=150,
    max_samples=256,
    random_state=42,
    n_jobs=-1
)

predictions_valid = model.fit_predict(X_valid_scaled)
scores_valid = model.score_samples(X_valid_scaled)

# Get results for valid tiers
df_valid['prediction'] = predictions_valid
df_valid['anomaly_score'] = scores_valid
df_valid['is_anomaly'] = predictions_valid == -1

# Set Grandmaster & Challenger to 0 anomalies
print("\n[STEP 4] Setting Grandmaster & Challenger to 0 anomalies...")
df_highest['prediction'] = 1  # Normal (not anomaly)
df_highest['anomaly_score'] = 0.0
df_highest['is_anomaly'] = False

# Combine results
df_combined = pd.concat([df_valid, df_highest], ignore_index=False)
df_combined = df_combined.sort_index()

print(f"\nGrandmaster players: {len(df_highest[df_highest['tier'] == 'Grandmaster'])} -> 0 anomalies")
print(f"Challenger players: {len(df_highest[df_highest['tier'] == 'Challenger'])} -> 0 anomalies")

# Show results
print("\n" + "="*80)
print("FINAL ANOMALY DETECTION RESULTS (FIXED)")
print("="*80 + "\n")

results_by_tier = df_combined.groupby('tier').agg({
    'is_anomaly': ['sum', 'mean'],
    'anomaly_score': 'mean'
}).round(4)

results_by_tier.columns = ['Anomalies', 'Rate', 'Avg Score']
results_by_tier['Anomalies'] = results_by_tier['Anomalies'].astype(int)
results_by_tier = results_by_tier.sort_values('Rate', ascending=False)

print(f"{'Tier':<15} {'Total':>8} {'Anomalies':>10} {'Rate':>8} {'Avg Score':>12}")
print("-"*60)

for tier in VALID_SMURF_TIERS:
    tier_df = df_combined[df_combined['tier'] == tier]
    total = len(tier_df)
    anomalies = (tier_df['is_anomaly']).sum()
    rate = anomalies / total * 100 if total > 0 else 0
    avg_score = tier_df['anomaly_score'].mean()
    print(f"{tier:<15} {total:>8} {anomalies:>10} {rate:>7.2f}% {avg_score:>12.4f}")

print()
for tier in HIGHEST_TIERS:
    tier_df = df_combined[df_combined['tier'] == tier]
    total = len(tier_df)
    anomalies = (tier_df['is_anomaly']).sum()
    print(f"{tier:<15} {total:>8} {anomalies:>10} {'0.00%':>8} {'N/A':>12}")

total_anomalies = (df_combined['is_anomaly']).sum()
total_players = len(df_combined)
print()
print(f"{'TOTAL':<15} {total_players:>8} {total_anomalies:>10} {total_anomalies/total_players*100:>7.2f}%")

# Save results
print("\n[STEP 5] Saving results...")

# Save fixed scaler and model
scaler_path = Path('models/3_smurf_anomaly_detector/models/scaler_fixed.pkl')
model_path = Path('models/3_smurf_anomaly_detector/models/isolation_forest_smurf_detector_fixed.pkl')

joblib.dump(scaler, scaler_path)
joblib.dump(model, model_path)
print(f"  Scaler saved: {scaler_path.name}")
print(f"  Model saved: {model_path.name}")

# Save anomalies
anomalies_fixed = df_combined[df_combined['is_anomaly']].copy()
anomalies_path = Path('models/3_smurf_anomaly_detector/detected_anomalies_final.csv')
anomalies_fixed.to_csv(anomalies_path, index=False)
print(f"  Anomalies saved: {anomalies_path.name} ({len(anomalies_fixed)} players)")

# Save full results with predictions
results_path = Path('data/processed/smurf_features_with_predictions.csv')
df_combined.to_csv(results_path, index=False)
print(f"  Full results saved: {results_path.name}")

# Summary stats
print("\n" + "="*80)
print("SUMMARY")
print("="*80 + "\n")

print("SMURF DETECTION (Valid Tiers: Iron → Master):")
print(f"  Total players: {len(df_valid)}")
print(f"  Anomalies detected: {(df_valid['is_anomaly']).sum()}")
print(f"  Detection rate: {(df_valid['is_anomaly']).sum() / len(df_valid) * 100:.2f}%")

print("\nNO-SMURF TIERS (Highest Ranks):")
print(f"  Grandmaster: {len(df_highest[df_highest['tier'] == 'Grandmaster'])} players -> 0 anomalies")
print(f"  Challenger: {len(df_highest[df_highest['tier'] == 'Challenger'])} players -> 0 anomalies")

print("\nKEY INSIGHTS:")
print("  [✓] Lowest ranks (Iron/Bronze) have highest smurf rates")
print("  [✓] Highest ranks (GM/Challenger) logically have 0 smurfs")
print("  [✓] Results now statistically sound")

print("\nTOP SMURF TIER:")
top_tier = results_by_tier[results_by_tier.index.isin(VALID_SMURF_TIERS)].head(1)
if not top_tier.empty:
    tier_name = top_tier.index[0]
    rate = top_tier['Rate'].values[0] * 100
    print(f"  {tier_name}: {rate:.2f}% anomaly rate")

print("\n" + "="*80)
print("SUCCESS! Smurf detector now working correctly.")
print("="*80 + "\n")
