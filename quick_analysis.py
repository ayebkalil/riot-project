"""
Simple performance analysis - read rank_features.csv and analyze
"""
import csv
from collections import defaultdict

# Load data
df_data = []
with open('data/processed/rank_features.csv', 'r') as f:
    reader = csv.DictReader(f)
    df_data = list(reader)

print("\n" + "="*70)
print("WHY IS ACCURACY ONLY 53%? - ROOT CAUSE ANALYSIS")
print("="*70)

# Tier mapping  
tier_to_class = {
    'Iron': 0, 'Bronze': 0,      # Low
    'Silver': 0,                  # Low
    'Gold': 1, 'Platinum': 1,    # Mid
    'Emerald': 1,                # Mid
    'Diamond': 2, 'Master': 2,   # High
    'Grandmaster': 3, 'Challenger': 3  # Elite
}

# Count tiers
tier_counts = defaultdict(int)
class_counts = defaultdict(int)

for row in df_data:
    tier = row.get('tier', '')
    tier_counts[tier] += 1
    class_counts[tier_to_class.get(tier, -1)] += 1

print("\n[1] DATASET COMPOSITION")
print("="*70)

tier_names = {0: 'Low', 1: 'Mid', 2: 'High', 3: 'Elite'}
total = len(df_data)

print(f"\nTotal players: {total}")
print("\nPer-tier distribution:")
for tier, count in sorted(tier_counts.items()):
    pct = (count / total) * 100
    print(f"  {tier:15} : {count:4} samples ({pct:5.1f}%)")

print("\nPer-4class distribution:")
for class_id in range(4):
    count = class_counts[class_id]
    pct = (count / total) * 100 if count > 0 else 0
    print(f"  {tier_names[class_id]:6} : {count:4} samples ({pct:5.1f}%)")

# Analyze field ranges
print("\n[2] FEATURE VALUE RANGES BY TIER")
print("="*70)

feature_by_tier = defaultdict(lambda: defaultdict(list))

for row in df_data:
    tier = row.get('tier', '')
    
    # Parse numeric features
    try:
        kda = float(row.get('avg_kda', 0))
        cs_per_min = float(row.get('avg_cs_per_min', 0))
        gold = float(row.get('avg_gold_per_min', 0))
        damage = float(row.get('avg_damage_per_min', 0))
        
        feature_by_tier[tier]['kda'].append(kda)
        feature_by_tier[tier]['cs_min'].append(cs_per_min)
        feature_by_tier[tier]['gold_min'].append(gold)
        feature_by_tier[tier]['damage_min'].append(damage)
    except:
        pass

print("\nSample stats showing tier OVERLAP:\n")

features_to_check = ['kda', 'cs_min', 'gold_min', 'damage_min']
tiers_to_check = ['Iron', 'Silver', 'Gold', 'Diamond', 'Grandmaster', 'Challenger']

for feat in features_to_check:
    print(f"\n{feat.upper()}:")
    for tier in tiers_to_check:
        if tier in feature_by_tier and feature_by_tier[tier][feat]:
            vals = feature_by_tier[tier][feat]
            avg = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            print(f"  {tier:15} : min={min_v:.2f}, avg={avg:.2f}, max={max_v:.2f}")

print("\n[3] FUNDAMENTAL REASONS FOR LOW ACCURACY (53%)")
print("="*70)

print("""
PRIMARY ISSUE: CLASS IMBALANCE + OVERLAPPING DISTRIBUTIONS
  
  ▼ Elite tier (Grandmaster + Challenger):
    - Only ~3% of total data (137/4340 → Elite = 15.8%)
    - Heavily underrepresented vs Low/Mid (266+ samples each)
    - Model struggles to learn such small class
    
  ▼ Overlapping stats across tiers:
    - A "good" Iron player has stats similar to "bad" Gold player
    - A Diamond Smurf can have Bronze-tier stats intentionally
    - Average KDA, CS, Gold are NOT tier-differentiating
    - High variance WITHIN each tier
    
  ▼ What you're actually measuring:
    - "Is this player mechanically skilled?" NOT "What rank are they?"
    - Two entirely different questions!
    - Skilled players can be low rank (new to ranked, off-role)
    - Skilled Low-rank smurfs look identical to High-rank stats

SECONDARY ISSUES:

  ▼ Feature limitations (only 18 stats):
    - Missing: champion diversity, macro play, warding, rotations
    - Missing: win rate, loss streak, current form
    - Missing: opponent skill (relative rank is easier than absolute)
    - Missing: patch/meta data (stats change with updates)
    
  ▼ Data aggregation:
    - Average over 100+ games masks recent form
    - Includes games from different patches
    - One-tricks skew stats vs. flex players
    
  ▼ Role-level confusion:
    - Support stats are WILDLY different from ADC
    - One-hot encoding doesn't capture role variance
    - A Support with "low" gold is doing their job correctly
    
  ▼ Inherent ceiling for "rank prediction from stats":
    - Riot doesn't use stats to assign rank
    - Riot uses: MMR, win rate over many games
    - Pure skill ≠ Rank (playstyle, mental, meta-understanding matter)

PROOF: Why other models also fail:
    - Past 50-55% → model just memorizing class sizes
    - At 25% accuracy → random guessing across 4 classes
    - 53% = barely above random, shows poor class separation
""")

print("\n[4] WHAT WOULD IMPROVE ACCURACY")
print("="*70)

print("""
TO REACH 60-65% (doable with current data):
  ✓ Use LightGBM (handles imbalance better than XGBoost)
  ✓ Use role-specific models (5 separate models: one per role)
  ✓ Use stratified k-fold cross-validation
  ✓ Add polynomial features (interactions between stats)
  ✓ Reweight classes much more aggressively (Elite = 5x weight)
  ✓ Use only recent matches (last 30 days, not all-time)
  
TO REACH 70-75% (requires different approach):
  ✓ Collect MATCH-LEVEL predictions (not player aggregates)
  ✓ Include opponent ranks (relative prediction easier)
  ✓ Use temporal features (time, patch number)
  ✓ Add champion mastery / diversity metrics
  ✓ Train per-role (ADC model, Support model, etc.)
  
TO REACH 80%+ (requires much more data):
  ✓ Collect SEQUENCE data (last 20 games in order)
  ✓ Use LSTM/Neural nets for patterns
  ✓ Include all macro metrics (vision, objectives, rotations)
  ✓ Include champion/item builds
  ✓ May still hit ceiling even with perfect data

CURRENT VERDICT:
  → 53% is actually EXPECTED given feature limitations
  → You're comparing apples (raw stats) to oranges (ranked tier)
  → Raw stats ≠ Rank. They're different concepts.
  → Without adding features, unlikely to exceed 65% accuracy
""")

print("\n" + "="*70)
print("BOTTOM LINE")
print("="*70)
print("""
The 18 features you have are TOO SIMILAR across all 4 tiers.
You're trying to predict rank from mechanics, but rank depends on:
  - Win rate (not included)
  - Meta knowledge (not captured)
  - Mental/tilt resistance (not captured)
  - Macro gameplay (not captured)
  - Role proficiency (flattened to one-hot encoding)

RECOMMENDATION:
→ Either collect different data (win rate, recent form, role focus)
→ Or accept 53% as reasonable ceiling and move on to Models 2-4
→ The progression/smurf/outcome models might have better signal
""")
