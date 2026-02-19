"""
Create enriched progression features with delta metrics.
Converts match-level temporal features to delta (change) metrics for Model 2.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def create_progression_enriched_features():
    """Generate enriched progression features with temporal deltas"""
    
    print("=" * 80)
    print("CREATING ENRICHED PROGRESSION FEATURES")
    print("=" * 80)
    
    # Load raw match features
    print("\nLoading raw match features...")
    match_features = pd.read_csv('data/processed/match_features_raw.csv')
    print(f"✓ Loaded {len(match_features):,} player-match records")
    
    # Load original progression features
    print("Loading original progression features...")
    progression = pd.read_csv('data/processed/progression_features.csv')
    print(f"✓ Loaded {len(progression):,} players")
    
    # For each player, calculate temporal deltas
    # We'll use first half vs second half of matches to compute changes
    
    enriched_data = []
    
    print("\nCalculating temporal deltas for each player...")
    for puuid in progression['puuid'].unique():
        player_matches = match_features[match_features['puuid'] == puuid].sort_values('gameDuration')
        
        if len(player_matches) < 4:  # Need at least 4 matches to split
            continue
        
        # Split into first half and second half
        split_idx = len(player_matches) // 2
        first_half = player_matches.iloc[:split_idx]
        second_half = player_matches.iloc[split_idx:]
        
        # Calculate average metrics for each half
        first_gpm = first_half['goldPerMinute'].mean()
        second_gpm = second_half['goldPerMinute'].mean()
        delta_gpm = second_gpm - first_gpm
        
        first_dpm = first_half['damagePerMinute'].mean()
        second_dpm = second_half['damagePerMinute'].mean()
        delta_dpm = second_dpm - first_dpm
        
        first_vspm = first_half['visionScorePerMinute'].mean()
        second_vspm = second_half['visionScorePerMinute'].mean()
        delta_vspm = second_vspm - first_vspm
        
        first_ssa = first_half['skillshotAccuracy'].mean()
        second_ssa = second_half['skillshotAccuracy'].mean()
        delta_ssa = second_ssa - first_ssa
        
        # Champion pool growth
        first_champ_pool = first_half['championId'].nunique()
        second_champ_pool = second_half['championId'].nunique()
        champ_pool_growth = second_champ_pool - first_champ_pool
        
        # Store enriched data
        enriched_data.append({
            'puuid': puuid,
            'delta_goldPerMinute': delta_gpm,
            'delta_damagePerMinute': delta_dpm,
            'delta_visionScorePerMinute': delta_vspm,
            'delta_skillshotAccuracy': delta_ssa,
            'champion_pool_growth': champ_pool_growth,
            'total_matches_analyzed': len(player_matches)
        })
    
    enriched_df = pd.DataFrame(enriched_data)
    print(f"✓ Generated delta features for {len(enriched_df):,} players")
    
    # Merge with original progression features
    print("\nMerging with original progression features...")
    merged = progression.merge(enriched_df, on='puuid', how='left')
    
    # Fill missing values
    delta_cols = [col for col in merged.columns if col.startswith('delta_')]
    for col in delta_cols:
        merged[col] = merged[col].fillna(0.0)
    
    merged['champion_pool_growth'] = merged['champion_pool_growth'].fillna(0)
    merged['total_matches_analyzed'] = merged['total_matches_analyzed'].fillna(0)
    
    # Save
    output_path = 'data/processed/progression_features_enriched_v2.csv'
    merged.to_csv(output_path, index=False)
    print(f"✓ Saved enriched progression features: {output_path}")
    print(f"  Dimensions: {merged.shape[0]} players × {merged.shape[1]} columns")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ENRICHED PROGRESSION FEATURES CREATED")
    print("=" * 80)
    print(f"""
New columns:
1. delta_goldPerMinute - Income improvement trajectory
2. delta_damagePerMinute - Combat improvement
3. delta_visionScorePerMinute - Vision awareness improvement
4. delta_skillshotAccuracy - Mechanical skill improvement
5. champion_pool_growth - Champion pool expansion
6. total_matches_analyzed - Sample size

Output: {output_path}
Rows: {merged.shape[0]}
Total columns: {merged.shape[1]} (original 13 + new 5)
    """)

if __name__ == '__main__':
    create_progression_enriched_features()
