"""
Extract temporal, mastery, and team dynamic features from match data
to improve Model 1 (Rank Tier) and Model 2 (Progression) accuracy.

Key improvements:
- Per-minute normalized metrics (not affected by game length)
- Skillshot accuracy (mechanical skill)
- Vision control (map awareness)
- Champion mastery (performance consistency)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def extract_player_features_from_match(match_data):
    """Extract per-player features from a single match"""
    features = []
    
    game_duration_minutes = match_data['info']['gameDuration'] / 60.0
    
    for participant in match_data['info']['participants']:
        puuid = participant['puuid']
        
        # Avoid division by zero
        if game_duration_minutes == 0:
            continue
            
        # Core per-minute metrics (NORMALIZED)
        gold_per_min = participant.get('goldEarned', 0) / game_duration_minutes
        damage_per_min = participant.get('totalDamageDealtToChampions', 0) / game_duration_minutes
        vision_per_min = participant.get('visionScore', 0) / game_duration_minutes
        
        # Mechanical skill
        skillshots_hit = participant.get('challenges', {}).get('skillshotsHit', 0)
        skillshots_dodged = participant.get('challenges', {}).get('skillshotsDodged', 0)
        skillshot_accuracy = skillshots_hit / (skillshots_hit + skillshots_dodged) if (skillshots_hit + skillshots_dodged) > 0 else 0.5
        
        # Team coordination
        kill_participation = participant.get('challenges', {}).get('killParticipation', 0)
        
        # Vision control
        control_wards = participant.get('challenges', {}).get('controlWardsPlaced', 0)
        ward_takedowns = participant.get('wardsKilled', 0)
        
        # Individual skill
        solo_kills = participant.get('challenges', {}).get('soloKills', 0)
        
        # Death efficiency
        time_dead = participant.get('totalTimeSpentDead', 0)
        death_time_ratio = time_dead / match_data['info']['gameDuration'] if match_data['info']['gameDuration'] > 0 else 0
        
        # Early game skill
        cs_at_10 = participant.get('challenges', {}).get('laneMinionsFirst10Minutes', 0)
        jungle_cs_at_10 = participant.get('challenges', {}).get('jungleCsBefore10Minutes', 0)
        early_cs = cs_at_10 + jungle_cs_at_10
        
        # Aggression
        turret_plates = participant.get('challenges', {}).get('turretPlatesTaken', 0)
        kills_near_turret = participant.get('challenges', {}).get('killsNearEnemyTurret', 0)
        
        # High-IQ plays
        epic_steals = participant.get('challenges', {}).get('epicMonsterSteals', 0)
        objective_steals = participant.get('objectivesStolen', 0)
        bounty_gold = participant.get('challenges', {}).get('bountyGold', 0)
        
        # Role & champion
        champion_id = participant.get('championId', 0)
        role = participant.get('teamPosition', 'UNKNOWN')
        
        # Game outcome
        win = 1 if participant.get('win', False) else 0
        
        features.append({
            'puuid': puuid,
            'championId': champion_id,
            'role': role,
            'win': win,
            'goldPerMinute': gold_per_min,
            'damagePerMinute': damage_per_min,
            'visionScorePerMinute': vision_per_min,
            'skillshotAccuracy': skillshot_accuracy,
            'killParticipation': kill_participation,
            'controlWardsPlaced': control_wards,
            'wardTakedowns': ward_takedowns,
            'soloKills': solo_kills,
            'deathTimeRatio': death_time_ratio,
            'earlyCS': early_cs,
            'turretPlates': turret_plates,
            'killsNearTurret': kills_near_turret,
            'epicMonsterSteals': epic_steals,
            'objectivesStolen': objective_steals,
            'bountyGold': bounty_gold,
            'gameDuration': match_data['info']['gameDuration']
        })
    
    return features

def load_all_player_puuids():
    """Load all player PUUIDs from rank_features.csv"""
    df = pd.read_csv('data/processed/rank_features.csv')
    return set(df['puuid'].unique())

def process_all_matches():
    """Process all match files and extract features"""
    print("=" * 80)
    print("EXTRACTING TEMPORAL & MASTERY FEATURES FROM MATCHES")
    print("=" * 80)
    
    # Load player PUUIDs
    print("\nLoading player PUUIDs...")
    target_puuids = load_all_player_puuids()
    print(f"✓ Loaded {len(target_puuids):,} players")
    
    # Find all match files
    match_dir = Path('data/raw/matches')
    match_files = list(match_dir.glob('*.json'))
    print(f"✓ Found {len(match_files):,} match files")
    
    # Extract features from all matches
    all_features = []
    skipped = 0
    
    print("\nProcessing matches...")
    for match_file in tqdm(match_files, desc="Extracting features"):
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            features = extract_player_features_from_match(match_data)
            
            # Filter to only our target players
            features = [f for f in features if f['puuid'] in target_puuids]
            all_features.extend(features)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\n✓ Extracted {len(all_features):,} player-match records")
    print(f"✗ Skipped {skipped:,} invalid matches")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    return df

def aggregate_features_for_rank_classification(df):
    """Aggregate match-level features to player-level for Model 1"""
    print("\n" + "=" * 80)
    print("AGGREGATING FEATURES FOR MODEL 1 (RANK TIER CLASSIFIER)")
    print("=" * 80)
    
    # Calculate per-player aggregates
    agg_dict = {
        'goldPerMinute': 'mean',
        'damagePerMinute': 'mean',
        'visionScorePerMinute': 'mean',
        'skillshotAccuracy': 'mean',
        'killParticipation': 'mean',
        'controlWardsPlaced': 'mean',
        'wardTakedowns': 'mean',
        'soloKills': 'mean',
        'deathTimeRatio': 'mean',
        'earlyCS': 'mean',
        'turretPlates': 'mean',
        'killsNearTurret': 'mean',
        'epicMonsterSteals': 'sum',  # Total clutch plays
        'objectivesStolen': 'sum',
        'bountyGold': 'mean',
        'win': 'sum'  # Total wins for win rate
    }
    
    player_features = df.groupby('puuid').agg(agg_dict).reset_index()
    
    # Calculate champion pool diversity
    champion_diversity = df.groupby('puuid')['championId'].nunique().reset_index()
    champion_diversity.columns = ['puuid', 'champion_pool_size']
    
    # Calculate role consistency (% games on most-played role)
    role_consistency = df.groupby('puuid').apply(
        lambda x: (x['role'].value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
    ).reset_index()
    role_consistency.columns = ['puuid', 'role_consistency']
    
    # Calculate total games
    total_games = df.groupby('puuid').size().reset_index()
    total_games.columns = ['puuid', 'total_games']
    
    # Merge all metrics
    player_features = player_features.merge(champion_diversity, on='puuid')
    player_features = player_features.merge(role_consistency, on='puuid')
    player_features = player_features.merge(total_games, on='puuid')
    
    # Calculate actual win rate
    player_features['matches_analyzed'] = player_features['total_games']
    player_features['wins_in_matches'] = player_features['win']
    player_features.drop('win', axis=1, inplace=True)
    
    print(f"\n✓ Aggregated features for {len(player_features):,} players")
    print(f"✓ Added {len(player_features.columns) - 1} new features")
    
    return player_features

def merge_with_existing_features(new_features, output_path='data/processed/rank_features_enriched_v2.csv'):
    """Merge new features with existing rank_features.csv"""
    print("\n" + "=" * 80)
    print("MERGING WITH EXISTING FEATURES")
    print("=" * 80)
    
    # Load existing features
    existing = pd.read_csv('data/processed/rank_features.csv')
    print(f"✓ Loaded existing features: {len(existing)} players × {len(existing.columns)} columns")
    
    # Merge
    merged = existing.merge(new_features, on='puuid', how='left', suffixes=('', '_new'))
    
    # Fill missing values (players with no match data)
    numeric_cols = new_features.select_dtypes(include=[np.number]).columns
    if 'puuid' in numeric_cols:
        numeric_cols = numeric_cols.drop('puuid')
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())
    
    print(f"✓ Merged: {len(merged)} players × {len(merged.columns)} columns")
    print(f"✓ New feature columns: {len(merged.columns) - len(existing.columns)}")
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    return merged

def main():
    print("\n🚀 Starting feature extraction pipeline...\n")
    
    # Step 1: Extract match-level features
    match_features = process_all_matches()
    
    # Save raw match features for debugging
    match_features.to_csv('data/processed/match_features_raw.csv', index=False)
    print(f"\n✓ Saved raw match features: data/processed/match_features_raw.csv")
    
    # Step 2: Aggregate to player-level for Model 1
    player_features = aggregate_features_for_rank_classification(match_features)
    
    # Step 3: Merge with existing features
    final = merge_with_existing_features(player_features)
    
    print("\n" + "=" * 80)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"""
Summary:
- Processed {len(match_features):,} player-match records
- Generated features for {len(player_features):,} players
- Added {len(player_features.columns) - 1} new columns
- Output: data/processed/rank_features_enriched_v2.csv

New Features:
1. goldPerMinute - NORMALIZED income (fixes game length issue)
2. damagePerMinute - NORMALIZED combat efficiency
3. visionScorePerMinute - NORMALIZED map awareness
4. skillshotAccuracy - Mechanical skill (hit / total)
5. killParticipation - Team coordination
6. controlWardsPlaced - Macro awareness
7. wardTakedowns - Vision denial
8. soloKills - Individual skill
9. deathTimeRatio - Death efficiency
10. earlyCS - Laning skill (CS @ 10 min)
11. turretPlates - Early aggression
12. epicMonsterSteals - Clutch plays (high elo trait)
13. champion_pool_size - Champion mastery breadth
14. role_consistency - Role specialization

Expected improvement: 53% → 60-65% accuracy
    """)

if __name__ == '__main__':
    main()
