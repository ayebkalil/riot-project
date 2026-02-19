"""
Generate match-level TEAM features with DIFFERENTIALS for Model 4
Creates 306,312 rows (153,156 matches x 2 teams) with 15 columns

Each match creates 2 rows:
- Blue team (team_id=100): differentials = blue - red
- Red team (team_id=200): differentials = red - blue (opposite sign)

Target: team_won (1 for winning team, 0 for losing team)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_team_features(match_data):
    """Extract team-level features with differentials from a single match"""
    metadata = match_data['metadata']
    info = match_data['info']
    
    # Get match ID from metadata
    match_id = metadata.get('matchId', '')
    
    # Get both teams
    teams = {100: {'players': []}, 200: {'players': []}}
    
    for participant in info['participants']:
        team_id = participant['teamId']
        teams[team_id]['players'].append(participant)
    
    # Aggregate team stats
    team_stats = {}
    for team_id, team_data in teams.items():
        players = team_data['players']
        
        team_stats[team_id] = {
            'gold': sum(p.get('goldEarned', 0) for p in players),
            'damage': sum(p.get('totalDamageDealtToChampions', 0) for p in players),
            'kills': sum(p.get('kills', 0) for p in players),
            'deaths': sum(p.get('deaths', 0) for p in players),
            'assists': sum(p.get('assists', 0) for p in players),
            'vision_score': sum(p.get('visionScore', 0) for p in players),
            'turrets': sum(p.get('turretKills', 0) for p in players),
            'dragons': sum(p.get('dragonKills', 0) for p in players),
            'barons': sum(p.get('baronKills', 0) for p in players),
            'cs': sum(p.get('totalMinionsKilled', 0) + p.get('neutralMinionsKilled', 0) for p in players),
        }
    
    # Determine winner from teams info
    winner_team = None
    for team_info in info['teams']:
        if team_info['win']:
            winner_team = team_info['teamId']
    
    # Create rows for BOTH teams with differentials
    team_rows = []
    
    for team_id in [100, 200]:
        enemy_id = 200 if team_id == 100 else 100
        
        team_won = 1 if team_id == winner_team else 0
        
        # Differentials: this team - enemy team
        row = {
            'match_id': match_id,
            'team_id': team_id,
            'team_won': team_won,
            'gold_diff': team_stats[team_id]['gold'] - team_stats[enemy_id]['gold'],
            'damage_diff': team_stats[team_id]['damage'] - team_stats[enemy_id]['damage'],
            'kills_diff': team_stats[team_id]['kills'] - team_stats[enemy_id]['kills'],
            'deaths_diff': team_stats[team_id]['deaths'] - team_stats[enemy_id]['deaths'],
            'assists_diff': team_stats[team_id]['assists'] - team_stats[enemy_id]['assists'],
            'vision_diff': team_stats[team_id]['vision_score'] - team_stats[enemy_id]['vision_score'],
            'turrets_diff': team_stats[team_id]['turrets'] - team_stats[enemy_id]['turrets'],
            'dragons_diff': team_stats[team_id]['dragons'] - team_stats[enemy_id]['dragons'],
            'barons_diff': team_stats[team_id]['barons'] - team_stats[enemy_id]['barons'],
            'cs_diff': team_stats[team_id]['cs'] - team_stats[enemy_id]['cs'],
        }
        
        team_rows.append(row)
    
    return team_rows

def main():
    print("\n=== Generating Match Features Team-Level ===\n")
    
    # Find all match JSON files
    match_dir = Path('data/raw/matches')
    if not match_dir.exists():
        print(f"[ERROR] Directory not found: {match_dir}")
        print(f"[INFO] Expected path: {match_dir.absolute()}")
        return
    
    match_files = list(match_dir.glob('*.json'))
    print(f"[OK] Found {len(match_files)} match files\n")
    
    # Extract features from each match
    all_features = []
    successful_matches = 0
    skipped_files = 0
    
    print("[PROCESSING] Extracting team-level features...\n")
    
    for match_file in tqdm(match_files, desc="Matches processed"):
        try:
            with open(match_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Validate structure
            if 'metadata' not in match_data or 'matchId' not in match_data['metadata']:
                skipped_files += 1
                continue
            if 'info' not in match_data or 'participants' not in match_data['info']:
                skipped_files += 1
                continue
            
            team_features = extract_team_features(match_data)
            all_features.extend(team_features)
            successful_matches += 1
        
        except Exception as e:
            skipped_files += 1
            continue
    
    # Save to CSV
    df_features = pd.DataFrame(all_features)
    
    output_path = Path('data/processed/match_features.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    print(f"\n[OK] Processed {successful_matches:,} valid matches")
    print(f"[OK] Skipped {skipped_files:,} invalid files")
    print(f"[OK] Generated {len(all_features):,} team records")
    print(f"[OK] Saved to: {output_path}\n")
    print(f"Dataset shape: {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")
    print(f"Columns: {list(df_features.columns)}\n")
    print("Target distribution:")
    print(df_features['team_won'].value_counts().to_string())

if __name__ == '__main__':
    main()
