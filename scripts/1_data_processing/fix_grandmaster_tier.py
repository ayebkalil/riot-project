"""
Fix Grandmaster tier issue in smurf_features.csv

The problem:
- Grandmaster players were combined with Master tier in smurf_features.csv
- opgg/by_rank/grandmaster.csv has the list of actual Grandmaster players
- Need to update smurf_features.csv to separate them

Approach:
1. Load the Grandmaster and Master player lists from opgg
2. Load the smurf_features.csv
3. Update players in smurf_features to have correct tier
4. Also check and remove any Challenger players (they shouldn't be smurfs)
"""

import pandas as pd
from pathlib import Path
import json

# Load data
print("\n" + "="*80)
print("FIXING GRANDMASTER TIER IN SMURF FEATURES")
print("="*80)

# Load opgg leaderboards for all tiers
gm_players = pd.read_csv('opgg/by_rank/grandmaster.csv')
master_players = pd.read_csv('opgg/by_rank/master.csv')
challenger_players = pd.read_csv('opgg/by_rank/challenger.csv')

print(f"\n[OPGG DATA]")
print(f"Grandmaster players: {len(gm_players)}")
print(f"Master players: {len(master_players)}")
print(f"Challenger players: {len(challenger_players)}")

# Create summoner name to tier mapping
gm_names = set(gm_players['summoner_name'].str.lower())
master_names = set(master_players['summoner_name'].str.lower())
challenger_names = set(challenger_players['summoner_name'].str.lower())

print(f"\nSample GM players: {list(gm_players['summoner_name'].head(3).values)}")
print(f"Sample Master players: {list(master_players['summoner_name'].head(3).values)}")

# Load smurf features
smurf_df = pd.read_csv('data/processed/smurf_features.csv')
print(f"\n[SMURF FEATURES]")
print(f"Current tier distribution:")
print(smurf_df['tier'].value_counts().to_string())

# Issue: We need to find a way to match summoner names from opgg with puuids in smurf_features
# Since smurf_features only has puuid, not summoner_name, we need to use the source data

# Load players_by_rank to find the mapping
print(f"\n[CHECKING PLAYER SOURCE DATA]")
players_by_rank_dir = Path('data/processed/players_by_rank')

# Collect all player data with their names and puuids
all_players = []
for player_file in sorted(players_by_rank_dir.glob('*.csv')):
    rank_name = player_file.stem
    df = pd.read_csv(player_file)
    if 'summoner_name' in df.columns:
        df['file_tier'] = rank_name
        all_players.append(df)
        print(f"  {rank_name}: {len(df)} players with summoner_name")
    else:
        print(f"  {rank_name}: NO summoner_name column!")

if all_players:
    players_combined = pd.concat(all_players, ignore_index=True)
    print(f"\nTotal players from source: {len(players_combined)}")
    print(f"Unique puuids: {players_combined['puuid'].nunique()}")
    
    # Now we can fix the smurf features by using summoner_name as a proxy
    # Build a mapping from puuid to correct tier
    puuid_to_tier = {}
    
    for _, row in players_combined.iterrows():
        puuid = row['puuid']
        summoner_name = row['summoner_name'].lower()
        
        # Determine the correct tier based on opgg data
        if summoner_name in gm_names:
            correct_tier = 'Grandmaster'
        elif summoner_name in master_names:
            correct_tier = 'Master'
        elif summoner_name in challenger_names:
            correct_tier = 'Challenger'
        else:
            correct_tier = row.get('tier', 'Unknown')
        
        puuid_to_tier[puuid] = correct_tier
    
    print(f"\nBuilt mapping for {len(puuid_to_tier)} puuid->tier pairs")
    
    # Apply the fix
    print(f"\n[FIXING TIERS]")
    fixes_made = 0
    tier_changes = {}
    
    for idx, row in smurf_df.iterrows():
        puuid = row['puuid']
        old_tier = row['tier']
        
        if puuid in puuid_to_tier:
            new_tier = puuid_to_tier[puuid]
            if new_tier != old_tier:
                smurf_df.at[idx, 'tier'] = new_tier
                fixes_made += 1
                change_key = f"{old_tier} → {new_tier}"
                tier_changes[change_key] = tier_changes.get(change_key, 0) + 1
    
    print(f"Fixed {fixes_made} tier assignments")
    if tier_changes:
        print(f"\nTier changes made:")
        for change, count in sorted(tier_changes.items()):
            print(f"  {change}: {count} players")
    
    # Show new distribution
    print(f"\n[NEW TIER DISTRIBUTION]")
    print(smurf_df['tier'].value_counts().to_string())
    
    # Save the fixed file
    output_path = 'data/processed/smurf_features.csv'
    smurf_df.to_csv(output_path, index=False)
    print(f"\n✅ FIXED! Saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF CHANGES")
    print(f"{'='*80}")
    print(f"Grandmaster should now have proper count")
    print(f"Challenger players are now separated (no smurfs expected there)")
    print(f"Master tier reduced by ~{tier_changes.get('Master → Grandmaster', 0)} to reflect true Masters")
    
else:
    print("\n❌ ERROR: Could not find player data with summoner_name!")
    print("The smurf_features.csv only has puuid, not summoner_name")
    print("Need to regenerate from source data...")
