"""
Extract early-game features from FLAT MATCH JSON (no timeline required).

These features reconstruct early-game advantage signals available in Riot API v5
without needing the separate timeline endpoint. Works on your existing match JSON files.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd


def extract_early_team_features(match_data: dict) -> List[dict]:
    """
    Extract early-game team features from flat match JSON (no timeline needed).
    
    Available signals:
    - laneMinionsFirst10Minutes, jungleCsBefore10Minutes (CS at 10m)
    - firstTurretKilled / firstTurretKilledTime (early objective control)
    - earliestBaron / earliestDragonTakedown (objective timing)
    - earlyLaningPhaseGoldExpAdvantage (explicit early game metric!)
    - killParticipation (early involvement)
    - controlWardsPlaced (early vision)
    - acesBefore15Minutes (team fight presence)
    
    Returns list of 2 dicts (one per team) with early feature aggregates.
    """
    
    metadata = match_data.get("metadata", {})
    info = match_data.get("info", {})
    match_id = metadata.get("matchId", "")
    game_duration = info.get("gameDuration", 0)  # seconds
    
    participants = info.get("participants", [])
    teams = {t["teamId"]: t for t in info.get("teams", [])}
    
    # Determine winner
    winner_id = None
    for team in info.get("teams", []):
        if team.get("win"):
            winner_id = team["teamId"]
            break
    
    # Group participants by team
    team_100_players = [p for p in participants if p["teamId"] == 100]
    team_200_players = [p for p in participants if p["teamId"] == 200]
    
    rows = []
    
    for team_id, team_players in [(100, team_100_players), (200, team_200_players)]:
        if not team_players:
            continue
            
        # === EARLY GAME AGGREGATES ===
        
        # 1. CS at 10 minutes (from challenges)
        lane_cs_10m = sum(p.get("challenges", {}).get("laneMinionsFirst10Minutes", 0) 
                          for p in team_players)
        jungle_cs_10m = sum(p.get("challenges", {}).get("jungleCsBefore10Minutes", 0) 
                            for p in team_players)
        total_cs_10m = lane_cs_10m + jungle_cs_10m
        
        # 2. Early tournament/kill pressure
        takedowns_early = sum(p.get("challenges", {}).get("takedownsFirstXMinutes", 0) 
                              for p in team_players)
        aces_before_15m = sum(p.get("challenges", {}).get("acesBefore15Minutes", 0) 
                              for p in team_players)
        
        # 3. First objective control
        first_turret_kills = sum(1 for p in team_players 
                                 if p.get("challenges", {}).get("firstTurretKilled", False))
        first_turret_time = min(
            (p.get("challenges", {}).get("firstTurretKilledTime", float('inf')) 
             for p in team_players),
            default=float('inf')
        )
        
        # 4. Dragon/Baron timing (earliest indicates team fighting early)
        earliest_dragon = min(
            (p.get("challenges", {}).get("earliestDragonTakedown", float('inf')) 
             for p in team_players),
            default=float('inf')
        )
        earliest_baron = min(
            (p.get("challenges", {}).get("earliestBaron", float('inf')) 
             for p in team_players),
            default=float('inf')
        )
        
        # 5. Early laning phase advantage (RIOT'S OWN "early game" metric!)
        early_laning_advantage = sum(
            p.get("challenges", {}).get("earlyLaningPhaseGoldExpAdvantage", 0) 
            for p in team_players
        )
        
        # 6. Vision control early
        control_wards = sum(p.get("challenges", {}).get("controlWardsPlaced", 0) 
                            for p in team_players)
        
        # 7. Fight participation (% of early kills team is involved in)
        avg_kill_participation = sum(p.get("challenges", {}).get("killParticipation", 0) 
                                     for p in team_players) / len(team_players)
        
        # 8. Pre-match composition strength (weak proxy: count of meta champions)
        # You could add a static meta tier dict here if desired
        champion_ids = [p["championId"] for p in team_players]
        
        # 9. Gold/XP differences (team totals, as proxy for early advantage)
        total_gold = sum(p.get("goldEarned", 0) for p in team_players)
        total_xp = sum(p.get("champExperience", 0) for p in team_players)
        avg_level = sum(p.get("champLevel", 0) for p in team_players) / len(team_players)
        
        rows.append({
            "match_id": match_id,
            "game_duration_sec": game_duration,
            "team_id": team_id,
            "is_winner": int(team_id == winner_id),
            
            # === EARLY GAME FEATURES ===
            "lane_cs_10m": lane_cs_10m,
            "jungle_cs_10m": jungle_cs_10m,
            "total_cs_10m": total_cs_10m,
            
            "takedowns_early": takedowns_early,
            "aces_before_15m": aces_before_15m,
            
            "first_turret_kills": first_turret_kills,
            "first_turret_time_sec": first_turret_time if first_turret_time != float('inf') else None,
            
            "earliest_dragon_time_sec": earliest_dragon if earliest_dragon != float('inf') else None,
            "earliest_baron_time_sec": earliest_baron if earliest_baron != float('inf') else None,
            
            "early_laning_advantage": early_laning_advantage,
            "control_wards_placed": control_wards,
            "avg_kill_participation": avg_kill_participation,
            
            "total_gold_earned": total_gold,
            "total_xp": total_xp,
            "avg_champion_level": avg_level,
        })
    
    return rows


def build_features_from_matches(
    matches_dir: Path,
    output_csv: Path,
    max_matches: Optional[int] = None,
    min_game_duration: int = 600,
) -> None:
    """Build early-game features from all match JSON files."""
    
    match_files = sorted(matches_dir.glob("*.json"))
    if max_matches:
        match_files = match_files[:max_matches]
    
    all_rows = []
    skipped_short_games = 0
    
    for i, match_file in enumerate(match_files, 1):
        try:
            with open(match_file, encoding='utf-8', errors='ignore') as f:
                match_data = json.load(f)

            game_duration = match_data.get("info", {}).get("gameDuration", 0)
            if game_duration < min_game_duration:
                skipped_short_games += 1
                continue
            
            team_rows = extract_early_team_features(match_data)
            all_rows.extend(team_rows)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(match_files)} matches...")
        
        except Exception as e:
            print(f"Error processing {match_file.name}: {e}")
            continue
    
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(df)} team samples to {output_csv}")
    print(f"✓ Skipped {skipped_short_games} short matches (< {min_game_duration}s)")
    print(f"\nFeatures extracted:")
    print(f"  • CS at 10m (lane + jungle)")
    print(f"  • Early takedowns & aces")
    print(f"  • First turret timing")
    print(f"  • Dragon/Baron earliest times")
    print(f"  • Laning phase gold/XP advantage (RIOT'S OWN metric)")
    print(f"  • Vision control (control wards)")
    print(f"  • Kill participation")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Extract early-game features from flat match JSON")
    parser.add_argument(
        "--matches-dir",
        type=Path,
        default=Path("data/raw/matches"),
        help="Directory with match JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/match_features_early_simple.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        help="Limit matches to process (for testing)"
    )
    parser.add_argument(
        "--min-game-duration",
        type=int,
        default=600,
        help="Minimum game duration in seconds to keep (default: 600 = 10 minutes)"
    )
    
    args = parser.parse_args()
    args.matches_dir = args.matches_dir.resolve()
    args.output = args.output.resolve()
    
    print(f"Reading matches from: {args.matches_dir}")
    print(f"Output to: {args.output}\n")
    
    build_features_from_matches(
        args.matches_dir,
        args.output,
        args.max_matches,
        args.min_game_duration,
    )
