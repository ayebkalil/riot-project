"""
Analyze early-game features dataset and impact of filtering by game duration.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_dataset(csv_path: Path, game_duration_threshold: int = 600):
    """
    Analyze early features dataset and show impact of duration filtering.
    
    Args:
        csv_path: Path to match_features_early_simple.csv
        game_duration_threshold: Minimum game duration in seconds (default 10 min = 600s)
    """
    
    print("LOADING DATASET...")
    df = pd.read_csv(csv_path)
    
    print(f"\n=== FULL DATASET STATS ===")
    print(f"Total rows: {len(df):,}")
    print(f"Total matches: {len(df) // 2:,}")
    print(f"Game duration range: {df['game_duration_sec'].min()} to {df['game_duration_sec'].max()} seconds")
    print(f"Average game duration: {df['game_duration_sec'].mean():.0f} seconds ({df['game_duration_sec'].mean()/60:.1f} min)")
    
    # Show distribution of game durations
    short_games = (df['game_duration_sec'] < game_duration_threshold).sum() // 2
    long_games = (df['game_duration_sec'] >= game_duration_threshold).sum() // 2
    
    print(f"\nGames < 10 min: {short_games:,} ({100*short_games/(short_games+long_games):.1f}%)")
    print(f"Games >= 10 min: {long_games:,} ({100*long_games/(short_games+long_games):.1f}%)")
    
    # Filter to valid games
    df_valid = df[df['game_duration_sec'] >= game_duration_threshold].copy()
    
    print(f"\n=== AFTER FILTERING OUT SHORT GAMES (<10min) ===")
    print(f"Remaining rows: {len(df_valid):,}")
    print(f"Remaining matches: {len(df_valid) // 2:,}")
    
    # Analyze feature quality
    print(f"\n=== FEATURE STATISTICS (VALID GAMES ONLY) ===\n")
    
    feature_cols = [
        'lane_cs_10m', 'jungle_cs_10m', 'total_cs_10m', 'takedowns_early',
        'aces_before_15m', 'first_turret_kills', 'early_laning_advantage',
        'control_wards_placed', 'avg_kill_participation'
    ]
    
    for col in feature_cols:
        mean_val = df_valid[col].mean()
        std_val = df_valid[col].std()
        min_val = df_valid[col].min()
        max_val = df_valid[col].max()
        print(f"{col:30} Mean: {mean_val:8.2f}  Std: {std_val:8.2f}  Range: [{min_val:6.1f}, {max_val:8.1f}]")
    
    # Win rate signal check
    print(f"\n=== WIN RATE PREDICTION SIGNAL ===\n")
    
    winners = df_valid[df_valid['is_winner'] == 1]
    losers = df_valid[df_valid['is_winner'] == 0]
    
    print(f"Winners vs Losers early game stats:\n")
    print(f"{'Feature':<30} {'Winners':<12} {'Losers':<12} {'Diff':<12} {'Win Advantage %':<15}")
    print("-" * 80)
    
    for col in feature_cols:
        winner_mean = winners[col].mean()
        loser_mean = losers[col].mean()
        diff = winner_mean - loser_mean
        if loser_mean != 0:
            pct = 100 * diff / loser_mean
        else:
            pct = 0
        print(f"{col:<30} {winner_mean:<12.2f} {loser_mean:<12.2f} {diff:<12.2f} {pct:<14.1f}%")
    
    # Data quality issues
    print(f"\n=== DATA QUALITY ISSUES ===\n")
    
    time_features = ['first_turret_time_sec', 'earliest_dragon_time_sec', 'earliest_baron_time_sec']
    for col in time_features:
        nulls = df_valid[col].isna().sum()
        pct = 100 * nulls / len(df_valid)
        print(f"{col:<30} {nulls:>8,} missing ({pct:>5.1f}%) - Expected! Not all games reach these objectives")
    
    # Export summary stats for reference
    summary = {
        'Total matches': len(df) // 2,
        'Valid matches (>=10min)': len(df_valid) // 2,
        'Short matches (<10min)': short_games,
        'Avg game duration (min)': df['game_duration_sec'].mean() / 60,
        'Valid games avg duration (min)': df_valid['game_duration_sec'].mean() / 60,
    }
    
    print(f"\n=== SUMMARY ===\n")
    for key, val in summary.items():
        if 'avg' in key.lower():
            print(f"{key}: {val:.1f}")
        else:
            print(f"{key}: {val:,}")
    
    return df_valid

if __name__ == "__main__":
    csv_path = Path(r'data/processed/match_features_early_simple.csv')
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run build_match_features_early_simple.py first.")
    else:
        df_clean = analyze_dataset(csv_path)
        
        # Save clean version
        output_path = Path(r'data/processed/match_features_early_simple_CLEAN.csv')
        df_clean.to_csv(output_path, index=False)
        print(f"\nSaved clean dataset to: {output_path}")
