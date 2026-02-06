"""
Feature Engineering Pipeline
Consolidates rank-stratified data into 4 ML-ready datasets
"""

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


# ================= CONFIG =================

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_BY_RANK_DIR = PROCESSED_DIR / "players_by_rank"
MATCHES_BY_RANK_DIR = PROCESSED_DIR / "player_matches_by_rank"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ================= UTILITIES =================

def entropy(values: Iterable[int]) -> float:
    """Calculate Shannon entropy for champion pool diversity"""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p + 1e-12, 2) for p in probs)


# ================= FEATURE BUILDERS =================

def build_rank_features(matches_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Build player performance features for RANK CLASSIFICATION MODEL
    
    Features: Aggregated stats over last 20 matches
    Label: Current rank tier/division
    
    Use case: Predict a summoner's rank from their performance stats
    """
    print("[FEATURE] Building rank_features.csv...")
    features = []
    
    for puuid, group in matches_df.sort_values("game_creation", ascending=False).groupby("puuid"):
        recent = group.head(window)
        if recent.empty:
            continue
        
        duration_min = recent["game_duration"].replace(0, pd.NA) / 60
        kda_series = (recent["kills"] + recent["assists"]) / recent["deaths"].replace(0, 1)

        features.append({
            "puuid": puuid,
            "avg_kda": kda_series.mean(),
            "avg_cs_per_min": (recent["cs"] / duration_min).mean(),
            "avg_gold_per_min": (recent["gold"] / duration_min).mean(),
            "avg_damage_per_min": (recent["damage"] / duration_min).mean(),
            "avg_vision": recent["vision"].mean(),
            "win_rate": recent["win"].mean(),
            "champ_pool_size": recent["champ"].nunique(),
            "main_role": recent["role"].mode().iloc[0] if not recent["role"].mode().empty else "",
            "matches_used": len(recent),
        })

    df = pd.DataFrame(features)
    print(f"  ✓ Generated {len(df)} rank features")
    return df


def build_progression_features(matches_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Build temporal delta features for RANK PROGRESSION MODEL
    
    Features: Changes in stats between two 20-match windows (old vs recent)
    Label: Did player rank up, down, or stay same?
    
    Use case: Predict if player will climb or fall in next season
    """
    print("[FEATURE] Building progression_features.csv...")
    features = []
    
    for puuid, group in matches_df.sort_values("game_creation", ascending=False).groupby("puuid"):
        recent = group.head(window)
        if len(recent) < window:
            continue
        
        window_a = recent.head(window // 2)  # Older matches
        window_b = recent.tail(window // 2)  # Recent matches

        def summarize(df):
            duration_min = df["game_duration"].replace(0, pd.NA) / 60
            kda_series = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)
            return {
                "winrate": df["win"].mean(),
                "kda": kda_series.mean(),
                "cs_min": (df["cs"] / duration_min).mean(),
                "gold_min": (df["gold"] / duration_min).mean(),
            }

        a = summarize(window_a)
        b = summarize(window_b)

        # Calculate win streak
        streak = recent["win"].astype(int).tolist()
        longest = 0
        current = 0
        for w in streak:
            if w == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        features.append({
            "puuid": puuid,
            "delta_winrate": b["winrate"] - a["winrate"],
            "delta_kda": b["kda"] - a["kda"],
            "delta_cs": b["cs_min"] - a["cs_min"],
            "delta_gold": b["gold_min"] - a["gold_min"],
            "win_streak": longest,
            "matches_used": len(recent),
        })

    df = pd.DataFrame(features)
    print(f"  ✓ Generated {len(df)} progression features")
    return df


def build_smurf_features(matches_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rank-normalized z-score features for SMURF DETECTION MODEL
    
    Features: How player stats compare to their rank's baseline (z-scores)
    Label: Is player a smurf? (performing like higher rank)
    
    Use case: Detect boosted/smurf accounts
    """
    print("[FEATURE] Building smurf_features.csv...")
    
    merged = matches_df.merge(players_df[["puuid", "tier"]], on="puuid", how="left")
    merged["kda"] = (merged["kills"] + merged["assists"]) / merged["deaths"].replace(0, 1)
    
    # Calculate tier-wide statistics
    rank_stats = merged.groupby("tier").agg(
        winrate_mean=("win", "mean"),
        winrate_std=("win", "std"),
        kda_mean=("kda", "mean"),
        kda_std=("kda", "std"),
    )

    player_features = []
    for puuid, group in merged.groupby("puuid"):
        if group.empty:
            continue
        
        tier = group["tier"].iloc[0]
        baseline = rank_stats.loc[tier] if tier in rank_stats.index else None
        
        kda_series = (group["kills"] + group["assists"]) / group["deaths"].replace(0, 1)
        winrate = group["win"].mean()
        kda_avg = kda_series.mean()

        # Z-score: how many std devs away from tier mean
        winrate_std = (baseline["winrate_std"] if baseline is not None else 0) or 0
        kda_std = (baseline["kda_std"] if baseline is not None else 0) or 0
        
        winrate_z = ((winrate - baseline["winrate_mean"]) / (winrate_std + 1e-9)) if baseline is not None else 0
        kda_z = ((kda_avg - baseline["kda_mean"]) / (kda_std + 1e-9)) if baseline is not None else 0

        # Champion pool entropy (high = diverse, low = one-trick)
        champ_counts = group["champ"].value_counts()
        mastery_entropy = entropy(champ_counts.values)

        player_features.append({
            "puuid": puuid,
            "tier": tier,
            "winrate_zscore": winrate_z,
            "kda_zscore": kda_z,
            "dmg_share": group["dmg_share"].mean(),
            "gold_share": group["gold_share"].mean(),
            "avg_game_time": group["game_duration"].mean(),
            "champ_mastery_entropy": mastery_entropy,
        })

    df = pd.DataFrame(player_features)
    print(f"  ✓ Generated {len(df)} smurf detection features")
    return df


def build_match_features(matches_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build team-level differential features for MATCH OUTCOME PREDICTION MODEL
    
    Features: Team avg stats vs opponent team avg stats
    Label: Match win/loss
    
    Use case: Predict match outcome from team composition/performance
    """
    print("[FEATURE] Building match_features.csv...")
    
    # Merge rank info with matches
    matches_df = matches_df.merge(players_df[["puuid", "rank_numeric"]], on="puuid", how="left")

    # Aggregate to team level
    team_agg = matches_df.groupby(["match_id", "team_id"]).agg(
        team_win=("win", "max"),
        team_avg_rank=("rank_numeric", "mean"),
        team_avg_kda=("kills", "mean"),
        team_avg_cs=("cs", "mean"),
        team_avg_gold=("gold", "mean"),
        team_avg_damage=("damage", "mean"),
        team_avg_vision=("vision", "mean"),
    ).reset_index()

    # Join both teams in each match
    merged = team_agg.merge(
        team_agg,
        on="match_id",
        suffixes=("", "_opp"),
    )
    merged = merged[merged["team_id"] != merged["team_id_opp"]]

    # Calculate differentials (team metrics - opponent metrics)
    merged["rank_diff"] = merged["team_avg_rank"] - merged["team_avg_rank_opp"]
    merged["kda_diff"] = merged["team_avg_kda"] - merged["team_avg_kda_opp"]
    merged["cs_diff"] = merged["team_avg_cs"] - merged["team_avg_cs_opp"]
    merged["gold_diff"] = merged["team_avg_gold"] - merged["team_avg_gold_opp"]
    merged["damage_diff"] = merged["team_avg_damage"] - merged["team_avg_damage_opp"]
    merged["vision_diff"] = merged["team_avg_vision"] - merged["team_avg_vision_opp"]

    result = merged[[
        "match_id",
        "team_id",
        "team_win",
        "rank_diff",
        "kda_diff",
        "cs_diff",
        "gold_diff",
        "damage_diff",
        "vision_diff",
    ]]
    
    print(f"  ✓ Generated {len(result)} match outcome features")
    return result


# ================= MAIN PIPELINE =================

def consolidate_features() -> None:
    """
    Load all rank-stratified data and generate 4 feature datasets
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # ===== STEP 1: Load all player & match data =====
    print("\n[LOAD] Consolidating player & match data...")
    
    player_dfs = []
    match_dfs = []
    
    for player_file in sorted(PLAYERS_BY_RANK_DIR.glob("*.csv")):
        rank_name = player_file.stem
        print(f"  Loading {rank_name}...")
        player_dfs.append(pd.read_csv(player_file))
        
        match_file = MATCHES_BY_RANK_DIR / f"{rank_name}.csv"
        if match_file.exists():
            match_dfs.append(pd.read_csv(match_file))
    
    if not player_dfs or not match_dfs:
        print("\n❌ ERROR: No player or match data found!")
        print(f"   Players: {len(list(PLAYERS_BY_RANK_DIR.glob('*.csv')))} files")
        print(f"   Matches: {len(list(MATCHES_BY_RANK_DIR.glob('*.csv')))} files")
        return
    
    players_df = pd.concat(player_dfs, ignore_index=True).drop_duplicates("puuid")
    matches_df = pd.concat(match_dfs, ignore_index=True)
    
    print(f"\n✓ Loaded {len(players_df)} unique players")
    print(f"✓ Loaded {len(matches_df)} total matches")
    print(f"✓ Average {len(matches_df) / len(players_df):.1f} matches per player")
    
    # ===== STEP 2: Build 4 feature datasets =====
    print("\n" + "-"*60)
    print("GENERATING FEATURES")
    print("-"*60 + "\n")
    
    rank_features = build_rank_features(matches_df)
    rank_features = rank_features.merge(
        players_df[["puuid", "tier", "division", "rank_numeric"]],
        on="puuid",
        how="left",
    )
    
    progression_features = build_progression_features(matches_df)
    smurf_features = build_smurf_features(matches_df, players_df)
    match_features = build_match_features(matches_df, players_df)
    
    # ===== STEP 3: Save feature datasets =====
    print("\n" + "-"*60)
    print("SAVING DATASETS")
    print("-"*60 + "\n")
    
    rank_features_path = PROCESSED_DIR / "rank_features.csv"
    progression_features_path = PROCESSED_DIR / "progression_features.csv"
    smurf_features_path = PROCESSED_DIR / "smurf_features.csv"
    match_features_path = PROCESSED_DIR / "match_features.csv"
    
    rank_features.to_csv(rank_features_path, index=False)
    print(f"✓ Saved {len(rank_features)} rows → {rank_features_path}")
    
    progression_features.to_csv(progression_features_path, index=False)
    print(f"✓ Saved {len(progression_features)} rows → {progression_features_path}")
    
    smurf_features.to_csv(smurf_features_path, index=False)
    print(f"✓ Saved {len(smurf_features)} rows → {smurf_features_path}")
    
    match_features.to_csv(match_features_path, index=False)
    print(f"✓ Saved {len(match_features)} rows → {match_features_path}")
    
    # ===== STEP 4: Summary statistics =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n📊 RANK FEATURES (Rank Classification)")
    print(f"   Rows: {len(rank_features)} players")
    print(f"   Features: avg_kda, avg_cs_per_min, avg_gold_per_min, avg_damage_per_min, avg_vision, win_rate, champ_pool_size, main_role")
    print(f"   Labels: tier, division, rank_numeric")
    
    print(f"\n📈 PROGRESSION FEATURES (Rank Progression Prediction)")
    print(f"   Rows: {len(progression_features)} players")
    print(f"   Features: delta_winrate, delta_kda, delta_cs, delta_gold, win_streak")
    print(f"   Task: Predict if player will rank up/down/stay")
    
    print(f"\n👤 SMURF FEATURES (Smurf Detection)")
    print(f"   Rows: {len(smurf_features)} players")
    print(f"   Features: winrate_zscore, kda_zscore, dmg_share, gold_share, avg_game_time, champ_mastery_entropy")
    print(f"   Task: Detect if player is smurfing (performing above rank level)")
    
    print(f"\n🎮 MATCH FEATURES (Match Outcome Prediction)")
    print(f"   Rows: {len(match_features)} matches")
    print(f"   Features: rank_diff, kda_diff, cs_diff, gold_diff, damage_diff, vision_diff")
    print(f"   Label: team_win")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    consolidate_features()
