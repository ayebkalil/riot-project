"""
Add temporal/streak features for smurf detection from match history.

Outputs:
- data/processed/smurf_features.csv (updated with new columns)
- data/processed/smurf_features_with_predictions.csv (updated with new columns)
"""

from pathlib import Path
import numpy as np
import pandas as pd


MATCHES_DIR = Path("data/processed/player_matches_by_rank")
SMURF_FEATURES = Path("data/processed/smurf_features.csv")
SMURF_WITH_PRED = Path("data/processed/smurf_features_with_predictions.csv")


def longest_streak(values, target=1):
    best = 0
    cur = 0
    for value in values:
        if value == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def current_streak(values, target=1):
    cur = 0
    for value in values:
        if value == target:
            cur += 1
        else:
            break
    return cur


def build_temporal_features(matches_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rows = []

    for puuid, group in matches_df.groupby("puuid"):
        recent = group.sort_values("game_creation", ascending=False).head(window).copy()
        if recent.empty:
            continue

        wins = recent["win"].astype(int).tolist()
        kda_series = ((recent["kills"] + recent["assists"]) / recent["deaths"].replace(0, 1)).tolist()

        first_5_wins = wins[:5]
        first_10_wins = wins[:10]
        prev_10_wins = wins[10:20]

        first_5_kda = kda_series[:5]
        first_10_kda = kda_series[:10]
        prev_10_kda = kda_series[10:20]

        recent_winrate_5 = float(np.mean(first_5_wins)) if first_5_wins else 0.0
        recent_winrate_10 = float(np.mean(first_10_wins)) if first_10_wins else 0.0
        prev_winrate_10 = float(np.mean(prev_10_wins)) if prev_10_wins else recent_winrate_10

        recent_kda_5 = float(np.mean(first_5_kda)) if first_5_kda else 0.0
        recent_kda_10 = float(np.mean(first_10_kda)) if first_10_kda else 0.0
        prev_kda_10 = float(np.mean(prev_10_kda)) if prev_10_kda else recent_kda_10

        rows.append(
            {
                "puuid": puuid,
                "current_win_streak": int(current_streak(wins, 1)),
                "current_loss_streak": int(current_streak(wins, 0)),
                "longest_win_streak_20": int(longest_streak(wins, 1)),
                "longest_loss_streak_20": int(longest_streak(wins, 0)),
                "recent_winrate_5": recent_winrate_5,
                "recent_winrate_10": recent_winrate_10,
                "winrate_trend_10": recent_winrate_10 - prev_winrate_10,
                "recent_kda_5": recent_kda_5,
                "recent_kda_10": recent_kda_10,
                "kda_trend_10": recent_kda_10 - prev_kda_10,
                "kda_volatility_10": float(np.std(first_10_kda)) if first_10_kda else 0.0,
            }
        )

    return pd.DataFrame(rows)


def load_all_matches() -> pd.DataFrame:
    frames = []
    for file in sorted(MATCHES_DIR.glob("*.csv")):
        frames.append(pd.read_csv(file))
    if not frames:
        raise RuntimeError("No match files found in data/processed/player_matches_by_rank")
    return pd.concat(frames, ignore_index=True)


def merge_and_save(target_path: Path, temporal_df: pd.DataFrame):
    if not target_path.exists():
        return

    df = pd.read_csv(target_path)
    existing_temporal = [col for col in temporal_df.columns if col != "puuid" and col in df.columns]
    if existing_temporal:
        df = df.drop(columns=existing_temporal)

    merged = df.merge(temporal_df, on="puuid", how="left")

    new_cols = [c for c in temporal_df.columns if c != "puuid"]
    for col in new_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    merged.to_csv(target_path, index=False)
    print(f"Updated: {target_path} (+{len(new_cols)} temporal features)")


def main():
    print("\n" + "=" * 80)
    print("ADDING TEMPORAL FEATURES TO SMURF DATA")
    print("=" * 80)

    matches_df = load_all_matches()
    print(f"Loaded matches: {len(matches_df):,}")

    temporal_df = build_temporal_features(matches_df, window=20)
    print(f"Built temporal rows: {len(temporal_df):,}")

    merge_and_save(SMURF_FEATURES, temporal_df)
    merge_and_save(SMURF_WITH_PRED, temporal_df)

    print("\nNew features added:")
    print("- current_win_streak")
    print("- current_loss_streak")
    print("- longest_win_streak_20")
    print("- longest_loss_streak_20")
    print("- recent_winrate_5")
    print("- recent_winrate_10")
    print("- winrate_trend_10")
    print("- recent_kda_5")
    print("- recent_kda_10")
    print("- kda_trend_10")
    print("- kda_volatility_10")


if __name__ == "__main__":
    main()
