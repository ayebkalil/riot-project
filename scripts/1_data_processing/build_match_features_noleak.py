from pathlib import Path
import pandas as pd

MATCHES_DIR = Path("data/processed/player_matches_by_rank")
PLAYERS_DIR = Path("data/processed/players_by_rank")
OUT_PATH = Path("data/processed/match_features_noleak.csv")

TIER_TO_NUM = {
    "IRON": 1,
    "BRONZE": 2,
    "SILVER": 3,
    "GOLD": 4,
    "PLATINUM": 5,
    "EMERALD": 6,
    "DIAMOND": 7,
    "MASTER": 8,
    "GRANDMASTER": 9,
    "CHALLENGER": 10,
}


def load_players():
    frames = []
    for file in sorted(PLAYERS_DIR.glob("*.csv")):
        df = pd.read_csv(file)
        frames.append(df)
    players = pd.concat(frames, ignore_index=True).drop_duplicates("puuid")
    players["tier"] = players["tier"].astype(str).str.upper()
    players["rank_numeric"] = players["tier"].map(TIER_TO_NUM)
    players["rank_numeric"] = players["rank_numeric"].fillna(players["rank_numeric"].median())
    return players[["puuid", "rank_numeric"]]


def load_matches():
    frames = []
    for file in sorted(MATCHES_DIR.glob("*.csv")):
        frames.append(pd.read_csv(file))
    return pd.concat(frames, ignore_index=True)


def main():
    print("\n" + "=" * 80)
    print("BUILDING NO-LEAK MATCH FEATURES")
    print("=" * 80)

    players = load_players()
    matches = load_matches()

    print(f"Players: {len(players):,}")
    print(f"Match rows: {len(matches):,}")

    merged = matches.merge(players, on="puuid", how="left")
    merged["rank_numeric"] = merged["rank_numeric"].fillna(merged["rank_numeric"].median())

    team_agg = merged.groupby(["match_id", "team_id"], as_index=False).agg(
        team_won=("win", "max"),
        team_avg_rank=("rank_numeric", "mean"),
    )

    pairs = team_agg.merge(team_agg, on="match_id", suffixes=("", "_opp"))
    pairs = pairs[pairs["team_id"] != pairs["team_id_opp"]].copy()

    pairs["rank_diff"] = pairs["team_avg_rank"] - pairs["team_avg_rank_opp"]

    result = pairs[["match_id", "team_id", "team_won", "rank_diff"]]
    result.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(result):,}")
    print("Columns: match_id, team_id, team_won, rank_diff")


if __name__ == "__main__":
    main()
