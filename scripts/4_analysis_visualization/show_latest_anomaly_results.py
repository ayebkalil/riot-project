from pathlib import Path
import pandas as pd

base_path = Path("data/processed/smurf_features_with_predictions.csv")
if not base_path.exists():
    raise FileNotFoundError(f"Missing file: {base_path}")

df = pd.read_csv(base_path)

# Merge optional summoner names from players_by_rank
players_dir = Path("data/processed/players_by_rank")
name_map_frames = []
for csv_file in sorted(players_dir.glob("*.csv")):
    try:
        p = pd.read_csv(csv_file)
        if "puuid" in p.columns and "summoner_name" in p.columns:
            name_map_frames.append(p[["puuid", "summoner_name"]])
    except Exception:
        pass

if name_map_frames:
    names_df = pd.concat(name_map_frames, ignore_index=True).drop_duplicates("puuid")
    df = df.merge(names_df, on="puuid", how="left")
else:
    df["summoner_name"] = ""

# Ensure bool
if df["is_anomaly"].dtype != bool:
    df["is_anomaly"] = df["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])

anomalies = df[df["is_anomaly"]].copy()
normal = df[~df["is_anomaly"]].copy()

# Top anomalies (lowest score = most anomalous for IsolationForest score_samples)
top_anomalies = anomalies.sort_values("anomaly_score", ascending=True).head(50).copy()

# Save outputs
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

anomalies_path = out_dir / "anomalies_only_latest.csv"
normal_path = out_dir / "non_anomalies_only_latest.csv"
top_path = out_dir / "top_50_anomalies_latest.csv"

anomalies.to_csv(anomalies_path, index=False)
normal.to_csv(normal_path, index=False)
top_anomalies.to_csv(top_path, index=False)

print("\n" + "=" * 90)
print("LATEST SMURF MODEL RESULTS")
print("=" * 90)
print(f"Total players: {len(df):,}")
print(f"Anomalies: {len(anomalies):,}")
print(f"Non-anomalies: {len(normal):,}")
print(f"Anomaly rate: {len(anomalies) / len(df) * 100:.2f}%")

print("\nBy tier (anomaly count / total):")
summary = df.groupby("tier").agg(total=("puuid", "count"), anomalies=("is_anomaly", "sum"))
summary["rate_pct"] = (summary["anomalies"] / summary["total"] * 100).round(2)
summary = summary.sort_values(["rate_pct", "anomalies"], ascending=False)
print(summary.to_string())

print("\nTop 15 most anomalous players:")
cols = ["puuid", "summoner_name", "tier", "anomaly_score", "prediction", "is_anomaly"]
for col in cols:
    if col not in top_anomalies.columns:
        top_anomalies[col] = ""
print(top_anomalies[cols].head(15).to_string(index=False))

print("\nSaved files:")
print(f"- {anomalies_path}")
print(f"- {normal_path}")
print(f"- {top_path}")
