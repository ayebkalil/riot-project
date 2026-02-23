"""
Split op.gg leaderboards CSV by rank tier.
Creates individual CSV files for each rank with headers.
"""

import pandas as pd
from pathlib import Path

# File paths
input_csv = Path("data/opgg/opgg_leaderboards.csv")
output_dir = Path("data/opgg/by_rank")

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV
df = pd.read_csv(input_csv)

print(f"Total players: {len(df)}")
print(f"Rank tiers found: {df['rank_tier'].unique().tolist()}")
print()

# Split by rank tier
for rank in sorted(df['rank_tier'].unique()):
    rank_df = df[df['rank_tier'] == rank]
    output_file = output_dir / f"{rank.lower()}.csv"
    rank_df.to_csv(output_file, index=False)
    print(f"✓ {rank.capitalize():12} - {len(rank_df):5} players → {output_file}")

print()
print(f"✓ All rank files created in {output_dir}")
