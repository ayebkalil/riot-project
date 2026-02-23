import pandas as pd
from pathlib import Path

by_rank_dir = Path("opgg/by_rank")

for csv_file in by_rank_dir.glob("*.csv"):
    print(f"Cleaning {csv_file.name}...")
    df = pd.read_csv(csv_file)
    
    # Remove trailing backslashes from summoner_name
    if 'summoner_name' in df.columns:
        df['summoner_name'] = df['summoner_name'].str.rstrip('\\')
    
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Cleaned {csv_file.name}")

print("\n✓ All rank CSV files cleaned!")
