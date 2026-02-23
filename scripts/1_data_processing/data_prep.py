import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= CONFIG =================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MATCH_DIR = RAW_DIR / "matches"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_BY_RANK_DIR = PROCESSED_DIR / "players_by_rank"
MATCHES_BY_RANK_DIR = PROCESSED_DIR / "player_matches_by_rank"

RAW_DIR.mkdir(parents=True, exist_ok=True)
MATCH_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLAYERS_BY_RANK_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_BY_RANK_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("RIOT_API_KEY", "").strip()

if not API_KEY:
    print("\n❌ ERROR: RIOT_API_KEY environment variable not set!")
    print("   Set it with: $env:RIOT_API_KEY = 'your-api-key'")
else:
    print(f"✓ API Key loaded: {API_KEY[:20]}...")

REGIONS = {
    "europe": "https://europe.api.riotgames.com",
    "americas": "https://americas.api.riotgames.com",
    "asia": "https://asia.api.riotgames.com",
    "sea": "https://sea.api.riotgames.com",
    "euw1": "https://euw1.api.riotgames.com",
    "eune1": "https://eune1.api.riotgames.com",
    "na1": "https://na1.api.riotgames.com",
    "kr": "https://kr.api.riotgames.com",
    "br1": "https://br1.api.riotgames.com",
    "jp1": "https://jp1.api.riotgames.com",
    "la1": "https://la1.api.riotgames.com",
    "la2": "https://la2.api.riotgames.com",
    "oc1": "https://oc1.api.riotgames.com",
    "ph2": "https://ph2.api.riotgames.com",
    "sg2": "https://sg2.api.riotgames.com",
    "th2": "https://th2.api.riotgames.com",
    "tr1": "https://tr1.api.riotgames.com",
    "tw2": "https://tw2.api.riotgames.com",
    "vn2": "https://vn2.api.riotgames.com",
}

REGION_TO_GROUP = {
    "euw1": "europe", "eune1": "europe", "tr1": "europe",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}

TIERS = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
    "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]

DIVISIONS = ["IV", "III", "II", "I"]


# ================= UTILITIES =================

@dataclass
class RiotClient:
    api_key: str

    def __post_init__(self):
        self.session = requests.Session()
        self.request_count = 0
        self.window_start = time.time()
        self.window_duration = 120  # 2 minutes
        self.max_requests_per_window = 100
        
        retries = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        # Rate limiting: 100 requests per 2 minutes
        elapsed = time.time() - self.window_start
        if elapsed < self.window_duration:
            if self.request_count >= self.max_requests_per_window:
                wait_time = self.window_duration - elapsed
                print(f"\n⏰ RATE LIMIT: Reached 100 requests. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time + 1)
                self.request_count = 0
                self.window_start = time.time()
        else:
            self.request_count = 0
            self.window_start = time.time()
        
        self.request_count += 1
        
        headers = {"X-Riot-Token": self.api_key}
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=20)
        except requests.RequestException as e:
            self.request_count -= 1  # Don't count failed requests
            print(f"  [HTTP ERROR] {e}")
            return None

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 90))
            print(f"  [429 RATE LIMITED] Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self.get(url, params)

        if response.status_code == 403:
            print(f"  [403 FORBIDDEN] API key may be invalid or expired")
        elif response.status_code == 404:
            pass  # Silently skip 404s (summoner not found)
        else:
            print(f"  [HTTP {response.status_code}] {response.text[:100]}")

        return None


# ================= RANK HELPERS =================


def parse_rank_tier_division(rank_text: str) -> Tuple[str, str]:
    text = (rank_text or "").strip().upper()
    if not text:
        return "", ""

    tier = ""
    division = ""

    for t in TIERS:
        if t in text:
            tier = t
            break

    for d in DIVISIONS:
        if d in text:
            division = d
            break

    return tier, division


def rank_to_numeric(tier: str, division: str) -> Optional[int]:
    if not tier:
        return None

    tier = tier.upper()
    if tier not in TIERS:
        return None

    tier_index = TIERS.index(tier)

    if tier in {"MASTER", "GRANDMASTER", "CHALLENGER"}:
        division_index = 0
    else:
        division_index = DIVISIONS.index(division) if division in DIVISIONS else 3

    return tier_index * 4 + (3 - division_index)


def split_riot_id(summoner_name: str) -> Tuple[str, str]:
    if "#" in summoner_name:
        game_name, tag_line = summoner_name.split("#", 1)
        return game_name.strip(), tag_line.strip()
    return summoner_name.strip(), ""


def kda(kills: int, deaths: int, assists: int) -> float:
    return (kills + assists) / max(1, deaths)


def entropy(values: Iterable[int]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p + 1e-12, 2) for p in probs)


# ================= DATA EXTRACTION =================


def get_account_by_riot_id(client: RiotClient, game_name: str, tag_line: str, region_group: str) -> Optional[dict]:
    url = f"{REGIONS[region_group]}/riot/account/v1/accounts/by-riot-id/{requests.utils.quote(game_name)}/{requests.utils.quote(tag_line)}"
    return client.get(url)


def get_match_ids(client: RiotClient, puuid: str, region_group: str, count: int = 50) -> List[str]:
    url = f"{REGIONS[region_group]}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"queue": 420, "count": count}
    data = client.get(url, params=params)
    return data if isinstance(data, list) else []


def get_match_details(client: RiotClient, match_id: str, region_group: str) -> Optional[dict]:
    cache_path = MATCH_DIR / f"{match_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = f"{REGIONS[region_group]}/lol/match/v5/matches/{match_id}"
    data = client.get(url)
    if data:
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def extract_player_match_row(match: dict, puuid: str) -> Optional[dict]:
    info = match.get("info", {})
    participants = info.get("participants", [])

    participant = next((p for p in participants if p.get("puuid") == puuid), None)
    if not participant:
        return None

    team_id = participant.get("teamId")
    team_participants = [p for p in participants if p.get("teamId") == team_id]

    team_total_damage = sum(p.get("totalDamageDealtToChampions", 0) for p in team_participants)
    team_total_gold = sum(p.get("goldEarned", 0) for p in team_participants)

    return {
        "match_id": match.get("metadata", {}).get("matchId"),
        "puuid": puuid,
        "team_id": team_id,
        "win": int(participant.get("win", False)),
        "kills": participant.get("kills", 0),
        "deaths": participant.get("deaths", 0),
        "assists": participant.get("assists", 0),
        "cs": participant.get("totalMinionsKilled", 0),
        "gold": participant.get("goldEarned", 0),
        "damage": participant.get("totalDamageDealtToChampions", 0),
        "vision": participant.get("visionScore", 0),
        "champ": participant.get("championName", ""),
        "role": participant.get("teamPosition", ""),
        "game_duration": info.get("gameDuration", 0),
        "game_creation": info.get("gameCreation", 0),
        "dmg_share": (participant.get("totalDamageDealtToChampions", 0) / team_total_damage) if team_total_damage else 0,
        "gold_share": (participant.get("goldEarned", 0) / team_total_gold) if team_total_gold else 0,
    }


# ================= FEATURE BUILDERS =================


def build_rank_features(matches_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
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

    return pd.DataFrame(features)


def build_progression_features(matches_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    features = []
    for puuid, group in matches_df.sort_values("game_creation", ascending=False).groupby("puuid"):
        recent = group.head(window)
        if len(recent) < window:
            continue
        window_a = recent.head(window // 2)
        window_b = recent.tail(window // 2)

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

        streak = (recent["win"].astype(int).tolist())
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

    return pd.DataFrame(features)


def build_smurf_features(matches_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    merged = matches_df.merge(players_df[["puuid", "tier"]], on="puuid", how="left")
    merged["kda"] = (merged["kills"] + merged["assists"]) / merged["deaths"].replace(0, 1)
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

        winrate_std = (baseline["winrate_std"] if baseline is not None else 0) or 0
        kda_std = (baseline["kda_std"] if baseline is not None else 0) or 0
        winrate_z = ((winrate - baseline["winrate_mean"]) / (winrate_std + 1e-9)) if baseline is not None else 0
        kda_z = ((kda_avg - baseline["kda_mean"]) / (kda_std + 1e-9)) if baseline is not None else 0

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

    return pd.DataFrame(player_features)


def build_match_features(matches_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    matches_df = matches_df.merge(players_df[["puuid", "rank_numeric"]], on="puuid", how="left")

    team_agg = matches_df.groupby(["match_id", "team_id"]).agg(
        team_win=("win", "max"),
        team_avg_rank=("rank_numeric", "mean"),
        team_avg_kda=("kills", "mean"),
        team_avg_cs=("cs", "mean"),
        team_avg_gold=("gold", "mean"),
        team_avg_damage=("damage", "mean"),
        team_avg_vision=("vision", "mean"),
    ).reset_index()

    merged = team_agg.merge(
        team_agg,
        on="match_id",
        suffixes=("", "_opp"),
    )
    merged = merged[merged["team_id"] != merged["team_id_opp"]]

    merged["rank_diff"] = merged["team_avg_rank"] - merged["team_avg_rank_opp"]
    merged["kda_diff"] = merged["team_avg_kda"] - merged["team_avg_kda_opp"]
    merged["cs_diff"] = merged["team_avg_cs"] - merged["team_avg_cs_opp"]
    merged["gold_diff"] = merged["team_avg_gold"] - merged["team_avg_gold_opp"]
    merged["damage_diff"] = merged["team_avg_damage"] - merged["team_avg_damage_opp"]
    merged["vision_diff"] = merged["team_avg_vision"] - merged["team_avg_vision_opp"]

    return merged[[
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


# ================= MAIN PIPELINE =================


def build_datasets(
    summoners_csv: Path,
    region: str,
    match_count: int = 50,
) -> None:
    if not API_KEY:
        raise SystemExit("RIOT_API_KEY is missing. Set it in your environment before running.")

    client = RiotClient(API_KEY)
    summoners = pd.read_csv(summoners_csv)

    players_rows = []
    matches_rows = []

    for _, row in summoners.iterrows():
        summoner_name = str(row.get("summoner_name", "")).strip()
        if not summoner_name:
            continue

        tier, division = parse_rank_tier_division(str(row.get("rank_tier", "")))
        rank_numeric = rank_to_numeric(tier, division)

        game_name, tag_line = split_riot_id(summoner_name)
        if not tag_line:
            continue

        region_group = REGION_TO_GROUP.get(region, "europe")
        account = get_account_by_riot_id(client, game_name, tag_line, region_group)
        if not account:
            continue

        puuid = account.get("puuid")
        players_rows.append({
            "puuid": puuid,
            "summoner_name": summoner_name,
            "tier": tier.title() if tier else "",
            "division": division,
            "rank_numeric": rank_numeric,
            "region": region.upper(),
        })

        match_ids = get_match_ids(client, puuid, region_group, count=match_count)
        for match_id in match_ids:
            match = get_match_details(client, match_id, region_group)
            if not match:
                continue

            row_match = extract_player_match_row(match, puuid)
            if row_match:
                matches_rows.append(row_match)

        time.sleep(1.1)

    players_df = pd.DataFrame(players_rows).drop_duplicates("puuid")
    matches_df = pd.DataFrame(matches_rows)

    players_path = PROCESSED_DIR / "players.csv"
    matches_path = PROCESSED_DIR / "player_matches.csv"

    players_df.to_csv(players_path, index=False)
    matches_df.to_csv(matches_path, index=False)

    rank_features = build_rank_features(matches_df)
    rank_features = rank_features.merge(
        players_df[["puuid", "tier", "division", "rank_numeric"]],
        on="puuid",
        how="left",
    )
    progression_features = build_progression_features(matches_df)
    smurf_features = build_smurf_features(matches_df, players_df)
    match_features = build_match_features(matches_df, players_df)

    rank_features.to_csv(PROCESSED_DIR / "rank_features.csv", index=False)
    progression_features.to_csv(PROCESSED_DIR / "progression_features.csv", index=False)
    smurf_features.to_csv(PROCESSED_DIR / "smurf_features.csv", index=False)
    match_features.to_csv(PROCESSED_DIR / "match_features.csv", index=False)


def build_datasets_by_rank(
    by_rank_dir: Path,
    region: str,
    match_count: int = 50,
) -> None:
    if not API_KEY:
        raise SystemExit("RIOT_API_KEY is missing. Set it in your environment before running.")

    client = RiotClient(API_KEY)
    region_group = REGION_TO_GROUP.get(region, "europe")

    rank_files = sorted(by_rank_dir.glob("*.csv"))
    if not rank_files:
        raise SystemExit(f"No rank CSV files found in {by_rank_dir}")

    for rank_file in rank_files:
        rank_name = rank_file.stem.lower()
        print(f"\n{'='*60}")
        print(f"Processing rank: {rank_name.upper()}")
        print(f"{'='*60}")
        
        summoners = pd.read_csv(rank_file)
        players_path = PLAYERS_BY_RANK_DIR / f"{rank_name}.csv"

        # ===== STEP 1: Fetch PUUIDs (skip if already exists) =====
        if players_path.exists():
            print(f"✓ PUUIDs already saved. Loading from {players_path}")
            players_df = pd.read_csv(players_path)
        else:
            print(f"[STEP 1] Fetching PUUIDs...")
            players_rows = []
            failed_count = 0

            for idx, row in summoners.iterrows():
                summoner_name = str(row.get("summoner_name", "")).strip()
                if not summoner_name:
                    continue

                tier, division = parse_rank_tier_division(str(row.get("rank_tier", "")))
                rank_numeric = rank_to_numeric(tier, division)

                game_name, tag_line = split_riot_id(summoner_name)
                if not tag_line:
                    continue

                account = get_account_by_riot_id(client, game_name, tag_line, region_group)
                if not account:
                    failed_count += 1
                    continue

                puuid = account.get("puuid")
                players_rows.append({
                    "puuid": puuid,
                    "summoner_name": summoner_name,
                    "tier": tier.title() if tier else "",
                    "division": division,
                    "rank_numeric": rank_numeric,
                    "region": region.upper(),
                })

                if (idx + 1) % 10 == 0:
                    print(f"  [PUUID] {idx + 1}/{len(summoners)} summoners | {len(players_rows)} PUUIDs found | Failed: {failed_count} | Requests: {client.request_count}")
                
                time.sleep(0.1)

            players_df = pd.DataFrame(players_rows).drop_duplicates("puuid")
            players_df.to_csv(players_path, index=False)
            print(f"\n✓ Saved {len(players_df)} unique PUUIDs → {players_path}")

        # ===== STEP 2: Fetch Match IDs & Details (skip if already exists) =====
        matches_path = MATCHES_BY_RANK_DIR / f"{rank_name}.csv"
        if matches_path.exists():
            print(f"✓ Matches already saved. Skipping match fetch for {rank_name.upper()}")
            print(f"{'='*60}\n")
            continue

        print(f"[STEP 2] Fetching matches for {len(players_df)} players...")
        matches_rows = []

        for idx, row in players_df.iterrows():
            puuid = row.get("puuid")
            summoner_name = row.get("summoner_name")
            if not puuid:
                continue

            match_ids = get_match_ids(client, puuid, region_group, count=match_count)
            
            for match_idx, match_id in enumerate(match_ids):
                match = get_match_details(client, match_id, region_group)
                if not match:
                    continue

                row_match = extract_player_match_row(match, puuid)
                if row_match:
                    matches_rows.append(row_match)

            if (idx + 1) % 10 == 0:
                print(f"  [MATCHES] {idx + 1}/{len(players_df)} players | {len(matches_rows)} matches | Requests: {client.request_count}")

            time.sleep(0.1)

        matches_df = pd.DataFrame(matches_rows)
        matches_path = MATCHES_BY_RANK_DIR / f"{rank_name}.csv"
        matches_df.to_csv(matches_path, index=False)
        print(f"\n✓ Saved {len(matches_df)} match records → {matches_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    build_datasets_by_rank(
        by_rank_dir=Path("opgg/by_rank"),
        region="euw1",
        match_count=50,
    )
