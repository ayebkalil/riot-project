"""
Download Riot match timelines for locally cached match JSON files.

Reads match IDs from data/raw/matches/*.json and saves timeline files to data/raw/timelines.
Supports retry handling for 429/5xx and auto-detects routing region from match ID prefix.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests


REGIONAL_BASE_URLS = {
    "europe": "https://europe.api.riotgames.com",
    "americas": "https://americas.api.riotgames.com",
    "asia": "https://asia.api.riotgames.com",
    "sea": "https://sea.api.riotgames.com",
}


PLATFORM_TO_REGIONAL = {
    "EUW1": "europe",
    "EUN1": "europe",
    "TR1": "europe",
    "RU": "europe",
    "NA1": "americas",
    "BR1": "americas",
    "LA1": "americas",
    "LA2": "americas",
    "KR": "asia",
    "JP1": "asia",
    "OC1": "sea",
    "PH2": "sea",
    "SG2": "sea",
    "TH2": "sea",
    "TW2": "sea",
    "VN2": "sea",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download match timelines from Riot API.")
    parser.add_argument(
        "--matches-dir",
        type=Path,
        default=Path("data/raw/matches"),
        help="Directory with existing match JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/timelines"),
        help="Directory to save timeline JSON files",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Riot API key (optional, defaults to RIOT_API_KEY env)",
    )
    parser.add_argument(
        "--default-region",
        type=str,
        default="europe",
        choices=list(REGIONAL_BASE_URLS.keys()),
        help="Fallback regional routing when match ID prefix is unknown",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Max number of match files to process (0 = all)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Sleep seconds between successful requests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download timelines even if output file already exists",
    )
    return parser.parse_args()


def extract_match_id(match_file: Path) -> Optional[str]:
    try:
        with open(match_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        match_id = payload.get("metadata", {}).get("matchId")
        if isinstance(match_id, str) and "_" in match_id:
            return match_id
    except Exception:
        pass

    stem = match_file.stem
    return stem if "_" in stem else None


def infer_regional_group(match_id: str, default_region: str) -> str:
    platform = match_id.split("_", 1)[0].upper()
    return PLATFORM_TO_REGIONAL.get(platform, default_region)


def request_timeline(
    session: requests.Session,
    api_key: str,
    match_id: str,
    regional_group: str,
    max_attempts: int = 5,
) -> Optional[dict]:
    base_url = REGIONAL_BASE_URLS[regional_group]
    url = f"{base_url}/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": api_key}

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, headers=headers, timeout=30)
        except requests.RequestException:
            if attempt == max_attempts:
                return None
            time.sleep(min(2**attempt, 20))
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return None

        if response.status_code == 404:
            return None

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "2"))
            time.sleep(max(retry_after, 1))
            continue

        if response.status_code in {500, 502, 503, 504}:
            if attempt == max_attempts:
                return None
            time.sleep(min(2**attempt, 20))
            continue

        return None

    return None


def main() -> None:
    args = parse_args()
    api_key = (args.api_key or os.getenv("RIOT_API_KEY", "")).strip()

    if not api_key:
        raise SystemExit("RIOT_API_KEY is missing. Set env var or pass --api-key.")

    if not args.matches_dir.exists():
        raise SystemExit(f"Matches directory not found: {args.matches_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    match_files = sorted(args.matches_dir.glob("*.json"))
    if args.max_matches and args.max_matches > 0:
        match_files = match_files[: args.max_matches]

    print("\n" + "=" * 80)
    print("DOWNLOAD MATCH TIMELINES")
    print("=" * 80)
    print(f"matches dir : {args.matches_dir}")
    print(f"output dir  : {args.output_dir}")
    print(f"files found : {len(match_files):,}")
    print(f"overwrite   : {args.overwrite}")
    print(f"default reg : {args.default_region}")

    stats = {
        "processed": 0,
        "downloaded": 0,
        "skipped_existing": 0,
        "missing_match_id": 0,
        "failed": 0,
    }

    session = requests.Session()

    for idx, match_file in enumerate(match_files, start=1):
        stats["processed"] += 1
        match_id = extract_match_id(match_file)
        if not match_id:
            stats["missing_match_id"] += 1
            continue

        out_file = args.output_dir / f"{match_id}.json"
        if out_file.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            continue

        regional_group = infer_regional_group(match_id, args.default_region)
        timeline_data = request_timeline(session, api_key, match_id, regional_group)

        if not timeline_data:
            stats["failed"] += 1
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, ensure_ascii=False)

        stats["downloaded"] += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 200 == 0:
            print(
                f"Progress {idx:,}/{len(match_files):,} | downloaded={stats['downloaded']:,} "
                f"skipped={stats['skipped_existing']:,} failed={stats['failed']:,}"
            )

    print("\n" + "=" * 80)
    print("TIMELINE DOWNLOAD COMPLETE")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key:>17}: {value:,}")
    print(f"saved dir        : {args.output_dir}")


if __name__ == "__main__":
    main()
