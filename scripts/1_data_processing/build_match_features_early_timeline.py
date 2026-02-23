"""
Build match outcome features from early-game timeline snapshots (e.g., 10m or 15m).

This keeps the same feature schema as `generate_match_features.py`:
gold_diff, damage_diff, kills_diff, deaths_diff, assists_diff,
vision_diff, turrets_diff, dragons_diff, barons_diff, cs_diff

but computes values only up to a target minute from timeline data.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


FEATURE_COLUMNS = [
    "match_id",
    "team_id",
    "team_won",
    "gold_diff",
    "damage_diff",
    "kills_diff",
    "deaths_diff",
    "assists_diff",
    "vision_diff",
    "turrets_diff",
    "dragons_diff",
    "barons_diff",
    "cs_diff",
]


def load_json(file_path: Path) -> Optional[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_timeline_file(match_id: str, timelines_dir: Path) -> Optional[Path]:
    direct_candidates = [
        timelines_dir / f"{match_id}.json",
        timelines_dir / f"timeline_{match_id}.json",
        timelines_dir / f"{match_id}_timeline.json",
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    fuzzy = list(timelines_dir.glob(f"*{match_id}*.json"))
    return fuzzy[0] if fuzzy else None


def build_participant_team_map(match_data: dict) -> Dict[int, int]:
    participant_team = {}
    for participant in match_data.get("info", {}).get("participants", []):
        participant_id = participant.get("participantId")
        team_id = participant.get("teamId")
        if participant_id is not None and team_id in (100, 200):
            participant_team[int(participant_id)] = int(team_id)
    return participant_team


def get_winner_team_id(match_data: dict) -> Optional[int]:
    for team in match_data.get("info", {}).get("teams", []):
        if team.get("win"):
            return team.get("teamId")
    return None


def select_frame_at_or_before(frames: List[dict], target_ms: int) -> Optional[dict]:
    if not frames:
        return None

    selected = None
    for frame in frames:
        timestamp = int(frame.get("timestamp", 0))
        if timestamp <= target_ms:
            selected = frame
        else:
            break

    if selected is None:
        selected = frames[0]
    return selected


def _safe_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def aggregate_frame_stats(frame: dict, participant_team: Dict[int, int]) -> Dict[int, Dict[str, int]]:
    team_stats = {
        100: {"gold": 0, "damage": 0, "cs": 0},
        200: {"gold": 0, "damage": 0, "cs": 0},
    }

    participant_frames = frame.get("participantFrames", {})

    if isinstance(participant_frames, dict):
        iterable = participant_frames.items()
    elif isinstance(participant_frames, list):
        iterable = [(pf.get("participantId"), pf) for pf in participant_frames if isinstance(pf, dict)]
    else:
        iterable = []

    for raw_pid, pf in iterable:
        if not isinstance(pf, dict):
            continue

        participant_id = pf.get("participantId", raw_pid)
        participant_id = _safe_int(participant_id)
        team_id = participant_team.get(participant_id)
        if team_id not in (100, 200):
            continue

        damage_stats = pf.get("damageStats", {}) if isinstance(pf.get("damageStats", {}), dict) else {}

        team_stats[team_id]["gold"] += _safe_int(pf.get("totalGold", 0))
        team_stats[team_id]["cs"] += _safe_int(pf.get("minionsKilled", 0)) + _safe_int(
            pf.get("jungleMinionsKilled", 0)
        )
        team_stats[team_id]["damage"] += _safe_int(damage_stats.get("totalDamageDoneToChampions", 0))

    return team_stats


def _resolve_event_team(event: dict, participant_team: Dict[int, int]) -> Optional[int]:
    team_id = event.get("killerTeamId")
    if team_id in (100, 200):
        return team_id

    team_id = event.get("teamId")
    if team_id in (100, 200):
        return team_id

    killer_id = _safe_int(event.get("killerId", 0))
    if killer_id > 0:
        return participant_team.get(killer_id)

    creator_id = _safe_int(event.get("creatorId", 0))
    if creator_id > 0:
        return participant_team.get(creator_id)

    return None


def aggregate_event_stats(frames: List[dict], participant_team: Dict[int, int], target_ms: int) -> Dict[int, Dict[str, int]]:
    team_stats = {
        100: {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "vision": 0,
            "turrets": 0,
            "dragons": 0,
            "barons": 0,
        },
        200: {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "vision": 0,
            "turrets": 0,
            "dragons": 0,
            "barons": 0,
        },
    }

    for frame in frames:
        events = frame.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue

            timestamp = _safe_int(event.get("timestamp", 0))
            if timestamp > target_ms:
                continue

            event_type = str(event.get("type", "")).upper()

            if event_type == "CHAMPION_KILL":
                killer_id = _safe_int(event.get("killerId", 0))
                victim_id = _safe_int(event.get("victimId", 0))
                killer_team = participant_team.get(killer_id)
                victim_team = participant_team.get(victim_id)

                if killer_team in (100, 200) and killer_id > 0:
                    team_stats[killer_team]["kills"] += 1

                if victim_team in (100, 200):
                    team_stats[victim_team]["deaths"] += 1

                assists = event.get("assistingParticipantIds", [])
                if isinstance(assists, list):
                    for assist_pid in assists:
                        assist_team = participant_team.get(_safe_int(assist_pid))
                        if assist_team in (100, 200):
                            team_stats[assist_team]["assists"] += 1

            elif event_type in ("WARD_PLACED", "WARD_KILL"):
                team_id = _resolve_event_team(event, participant_team)
                if team_id in (100, 200):
                    team_stats[team_id]["vision"] += 1

            elif event_type == "BUILDING_KILL":
                building_type = str(event.get("buildingType", "")).upper()
                if "TOWER" in building_type or "TURRET" in building_type:
                    team_id = _resolve_event_team(event, participant_team)
                    if team_id in (100, 200):
                        team_stats[team_id]["turrets"] += 1

            elif event_type == "ELITE_MONSTER_KILL":
                team_id = _resolve_event_team(event, participant_team)
                if team_id not in (100, 200):
                    continue

                monster_type = str(event.get("monsterType", "")).upper()
                if "DRAGON" in monster_type:
                    team_stats[team_id]["dragons"] += 1
                elif "BARON" in monster_type:
                    team_stats[team_id]["barons"] += 1

    return team_stats


def extract_early_team_rows(match_data: dict, timeline_data: dict, minute: int) -> List[dict]:
    match_id = match_data.get("metadata", {}).get("matchId", "")
    winner_team = get_winner_team_id(match_data)
    participant_team = build_participant_team_map(match_data)

    frames = timeline_data.get("info", {}).get("frames", [])
    if not isinstance(frames, list) or not frames:
        return []

    target_ms = minute * 60 * 1000
    frame = select_frame_at_or_before(frames, target_ms)
    if frame is None:
        return []

    frame_stats = aggregate_frame_stats(frame, participant_team)
    event_stats = aggregate_event_stats(frames, participant_team, target_ms)

    team_stats = {
        100: {
            "gold": frame_stats[100]["gold"],
            "damage": frame_stats[100]["damage"],
            "kills": event_stats[100]["kills"],
            "deaths": event_stats[100]["deaths"],
            "assists": event_stats[100]["assists"],
            "vision": event_stats[100]["vision"],
            "turrets": event_stats[100]["turrets"],
            "dragons": event_stats[100]["dragons"],
            "barons": event_stats[100]["barons"],
            "cs": frame_stats[100]["cs"],
        },
        200: {
            "gold": frame_stats[200]["gold"],
            "damage": frame_stats[200]["damage"],
            "kills": event_stats[200]["kills"],
            "deaths": event_stats[200]["deaths"],
            "assists": event_stats[200]["assists"],
            "vision": event_stats[200]["vision"],
            "turrets": event_stats[200]["turrets"],
            "dragons": event_stats[200]["dragons"],
            "barons": event_stats[200]["barons"],
            "cs": frame_stats[200]["cs"],
        },
    }

    rows = []
    for team_id in (100, 200):
        enemy_id = 200 if team_id == 100 else 100
        rows.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "team_won": 1 if winner_team == team_id else 0,
                "gold_diff": team_stats[team_id]["gold"] - team_stats[enemy_id]["gold"],
                "damage_diff": team_stats[team_id]["damage"] - team_stats[enemy_id]["damage"],
                "kills_diff": team_stats[team_id]["kills"] - team_stats[enemy_id]["kills"],
                "deaths_diff": team_stats[team_id]["deaths"] - team_stats[enemy_id]["deaths"],
                "assists_diff": team_stats[team_id]["assists"] - team_stats[enemy_id]["assists"],
                "vision_diff": team_stats[team_id]["vision"] - team_stats[enemy_id]["vision"],
                "turrets_diff": team_stats[team_id]["turrets"] - team_stats[enemy_id]["turrets"],
                "dragons_diff": team_stats[team_id]["dragons"] - team_stats[enemy_id]["dragons"],
                "barons_diff": team_stats[team_id]["barons"] - team_stats[enemy_id]["barons"],
                "cs_diff": team_stats[team_id]["cs"] - team_stats[enemy_id]["cs"],
            }
        )
    return rows


def build_dataset(matches_dir: Path, timelines_dir: Path, minute: int) -> Tuple[pd.DataFrame, dict]:
    rows: List[dict] = []
    stats = {
        "match_files": 0,
        "with_timeline": 0,
        "processed_matches": 0,
        "missing_timeline": 0,
        "invalid_match": 0,
        "invalid_timeline": 0,
        "empty_rows": 0,
    }

    match_files = sorted(matches_dir.glob("*.json"))
    stats["match_files"] = len(match_files)

    for match_file in match_files:
        match_data = load_json(match_file)
        if not match_data:
            stats["invalid_match"] += 1
            continue

        match_id = match_data.get("metadata", {}).get("matchId")
        if not match_id:
            stats["invalid_match"] += 1
            continue

        timeline_file = find_timeline_file(match_id, timelines_dir)
        if timeline_file is None:
            stats["missing_timeline"] += 1
            continue

        timeline_data = load_json(timeline_file)
        if not timeline_data:
            stats["invalid_timeline"] += 1
            continue

        stats["with_timeline"] += 1

        team_rows = extract_early_team_rows(match_data, timeline_data, minute)
        if not team_rows:
            stats["empty_rows"] += 1
            continue

        rows.extend(team_rows)
        stats["processed_matches"] += 1

    if rows:
        df = pd.DataFrame(rows)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_COLUMNS]
    else:
        df = pd.DataFrame(columns=FEATURE_COLUMNS)

    return df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build early-game match features from timeline data.")
    parser.add_argument("--minute", type=int, default=15, help="Target minute for snapshot features (default: 15)")
    parser.add_argument(
        "--matches-dir",
        type=Path,
        default=Path("data/raw/matches"),
        help="Directory containing match JSON files",
    )
    parser.add_argument(
        "--timelines-dir",
        type=Path,
        default=Path("data/raw/timelines"),
        help="Directory containing timeline JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed/match_features_early_<minute>m.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.minute <= 0:
        raise ValueError("--minute must be > 0")

    output_path = args.output or Path(f"data/processed/match_features_early_{args.minute}m.csv")

    print("\n" + "=" * 80)
    print(f"BUILD EARLY MATCH FEATURES @ {args.minute} MIN")
    print("=" * 80)
    print(f"Matches dir  : {args.matches_dir}")
    print(f"Timelines dir: {args.timelines_dir}")
    print(f"Output       : {output_path}\n")

    if not args.matches_dir.exists():
        print(f"[ERROR] matches directory not found: {args.matches_dir}")
        return

    if not args.timelines_dir.exists():
        print(f"[ERROR] timelines directory not found: {args.timelines_dir}")
        print("        Create timeline JSON files first, then run this script again.")
        return

    df, stats = build_dataset(args.matches_dir, args.timelines_dir, args.minute)

    if df.empty:
        print("[ERROR] No rows generated. Check timeline files and match IDs.")
        print(f"Stats: {stats}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[OK] Saved: {output_path}")
    print(f"[OK] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[OK] Columns: {list(df.columns)}")
    print(f"[OK] Team won distribution: {df['team_won'].value_counts().to_dict()}")
    print(f"[INFO] Stats: {stats}")


if __name__ == "__main__":
    main()
