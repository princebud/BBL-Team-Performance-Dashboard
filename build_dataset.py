#!/usr/bin/env python3
"""Step 1: turn a Cricsheet JSON download into the two CSV files this project uses.

Usage:
    python build_dataset.py data/<the_zip_you_downloaded>.zip
    python build_dataset.py path/to/a/folder/of/json/files

Writes into data/ (change with --out-dir):
    bbl_matches.csv        one row per match
    bbl_ball_by_ball.csv   one row per delivery

Why this step exists: Cricsheet publishes one JSON file per match. The rest of
the project (data_processing.py, app.py, the notebook) works on two flat tables,
so this script flattens the JSON into them. Nothing is cleaned or filtered here;
cleaning happens in data_processing.py.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

MATCH_COLUMNS = [
    "match_id", "date", "city", "venue", "team1", "team2", "toss_winner", "toss_decision",
    "winner", "result_type", "player_of_match", "season", "event_name", "match_type",
    "gender", "overs", "balls_per_over", "team1_won", "toss_winner_won",
]
BALL_COLUMNS = [
    "match_id", "date", "season", "venue", "city", "innings", "batting_team", "bowling_team",
    "over", "ball_in_over", "ball_number", "batter", "bowler", "non_striker", "batter_runs",
    "extra_runs", "total_runs", "wides", "noballs", "byes", "legbyes", "penalty", "is_wicket",
    "wicket_player_out", "wicket_kind", "match_winner", "toss_winner", "toss_decision",
    "team1", "team2", "is_powerplay", "is_middle_overs", "is_death_overs", "batting_team_won",
]


def read_json_files(source: Path):
    """Yield (match_id, parsed_json) for every match file in a zip or a folder."""
    if source.is_dir():
        for path in sorted(source.glob("*.json")):
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
    else:
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".json"):
                    yield Path(name).stem, json.loads(zf.read(name))


def parse_match(match_id: str, data: dict):
    """Return (match_row, list_of_delivery_rows), or None if the file is not a usable match."""
    info = data.get("info", {})
    teams = info.get("teams", [])
    if len(teams) != 2 or not data.get("innings"):
        return None
    team1, team2 = teams
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})
    winner = outcome.get("winner", "")          # blank for ties and no-results
    balls_per_over = int(info.get("balls_per_over", 6))
    date = (info.get("dates") or [""])[0]
    season = str(info.get("season", ""))
    venue, city = info.get("venue", ""), info.get("city", "")
    toss_winner = toss.get("winner", "")
    toss_decision = toss.get("decision", "")

    match_row = {
        "match_id": match_id, "date": date, "city": city, "venue": venue,
        "team1": team1, "team2": team2, "toss_winner": toss_winner,
        "toss_decision": toss_decision, "winner": winner,
        "result_type": str(list(outcome.keys())),
        "player_of_match": (info.get("player_of_match") or [""])[0],
        "season": season, "event_name": info.get("event", {}).get("name", ""),
        "match_type": info.get("match_type", ""), "gender": info.get("gender", ""),
        "overs": info.get("overs", ""), "balls_per_over": balls_per_over,
        "team1_won": int(winner == team1), "toss_winner_won": int(bool(winner) and toss_winner == winner),
    }

    rows = []
    # Cricsheet lists super-over innings after the main two, so position gives innings 3, 4, ...
    for innings_no, innings in enumerate(data["innings"], start=1):
        batting = innings["team"]
        bowling = team2 if batting == team1 else team1
        for over_block in innings.get("overs", []):
            over = over_block["over"]                       # 0-based: 0 is the first over
            for ball_in_over, d in enumerate(over_block["deliveries"], start=1):
                runs, extras = d["runs"], d.get("extras", {})
                wickets = d.get("wickets") or []
                first_wicket = wickets[0] if wickets else {}
                rows.append({
                    "match_id": match_id, "date": date, "season": season,
                    "venue": venue, "city": city, "innings": innings_no,
                    "batting_team": batting, "bowling_team": bowling,
                    "over": over, "ball_in_over": ball_in_over,
                    "ball_number": over * balls_per_over + ball_in_over,
                    "batter": d["batter"], "bowler": d["bowler"], "non_striker": d["non_striker"],
                    "batter_runs": runs["batter"], "extra_runs": runs["extras"],
                    "total_runs": runs["total"],
                    "wides": extras.get("wides", 0), "noballs": extras.get("noballs", 0),
                    "byes": extras.get("byes", 0), "legbyes": extras.get("legbyes", 0),
                    "penalty": extras.get("penalty", 0),
                    "is_wicket": int(bool(wickets)),
                    "wicket_player_out": first_wicket.get("player_out", ""),
                    "wicket_kind": first_wicket.get("kind", ""),
                    "match_winner": winner, "toss_winner": toss_winner,
                    "toss_decision": toss_decision, "team1": team1, "team2": team2,
                    "is_powerplay": int(over <= 5),
                    "is_middle_overs": int(6 <= over <= 14),
                    "is_death_overs": int(over >= 15),
                    "batting_team_won": int(bool(winner) and batting == winner),
                })
    return match_row, rows


def main():
    ap = argparse.ArgumentParser(description="Flatten a Cricsheet JSON download into two CSV files.")
    ap.add_argument("source", type=Path, help="The downloaded .zip, or a folder of .json files")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()
    if not args.source.exists():
        raise SystemExit(f"Cannot find {args.source}. Check the path and try again.")

    matches, balls, skipped = [], [], []
    for match_id, data in read_json_files(args.source):
        parsed = parse_match(match_id, data)
        if parsed is None:
            skipped.append(match_id)
            continue
        matches.append(parsed[0])
        balls.extend(parsed[1])
    if not matches:
        raise SystemExit("No matches found. Is this a Cricsheet JSON download?")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    b = pd.DataFrame(balls, columns=BALL_COLUMNS)
    m.to_csv(args.out_dir / "bbl_matches.csv", index=False)
    b.to_csv(args.out_dir / "bbl_ball_by_ball.csv", index=False)

    print(f"Matches:    {len(m):,}")
    print(f"Deliveries: {len(b):,}")
    print(f"Seasons:    {m['season'].nunique()} ({m['date'].min()} to {m['date'].max()})")
    print(f"Competition(s) in the file: {sorted(m['event_name'].unique())}")
    print(f"Teams ({len(set(m['team1']) | set(m['team2']))}): {sorted(set(m['team1']) | set(m['team2']))}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) that were not usable matches: {skipped[:5]}")
    print(f"Wrote {args.out_dir / 'bbl_matches.csv'} and {args.out_dir / 'bbl_ball_by_ball.csv'}")


if __name__ == "__main__":
    main()
