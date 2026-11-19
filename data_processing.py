#!/usr/bin/env python3
"""Step 2: clean the BBL tables, compute the metrics, and write everything to output/.

Usage (after build_dataset.py has created the two CSVs in data/):
    python data_processing.py
or, with your own paths:
    python data_processing.py path/to/bbl_ball_by_ball.csv path/to/bbl_matches.csv --out output

Writes into the output folder:
    cleaned_ball_by_ball.csv   cleaned deliveries plus three derived columns
    cleaned_matches.csv        cleaned matches
    data_quality_report.md     what was found and changed, per column
    cleaning_summary.json      the same counts, machine-readable
    outlier_report.csv         innings totals flagged by the IQR rule (kept, never deleted)
    kpi_summary.csv            one row per team, all data, no filters
    findings.md                headline findings, written from the data (no typed-in numbers)
    charts/*.png               four static charts

app.py and the notebook import the metric functions in this file, so the numbers
on every surface come from the same code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ------------------------------------------------------------------ settings
LEAGUE_NAME = "Big Bash League"
LEAGUE_SHORT = "BBL"

ROOT = Path(__file__).resolve().parent
DEFAULT_BALL_CSV = ROOT / "data" / "bbl_ball_by_ball.csv"
DEFAULT_MATCH_CSV = ROOT / "data" / "bbl_matches.csv"
DEFAULT_OUT = ROOT / "output"

# If a franchise ever changes its name, add  "Old name": "Current name"  here and
# re-run this script. The cleaning step then merges the two into one team.
TEAM_RENAMES: dict[str, str] = {}

NULL_MARKERS = {"", "na", "n/a", "none", "null", "nan", "-"}
NON_DISMISSAL_KINDS = {"retired hurt", "retired out", "obstructing the field"}
PHASE_LABELS = ["Powerplay", "Middle", "Death"]

# Reliability thresholds: below these, a number is shown with a warning.
MIN_DECIDED_RELIABLE = 20   # team-level decided matches
MIN_DECIDED_SEASON = 5      # season-level decided matches
MIN_BALLS_PHASE = 120       # heatmap cells with fewer legal balls are left blank
MIN_DISMISSALS = 30         # dismissal mix needs at least this many

# Chart colours, shared by the dashboard.
PRIMARY, HIGHLIGHT, MUTED = "#d9822b", "#2a9d8f", "#9e9e9e"

MATCH_REQUIRED = ["match_id", "date", "team1", "team2", "toss_winner", "toss_decision", "winner", "season"]
MATCH_NUMERIC = ["match_id", "overs", "balls_per_over", "team1_won", "toss_winner_won"]
MATCH_DATES = ["date"]
MATCH_RULES = {"overs": (1, 50), "balls_per_over": (1, 12), "team1_won": (0, 1), "toss_winner_won": (0, 1)}
MATCH_TEAM_COLS = ["team1", "team2", "toss_winner", "winner"]

BALL_REQUIRED = ["match_id", "season", "innings", "batting_team", "over", "batter_runs", "total_runs",
                 "wides", "noballs", "is_wicket", "wicket_kind", "is_powerplay", "is_middle_overs",
                 "is_death_overs"]
BALL_NUMERIC = ["match_id", "innings", "over", "ball_in_over", "ball_number", "batter_runs", "extra_runs",
                "total_runs", "wides", "noballs", "byes", "legbyes", "penalty", "is_wicket",
                "batting_team_won", "is_powerplay", "is_middle_overs", "is_death_overs"]
BALL_DATES = ["date"]
# Plausibility bounds, deliberately generous. They catch impossible values (negative runs,
# 40 off one ball), not rare real ones: a six off a no-ball is 7, and 5 penalty runs can be
# added on top (12). A value outside its range is set to missing and listed in the quality report.
BALL_RULES = {"innings": (1, 6), "over": (0, 19), "ball_in_over": (1, 12), "ball_number": (1, 200),
              "batter_runs": (0, 8), "extra_runs": (0, 12), "total_runs": (0, 15), "wides": (0, 12),
              "noballs": (0, 12), "byes": (0, 8), "legbyes": (0, 8), "penalty": (0, 5),
              "is_wicket": (0, 1), "batting_team_won": (0, 1), "is_powerplay": (0, 1),
              "is_middle_overs": (0, 1), "is_death_overs": (0, 1)}
BALL_TEAM_COLS = ["batting_team", "bowling_team", "team1", "team2", "toss_winner", "match_winner"]


# ------------------------------------------------------------------ cleaning
def _to_number(raw: pd.Series, col: str, failures: dict) -> pd.Series:
    """Text to number. Whole-number columns become Int64 so missing values stay missing."""
    num = pd.to_numeric(raw, errors="coerce")
    failed = raw.notna() & num.isna()
    failures[col] = {"count": int(failed.sum()), "examples": raw[failed].unique()[:5].tolist()}
    if num.dropna().mod(1).eq(0).all():
        num = num.astype("Int64")
    return num


def _to_date(raw: pd.Series, col: str, failures: dict) -> pd.Series:
    parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    failed = raw.notna() & parsed.isna()
    failures[col] = {"count": int(failed.sum()), "examples": raw[failed].unique()[:5].tolist()}
    return parsed


def clean_table(raw: pd.DataFrame, name: str, numeric, dates, rules, report: dict) -> pd.DataFrame:
    """Dedupe, trim text, turn null markers into missing, parse types, apply business ranges.
    Invalid values become missing; nothing is guessed or imputed."""
    info = {"raw_rows": int(len(raw))}
    df = raw.drop_duplicates().copy()
    info["duplicates_removed"] = int(len(raw) - len(df))

    nulls, trimmed = {}, 0
    for col in df.columns:
        stripped = df[col].str.strip()
        trimmed += int((stripped != df[col]).sum())
        is_null = stripped.str.lower().isin(NULL_MARKERS)
        nulls[col] = int(is_null.sum())
        df[col] = stripped.mask(is_null)
    info["whitespace_trimmed"] = trimmed
    info["null_markers_converted"] = nulls

    failures: dict = {}
    for col in numeric:
        if col in df:
            df[col] = _to_number(df[col], col, failures)
    for col in dates:
        if col in df:
            df[col] = _to_date(df[col], col, failures)
    info["conversion_failures"] = failures

    invalid = {}
    for col, (lo, hi) in rules.items():
        if col not in df:
            continue
        s = df[col]
        in_range = ((s >= lo) & (s <= hi)).fillna(False).astype(bool)
        bad = s.notna() & ~in_range
        invalid[col] = {"count": int(bad.sum()), "allowed": [lo, hi]}
        df.loc[bad, col] = pd.NA
    info["invalid_values_set_missing"] = invalid
    info["clean_rows"] = int(len(df))
    info["missing_after_cleaning"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    report[name] = info
    return df


def rename_teams(df: pd.DataFrame, cols, name: str, report: dict) -> pd.DataFrame:
    changes = {}
    for col in cols:
        if col in df and TEAM_RENAMES:
            new = df[col].replace(TEAM_RENAMES)
            changes[col] = int((df[col].notna() & (df[col] != new)).sum())
            df[col] = new
    report.setdefault("team_name_changes", {})[name] = changes
    return df


def add_ball_fields(b: pd.DataFrame) -> pd.DataFrame:
    """Three derived columns: legal_ball, counts_as_dismissal, phase."""
    def flag(s):  # 1 -> True; 0 or missing -> False
        return s.eq(1).fillna(False).astype(bool).to_numpy()

    b["legal_ball"] = ((b["wides"].fillna(0) == 0) & (b["noballs"].fillna(0) == 0)).astype(int)
    b["counts_as_dismissal"] = (flag(b["is_wicket"]) & ~b["wicket_kind"].isin(NON_DISMISSAL_KINDS)).astype(int)
    phase = np.select([flag(b["is_powerplay"]), flag(b["is_middle_overs"]), flag(b["is_death_overs"])],
                      PHASE_LABELS, default="")
    b["phase"] = pd.Series(phase, index=b.index).replace("", np.nan)
    return b


# ------------------------------------------------------------------ metrics (shared)
def _pct(n, d):
    """100 * n / d, left missing where d is zero."""
    return (100 * n / d).where(d > 0).round(2)


def season_order(matches: pd.DataFrame) -> list:
    """Seasons in the order they were played (by first match date), not alphabetical."""
    return matches.groupby("season")["date"].min().sort_values().index.tolist()


def scope_filters(matches, balls, *, start=None, end=None, status="All matches",
                  season="All seasons", toss="All", team=None, opponent=None):
    """Apply the dashboard filters to both tables. With no arguments, returns everything."""
    m = matches
    if start is not None:
        m = m[m["date"] >= start]
    if end is not None:
        m = m[m["date"] <= end]
    if status == "Decided matches only":
        m = m[m["winner"].notna()]
    elif status == "No result only":
        m = m[m["winner"].isna()]
    if season and season != "All seasons":
        m = m[m["season"] == season]
    if toss and toss != "All":
        m = m[m["toss_decision"] == toss]
    if team:
        m = m[(m["team1"] == team) | (m["team2"] == team)]
    if team and opponent and opponent != "All teams":
        m = m[((m["team1"] == team) & (m["team2"] == opponent)) |
              ((m["team2"] == team) & (m["team1"] == opponent))]
    return m, balls[balls["match_id"].isin(m["match_id"])]


def phase_table(balls: pd.DataFrame) -> pd.DataFrame:
    """Runs and legal balls for each batting team, season and phase."""
    d = balls[balls["phase"].notna()]
    key = ["batting_team", "season", "phase"]
    runs = d.groupby(key)["total_runs"].sum()
    legal = d[d["legal_ball"] == 1].groupby(key).size()
    out = pd.concat([runs.rename("runs"), legal.rename("legal_balls")], axis=1).fillna(0).reset_index()
    out["runs_per_100_balls"] = _pct(out["runs"], out["legal_balls"])
    return out


def season_win_table(matches: pd.DataFrame, team: str, order: list) -> pd.DataFrame:
    """One row per season for one team: matches, decided matches, wins, win rate."""
    t = matches[(matches["team1"] == team) | (matches["team2"] == team)].copy()
    t["won"] = (t["winner"] == team).astype(int)
    t["decided"] = t["winner"].notna().astype(int)
    g = t.groupby("season").agg(matches=("match_id", "size"), decided=("decided", "sum"), wins=("won", "sum"))
    g["win_rate_pct"] = _pct(g["wins"], g["decided"])
    g["enough_matches"] = np.where(g["decided"] >= MIN_DECIDED_SEASON, "Enough matches", "Too few matches")
    return g.reindex([s for s in order if s in g.index]).reset_index()


def team_kpis(matches: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """One row per team. The dashboard, notebook and KPI file all use this function."""
    decided = matches["winner"].notna().to_numpy()
    sides = []
    for col in ("team1", "team2"):
        sides.append(pd.DataFrame({"team": matches[col].to_numpy(), "decided": decided,
                                   "won": (matches["winner"] == matches[col]).to_numpy(),
                                   "won_toss": (matches["toss_winner"] == matches[col]).to_numpy()}))
    long = pd.concat(sides, ignore_index=True)
    g = long.groupby("team")
    out = pd.DataFrame({"matches_played": g.size(),
                        "decided_matches": g["decided"].sum().astype(int),
                        "wins": g["won"].sum().astype(int)})
    out["win_rate_pct"] = _pct(out["wins"], out["decided_matches"])

    dec = long[long["decided"]]
    for label, subset in (("win", dec[dec["won_toss"]]), ("loss", dec[~dec["won_toss"]])):
        grp = subset.groupby("team")["won"]
        n = grp.size().reindex(out.index).fillna(0).astype(int)
        out[f"decided_after_toss_{label}"] = n
        out[f"win_rate_after_toss_{label}_pct"] = _pct(grp.sum().reindex(out.index).fillna(0), n)

    b = balls.assign(boundary=balls["batter_runs"].isin([4, 6]).astype(int))
    bg = b.groupby("batting_team").agg(runs_scored=("total_runs", "sum"),
                                       dismissals=("counts_as_dismissal", "sum"),
                                       boundaries=("boundary", "sum"),
                                       legal_balls_faced=("legal_ball", "sum"))
    out = out.join(bg.reindex(out.index).fillna(0))
    out["runs_per_100_balls"] = _pct(out["runs_scored"], out["legal_balls_faced"])
    out["boundaries_per_100_balls"] = _pct(out["boundaries"], out["legal_balls_faced"])
    out["wickets_lost_per_100_balls"] = _pct(out["dismissals"], out["legal_balls_faced"])

    pt = phase_table(balls)
    death = pt[pt["phase"] == "Death"].groupby("batting_team")[["runs", "legal_balls"]].sum()
    death = death.reindex(out.index).fillna(0)
    out["death_runs_per_100_balls"] = _pct(death["runs"], death["legal_balls"])

    cols = ["matches_played", "decided_matches", "wins", "win_rate_pct",
            "decided_after_toss_win", "win_rate_after_toss_win_pct",
            "decided_after_toss_loss", "win_rate_after_toss_loss_pct",
            "runs_scored", "legal_balls_faced", "runs_per_100_balls",
            "boundaries_per_100_balls", "wickets_lost_per_100_balls", "death_runs_per_100_balls"]
    out = out[cols].round(2)
    out.index.name = "team"
    return out


def innings_outliers(balls: pd.DataFrame):
    """IQR rule on first- and second-innings totals. Flags only; nothing is deleted."""
    main = balls[balls["innings"].isin([1, 2])]
    t = (main.groupby(["match_id", "innings", "batting_team"])["total_runs"].sum()
         .reset_index(name="value"))
    q1, q3 = t["value"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    t["lower_bound"], t["upper_bound"] = lo, hi
    flagged = t[(t["value"] < lo) | (t["value"] > hi)].copy()
    flagged.insert(0, "metric", "innings_total_runs")
    return flagged, {"metric": "innings_total_runs", "q1": float(q1), "q3": float(q3),
                     "lower_bound": float(lo), "upper_bound": float(hi),
                     "flagged_rows": int(len(flagged)), "innings_checked": int(len(t))}


# ------------------------------------------------------------------ findings (written from the data)
def build_findings(kpis: pd.DataFrame, matches: pd.DataFrame, balls: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (heading, sentence) pairs. Every number is computed here, none is typed in."""
    out: list[tuple[str, str]] = []
    order = season_order(matches)

    # 1. Win rates (teams with a small record are left out of the ranking)
    ok = kpis[kpis["decided_matches"] >= MIN_DECIDED_RELIABLE].sort_values("win_rate_pct", ascending=False)
    if not ok.empty:
        top, low = ok.iloc[0], ok.iloc[-1]
        text = (f"{ok.index[0]} have the highest win rate, {top.win_rate_pct:.1f}% of {int(top.decided_matches)} "
                f"decided matches. {ok.index[-1]} have the lowest, {low.win_rate_pct:.1f}% of "
                f"{int(low.decided_matches)}.")
        small = kpis[kpis["decided_matches"] < MIN_DECIDED_RELIABLE].index.tolist()
        if small:
            text += (f" Teams with fewer than {MIN_DECIDED_RELIABLE} decided matches are left out of this "
                     f"ranking: {', '.join(small)}.")
        out.append(("Win rates", text))

    # 2. Toss
    dec = matches[matches["winner"].notna()]
    if not dec.empty:
        won_toss = dec["toss_winner"] == dec["winner"]
        text = f"Toss winners won {100 * won_toss.mean():.1f}% of {len(dec):,} decided matches."
        parts = []
        for choice, label in (("field", "chose to field"), ("bat", "chose to bat")):
            sub = dec[(dec["toss_decision"] == choice) & dec["toss_winner"].notna()]
            if len(sub):
                parts.append(f"teams that won the toss and {label} won "
                             f"{100 * (sub['toss_winner'] == sub['winner']).mean():.1f}% of {len(sub):,} matches")
        if parts:
            text += " " + "; ".join(parts).capitalize() + "."
        out.append(("Toss", text))

    # 3. Scoring by phase and over time
    pt = phase_table(balls)
    if not pt.empty:
        ph = pt.groupby("phase")[["runs", "legal_balls"]].sum()
        ph["rate"] = 100 * ph["runs"] / ph["legal_balls"]
        ph = ph.reindex([p for p in PHASE_LABELS if p in ph.index])
        fastest = ph["rate"].idxmax()
        text = (f"{fastest} overs are the fastest phase. Runs per 100 legal balls: "
                + ", ".join(f"{p.lower()} {r:.0f}" for p, r in ph["rate"].items()) + ".")
        death = (pt[pt["phase"] == "Death"].groupby("season")[["runs", "legal_balls"]].sum()
                 .reindex([s for s in order if s in set(pt["season"])]))
        death = death[death["legal_balls"] > 0]
        if len(death) >= 2:
            rate = 100 * death["runs"] / death["legal_balls"]
            text += (f" The death-overs rate was {rate.iloc[0]:.0f} in {rate.index[0]} and {rate.iloc[-1]:.0f} "
                     f"in {rate.index[-1]}, with a peak of {rate.max():.0f} in {rate.idxmax()}.")
        out.append(("Scoring", text))

    # 4. Dismissals
    d = balls[balls["counts_as_dismissal"] == 1]["wicket_kind"].value_counts(normalize=True) * 100
    if not d.empty:
        top3 = ", ".join(f"{k} {v:.1f}%" for k, v in d.head(3).items())
        out.append(("Dismissals", f"The most common ways to get out (retirements excluded): {top3}."))

    # 5. Caveats
    per_season = matches.groupby("season").size()
    caveats = [f"{int(matches['winner'].isna().sum())} matches with no winner (no result, or tied without a "
               "winner recorded) are left out of win rates.",
               "Win rates are not adjusted for the strength of the opposition."]
    if len(order) >= 3 and per_season[order[-1]] < 0.8 * per_season.median():
        caveats.append(f"The latest season in the data ({order[-1]}) has only {per_season[order[-1]]} matches, "
                       "fewer than a typical season, so it may still be in progress.")
    out.append(("Caveats", " ".join(caveats)))
    return out


def findings_markdown(findings) -> str:
    lines = ["# Main findings", "", f"Written by `data_processing.py` from the {LEAGUE_NAME} data. "
             "Re-run the script and this file updates.", ""]
    lines += [f"- **{h}.** {t}" for h, t in findings]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ charts
def make_charts(kpis: pd.DataFrame, matches: pd.DataFrame, balls: pd.DataFrame, out: Path):
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    charts = out / "charts"

    k = kpis.sort_values("win_rate_pct")
    colors = [PRIMARY if n >= MIN_DECIDED_RELIABLE else MUTED for n in k["decided_matches"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(k) + 1.5)))
    ax.barh(k.index, k["win_rate_pct"], color=colors)
    ax.set_xlabel("Win rate (% of decided matches)")
    ax.set_title(f"Which team wins the largest share of its decided matches?\n"
                 f"Grey = fewer than {MIN_DECIDED_RELIABLE} decided matches")
    fig.tight_layout(); fig.savefig(charts / "win_rate_by_team.png"); plt.close(fig)

    k = kpis.sort_values("runs_per_100_balls")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(k) + 1.5)))
    ax.barh(k.index, k["runs_per_100_balls"], color=PRIMARY)
    ax.set_xlabel("Runs per 100 legal balls faced")
    ax.set_title("Which team scores fastest?")
    fig.tight_layout(); fig.savefig(charts / "runs_per_100_balls_by_team.png"); plt.close(fig)

    order = season_order(matches)
    counts = matches.groupby("season").size().reindex(order)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts.values, color=PRIMARY)
    ax.set_ylabel("Matches")
    ax.set_title("How many matches were played each season?")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(charts / "matches_per_season.png"); plt.close(fig)

    d = balls[balls["counts_as_dismissal"] == 1]["wicket_kind"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d.index[::-1], d.values[::-1], color=PRIMARY)
    ax.set_xlabel("Share of dismissals (%)")
    ax.set_title("How are batters dismissed? (retirements excluded)")
    fig.tight_layout(); fig.savefig(charts / "dismissal_types.png"); plt.close(fig)


# ------------------------------------------------------------------ pipeline
def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Cannot find {path}. Run build_dataset.py first, or pass the file paths.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}. Are the two files the right way round "
                         "(ball-by-ball first, matches second)?")
    return df


def run(ball_path: Path, matches_path: Path, out: Path):
    (out / "charts").mkdir(parents=True, exist_ok=True)
    raw_m = _read(matches_path, MATCH_REQUIRED)
    raw_b = _read(ball_path, BALL_REQUIRED)

    report: dict = {}
    tables: dict = {}
    m = clean_table(raw_m, "matches", MATCH_NUMERIC, MATCH_DATES, MATCH_RULES, tables)
    b = clean_table(raw_b, "ball_by_ball", BALL_NUMERIC, BALL_DATES, BALL_RULES, tables)
    report["tables"] = tables
    m = rename_teams(m, MATCH_TEAM_COLS, "matches", report)
    b = rename_teams(b, BALL_TEAM_COLS, "ball_by_ball", report)
    b = add_ball_fields(b)

    m_ids, b_ids = set(m["match_id"].dropna()), set(b["match_id"].dropna())
    report["unmatched_ids"] = {"in_balls_not_matches": len(b_ids - m_ids),
                               "in_matches_not_balls": len(m_ids - b_ids)}
    report["super_over_deliveries"] = int((b["innings"] >= 3).sum())
    report["teams"] = sorted(set(m["team1"].dropna()) | set(m["team2"].dropna()))

    m.to_csv(out / "cleaned_matches.csv", index=False, date_format="%Y-%m-%d")
    b.to_csv(out / "cleaned_ball_by_ball.csv", index=False, date_format="%Y-%m-%d")

    kpis = team_kpis(m, b)
    kpis.reset_index().to_csv(out / "kpi_summary.csv", index=False)

    flagged, outlier_info = innings_outliers(b)
    flagged.to_csv(out / "outlier_report.csv", index=False)
    report["outliers"] = outlier_info

    make_charts(kpis, m, b, out)
    (out / "findings.md").write_text(findings_markdown(build_findings(kpis, m, b)), encoding="utf-8")
    with open(out / "cleaning_summary.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    write_quality_report(report, m, b, out / "data_quality_report.md")
    # Clearly label synthetic sample outputs. Real source files remove the marker.
    demo_inputs = "demo_data" in str(ball_path) or "demo_data" in str(matches_path)
    demo_marker = out / "DEMO_DATA_NOTICE.md"
    if demo_inputs:
        demo_marker.write_text(
            "# Demo output only\\n\\nThese outputs use synthetic demo data, not real BBL records. "
            "Replace them by processing Cricsheet data.\\n", encoding="utf-8"
        )
    elif demo_marker.exists():
        demo_marker.unlink()

    print(f"Matches: {len(m):,} | Deliveries: {len(b):,} | Teams: {kpis.shape[0]}")
    print(f"Teams found: {', '.join(report['teams'])}")
    print(f"Outputs written to {out.resolve()}")


def write_quality_report(report: dict, m: pd.DataFrame, b: pd.DataFrame, path: Path):
    L = ["# Data quality report", "",
         "Generated by `data_processing.py`. Counts describe what was found and changed.", ""]
    for name, info in report["tables"].items():
        L += [f"## {name}", "",
              f"- Raw rows: {info['raw_rows']:,}",
              f"- Exact duplicate rows removed: {info['duplicates_removed']:,}",
              f"- Cells with stray whitespace trimmed: {info['whitespace_trimmed']:,}",
              f"- Clean rows: {info['clean_rows']:,}", "",
              "### Null markers converted to missing (NA, none, -, blank)", ""]
        nz = {k: v for k, v in info["null_markers_converted"].items() if v}
        L += ([f"- `{k}`: {v:,}" for k, v in nz.items()] or ["- None found."]) + [""]
        L += ["### Values that failed to convert", ""]
        bad = {k: v for k, v in info["conversion_failures"].items() if v["count"]}
        L += ([f"- `{k}`: {v['count']:,} failed, e.g. {v['examples']}" for k, v in bad.items()]
              or ["- None. Every number and date parsed."]) + [""]
        L += ["### Business-rule violations set to missing (no imputation)", ""]
        badr = {k: v for k, v in info["invalid_values_set_missing"].items() if v["count"]}
        L += ([f"- `{k}` outside {v['allowed']}: {v['count']:,}" for k, v in badr.items()]
              or ["- None. All values are within their valid ranges."]) + [""]
        L += ["### Columns still missing after cleaning", ""]
        miss = info["missing_after_cleaning"]
        L += ([f"- `{k}`: {v:,}" for k, v in miss.items()] or ["- None."]) + [""]

    L += ["## Team names", "", f"Teams in the data ({len(report['teams'])}): " + ", ".join(report["teams"]), ""]
    if TEAM_RENAMES:
        L += ["Renamed franchises were merged to their current name:", ""]
        L += [f"- `{old}` to `{new}`" for old, new in TEAM_RENAMES.items()]
        L += ["", "Cells changed, per table: " + json.dumps(report["team_name_changes"]), ""]
    else:
        L += ["No team renames are configured, so no names were merged. To merge a renamed franchise, "
              "add it to `TEAM_RENAMES` in `data_processing.py`.", ""]

    L += ["## Derived columns added", "",
          "- `legal_ball`: 1 when the delivery has no wide or no-ball, otherwise 0. Balls faced exclude extras.",
          "- `counts_as_dismissal`: 1 for a wicket except retirements and obstructing the field.",
          "- `phase`: Powerplay (overs 1 to 6), Middle (overs 7 to 15), Death (overs 16 to 20), from the source flags.", ""]

    L += ["## Linking and coverage", "",
          f"- Match IDs in ball-by-ball but not in matches: {report['unmatched_ids']['in_balls_not_matches']:,}",
          f"- Match IDs in matches but not in ball-by-ball: {report['unmatched_ids']['in_matches_not_balls']:,}",
          f"- Date range: {m['date'].min().date()} to {m['date'].max().date()}",
          f"- Matches with no winner (no result, or tied without a winner recorded): {int(m['winner'].isna().sum()):,}",
          f"- Super-over deliveries (innings 3 and above): {report['super_over_deliveries']:,}. "
          "Kept in the totals; they are too few to change any rate meaningfully.", ""]

    o = report["outliers"]
    L += ["## Outliers", "",
          f"IQR rule on first and second innings totals. Bounds: {o['lower_bound']:.1f} to {o['upper_bound']:.1f} runs.",
          f"{o['flagged_rows']:,} of {o['innings_checked']:,} innings were flagged for review in `outlier_report.csv`. "
          "They are kept in every metric; flagging is for review only.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=f"Clean the {LEAGUE_SHORT} datasets and build the outputs.")
    ap.add_argument("ball_csv", nargs="?", type=Path, default=DEFAULT_BALL_CSV,
                    help="ball-by-ball CSV (default: data/bbl_ball_by_ball.csv)")
    ap.add_argument("matches_csv", nargs="?", type=Path, default=DEFAULT_MATCH_CSV,
                    help="matches CSV (default: data/bbl_matches.csv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: output)")
    args = ap.parse_args()
    run(args.ball_csv, args.matches_csv, args.out)


if __name__ == "__main__":
    main()
