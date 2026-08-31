"""
Adapter around the existing league_analysis_final.py pitching-report
pipeline -- reused as-is rather than reinventing pitcher matchup logic.
That module already computes an A-F matchup grade per pitcher start
(K%/BB%/wOBA-against blended with the opponent's home/away OPS) plus the
opponent's rank among all 30 teams in OPS/wOBA/K%/HR rate. It just wasn't
wired into any downstream decision output before now -- see chat.

get_all_team_stats() is expensive (~2 min, 30 teams x Savant calls per
the module's own docstring) and shouldn't re-run on every daily cron
invocation. No existing cache was available to reuse, so this adds a
simple weekly JSON file cache around it. If a shared cache location
already exists elsewhere in the project by the time this runs, point
TEAM_STATS_CACHE_PATH at that instead of this local one.

Import assumes league_analysis_final.py is importable from this
directory (confirmed: same repo).
"""

from __future__ import annotations
import datetime as dt
import json
import os

from league_analysis_final import get_pitching_report, get_all_team_stats

TEAM_STATS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "team_stats_cache.json")
TEAM_STATS_TTL_DAYS = 7


def _load_cache() -> dict | None:
    if not os.path.exists(TEAM_STATS_CACHE_PATH):
        return None
    try:
        with open(TEAM_STATS_CACHE_PATH) as f:
            payload = json.load(f)
        cached_at = dt.date.fromisoformat(payload["cached_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None  # corrupt/old-format cache file -- treat as a miss, not a crash
    if (dt.date.today() - cached_at).days > TEAM_STATS_TTL_DAYS:
        return None
    return payload["team_stats"]


def _save_cache(team_stats: dict) -> None:
    os.makedirs(os.path.dirname(TEAM_STATS_CACHE_PATH), exist_ok=True)
    with open(TEAM_STATS_CACHE_PATH, "w") as f:
        json.dump({"cached_at": dt.date.today().isoformat(), "team_stats": team_stats}, f)


def get_team_stats_cached(force_refresh: bool = False) -> dict:
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached
    print("Refreshing all-30-team Savant stats (this takes ~2 minutes; cached for 7 days after)...")
    team_stats = get_all_team_stats()
    _save_cache(team_stats)
    return team_stats


def get_rotation_info(pitcher_names: list[str],
                       team_stats: dict | None = None) -> dict[str, dict]:
    """Returns {pitcher_name: rotation_dict} straight from
    get_pitching_report's 'rotation' field -- last_start,
    avg_days_between_starts, next_predicted_start, is_starter,
    recent_starts_count. Empty dict per pitcher if lookup failed or too
    few recent starts to predict from."""
    if not pitcher_names:
        return {}
    if team_stats is None:
        team_stats = get_team_stats_cached()
    report = get_pitching_report(pitcher_names, team_stats)
    return {p["name"]: p.get("rotation", {}) for p in report}


def filter_to_predicted_starters(
    pitcher_names: list[str], target_date: dt.date,
    team_stats: dict | None = None, tolerance_days: int = 1,
) -> tuple[list[str], list[str]]:
    """Narrows a list of SP-slot candidates down to who's actually
    predicted to start on target_date, using real rotation-gap
    prediction rather than "does his team have a game" (every rostered
    SP's team has a game on plenty of days he isn't the one pitching --
    see chat history for why the roster page alone can't answer this).

    Returns (confirmed, uncertain): confirmed = predicted start date
    falls within tolerance_days of target_date. uncertain = rotation
    info unavailable (e.g. recent call-up, too few tracked starts) --
    these aren't excluded outright, just flagged so the person can weigh
    them with lower confidence rather than silently dropping them.
    """
    rotations = get_rotation_info(pitcher_names, team_stats)
    confirmed, uncertain = [], []
    for name in pitcher_names:
        rotation = rotations.get(name) or {}
        predicted = rotation.get("next_predicted_start")
        if predicted is None:
            print(f"  {name}: no rotation prediction available "
                  f"(recent_starts_count={rotation.get('recent_starts_count', 0)}) -> uncertain")
            uncertain.append(name)
            continue
        predicted_date = predicted.date() if hasattr(predicted, "date") else predicted
        gap = (predicted_date - target_date).days
        if abs(gap) <= tolerance_days:
            print(f"  {name}: predicted next start {predicted_date} (target {target_date}, "
                  f"gap {gap:+d}d) -> confirmed")
            confirmed.append(name)
        else:
            print(f"  {name}: predicted next start {predicted_date} (target {target_date}, "
                  f"gap {gap:+d}d, outside {tolerance_days}d tolerance) -> excluded")
    return confirmed, uncertain


def get_matchups_for_date(pitcher_names: list[str], target_date: dt.date,
                           team_stats: dict | None = None) -> dict[str, dict | None]:
    """Returns {pitcher_name: matchup_dict_or_None} for the given date.

    matchup_dict keys (straight from get_pitching_report, unmodified):
    opponent, home_away, date, grade ('A'-'F'), rankings (per-stat rank
    among 30 teams), opp_stats (opponent's OPS/wOBA/K_percent/HR_rate at
    that location).

    Note: get_pitcher_schedule (inside get_pitching_report) looks at a
    rolling 7-day window starting from *today*, not from target_date.
    Fine for config.LOOKAHEAD_DAYS=1 (tomorrow); if that value ever grows
    past ~6 the target date could fall outside the window and every
    pitcher will come back with matchup=None here.
    """
    if not pitcher_names:
        return {}
    if team_stats is None:
        team_stats = get_team_stats_cached()

    report = get_pitching_report(pitcher_names, team_stats)
    target_str = target_date.isoformat()

    result: dict[str, dict | None] = {}
    for pitcher in report:
        result[pitcher["name"]] = next(
            (m for m in pitcher["matchups"] if m["date"] == target_str), None
        )
    return result
