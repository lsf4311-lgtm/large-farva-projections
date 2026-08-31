"""
Turns raw Ottoneu stat lines (season / vsR / vsL / last15) into the
blended "expected points today" signal reasoning.py consumes, plus
season-level advanced rate stats (AVG/OBP/SLG/OPS/wOBA) for display
context alongside the Ottoneu-scoring numbers.

Replaces the old Statcast-reconstruction approach in splits.py now that
Ottoneu's own setlineups page gives us real, league-native stat lines for
each split -- see ottoneu_roster.py's docstring for the verification.
"""

from __future__ import annotations
from dataclasses import dataclass

from ottoneu_roster import RosterPlayer, BatterStatLine, PitcherStatLine
from scoring import BatterLine, PitcherLine

# Same wOBA linear weights as league_analysis_final.WOBA_WEIGHTS, kept as
# a local copy rather than an import so this module has no dependency on
# that file's Savant-calling code paths. Worth factoring into one shared
# constants module if the two codebases merge further -- flagging so this
# doesn't silently drift out of sync if either copy gets tuned later.
WOBA_WEIGHTS = {"bb": 0.69, "hbp": 0.72, "1b": 0.89, "2b": 1.27, "3b": 1.62, "hr": 2.15}


@dataclass
class AdvancedBattingStats:
    avg: float
    obp: float
    slg: float
    ops: float
    woba: float


def advanced_stats(stat: BatterStatLine) -> AdvancedBattingStats | None:
    """AVG/OBP/SLG/OPS/wOBA from the same raw counts Ottoneu already gives
    us. Note: Ottoneu's columns don't break out sac flies separately, so
    OBP/PA here use AB+BB+HBP as the denominator (same simplification as
    scoring.BatterLine.plate_appearances) -- close enough for a matchup
    signal, not meant to reconcile exactly to a stats site's OBP."""
    if stat.at_bats == 0:
        return None

    singles = max(stat.hits - stat.doubles - stat.triples - stat.home_runs, 0)
    pa = stat.at_bats + stat.walks + stat.hbp
    if pa == 0:
        return None

    avg = stat.hits / stat.at_bats
    obp = (stat.hits + stat.walks + stat.hbp) / pa
    total_bases = singles + 2 * stat.doubles + 3 * stat.triples + 4 * stat.home_runs
    slg = total_bases / stat.at_bats
    ops = obp + slg

    woba_num = (
        WOBA_WEIGHTS["bb"] * stat.walks + WOBA_WEIGHTS["hbp"] * stat.hbp
        + WOBA_WEIGHTS["1b"] * singles + WOBA_WEIGHTS["2b"] * stat.doubles
        + WOBA_WEIGHTS["3b"] * stat.triples + WOBA_WEIGHTS["hr"] * stat.home_runs
    )
    woba = woba_num / pa

    return AdvancedBattingStats(
        avg=round(avg, 3), obp=round(obp, 3), slg=round(slg, 3),
        ops=round(obp + slg, 3), woba=round(woba, 3),
    )


def batter_line(stat: BatterStatLine) -> BatterLine:
    singles = stat.hits - stat.doubles - stat.triples - stat.home_runs
    return BatterLine(
        at_bats=stat.at_bats, singles=max(singles, 0), doubles=stat.doubles,
        triples=stat.triples, home_runs=stat.home_runs, walks=stat.walks,
        hbp=stat.hbp, stolen_bases=stat.stolen_bases, caught_stealing=stat.caught_stealing,
    )


def pitcher_line(stat: PitcherStatLine) -> PitcherLine:
    return PitcherLine(
        outs=stat.outs, strikeouts=stat.strikeouts, hits_allowed=stat.hits_allowed,
        walks_allowed=stat.walks_allowed, hit_batsmen=stat.hit_batsmen,
        home_runs_allowed=stat.home_runs_allowed, saves=stat.saves, holds=stat.holds,
    )


@dataclass
class BatterMatchupSignal:
    name: str
    season_pts_per_pa: float
    platoon_pts_per_pa: float
    platoon_pa: int
    recent_pts_per_pa: float
    recent_pa: int
    expected_pts_per_pa: float  # the blended number reasoning.py should use
    season_advanced: AdvancedBattingStats | None = None  # AVG/OBP/SLG/OPS/wOBA, season


def _index_by_id(players: list[RosterPlayer]) -> dict[int, RosterPlayer]:
    return {p.ottoneu_id: p for p in players}


def build_batter_signals(
    season_players: list[RosterPlayer],
    vsr_players: list[RosterPlayer],
    vsl_players: list[RosterPlayer],
    last15_players: list[RosterPlayer],
) -> dict[int, BatterMatchupSignal]:
    """One fetch per split, done once per run for the whole roster --
    call this rather than re-fetching per player."""
    by_id_season = _index_by_id(season_players)
    by_id_vsr = _index_by_id(vsr_players)
    by_id_vsl = _index_by_id(vsl_players)
    by_id_recent = _index_by_id(last15_players)

    signals: dict[int, BatterMatchupSignal] = {}

    for ottoneu_id, player in by_id_season.items():
        if not player.batting:
            continue
        season_line = batter_line(player.batting)
        season_pts_per_pa = season_line.points_per_pa()

        recent_player = by_id_recent.get(ottoneu_id)
        recent_line = batter_line(recent_player.batting) if recent_player and recent_player.batting else BatterLine()
        recent_pts_per_pa = recent_line.points_per_pa()
        recent_pa = recent_line.plate_appearances()

        # Opponent's probable-pitcher hand comes off the season view's
        # matchup fields (identical across filters -- it's today's game,
        # not stat-dependent).
        hand = player.opposing_pitcher_hand
        if hand == "L":
            platoon_player = by_id_vsl.get(ottoneu_id)
        elif hand == "R":
            platoon_player = by_id_vsr.get(ottoneu_id)
        else:
            platoon_player = None

        if platoon_player and platoon_player.batting:
            platoon_line = batter_line(platoon_player.batting)
            platoon_pts_per_pa = platoon_line.points_per_pa()
            platoon_pa = platoon_line.plate_appearances()
        else:
            platoon_pts_per_pa = season_pts_per_pa
            platoon_pa = 0

        # Small-sample platoon splits are noisy -- blend toward season
        # rate, weight rising with PA, capped at 60+ PA = "trust it".
        weight = min(platoon_pa / 60.0, 1.0)
        platoon_blend = weight * platoon_pts_per_pa + (1 - weight) * season_pts_per_pa
        expected = 0.75 * platoon_blend + 0.25 * recent_pts_per_pa

        signals[ottoneu_id] = BatterMatchupSignal(
            name=player.name,
            season_pts_per_pa=season_pts_per_pa,
            platoon_pts_per_pa=platoon_pts_per_pa,
            platoon_pa=platoon_pa,
            recent_pts_per_pa=recent_pts_per_pa,
            recent_pa=recent_pa,
            expected_pts_per_pa=expected,
            season_advanced=advanced_stats(player.batting),
        )

    return signals


@dataclass
class PitcherSignal:
    name: str
    season_pts_per_out: float
    recent_pts_per_out: float
    recent_outs: int


def build_pitcher_signals(
    season_players: list[RosterPlayer],
    last15_players: list[RosterPlayer],
) -> dict[int, PitcherSignal]:
    by_id_season = _index_by_id(season_players)
    by_id_recent = _index_by_id(last15_players)

    signals: dict[int, PitcherSignal] = {}
    for ottoneu_id, player in by_id_season.items():
        if not player.pitching:
            continue
        season_line = pitcher_line(player.pitching)

        recent_player = by_id_recent.get(ottoneu_id)
        recent_line = pitcher_line(recent_player.pitching) if recent_player and recent_player.pitching else PitcherLine()

        signals[ottoneu_id] = PitcherSignal(
            name=player.name,
            season_pts_per_out=season_line.points_per_out(),
            recent_pts_per_out=recent_line.points_per_out(),
            recent_outs=recent_line.outs,
        )

    return signals
