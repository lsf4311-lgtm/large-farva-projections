"""
Turns raw counting stats into Ottoneu FanGraphs Points (FGP), using the
confirmed formula for league 569 (see config.py for source + notes).

Kept deliberately dumb and centralized: every other module that needs a
point value (batter splits, opposing-lineup-vs-pitcher proxy, recent form)
routes through here, so there's exactly one place that encodes the scoring
rules and exactly one place to fix if config.SCORING is ever wrong.
"""

from __future__ import annotations
from dataclasses import dataclass

from config import SCORING


@dataclass
class BatterLine:
    at_bats: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    hbp: int = 0
    stolen_bases: int = 0
    caught_stealing: int = 0

    @property
    def hits(self) -> int:
        return self.singles + self.doubles + self.triples + self.home_runs

    def points(self) -> float:
        s = SCORING.batting
        return (
            self.at_bats * s.at_bat
            + self.hits * s.hit
            + self.doubles * s.double_bonus
            + self.triples * s.triple_bonus
            + self.home_runs * s.home_run_bonus
            + self.walks * s.walk
            + self.hbp * s.hit_by_pitch
            + self.stolen_bases * s.stolen_base
            + self.caught_stealing * s.caught_stealing
        )

    def plate_appearances(self) -> int:
        # Close enough for rate-stat purposes: AB + BB + HBP. Undercounts
        # sac flies/bunts slightly since those aren't tracked as a
        # separate field here -- fine for "points per PA" as a matchup
        # signal, not meant to reconcile exactly to official PA.
        return self.at_bats + self.walks + self.hbp

    def points_per_pa(self) -> float:
        pa = self.plate_appearances()
        return self.points() / pa if pa else 0.0


@dataclass
class PitcherLine:
    outs: int = 0
    strikeouts: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    hit_batsmen: int = 0
    home_runs_allowed: int = 0
    saves: int = 0
    holds: int = 0

    def points(self) -> float:
        s = SCORING.pitching
        innings = self.outs / 3.0
        return (
            innings * s.inning_pitched
            + self.strikeouts * s.strikeout
            + self.hits_allowed * s.hit_allowed
            + self.walks_allowed * s.walk_allowed
            + self.hit_batsmen * s.hit_batsman
            + self.home_runs_allowed * s.home_run_allowed
            + self.saves * s.save
            + self.holds * s.hold
        )

    def points_per_out(self) -> float:
        return self.points() / self.outs if self.outs else 0.0


# Statcast "events" values that do NOT count as an at-bat (matters because
# our formula's -1.0 AB penalty only applies to actual at-bats).
NON_AB_EVENTS = {
    "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "catcher_interf", "intent_walk",
}

# Statcast "events" -> BatterLine field, for the subset that are hits.
HIT_EVENT_MAP = {
    "single": "singles",
    "double": "doubles",
    "triple": "triples",
    "home_run": "home_runs",
}


def batter_line_from_events(event_counts: dict[str, int],
                             sb: int = 0, cs: int = 0) -> BatterLine:
    """event_counts: {statcast 'events' value: count}, e.g. from a
    value_counts() on the statcast 'events' column for one batter (or one
    team, for the opposing-lineup proxy). Computes AB as total PA-ending
    events minus the ones that don't count as an at-bat."""
    line = BatterLine(stolen_bases=sb, caught_stealing=cs)

    total_events = sum(event_counts.values())
    non_ab = sum(count for event, count in event_counts.items() if event in NON_AB_EVENTS)
    line.at_bats = total_events - non_ab
    line.walks = event_counts.get("walk", 0) + event_counts.get("intent_walk", 0)
    line.hbp = event_counts.get("hit_by_pitch", 0)

    for event, field_name in HIT_EVENT_MAP.items():
        setattr(line, field_name, event_counts.get(event, 0))

    return line
