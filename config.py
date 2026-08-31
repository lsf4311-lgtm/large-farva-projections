"""
Central configuration for the sit/start agent.

Fill in the ALL_CAPS values below before running. Anything marked
"VERIFY" is something I could not confirm from outside your league
(Ottoneu's exact scoring page and roster export aren't publicly
browsable) -- you can confirm it in about 30 seconds from inside
your league, see the notes next to each one.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# League identity
# ---------------------------------------------------------------------------

OTTONEU_LEAGUE_ID = "569"    # Kaboom's Pepper Palace
OTTONEU_TEAM_ID = "4067"     # Large Farva

# VERIFY: log into your Ottoneu team page and view source (or Network tab)
# looking for a link/button labeled "Export" or "Roster Export" near the
# roster table. If one exists, it's usually a CSV link you can hit directly
# with a saved cookie -- swap that in as ROSTER_SOURCE = "csv_export" and
# drop the URL into ottoneu_roster.py's CSV_EXPORT_URL. If there's no such
# link in your league, leave this as "scrape" and use the scraper as-is.
ROSTER_SOURCE = "scrape"  # "scrape" | "csv_export"

# Path to a saved browser cookie string for authenticated scraping/export.
# Ottoneu team pages are login-gated for roster/lineup detail.
OTTONEU_SESSION_COOKIE = "REPLACE_ME"


# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

TIMEZONE = "America/New_York"
LOOKAHEAD_DAYS = 1          # evaluate tomorrow's slate (run agent the night before lock)
RECENT_FORM_GAMES = 15      # rolling window for "hot/cold" signal
EVALUATE_HITTERS = False    # scoped to pitching-only for now (2026-08-27) -- hitter matchup
                             # logic stays in the codebase (matchup_signal.py, the vsR/vsL/
                             # last15 splits fetching) and is a one-line flip back on later.
                             # Batters were part of the original design; pitching matters
                             # more right now, and dropping the vsR/vsL fetches when this is
                             # False also roughly halves the page-fetch work per run.
SPLIT_SEASON = None         # None = current season, inferred at runtime


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # start cheap; step up to claude-sonnet-5 only if reasoning quality falls short
CLAUDE_MAX_TOKENS = 6000     # raised from 2000 -- a full 20+ batter roster with per-player rationale genuinely needs this much room; 2000 truncated mid-response


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
EMAIL_SUBJECT_PREFIX = "Lineup Recommendation"
# Real sending is intentionally not wired up yet (see README) -- the agent
# writes a .html file you can open, and an .eml you could attach/forward.
SEND_EMAIL = False
SMTP_CONFIG = {
    "host": "REPLACE_ME",       # e.g. smtp.gmail.com
    "port": 587,
    "username": "REPLACE_ME",
    "app_password": "REPLACE_ME",  # use an app password, never your real password
    "from_addr": "REPLACE_ME",
    "to_addr": "REPLACE_ME",
}


# ---------------------------------------------------------------------------
# Ottoneu FanGraphs Points (FGP) scoring
# ---------------------------------------------------------------------------
# CONFIRMED for league 569 (Kaboom's Pepper Palace): the league settings
# page (https://ottoneu.fangraphs.com/569/settings) shows "Scoring System:
# FanGraphs Points" linking to Ottoneu's own scoring-options page, meaning
# this league runs the standard FGP table rather than a commissioner
# customization. Values below are copied directly from
# https://ottoneu.fangraphs.com/scoringoptions (credited there to work by
# Justin Merry). If your league ever shows custom numbers on that settings
# page instead of a link to the standard table, use those instead.
#
# Important structural note: this is AB-based, not single/double/triple-
# based. Every at-bat is -1.0 regardless of outcome; a hit adds +5.6 on
# top of that, and extra-base hits add a further bonus on top of that. So
# net values work out to single=4.6, double=7.5, triple=10.3, HR=14.0 --
# but keep the formula additive (AB, then H, then XBH bonus) rather than
# collapsing to those net numbers, since AB also has to cover plain outs
# (which have no H/2B/3B/HR component at all, just the -1.0).
#
# Pitching has NO wins, quality starts, or earned runs in this system --
# it's a FIP-style formula off IP/K/H/BB/HBP/HR/SV/HLD only. Don't add
# W/QS/ERA scoring back in; it isn't part of this league's math.

@dataclass
class BattingScoring:
    at_bat: float = -1.0     # every AB, hit or not
    hit: float = 5.6         # on top of the AB value, for any hit
    double_bonus: float = 2.9   # additional, on top of `hit`, for a 2B
    triple_bonus: float = 5.7   # additional, on top of `hit`, for a 3B
    home_run_bonus: float = 9.4  # additional, on top of `hit`, for a HR
    walk: float = 3.0
    hit_by_pitch: float = 3.0
    stolen_base: float = 1.9
    caught_stealing: float = -2.8


@dataclass
class PitchingScoring:
    inning_pitched: float = 7.4   # per full inning; use outs/3 for partials
    strikeout: float = 2.0
    hit_allowed: float = -2.6
    walk_allowed: float = -3.0
    hit_batsman: float = -3.0
    home_run_allowed: float = -12.3
    save: float = 5.0
    hold: float = 4.0


@dataclass
class ScoringConfig:
    batting: BattingScoring = field(default_factory=BattingScoring)
    pitching: PitchingScoring = field(default_factory=PitchingScoring)


SCORING = ScoringConfig()
