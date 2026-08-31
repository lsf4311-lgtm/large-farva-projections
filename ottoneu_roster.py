"""
Ottoneu roster + stat-line fetch.

Confirmed live for league 569 (Kaboom's Pepper Palace) / team 4067
(Large Farva): https://ottoneu.fangraphs.com/569/setlineups?team=4067 is
public, no login needed. Confirmed further: the `statFilter` query param
genuinely switches the server-rendered stat line (not just a client-side
tab), and the computed FGP total from scoring.py matches Ottoneu's own
"Pts" column to the cent for every filter tested:

    statFilter=season   -> Yordan Alvarez 1044.2  vs Ottoneu 1044.20
    statFilter=vsR       -> Gabriel Moreno 433.8   vs Ottoneu 433.80
    statFilter=last15    -> Gabriel Moreno 121.7   vs Ottoneu 121.70

Confirmed valid statFilter values: season, vsR, vsL, last15, last7
(last7 inferred from the same naming pattern as last15 and the person's
own URL; not independently re-verified against a points total the way
the others were -- worth a spot-check the first time you rely on it).

This means the whole pipeline can run on Ottoneu's own numbers end to
end: no Statcast, no pybaseball, no MLBAM ID resolution. The trade-off is
no opponent-specific pitcher matchup signal (Ottoneu doesn't expose
opposing-lineup splits) -- see the README for how that's handled instead.

Same caveat as before on markup: I'm parsing against real column headers
and confirmed content, but I only ever see this page as fetch-rendered
text, never raw DOM, so cell-index assumptions should be spot-checked
against the live page if parsing ever comes back empty or clearly wrong.
"""

from __future__ import annotations
import datetime as dt
import re
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

import config

PLAYER_URL_RE = re.compile(r"/players/(\d+)")
TEAM_POS_HAND_RE = re.compile(r"\b([A-Z]{2,3})\b\s+((?:[A-Z0-9]+/)*[A-Z0-9]+)\s+([LRS])\b")

# Confirmed against real (non-markdown) cell text from the Opponent column,
# e.g. "SDP 7:10 PM EDT Casey Mize R" or "@MIN 8:10 PM EDT Dean Kremer R".
# An earlier version of this regex was accidentally written against
# web_fetch's Markdown-rendered view of the page (which showed link syntax
# like `[Name](url)`) rather than the real HTML `.get_text()` sees, so it
# never matched anything -- see chat history. This version is verified
# against 5 real rows across a range of times/teams/hands.
OPPONENT_CELL_RE = re.compile(
    r"^(@)?([A-Z]{2,3})\s+\d{1,2}:\d{2}\s*[AP]M\s*[A-Z]{2,4}\s+(.+?)\s+([LRS])$"
)

VALID_STAT_FILTERS = {"season", "vsR", "vsL", "last15", "last7"}

NON_ACTIVE_SLOTS = {"Bench", "Minors", "IL"}


@dataclass
class BatterStatLine:
    games: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    hbp: int = 0
    stolen_bases: int = 0
    caught_stealing: int = 0
    ottoneu_pts: float = 0.0


@dataclass
class PitcherStatLine:
    games: int = 0
    games_started: int = 0
    outs: int = 0
    strikeouts: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    hit_batsmen: int = 0
    home_runs_allowed: int = 0
    saves: int = 0
    holds: int = 0
    ottoneu_pts: float = 0.0


@dataclass
class RosterPlayer:
    ottoneu_id: int
    name: str
    roster_slot: str
    position_eligibility: list[str] = field(default_factory=list)
    mlb_team_abbrev: str = ""
    bats_or_throws: str = ""
    opponent_team_abbrev: str | None = None
    opposing_pitcher_name: str | None = None
    opposing_pitcher_hand: str | None = None
    is_probable_starter_today: bool = False
    is_pitcher: bool = False
    batting: BatterStatLine | None = None
    pitching: PitcherStatLine | None = None


def _session() -> requests.Session:
    s = requests.Session()
    if config.OTTONEU_SESSION_COOKIE and config.OTTONEU_SESSION_COOKIE != "REPLACE_ME":
        s.headers.update({"Cookie": config.OTTONEU_SESSION_COOKIE})
    return s


def _setlineups_url(target_date: dt.date | None, stat_filter: str) -> str:
    if stat_filter not in VALID_STAT_FILTERS:
        raise ValueError(f"Unrecognized statFilter {stat_filter!r}; confirmed values: {VALID_STAT_FILTERS}")
    url = (
        f"https://ottoneu.fangraphs.com/{config.OTTONEU_LEAGUE_ID}/"
        f"setlineups?team={config.OTTONEU_TEAM_ID}"
    )
    if target_date:
        url += f"&date={target_date.isoformat()}"
    url += f"&statFilter={stat_filter}"
    return url


def _int(text: str) -> int:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return 0


def _float(text: str) -> float:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_row(row, is_pitcher_table: bool) -> RosterPlayer | None:
    link = row.find("a", href=PLAYER_URL_RE)
    if not link:
        return None  # "Empty" slots, spacer rows

    ottoneu_id = int(PLAYER_URL_RE.search(link["href"]).group(1))
    name = link.get_text(strip=True)

    cells = row.find_all("td")
    if not cells:
        return None
    roster_slot = cells[0].get_text(strip=True)
    name_cell_text = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
    opp_cell_text = cells[2].get_text(" ", strip=True) if len(cells) > 2 else ""

    team_match = TEAM_POS_HAND_RE.search(name_cell_text)
    mlb_team = team_match.group(1) if team_match else ""
    positions = team_match.group(2).split("/") if team_match else []
    hand = team_match.group(3) if team_match else ""

    player = RosterPlayer(
        ottoneu_id=ottoneu_id,
        name=name,
        roster_slot=roster_slot,
        position_eligibility=positions,
        mlb_team_abbrev=mlb_team,
        bats_or_throws=hand,
        is_pitcher=is_pitcher_table,
    )

    has_game_today = opp_cell_text not in ("", "Not Available", "---")
    if has_game_today:
        opp_match = OPPONENT_CELL_RE.match(opp_cell_text)
        if opp_match:
            _, team, opp_name, opp_hand = opp_match.groups()
            player.opponent_team_abbrev = team
            player.opposing_pitcher_name = opp_name
            player.opposing_pitcher_hand = opp_hand
        else:
            # Pitcher rows may not carry a trailing "opposing pitcher +
            # hand" the way batter rows do (a pitcher doesn't face an
            # opposing pitcher) -- confirmed batter format, NOT yet
            # confirmed for pitcher rows. Fall back to just the leading
            # team abbreviation so we at least know there's a game.
            team_only = re.match(r"^(@)?([A-Z]{2,3})\b", opp_cell_text)
            if team_only:
                player.opponent_team_abbrev = team_only.group(2)
        # NOTE: "is a probable starter today" has no separate text
        # signal on a future-dated fetch (no "Batting N" / "Not
        # starting" the way there is on today's/past slates) -- using
        # has_game_today as the proxy. Confirmed via real data: RP-slot
        # pitchers ALSO show game info in this cell (same format as SP
        # rows), so for pitchers specifically this must be narrowed to
        # roster_slot == "SP" or every reliever with a game gets treated
        # as a probable starter too.
        if is_pitcher_table:
            player.is_probable_starter_today = roster_slot == "SP"
        else:
            player.is_probable_starter_today = True

    # Stat columns: index from the right since leading columns (Opponent,
    # PC counts) are the messiest part of the rendered markup I've seen.
    # Batters (from confirmed headers): ... G AB H 2B 3B HR BB HBP SB CS
    # Pitchers: ... G GS IP SV HLD K H BB HBP HR
    values = [c.get_text(strip=True) for c in cells]
    try:
        if is_pitcher_table:
            g, gs, ip, sv, hld, k, h, bb, hbp, hr = values[-10:]
            pts = values[-11] if len(values) >= 11 else "0"
            innings_whole, _, innings_partial = ip.partition(".")
            outs = _int(innings_whole) * 3 + _int(innings_partial)  # e.g. "5.1" IP = 16 outs
            player.pitching = PitcherStatLine(
                games=_int(g), games_started=_int(gs), outs=outs,
                strikeouts=_int(k), hits_allowed=_int(h), walks_allowed=_int(bb),
                hit_batsmen=_int(hbp), home_runs_allowed=_int(hr),
                saves=_int(sv), holds=_int(hld), ottoneu_pts=_float(pts),
            )
        else:
            g, ab, h, tb2, tb3, hr, bb, hbp, sb, cs = values[-10:]
            pts = values[-11] if len(values) >= 11 else "0"
            player.batting = BatterStatLine(
                games=_int(g), at_bats=_int(ab), hits=_int(h),
                doubles=_int(tb2), triples=_int(tb3), home_runs=_int(hr),
                walks=_int(bb), hbp=_int(hbp), stolen_bases=_int(sb),
                caught_stealing=_int(cs), ottoneu_pts=_float(pts),
            )
    except ValueError:
        pass  # row didn't have the expected stat columns (e.g. summary row) -- skip stats, keep identity fields

    return player


def fetch_roster_and_matchups(target_date: dt.date | None = None,
                               stat_filter: str = "season") -> list[RosterPlayer]:
    resp = _session().get(_setlineups_url(target_date, stat_filter), timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players: list[RosterPlayer] = []
    for table in soup.find_all("table"):
        header_text = table.get_text(" ", strip=True)[:250]
        if "Position" not in header_text or "Name" not in header_text:
            continue
        is_pitcher_table = "IP" in header_text and "AB" not in header_text
        for row in table.find_all("tr"):
            parsed = _parse_row(row, is_pitcher_table)
            if parsed:
                players.append(parsed)

    return players


def fetch_roster() -> list[RosterPlayer]:
    return fetch_roster_and_matchups(target_date=None)
