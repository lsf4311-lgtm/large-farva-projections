"""
The actual "agentic" step: hand Claude pre-shaped matchup data and ask for
a ranked recommendation with rationale, not just a sort by points/PA.

Deliberately NOT just "sort players by matchup_points_per_pa and call it a
day" -- that's the automation you already had. What Claude adds here is
judging close calls, flagging small-sample splits it shouldn't be trusted,
and writing the "why" a human can sanity-check in five seconds.
"""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass

import anthropic

import config


@dataclass
class BatterInput:
    name: str
    position_eligibility: list[str]
    opponent_pitcher_name: str
    opponent_pitcher_hand: str
    matchup_pts_per_pa: float
    season_pts_per_pa: float
    platoon_pa_sample: int
    recent_pts_per_pa: float
    season_avg: float | None = None
    season_obp: float | None = None
    season_slg: float | None = None
    season_ops: float | None = None
    season_woba: float | None = None


@dataclass
class PitcherInput:
    name: str
    opponent_team: str
    season_pts_per_out: float
    recent_pts_per_out: float
    recent_outs_sample: int
    matchup_grade: str | None = None          # 'A'-'F' from league_analysis_final.get_matchup_grade, or None if unavailable
    opponent_ops_rank: str | None = None       # e.g. "5/30" -- lower means tougher lineup
    opponent_ops: float | None = None
    rotation_confirmed: bool = True            # False if his start-day prediction is uncertain (see main.py)


@dataclass
class LineupSlateInput:
    date: str
    available_batters: list[BatterInput]
    available_pitchers: list[PitcherInput]
    roster_slots: dict[str, int]   # e.g. {"OF": 3, "IF": 4, "SP": 2, ...}


SYSTEM_PROMPT = """You are a fantasy baseball sit/start assistant for an \
Ottoneu FanGraphs Points (FGP) league. You will be given pre-computed \
matchup data for today's available batters and starting pitchers: each \
number is already expressed in Ottoneu FGP terms (points per plate \
appearance, or points per out for pitchers), computed with this league's \
own confirmed scoring formula, so higher is always better and the units \
are directly comparable across players.

Batter data includes season rate, platoon split vs today's opposing \
starter's handedness (blended toward the season rate when the platoon \
sample is small), and recent (last-15-day) form -- `matchup_pts_per_pa` \
is the blended number to weight most heavily. Season AVG/OBP/SLG/OPS/\
wOBA are also included as supporting context (wOBA in particular is a \
more stable, better overall-quality indicator than raw AVG) -- use them \
to explain *why* a matchup number looks the way it does, not as a \
separate scoring system. Ottoneu points-per-PA is still the number that \
determines this league's outcome; treat AVG/OBP/SLG/OPS/wOBA as \
supporting evidence in your rationale, not a competing ranking.

Pitcher data is season and last-15-day Ottoneu-scoring rate, PLUS a real \
opponent-specific matchup grade where available: `matchup_grade` is an \
A-F letter grade (A = best matchup, F = worst) computed from the \
pitcher's own K%/BB%/wOBA-against blended with the opposing lineup's \
OPS at that game's location, and `opponent_ops_rank` shows where that \
opponent ranks league-wide (e.g. "5/30" means the 5th-toughest lineup in \
MLB by that measure -- lower number is tougher). Rank 1 is the TOUGHEST \
lineup in MLB (highest OPS), rank 30 is the WEAKEST (lowest OPS) -- a \
single-digit or low-double-digit rank is a hard matchup, not a "weak" \
one, and calling it weak is backwards. Only describe an opponent as \
weak when their rank is in roughly the bottom third (around 20+/30). When `matchup_grade` is \
null, no opponent data was available (e.g. early season, or the matchup \
fell outside the data window) -- fall back to the pitcher's own rate \
stats and say so rather than guessing at matchup quality. When \
`rotation_confirmed` is false, this pitcher's presence in the list at \
all is based on an uncertain rotation prediction (not enough recent \
starts to confidently predict his next one) rather than a confirmed \
start -- flag that explicitly as a separate, more basic uncertainty \
than a small-sample matchup stat. CONSISTENCY RULE: when \
`rotation_confirmed` is false, the word "confirmed" (or equivalent \
certain language like "locked in", "set to start") must not appear \
anywhere in that pitcher's "reason" text in the "start" list -- say \
"expected to start" or "likely starting" instead. A reader who only \
sees the "start" section, without cross-referencing \
low_confidence_flags, should still come away with an accurate sense of \
confidence.

IMPORTANT: `available_pitchers` includes BOTH pitchers currently sitting \
in an active SP roster slot AND bench pitchers predicted to start that \
day. Do not assume a pitcher is already in the active lineup just \
because he's in this list. This means your "start" recommendations can \
and should include benching a currently-active starter in favor of a \
benched one with a better matchup or rate -- that reassignment is \
exactly the kind of call this data is meant to support, not an edge \
case to avoid.
than a small-sample matchup stat.

Your job:
1. Propose a starting lineup that fills the given roster slots, choosing \
the batters/pitchers you'd start over the ones you'd sit, using the \
matchup and recent-form data provided.
2. Give a short rationale for each start/sit call, especially close ones.
3. Explicitly flag any recommendation that rests on a small sample \
(platoon_pa_sample under ~40 PA, or recent_outs_sample under ~15 outs \
for pitchers) as low-confidence rather than stating it with the same \
certainty as a well-sampled one.
4. Do not invent stats you weren't given. If two players look close, say \
so rather than forcing a confident pick.
5. Keep each "reason" to ONE concise sentence (roughly 20 words) citing \
at most 2 of the most decision-relevant numbers -- not every stat you \
were given. A rationale someone can read in two seconds is more useful \
than an exhaustive one, and shorter reasons also matter mechanically \
here: with 20+ players in a slate, verbose per-player reasoning risks \
the response running out of room before it finishes.
6. If `available_batters` is empty, this is a pitching-only run by \
design -- do not comment on or fabricate hitter recommendations, and \
don't treat the empty list as missing data.

Respond ONLY with JSON matching this shape, no other text -- no leading \
or trailing commentary, no markdown code fences, nothing before the \
opening brace or after the closing one. If there are fewer candidates \
than roster_slots calls for (a thin day for available starters, for \
example), that's expected and not an error: recommend starting all \
available candidates and note the shortfall as a close_calls or \
low_confidence_flags entry, not as text outside the JSON structure.
{
  "date": "...",
  "start": [{"name": "...", "role": "batter|pitcher", "reason": "..."}],
  "sit": [{"name": "...", "role": "batter|pitcher", "reason": "..."}],
  "close_calls": [{"players": ["...", "..."], "note": "..."}],
  "low_confidence_flags": [{"name": "...", "reason": "small sample size etc"}]
}
"""


def get_recommendation(slate: LineupSlateInput) -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    payload = {
        "date": slate.date,
        "roster_slots": slate.roster_slots,
        "available_batters": [asdict(b) for b in slate.available_batters],
        "available_pitchers": [asdict(p) for p in slate.available_pitchers],
    }

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ],
    )

    text = "".join(block.text for block in message.content if block.type == "text").strip()

    if not text:
        raise RuntimeError(
            f"Claude returned no text content. stop_reason={message.stop_reason!r}, "
            f"content blocks={message.content!r}. This usually means either the "
            f"response got cut off (try raising config.CLAUDE_MAX_TOKENS) or the "
            f"model returned a non-text block type this code doesn't expect yet."
        )

    # Defensive: strip an accidental leading markdown code fence.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    # Use raw_decode instead of json.loads: raw_decode parses the first
    # complete JSON value and tells us where it ends, ignoring anything
    # after -- json.loads requires the ENTIRE string to be valid JSON
    # with nothing trailing, which breaks if the model appends a closing
    # code fence or a stray explanatory note after the JSON object
    # (confirmed happening in practice despite the prompt saying JSON
    # only, no other text).
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Claude's response wasn't valid JSON ({e}). First 2000 chars of what "
            f"it actually returned:\n\n{text[:2000]}"
        ) from e
