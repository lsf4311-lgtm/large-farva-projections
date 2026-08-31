# Fantasy Baseball Sit/Start Agent

Scaffold for the agent described in the steering doc: scheduled, bounded,
proposes a lineup with rationale and stops. No auto-actions.

## Pipeline

```
Ottoneu setlineups page, fetched 4x
(season / vsR / vsL / last15)         For league 569 (Kaboom's Pepper
        |                             Palace), team 4067 (Large Farva).
        v                             Confirmed public, no login needed.
scoring.py: raw stats -> Ottoneu FGP
(verified to the cent against
Ottoneu's own "Pts" column)
        |
        v
matchup_signal.py: blends season +
platoon (vs today's opposing
starter's hand) + last-15-day form
into one expected-points number
        |
        v
reasoning.py: shaped data -> Claude
-> ranked start/sit + rationale +
flagged low-confidence calls
        |
        v
email_output.py: renders .html/.txt
locally (SMTP sending stubbed, not
wired up -- see config.SEND_EMAIL)
```

Pitcher matchup grades come from a separate, already-existing pipeline
(`league_analysis_final.get_pitching_report`, reused via
`pitcher_matchup_report.py`) rather than from Ottoneu's page -- see
"Pitcher matchups reuse the existing Streamlit tool's pipeline" below.

## What changed from the first draft

The original version reconstructed platoon splits from Statcast via
pybaseball, because Ottoneu doesn't have a documented public API. Turned
out unnecessary: Ottoneu's own setlineups page already computes vs-R,
vs-L, and last-15-day splits in the league's *own* scoring, server-side,
behind a `statFilter` query param. That's a strictly better source than
reconstructing it -- no Statcast dependency, no MLBAM ID name-matching
(a real fragility point), no proxy math.

**Verified, not assumed:** I computed Ottoneu FGP by hand from the raw
stat lines Ottoneu displays and it matched their own "Pts" column to the
cent, for three different players/splits (season, vsR, last15). See the
docstrings in `scoring.py` and `ottoneu_roster.py` for the exact checks.

**Confirmed valid `statFilter` values:** `season`, `vsR`, `vsL`,
`last15`, `last7`. The first four were independently checked against
real point totals; `last7` was inferred from the same naming pattern and
the URL the person found, not independently re-verified -- worth a
quick sanity check the first time the agent relies on it.

## What's still a gap

- **Roster/lineup-slot parsing is regex-over-rendered-text, not
  DOM-verified.** I don't have raw HTML access to this page (my fetch
  tool renders to text), so `ottoneu_roster.py`'s row parser is written
  against confirmed column headers and link patterns rather than
  confirmed CSS selectors. If a run comes back with empty or obviously
  wrong stat lines, that's the first place to look -- add a quick
  `print(resp.text[:2000])` in `fetch_roster_and_matchups` and compare
  against what you see in a browser.
- **`reasoning.LineupSlateInput.roster_slots`** is still an empty dict in
  `main.py` -- fill in your league's actual starting lineup requirements
  (e.g. `{"C": 1, "1B": 1, "OF": 3, "SP": 3, ...}`).
- **Relief pitchers are skipped entirely** (`main.py` only evaluates
  probable starters). Holds/saves streaming would need its own logic.
- **The matchup-grade lookup relies on a rolling 7-day window from
  *today*, not from `target_date`** (that's how `get_pitcher_schedule`
  in `league_analysis_final.py` already works). Fine for
  `config.LOOKAHEAD_DAYS=1`; if that value ever grows past ~6, matchup
  grades will start coming back `None` for everyone and pitcher
  recommendations will silently fall back to rate stats only.
- **`pitcher_matchup_report.py`'s weekly file cache is new and
  untested against the real `get_all_team_stats()`** -- I verified the
  caching logic (refresh-once, reuse-on-second-call) against a stub,
  but not against the real ~2-minute Savant pull. First real run will
  be slow; every run within 7 days after should be fast.

## Pitcher matchups reuse the existing Streamlit tool's pipeline

`pitcher_matchup_report.py` imports `get_pitching_report` and
`get_all_team_stats` directly from `league_analysis_final.py` rather
than reinventing pitcher matchup logic -- that module already computes
a real A-F grade per start (pitcher's K%/BB%/wOBA-against blended with
the opponent's home/away OPS) plus the opponent's rank among all 30
teams. It just wasn't wired into a decision output before; this agent
is that wiring. The only thing added on top is a weekly JSON file cache
around the expensive team-stats pull, since a daily cron job shouldn't
re-run a 2-minute, 30-team Savant scrape every time.

## Advanced hitting stats (AVG/OBP/SLG/OPS/wOBA)

No new data source needed for this -- Ottoneu's raw AB/H/2B/3B/HR/BB/HBP
counts (already being pulled for scoring) are enough to compute all five
directly. `matchup_signal.advanced_stats()` does that, using the same
wOBA linear weights as `league_analysis_final.WOBA_WEIGHTS` (kept as a
local copy to avoid pulling in that module's Savant-calling code paths
just for a formula -- worth consolidating into one shared constants file
if the two codebases merge further). Sanity-checked against a known
season line (.317/.433/.610, 1.043 OPS) and it came out right.

These are season-level only for now (platoon/last-15 samples are small
enough that AVG/SLG get noisy fast) and are additive context, not a
second scoring system -- Ottoneu FGP is still what determines the
league. They show up in two places: as extra fields on
`reasoning.BatterInput` (so Claude's rationale can reference them), and
as a standalone table at the bottom of the email, built directly from
the computed numbers rather than routed through Claude's response --
that keeps the figures exact rather than model-transcribed.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```

`config.py` already has league 569 / team 4067 filled in. If you're
running this for a different team or league, update
`OTTONEU_LEAGUE_ID` / `OTTONEU_TEAM_ID`.

```bash
python main.py
```

This writes `output/lineup_recommendation_<date>.html` and `.txt`. Open
the HTML file to review. Nothing gets submitted to Ottoneu automatically.

## Guardrails checklist (from the steering doc -- don't relax these casually)

- [x] No automated actions on Ottoneu -- the agent stops at "here's a file to read."
- [x] Scheduled only (cron), no polling loop.
- [x] Data pulled from Ottoneu's own live pages, not model memory.
- [x] Scoring formula verified against real output, not assumed.
- [x] Pitcher matchup grades reuse an already-built, already-tested
      pipeline instead of a new guess.
- [x] Recommendations include rationale via the reasoning.py prompt contract.
- [ ] Log of recommendations vs. outcomes -- not built yet. Simplest
      version: append each day's `recommendation` dict + the next day's
      actual results (Ottoneu shows those too) to a CSV/SQLite table.
