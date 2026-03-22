# 2026 Ottoneu Fantasy Baseball Analysis — Project Brief

## Overview
A Python-based analytics tool for a 12-team Ottoneu fantasy baseball league (League ID: 569, FanGraphs Points scoring). Projects team performance using OOPSY/ATC/THE BAT X projections with an optimized lineup model. Deployed as a public Hugging Face Space shared with a co-manager.

## Links
- Live app: https://huggingface.co/spaces/lsf4311/Farva-Operations-Center
- HF Space git remote: `hf` → https://huggingface.co/spaces/lsf4311/Farva-Operations-Center
- GitHub repo: https://github.com/lsf4311-lgtm/large-farva-projections (public)

## Deployment
- **Platform:** Hugging Face Spaces (migrated from Streamlit Community Cloud March 2026)
- **Why migrated:** Streamlit suspended account for fair use violations (cause unclear — possibly scraping volume or optimizer CPU). HF Spaces has no such restrictions for this use case.
- **Secrets:** FG_USER and FG_PASS stored in HF Space Settings → Variables and Secrets (not in code or repo)
- **Rebuilds:** triggered by `git push hf main`
- **Push to both remotes when updating:**
  ```
  git push origin main   # GitHub
  git push hf main       # Hugging Face
  ```
- **NOTE:** `st.dataframe(width='stretch')` is not supported on newer Streamlit — use `use_container_width=True` instead. All instances updated in app.py.

## File Structure
Local project folder: `C:\Users\lsf43\Desktop\2026 ottoneu analysis with Claude\`

Tracked in both GitHub and HF (committed to git):
  league_analysis_final.py     - main pipeline + pitching report functions
  app.py                       - Streamlit app (7 pages)
  requirements.txt             - pinned: streamlit==1.32.0, altair==4.2.2, beautifulsoup4 added
  README.md                    - HF Space config (sdk, app_file, etc.)
  data/
    league_rosters.csv
    players_with_projections.csv
    team_projections.csv
    roster_scrape_timestamp.txt
    sfbb_crosswalk.csv
    fangraphs-leaderboard-projections_oopsy hitting 2026.csv
    fangraphs-leaderboard-projections_oopsy pitching 2026.csv
    fangraphs-leaderboard-projections_2026 hitting_atc.csv
    fangraphs-leaderboard-projections_2026 pitching_atc.csv
    fangraphs-leaderboard-projections_thebatx hitting 2026.csv
    fangraphs-leaderboard-projections_thebatx pitching 2026.csv

Local only (never push):
  .streamlit/secrets.toml      <- FG_USER and FG_PASS (in .gitignore)
  keepwarm.py                  <- pings HF Space URL daily
  keepwarm.log                 <- keepwarm run history
  run_keepwarm.bat             <- Task Scheduler wrapper

## Stack
- Python 3.13 in VS Code
- Libraries: requests, pandas, beautifulsoup4, rapidfuzz, pulp, streamlit==1.32.0, altair==4.2.2
- Deployed on Hugging Face Spaces (free tier)
- Secrets: FG_USER, FG_PASS stored in HF Space settings

## League Details
- 12 teams, FanGraphs Points scoring
- Active lineup: 2C, 1B, 2B, SS, 3B, MI (2B/SS only), 5 OF, Util (any hitting position) = 13 hitters; 5 SP, 5 RP = 10 pitchers
- Season caps: 162 games per batting slot (catchers share 162 across 2 spots), 1500 IP shared across all pitchers
- Keeper league with annual auction draft
- Ottoneu roster page: salary at cells[1], position at cells[2]

## Pipeline (league_analysis_final.py)

### Step 1: get_league_rosters()
Scrapes all 12 team roster pages from Ottoneu. Returns player name, Ottoneu fg_id, position, salary, player_type (hitters/pitchers). Saves roster_scrape_timestamp.txt on success.

### Step 2: Projection fetch (auto-refresh)
fetch_projections(system, username, password) logs into FanGraphs via WordPress auth (blogs.fangraphs.com/wp-login.php), hits api/projections endpoint for all 4 CSVs (OOPSY/ATC hitting/pitching), saves fresh copies to disk. Falls back to CSV if login fails. Credentials stored in HF Secrets (FG_USER, FG_PASS). Sidebar shows "✓ live" or "⚠️ cached" with timestamp.

Projection endpoints:
  api/projections?type=oopsy&stats=bat  (and pit, atc, atcdc, oopsydc, thebatx, thebatxdc variants)
  OOPSY DC / ATC DC / THE BAT X DC unlock on Opening Day for in-season use.

### Step 3: Crosswalk merge
SFBB crosswalk bridges Ottoneu IDs to FanGraphs IDs via OTTONEUID -> IDFANGRAPHS. Merges with projections on FanGraphs ID.

### Step 4: fuzzy_match_players()
rapidfuzz fallback (threshold=90) for ~70+ players missing from crosswalk. Flags matches below 95% as needs_review. Only fires for players with FPTS=0 after crosswalk merge.

### Step 5: optimize_lineup() - TWO-PHASE
Phase 1: PuLP linear programming fills all constrained slots optimally, excluding Util.
Phase 2: Best remaining Util-eligible player (C/1B/2B/SS/3B/OF) fills Util slot.
Post-processing: SP and RP slots re-ordered by FPTS descending so SP1 is always the best starter.

### Step 6: FA position data
get_fa_positions() fetches accurate position eligibility from FanGraphs JSON API:
- Hitters: one request per position (C/1B/2B/SS/3B/OF) using api/leaders/major-league/data
- Pitchers: stats=sta → SP, stats=rel → RP (combined if both)
- Crosswalk fallback for hitters missing from API
- Name-based filter (rostered_names) catches rostered players missing from crosswalk (e.g. Luis Castillo IDFANGRAPHS=NaN bug)

### Key Fixes Baked In
- Crosswalk OTTONEUID float suffix (.0) - fixed with int(float(x))
- Crosswalk duplicate Ohtani entry - drop_duplicates(subset='OTTONEUID')
- Ohtani two-way - manually adds hitting FPTS on top of pitching FPTS
- Negative FPTS (-1.0) = 1 AB projection, treated as legitimate
- File paths use os.path.join for cross-platform compatibility
- FA filter uses BOTH fg_id and player name to prevent rostered players leaking into FA pool
- Player Search now searches both rostered players AND free agents (fix: March 2026)

### Known Limitations
- Ohtani appears as pitcher only in optimizer. Hitting FPTS added to his pitching row. Will not appear in Util or hitting slots. Noted with asterisk on Positional Breakdown page.
- ~3 missing projections per team are true prospects, treated as 0.
- Salary figures do not account for cap penalties.
- Scraper can get rate-limited by Ottoneu if run too frequently. Fallback to cached CSV handles this gracefully.
- Pitching Report page (stats/grades/rankings) is blank pre-season — Baseball Savant has no data until Opening Day.
- ~70+ rostered players have no IDFANGRAPHS in SFBB crosswalk (mostly prospects/international). These rely on name-based filter. Crosswalk updates over time as SFBB adds entries.

## Streamlit App (app.py)

### Design
Dark theme, IBM Plex Mono/Sans fonts, navy/blue palette. Cached weekly via @st.cache_data(ttl=604800). Manual refresh button in sidebar clears cache and re-scrapes everything.

### Load Order (important)
1. Page config + styling
2. Imports from league_analysis_final
3. PROJECTION_FILES dict (all 6 systems: OOPSY, ATC, THE BAT X, OOPSY DC, ATC DC, THE BAT X DC)
4. Projection system selector in sidebar (MUST be declared before load_all_data call)
5. load_all_data(projection_system) definition
6. Data load call
7. Sidebar freshness widgets
8. Pages

### Sidebar
- Projection system toggle: OOPSY / ATC / THE BAT X (base), Preseason / In-Season (DC) (type)
- Refresh button
- Data Freshness: roster timestamp (amber=cached, green=live), projection freshness (amber=cached, green=live with date)

### Pages
1. Standings - full league table with top-line metrics
2. Team Detail - defaults to Large Farva, roster split into starters/bench sorted by FPTS
3. Positional Breakdown - slot-level detail + position group summary. Ohtani asterisk.
4. Free Agent Targets - Best Available by Position (one row per position, gain colored) + Full FA list with position filter
5. Player Search - search by name, shows player/team/position/salary/FPTS/status. Searches both rostered players AND free agents (FAs show Team="Free Agent", Status="FA")
6. Head to Head - FPTS gap + side-by-side rosters
7. Pitching Report - see below

### Pitching Report Page
Caching: team stats weekly (ttl=604800, slow ~2 min), pitcher data daily (ttl=86400).
Data sources: MLB Stats API (schedule, player IDs, rotation), Baseball Savant (pitcher stats, team batting stats).
Sections per pitcher:
  - Summary ranking table at top (sorted by best matchup grade, then games this week)
  - 5 metric cards: IP, K%, BB%, wOBA Against, Hard Hit%
  - Rotation info: last start, avg rest days, next predicted start
  - Home/Away splits table
  - Per-matchup grade cards (A-F) with opponent batting stats and MLB rankings
Grade logic: A-F based on pitcher K%/BB%/wOBA splits vs opponent OPS at relevant location.
Opponent rankings: OPS, wOBA, K%, HR rate vs all 30 teams at home or away.
Pre-season: page loads but stats/grades are blank (no Savant data until Opening Day).

### load_all_data() returns
all_players, standings, free_agents, rosters_with_fgid, crosswalk, atc_hitting, atc_pitching, roster_source, proj_source, last_updated, projection_system

### Scraper Fallback
If Ottoneu scrape fails → cached CSV, roster_source='cached', amber sidebar warning.
If FanGraphs projection fetch fails → cached CSV, proj_source='cached', amber sidebar warning.

## Current Projected Standings (OOPSY, Two-Phase Optimized Lineups)
1  Large Farva               389    16205.8    40
2  Big Trouble               378    16108.2    32
3  Hollyhood                 396    16063.1    40
4  The Milwaukee Beers       394    15708.7    38
5  Chyna Jr                  393    15630.4    31
6  Bartolo's Meatballs       394    15587.7    37
7  Titty City                365    15441.6    40
8  Cat & Kaboom              400    15315.0    40
9  Busch Banditos            364    15025.2    40
10 Vance Munson VagHawks     375    14648.7    37
11 Rick Vaughn's Wild Things 391    14544.2    39
12 The Baseball Team         346    12106.6    38

## Roadmap

### ✅ Completed
- ATC vs OOPSY projection toggle (sidebar dropdown)
- OOPSY DC / ATC DC in-season variants (unlock Opening Day)
- THE BAT X and THE BAT X DC projection systems added
- Auto-refresh projections from FanGraphs API (daily, with CSV fallback)
- FA position accuracy: FanGraphs API for hitters (C/1B/2B/SS/3B/OF), sta/rel for pitchers (SP/RP)
- Rostered player leak fix: name-based secondary filter catches crosswalk ID gaps
- Pitching Report page: schedule, matchup grades, season stats, rotation prediction, opponent rankings, summary ranking table
- Player Search fix: now searches both rostered players and free agents
- Migrated from Streamlit Community Cloud to Hugging Face Spaces
- Keepwarm script: Windows Task Scheduler job fixed (cmd.exe wrapper handles path spaces), pings HF Space URL daily at 8am

### P2 - Next
- **Inflation model** (same app or separate TBD — scope further first)
  
  **What it needs to do:**
  - Per position: total projected FPTS, keeper FPTS locked up, keeper salary, implied $/FPTS for kept players
  - Remaining auction budget = $400 × 12 teams minus all keeper salaries
  - Remaining FPTS = total FA pool FPTS
  - Implied auction $/FPTS rate → dollar values assigned to each FA
  - Positional scarcity adjustment: positions with more FPTS locked in keepers get inflated prices
  - Historical calibration: validate model vs actual auction prices, identify position premiums, star player premiums, scarcity patterns
  
  **Data pipeline:**
  - Current keeper data: already have via roster scrape (all 12 teams public)
  - Historical data: Ottoneu transaction log (every add/drop/trade with salary and date). Auction acquisitions identified by clustering transactions on auction day (spike of adds on same date). Need sample of transaction log to assess format and cleaning required.
  - Historical projections: need OOPSY/ATC CSVs from prior years to pair with auction prices for validation
  
  **V1 scope (no historicals):** positional scarcity + implied dollar values — buildable in one session, ~80% of the value
  **V2 scope (with historicals):** calibrated model with position/player premiums — more accurate, requires transaction log parsing first
  
  **Decision pending:** same app (new page) vs separate app — leaning separate given it's a pre-draft tool with different workflow, but decide after scoping session

### P3 - Medium Term
- Prospect breakout page: cross-reference top prospects against rosters, flag unowned ones worth stashing.
- Load time optimization (if needed): parallelize get_fa_positions() with ThreadPoolExecutor (8 concurrent requests instead of sequential). Risk: more likely to trigger FanGraphs rate limiting. Only worth doing if load time becomes a real issue in daily use — current weekly cache means slow load only hits once per week anyway.

### P4 - Offseason/Long Term
- In-season actuals + weekly snapshots: blend actual FPTS with remaining projected, standings movement over time.
- Ohtani two-way fix: inject synthetic hitter row so he appears in Util.
- Salary efficiency view: FPTS per dollar, identify over/underpaid players.

## Key Technical Notes
- Player ID systems: Ottoneu IDs != FanGraphs IDs. SFBB crosswalk bridges via OTTONEUID -> IDFANGRAPHS
- FanGraphs WordPress login: POST to blogs.fangraphs.com/wp-login.php with log/pwd/rememberme. Auth confirmed by wordpress_sec_* cookie (not wordpress_logged_in — duplicate cookie names cause CookieConflictError in dict() but session still works).
- FanGraphs projection API: api/projections?type={oopsy|atc|oopsydc|atcdc|thebatx|thebatxdc}&stats={bat|pit}&pos=all&team=0&players=0&lg=all — requires auth cookie, returns JSON list directly.
- FanGraphs FA position API: api/leaders/major-league/data?pos={pos}&stats={bat|sta|rel}&fl=569&ft=-1 — no auth needed.
- Baseball Savant data: baseballsavant.mlb.com/statcast_search/csv — pitch-level CSV, no auth needed.
- requirements.txt must pin streamlit==1.32.0 and altair==4.2.2 to avoid altair.vegalite.v4 error. beautifulsoup4 must be explicitly listed.
- HF Spaces free tier sleeps inactive apps — keepwarm.py pings daily at 8am to prevent this
- DATA_DIR auto-detects: uses data subfolder locally, falls back to script directory on cloud
- .streamlit/secrets.toml must be in .gitignore — never push credentials to GitHub or HF
- st.dataframe() width parameter: use use_container_width=True, NOT width='stretch' (breaks on newer Streamlit versions)
- Keepwarm Task Scheduler: use cmd.exe as program with /c "full\path\to\run_keepwarm.bat" as argument — direct .bat path fails when folder name contains spaces
