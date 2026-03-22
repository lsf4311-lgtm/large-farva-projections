# 2026 Ottoneu Fantasy Baseball Analysis — Project Brief

## Overview
A Python-based analytics tool for a 12-team Ottoneu fantasy baseball league (League ID: 569, FanGraphs Points scoring). Projects team performance using OOPSY/ATC projections with an optimized lineup model. Deployed as a public Streamlit app shared with a co-manager.

## Links
- Live app: https://large-farva-projections-alpha.streamlit.app
- GitHub repo: https://github.com/[your-username]/large-farva-projections (public)

## File Structure
GitHub repo (large-farva-projections):
  league_analysis_final.py     - main pipeline script
  app.py                       - Streamlit app
  requirements.txt             - pinned: streamlit==1.32.0, altair==4.2.2
  fangraphs-leaderboard-projections_oopsy hitting 2026.csv
  fangraphs-leaderboard-projections_oopsy pitching 2026.csv
  fangraphs-leaderboard-projections_2026 hitting.csv
  fangraphs-leaderboard-projections_2026 pitching.csv
  sfbb_crosswalk.csv

Local only (not in GitHub):
  C:\Users\lsf43\Desktop\2026 ottoneu analysis with Claude\
    data\
      league_rosters.csv
      players_with_projections.csv
      team_projections.csv
      roster_scrape_timestamp.txt     <- saved on each successful scrape

## Stack
- Python 3.13 in VS Code
- Libraries: requests, pandas, beautifulsoup4, rapidfuzz, pulp, streamlit==1.32.0, altair==4.2.2
- Separate scheduled email script (untouched) for weekly pitching reports
- Deployed on Streamlit Community Cloud (free tier)

## League Details
- 12 teams, FanGraphs Points scoring
- Active lineup: 2C, 1B, 2B, SS, 3B, MI (2B/SS only), 5 OF, Util (any hitting position) = 13 hitters; 5 SP, 5 RP = 10 pitchers
- Season caps: 162 games per batting slot (catchers share 162 across 2 spots), 1500 IP shared across all pitchers
- Keeper league with annual auction draft
- Ottoneu roster page: salary at cells[1], position at cells[2]

## Pipeline (league_analysis_final.py)

### Step 1: get_league_rosters()
Scrapes all 12 team roster pages from Ottoneu. Returns player name, Ottoneu fg_id, position, salary, player_type (hitters/pitchers). Saves roster_scrape_timestamp.txt on success.

### Step 2: Projection merge
SFBB crosswalk bridges Ottoneu IDs to FanGraphs IDs via OTTONEUID -> IDFANGRAPHS. Merges with OOPSY projections on FanGraphs ID.

### Step 3: fuzzy_match_players()
rapidfuzz fallback (threshold=90) for ~70 players missing from crosswalk. Flags matches below 95% as needs_review. Only fires for players with FPTS=0 after crosswalk merge.

### Step 4: optimize_lineup() - TWO-PHASE
Phase 1: PuLP linear programming fills all constrained slots (C, 1B, 2B, SS, 3B, MI, OF1-5, SP1-5, RP1-5) optimally, excluding Util.
Phase 2: Best remaining Util-eligible player (C/1B/2B/SS/3B/OF) fills Util slot.
Post-processing: SP and RP slots re-ordered by FPTS descending so SP1 is always the best starter.

### Key Fixes Baked In
- Crosswalk OTTONEUID float suffix (.0) - fixed with int(float(x))
- Crosswalk duplicate Ohtani entry - drop_duplicates(subset='OTTONEUID')
- Ohtani two-way - manually adds hitting FPTS on top of pitching FPTS (pitchers-only on Ottoneu)
- Negative FPTS (-1.0) = 1 AB projection, treated as legitimate
- File paths use os.path.join for cross-platform compatibility
- from datetime import datetime needed in league_analysis_final.py for timestamp

### Known Limitations
- Ohtani appears as pitcher only in optimizer. Hitting FPTS added to his pitching row. Will not appear in Util or hitting slots. Noted with asterisk on Positional Breakdown page.
- ~3 missing projections per team are true prospects, treated as 0.
- Salary figures do not account for cap penalties.
- Scraper can get rate-limited by Ottoneu if run too frequently. Fallback to cached CSV handles this gracefully.

## Streamlit App (app.py)

### Design
Dark theme, IBM Plex Mono/Sans fonts, navy/blue palette. Cached weekly via @st.cache_data(ttl=604800). Manual refresh button in sidebar clears cache and re-scrapes everything.

### Load Order (important)
1. Page config + styling
2. Imports from league_analysis_final
3. load_all_data() definition
4. Load data call (all_players, standings, free_agents, rosters_with_fgid, crosswalk, atc_hitting, atc_pitching, roster_source, last_updated = load_all_data())
5. Sidebar (must be AFTER data load to access roster_source)
6. Pages

### Sidebar
Shows navigation, refresh button, and Data Freshness section with:
- Roster timestamp (from roster_scrape_timestamp.txt) - amber warning if cached, green if live
- Projection freshness (file modification date of OOPSY hitting CSV)

### Pages
1. Standings - full league table (all 12 rows visible) with top-line metrics
2. Team Detail - defaults to Large Farva, roster split into starters/bench sorted by FPTS
3. Positional Breakdown - slot-level detail table + position group summary (Util player reassigned to primary position using first-listed position as tiebreaker). Ohtani asterisk at top.
4. Free Agent Targets - two sections:
   - Best Available by Position: one row per position (C,1B,2B,SS,3B,OF,SP,RP), best FA vs weakest rostered player (starters+bench), gain colored green/red. Team selector defaults to Large Farva.
   - Full Free Agent List: top 50 FAs with position filter dropdown
5. Player Search - search by name, shows player/team/position/salary/FPTS/status
6. Head to Head - defaults Large Farva on left, FPTS gap + side-by-side rosters

### load_all_data() returns
all_players, standings, free_agents, rosters_with_fgid, crosswalk, atc_hitting, atc_pitching, roster_source, last_updated

### Scraper Fallback
If Ottoneu scrape fails, loads league_rosters.csv from disk and sets roster_source='cached'. Sidebar shows amber warning with last successful scrape date.

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

## Roadmap (Priority Order)

### P1 - Next Session (quick wins)
1. ATC vs OOPSY toggle - sidebar dropdown to switch projection systems. Both CSVs already in GitHub. Estimated 20-30 mins.

### P2 - Soon
2. Inflation model - keeper salaries, free agent pool, team budgets, positional scarcity, retroactive validation. Post-draft this year but valuable for next year's prep. Notable: this year's draft had significant positional inflation trends worth analyzing.
3. Auto-refresh projections - scrape OOPSY/ATC exports from FanGraphs automatically. Requires FanGraphs login handling (co-manager has paid account, credentials storable as Streamlit secrets). Medium complexity.

### P3 - Medium Term
4. Fold in pitching email script - integrate weekly pitcher matchup report (Baseball Savant stats, schedule scraping, matchup analysis) as a Streamlit page. Email can still send optionally.
5. Prospect-specific breakout - cross-reference top prospects against current rosters, flag unowned prospects worth stashing.

### P4 - Offseason/Long Term
6. In-season actuals + weekly snapshots - blend actual FPTS with remaining projected, standings movement chart over time.
7. Ohtani two-way fix - inject synthetic hitter row so he appears in Util.
8. Salary efficiency view - FPTS per dollar, identify over/underpaid players.

## Key Technical Notes
- Player ID systems: Ottoneu IDs != FanGraphs IDs. SFBB crosswalk bridges via OTTONEUID -> IDFANGRAPHS
- Two projection files needed (hitting + pitching), both with PlayerId renamed to fg_id
- app.py imports from league_analysis_final via sys.path.append + os.path.dirname(__file__)
- requirements.txt must pin streamlit==1.32.0 and altair==4.2.2 to avoid altair.vegalite.v4 error
- Streamlit free tier sleeps inactive apps - first visit after dormancy is slow (~30-60 sec)
- Streamlit free tier may delete inactive apps - redeploy from GitHub if needed, all code is safe there
- DATA_DIR auto-detects: uses data subfolder locally, falls back to script directory on cloud
- Projection CSVs are static in GitHub - manual refresh recommended monthly during season
- FanGraphs projection exports are free-account accessible, manual download takes ~5 mins
