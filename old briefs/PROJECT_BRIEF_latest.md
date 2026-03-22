# 2026 Ottoneu Fantasy Baseball Analysis — Project Brief

## Overview
Building a Python-based analytics tool for a 12-team Ottoneu fantasy baseball league (League ID: 569, FanGraphs Points scoring). The goal is to project team performance and model auction draft inflation. Deployed as a Streamlit app shared with a co-manager.

## File Structure
```
C:\Users\lsf43\Desktop\2026 ottoneu analysis with Claude\
  data\
    league_rosters.csv
    players_with_projections.csv
    team_projections.csv
    fangraphs-leaderboard-projections_oopsy hitting 2026.csv
    fangraphs-leaderboard-projections_oopsy pitching 2026.csv
    fangraphs-leaderboard-projections_2026 hitting.csv
    fangraphs-leaderboard-projections_2026 pitching.csv
    sfbb_crosswalk.csv
  league_analysis_final.py           <- main pipeline script
  app.py                             <- Streamlit app
```

## Stack
- Python 3.13 in VS Code
- Libraries: requests, pandas, beautifulsoup4, rapidfuzz, pulp, streamlit
- Separate scheduled email script (untouched) for weekly pitching reports

## League Details
- 12 teams, FanGraphs Points scoring
- Active lineup: 2C, 1B, 2B, SS, 3B, MI (2B/SS), 5 OF, Util (any) = 13 hitters; 5 SP, 5 RP = 10 pitchers
- Season caps: 162 games per batting slot (catchers share 162 across 2 spots), 1500 IP shared across all pitchers
- Keeper league with annual auction draft
- Ottoneu roster page: salary at cells[1], position at cells[2]

## Pipeline (league_analysis_final.py)
1. get_league_rosters() - scrapes all 12 team roster pages
2. Projection merge - SFBB crosswalk bridges Ottoneu IDs to FanGraphs IDs, merges with OOPSY
3. fuzzy_match_players() - rapidfuzz fallback (threshold=90) for players missing crosswalk IDs
4. optimize_lineup() - PuLP linear programming assigns players to optimal positional slots
5. Standings output - sorted by projected FPTS, saved to CSV

### Key Fixes Baked In
- Crosswalk OTTONEUID float suffix (.0) - fixed with int(float(x))
- Crosswalk duplicate Ohtani entry - drop_duplicates(subset='OTTONEUID')
- Ohtani two-way - manually adds hitting FPTS on top of pitching FPTS
- Negative FPTS (-1.0) = 1 AB projection, treated as legitimate

## Streamlit App (app.py)
Four pages, dark theme (IBM Plex fonts, navy/blue palette):
- Standings - full league table with top-line metrics
- Team Detail - roster split into starters/bench with salary and FPTS
- Player Search - search by name across all rosters
- Head to Head - two-team FPTS comparison + side by side rosters

Data cached weekly via @st.cache_data(ttl=604800). Manual refresh button in sidebar.
Currently running locally at http://localhost:8501

## Current Projected Standings (OOPSY, Optimized Lineups)
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

## Next Steps (Priority Order)
1. Deploy to Streamlit Community Cloud - GitHub setup, handle hardcoded file paths, public URL for co-manager
2. Positional breakdown table - each team's projected FPTS by position (C, 1B, 2B, SS, 3B, OF, SP, RP)
3. ATC vs OOPSY toggle - sidebar dropdown to switch projection systems
4. Inflation model - keeper salaries, remaining free agent pool, team budgets, positional scarcity, retroactive validation
5. In-season actuals + weekly snapshots - blend actual FPTS with remaining projected, weekly standings movement chart

## Key Technical Notes
- Player ID systems: Ottoneu IDs != FanGraphs ATC/OOPSY IDs. SFBB crosswalk bridges via OTTONEUID -> IDFANGRAPHS
- Two projection files needed (hitting + pitching), both with PlayerId renamed to fg_id
- app.py imports from league_analysis_final via sys.path.append
- File paths currently hardcoded to local machine - needs fixing for cloud deployment
- GitHub repo needed before Streamlit Cloud deployment
