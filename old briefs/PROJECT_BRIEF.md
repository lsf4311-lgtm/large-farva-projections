# 2026 Ottoneu Fantasy Baseball Analysis — Project Brief

## Overview
Building a Python-based analytics tool for a 12-team Ottoneu fantasy baseball league (League ID: 569, FanGraphs Points scoring). The goal is to project team performance and model auction draft inflation. Eventually to be deployed as a Streamlit app shared with a co-manager.

## File Structure
```
C:\Users\lsf43\Desktop\2026 ottoneu analysis with Claude\
  data\
    league_rosters.csv               ← scraped from Ottoneu
    players_with_projections.csv     ← rosters merged with OOPSY projections
    team_projections.csv             ← final optimized standings output
    fangraphs-leaderboard-projections_oopsy hitting 2026.csv
    fangraphs-leaderboard-projections_oopsy pitching 2026.csv
    sfbb_crosswalk.csv               ← SFBB Player ID Map (bridges Ottoneu IDs to FanGraphs IDs)
  python league_analysis.py          ← main script (clean version as of session 1)
```

## Stack
- Python 3.13 in VS Code
- Libraries: requests, pandas, beautifulsoup4, rapidfuzz, pulp
- Scheduled email script (separate file, untouched) using same Ottoneu scraping approach

## League Details
- 12 teams, FanGraphs Points scoring
- Ottoneu roster page scraping works (salary at cells[1], position at cells[2])
- Active lineup: 2C, 1B, 2B, SS, 3B, MI (2B/SS), 5 OF, Util (any) = 13 hitters; 5 SP, 5 RP = 10 pitchers
- Season caps: 162 games per batting slot (catchers share 162 across 2 spots), 1500 IP shared across all pitchers
- Keeper league with annual auction draft

## What's Built and Working
### Step 1: Roster Scraper (get_league_rosters)
- Scrapes all 12 team roster pages from Ottoneu
- Returns player name, Ottoneu fg_id, position, salary, player_type (hitters/pitchers)
- 452 players total

### Step 2: Projection Merge
- Uses SFBB crosswalk to bridge Ottoneu IDs → FanGraphs IDs
- Merges with OOPSY projections on FanGraphs ID
- Key fixes:
  - Crosswalk OTTONEUID had .0 float suffix — fixed with int(float(x)) conversion
  - Crosswalk has duplicate Ohtani entry — fixed with drop_duplicates(subset='OTTONEUID')
  - Ohtani is pitchers-only on Ottoneu but is two-way — manually adds hitting FPTS on top of pitching FPTS
  - ~70 players missing from crosswalk — handled by rapidfuzz fallback name matcher (threshold=90, flags needs_review if <95)
  - Negative FPTS (-1.0) = players projected for 1 AB only, treated as legitimate minimal projections

### Step 3: Lineup Optimizer (optimize_lineup)
- Uses PuLP linear programming to assign players to optimal positional slots
- Maximizes total FPTS subject to positional eligibility constraints
- Handles multi-position players correctly (e.g. 2B/3B/OF slotted optimally)
- Returns both total_fpts and full lineup assignment dataframe

### Step 4: Projected Standings
```
1  Large Farva              389    16205.8    40
2  Big Trouble              378    16108.2    32
3  Hollyhood                396    16063.1    40
4  The Milwaukee Beers      394    15708.7    38
5  Chyna Jr                 393    15630.4    31
6  Bartolo's Meatballs      394    15587.7    37
7  Titty City               365    15441.6    40
8  Cat & Kaboom             400    15315.0    40
9  Busch Banditos           364    15025.2    40
10 Vance Munson VagHawks    375    14648.7    37
11 Rick Vaughn's Wild Things 391   14544.2    39
12 The Baseball Team        346    12106.6    38
```

## Known Limitations
- Salary figures don't account for cap penalties (e.g. cut players)
- ~3 missing projections per team are true prospects with no OOPSY projection — treated as 0
- Season caps (162 G, 1500 IP) not yet modeled in optimizer — currently just optimizes best possible lineup

## Next Steps
1. **Inflation model** — model auction draft inflation based on keeper salaries, remaining free agent pool value, and team budgets. Account for positional scarcity. Eventually build retroactive validation.
2. **Streamlit UI** — deploy as web app for co-manager access
3. **Retroactive inflation analysis** — reconstruct historical draft states from transaction history

## Key Technical Notes
- Player ID systems: Ottoneu uses its own IDs (scraped as fg_id from roster page hrefs), ATC/OOPSY exports use a different FanGraphs ID. SFBB crosswalk bridges them via OTTONEUID → IDFANGRAPHS.
- Two projection files needed: hitting and pitching separately, both with PlayerId column renamed to fg_id
- PuLP solver runs silently (msg=0), solves in ~1 second per team
