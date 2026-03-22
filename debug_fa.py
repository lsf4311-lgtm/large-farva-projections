import pandas as pd
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))
from league_analysis_final import get_league_rosters, DATA_DIR
from league_analysis_final import fuzzy_match_players

def normalize_name(n):
    n = str(n).lower()
    n = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', n)
    n = re.sub(r'[^a-z\s]', '', n)
    return ' '.join(n.split())

# Load data from disk (no scraping needed)
rosters = pd.read_csv(os.path.join(DATA_DIR, 'league_rosters.csv'))
rosters['fg_id'] = rosters['fg_id'].astype(str).str.strip()
atc_hitting = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy hitting 2026.csv'))

crosswalk = pd.read_csv(os.path.join(DATA_DIR, 'sfbb_crosswalk.csv'))
crosswalk = crosswalk.drop_duplicates(subset='OTTONEUID')
crosswalk['OTTONEUID'] = crosswalk['OTTONEUID'].fillna('').apply(
    lambda x: str(int(float(x))) if x != '' else '').str.strip()
crosswalk['IDFANGRAPHS'] = crosswalk['IDFANGRAPHS'].astype(str).str.strip()

if 'PlayerId' in atc_hitting.columns:
    atc_hitting = atc_hitting.rename(columns={'PlayerId': 'fg_id'})
atc_hitting['fg_id'] = atc_hitting['fg_id'].astype(str).str.strip()

rosters_with_fgid = rosters.merge(
    crosswalk[['OTTONEUID', 'IDFANGRAPHS']],
    left_on='fg_id', right_on='OTTONEUID', how='left'
)

rostered_fg_ids = set(rosters_with_fgid['IDFANGRAPHS'].dropna().tolist())
rostered_names = set(rosters['player_name'].apply(normalize_name).tolist())

# What does the ID filter alone catch?
id_filtered = atc_hitting[~atc_hitting['fg_id'].isin(rostered_fg_ids)]

# What does the name filter catch on top of that?
name_filtered = id_filtered[id_filtered['Name'].apply(normalize_name).isin(rostered_names)]

print(f"\nPlayers ID filter missed that name filter catches ({len(name_filtered)}):")
print(name_filtered[['Name', 'fg_id', 'FPTS']].to_string())

# Also check how many NaN crosswalk IDs we have
nan_ids = rosters_with_fgid[rosters_with_fgid['IDFANGRAPHS'].isna()]
print(f"\nRostered players with no crosswalk IDFANGRAPHS ({len(nan_ids)}):")
print(nan_ids[['player_name', 'fg_id']].to_string())

# Check Wrobleski specifically
print("\n--- Wrobleski debug ---")
w_roster = rosters_with_fgid[rosters_with_fgid['player_name'].str.contains('Wrobleski', case=False)]
print("In rosters_with_fgid:")
print(w_roster[['player_name', 'fg_id', 'IDFANGRAPHS']].to_string())

w_proj = atc_hitting[atc_hitting['Name'].str.contains('Wrobleski', case=False)]
print("\nIn atc_hitting projections:")
print(w_proj[['Name', 'fg_id', 'FPTS']].to_string())

w_pitch = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy pitching 2026.csv'))
if 'PlayerId' in w_pitch.columns:
    w_pitch = w_pitch.rename(columns={'PlayerId': 'fg_id'})
w_pitch['fg_id'] = w_pitch['fg_id'].astype(str).str.strip()
w_proj2 = w_pitch[w_pitch['Name'].str.contains('Wrobleski', case=False)]
print("\nIn atc_pitching projections:")
print(w_proj2[['Name', 'fg_id', 'FPTS']].to_string())

print("\n--- Wrobleski roster check ---")
# Check raw rosters before crosswalk merge
w_raw = rosters[rosters['player_name'].str.contains('Wrobleski', case=False)]
print("In raw rosters:")
print(w_raw[['player_name', 'fg_id', 'team_name']].to_string())

# Check crosswalk
w_cross = crosswalk[crosswalk['IDFANGRAPHS'] == '31204']
print("\nCrosswalk entry for fg_id 31204:")
print(w_cross[['OTTONEUID', 'IDFANGRAPHS']].to_string())

print("\n--- Check which team has Wrobleski on Ottoneu ---")
import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

# Search all 12 team pages for Wrobleski
from league_analysis_final import HEADERS
resp = requests.get(f"https://ottoneu.fangraphs.com/569/standings", headers=HEADERS)
soup = BeautifulSoup(resp.text, 'html.parser')
team_links = {}
for link in soup.find_all('a', href=True):
    href = link['href']
    if '/569/team/' in href:
        team_id = href.split('/team/')[-1].strip('/')
        team_name = link.get_text(strip=True)
        if team_id.isdigit() and team_name:
            team_links[team_id] = team_name

for team_id, team_name in team_links.items():
    resp = requests.get(f"https://ottoneu.fangraphs.com/569/team/{team_id}", headers=HEADERS)
    if 'Wrobleski' in resp.text:
        print(f"Found Wrobleski on: {team_name} (team_id {team_id})")
        break
else:
    print("Wrobleski not found on any roster page")

    # Check if Wrobleski is in pitching projections and would survive FA filter
rostered_fg_ids_pitch = set(rosters_with_fgid['IDFANGRAPHS'].dropna().tolist())
rostered_names_pitch = set(rosters['player_name'].str.lower().tolist())

w_in_fa = atc_pitching[
    (~atc_pitching['fg_id'].isin(rostered_fg_ids_pitch)) &
    (~atc_pitching['Name'].str.lower().isin(rostered_names_pitch))
]
print("\nWrobleski in FA pitching pool:")
print(w_in_fa[w_in_fa['Name'].str.contains('Wrobleski', case=False)][['Name', 'fg_id', 'FPTS']])