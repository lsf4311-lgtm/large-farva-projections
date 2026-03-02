import requests
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, value, PULP_CBC_CMD
import time

# ── Configuration ────────────────────────────────────────────────────────────
LEAGUE_ID = "569"
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.dirname(__file__)

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_api_request(url, timeout=30):
    try:
        response = requests.get(url, timeout=timeout)
        return response if response.status_code == 200 else None
    except Exception as e:
        print(f"API request failed for {url}: {e}")
        return None


# ── Step 1: Scrape League Rosters ────────────────────────────────────────────
def get_league_rosters():
    """Scrape all team rosters and salaries for the full league."""
    all_players = []

    league_url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/standings"
    response = make_api_request(league_url)
    if not response:
        print("Failed to fetch league standings page")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')

    team_links = {}
    for link in soup.find_all('a', href=True):
        href = link['href']
        if f'/{LEAGUE_ID}/team/' in href:
            team_id = href.split('/team/')[-1].strip('/')
            team_name = link.get_text(strip=True)
            if team_id.isdigit() and team_name:
                team_links[team_id] = team_name

    print(f"Found {len(team_links)} teams")

    for team_id, team_name in team_links.items():
        print(f"  Scraping {team_name}...")
        roster_url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/team/{team_id}"
        response = make_api_request(roster_url)
        if not response:
            print(f"  Failed to fetch roster for {team_name}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        for table_id in ['hitters', 'pitchers']:
            table = soup.find('table', {'id': table_id})
            if not table:
                continue

            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue

                player_link = cells[0].find('a')
                if not player_link:
                    continue

                player_name = player_link.get_text(strip=True)

                href = player_link.get('href', '')
                fg_id = None
                if 'playerid=' in href:
                    fg_id = href.split('playerid=')[-1].split('&')[0]
                elif '/players/' in href:
                    fg_id = href.split('/players/')[-1].strip('/')

                position = cells[2].get_text(strip=True)
                salary_text = cells[1].get_text(strip=True).replace('$', '').replace(',', '')
                try:
                    salary = int(salary_text)
                except ValueError:
                    salary = 0

                all_players.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'player_name': player_name,
                    'fg_id': fg_id,
                    'position': position,
                    'salary': salary,
                    'player_type': table_id
                })

        time.sleep(1)

    df = pd.DataFrame(all_players)
    print(f"\nTotal players scraped: {len(df)} across {df['team_name'].nunique()} teams")
    return df


# ── Step 2: Fuzzy Match Fallback ─────────────────────────────────────────────
def fuzzy_match_players(unmatched_rosters, projection_df, threshold=90):
    """Fallback name matcher for players missing crosswalk IDs."""
    projection_names = projection_df['Name'].tolist()
    results = []

    for _, row in unmatched_rosters.iterrows():
        match = process.extractOne(
            row['player_name'],
            projection_names,
            scorer=fuzz.token_sort_ratio
        )

        if match and match[1] >= threshold:
            matched_row = projection_df[projection_df['Name'] == match[0]].iloc[0]
            results.append({
                'player_name': row['player_name'],
                'matched_name': match[0],
                'confidence': match[1],
                'fg_id': matched_row['fg_id'],
                'FPTS': matched_row['FPTS'],
                'team_name': row['team_name'],
                'needs_review': match[1] < 95
            })
        else:
            results.append({
                'player_name': row['player_name'],
                'matched_name': None,
                'confidence': match[1] if match else 0,
                'fg_id': None,
                'FPTS': 0,
                'team_name': row['team_name'],
                'needs_review': True
            })

    return pd.DataFrame(results)


# ── Step 3: Lineup Optimizer ─────────────────────────────────────────────────
def optimize_lineup(team_df):
    """Optimize lineup assignment for a single team using linear programming."""
    slots = {
        'C':    ['C'],
        '1B':   ['1B'],
        '2B':   ['2B'],
        'SS':   ['SS'],
        '3B':   ['3B'],
        'MI':   ['2B', 'SS'],
        'OF1':  ['OF'],
        'OF2':  ['OF'],
        'OF3':  ['OF'],
        'OF4':  ['OF'],
        'OF5':  ['OF'],
        'Util': ['C', '1B', '2B', 'SS', '3B', 'OF'],
        'SP1':  ['SP'],
        'SP2':  ['SP'],
        'SP3':  ['SP'],
        'SP4':  ['SP'],
        'SP5':  ['SP'],
        'RP1':  ['RP'],
        'RP2':  ['RP'],
        'RP3':  ['RP'],
        'RP4':  ['RP'],
        'RP5':  ['RP'],
    }

    def get_positions(pos_string):
        if pd.isna(pos_string):
            return []
        return [p.strip() for p in str(pos_string).split('/')]

    team_df = team_df.copy()
    team_df['pos_list'] = team_df['position'].apply(get_positions)
    team_df['FPTS'] = team_df['FPTS'].fillna(0).clip(lower=0)

    players = team_df.index.tolist()
    slot_names = list(slots.keys())

    prob = LpProblem("lineup_optimization", LpMaximize)
    x = {(p, s): LpVariable(f"x_{p}_{s}", cat=LpBinary)
         for p in players for s in slot_names}

    prob += lpSum(team_df.loc[p, 'FPTS'] * x[p, s]
                  for p in players for s in slot_names)

    for p in players:
        prob += lpSum(x[p, s] for s in slot_names) <= 1

    for s in slot_names:
        prob += lpSum(x[p, s] for p in players) <= 1

    for p in players:
        player_positions = team_df.loc[p, 'pos_list']
        for s in slot_names:
            if not any(pos in slots[s] for pos in player_positions):
                prob += x[p, s] == 0

    prob.solve(PULP_CBC_CMD(msg=0))

    total_fpts = value(prob.objective) or 0
    assigned = []
    for p in players:
        for s in slot_names:
            if value(x[p, s]) == 1:
                assigned.append({
                    'player_name': team_df.loc[p, 'player_name'],
                    'slot': s,
                    'position': team_df.loc[p, 'position'],
                    'FPTS': team_df.loc[p, 'FPTS']
                })

    return total_fpts, pd.DataFrame(assigned)


# ── Main Pipeline ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Scrape rosters
    rosters = get_league_rosters()
    rosters.to_csv(os.path.join(DATA_DIR, 'league_rosters.csv'), index=False)

    # 2. Load projections
    atc_hitting = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy hitting 2026.csv'))
    atc_pitching = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy pitching 2026.csv'))
    atc_hitting = atc_hitting.rename(columns={'PlayerId': 'fg_id'})
    atc_pitching = atc_pitching.rename(columns={'PlayerId': 'fg_id'})

    # 3. Load and clean crosswalk
    crosswalk = pd.read_csv(os.path.join(DATA_DIR, 'sfbb_crosswalk.csv'))
    crosswalk = crosswalk.drop_duplicates(subset='OTTONEUID')
    crosswalk['OTTONEUID'] = crosswalk['OTTONEUID'].fillna('').apply(
        lambda x: str(int(float(x))) if x != '' else '').str.strip()
    crosswalk['IDFANGRAPHS'] = crosswalk['IDFANGRAPHS'].astype(str).str.strip()

    # 4. Standardize IDs
    rosters['fg_id'] = rosters['fg_id'].astype(str).str.strip()
    atc_hitting['fg_id'] = atc_hitting['fg_id'].astype(str).str.strip()
    atc_pitching['fg_id'] = atc_pitching['fg_id'].astype(str).str.strip()

    # 5. Merge rosters with crosswalk, then with projections
    rosters_with_fgid = rosters.merge(
        crosswalk[['OTTONEUID', 'IDFANGRAPHS']],
        left_on='fg_id', right_on='OTTONEUID', how='left'
    )

    roster_hitters = rosters_with_fgid[rosters_with_fgid['player_type'] == 'hitters']
    roster_pitchers = rosters_with_fgid[rosters_with_fgid['player_type'] == 'pitchers']

    hitters_merged = roster_hitters.merge(
        atc_hitting[['fg_id', 'Name', 'FPTS']], left_on='IDFANGRAPHS', right_on='fg_id', how='left')
    pitchers_merged = roster_pitchers.merge(
        atc_pitching[['fg_id', 'Name', 'FPTS']], left_on='IDFANGRAPHS', right_on='fg_id', how='left')

    all_players_projected = pd.concat([hitters_merged, pitchers_merged], ignore_index=True)

    # 6. Fill missing FPTS
    all_players_projected['FPTS'] = all_players_projected['FPTS'].fillna(0)

    # 7. Fix Ohtani two-way contribution
    ohtani_hitting_fpts = atc_hitting[
        atc_hitting['Name'].str.contains('Ohtani', case=False)]['FPTS'].values[0]
    ohtani_mask = all_players_projected['player_name'].str.contains('Ohtani', case=False)
    all_players_projected.loc[ohtani_mask, 'FPTS'] += ohtani_hitting_fpts

    # 8. Fuzzy match fallback for remaining missing projections
    missing_hitters = all_players_projected[
        (all_players_projected['FPTS'] == 0) & (all_players_projected['player_type'] == 'hitters')]
    missing_pitchers = all_players_projected[
        (all_players_projected['FPTS'] == 0) & (all_players_projected['player_type'] == 'pitchers')]

    fuzzy_hitters = fuzzy_match_players(missing_hitters, atc_hitting)
    fuzzy_pitchers = fuzzy_match_players(missing_pitchers, atc_pitching)

    for _, match_row in pd.concat([fuzzy_hitters, fuzzy_pitchers]).iterrows():
        if match_row['FPTS'] > 0:
            mask = ((all_players_projected['player_name'] == match_row['player_name']) &
                    (all_players_projected['team_name'] == match_row['team_name']))
            all_players_projected.loc[mask, 'FPTS'] = match_row['FPTS']

    # 9. Save full player projections
    all_players_projected.to_csv(os.path.join(DATA_DIR, 'players_with_projections.csv'), index=False)

    # 10. Optimize lineups and build standings
    print("\nOptimizing lineups...")
    team_results = []

    for team_name, team_df in all_players_projected.groupby('team_name'):
        total_fpts, lineup = optimize_lineup(team_df)
        team_results.append({
            'team_name': team_name,
            'total_salary': team_df['salary'].sum(),
            'projected_fpts': total_fpts,
            'num_players': len(team_df),
        })
        print(f"  {team_name}: {total_fpts:.1f} pts")

    team_projections = pd.DataFrame(team_results)
    team_projections = team_projections.sort_values(
        'projected_fpts', ascending=False).reset_index(drop=True)
    team_projections.index += 1

    print("\n=== 2026 Projected Standings (Optimized Lineups) ===")
    print(team_projections.to_string())

    team_projections.to_csv(os.path.join(DATA_DIR, 'team_projections.csv'), index=True)