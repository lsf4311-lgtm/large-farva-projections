import requests
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, value, PULP_CBC_CMD
import time
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
LEAGUE_ID = "569"
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.dirname(__file__)

# ── Helpers ──────────────────────────────────────────────────────────────────
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.fangraphs.com/',
    'Origin': 'https://www.fangraphs.com',
    'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
}

def make_api_request(url, timeout=30):
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} for {url[:80]}...")
            return None
        return response
    except Exception as e:
        print(f"API request failed for {url}: {e}")
        return None


# ── FanGraphs Auth & Projection Fetcher ──────────────────────────────────────
FG_LOGIN_URL = 'https://blogs.fangraphs.com/wp-login.php'
FG_PROJ_URL = 'https://www.fangraphs.com/api/projections?type={type}&stats={stats}&pos=all&team=0&players=0&lg=all'

PROJECTION_ENDPOINTS = {
    'OOPSY': {
        'hitting': {'type': 'oopsy', 'stats': 'bat'},
        'pitching': {'type': 'oopsy', 'stats': 'pit'},
    },
    'ATC': {
        'hitting': {'type': 'atc', 'stats': 'bat'},
        'pitching': {'type': 'atc', 'stats': 'pit'},
    },
    'OOPSY DC': {
        'hitting': {'type': 'oopsydc', 'stats': 'bat'},
        'pitching': {'type': 'oopsydc', 'stats': 'pit'},
    },
    'ATC DC': {
        'hitting': {'type': 'atcdc', 'stats': 'bat'},
        'pitching': {'type': 'atcdc', 'stats': 'pit'},
    },
    'THE BAT X': {
        'hitting': {'type': 'thebatx', 'stats': 'bat'},
        'pitching': {'type': 'thebatx', 'stats': 'pit'},
    },
    'THE BAT X DC': {
        'hitting': {'type': 'thebatxdc', 'stats': 'bat'},
        'pitching': {'type': 'thebatxdc', 'stats': 'pit'},
    },
}

def get_fangraphs_session(username, password):
    """Log into FanGraphs via WordPress and return an authenticated session."""
    session = requests.Session()
    session.headers.update(HEADERS)

    # First GET the login page to grab the test cookie
    session.get(FG_LOGIN_URL, timeout=30)

    login_data = {
        'log': username,
        'pwd': password,
        'wp-submit': 'Sign In',
        'rememberme': 'forever',
        'redirect_to': 'https://www.fangraphs.com/',
        'testcookie': '1',
    }

    resp = session.post(
        FG_LOGIN_URL,
        data=login_data,
        headers={
            'Referer': FG_LOGIN_URL,
            'Origin': 'https://blogs.fangraphs.com',
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        timeout=30,
        allow_redirects=True
    )

    # Check we got a logged-in cookie (either wordpress_logged_in or wordpress_sec)
    logged_in = any(
        'wordpress_logged_in' in c or 'wordpress_sec' in c 
        for c in session.cookies.keys()
    )
    if not logged_in:
        raise Exception(f"FanGraphs login failed (status {resp.status_code})")

    print("  FanGraphs login successful")
    return session


def fetch_projections(projection_system='OOPSY', username=None, password=None):
    """Fetch projection data from FanGraphs API for the given system.
    Returns (hitting_df, pitching_df) or raises on failure.
    """
    session = get_fangraphs_session(username, password)
    endpoints = PROJECTION_ENDPOINTS[projection_system]
    results = {}

    for slot, params in endpoints.items():
        url = FG_PROJ_URL.format(**params)
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Projection fetch failed for {projection_system} {slot}: HTTP {resp.status_code}")

        data = resp.json()
        # API returns a list directly or {"data": [...]}
        rows = data if isinstance(data, list) else data.get('data', [])
        if not rows:
            raise Exception(f"No projection data returned for {projection_system} {slot}")

        df = pd.DataFrame(rows)
        df = df.rename(columns={'playerid': 'fg_id', 'PlayerName': 'Name'})
        # Ensure PlayerId rename also covered
        if 'PlayerId' in df.columns:
            df = df.rename(columns={'PlayerId': 'fg_id'})
        df['fg_id'] = df['fg_id'].astype(str).str.strip()

        # Ensure FPTS column exists
        if 'FPTS' not in df.columns:
            fpts_candidates = [c for c in df.columns if 'fpts' in c.lower() or 'pts' in c.lower()]
            if fpts_candidates:
                df = df.rename(columns={fpts_candidates[0]: 'FPTS'})
            else:
                raise Exception(f"No FPTS column found in {projection_system} {slot} projection data. Columns: {df.columns.tolist()}")

        results[slot] = df
        print(f"  Fetched {projection_system} {slot} projections: {len(df)} players")
        time.sleep(1)

    return results['hitting'], results['pitching']


# ── FA Position Scraper ───────────────────────────────────────────────────────
# Hitter positions: scraped from FanGraphs JSON API (accurate, no auth needed)
# Pitcher positions: scraped from FanGraphs JSON API using stats=sta/rel
HITTER_POSITIONS = ['c', '1b', '2b', 'ss', '3b', 'of']
PITCHER_POSITIONS = [('sta', 'SP'), ('rel', 'RP')]
FANGRAPHS_API_URL = (
    "https://www.fangraphs.com/api/leaders/major-league/data"
    "?pos={pos}&stats={stats}&lg=all&qual=0&season=2025&season1=2025"
    "&month=0&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0"
    "&type=8&fl={league_id}&ft=-1"
)

def get_fa_positions():
    """Build an accurate fg_id -> position map for FA hitters and pitchers
    using the FanGraphs JSON API.
    Hitters: one request per position (C, 1B, 2B, SS, 3B, OF)
    Pitchers: stats=sta -> SP, stats=rel -> RP (combined if both)
    """
    fa_pos_map = {}  # fg_id -> position string

    # ── Hitters ──────────────────────────────────────────────────────────────
    for pos in HITTER_POSITIONS:
        url = FANGRAPHS_API_URL.format(pos=pos, stats='bat', league_id=LEAGUE_ID)
        response = make_api_request(url)
        if not response:
            print(f"  FA position fetch failed for {pos.upper()}")
            time.sleep(2)
            continue

        try:
            data = response.json()
        except Exception as e:
            print(f"  FA position JSON parse failed for {pos.upper()}: {e}")
            time.sleep(2)
            continue

        rows = data.get('data', [])
        if not rows:
            print(f"  No data rows for {pos.upper()}")
            time.sleep(2)
            continue

        pos_label = pos.upper()
        count = 0
        for row in rows:
            fg_id = str(row.get('playerid', '')).strip()
            if not fg_id or fg_id == 'nan':
                continue
            if fg_id in fa_pos_map:
                if pos_label not in fa_pos_map[fg_id]:
                    fa_pos_map[fg_id] = fa_pos_map[fg_id] + '/' + pos_label
            else:
                fa_pos_map[fg_id] = pos_label
            count += 1

        print(f"  FA positions fetched: {pos_label} ({count} players)")
        time.sleep(1)

    # ── Pitchers ─────────────────────────────────────────────────────────────
    for stats_param, pos_label in PITCHER_POSITIONS:
        url = FANGRAPHS_API_URL.format(pos='all', stats=stats_param, league_id=LEAGUE_ID)
        response = make_api_request(url)
        if not response:
            print(f"  FA position fetch failed for {pos_label}")
            time.sleep(2)
            continue

        try:
            data = response.json()
        except Exception as e:
            print(f"  FA position JSON parse failed for {pos_label}: {e}")
            time.sleep(2)
            continue

        rows = data.get('data', [])
        if not rows:
            print(f"  No data rows for {pos_label}")
            time.sleep(2)
            continue

        count = 0
        for row in rows:
            fg_id = str(row.get('playerid', '')).strip()
            if not fg_id or fg_id == 'nan':
                continue
            if fg_id in fa_pos_map:
                if pos_label not in fa_pos_map[fg_id]:
                    fa_pos_map[fg_id] = fa_pos_map[fg_id] + '/' + pos_label
            else:
                fa_pos_map[fg_id] = pos_label
            count += 1

        print(f"  FA positions fetched: {pos_label} ({count} players)")
        time.sleep(1)

    print(f"  Total FA position entries: {len(fa_pos_map)}")
    return fa_pos_map


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

        time.sleep(3)

    df = pd.DataFrame(all_players)
    print(f"\nTotal players scraped: {len(df)} across {df['team_name'].nunique()} teams")

    # Save timestamp of successful scrape
    timestamp_path = os.path.join(DATA_DIR, 'roster_scrape_timestamp.txt')
    with open(timestamp_path, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M'))
    
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


def optimize_lineup(team_df):
    """Optimize lineup assignment using two-phase approach:
    Phase 1: Fill all constrained positions optimally (excluding Util)
    Phase 2: Best remaining eligible player goes to Util
    """
    slots_phase1 = {
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

    util_eligible = ['C', '1B', '2B', 'SS', '3B', 'OF']

    def get_positions(pos_string):
        if pd.isna(pos_string):
            return []
        return [p.strip() for p in str(pos_string).split('/')]

    team_df = team_df.copy()
    team_df['pos_list'] = team_df['position'].apply(get_positions)
    team_df['FPTS'] = team_df['FPTS'].fillna(0).clip(lower=0)

    players = team_df.index.tolist()
    slot_names = list(slots_phase1.keys())

    # ── Phase 1: Optimize all positions except Util ───────────────────────────
    prob = LpProblem("lineup_phase1", LpMaximize)
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
            if not any(pos in slots_phase1[s] for pos in player_positions):
                prob += x[p, s] == 0

    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract Phase 1 assignments
    assigned = []
    assigned_players = set()

    for p in players:
        for s in slot_names:
            if value(x[p, s]) == 1:
                assigned.append({
                    'player_name': team_df.loc[p, 'player_name'],
                    'slot': s,
                    'position': team_df.loc[p, 'position'],
                    'FPTS': team_df.loc[p, 'FPTS']
                })
                assigned_players.add(p)
    
    # Sort SP and RP slots by FPTS descending for clean display
    pitcher_slots = [a for a in assigned if a['slot'].startswith('SP') or a['slot'].startswith('RP')]
    non_pitcher_slots = [a for a in assigned if not a['slot'].startswith('SP') and not a['slot'].startswith('RP')]

    sp_assigned = sorted([a for a in pitcher_slots if a['slot'].startswith('SP')], 
                        key=lambda x: x['FPTS'], reverse=True)
    rp_assigned = sorted([a for a in pitcher_slots if a['slot'].startswith('RP')], 
                        key=lambda x: x['FPTS'], reverse=True)

    for i, a in enumerate(sp_assigned):
        a['slot'] = f'SP{i+1}'
    for i, a in enumerate(rp_assigned):
        a['slot'] = f'RP{i+1}'

    assigned = non_pitcher_slots + sp_assigned + rp_assigned

    # ── Phase 2: Best remaining Util-eligible player fills Util ──────────────
    unassigned = team_df[~team_df.index.isin(assigned_players)].copy()
    util_candidates = unassigned[
        unassigned['pos_list'].apply(
            lambda positions: any(pos in util_eligible for pos in positions)
        )
    ].sort_values('FPTS', ascending=False)

    if len(util_candidates) > 0:
        util_player = util_candidates.iloc[0]
        assigned.append({
            'player_name': util_player['player_name'],
            'slot': 'Util',
            'position': util_player['position'],
            'FPTS': util_player['FPTS']
        })

    total_fpts = sum(a['FPTS'] for a in assigned)
    return total_fpts, pd.DataFrame(assigned)


# ── Pitching Report ───────────────────────────────────────────────────────────
from urllib.parse import quote
from io import StringIO

WOBA_WEIGHTS = {'bb': 0.69, 'hbp': 0.72, '1b': 0.89, '2b': 1.27, '3b': 1.62, 'hr': 2.15}
MLB_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE',
    'COL', 'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL',
    'MIN', 'NYM', 'NYY', 'ATH', 'PHI', 'PIT', 'SD', 'SEA',
    'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH'
]
TEAM_MAPPING = {'KCR': 'KC', 'SDP': 'SD', 'SFG': 'SF', 'TBR': 'TB', 'OAK': 'ATH'}
HIT_EVENTS = ['single', 'double', 'triple', 'home_run']
OUT_EVENTS = ['strikeout', 'field_out', 'force_out', 'grounded_into_double_play',
              'fielders_choice_out', 'sac_fly', 'sac_bunt']
NON_AB_EVENTS = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'intent_walk']


def _mlb_request(url, params=None, timeout=30):
    """Lightweight MLB/Savant API request — no FanGraphs headers needed."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        return resp if resp.status_code == 200 else None
    except Exception as e:
        print(f"MLB API request failed: {e}")
        return None


def _get_player_id(name):
    """Look up MLB player ID by name."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={quote(name.strip())}"
        resp = _mlb_request(url)
        if not resp:
            return None
        people = resp.json().get('people', [])
        for p in people:
            if p.get('active'):
                return p.get('id')
        return people[0].get('id') if people else None
    except Exception as e:
        print(f"Player ID lookup failed for {name}: {e}")
        return None


def _get_player_team(player_id):
    """Get player's current team ID and name."""
    try:
        resp = _mlb_request(f"https://statsapi.mlb.com/api/v1/people/{player_id}")
        if resp:
            people = resp.json().get('people', [])
            if people:
                team = people[0].get('currentTeam', {})
                if team:
                    return team.get('id'), team.get('name', 'Unknown')
    except Exception as e:
        print(f"Team lookup failed for player {player_id}: {e}")
    return None, 'Unknown'


def _get_team_abbr(team_info):
    """Extract team abbreviation from MLB API team dict."""
    abbr = team_info.get('abbreviation') or team_info.get('teamCode') or team_info.get('fileCode', '').upper()
    if not abbr:
        tid = team_info.get('id')
        if tid:
            try:
                resp = _mlb_request(f"https://statsapi.mlb.com/api/v1/teams/{tid}")
                if resp:
                    data = resp.json()
                    teams = data.get('teams', [data])
                    abbr = teams[0].get('abbreviation', 'UNK')
            except:
                pass
    return abbr or 'UNK'


def _get_savant_data(params, timeout=45):
    """Fetch pitch-level data from Baseball Savant."""
    try:
        url = "https://baseballsavant.mlb.com/statcast_search/csv"
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200 and resp.text.strip():
            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            return df if not df.empty else None
    except Exception as e:
        print(f"Savant request failed: {e}")
    return None


def _calc_basic_stats(df):
    """Calculate counting stats from a Savant pitch-level dataframe."""
    pa_df = df[df['events'].notna() & (df['events'] != '')]
    hits = pa_df[pa_df['events'].isin(HIT_EVENTS)]
    strikeouts = len(pa_df[pa_df['events'] == 'strikeout'])
    walks = len(pa_df[pa_df['events'] == 'walk'])
    hbp = len(pa_df[pa_df['events'] == 'hit_by_pitch'])
    singles = len(hits[hits['events'] == 'single'])
    doubles = len(hits[hits['events'] == 'double'])
    triples = len(hits[hits['events'] == 'triple'])
    home_runs = len(hits[hits['events'] == 'home_run'])
    total_hits = len(hits)
    non_ab = len(pa_df[pa_df['events'].isin(NON_AB_EVENTS)])
    at_bats = len(pa_df) - non_ab
    batters_faced = len(pa_df)
    outs = len(pa_df[pa_df['events'].isin(OUT_EVENTS)])
    innings = outs / 3
    return {
        'innings_pitched': innings, 'strikeouts': strikeouts, 'walks': walks,
        'hbp': hbp, 'singles': singles, 'doubles': doubles, 'triples': triples,
        'home_runs': home_runs, 'hits': total_hits, 'at_bats': at_bats,
        'batters_faced': batters_faced, 'outs': outs,
    }


def _calc_woba(singles, doubles, triples, hr, bb, hbp, pa):
    """Calculate wOBA from counting stats."""
    if pa <= 0:
        return 0.320
    num = (WOBA_WEIGHTS['bb'] * bb + WOBA_WEIGHTS['hbp'] * hbp +
           WOBA_WEIGHTS['1b'] * singles + WOBA_WEIGHTS['2b'] * doubles +
           WOBA_WEIGHTS['3b'] * triples + WOBA_WEIGHTS['hr'] * hr)
    return round(num / pa, 3)


def get_pitcher_schedule(pitcher_name):
    """Get rolling 7-day schedule for a pitcher's team."""
    try:
        player_id = _get_player_id(pitcher_name)
        if not player_id:
            return None
        team_id, team_name = _get_player_team(player_id)
        if not team_id:
            return None

        today = datetime.now()
        end = today + timedelta(days=6)
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
               f"&teamId={team_id}&startDate={today.strftime('%Y-%m-%d')}"
               f"&endDate={end.strftime('%Y-%m-%d')}")
        resp = _mlb_request(url)
        if not resp:
            return None

        info = {'games_this_week': 0, 'opponents': [], 'dates': [], 'home_away': [], 'team_name': team_name}
        for date_info in resp.json().get('dates', []):
            for game in date_info.get('games', []):
                info['games_this_week'] += 1
                info['dates'].append(date_info['date'])
                home = game['teams']['home']['team']
                away = game['teams']['away']['team']
                if home.get('id') == team_id:
                    info['opponents'].append(_get_team_abbr(away))
                    info['home_away'].append('Home')
                else:
                    info['opponents'].append(_get_team_abbr(home))
                    info['home_away'].append('Away')
        return info
    except Exception as e:
        print(f"Schedule fetch failed for {pitcher_name}: {e}")
        return None


def get_pitcher_stats(pitcher_name):
    """Get season stats and home/away splits from Baseball Savant."""
    try:
        player_id = _get_player_id(pitcher_name)
        if not player_id:
            return {}
        year = datetime.now().year
        params = {
            'all': 'true', 'hfSea': f'{year}|', 'hfGT': 'R|',
            'player_type': 'pitcher', 'pitchers_lookup[]': str(player_id),
            'game_date_gt': f'{year}-03-01', 'game_date_lt': f'{year}-11-30',
            'type': 'details'
        }
        df = _get_savant_data(params)
        if df is None:
            return {}

        pa = df[df['events'].notna() & (df['events'] != '')]
        if pa.empty:
            return {}

        s = _calc_basic_stats(pa)
        if s['innings_pitched'] <= 0 or s['batters_faced'] <= 0:
            return {}

        stats = {
            'IP': round(s['innings_pitched'], 1),
            'K_percent': round(s['strikeouts'] / s['batters_faced'] * 100, 1),
            'BB_percent': round(s['walks'] / s['batters_faced'] * 100, 1),
            'wOBA_against': _calc_woba(s['singles'], s['doubles'], s['triples'],
                                       s['home_runs'], s['walks'], s['hbp'], s['batters_faced']),
            'batters_faced': s['batters_faced'],
        }

        contact = df[df['launch_speed'].notna()]
        if not contact.empty:
            stats['avg_exit_velo'] = round(contact['launch_speed'].mean(), 1)
            stats['hard_hit_percent'] = round(len(contact[contact['launch_speed'] >= 95]) / len(contact) * 100, 1)

        # Home/away splits
        if 'inning_topbot' in df.columns:
            for split_name, top_bot in [('home_splits', 'Top'), ('away_splits', 'Bot')]:
                split_pa = pa[pa['inning_topbot'] == top_bot]
                if not split_pa.empty:
                    ss = _calc_basic_stats(split_pa)
                    if ss['innings_pitched'] > 0 and ss['batters_faced'] > 0:
                        stats[split_name] = {
                            'K_percent': round(ss['strikeouts'] / ss['batters_faced'] * 100, 1),
                            'BB_percent': round(ss['walks'] / ss['batters_faced'] * 100, 1),
                            'wOBA_against': _calc_woba(ss['singles'], ss['doubles'], ss['triples'],
                                                       ss['home_runs'], ss['walks'], ss['hbp'], ss['batters_faced']),
                            'innings': round(ss['innings_pitched'], 1),
                        }
        return stats
    except Exception as e:
        print(f"Pitcher stats failed for {pitcher_name}: {e}")
        return {}


def get_pitcher_rotation_info(pitcher_name):
    """Predict next start based on recent game log."""
    try:
        player_id = _get_player_id(pitcher_name)
        if not player_id:
            return {}
        year = datetime.now().year
        url = (f"https://statsapi.mlb.com/api/v1/people?personIds={player_id}"
               f"&hydrate=stats(group=[pitching],type=[gameLog],season={year})")
        resp = _mlb_request(url)
        if not resp:
            return {}

        people = resp.json().get('people', [])
        if not people:
            return {}

        starts = []
        for stat_group in people[0].get('stats', []):
            if stat_group.get('type', {}).get('displayName') == 'gameLog':
                for game in stat_group.get('splits', [])[-10:]:
                    gs = game.get('stat', {}).get('gamesStarted', 0)
                    if gs and int(gs) > 0:
                        date_str = game.get('date')
                        if date_str:
                            try:
                                starts.append(datetime.strptime(date_str, '%Y-%m-%d'))
                            except:
                                pass

        if len(starts) < 2:
            return {'is_starter': len(starts) > 0, 'recent_starts_count': len(starts)}

        starts.sort()
        gaps = [(starts[i+1] - starts[i]).days for i in range(len(starts)-1)]
        avg_gap = round(sum(gaps) / len(gaps), 1)
        last_start = starts[-1]
        next_predicted = last_start + timedelta(days=round(avg_gap))

        return {
            'is_starter': True,
            'last_start': last_start,
            'avg_days_between_starts': avg_gap,
            'next_predicted_start': next_predicted,
            'recent_starts_count': len(starts),
        }
    except Exception as e:
        print(f"Rotation info failed for {pitcher_name}: {e}")
        return {}


def get_all_team_stats():
    """Fetch home/away batting stats for all 30 MLB teams from Savant.
    This is slow (~2 min) — cache weekly.
    """
    year = datetime.now().year
    all_stats = {}
    for i, team in enumerate(MLB_TEAMS, 1):
        savant_team = TEAM_MAPPING.get(team, team)
        team_stats = {}
        print(f"  Team stats: {team} ({i}/{len(MLB_TEAMS)})...")
        for location, home_road in [('home', 'Home'), ('away', 'Road')]:
            params = {
                'all': 'true', 'hfSea': f'{year}|', 'hfGT': 'R|',
                'player_type': 'batter', 'team': savant_team,
                'home_road': home_road,
                'game_date_gt': f'{year}-03-01', 'game_date_lt': f'{year}-11-30',
                'min_pitches': '0', 'type': 'details'
            }
            df = _get_savant_data(params, timeout=45)
            if df is not None:
                pa = df[df['events'].notna() & (df['events'] != '')]
                if len(pa) > 50:
                    s = _calc_basic_stats(pa)
                    if s['at_bats'] > 0 and s['batters_faced'] > 0:
                        tb = s['singles'] + 2*s['doubles'] + 3*s['triples'] + 4*s['home_runs']
                        obp = (s['hits'] + s['walks'] + s['hbp']) / s['batters_faced']
                        slg = tb / s['at_bats']
                        team_stats[f'{location}_AVG'] = round(s['hits'] / s['at_bats'], 3)
                        team_stats[f'{location}_OPS'] = round(obp + slg, 3)
                        team_stats[f'{location}_wOBA'] = _calc_woba(
                            s['singles'], s['doubles'], s['triples'],
                            s['home_runs'], s['walks'], s['hbp'], s['batters_faced'])
                        team_stats[f'{location}_K_percent'] = round(s['strikeouts'] / s['batters_faced'] * 100, 1)
                        team_stats[f'{location}_HR_rate'] = round(s['home_runs'] / s['batters_faced'] * 100, 2)
            time.sleep(1)

        has_home = any(k.startswith('home_') for k in team_stats)
        has_away = any(k.startswith('away_') for k in team_stats)
        if has_home and has_away:
            all_stats[team] = team_stats
        time.sleep(1)

    print(f"  Team stats complete: {len(all_stats)}/30 teams")
    return all_stats


def get_matchup_grade(pitcher_splits, opponent_stats, location):
    """Grade a matchup A-F based on pitcher splits and opponent stats."""
    score = 0
    k_rate = pitcher_splits.get('K_percent', 20)
    bb_rate = pitcher_splits.get('BB_percent', 8)
    woba = pitcher_splits.get('wOBA_against', 0.320)
    opp_ops = opponent_stats.get(f'{location}_OPS', 0.720)

    if k_rate > 28: score += 2
    elif k_rate > 23: score += 1
    elif k_rate < 16: score -= 1

    if bb_rate < 6: score += 1
    elif bb_rate > 10: score -= 1

    if woba < 0.290: score += 2
    elif woba < 0.310: score += 1
    elif woba > 0.350: score -= 1
    elif woba > 0.370: score -= 2

    if opp_ops < 0.680: score += 2
    elif opp_ops < 0.720: score += 1
    elif opp_ops > 0.800: score -= 2
    elif opp_ops > 0.760: score -= 1

    if score >= 4: return 'A'
    elif score >= 2: return 'B'
    elif score >= 0: return 'C'
    elif score >= -2: return 'D'
    else: return 'F'


def get_team_rankings(target_team, location, all_team_stats):
    """Rank target team among all 30 for key batting stats."""
    stats_to_rank = ['OPS', 'wOBA', 'K_percent', 'HR_rate']
    rankings = {}
    for stat in stats_to_rank:
        key = f'{location}_{stat}'
        values = [(t, s[key]) for t, s in all_team_stats.items() if key in s]
        target = all_team_stats.get(target_team, {}).get(key)
        if len(values) >= 20 and target is not None:
            reverse = stat != 'K_percent'
            values.sort(key=lambda x: x[1], reverse=reverse)
            rank = next((i+1 for i, (t, _) in enumerate(values) if t == target_team), None)
            if rank:
                rankings[stat] = {'rank': rank, 'total': len(values), 'value': target}
    return rankings


def get_pitching_report(pitcher_names, all_team_stats):
    """Build the full pitching report for a list of pitcher names.
    Returns a list of dicts, one per pitcher.
    """
    report = []
    for name in pitcher_names:
        print(f"  Building report for {name}...")
        schedule = get_pitcher_schedule(name)
        stats = get_pitcher_stats(name)
        rotation = get_pitcher_rotation_info(name)

        matchups = []
        if schedule and schedule.get('opponents'):
            for i, opp in enumerate(schedule['opponents']):
                home_away = schedule['home_away'][i] if i < len(schedule['home_away']) else 'Home'
                opp_location = 'away' if home_away == 'Home' else 'home'
                splits = stats.get(f'{home_away.lower()}_splits', stats)
                grade = get_matchup_grade(splits, all_team_stats.get(opp, {}), opp_location)
                rankings = get_team_rankings(opp, opp_location, all_team_stats) if all_team_stats else {}
                matchups.append({
                    'opponent': opp,
                    'home_away': home_away,
                    'date': schedule['dates'][i] if i < len(schedule['dates']) else 'TBD',
                    'grade': grade,
                    'rankings': rankings,
                    'opp_stats': {k.replace(f'{opp_location}_', ''): v
                                  for k, v in all_team_stats.get(opp, {}).items()
                                  if k.startswith(opp_location)},
                })

        report.append({
            'name': name,
            'schedule': schedule,
            'stats': stats,
            'rotation': rotation,
            'matchups': matchups,
        })
        time.sleep(0.5)

    return report


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