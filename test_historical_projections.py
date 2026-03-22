"""
test_historical_projections.py
-------------------------------
Probes the FanGraphs API to find which projection systems and seasons
are actually accessible with your credentials.

Usage:
    python test_historical_projections.py

Paste FG_USER and FG_PASS when prompted, or hardcode them temporarily.
DELETE this file or remove credentials after running.
"""

import requests
import getpass

# ── Auth ──────────────────────────────────────────────────────────────────────
FG_LOGIN_URL = 'https://blogs.fangraphs.com/wp-login.php'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.fangraphs.com/',
}

def get_session(username, password):
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(FG_LOGIN_URL, timeout=30)
    login_data = {
        'log': username,
        'pwd': password,
        'wp-submit': 'Log In',
        'redirect_to': 'https://www.fangraphs.com',
        'rememberme': 'forever',
        'testcookie': '1',
    }
    session.post(FG_LOGIN_URL, data=login_data, timeout=30)
    print("  Login attempted.")
    return session


# ── Probe ─────────────────────────────────────────────────────────────────────
SYSTEMS = {
    'steamer':  {'bat': 'steamer', 'pit': 'steamer'},
    'zips':     {'bat': 'zips',    'pit': 'zips'},
    'atc':      {'bat': 'atc',     'pit': 'atc'},
    'oopsy':    {'bat': 'oopsy',   'pit': 'oopsy'},
}

# Auction years we care about (excluding 2015 startup draft)
TEST_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

def probe(session, proj_type, stats, season):
    """Try one API call. Returns (success, row_count, note)."""
    url = (
        f'https://www.fangraphs.com/api/projections'
        f'?type={proj_type}&stats={stats}&pos=all'
        f'&team=0&players=0&lg=all&season={season}'
    )
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            return False, 0, f'HTTP {resp.status_code}'
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return False, 0, 'empty or non-list response'
        # Check if it has a meaningful FPTS-relevant column
        sample = data[0]
        has_data = any(k in sample for k in ['wRC+', 'ERA', 'SO', 'HR', 'playerid'])
        return True, len(data), 'ok' if has_data else 'unexpected shape'
    except Exception as e:
        return False, 0, str(e)


def run_probe(session):
    print(f"\n{'System':<10} {'Year':<6} {'Bat':<20} {'Pit':<20}")
    print("-" * 60)

    results = {}
    for system, params in SYSTEMS.items():
        for year in TEST_YEARS:
            bat_ok, bat_n, bat_note = probe(session, params['bat'], 'bat', year)
            pit_ok, pit_n, pit_note = probe(session, params['pit'], 'pit', year)

            bat_str = f"✓ {bat_n} rows" if bat_ok else f"✗ ({bat_note})"
            pit_str = f"✓ {pit_n} rows" if pit_ok else f"✗ ({pit_note})"
            print(f"{system:<10} {year:<6} {bat_str:<20} {pit_str:<20}")

            results[(system, year)] = {
                'bat_ok': bat_ok, 'bat_n': bat_n,
                'pit_ok': pit_ok, 'pit_n': pit_n,
            }

    print("\n=== USABLE COMBINATIONS (both bat + pit accessible) ===")
    for (system, year), r in results.items():
        if r['bat_ok'] and r['pit_ok']:
            print(f"  {system} {year}: {r['bat_n']} hitters, {r['pit_n']} pitchers")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("FanGraphs Historical Projection Probe")
    print("======================================")
    print("Enter your FanGraphs credentials (not stored anywhere):")
    username = "matthewchoman@gmail.com"
    password = "brettbotchomanager"

    print("\nLogging in...")
    session = get_session(username, password)

    print("Probing API (this takes ~60 seconds)...\n")
    run_probe(session)
