"""
fetch_historical_projections.py
--------------------------------
Fetches Steamer historical projections (2016-2025) from FanGraphs member API.
Saves one CSV per year: steamer_{year}_bat.csv, steamer_{year}_pit.csv
Also saves a combined all-years file: historical_projections.csv

Uses the same WordPress auth pattern as league_analysis_final.py.

Runtime: ~2-3 minutes (20 API calls + processing)

Usage:
    python fetch_historical_projections.py
"""

import requests
import pandas as pd
import os
import getpass
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
LEAGUE_ID = "569"
FG_LOGIN_URL = 'https://blogs.fangraphs.com/wp-login.php'

# Years to fetch — 2026 handled by existing OOPSY pipeline
HISTORICAL_YEARS = list(range(2016, 2027))  # 2016–2026 inclusive

# Columns to keep — everything else is noise for our purposes
BAT_COLS  = ['playerid', 'PlayerName', 'minpos', 'G', 'FPTS', 'Team']
PIT_COLS  = ['playerid', 'PlayerName', 'minpos', 'G', 'GS', 'IP', 'SV', 'HLD', 'FPTS', 'Team']

# Output directory — same data\ folder as transaction_log.csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = SCRIPT_DIR

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

# ── Auth ──────────────────────────────────────────────────────────────────────
def get_session(username, password):
    """Log into FanGraphs via WordPress and return authenticated session."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(FG_LOGIN_URL, timeout=30)
    session.post(FG_LOGIN_URL, data={
        'log': username,
        'pwd': password,
        'wp-submit': 'Log In',
        'redirect_to': 'https://www.fangraphs.com',
        'rememberme': 'forever',
        'testcookie': '1',
    }, timeout=30)

    # Verify login succeeded
    auth_ok = any(
        'wordpress_sec' in c or 'wordpress_logged_in' in c
        for c in session.cookies.keys()
    )
    if not auth_ok:
        raise Exception("FanGraphs login failed — check credentials")
    print("  ✓ FanGraphs login successful")
    return session


# ── Fetcher ───────────────────────────────────────────────────────────────────
def fetch_projections(session, year, stats):
    """
    Fetch one year/stats combo from the member API.
    stats: 'bat' or 'pit'
    Returns a DataFrame with just the columns we need.
    """
    # Current year uses /api/projections, historical uses /api/projections/member
    # with year suffix in the type param
    if year == datetime.now().year:
        url = (
            f'https://www.fangraphs.com/api/projections'
            f'?type=steamer&stats={stats}&pos=all&team=0&players=0&lg=all'
        )
    else:
        url = (
            f'https://www.fangraphs.com/api/projections/member'
            f'?type=steamer_{year}&stats={stats}&pos=all&team=0&players=0&lg=all'
        )

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {year} {stats}")
            return pd.DataFrame()

        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            print(f"    Empty response for {year} {stats}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Keep only relevant columns (ignore missing ones gracefully)
        keep = BAT_COLS if stats == 'bat' else PIT_COLS
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # Add metadata
        df['season'] = year
        df['stats']  = stats

        # Infer SP/RP for pitchers
        if stats == 'pit' and 'GS' in df.columns and 'G' in df.columns:
            df['role'] = df.apply(
                lambda r: 'SP' if pd.notna(r['GS']) and pd.notna(r['G'])
                          and r['G'] > 0 and (r['GS'] / r['G']) >= 0.5
                          else 'RP',
                axis=1
            )
        elif stats == 'bat':
            # Use minpos directly for hitters
            df['role'] = df['minpos']

        # Ensure playerid is string for consistent joining
        df['playerid'] = df['playerid'].astype(str).str.strip()

        # Drop rows with no FPTS (true blanks, not just 0)
        df = df[df['FPTS'].notna()]

        return df

    except Exception as e:
        print(f"    Error fetching {year} {stats}: {e}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Historical Projection Fetcher")
    print("=" * 40)
    print("Enter your FanGraphs credentials:")
    username = input("  FG_USER: ").strip()
    password = getpass.getpass("  FG_PASS: ")

    print("\nLogging in...")
    session = get_session(username, password)

    all_frames = []
    print(f"\nFetching {len(HISTORICAL_YEARS)} years × 2 stat types = "
          f"{len(HISTORICAL_YEARS) * 2} API calls\n")

    for year in HISTORICAL_YEARS:
        for stats in ['bat', 'pit']:
            print(f"  Fetching {year} {stats}...", end=' ')
            df = fetch_projections(session, year, stats)

            if df.empty:
                print("FAILED")
                continue

            print(f"{len(df)} rows")

            # Save individual year file
            fname = os.path.join(DATA_DIR, f'steamer_{year}_{stats}.csv')
            df.to_csv(fname, index=False)

            all_frames.append(df)
            time.sleep(0.5)  # polite delay

    if not all_frames:
        print("\nERROR: No data fetched. Check credentials and try again.")
        return

    # Save combined file
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(DATA_DIR, 'historical_projections.csv')
    combined.to_csv(combined_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"✓ Done! Combined file: {combined_path}")
    print(f"\n=== Coverage by year ===")
    summary = combined.groupby(['season', 'stats']).agg(
        players=('playerid', 'count'),
        fpts_max=('FPTS', 'max'),
        fpts_median=('FPTS', 'median'),
    ).round(1)
    print(summary.to_string())

    print(f"\n=== SP/RP/position breakdown (sample year 2022) ===")
    y2022 = combined[combined['season'] == 2022]
    print(y2022['role'].value_counts().to_string())

    print(f"\n=== Sample: top 10 hitters by FPTS (2022) ===")
    top = (y2022[y2022['stats'] == 'bat']
           .nlargest(10, 'FPTS')[['PlayerName', 'role', 'FPTS', 'Team']])
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
