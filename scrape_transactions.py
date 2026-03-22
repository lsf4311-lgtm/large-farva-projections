"""
scrape_transactions.py
----------------------
One-time scrape of all Ottoneu League 569 transaction history.
Saves to transaction_log.csv in the same folder as this script (or DATA_DIR if set).

Runtime: ~15-20 minutes for 393 pages (1.5s delay between requests).
Run once, then use the cached CSV for the inflation model.

Usage:
    python scrape_transactions.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
LEAGUE_ID = "569"
BASE_URL = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/transactions"
TOTAL_PAGES = 393          # Update this if the league grows
DELAY_SECONDS = 1.5        # Be polite to Ottoneu's servers
OUTPUT_FILENAME = "transaction_log.csv"

# Output goes to data\ subfolder if it exists, otherwise same dir as script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = SCRIPT_DIR
OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_FILENAME)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': f'https://ottoneu.fangraphs.com/{LEAGUE_ID}/transactions',
}

# ── Scraper ───────────────────────────────────────────────────────────────────
def scrape_page(session, page_num):
    """Scrape a single transaction log page. Returns list of row dicts."""
    url = f"{BASE_URL}?page={page_num}"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} on page {page_num} — skipping")
            return []
    except Exception as e:
        print(f"  Request failed on page {page_num}: {e} — skipping")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find the transactions table — look for table containing transaction rows
    table = soup.find('table')
    if not table:
        print(f"  No table found on page {page_num}")
        return []

    rows = []
    for tr in table.find_all('tr')[1:]:  # skip header row
        cells = tr.find_all('td')
        if len(cells) < 4:
            continue

        # Column order from your sample:
        # Date | Transaction Type | Player Name | Team Name | From Team | Salary
        date_text     = cells[0].get_text(strip=True)
        trans_type    = cells[1].get_text(strip=True)
        
        # Player name cell — extract both name and Ottoneu player ID from href
        player_cell   = cells[2]
        player_link   = player_cell.find('a')
        player_name   = player_link.get_text(strip=True) if player_link else player_cell.get_text(strip=True)
        ottoneu_id    = None
        if player_link:
            href = player_link.get('href', '')
            # Pattern: /569/players/6305
            if '/players/' in href:
                try:
                    ottoneu_id = int(href.split('/players/')[-1].strip('/'))
                except ValueError:
                    pass

        team_name     = cells[3].get_text(strip=True)

        # From Team (populated for trades, empty otherwise)
        from_team     = cells[4].get_text(strip=True) if len(cells) > 4 else ''
        from_team     = from_team if from_team else None

        # Salary — strip $ sign
        salary_text   = cells[5].get_text(strip=True) if len(cells) > 5 else ''
        salary        = None
        if salary_text:
            try:
                salary = int(salary_text.replace('$', '').replace(',', '').strip())
            except ValueError:
                pass

        rows.append({
            'date':          date_text,
            'trans_type':    trans_type,
            'player_name':   player_name,
            'ottoneu_id':    ottoneu_id,
            'team_name':     team_name,
            'from_team':     from_team,
            'salary':        salary,
            'page':          page_num,
        })

    return rows


def scrape_all_pages(start_page=1, end_page=TOTAL_PAGES):
    """Scrape all pages and return a combined DataFrame."""
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    start_time = datetime.now()

    print(f"Starting scrape: pages {start_page}–{end_page}")
    print(f"Estimated time: {((end_page - start_page + 1) * DELAY_SECONDS / 60):.1f} minutes")
    print(f"Output: {OUTPUT_PATH}\n")

    for page_num in range(start_page, end_page + 1):
        rows = scrape_page(session, page_num)
        all_rows.extend(rows)

        # Progress update every 25 pages
        if page_num % 25 == 0 or page_num == end_page:
            elapsed = (datetime.now() - start_time).seconds
            pct = (page_num - start_page + 1) / (end_page - start_page + 1)
            eta_sec = int(elapsed / pct * (1 - pct)) if pct > 0 else 0
            print(
                f"  Page {page_num}/{end_page} | "
                f"{len(all_rows):,} rows | "
                f"Elapsed: {elapsed//60}m{elapsed%60:02d}s | "
                f"ETA: {eta_sec//60}m{eta_sec%60:02d}s"
            )

        # Save checkpoint every 50 pages in case of interruption
        if page_num % 50 == 0:
            checkpoint_path = OUTPUT_PATH.replace('.csv', f'_checkpoint_p{page_num}.csv')
            pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        time.sleep(DELAY_SECONDS)

    return pd.DataFrame(all_rows)


def clean_transactions(df):
    """Parse dates, standardize types, sort oldest-first."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Normalize transaction type
    df['trans_type'] = df['trans_type'].str.lower().str.strip()

    # Flag likely auction transactions:
    # Auction days = dates where 50+ adds happen across 8+ teams in a single day
    df['date_only'] = df['date'].dt.date
    daily = df[df['trans_type'] == 'add'].groupby('date_only').agg(
        add_count=('trans_type', 'count'),
        team_count=('team_name', 'nunique')
    ).reset_index()
    auction_dates = daily[(daily['add_count'] >= 50) & (daily['team_count'] >= 8)]['date_only']
    df['is_auction'] = df['date_only'].isin(auction_dates) & (df['trans_type'] == 'add')

    df = df.drop(columns=['date_only'])
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Optional: resume from a checkpoint if interrupted
    # Change start_page to resume from where you left off
    df = scrape_all_pages(start_page=1, end_page=TOTAL_PAGES)

    if df.empty:
        print("ERROR: No data scraped. Check your network connection and try again.")
    else:
        df = clean_transactions(df)

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✓ Done! {len(df):,} transactions saved to {OUTPUT_PATH}")

        # Summary
        print(f"\n=== Summary ===")
        print(f"Date range:      {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"Transaction types:\n{df['trans_type'].value_counts().to_string()}")
        print(f"Auction days detected: {df['is_auction'].sum()} transactions flagged")
        print(f"Ottoneu IDs captured: {df['ottoneu_id'].notna().sum():,} / {len(df):,}")
        auction_days = df[df['is_auction']]['date'].dt.date.unique()
        print(f"Auction dates: {sorted(auction_days)}")
