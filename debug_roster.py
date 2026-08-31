"""
One-off diagnostic: prints the RAW text BeautifulSoup actually sees for a
handful of roster rows, so we can fix ottoneu_roster.py's opponent/pitcher
parsing against real data instead of guessing again.

Run: python debug_roster.py
"""
import datetime as dt
import requests
from bs4 import BeautifulSoup
import config

target_date = dt.date.today() + dt.timedelta(days=config.LOOKAHEAD_DAYS)
url = (
    f"https://ottoneu.fangraphs.com/{config.OTTONEU_LEAGUE_ID}/"
    f"setlineups?team={config.OTTONEU_TEAM_ID}&date={target_date.isoformat()}&statFilter=season"
)
print(f"Fetching: {url}\n")

resp = requests.get(url, timeout=15)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

shown = 0
for table in soup.find_all("table"):
    header_text = table.get_text(" ", strip=True)[:250]
    if "Position" not in header_text or "Name" not in header_text:
        continue
    is_pitcher_table = "IP" in header_text and "AB" not in header_text
    label = "PITCHER" if is_pitcher_table else "BATTER"

    if not is_pitcher_table:
        continue  # this run only cares about the full pitcher table

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        link = row.find("a", href=True)
        roster_slot = cells[0].get_text(strip=True) if cells else "?"
        if not cells:
            continue
        if not link:
            print(f"  [{roster_slot}] (empty slot, no player)")
            continue
        name = link.get_text(strip=True)
        opp_text = cells[2].get_text(" ", strip=True) if len(cells) > 2 else "?"
        print(f"  [{roster_slot}] {name!r}  opponent_cell={opp_text!r}")
        shown += 1
print(f"\nTotal pitcher rows with a player: {shown}")
