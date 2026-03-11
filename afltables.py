"""
AFL Tables scraper — fetches team season-level stats (clearances, inside 50s,
contested possessions, tackles etc.) from the static HTML pages that don't
require JavaScript.

URL pattern: https://afltables.com/afl/stats/{year}t.html
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AFL-Predictor/1.0; +https://github.com/nwowhar/afltip)"
}

# Map AFL Tables team name variants → our standard names
TEAM_NAME_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS Giants": "GWS Giants",
    "Greater Western Sydney": "GWS Giants",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Kangaroos": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    "Footscray": "Western Bulldogs",
}

STAT_COLS = [
    "kicks", "marks", "handballs", "disposals",
    "goals", "behinds", "hitouts", "tackles",
    "rebound_50s", "inside_50s", "clearances", "clangers",
    "frees_for", "frees_against",
    "contested_possessions", "uncontested_possessions",
    "contested_marks", "marks_inside_50",
]


def _normalise_team(name: str) -> str:
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def scrape_team_season_stats(year: int) -> pd.DataFrame:
    """
    Scrape team stat totals for a given year from AFL Tables.
    Returns a DataFrame with one row per team with per-game averages.
    """
    url = f"https://afltables.com/afl/stats/{year}t.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  AFL Tables fetch failed for {year}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")

    records = []
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 10:
                continue
            # First cell should be team name
            team_raw = cells[0]
            if team_raw not in TEAM_NAME_MAP:
                continue
            team = _normalise_team(team_raw)
            try:
                # AFL Tables table 1 columns (approximate):
                # Team | GP | KI | MK | HB | DI | GL | BH | HO | TK | RB | IF | CL | CG | FF | FA | ...
                gp = int(cells[1]) if cells[1].isdigit() else None
                if not gp or gp == 0:
                    continue

                def safe_avg(idx):
                    try:
                        val = float(cells[idx].replace(",", ""))
                        return round(val / gp, 2)
                    except Exception:
                        return None

                record = {
                    "team": team,
                    "year": year,
                    "games_played": gp,
                    "avg_kicks":                 safe_avg(2),
                    "avg_marks":                 safe_avg(3),
                    "avg_handballs":             safe_avg(4),
                    "avg_disposals":             safe_avg(5),
                    "avg_goals":                 safe_avg(6),
                    "avg_behinds":               safe_avg(7),
                    "avg_hitouts":               safe_avg(8),
                    "avg_tackles":               safe_avg(9),
                    "avg_rebound_50s":           safe_avg(10),
                    "avg_inside_50s":            safe_avg(11),
                    "avg_clearances":            safe_avg(12),
                    "avg_clangers":              safe_avg(13),
                    "avg_frees_for":             safe_avg(14),
                    "avg_frees_against":         safe_avg(15),
                }
                records.append(record)
            except Exception:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(subset=["team", "year"])
    return df


def get_all_team_season_stats(start_year: int = 2013) -> pd.DataFrame:
    """Fetch team season stats for all years from start_year to present."""
    current_year = datetime.now().year
    frames = []
    for year in range(start_year, current_year + 1):
        df = scrape_team_season_stats(year)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # be polite to the server
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_team_stats_for_game(team: str, year: int,
                             season_stats: pd.DataFrame) -> dict:
    """
    Look up a team's season stats for a given year.
    Returns dict of avg stats, or empty defaults if not found.
    """
    defaults = {
        "avg_clearances": 35.0,
        "avg_inside_50s": 50.0,
        "avg_contested_possessions": 130.0,
        "avg_tackles": 55.0,
        "avg_hitouts": 35.0,
        "avg_disposals": 370.0,
        "avg_clangers": 50.0,
        "avg_marks_inside_50": 10.0,
    }
    if season_stats is None or season_stats.empty:
        return defaults

    row = season_stats[(season_stats["team"] == team) &
                       (season_stats["year"] == year)]
    if row.empty:
        # Try previous year
        row = season_stats[(season_stats["team"] == team) &
                           (season_stats["year"] == year - 1)]
    if row.empty:
        return defaults

    r = row.iloc[0].to_dict()
    return {k: (r.get(k) or defaults.get(k, 0)) for k in defaults}
