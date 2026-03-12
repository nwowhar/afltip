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

    The page structure is one table per team, with:
      - A caption like "Brisbane Lions Team Statistics [Players]"
      - Rows:  Round | Opponent | KI | MK | HB | DI | GL | BH | HO | TK | RB | IF | CL | CG | ...
      - Each stat cell is "home-away" e.g. "254-211" — we take the left (home team) value
      - A summary row starting with "W-D-L" that we skip

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

    # Column indices in the per-game rows (0=round, 1=opponent, then stats)
    # KI MK HB DI GL BH HO TK RB IF CL CG FF FA
    COL = {
        "kicks": 2, "marks": 3, "handballs": 4, "disposals": 5,
        "goals": 6, "behinds": 7, "hitouts": 8, "tackles": 9,
        "rebound_50s": 10, "inside_50s": 11, "clearances": 12,
        "clangers": 13, "frees_for": 14, "frees_against": 15,
    }

    records = []
    for table in tables:
        # Team name is in the caption
        caption = table.find("caption")
        if not caption:
            continue
        caption_text = caption.get_text(strip=True)
        # e.g. "Brisbane Lions Team Statistics [Players]"
        team_raw = caption_text.split(" Team Statistics")[0].strip()
        if team_raw not in TEAM_NAME_MAP:
            continue
        team = _normalise_team(team_raw)

        # Accumulate stat totals across all game rows
        totals = {k: 0.0 for k in COL}
        games = 0

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 10:
                continue
            # Skip header-like rows and the W-D-L summary row
            if not cells[0] or cells[0].startswith("W-D-L") or cells[0].startswith("R") is False:
                # Round cells start with "R" e.g. "R1", "R2", "EF", "SF", "PF", "GF"
                pass
            # Accept any row where cell[0] looks like a round label
            round_label = cells[0]
            if not round_label or cells[1] == "":
                continue
            # Skip W-D-L summary
            if "-" in round_label and round_label.count("-") == 2:
                continue

            games += 1
            for stat, idx in COL.items():
                if idx >= len(cells):
                    continue
                raw = cells[idx]
                # Values are "home-away" e.g. "254-211" — take left side
                home_val = raw.split("-")[0] if "-" in raw else raw
                try:
                    totals[stat] += float(home_val.replace(",", ""))
                except ValueError:
                    pass

        if games == 0:
            continue

        record = {
            "team": team,
            "year": year,
            "games_played": games,
            "avg_kicks":          round(totals["kicks"]          / games, 2),
            "avg_marks":          round(totals["marks"]          / games, 2),
            "avg_handballs":      round(totals["handballs"]      / games, 2),
            "avg_disposals":      round(totals["disposals"]      / games, 2),
            "avg_goals":          round(totals["goals"]          / games, 2),
            "avg_behinds":        round(totals["behinds"]        / games, 2),
            "avg_hitouts":        round(totals["hitouts"]        / games, 2),
            "avg_tackles":        round(totals["tackles"]        / games, 2),
            "avg_rebound_50s":    round(totals["rebound_50s"]    / games, 2),
            "avg_inside_50s":     round(totals["inside_50s"]     / games, 2),
            "avg_clearances":     round(totals["clearances"]     / games, 2),
            "avg_clangers":       round(totals["clangers"]       / games, 2),
            "avg_frees_for":      round(totals["frees_for"]      / games, 2),
            "avg_frees_against":  round(totals["frees_against"]  / games, 2),
            # Derived / aliased columns used elsewhere
            "avg_contested_possessions": None,
            "avg_uncontested_possessions": None,
            "avg_contested_marks": None,
            "avg_marks_inside_50": None,
        }
        records.append(record)

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
