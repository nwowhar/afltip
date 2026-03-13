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
    Scrape team season stat totals from AFL Tables {year}s.html.

    That page has a clean "Team Totals For" table:
      Team | KI | MK | HB | DI | GL | BH | HO | TK | RB | IF | CL | CG | FF | BR | CP | UP | CM | MI ...
    Values are season TOTALS. We also fetch games played from {year}t.html (game-by-game)
    by counting game rows per team, then divide totals by GP to get per-game averages.
    """
    # --- Step 1: fetch season totals from {year}s.html ---
    url_s = f"https://afltables.com/afl/stats/{year}s.html"
    try:
        resp = requests.get(url_s, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  AFL Tables fetch failed for {year}s: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")

    # Column order in "Team Totals For" table (0-indexed after team name)
    # KI MK HB DI GL BH HO TK RB IF CL CG FF BR CP UP CM MI 1% BO GA
    COL = {
        "kicks": 1, "marks": 2, "handballs": 3, "disposals": 4,
        "goals": 6, "behinds": 7, "hitouts": 8, "tackles": 9,
        "rebound_50s": 10, "inside_50s": 11, "clearances": 12,
        "clangers": 13, "frees_for": 14,
        "contested_possessions": 16, "uncontested_possessions": 17,
        "contested_marks": 18, "marks_inside_50": 19,
    }

    totals_by_team = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        # Find the header row with "KI" to confirm this is the right table
        header_cells = [th.get_text(strip=True) for th in (rows[0].find_all("th") if rows else [])]
        if "KI" not in header_cells and "HO" not in header_cells:
            continue
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 10:
                continue
            # First cell has team name as link text
            team_cell = row.find("td")
            if not team_cell:
                continue
            team_raw = team_cell.get_text(strip=True)
            if team_raw not in TEAM_NAME_MAP:
                continue
            team = _normalise_team(team_raw)
            record = {}
            for stat, idx in COL.items():
                if idx < len(cells):
                    try:
                        record[stat] = float(cells[idx].replace(",", ""))
                    except ValueError:
                        record[stat] = None
            totals_by_team[team] = record
        if totals_by_team:
            break  # found the right table

    if not totals_by_team:
        return pd.DataFrame()

    # --- Step 2: get games played from {year}t.html (game-by-game page) ---
    gp_by_team = {}
    url_t = f"https://afltables.com/afl/stats/{year}t.html"
    try:
        resp_t = requests.get(url_t, headers=HEADERS, timeout=20)
        resp_t.raise_for_status()
        soup_t = BeautifulSoup(resp_t.text, "lxml")
        for table in soup_t.find_all("table"):
            caption = table.find("caption")
            if not caption:
                continue
            team_raw = caption.get_text(strip=True).split(" Team Statistics")[0].strip()
            if team_raw not in TEAM_NAME_MAP:
                continue
            team = _normalise_team(team_raw)
            # Count game rows (round label in cell 0, not W-D-L)
            gp = 0
            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 10:
                    continue
                r = cells[0]
                if r and r.count("-") != 2:
                    gp += 1
            if gp > 0:
                gp_by_team[team] = gp
    except Exception as e:
        print(f"  AFL Tables GP fetch failed for {year}: {e}")

    # --- Step 3: build per-game averages ---
    records = []
    for team, totals in totals_by_team.items():
        gp = gp_by_team.get(team)
        if not gp:
            # Fallback: estimate GP from disposals (league avg ~375 disp/game)
            disp = totals.get("disposals")
            gp = max(1, round(disp / 375)) if disp else 1

        def avg(stat):
            v = totals.get(stat)
            return round(v / gp, 2) if v is not None else None

        records.append({
            "team": team, "year": year, "games_played": gp,
            "avg_kicks":                    avg("kicks"),
            "avg_marks":                    avg("marks"),
            "avg_handballs":                avg("handballs"),
            "avg_disposals":                avg("disposals"),
            "avg_goals":                    avg("goals"),
            "avg_behinds":                  avg("behinds"),
            "avg_hitouts":                  avg("hitouts"),
            "avg_tackles":                  avg("tackles"),
            "avg_rebound_50s":              avg("rebound_50s"),
            "avg_inside_50s":               avg("inside_50s"),
            "avg_clearances":               avg("clearances"),
            "avg_clangers":                 avg("clangers"),
            "avg_frees_for":                avg("frees_for"),
            "avg_frees_against":            None,
            "avg_contested_possessions":    avg("contested_possessions"),
            "avg_uncontested_possessions":  avg("uncontested_possessions"),
            "avg_contested_marks":          avg("contested_marks"),
            "avg_marks_inside_50":          avg("marks_inside_50"),
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).drop_duplicates(subset=["team", "year"])


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
