import requests
import pandas as pd
from datetime import datetime

SQUIGGLE_BASE = "https://api.squiggle.com.au/"

def get_games(year: int) -> pd.DataFrame:
    """Fetch all completed games for a given year."""
    url = f"{SQUIGGLE_BASE}?q=games;year={year}"
    headers = {"User-Agent": "AFL-Predictor/1.0 (contact@example.com)"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    games = r.json().get("games", [])
    df = pd.DataFrame(games)
    if df.empty:
        return df
    # Keep completed games only
    df = df[df["complete"] == 100].copy()
    df["year"] = year
    return df

def get_all_games(start_year: int = 2010) -> pd.DataFrame:
    """Fetch all games from start_year to current year."""
    current_year = datetime.now().year
    frames = []
    for year in range(start_year, current_year + 1):
        try:
            df = get_games(year)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Error fetching {year}: {e}")
    if not frames:
        return pd.DataFrame()
    all_games = pd.concat(frames, ignore_index=True)
    return all_games

def get_upcoming_games() -> pd.DataFrame:
    """Fetch upcoming / in-progress games for the current year."""
    current_year = datetime.now().year
    url = f"{SQUIGGLE_BASE}?q=games;year={current_year}"
    headers = {"User-Agent": "AFL-Predictor/1.0 (contact@example.com)"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    games = r.json().get("games", [])
    df = pd.DataFrame(games)
    if df.empty:
        return df
    # Incomplete games
    upcoming = df[df["complete"] < 100].copy()
    return upcoming

def get_teams() -> list:
    """Return sorted list of all team names."""
    url = f"{SQUIGGLE_BASE}?q=teams"
    headers = {"User-Agent": "AFL-Predictor/1.0 (contact@example.com)"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    teams = r.json().get("teams", [])
    return sorted([t["name"] for t in teams])
