"""
Lineup and PAV (Player Approximate Value) fetcher from Squiggle API.

Lineup endpoint:  api.squiggle.com.au/?q=lineup;year=YYYY;round=R
PAV endpoint:     api.squiggle.com.au/?q=pav;year=YYYY

PAV breaks each player's value into offensive, defensive, and midfield
components. When we sum the selected 22's PAV we get a "team strength today"
number that automatically accounts for injuries and omissions.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

BASE = "https://api.squiggle.com.au/"
HEADERS = {"User-Agent": "AFL-Predictor/1.0 (github.com/nwowhar/afltip)"}


# ── PAV ───────────────────────────────────────────────────────────────────────

def get_pav(year: int) -> pd.DataFrame:
    """Fetch Player Approximate Value ratings for a given year."""
    url = f"{BASE}?q=pav;year={year}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json().get("pav", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["year"] = year
        return df
    except Exception as e:
        print(f"PAV fetch failed {year}: {e}")
        return pd.DataFrame()


def get_pav_multi_year(start_year: int = 2013) -> pd.DataFrame:
    """Fetch PAV ratings for all years from start_year to present."""
    current_year = datetime.now().year
    frames = []
    for year in range(start_year, current_year + 1):
        df = get_pav(year)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_player_career_pav(pav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each player's rolling average PAV over their career.
    Returns a DataFrame with player name, latest team, and avg PAV stats.
    """
    if pav_df.empty:
        return pd.DataFrame()

    # Use most recent year's data per player
    latest = pav_df.sort_values("year").groupby(
        ["firstname", "surname"], as_index=False
    ).last()

    numeric_cols = ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]
    for col in numeric_cols:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0)

    return latest


# ── Lineups ───────────────────────────────────────────────────────────────────

def get_lineup(year: int, round_num: int) -> pd.DataFrame:
    """
    Fetch announced team lineups for a given year and round.
    Returns DataFrame with columns: gameid, team, player, status, position.
    """
    url = f"{BASE}?q=lineup;year={year};round={round_num}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json().get("lineups", [])
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Lineup fetch failed {year} R{round_num}: {e}")
        return pd.DataFrame()


def get_current_lineups() -> pd.DataFrame:
    """Try to fetch lineups for the current/upcoming round."""
    year = datetime.now().year
    # Try rounds 1-28, find the latest with lineup data
    for rnd in range(28, 0, -1):
        df = get_lineup(year, rnd)
        if not df.empty:
            return df
    return pd.DataFrame()


# ── Team strength from lineups + PAV ─────────────────────────────────────────

TEAM_NAME_SQUIGGLE_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS": "GWS Giants",
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
}


def compute_lineup_strength(lineup_df: pd.DataFrame,
                             pav_df: pd.DataFrame,
                             game_id=None) -> dict:
    """
    Given announced lineups and PAV data, compute team strength scores.

    Returns dict: {team_name: {pav_total, pav_off, pav_def, pav_mid, n_players}}
    """
    if lineup_df.empty or pav_df.empty:
        return {}

    # Filter to specific game if provided
    if game_id and "gameid" in lineup_df.columns:
        lineup_df = lineup_df[lineup_df["gameid"] == game_id]

    if lineup_df.empty:
        return {}

    # Get latest PAV per player
    player_pav = get_player_career_pav(pav_df)
    if player_pav.empty:
        return {}

    # Build lookup: (firstname, surname) → PAV stats
    pav_lookup = {}
    for _, row in player_pav.iterrows():
        key = (str(row.get("firstname", "")).strip().lower(),
               str(row.get("surname", "")).strip().lower())
        pav_lookup[key] = {
            "PAV_total": float(row.get("PAV_total", 0) or 0),
            "PAV_off":   float(row.get("PAV_off",   0) or 0),
            "PAV_def":   float(row.get("PAV_def",   0) or 0),
            "PAV_mid":   float(row.get("PAV_mid",   0) or 0),
        }

    results = {}

    # Group lineups by team
    team_col = "teamname" if "teamname" in lineup_df.columns else "team"
    fn_col = "firstname" if "firstname" in lineup_df.columns else "givenname"
    sn_col = "surname"

    if team_col not in lineup_df.columns:
        return {}

    for team_raw, group in lineup_df.groupby(team_col):
        team = TEAM_NAME_SQUIGGLE_MAP.get(team_raw, team_raw)
        totals = {"PAV_total": 0, "PAV_off": 0, "PAV_def": 0, "PAV_mid": 0}
        n = 0

        for _, player in group.iterrows():
            fn = str(player.get(fn_col, "") or "").strip().lower()
            sn = str(player.get(sn_col, "") or "").strip().lower()
            pav = pav_lookup.get((fn, sn))
            if pav:
                for k in totals:
                    totals[k] += pav[k]
                n += 1

        results[team] = {**totals, "n_players_matched": n}

    return results


def get_lineup_pav_diff(home_team: str, away_team: str,
                        lineup_strength: dict) -> dict:
    """
    Compute PAV differential between two teams' selected lineups.
    Returns feature dict ready to add to prediction.
    """
    h = lineup_strength.get(home_team, {})
    a = lineup_strength.get(away_team, {})

    h_total = h.get("PAV_total", 0)
    a_total = a.get("PAV_total", 0)
    h_off   = h.get("PAV_off",   0)
    a_off   = a.get("PAV_off",   0)
    h_def   = h.get("PAV_def",   0)
    a_def   = a.get("PAV_def",   0)
    h_mid   = h.get("PAV_mid",   0)
    a_mid   = a.get("PAV_mid",   0)

    return {
        "pav_total_diff": h_total - a_total,
        "pav_off_diff":   h_off   - a_off,
        "pav_def_diff":   h_def   - a_def,
        "pav_mid_diff":   h_mid   - a_mid,
        "home_pav_total": h_total,
        "away_pav_total": a_total,
        "lineup_available": 1 if (h_total > 0 and a_total > 0) else 0,
    }
