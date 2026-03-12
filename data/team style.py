"""
data/team_style.py
------------------
Computes rolling team playing-style features from per-game stats.

Data source: AFL Tables match stats pages (already used in afltables.py)
URL pattern: https://afltables.com/afl/stats/games/{year}/{game_id}.html

These features capture HOW teams play, not just whether they win:
  - kick_ratio   : kicks / (kicks + handballs)  — style spectrum
  - tackle_rate  : tackles per game              — pressure intensity
  - hitout_rate  : hitouts per game              — ruck dominance
  - mark_rate    : marks per game                — aerial / corridor game

All features use a rolling window of the last N games BEFORE the target game
(no leakage). For early-season games where < N games exist, we fall back to
the previous season's average.

Matchup interaction features computed in build_prediction_features():
  - kick_ratio_diff     : home kick_ratio - away kick_ratio
  - tackle_diff         : home tackle_rate - away tackle_rate
  - hitout_diff         : home hitout_rate - away hitout_rate
  - mark_diff           : home mark_rate - away mark_rate
  - kick_vs_tackle      : interaction — does high-tackle team beat high-kick team?
  - ruck_advantage      : relative hitout dominance
"""

import re
import time
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

BASE = "https://afltables.com/afl"
HEADERS = {"User-Agent": "Mozilla/5.0 (AFL Predictor Research Bot)"}

STYLE_COLS = ["kicks", "handballs", "marks", "tackles", "hitouts"]
ROLLING_WINDOW = 8  # games


# ── AFL Tables team name normalisation ────────────────────────────────────

AFLT_TEAM_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane Bears": "Brisbane",
    "Brisbane Lions": "Brisbane",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fitzroy": None,         # pre-merger, skip
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS": "GWS",
    "Greater Western Sydney": "GWS",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Kangaroos": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "South Melbourne": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    "Footscray": "Western Bulldogs",
}


# ── Fetch per-game team stats from AFL Tables ─────────────────────────────

def fetch_game_stats_afltables(year: int, sleep: float = 0.3) -> pd.DataFrame:
    """
    Scrape the AFL Tables season stats page for a given year.
    Returns DataFrame with one row per team per game:
        year, round, game_id, team, opponent, venue,
        kicks, handballs, marks, tackles, hitouts,
        score, opp_score, won
    """
    # AFL Tables has a season-level stats index we can use to find game links
    url = f"{BASE}/stats/{year}t.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[team_style] Failed to fetch {year} index: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")

    # The season stats page has team-level aggregates but not per-game.
    # We need the per-game stats which are at:
    #   https://afltables.com/afl/stats/games/{year}/{YYYYMMDD}{home}{away}.html
    # Instead, use the season games index page which lists all game links:
    games_url = f"{BASE}/seas/{year}.html"
    try:
        r2 = requests.get(games_url, headers=HEADERS, timeout=20)
        r2.raise_for_status()
        soup2 = BeautifulSoup(r2.text, "html.parser")
    except Exception as e:
        print(f"[team_style] Failed to fetch {year} season page: {e}")
        return pd.DataFrame()

    # Find all game stat links
    game_links = []
    for a in soup2.find_all("a", href=re.compile(r"stats/games/\d{4}/\d+")):
        href = a["href"]
        if href not in game_links:
            game_links.append(href)

    if not game_links:
        print(f"[team_style] No game links found for {year}")
        return pd.DataFrame()

    all_rows = []
    for i, link in enumerate(game_links):
        full_url = f"https://afltables.com/afl/{link.lstrip('/')}"
        rows = _parse_game_stats_page(full_url, year)
        all_rows.extend(rows)
        if i % 20 == 0:
            print(f"[team_style] {year}: {i}/{len(game_links)} games...")
        time.sleep(sleep)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def _parse_game_stats_page(url: str, year: int) -> list:
    """
    Parse a single AFL Tables game stats page.
    Returns list of 2 dicts (one per team).
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []

    rows = []
    tables = soup.find_all("table")

    # Game stats page has tables for each quarter + totals
    # We want the "Totals" row for each team
    # Structure: Team name in heading, then stat table below

    # Extract round and teams from page title / headers
    title = soup.find("title")
    title_text = title.get_text() if title else ""

    # Find team stat rows — AFL Tables format varies by era
    # Look for rows containing kicks, handballs etc
    team_stats = {}

    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "kicks" not in headers and "k" not in headers:
            continue

        # Map header positions
        col_map = {}
        for idx, h in enumerate(headers):
            if h in ("kicks", "k"):       col_map["kicks"] = idx
            elif h in ("handballs", "hb"): col_map["handballs"] = idx
            elif h in ("marks", "m"):      col_map["marks"] = idx
            elif h in ("tackles", "tk"):   col_map["tackles"] = idx
            elif h in ("hitouts", "ho"):   col_map["hitouts"] = idx

        if not col_map:
            continue

        data_rows = table.find_all("tr")
        for row in data_rows:
            cells = row.find_all("td")
            if not cells:
                continue
            label = cells[0].get_text(strip=True)
            if label.lower() in ("totals", "total"):
                # Identify which team this belongs to by finding nearest heading
                # (simplified: collect both rows, assign to teams in order)
                try:
                    stat_dict = {}
                    for stat, idx in col_map.items():
                        if idx < len(cells):
                            val = cells[idx].get_text(strip=True)
                            stat_dict[stat] = float(val) if val.isdigit() else 0.0
                    if stat_dict:
                        key = f"team_{len(team_stats)}"
                        team_stats[key] = stat_dict
                except (ValueError, IndexError):
                    continue

    # Extract team names and scores from page
    # Look for score pattern e.g. "Richmond 15.12 (102) def Carlton 10.8 (68)"
    score_pattern = re.search(
        r"([A-Za-z ]+?)\s+(\d+\.\d+)\s+\((\d+)\).*?([A-Za-z ]+?)\s+(\d+\.\d+)\s+\((\d+)\)",
        title_text
    )

    game_id = url.split("/")[-1].replace(".html", "")

    # If we got exactly 2 team stat blocks, package them
    if len(team_stats) >= 2:
        teams = list(team_stats.values())
        for i, (team_key, stats) in enumerate(list(team_stats.items())[:2]):
            rows.append({
                "year":       year,
                "game_id":    game_id,
                "team_slot":  i,  # 0=home, 1=away (approximate)
                **stats,
            })

    return rows


# ── Higher-level: build rolling style profiles ────────────────────────────

def build_team_style_from_squiggle(enriched_df: pd.DataFrame,
                                    window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Build rolling team style features directly from the enriched games DataFrame
    (which already has per-game score, team names, round, year from Squiggle).

    Squiggle gives us scores but NOT kicks/handballs/tackles/hitouts per game.
    So we use AFL Tables season TOTALS (already in afltables.py) and compute
    per-game averages for the season-to-date as the proxy.

    This function computes rolling SEASON-AVERAGE style features for each team
    at each point in time, using AFL Tables season stats fetched in afltables.py.

    Parameters
    ----------
    enriched_df : the main games DataFrame (from fetcher.py), must have columns:
                  year, round, hteam, ateam, date
    window      : not used here (season averages used instead of rolling)

    Returns
    -------
    DataFrame with columns:
        year, team, kick_ratio, tackle_rate, hitout_rate, mark_rate
    (one row per team per season — used as season proxy)
    """
    # This is called at model build time; afltables season stats are already loaded
    # We just need to reshape them into per-team style features
    raise NotImplementedError("Use build_style_features_from_season_stats() instead")


def build_style_features_from_season_stats(season_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive team style features from AFL Tables season stats DataFrame.

    season_stats_df: from afltables.get_season_stats(), columns include:
        year, team, kicks, handballs, marks, tackles, hitouts, games_played, ...

    Returns per-team per-year style profile:
        year, team, kick_ratio, tackle_rate, hitout_rate, mark_rate
    """
    if season_stats_df is None or season_stats_df.empty:
        return pd.DataFrame()

    df = season_stats_df.copy()

    # Ensure numeric
    for col in ["kicks", "handballs", "marks", "tackles", "hitouts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Per-game rates (season totals / games played)
    games = df.get("games_played", df.get("games", pd.Series(22, index=df.index)))
    games = pd.to_numeric(games, errors="coerce").fillna(22).clip(lower=1)

    result = pd.DataFrame()
    result["year"]  = df["year"] if "year" in df.columns else df.index
    result["team"]  = df["team"]

    kicks     = pd.to_numeric(df.get("kicks",     0), errors="coerce").fillna(0)
    handballs = pd.to_numeric(df.get("handballs", 0), errors="coerce").fillna(0)
    marks     = pd.to_numeric(df.get("marks",     0), errors="coerce").fillna(0)
    tackles   = pd.to_numeric(df.get("tackles",   0), errors="coerce").fillna(0)
    hitouts   = pd.to_numeric(df.get("hitouts",   0), errors="coerce").fillna(0)

    total_disp = kicks + handballs
    result["kick_ratio"]  = kicks / total_disp.clip(lower=1)   # 0–1, higher = kick-heavy
    result["tackle_rate"] = tackles / games                      # tackles per game
    result["hitout_rate"] = hitouts / games                      # hitouts per game
    result["mark_rate"]   = marks / games                        # marks per game
    result["disposal_rate"] = total_disp / games                 # total disposals per game

    return result.reset_index(drop=True)


# ── Compute matchup features for a single game ────────────────────────────

def compute_style_matchup(home_team: str, away_team: str,
                           style_df: pd.DataFrame,
                           game_year: int,
                           use_prev_season: bool = False) -> dict:
    """
    Given a style profile DataFrame (team, year, kick_ratio, tackle_rate,
    hitout_rate, mark_rate), compute matchup features for one game.

    use_prev_season: if True, look up year-1 stats (for early season games
                     before current-year stats are meaningful)

    Returns dict of features (all zero if data unavailable).
    """
    lookup_year = (game_year - 1) if use_prev_season else game_year

    h_row = style_df[(style_df["team"] == home_team) &
                     (style_df["year"] == lookup_year)]
    a_row = style_df[(style_df["team"] == away_team) &
                     (style_df["year"] == lookup_year)]

    # Fallback to prev season if current year not available
    if h_row.empty or a_row.empty:
        h_row = style_df[(style_df["team"] == home_team) &
                         (style_df["year"] == game_year - 1)]
        a_row = style_df[(style_df["team"] == away_team) &
                         (style_df["year"] == game_year - 1)]

    if h_row.empty or a_row.empty:
        return {f: 0.0 for f in STYLE_FEATURES}

    h = h_row.iloc[0]
    a = a_row.iloc[0]

    kick_ratio_diff  = float(h["kick_ratio"])  - float(a["kick_ratio"])
    tackle_diff      = float(h["tackle_rate"]) - float(a["tackle_rate"])
    hitout_diff      = float(h["hitout_rate"]) - float(a["hitout_rate"])
    mark_diff        = float(h["mark_rate"])   - float(a["mark_rate"])

    # Interaction: does a tackle-heavy team neutralise a kick-heavy team?
    # Positive = home team kicks more AND away team tackles less (home advantage)
    # Negative = away team applies more pressure to home's kicking style
    kick_vs_tackle = kick_ratio_diff * (-tackle_diff)

    # Ruck advantage: pure hitout gap, sign = home advantage
    ruck_advantage = hitout_diff

    return {
        "kick_ratio_diff":  kick_ratio_diff,
        "tackle_diff":      tackle_diff,
        "hitout_diff":      hitout_diff,
        "mark_diff":        mark_diff,
        "kick_vs_tackle":   kick_vs_tackle,
        "ruck_advantage":   ruck_advantage,
    }


# Feature list for use in predictor.py
STYLE_FEATURES = [
    "kick_ratio_diff",   # kick-heavy vs handball-heavy styles
    "tackle_diff",       # pressure/contested game intensity
    "hitout_diff",       # ruck dominance
    "mark_diff",         # aerial vs ground game
    "kick_vs_tackle",    # interaction: kick style vs pressure defence
    "ruck_advantage",    # relative ruck dominance (same as hitout_diff, kept explicit)
]


# ── Attach style features to training DataFrame ────────────────────────────

def attach_style_features(df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game in df, look up prior-season style features for home and away team.
    Adds STYLE_FEATURES columns to df.

    Leakage-safe: uses year-1 style data for games in year Y.
    """
    if style_df is None or style_df.empty:
        for f in STYLE_FEATURES:
            df[f] = 0.0
        return df

    rows = []
    for _, game in df.iterrows():
        feat = compute_style_matchup(
            home_team=game["hteam"],
            away_team=game["ateam"],
            style_df=style_df,
            game_year=int(game["year"]),
            use_prev_season=True,  # always use prior season for training (no leakage)
        )
        rows.append(feat)

    style_game_df = pd.DataFrame(rows, index=df.index)
    for col in style_game_df.columns:
        df[col] = style_game_df[col].fillna(0.0)

    return df
