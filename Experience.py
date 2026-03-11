"""
Player Experience features for AFL Predictor.

Fetches career game counts from AFL Tables player pages, computes per-team averages,
and applies a finals multiplier (finals experience counts more than regular season).

Career stages (approximate):
  0–24 games  → Developing
  25–74 games → Emerging
  75–149 games → Prime
  150–199 games → Veteran
  200+ games  → Elite Veteran

Finals weighting: finals games count as FINALS_MULTIPLIER × regular games.
This means a player with 50 regular + 10 finals games has
50 + 10×FINALS_MULTIPLIER weighted experience.
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re

BASE_AFL_TABLES = "https://afltables.com/afl"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AFL-Predictor/1.0)"}

FINALS_MULTIPLIER = 2.5  # finals games worth this many regular games for experience

CAREER_STAGES = {
    "Developing":     (0,   24),
    "Emerging":       (25,  74),
    "Prime":          (75,  149),
    "Veteran":        (150, 199),
    "Elite Veteran":  (200, 9999),
}

TEAM_PAGES = {
    "Adelaide":        "https://afltables.com/afl/teams/adelaide/playeridx.html",
    "Brisbane Lions":  "https://afltables.com/afl/teams/brisbanel/playeridx.html",
    "Carlton":         "https://afltables.com/afl/teams/carlton/playeridx.html",
    "Collingwood":     "https://afltables.com/afl/teams/collingwood/playeridx.html",
    "Essendon":        "https://afltables.com/afl/teams/essendon/playeridx.html",
    "Fremantle":       "https://afltables.com/afl/teams/fremantle/playeridx.html",
    "Geelong":         "https://afltables.com/afl/teams/geelong/playeridx.html",
    "Gold Coast":      "https://afltables.com/afl/teams/goldcoast/playeridx.html",
    "GWS Giants":      "https://afltables.com/afl/teams/gws/playeridx.html",
    "Hawthorn":        "https://afltables.com/afl/teams/hawthorn/playeridx.html",
    "Melbourne":       "https://afltables.com/afl/teams/melbourne/playeridx.html",
    "North Melbourne": "https://afltables.com/afl/teams/northmelbourne/playeridx.html",
    "Port Adelaide":   "https://afltables.com/afl/teams/portadelaide/playeridx.html",
    "Richmond":        "https://afltables.com/afl/teams/richmond/playeridx.html",
    "St Kilda":        "https://afltables.com/afl/teams/stkilda/playeridx.html",
    "Sydney":          "https://afltables.com/afl/teams/sydney/playeridx.html",
    "West Coast":      "https://afltables.com/afl/teams/westcoast/playeridx.html",
    "Western Bulldogs":"https://afltables.com/afl/teams/bullldogs/playeridx.html",
}


def get_career_stage(weighted_games: float) -> str:
    for stage, (lo, hi) in CAREER_STAGES.items():
        if lo <= weighted_games <= hi:
            return stage
    return "Elite Veteran"


def fetch_player_career(player_url: str) -> dict:
    """
    Fetch a single player's career stats from their AFL Tables page.
    Returns dict with: regular_games, finals_games, weighted_games, years_active, last_year.
    """
    try:
        r = requests.get(player_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        regular_games = 0
        finals_games  = 0
        years_active  = set()

        # AFL Tables player pages have a table with rows for each season
        # Columns typically: Year | Team | #(games) | K | M | HB | D | G | B | HO | TK | RB | IF | CL | CG | FF | FA | BR | CP | UP | CM | MI | 1% | BO | GA
        tables = soup.find_all("table")
        for tbl in tables:
            rows = tbl.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 3:
                    continue
                year_text = cells[0]
                # Year cell looks like "2019" or "2019F" (finals) or "Totals"
                if year_text == "Totals" or not re.match(r'^\d{4}', year_text):
                    continue
                is_finals = year_text.endswith("F") or "final" in year_text.lower()
                year_num_match = re.match(r'(\d{4})', year_text)
                if not year_num_match:
                    continue
                year_num = int(year_num_match.group(1))

                # Games column — usually 3rd cell (index 2) but sometimes 2nd
                games_text = None
                for cell in cells[1:4]:
                    if re.match(r'^\d+$', cell):
                        games_text = cell
                        break
                if games_text is None:
                    continue

                try:
                    g = int(games_text)
                except ValueError:
                    continue

                if is_finals:
                    finals_games += g
                else:
                    regular_games += g
                    years_active.add(year_num)

        weighted = regular_games + finals_games * FINALS_MULTIPLIER
        return {
            "regular_games": regular_games,
            "finals_games":  finals_games,
            "weighted_games": weighted,
            "years_active":  len(years_active),
            "last_year":     max(years_active) if years_active else 0,
        }
    except Exception as e:
        return {"regular_games": 0, "finals_games": 0, "weighted_games": 0,
                "years_active": 0, "last_year": 0}


def get_team_current_players(team: str, cutoff_year: int = 2023) -> list:
    """
    Scrape the AFL Tables team player index to get URLs for players
    active in or after cutoff_year.
    Returns list of (name, url) tuples.
    """
    url = TEAM_PAGES.get(team)
    if not url:
        return []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        players = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Player links look like ../../players/a/Adams_Darcy.html
            if "/players/" in href:
                full_url = "https://afltables.com/afl/" + href.replace("../../", "")
                name = a.get_text(strip=True)
                if name:
                    players.append((name, full_url))
        return players
    except Exception as e:
        print(f"Team index fetch failed for {team}: {e}")
        return []


def build_team_experience_table(year: int = None, max_players_per_team: int = 35) -> pd.DataFrame:
    """
    For each team, fetch career stats for all current-ish players and compute
    team-level experience aggregates.

    This is slow (many HTTP requests) — use caching.

    Returns DataFrame: team | avg_weighted_games | median_weighted_games |
                       pct_prime | pct_veteran | pct_developing | avg_finals_games
    """
    if year is None:
        year = datetime.now().year - 1  # use previous year as proxy for current rosters

    records = []
    for team in TEAM_PAGES:
        players = get_team_current_players(team, cutoff_year=year - 2)
        if not players:
            continue

        team_games = []
        for name, url in players[:max_players_per_team]:
            stats = fetch_player_career(url)
            if stats["last_year"] >= year - 2 and stats["regular_games"] > 0:
                stats["name"] = name
                stats["team"] = team
                team_games.append(stats)
            time.sleep(0.3)  # be polite to AFL Tables

        if not team_games:
            continue

        wg = [p["weighted_games"] for p in team_games]
        fg = [p["finals_games"] for p in team_games]

        stage_counts = {}
        for p in team_games:
            s = get_career_stage(p["weighted_games"])
            stage_counts[s] = stage_counts.get(s, 0) + 1

        n = len(team_games)
        records.append({
            "team":                  team,
            "year":                  year,
            "n_players":             n,
            "avg_weighted_games":    np.mean(wg),
            "median_weighted_games": np.median(wg),
            "avg_finals_games":      np.mean(fg),
            "pct_developing":        stage_counts.get("Developing", 0) / n,
            "pct_emerging":          stage_counts.get("Emerging",   0) / n,
            "pct_prime":             stage_counts.get("Prime",      0) / n,
            "pct_veteran":           (stage_counts.get("Veteran", 0) +
                                      stage_counts.get("Elite Veteran", 0)) / n,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── PAV-based experience (fast path using Squiggle PAV data) ──────────────────

def compute_experience_from_pav(pav_df: pd.DataFrame,
                                 games_df: pd.DataFrame,
                                 year: int) -> pd.DataFrame:
    """
    Fast-path: approximate team experience from PAV game counts already in
    the Squiggle dataset rather than scraping AFL Tables per-player pages.

    PAV data typically includes a 'games' column with games played that season.
    We accumulate across years to get career totals.

    Returns DataFrame: team | year | avg_career_games | avg_finals_games |
                       exp_diff_ready_for_model
    """
    if pav_df is None or pav_df.empty:
        return pd.DataFrame()

    pav = pav_df.copy()

    # Standardise column names
    fn_col = "firstname" if "firstname" in pav.columns else "givenname"
    if fn_col not in pav.columns or "surname" not in pav.columns:
        return pd.DataFrame()

    pav["player_key"] = pav[fn_col].str.strip().str.lower() + "_" + pav["surname"].str.strip().str.lower()

    # Games column — Squiggle PAV may call it 'games', 'gms', or similar
    games_col = next((c for c in ["games", "gms", "g"] if c in pav.columns), None)
    if not games_col:
        return pd.DataFrame()

    pav[games_col] = pd.to_numeric(pav[games_col], errors="coerce").fillna(0)

    # Sum regular-season games per player across all years in dataset
    career_totals = (pav.groupby("player_key")[games_col]
                        .sum()
                        .reset_index()
                        .rename(columns={games_col: "career_games"}))

    pav = pav.merge(career_totals, on="player_key", how="left")

    # For finals we need the games_df — identify finals rounds (round > 23 typically)
    finals_rounds = set()
    if games_df is not None and not games_df.empty:
        if "round" in games_df.columns and "roundname" in games_df.columns:
            finals_mask = games_df["roundname"].str.contains(
                "final|elim|qual|semi|prelim|grand", case=False, na=False)
            finals_rounds = set(games_df.loc[finals_mask, "round"].unique())

    # Per-year team aggregates — what was this team's experience level in year Y?
    records = []
    for yr in sorted(pav["year"].unique()):
        yr_pav = pav[pav["year"] == yr].copy()
        team_col = "team" if "team" in yr_pav.columns else None
        if not team_col:
            continue
        for team, grp in yr_pav.groupby(team_col):
            wg = grp["career_games"].values
            records.append({
                "team":               team,
                "year":               int(yr),
                "avg_career_games":   float(np.mean(wg)) if len(wg) else 0,
                "med_career_games":   float(np.median(wg)) if len(wg) else 0,
                "pct_veterans":       float(np.mean(wg >= 150)) if len(wg) else 0,
                "pct_developing":     float(np.mean(wg < 25))   if len(wg) else 0,
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Data staleness analysis ────────────────────────────────────────────────────

def analyse_data_staleness(pav_df: pd.DataFrame,
                            games_df: pd.DataFrame,
                            start_year: int = 2013) -> dict:
    """
    Analyses how much of the historical training data still reflects
    the current player pool — helps recommend an optimal training start year.

    Returns a dict with per-year stats:
      - n_players_that_year: total players in PAV that year
      - n_still_active: how many were still playing in the latest year
      - pct_still_active: overlap %
      - n_in_prime: players aged 75–149 weighted games in that year
      - recommended: True if this year balances data volume vs relevance
    """
    if pav_df is None or pav_df.empty:
        return {}

    pav = pav_df.copy()
    fn_col = "firstname" if "firstname" in pav.columns else "givenname"
    if fn_col not in pav.columns:
        return {}
    pav["player_key"] = (pav[fn_col].str.strip().str.lower() + "_" +
                          pav["surname"].str.strip().str.lower())

    latest_year = int(pav["year"].max())
    active_now = set(pav[pav["year"] == latest_year]["player_key"].unique())

    games_col = next((c for c in ["games", "gms", "g"] if c in pav.columns), None)

    result = {}
    all_years = sorted(pav["year"].unique())
    for yr in all_years:
        if yr < start_year:
            continue
        yr_pav = pav[pav["year"] == int(yr)]
        players_that_year = set(yr_pav["player_key"].unique())
        still_active = players_that_year & active_now
        n_total = len(players_that_year)
        n_active = len(still_active)

        # Count players in "prime" career stage that year
        n_prime = 0
        if games_col:
            gc = pd.to_numeric(yr_pav[games_col], errors="coerce").fillna(0)
            n_prime = int(((gc >= 75) & (gc <= 149)).sum())

        # Number of completed seasons we'd have if training from this year
        seasons_available = latest_year - int(yr)

        pct_active = n_active / n_total if n_total else 0

        result[int(yr)] = {
            "n_players":         n_total,
            "n_still_active":    n_active,
            "pct_still_active":  round(pct_active * 100, 1),
            "n_prime_players":   n_prime,
            "seasons_available": seasons_available,
            # Score: balance of recency, volume, and prime player count
            "relevance_score":   round(pct_active * 0.5 + (seasons_available / 12) * 0.3
                                       + (n_prime / max(n_total, 1)) * 0.2, 3),
        }

    # Mark recommended year as the one with the best score that also has ≥6 seasons
    valid = {yr: v for yr, v in result.items() if v["seasons_available"] >= 6}
    if valid:
        best_yr = max(valid, key=lambda y: valid[y]["relevance_score"])
        result[best_yr]["recommended"] = True

    return result
