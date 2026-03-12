"""
data/footywire.py
-----------------
Scrapes FootyWire for:
  1. Player season stats by year (for squad quality backtest features)
  2. Current round team selections (for live named-22 features)

Squad quality approach:
  - Backtest: use PREVIOUS season's top-22 players (by games played) per team
  - Live prediction: use actual named 22 from current selections page
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.footywire.com/afl/footy"
HEADERS = {"User-Agent": "Mozilla/5.0 (AFL Predictor Research Bot)"}

# Map FootyWire team names → canonical names used in rest of app
FW_TEAM_MAP = {
    "Adelaide": "Adelaide",
    "Brisbane": "Brisbane",
    "Brisbane Lions": "Brisbane",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS": "GWS",
    "Greater Western Sydney": "GWS",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    # short names on selections page
    "Blues": "Carlton",
    "Bombers": "Essendon",
    "Bulldogs": "Western Bulldogs",
    "Cats": "Geelong",
    "Crows": "Adelaide",
    "Demons": "Melbourne",
    "Dockers": "Fremantle",
    "Eagles": "West Coast",
    "Giants": "GWS",
    "Hawks": "Hawthorn",
    "Kangaroos": "North Melbourne",
    "Lions": "Brisbane",
    "Magpies": "Collingwood",
    "Power": "Port Adelaide",
    "Saints": "St Kilda",
    "Suns": "Gold Coast",
    "Swans": "Sydney",
    "Tigers": "Richmond",
}


# ---------------------------------------------------------------------------
# 1. Player season stats
# ---------------------------------------------------------------------------

def get_player_stats(year: int, sleep: float = 0.5) -> pd.DataFrame:
    """
    Scrape FootyWire player rankings (League Totals) for a given year.
    Returns DataFrame with columns:
        player, team, games, kicks, handballs, disposals, goals, tackles, hitouts
    Only players with games > 0 are included.
    """
    stats = []
    stat_cols = ["kicks", "handballs", "disposals", "marks", "goals",
                 "behinds", "tackles", "frees_for", "frees_against", "hitouts"]

    for stat in ["disposals", "goals", "tackles", "hitouts", "kicks"]:
        url = f"{BASE}/ft_player_rankings?year={year}&rt=LT&pos=all&sort={stat}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", {"class": re.compile(r"sortable|data", re.I)})
            if table is None:
                # fallback: find any table with Rank header
                for t in soup.find_all("table"):
                    if t.find("th") and "Rank" in t.find("th").get_text():
                        table = t
                        break
            if table is None:
                continue

            rows = table.find_all("tr")[1:]  # skip header
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < 5:
                    continue
                # typical cols: Rank | Player | Team | Games | LastGame | Average
                try:
                    player_name = cells[1]
                    team_raw = cells[2]
                    games = int(cells[3])
                    avg = float(cells[5]) if len(cells) > 5 else float(cells[4])
                    team = FW_TEAM_MAP.get(team_raw, team_raw)
                    stats.append({
                        "player": player_name,
                        "team": team,
                        "games": games,
                        "stat": stat,
                        "avg": avg,
                        "year": year,
                    })
                except (ValueError, IndexError):
                    continue
            time.sleep(sleep)
        except Exception as e:
            print(f"[footywire] Failed to fetch {stat} for {year}: {e}")
            continue

    if not stats:
        return pd.DataFrame()

    df = pd.DataFrame(stats)
    # pivot so each player×year has one row with each stat as a column
    pivot = df.pivot_table(
        index=["player", "team", "year"],
        columns="stat",
        values="avg",
        aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    # merge in games (take max across stat pages to handle partial data)
    games_df = df.groupby(["player", "team", "year"])["games"].max().reset_index()
    pivot = pivot.merge(games_df, on=["player", "team", "year"], how="left")

    # ensure all expected stat cols exist
    for col in ["disposals", "goals", "tackles", "hitouts", "kicks"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    return pivot.fillna(0)


def get_squad_features_for_year(year: int) -> pd.DataFrame:
    """
    For a given season year, return one row per team summarising squad quality.
    Uses top 22 players by games played.
    
    Returns DataFrame with columns:
        team, squad_disp, squad_goals, squad_tackles, squad_hitouts, squad_games
    """
    players = get_player_stats(year)
    if players.empty:
        return pd.DataFrame()

    rows = []
    for team, grp in players.groupby("team"):
        top22 = grp.nlargest(22, "games")
        rows.append({
            "team": team,
            "year": year,
            "squad_disp": top22["disposals"].mean(),
            "squad_goals": top22["goals"].mean(),
            "squad_tackles": top22["tackles"].mean(),
            "squad_hitouts": top22["hitouts"].mean(),
            "squad_games": top22["games"].mean(),  # avg experience proxy
        })

    return pd.DataFrame(rows)


def get_all_squad_features(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Build squad feature lookup for all years in [start_year, end_year].
    Used to attach previous-season squad quality to each training game.
    Returns DataFrame indexed by (team, year) — where year is the GAME year
    (i.e. stats are from year-1).
    """
    all_rows = []
    for yr in range(start_year - 1, end_year + 1):
        print(f"[footywire] Fetching player stats for {yr}...")
        feats = get_squad_features_for_year(yr)
        if not feats.empty:
            feats["game_year"] = yr + 1  # these stats inform next year's games
            all_rows.append(feats)
        time.sleep(1)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined  # columns: team, year (stat year), game_year, squad_*


# ---------------------------------------------------------------------------
# 2. Current round team selections
# ---------------------------------------------------------------------------

def get_current_selections() -> dict:
    """
    Scrape FootyWire's current team selections page.
    Returns dict: { team_name: [player_name, ...] }
    Only includes named 22 + interchange (not emergencies).
    """
    url = f"{BASE}/afl_team_selections"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[footywire] Failed to fetch selections: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    selections = {}

    # Each game block has a header like "Sydney v Carlton (SCG)"
    # Teams appear as links with href matching pp-{team-slug}--{player-slug}
    # We detect team boundaries by looking for the position table structure

    # Find all player links grouped by their team context
    # The page structure alternates home/away in a two-column table
    game_headers = soup.find_all(string=re.compile(r" v "))

    for header_text in game_headers:
        header_text = str(header_text).strip()
        # Extract team names from "Team A v Team B (Venue)"
        match = re.match(r"^(.+?) v (.+?)\s*\(", header_text)
        if not match:
            continue

    # More robust: extract by team slug from all player links
    team_players = {}
    for link in soup.find_all("a", href=re.compile(r"^pp-")):
        href = link["href"]  # e.g. pp-sydney-swans--sam-walsh
        player_name = link.get_text(strip=True)
        if not player_name:
            continue

        # Extract team slug (between pp- and --)
        slug_match = re.match(r"pp-(.+?)--", href)
        if not slug_match:
            continue
        team_slug = slug_match.group(1)
        team = _slug_to_team(team_slug)

        if team not in team_players:
            team_players[team] = []
        if player_name not in team_players[team]:
            team_players[team].append(player_name)

    return team_players


def get_named22_features(selections: dict, player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Given current selections (team -> [player names]) and a player stats DataFrame
    (from get_player_stats for the previous/current season), compute squad quality
    features for the named 22 of each team.

    Returns DataFrame with columns: team, squad_disp, squad_goals, squad_tackles,
    squad_hitouts, squad_games
    """
    if player_stats.empty or not selections:
        return pd.DataFrame()

    rows = []
    for team, players in selections.items():
        team_stats = player_stats[player_stats["team"] == team]
        # Match named players by name (fuzzy: lowercase strip)
        team_stats = team_stats.copy()
        team_stats["player_lower"] = team_stats["player"].str.lower().str.strip()
        named_lower = [p.lower().strip() for p in players]

        named_stats = team_stats[team_stats["player_lower"].isin(named_lower)]

        # Fallback: if fewer than 15 matched, pad with top unmatched by games
        if len(named_stats) < 15:
            unmatched = team_stats[~team_stats["player_lower"].isin(named_lower)]
            pad = unmatched.nlargest(22 - len(named_stats), "games")
            named_stats = pd.concat([named_stats, pad])

        if named_stats.empty:
            continue

        rows.append({
            "team": team,
            "squad_disp": named_stats["disposals"].mean(),
            "squad_goals": named_stats["goals"].mean(),
            "squad_tackles": named_stats["tackles"].mean(),
            "squad_hitouts": named_stats["hitouts"].mean(),
            "squad_games": named_stats["games"].mean(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug_to_team(slug: str) -> str:
    """Convert FootyWire team URL slug to canonical team name."""
    slug_map = {
        "sydney-swans": "Sydney",
        "carlton-blues": "Carlton",
        "gold-coast-suns": "Gold Coast",
        "geelong-cats": "Geelong",
        "greater-western-sydney-giants": "GWS",
        "hawthorn-hawks": "Hawthorn",
        "brisbane-lions": "Brisbane",
        "western-bulldogs": "Western Bulldogs",
        "st-kilda-saints": "St Kilda",
        "collingwood-magpies": "Collingwood",
        "richmond-tigers": "Richmond",
        "melbourne-demons": "Melbourne",
        "essendon-bombers": "Essendon",
        "fremantle-dockers": "Fremantle",
        "west-coast-eagles": "West Coast",
        "port-adelaide-power": "Port Adelaide",
        "adelaide-crows": "Adelaide",
        "north-melbourne-kangaroos": "North Melbourne",
        "kangaroos": "North Melbourne",
    }
    return slug_map.get(slug, slug.replace("-", " ").title())


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing squad features for 2024...")
    feats = get_squad_features_for_year(2024)
    print(feats.to_string())

    print("\nTesting current selections...")
    sel = get_current_selections()
    for team, players in list(sel.items())[:3]:
        print(f"  {team}: {players[:5]}...")
