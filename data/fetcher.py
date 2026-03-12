import requests
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
HEADERS = {"User-Agent": "AFL-Predictor/1.0 (github.com/nwowhar/afltip)"}

# ── Venue coordinates ─────────────────────────────────────────────────────────
VENUE_COORDS = {
    "MCG":                            (-37.8200, 144.9834),
    "Marvel Stadium":                 (-37.8165, 144.9476),
    "Docklands":                      (-37.8165, 144.9476),
    "Etihad Stadium":                 (-37.8165, 144.9476),
    "GMHBA Stadium":                  (-38.1577, 144.3548),
    "Kardinia Park":                  (-38.1577, 144.3548),
    "Adelaide Oval":                  (-34.9157, 138.5962),
    "Optus Stadium":                  (-31.9505, 115.8890),
    "Perth Stadium":                  (-31.9505, 115.8890),
    "Gabba":                          (-27.4858, 153.0381),
    "The Gabba":                      (-27.4858, 153.0381),
    "Cazalys Stadium":                (-16.9186, 145.7781),
    "TIO Stadium":                    (-12.4000, 130.8833),
    "Marrara Oval":                   (-12.4000, 130.8833),
    "SCG":                            (-33.8914, 151.2246),
    "Sydney Cricket Ground":          (-33.8914, 151.2246),
    "Engie Stadium":                  (-33.8473, 151.0654),
    "Spotless Stadium":               (-33.8473, 151.0654),
    "GIANTS Stadium":                 (-33.8473, 151.0654),
    "University of Tasmania Stadium": (-41.4545, 147.1348),
    "York Park":                      (-41.4545, 147.1348),
    "Bellerive Oval":                 (-42.8821, 147.3673),
    "Manuka Oval":                    (-35.3213, 149.1258),
    "TIO Traeger Park":               (-23.6980, 133.8807),
    "Alice Springs":                  (-23.6980, 133.8807),
    "Norwood Oval":                   (-34.9200, 138.6400),
    "Football Park":                  (-34.8713, 138.5080),
}

# ── Team home city coords ─────────────────────────────────────────────────────
TEAM_HOME_COORDS = {
    "Adelaide":               (-34.9157, 138.5962),
    "Brisbane Lions":         (-27.4858, 153.0381),
    "Carlton":                (-37.8200, 144.9834),
    "Collingwood":            (-37.8200, 144.9834),
    "Essendon":               (-37.8200, 144.9834),
    "Fremantle":              (-31.9505, 115.8890),
    "Geelong":                (-38.1577, 144.3548),
    "Gold Coast":             (-28.0167, 153.4000),
    "GWS Giants":             (-33.8473, 151.0654),
    "Greater Western Sydney": (-33.8473, 151.0654),
    "Hawthorn":               (-37.8200, 144.9834),
    "Melbourne":              (-37.8200, 144.9834),
    "North Melbourne":        (-37.8200, 144.9834),
    "Port Adelaide":          (-34.9157, 138.5962),
    "Richmond":               (-37.8200, 144.9834),
    "St Kilda":               (-37.8200, 144.9834),
    "Sydney":                 (-33.8914, 151.2246),
    "West Coast":             (-31.9505, 115.8890),
    "Western Bulldogs":       (-37.8200, 144.9834),
}

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_venue_coords(venue: str):
    if not venue:
        return None
    if venue in VENUE_COORDS:
        return VENUE_COORDS[venue]
    for key, coords in VENUE_COORDS.items():
        if key.lower() in venue.lower() or venue.lower() in key.lower():
            return coords
    return None

def travel_distance_km(team: str, venue: str) -> float:
    vc = get_venue_coords(venue)
    hc = TEAM_HOME_COORDS.get(team)
    if not vc or not hc:
        return 0.0
    return round(haversine_km(*hc, *vc), 1)

# ── API ───────────────────────────────────────────────────────────────────────
def get_games(year: int) -> pd.DataFrame:
    r = requests.get(f"{SQUIGGLE_BASE}?q=games;year={year}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("games", []))
    if df.empty:
        return df
    return df[df["complete"] == 100].copy()

def get_all_games(start_year: int = 2010) -> pd.DataFrame:
    frames = []
    for year in range(start_year, datetime.now().year + 1):
        try:
            df = get_games(year)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Error fetching {year}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_upcoming_games() -> pd.DataFrame:
    r = requests.get(f"{SQUIGGLE_BASE}?q=games;year={datetime.now().year}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("games", []))
    if df.empty:
        return df
    incomplete = df[df["complete"] < 100].copy()
    if incomplete.empty:
        return incomplete
    # Only return the next/current round — the lowest round number with incomplete games
    next_round = incomplete["round"].min()
    return incomplete[incomplete["round"] == next_round].copy()

def get_teams() -> list:
    r = requests.get(f"{SQUIGGLE_BASE}?q=teams", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return sorted([t["name"] for t in r.json().get("teams", [])])

# ── Enrichment ────────────────────────────────────────────────────────────────

# Perth is qualitatively different — 2,700km from Melbourne, genuine fatigue
# and acclimatisation factor. Flag it separately from generic travel.
PERTH_VENUES = {"Optus Stadium", "Perth Stadium", "Subiaco Oval"}
PERTH_TRAVEL_THRESHOLD_KM = 2000   # anything this far = Perth-tier trip
LONG_TRAVEL_KM             = 1000  # interstate but not Perth

def enrich_games(df: pd.DataFrame) -> pd.DataFrame:
    """Add travel, rest days, streak, rolling margin, and travel record features."""
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    df = df.sort_values(["date_parsed", "id"]).reset_index(drop=True)

    # ── Static travel distances ───────────────────────────────────────────────
    df["travel_home_km"] = df.apply(
        lambda r: travel_distance_km(r.get("hteam",""), r.get("venue","")), axis=1)
    df["travel_away_km"] = df.apply(
        lambda r: travel_distance_km(r.get("ateam",""), r.get("venue","")), axis=1)
    df["travel_diff"] = df["travel_home_km"] - df["travel_away_km"]

    # ── Rest days — binary flags more predictive than raw counts ─────────────
    # Raw days_rest is noisy (7 days vs 9 days doesn't matter much)
    # What matters: short turnaround (<= 6 days) and bye-length rest (>= 14 days)
    df["short_rest_home"]   = (df["days_rest_home"] <= 6).astype(float)
    df["short_rest_away"]   = (df["days_rest_away"] <= 6).astype(float)
    df["short_rest_diff"]   = df["short_rest_home"] - df["short_rest_away"]
    df["bye_rest_home"]     = (df["days_rest_home"] >= 14).astype(float)
    df["bye_rest_away"]     = (df["days_rest_away"] >= 14).astype(float)
    df["bye_rest_diff"]     = df["bye_rest_home"] - df["bye_rest_away"]
    # Keep raw diff too — useful for the fatigue interaction
    df["days_rest_diff"]    = df["days_rest_home"] - df["days_rest_away"]

    # Perth game flag: 1 if venue is Perth, useful for the model to weight
    # interstate-to-Perth travel differently from Melbourne-to-Adelaide etc.
    df["is_perth_game"] = df["venue"].apply(
        lambda v: 1 if str(v) in PERTH_VENUES else 0)

    # ── Rolling travel record (computed row-by-row to avoid leakage) ──────────
    # Track each team's win rate and avg margin on long trips (>1000km) and
    # specifically on Perth trips (>2000km), using only games seen so far.
    travel_history = {}   # team -> list of (km, margin)

    h_travel_win_rate, a_travel_win_rate = [], []
    h_travel_avg_margin, a_travel_avg_margin = [], []
    h_perth_win_rate, a_perth_win_rate = [], []

    def _travel_stats(team, min_km):
        """Win rate + avg margin for trips >= min_km, from history so far."""
        hist = [x for x in travel_history.get(team, []) if x[0] >= min_km]
        if len(hist) < 3:
            return 0.5, 0.0   # not enough data → neutral
        margins = [x[1] for x in hist]
        win_rate = sum(1 for m in margins if m > 0) / len(margins)
        avg_margin = np.mean(margins)
        return round(win_rate, 3), round(float(avg_margin), 2)

    last_date, streak, margins = {}, {}, {}
    days_rest_h, days_rest_a = [], []
    h_streak, a_streak = [], []
    h_last, a_last = [], []
    h_last3, a_last3 = [], []
    h_last5, a_last5 = [], []

    NEUTRAL_REST = 7
    BYE_CAP      = 21

    for _, row in df.iterrows():
        h, a   = row.get("hteam",""), row.get("ateam","")
        gdate  = row["date_parsed"]
        h_km   = travel_distance_km(h, row.get("venue",""))
        a_km   = travel_distance_km(a, row.get("venue",""))

        # ── Days rest ─────────────────────────────────────────────────────────
        for team, store in [(h, days_rest_h), (a, days_rest_a)]:
            if pd.notna(gdate) and team in last_date and pd.notna(last_date[team]):
                raw  = int((gdate - last_date[team]).days)
                rest = raw if raw <= BYE_CAP else NEUTRAL_REST
                store.append(rest)
            else:
                store.append(NEUTRAL_REST)

        # ── Travel record BEFORE this game ────────────────────────────────────
        h_wr, h_am  = _travel_stats(h, LONG_TRAVEL_KM)
        a_wr, a_am  = _travel_stats(a, LONG_TRAVEL_KM)
        h_pwr, _    = _travel_stats(h, PERTH_TRAVEL_THRESHOLD_KM)
        a_pwr, _    = _travel_stats(a, PERTH_TRAVEL_THRESHOLD_KM)
        h_travel_win_rate.append(h_wr)
        a_travel_win_rate.append(a_wr)
        h_travel_avg_margin.append(h_am)
        a_travel_avg_margin.append(a_am)
        h_perth_win_rate.append(h_pwr)
        a_perth_win_rate.append(a_pwr)

        # ── Pre-game rolling stats ─────────────────────────────────────────────
        for team, ss, ls, l3s, l5s in [(h, h_streak, h_last, h_last3, h_last5),
                                        (a, a_streak, a_last, a_last3, a_last5)]:
            m = margins.get(team, [])
            ss.append(streak.get(team, 0))
            ls.append(m[-1] if m else 0)
            l3s.append(round(np.mean(m[-3:]), 2) if m else 0)
            l5s.append(round(np.mean(m[-5:]), 2) if m else 0)

        # ── Update state AFTER recording pre-game features ────────────────────
        hm = (row.get("hscore",0) or 0) - (row.get("ascore",0) or 0)
        for team, margin, km in [(h, hm, h_km), (a, -hm, a_km)]:
            prev = streak.get(team, 0)
            streak[team] = (max(prev,0)+1) if margin>0 else ((min(prev,0)-1) if margin<0 else 0)
            margins.setdefault(team, []).append(margin)
            if km >= LONG_TRAVEL_KM:
                travel_history.setdefault(team, []).append((km, margin))
            if pd.notna(gdate):
                last_date[team] = gdate

    # ── Assign columns ────────────────────────────────────────────────────────
    df["days_rest_home"]  = days_rest_h
    df["days_rest_away"]  = days_rest_a
    df["days_rest_diff"]  = df["days_rest_home"] - df["days_rest_away"]

    df["home_streak"]     = h_streak
    df["away_streak"]     = a_streak
    df["streak_diff"]     = df["home_streak"] - df["away_streak"]

    df["home_last_margin"]= h_last
    df["away_last_margin"]= a_last
    df["last_margin_diff"]= df["home_last_margin"] - df["away_last_margin"]

    df["home_last3_avg"]  = h_last3
    df["away_last3_avg"]  = a_last3
    df["home_last5_avg"]  = h_last5
    df["away_last5_avg"]  = a_last5
    df["last3_diff"]      = df["home_last3_avg"] - df["away_last3_avg"]
    df["last5_diff"]      = df["home_last5_avg"] - df["away_last5_avg"]

    # Travel record features
    df["home_travel_win_rate"]   = h_travel_win_rate
    df["away_travel_win_rate"]   = a_travel_win_rate
    df["home_travel_avg_margin"] = h_travel_avg_margin
    df["away_travel_avg_margin"] = a_travel_avg_margin
    df["home_perth_win_rate"]    = h_perth_win_rate
    df["away_perth_win_rate"]    = a_perth_win_rate
    df["travel_win_rate_diff"]   = df["home_travel_win_rate"]   - df["away_travel_win_rate"]
    df["travel_margin_diff"]     = df["home_travel_avg_margin"] - df["away_travel_avg_margin"]
    df["perth_win_rate_diff"]    = df["home_perth_win_rate"]    - df["away_perth_win_rate"]

    # Interaction: long travel + short rest = genuine fatigue signal
    # Only applies to the travelling team (away more often than home)
    df["home_travel_fatigue"] = df["travel_home_km"].clip(upper=3000) / 1000 * (
        (14 - df["days_rest_home"]).clip(lower=0))
    df["away_travel_fatigue"] = df["travel_away_km"].clip(upper=3000) / 1000 * (
        (14 - df["days_rest_away"]).clip(lower=0))
    df["travel_fatigue_diff"] = df["home_travel_fatigue"] - df["away_travel_fatigue"]

    return df

def get_team_current_stats(df: pd.DataFrame) -> dict:
    """Extract each team's current form stats from an enriched df."""
    stats = {}
    for _, row in df.sort_values(["date_parsed","id"]).iterrows():
        h, a = row.get("hteam",""), row.get("ateam","")
        hm = (row.get("hscore",0) or 0) - (row.get("ascore",0) or 0)

        # Always store last_date as a scalar Timestamp, never a DatetimeArray
        raw_date = row.get("date_parsed")
        try:
            last_date = pd.Timestamp(raw_date)
            if pd.isna(last_date):
                last_date = None
        except Exception:
            last_date = None

        for team, margin in [(h, hm), (a, -hm)]:
            prev = stats.get(team, {"streak":0, "margins":[], "last_date":None})
            m = prev["margins"] + [margin]
            s = prev["streak"]
            s = (max(s,0)+1) if margin>0 else ((min(s,0)-1) if margin<0 else 0)
            stats[team] = {
                "streak": s,
                "last_margin": margin,
                "last3_avg": round(np.mean(m[-3:]),1),
                "last5_avg": round(np.mean(m[-5:]),1),
                "last5_std": round(np.std(m[-5:]),1) if len(m)>=2 else 20.0,
                "margins": m,
                "last_date": last_date,
            }
    return stats

def get_squiggle_tips(year: int = None, round_num: int = None) -> pd.DataFrame:
    """Fetch tips/predictions from all Squiggle models for a given round."""
    if year is None:
        year = datetime.now().year
    url = f"{SQUIGGLE_BASE}?q=tips;year={year}"
    if round_num is not None:
        url += f";round={round_num}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    tips = r.json().get("tips", [])
    if not tips:
        return pd.DataFrame()
    df = pd.DataFrame(tips)
    return df

def get_squiggle_consensus(year: int = None, round_num: int = None) -> pd.DataFrame:
    """
    Returns per-game consensus home win probability across all Squiggle models.
    Columns: gameid, hteam, ateam, consensus_home_prob, n_models, tips_df
    """
    tips = get_squiggle_tips(year, round_num)
    if tips.empty:
        return pd.DataFrame()

    # hconfidence is the model's confidence the HOME team wins (0-100 or 0-1)
    # normalise to 0-1
    if "hconfidence" in tips.columns:
        tips["hconfidence"] = pd.to_numeric(tips["hconfidence"], errors="coerce")
        # Some models report 0-100, some 0-1
        if tips["hconfidence"].max() > 1.5:
            tips["hconfidence"] = tips["hconfidence"] / 100.0
        tips = tips.dropna(subset=["hconfidence", "gameid"])
        consensus = (
            tips.groupby("gameid")
            .agg(
                hteam=("hteam", "first"),
                ateam=("ateam", "first"),
                consensus_home_prob=("hconfidence", "mean"),
                n_models=("hconfidence", "count"),
            )
            .reset_index()
        )
        return consensus
    return pd.DataFrame()

def get_odds_api(api_key: str) -> pd.DataFrame:
    """
    Fetch current AFL head-to-head odds from The Odds API.
    Returns df with columns: home_team, away_team, bookmaker, home_odds, away_odds
    Requires a free API key from the-odds-api.com
    """
    url = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds"
    params = {
        "apiKey": api_key,
        "regions": "au",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        games = r.json()
    except Exception as e:
        return pd.DataFrame()

    rows = []
    for game in games:
        ht = game.get("home_team", "")
        at = game.get("away_team", "")
        commence = game.get("commence_time", "")
        for bm in game.get("bookmakers", []):
            bm_name = bm.get("title", "")
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                odds_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                h_odds = odds_map.get(ht, None)
                a_odds = odds_map.get(at, None)
                if h_odds and a_odds:
                    rows.append({
                        "home_team": ht,
                        "away_team": at,
                        "commence_time": commence,
                        "bookmaker": bm_name,
                        "home_odds": float(h_odds),
                        "away_odds": float(a_odds),
                    })
    return pd.DataFrame(rows)

# AFL team name mapping — Squiggle/Odds API names don't always match our training data names
TEAM_NAME_MAP = {
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
    "Greater Western Sydney Giants": "Greater Western Sydney",
    "GWS": "Greater Western Sydney",
    "Gold Coast Suns": "Gold Coast",
    "Gold Coast": "Gold Coast",
    "West Coast Eagles": "West Coast",
    "West Coast": "West Coast",
    "St Kilda Saints": "St Kilda",
    "North Melbourne Kangaroos": "North Melbourne",
    "Adelaide Crows": "Adelaide",
    "Geelong Cats": "Geelong",
    "Sydney Swans": "Sydney",
    "Collingwood Magpies": "Collingwood",
    "Melbourne Demons": "Melbourne",
    "Hawthorn Hawks": "Hawthorn",
    "Richmond Tigers": "Richmond",
    "Carlton Blues": "Carlton",
    "Essendon Bombers": "Essendon",
    "Fremantle Dockers": "Fremantle",
    "Western Bulldogs": "Western Bulldogs",
    "Port Adelaide Power": "Port Adelaide",
    "Port Adelaide": "Port Adelaide",
}

def normalise_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)
