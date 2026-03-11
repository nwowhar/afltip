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
    return df[df["complete"] < 100].copy()

def get_teams() -> list:
    r = requests.get(f"{SQUIGGLE_BASE}?q=teams", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return sorted([t["name"] for t in r.json().get("teams", [])])

# ── Enrichment ────────────────────────────────────────────────────────────────
def enrich_games(df: pd.DataFrame) -> pd.DataFrame:
    """Add travel, rest days, streak, and rolling margin features."""
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    df = df.sort_values(["date_parsed", "id"]).reset_index(drop=True)

    # Travel
    df["travel_home_km"] = df.apply(lambda r: travel_distance_km(r.get("hteam",""), r.get("venue","")), axis=1)
    df["travel_away_km"] = df.apply(lambda r: travel_distance_km(r.get("ateam",""), r.get("venue","")), axis=1)
    df["travel_diff"] = df["travel_home_km"] - df["travel_away_km"]

    last_date, streak, margins = {}, {}, {}
    days_rest_h, days_rest_a = [], []
    h_streak, a_streak = [], []
    h_last, a_last = [], []
    h_last3, a_last3 = [], []
    h_last5, a_last5 = [], []

    for _, row in df.iterrows():
        h, a = row.get("hteam",""), row.get("ateam","")
        gdate = row["date_parsed"]

        # Days rest
        for team, store in [(h, days_rest_h), (a, days_rest_a)]:
            if pd.notna(gdate) and team in last_date and pd.notna(last_date[team]):
                store.append(min(int((gdate - last_date[team]).days), 21))
            else:
                store.append(7)

        # Pre-game stats
        for team, ss, ls, l3s, l5s in [(h, h_streak, h_last, h_last3, h_last5),
                                        (a, a_streak, a_last, a_last3, a_last5)]:
            m = margins.get(team, [])
            ss.append(streak.get(team, 0))
            ls.append(m[-1] if m else 0)
            l3s.append(round(np.mean(m[-3:]), 2) if m else 0)
            l5s.append(round(np.mean(m[-5:]), 2) if m else 0)

        # Update state
        hm = (row.get("hscore",0) or 0) - (row.get("ascore",0) or 0)
        for team, margin in [(h, hm), (a, -hm)]:
            prev = streak.get(team, 0)
            streak[team] = (max(prev,0)+1) if margin>0 else ((min(prev,0)-1) if margin<0 else 0)
            margins.setdefault(team, []).append(margin)
            if pd.notna(gdate):
                last_date[team] = gdate

    df["days_rest_home"] = days_rest_h
    df["days_rest_away"] = days_rest_a
    df["days_rest_diff"] = df["days_rest_home"] - df["days_rest_away"]
    df["home_streak"] = h_streak
    df["away_streak"] = a_streak
    df["streak_diff"] = df["home_streak"] - df["away_streak"]
    df["home_last_margin"] = h_last
    df["away_last_margin"] = a_last
    df["last_margin_diff"] = df["home_last_margin"] - df["away_last_margin"]
    df["home_last3_avg"] = h_last3
    df["away_last3_avg"] = a_last3
    df["home_last5_avg"] = h_last5
    df["away_last5_avg"] = a_last5
    df["last3_diff"] = df["home_last3_avg"] - df["away_last3_avg"]
    df["last5_diff"] = df["home_last5_avg"] - df["away_last5_avg"]

    return df

def get_team_current_stats(df: pd.DataFrame) -> dict:
    """Extract each team's current form stats from an enriched df."""
    stats = {}
    for _, row in df.sort_values(["date_parsed","id"]).iterrows():
        h, a = row.get("hteam",""), row.get("ateam","")
        hm = (row.get("hscore",0) or 0) - (row.get("ascore",0) or 0)
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
                "last_date": row.get("date_parsed"),
            }
    return stats
