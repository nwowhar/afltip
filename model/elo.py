import pandas as pd
import numpy as np

INITIAL_ELO = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 50  # Elo points added to home team

def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected win probability for team A vs team B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a: float, elo_b: float, result: float, k: float = K_FACTOR):
    """
    Update Elo ratings after a game.
    result: 1 if A won, 0.5 if draw, 0 if B won.
    Returns (new_elo_a, new_elo_b)
    """
    exp = expected_score(elo_a, elo_b)
    new_a = elo_a + k * (result - exp)
    new_b = elo_b + k * ((1 - result) - (1 - exp))
    return new_a, new_b

def build_elo_ratings(games_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Build rolling Elo ratings from historical games.
    Returns:
        - games_df with pre-game Elo columns added
        - final elo dict {team: elo}
    """
    df = games_df.copy()

    # Ensure sorted by date
    df = df.sort_values(["year", "round", "id"]).reset_index(drop=True)

    elo_ratings = {}
    home_elos_pre = []
    away_elos_pre = []

    for _, row in df.iterrows():
        home = str(row["hteam"]).strip()
        away = str(row["ateam"]).strip()

        # Skip bad rows
        if not home or not away or home == "nan" or away == "nan":
            home_elos_pre.append(INITIAL_ELO)
            away_elos_pre.append(INITIAL_ELO)
            continue

        # Initialise if needed
        if home not in elo_ratings:
            elo_ratings[home] = float(INITIAL_ELO)
        if away not in elo_ratings:
            elo_ratings[away] = float(INITIAL_ELO)

        # Record pre-game Elos — always floats
        home_elo = float(elo_ratings[home]) + HOME_ADVANTAGE
        away_elo = float(elo_ratings[away])

        home_elos_pre.append(float(elo_ratings[home]))
        away_elos_pre.append(float(elo_ratings[away]))

        # Determine result
        hscore = float(row.get("hscore", 0) or 0)
        ascore = float(row.get("ascore", 0) or 0)
        if hscore > ascore:
            result = 1.0
        elif hscore < ascore:
            result = 0.0
        else:
            result = 0.5

        # Update
        new_home, new_away = update_elo(home_elo, away_elo, result)
        elo_ratings[home] = float(new_home) - HOME_ADVANTAGE
        elo_ratings[away] = float(new_away)

    df["home_elo_pre"] = home_elos_pre
    df["away_elo_pre"] = away_elos_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"] + HOME_ADVANTAGE

    return df, elo_ratings

def regress_elos_to_mean(elo_dict: dict, regress_factor: float = 0.3) -> dict:
    """Regress Elos toward the mean at season start (standard practice)."""
    # Force all values to float first — guards against any non-numeric contamination
    clean = {team: float(elo) for team, elo in elo_dict.items()
             if isinstance(elo, (int, float, np.floating, np.integer))}
    mean = np.mean(list(clean.values()))
    return {team: float(elo + regress_factor * (mean - elo))
            for team, elo in clean.items()}

def win_probability_from_elo(home_elo: float, away_elo: float) -> float:
    """Win probability for home team including home advantage."""
    return expected_score(home_elo + HOME_ADVANTAGE, away_elo)


# ── Offensive / Defensive Elo (ODELO) ────────────────────────────────────────
# Based on Holy Grail Ratings' ODELO system.
# Each team has separate attack and defence ratings representing how many
# points above/below the league average they score/concede.
# Expected margin = ((home_att - away_def) + (home_def - away_att)) / 2

ODELO_INITIAL   = 0.0    # Start at 0 = league average
ODELO_K         = 8.0    # Update speed (smaller than regular Elo — margins are noisier)
ODELO_REGRESS   = 0.3    # Regress to mean between seasons (same as regular Elo)
AVG_SCORE       = 85.0   # Historical AFL average score per team per game


def build_odelo_ratings(games_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Build offensive and defensive Elo ratings from historical games.
    
    Attack rating: how many pts above average the team scores
    Defence rating: how many pts above average the team concedes (positive = good defence)
    
    Returns:
        - games_df with pre-game ODELO columns added
        - final dict {team: {"att": float, "def": float}}
    """
    df = games_df.copy()
    df = df.sort_values(["year", "round", "id"]).reset_index(drop=True)

    ratings  = {}   # {team: {"att": float, "def": float}}
    prev_year = None

    h_att_pre, h_def_pre, a_att_pre, a_def_pre = [], [], [], []

    for _, row in df.iterrows():
        home = str(row["hteam"]).strip()
        away = str(row["ateam"]).strip()
        year = int(row.get("year", 0))

        if not home or not away or home == "nan" or away == "nan":
            for lst in [h_att_pre, h_def_pre, a_att_pre, a_def_pre]:
                lst.append(ODELO_INITIAL)
            continue

        # Season rollover — regress to mean
        if prev_year is not None and year != prev_year:
            for team in ratings:
                ratings[team]["att"] *= (1 - ODELO_REGRESS)
                ratings[team]["def"] *= (1 - ODELO_REGRESS)
        prev_year = year

        # Initialise new teams
        for team in [home, away]:
            if team not in ratings:
                ratings[team] = {"att": ODELO_INITIAL, "def": ODELO_INITIAL}

        # Record pre-game ratings
        h_att_pre.append(ratings[home]["att"])
        h_def_pre.append(ratings[home]["def"])
        a_att_pre.append(ratings[away]["att"])
        a_def_pre.append(ratings[away]["def"])

        # Actual scores
        hscore = float(row.get("hscore", 0) or 0)
        ascore = float(row.get("ascore", 0) or 0)

        # Expected scores from ODELO
        exp_home_score = AVG_SCORE + ratings[home]["att"] - ratings[away]["def"]
        exp_away_score = AVG_SCORE + ratings[away]["att"] - ratings[home]["def"]

        # Update attack: did we score more than expected?
        ratings[home]["att"] += ODELO_K * (hscore - exp_home_score) / AVG_SCORE
        ratings[away]["att"] += ODELO_K * (ascore - exp_away_score) / AVG_SCORE

        # Update defence: did we concede less than expected? (positive = good)
        ratings[home]["def"] += ODELO_K * (exp_away_score - ascore) / AVG_SCORE
        ratings[away]["def"] += ODELO_K * (exp_home_score - hscore) / AVG_SCORE

    df["h_att_elo"] = h_att_pre
    df["h_def_elo"] = h_def_pre
    df["a_att_elo"] = a_att_pre
    df["a_def_elo"] = a_def_pre

    # ODELO predicted margin differential (home perspective)
    df["odelo_diff"] = (
        (df["h_att_elo"] - df["a_def_elo"]) +
        (df["h_def_elo"] - df["a_att_elo"])
    ) / 2

    return df, ratings


def regress_odelo_to_mean(odelo_dict: dict, regress_factor: float = ODELO_REGRESS) -> dict:
    """Regress ODELO ratings toward zero (league average) at season start."""
    return {
        team: {
            "att": float(v["att"] * (1 - regress_factor)),
            "def": float(v["def"] * (1 - regress_factor)),
        }
        for team, v in odelo_dict.items()
    }
