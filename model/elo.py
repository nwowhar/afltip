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
        home = row["hteam"]
        away = row["ateam"]

        # Initialise if needed (regress toward mean each new season)
        if home not in elo_ratings:
            elo_ratings[home] = INITIAL_ELO
        if away not in elo_ratings:
            elo_ratings[away] = INITIAL_ELO

        # Record pre-game Elos
        home_elo = elo_ratings[home] + HOME_ADVANTAGE
        away_elo = elo_ratings[away]

        home_elos_pre.append(elo_ratings[home])
        away_elos_pre.append(elo_ratings[away])

        # Determine result
        hscore = row.get("hscore", 0) or 0
        ascore = row.get("ascore", 0) or 0
        if hscore > ascore:
            result = 1.0
        elif hscore < ascore:
            result = 0.0
        else:
            result = 0.5

        # Update
        new_home, new_away = update_elo(home_elo, away_elo, result)
        elo_ratings[home] = new_home - HOME_ADVANTAGE  # Remove home adjustment after update
        elo_ratings[away] = new_away

    df["home_elo_pre"] = home_elos_pre
    df["away_elo_pre"] = away_elos_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"] + HOME_ADVANTAGE

    return df, elo_ratings

def regress_elos_to_mean(elo_dict: dict, regress_factor: float = 0.3) -> dict:
    """Regress Elos toward the mean at season start (standard practice)."""
    mean = np.mean(list(elo_dict.values()))
    return {team: elo + regress_factor * (mean - elo) for team, elo in elo_dict.items()}

def win_probability_from_elo(home_elo: float, away_elo: float) -> float:
    """Win probability for home team including home advantage."""
    return expected_score(home_elo + HOME_ADVANTAGE, away_elo)
