"""
model/predictor.py  (with squad quality features added)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Feature group definitions ─────────────────────────────────────────────

BASE_FEATURES = [
    "elo_diff", "form_diff", "home_form", "away_form",
    "home_consistency", "away_consistency",
]

FATIGUE_FEATURES = [
    "travel_diff", "travel_home_km", "travel_away_km",
    "short_rest_diff", "short_rest_home", "short_rest_away",
    "bye_rest_diff",   "bye_rest_home",  "bye_rest_away",
    "days_rest_diff",  "days_rest_home", "days_rest_away",
    "travel_fatigue_diff", "home_travel_fatigue", "away_travel_fatigue",
    "travel_win_rate_diff", "travel_margin_diff",
    "perth_win_rate_diff", "is_perth_game",
]

CONTEXT_FEATURES = [
    "streak_diff", "home_streak", "away_streak",
    "last_margin_diff", "last3_diff", "last5_diff",
]

SEASON_STAT_FEATURES = [
    "cl_diff", "i50_diff", "cp_diff", "tk_diff", "ho_diff", "clanger_diff",
]

PAV_FEATURES = [
    "pav_total_diff", "pav_off_diff", "pav_def_diff", "pav_mid_diff",
]

EXPERIENCE_FEATURES = [
    "exp_avg_diff", "exp_veteran_diff", "exp_developing_diff",
]

STANDINGS_FEATURES = [
    "ladder_rank_diff", "ladder_pct_diff", "ladder_wins_diff",
]

# ── NEW: Squad quality features ────────────────────────────────────────────
SQUAD_FEATURES = [
    "squad_disp_diff",      # avg disposals diff (top-22 or named-22)
    "squad_goals_diff",     # avg goals diff
    "squad_tackles_diff",   # avg tackles diff
    "squad_hitouts_diff",   # avg hitouts diff (ruck quality)
    "squad_games_diff",     # avg career games diff (experience via selections)
]

ALL_FEATURE_GROUPS = {
    "base":         BASE_FEATURES,
    "fatigue":      FATIGUE_FEATURES,
    "context":      CONTEXT_FEATURES,
    "season_stats": SEASON_STAT_FEATURES,
    "pav":          PAV_FEATURES,
    "experience":   EXPERIENCE_FEATURES,
    "standings":    STANDINGS_FEATURES,
    "squad":        SQUAD_FEATURES,   # <-- new group
}

HOME_ADVANTAGE = 50.0
K_FACTOR       = 32
INITIAL_ELO    = 1500
PERTH_VENUES   = {"Optus Stadium", "Perth Stadium", "Subiaco Oval"}


# ── Elo helpers ────────────────────────────────────────────────────────────

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a, k=K_FACTOR):
    ea = expected_score(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


# ── Squad feature helpers ──────────────────────────────────────────────────

def _squad_row(squad_df, team):
    """Return squad quality dict for a team, or zeros if missing."""
    if squad_df is None or squad_df.empty:
        return {}
    rows = squad_df[squad_df["team"] == team]
    if rows.empty:
        return {}
    r = rows.iloc[0]
    return {
        "squad_disp":    float(r.get("squad_disp",    0)),
        "squad_goals":   float(r.get("squad_goals",   0)),
        "squad_tackles": float(r.get("squad_tackles", 0)),
        "squad_hitouts": float(r.get("squad_hitouts", 0)),
        "squad_games":   float(r.get("squad_games",   0)),
    }


def compute_squad_diffs(home_team, away_team, squad_df):
    """Return dict of squad_*_diff features. Returns zeros if data unavailable."""
    h = _squad_row(squad_df, home_team)
    a = _squad_row(squad_df, away_team)
    if not h or not a:
        return {f: 0.0 for f in SQUAD_FEATURES}
    return {
        "squad_disp_diff":    h["squad_disp"]    - a["squad_disp"],
        "squad_goals_diff":   h["squad_goals"]   - a["squad_goals"],
        "squad_tackles_diff": h["squad_tackles"] - a["squad_tackles"],
        "squad_hitouts_diff": h["squad_hitouts"] - a["squad_hitouts"],
        "squad_games_diff":   h["squad_games"]   - a["squad_games"],
    }


# ── Attach squad features to training DataFrame ────────────────────────────

def attach_squad_features(df: pd.DataFrame, squad_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prior-season squad quality features into game-level training DataFrame.

    squad_lookup must have columns:
        team, game_year, squad_disp, squad_goals, squad_tackles, squad_hitouts, squad_games
    where game_year = stat_year + 1 (stats from year N inform games in year N+1).

    Adds columns: squad_disp_diff, squad_goals_diff, squad_tackles_diff,
                  squad_hitouts_diff, squad_games_diff
    """
    if squad_lookup is None or squad_lookup.empty:
        for f in SQUAD_FEATURES:
            df[f] = 0.0
        return df

    sq_cols = ["team", "game_year", "squad_disp", "squad_goals",
               "squad_tackles", "squad_hitouts", "squad_games"]
    sq = squad_lookup[sq_cols].copy()

    # Home squad
    home_sq = sq.rename(columns={
        "team":          "hteam",
        "squad_disp":    "h_sq_disp",
        "squad_goals":   "h_sq_goals",
        "squad_tackles": "h_sq_tackles",
        "squad_hitouts": "h_sq_hitouts",
        "squad_games":   "h_sq_games",
    })
    # Away squad
    away_sq = sq.rename(columns={
        "team":          "ateam",
        "squad_disp":    "a_sq_disp",
        "squad_goals":   "a_sq_goals",
        "squad_tackles": "a_sq_tackles",
        "squad_hitouts": "a_sq_hitouts",
        "squad_games":   "a_sq_games",
    })

    df = df.merge(
        home_sq[["hteam", "game_year", "h_sq_disp", "h_sq_goals",
                 "h_sq_tackles", "h_sq_hitouts", "h_sq_games"]],
        left_on=["hteam", "year"], right_on=["hteam", "game_year"], how="left"
    ).drop(columns=["game_year"], errors="ignore")

    df = df.merge(
        away_sq[["ateam", "game_year", "a_sq_disp", "a_sq_goals",
                 "a_sq_tackles", "a_sq_hitouts", "a_sq_games"]],
        left_on=["ateam", "year"], right_on=["ateam", "game_year"], how="left"
    ).drop(columns=["game_year"], errors="ignore")

    df["squad_disp_diff"]    = df["h_sq_disp"]    - df["a_sq_disp"]
    df["squad_goals_diff"]   = df["h_sq_goals"]   - df["a_sq_goals"]
    df["squad_tackles_diff"] = df["h_sq_tackles"] - df["a_sq_tackles"]
    df["squad_hitouts_diff"] = df["h_sq_hitouts"] - df["a_sq_hitouts"]
    df["squad_games_diff"]   = df["h_sq_games"]   - df["a_sq_games"]

    for f in SQUAD_FEATURES:
        df[f] = df[f].fillna(0.0)

    # Cleanup
    drop_cols = ["h_sq_disp", "h_sq_goals", "h_sq_tackles", "h_sq_hitouts", "h_sq_games",
                 "a_sq_disp", "a_sq_goals", "a_sq_tackles", "a_sq_hitouts", "a_sq_games"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    return df


# ── build_prediction_features ─────────────────────────────────────────────

def build_prediction_features(
    home_team, away_team, venue, elo_ratings, team_stats,
    season_stats=None, lineup_pav=None, enriched_df=None,
    experience_df=None, standings_df=None,
    squad_df=None,           # ← NEW param: squad quality for this game
    home_advantage=HOME_ADVANTAGE,
    current_round=None,
) -> dict:
    """
    Build the full feature vector for a single game prediction.
    squad_df: per-team squad quality DataFrame (either named-22 or prior-season top-22).
              Columns: team, squad_disp, squad_goals, squad_tackles, squad_hitouts, squad_games
    """
    features = {}

    # ── Elo ───────────────────────────────────────────────────────────────
    home_elo = elo_ratings.get(home_team, INITIAL_ELO) + home_advantage
    away_elo = elo_ratings.get(away_team, INITIAL_ELO)
    features["elo_diff"] = home_elo - away_elo

    # ── Form / context ────────────────────────────────────────────────────
    if team_stats:
        ht = team_stats.get(home_team, {})
        at = team_stats.get(away_team, {})
        features["form_diff"]        = ht.get("form", 0)      - at.get("form", 0)
        features["home_form"]        = ht.get("form", 0)
        features["away_form"]        = at.get("form", 0)
        features["home_consistency"] = ht.get("consistency", 0)
        features["away_consistency"] = at.get("consistency", 0)
        features["streak_diff"]      = ht.get("streak", 0)    - at.get("streak", 0)
        features["home_streak"]      = ht.get("streak", 0)
        features["away_streak"]      = at.get("streak", 0)
        features["last_margin_diff"] = ht.get("last_margin", 0) - at.get("last_margin", 0)
        features["last3_diff"]       = ht.get("last3", 0)     - at.get("last3", 0)
        features["last5_diff"]       = ht.get("last5", 0)     - at.get("last5", 0)
        # Travel/rest
        features["travel_diff"]          = ht.get("travel_km", 0)         - at.get("travel_km", 0)
        features["travel_home_km"]       = ht.get("travel_km", 0)
        features["travel_away_km"]       = at.get("travel_km", 0)
        features["short_rest_diff"]      = ht.get("short_rest", 0)        - at.get("short_rest", 0)
        features["short_rest_home"]      = ht.get("short_rest", 0)
        features["short_rest_away"]      = at.get("short_rest", 0)
        features["bye_rest_diff"]        = ht.get("bye_rest", 0)          - at.get("bye_rest", 0)
        features["bye_rest_home"]        = ht.get("bye_rest", 0)
        features["bye_rest_away"]        = at.get("bye_rest", 0)
        features["days_rest_diff"]       = ht.get("days_rest", 7)         - at.get("days_rest", 7)
        features["days_rest_home"]       = ht.get("days_rest", 7)
        features["days_rest_away"]       = at.get("days_rest", 7)
        features["travel_fatigue_diff"]  = ht.get("travel_fatigue", 0)    - at.get("travel_fatigue", 0)
        features["home_travel_fatigue"]  = ht.get("travel_fatigue", 0)
        features["away_travel_fatigue"]  = at.get("travel_fatigue", 0)
        features["travel_win_rate_diff"] = ht.get("travel_win_rate", 0.5) - at.get("travel_win_rate", 0.5)
        features["travel_margin_diff"]   = ht.get("travel_margin", 0)     - at.get("travel_margin", 0)
        features["perth_win_rate_diff"]  = ht.get("perth_win_rate", 0.5)  - at.get("perth_win_rate", 0.5)
        features["is_perth_game"]        = 1 if venue in PERTH_VENUES else 0
    else:
        for f in FATIGUE_FEATURES + CONTEXT_FEATURES:
            features[f] = 0.0

    # ── Season stats ──────────────────────────────────────────────────────
    if season_stats is not None:
        hs = season_stats.get(home_team, {})
        as_ = season_stats.get(away_team, {})
        for stat in ["cl", "i50", "cp", "tk", "ho", "clanger"]:
            features[f"{stat}_diff"] = hs.get(stat, 0) - as_.get(stat, 0)
    else:
        for f in SEASON_STAT_FEATURES:
            features[f] = 0.0

    # ── PAV ───────────────────────────────────────────────────────────────
    if lineup_pav is not None:
        hp = lineup_pav.get(home_team, {})
        ap = lineup_pav.get(away_team, {})
        for k in ["pav_total", "pav_off", "pav_def", "pav_mid"]:
            features[f"{k}_diff"] = hp.get(k, 0) - ap.get(k, 0)
    else:
        for f in PAV_FEATURES:
            features[f] = 0.0

    # ── Experience ────────────────────────────────────────────────────────
    if experience_df is not None and not experience_df.empty:
        he = experience_df[experience_df["team"] == home_team]
        ae = experience_df[experience_df["team"] == away_team]
        hev = he.iloc[0] if not he.empty else {}
        aev = ae.iloc[0] if not ae.empty else {}
        def _get(row, key):
            try: return float(row[key])
            except: return 0.0
        features["exp_avg_diff"]       = _get(hev,"exp_avg")       - _get(aev,"exp_avg")
        features["exp_veteran_diff"]   = _get(hev,"exp_veteran")   - _get(aev,"exp_veteran")
        features["exp_developing_diff"]= _get(hev,"exp_developing")- _get(aev,"exp_developing")
    else:
        for f in EXPERIENCE_FEATURES:
            features[f] = 0.0

    # ── Standings / ladder ────────────────────────────────────────────────
    ladder_weight = 1.0
    if current_round is not None:
        ladder_weight = min(current_round / 8.0, 1.0)

    if standings_df is not None and not standings_df.empty:
        hs = standings_df[standings_df["team"] == home_team]
        as_ = standings_df[standings_df["team"] == away_team]
        hr = hs.iloc[0] if not hs.empty else {}
        ar = as_.iloc[0] if not as_.empty else {}
        def _gs(row, key, default=0):
            try: return float(row[key])
            except: return default
        raw_rank_diff = _gs(ar,"rank",10) - _gs(hr,"rank",10)  # lower rank = better
        features["ladder_rank_diff"] = raw_rank_diff * ladder_weight
        features["ladder_pct_diff"]  = (_gs(hr,"percentage",100) - _gs(ar,"percentage",100)) * ladder_weight
        features["ladder_wins_diff"] = (_gs(hr,"wins",0)         - _gs(ar,"wins",0))          * ladder_weight
    else:
        for f in STANDINGS_FEATURES:
            features[f] = 0.0

    # ── Squad quality (NEW) ───────────────────────────────────────────────
    squad_diffs = compute_squad_diffs(home_team, away_team, squad_df)
    features.update(squad_diffs)

    return features


# ── train_models ──────────────────────────────────────────────────────────

def train_models(df_games, start_year=2017, squad_lookup=None):
    """
    Train win classifier + margin regressor.

    Parameters
    ----------
    df_games    : enriched games DataFrame (from fetcher.py)
    start_year  : first season to include in training
    squad_lookup: DataFrame from footywire.get_all_squad_features()
                  (team, game_year, squad_disp, squad_goals, squad_tackles,
                   squad_hitouts, squad_games)
                  Pass None to train without squad features.

    Returns
    -------
    win_model, margin_model, metrics  (dict with cv scores, features_used, etc.)
    """
    df = df_games[df_games["year"] >= start_year].copy()
    df = df[df["complete"] == 1].copy()

    if len(df) < 100:
        return None, None, {"error": "Not enough training games"}

    # Build per-game features by iterating (same logic as existing code)
    # ... (existing feature engineering code) ...
    # After building df with all existing features, attach squad features:

    df = attach_squad_features(df, squad_lookup)

    # Determine which features are actually available (non-zero variance)
    candidate_features = (BASE_FEATURES + FATIGUE_FEATURES + CONTEXT_FEATURES +
                          SEASON_STAT_FEATURES + STANDINGS_FEATURES +
                          EXPERIENCE_FEATURES + SQUAD_FEATURES)
    # Drop PAV as it's mostly dead
    available = [f for f in candidate_features if f in df.columns and df[f].std() > 0]

    X = df[available].fillna(0)
    y_win    = (df["hscore"] > df["ascore"]).astype(int)
    y_margin = df["hscore"] - df["ascore"]

    win_model    = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                              learning_rate=0.05, random_state=42)
    margin_model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, random_state=42)

    cv_win    = cross_val_score(win_model, X, y_win,    cv=5, scoring="accuracy")
    cv_margin = cross_val_score(margin_model, X, y_margin, cv=5, scoring="r2")

    win_model.fit(X, y_win)
    margin_model.fit(X, y_margin)

    metrics = {
        "cv_win_accuracy":  cv_win.mean(),
        "cv_win_std":       cv_win.std(),
        "cv_margin_r2":     cv_margin.mean(),
        "n_games":          len(df),
        "features_used":    available,
        "squad_features_available": any(f in available for f in SQUAD_FEATURES),
    }

    return win_model, margin_model, metrics


# ── predict_game ──────────────────────────────────────────────────────────

def predict_game(
    home_team, away_team, venue,
    win_model, margin_model, metrics,
    elo_ratings, team_stats,
    season_stats=None, lineup_pav=None, enriched_df=None,
    experience_df=None, standings_df=None,
    squad_df=None,       # ← pass named-22 squad_df here for live predictions
    home_advantage=HOME_ADVANTAGE,
    current_round=None,
):
    """
    Predict win probability and margin for a single game.
    For live games: pass squad_df = named22_squad (from footywire.get_named22_features)
    For backtest:   squad_df is sliced from squad_lookup by game year (prior-season stats)
    """
    features = build_prediction_features(
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        elo_ratings=elo_ratings,
        team_stats=team_stats,
        season_stats=season_stats,
        lineup_pav=lineup_pav,
        enriched_df=enriched_df,
        experience_df=experience_df,
        standings_df=standings_df,
        squad_df=squad_df,
        home_advantage=home_advantage,
        current_round=current_round,
    )

    features_used = metrics.get("features_used", list(features.keys()))
    X = pd.DataFrame([{f: features.get(f, 0.0) for f in features_used}])

    win_prob = win_model.predict_proba(X)[0][1]
    margin   = margin_model.predict(X)[0]

    return {
        "win_prob":  win_prob,
        "margin":    margin,
        "features":  features,
    }
