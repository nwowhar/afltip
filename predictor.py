"""
Predictor v4 — all inputs cast to float to prevent DatetimeArray errors.

  1. Elo differential
  2. Rolling form (last 5 games avg margin, consistency)
  3. Travel distance differential
  4. Rest days differential
  5. Win/loss streak
  6. Last game margin + rolling 3/5 game averages
  7. Season stats (clearances, inside 50s, contested possessions, tackles)
  8. PAV lineup strength differential (when lineups available)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ── Feature sets ──────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "elo_diff",
    "form_diff", "home_form", "away_form",
    "home_consistency", "away_consistency",
]

FATIGUE_FEATURES = [
    # Raw distance (still useful for neutral venue games)
    "travel_diff", "travel_home_km", "travel_away_km",
    # Rest
    "days_rest_diff", "days_rest_home", "days_rest_away",
    # Travel × rest interaction — the killer combo
    "travel_fatigue_diff", "home_travel_fatigue", "away_travel_fatigue",
    # Rolling travel record — does this team actually travel well?
    "travel_win_rate_diff", "travel_margin_diff",
    # Perth-specific — qualitatively different from other interstate trips
    "perth_win_rate_diff", "is_perth_game",
]

CONTEXT_FEATURES = [
    "streak_diff", "home_streak", "away_streak",
    "last_margin_diff", "last3_diff", "last5_diff",
]

SEASON_STAT_FEATURES = [
    "cl_diff",       # clearances
    "i50_diff",      # inside 50s
    "cp_diff",       # contested possessions
    "tk_diff",       # tackles
    "ho_diff",       # hitouts
    "clanger_diff",  # clangers (negative = better)
]

PAV_FEATURES = [
    "pav_total_diff",
    "pav_off_diff",
    "pav_def_diff",
    "pav_mid_diff",
]

EXPERIENCE_FEATURES = [
    "exp_avg_diff",       # avg career games diff (home - away)
    "exp_veteran_diff",   # % veterans diff
    "exp_developing_diff",# % developing players diff (negative = better)
]

ALL_FEATURES = (BASE_FEATURES + FATIGUE_FEATURES +
                CONTEXT_FEATURES + SEASON_STAT_FEATURES +
                PAV_FEATURES + EXPERIENCE_FEATURES)

# Use these as the working set (PAV only available when lineups announced)
CORE_FEATURES = (BASE_FEATURES + FATIGUE_FEATURES +
                 CONTEXT_FEATURES + SEASON_STAT_FEATURES + EXPERIENCE_FEATURES)


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build rolling form features from a sorted games DataFrame.
    Expects enriched df from fetcher.enrich_games() with date_parsed column.
    """
    df = df.copy()
    sort_col = "date_parsed" if "date_parsed" in df.columns else "id"
    df = df.sort_values([sort_col, "id"]).reset_index(drop=True)

    df["margin"] = (df["hscore"].fillna(0) - df["ascore"].fillna(0)).astype(float)
    df["home_win"] = (df["margin"] > 0).astype(int)

    team_history = {}
    home_form_list, away_form_list = [], []
    home_cons_list, away_cons_list = [], []

    for _, row in df.iterrows():
        h, a = row.get("hteam", ""), row.get("ateam", "")
        hh = team_history.get(h, [])
        ah = team_history.get(a, [])
        home_form_list.append(np.mean(hh[-window:]) if hh else 0.0)
        away_form_list.append(np.mean(ah[-window:]) if ah else 0.0)
        home_cons_list.append(np.std(hh[-window:]) if len(hh) >= 2 else 30.0)
        away_cons_list.append(np.std(ah[-window:]) if len(ah) >= 2 else 30.0)
        team_history.setdefault(h, []).append(row["margin"])
        team_history.setdefault(a, []).append(-row["margin"])

    df["home_form"]        = home_form_list
    df["away_form"]        = away_form_list
    df["home_consistency"] = home_cons_list
    df["away_consistency"] = away_cons_list
    df["form_diff"]        = df["home_form"] - df["away_form"]
    return df


def add_season_stat_features(df: pd.DataFrame,
                              season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge season-level team stats into game rows as differential features.
    """
    if season_stats is None or season_stats.empty:
        for col in SEASON_STAT_FEATURES:
            df[col] = 0.0
        return df

    df = df.copy()

    stat_map = {
        "cl_diff":      "avg_clearances",
        "i50_diff":     "avg_inside_50s",
        "cp_diff":      "avg_contested_possessions",
        "tk_diff":      "avg_tackles",
        "ho_diff":      "avg_hitouts",
        "clanger_diff": "avg_clangers",
    }

    for diff_col, stat_col in stat_map.items():
        if stat_col not in season_stats.columns:
            df[diff_col] = 0.0
            continue

        # Build lookup dict: (team, year) -> stat value
        lookup = {
            (row["team"], int(row["year"])): row[stat_col]
            for _, row in season_stats.iterrows()
            if pd.notna(row.get(stat_col))
        }

        h_vals, a_vals = [], []
        for _, row in df.iterrows():
            year = int(row.get("year", 0))
            h = row.get("hteam", "")
            a = row.get("ateam", "")
            # Try current year, then previous year
            hv = lookup.get((h, year)) or lookup.get((h, year - 1)) or 0
            av = lookup.get((a, year)) or lookup.get((a, year - 1)) or 0
            h_vals.append(float(hv))
            a_vals.append(float(av))

        df[diff_col] = np.array(h_vals) - np.array(a_vals)

    return df


def add_pav_features(df: pd.DataFrame, pav_df: pd.DataFrame) -> pd.DataFrame:
    """Add PAV features to training data by matching player stats to games."""
    # For training data we don't have per-game lineups easily,
    # so we use season-level PAV totals as a proxy
    if pav_df is None or pav_df.empty:
        for col in PAV_FEATURES:
            df[col] = 0.0
        return df

    df = df.copy()

    # Aggregate PAV by team and year
    pav_numeric = pav_df.copy()
    for col in ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]:
        if col in pav_numeric.columns:
            pav_numeric[col] = pd.to_numeric(pav_numeric[col], errors="coerce").fillna(0)

    if "team" not in pav_numeric.columns or "year" not in pav_numeric.columns:
        for col in PAV_FEATURES:
            df[col] = 0.0
        return df

    team_pav = pav_numeric.groupby(["team", "year"])[
        [c for c in ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]
         if c in pav_numeric.columns]
    ].sum().reset_index()

    lookup = {}
    for _, row in team_pav.iterrows():
        lookup[(str(row["team"]), int(row["year"]))] = row.to_dict()

    pav_map = {
        "pav_total_diff": "PAV_total",
        "pav_off_diff":   "PAV_off",
        "pav_def_diff":   "PAV_def",
        "pav_mid_diff":   "PAV_mid",
    }

    for diff_col, pav_col in pav_map.items():
        h_vals, a_vals = [], []
        for _, row in df.iterrows():
            year = int(row.get("year", 0))
            h = str(row.get("hteam", ""))
            a = str(row.get("ateam", ""))
            hv = (lookup.get((h, year)) or lookup.get((h, year-1)) or {}).get(pav_col, 0)
            av = (lookup.get((a, year)) or lookup.get((a, year-1)) or {}).get(pav_col, 0)
            h_vals.append(float(hv or 0))
            a_vals.append(float(av or 0))
        df[diff_col] = np.array(h_vals) - np.array(a_vals)

    return df


def add_experience_features(df: pd.DataFrame, exp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team experience differential features to game rows.
    exp_df should have columns: team, year, avg_career_games, pct_veterans, pct_developing
    from experience.compute_experience_from_pav().
    """
    for col in EXPERIENCE_FEATURES:
        df[col] = 0.0

    if exp_df is None or exp_df.empty:
        return df

    df = df.copy()

    # Build lookup: (team, year) -> experience stats
    lookup = {}
    for _, row in exp_df.iterrows():
        lookup[(str(row["team"]), int(row["year"]))] = row.to_dict()

    h_avg, a_avg = [], []
    h_vet, a_vet = [], []
    h_dev, a_dev = [], []

    for _, row in df.iterrows():
        year = int(row.get("year", 0))
        h = str(row.get("hteam", ""))
        a = str(row.get("ateam", ""))
        hd = lookup.get((h, year)) or lookup.get((h, year - 1)) or {}
        ad = lookup.get((a, year)) or lookup.get((a, year - 1)) or {}
        h_avg.append(float(hd.get("avg_career_games", 0) or 0))
        a_avg.append(float(ad.get("avg_career_games", 0) or 0))
        h_vet.append(float(hd.get("pct_veterans",    0) or 0))
        a_vet.append(float(ad.get("pct_veterans",    0) or 0))
        h_dev.append(float(hd.get("pct_developing",  0) or 0))
        a_dev.append(float(ad.get("pct_developing",  0) or 0))

    df["exp_avg_diff"]        = np.array(h_avg) - np.array(a_avg)
    df["exp_veteran_diff"]    = np.array(h_vet) - np.array(a_vet)
    df["exp_developing_diff"] = np.array(h_dev) - np.array(a_dev)

    return df


def train_models(df: pd.DataFrame,
                 feature_set: list = None) -> tuple:
    """
    Train win probability and margin models.
    Returns (win_model, margin_model, metrics, feature_importance_df)
    """
    if feature_set is None:
        feature_set = CORE_FEATURES

    available = [f for f in feature_set if f in df.columns]
    base_avail = [f for f in BASE_FEATURES if f in df.columns]

    clean = df.dropna(subset=available + ["home_win", "margin"])
    if len(clean) < 50:
        raise ValueError(f"Not enough clean rows to train: {len(clean)}")

    X      = clean[available].values
    y_win  = clean["home_win"].values
    y_marg = clean["margin"].values

    # Win model
    win_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        ))
    ])
    win_cv = cross_val_score(win_model, X, y_win, cv=5, scoring="accuracy")
    win_model.fit(X, y_win)

    # Margin model
    margin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        ))
    ])
    margin_cv = cross_val_score(margin_model, X, y_marg, cv=5, scoring="r2")
    margin_model.fit(X, y_marg)

    # Base model comparison
    if set(base_avail) != set(available) and len(base_avail) >= 2:
        X_base = clean[base_avail].values
        base_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            ))
        ])
        base_cv = cross_val_score(base_model, X_base, y_win, cv=5, scoring="accuracy")
        base_acc = base_cv.mean()
    else:
        base_acc = win_cv.mean()

    # Feature importance
    clf = win_model.named_steps["clf"]
    fi_df = pd.DataFrame({
        "feature":    available,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    group_lookup = {}
    from model.backtest import FEATURE_GROUPS
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            group_lookup[f] = grp
    fi_df["group"] = fi_df["feature"].map(lambda f: group_lookup.get(f, "Other"))

    metrics = {
        "win_accuracy":     win_cv.mean(),
        "win_accuracy_std": win_cv.std(),
        "base_accuracy":    base_acc,
        "accuracy_gain":    win_cv.mean() - base_acc,
        "margin_r2":        margin_cv.mean(),
        "margin_r2_std":    margin_cv.std(),
        "n_games":          len(clean),
        "n_features":       len(available),
        "features_used":    available,
    }

    return win_model, margin_model, metrics, fi_df


def predict_game(win_model, margin_model,
                 features: dict,
                 feature_set: list = None) -> dict:
    """Predict a single game from a feature dict."""
    if feature_set is None:
        feature_set = CORE_FEATURES
    available = [f for f in feature_set if f in features]
    X = np.array([[features.get(f, 0) for f in available]])
    win_prob         = win_model.predict_proba(X)[0][1]
    predicted_margin = margin_model.predict(X)[0]
    return {
        "home_win_prob":    round(win_prob * 100, 1),
        "away_win_prob":    round((1 - win_prob) * 100, 1),
        "predicted_margin": round(predicted_margin, 1),
    }


def build_prediction_features(home_team: str, away_team: str,
                               venue: str,
                               elo_ratings: dict,
                               team_stats: dict,
                               season_stats: pd.DataFrame = None,
                               lineup_pav: dict = None,
                               enriched_df: pd.DataFrame = None,
                               experience_df: pd.DataFrame = None,
                               home_advantage: float = 50.0) -> dict:
    """Build a full feature dict for an upcoming game prediction."""
    from data.fetcher import (travel_distance_km, PERTH_VENUES,
                               LONG_TRAVEL_KM, PERTH_TRAVEL_THRESHOLD_KM)
    import pandas as _pd

    # Force everything to plain Python floats — guards against pandas
    # Series/DatetimeArray leaking in from cached DataFrames
    def _f(v, default=0.0):
        try:
            val = v
            if hasattr(val, 'iloc'):  val = val.iloc[-1]
            elif hasattr(val, 'item'): val = val.item()
            return float(val)
        except Exception:
            return float(default)

    h_elo = _f(elo_ratings.get(home_team, 1500), 1500)
    a_elo = _f(elo_ratings.get(away_team, 1500), 1500)

    hs  = team_stats.get(home_team, {})
    as_ = team_stats.get(away_team, {})

    h_form = _f(hs.get("last5_avg", 0))
    a_form = _f(as_.get("last5_avg", 0))
    h_std  = _f(hs.get("last5_std", 20), 20)
    a_std  = _f(as_.get("last5_std", 20), 20)

    h_travel = travel_distance_km(home_team, venue)
    a_travel = travel_distance_km(away_team, venue)

    is_perth = 1 if str(venue) in PERTH_VENUES else 0

    today = _pd.Timestamp.now()
    NEUTRAL_REST = 7
    BYE_CAP      = 21

    def calc_rest(last_date):
        try:
            if last_date is None:
                return NEUTRAL_REST
            # Unwrap Series, DatetimeArray, or list to a scalar
            ld = last_date
            if hasattr(ld, 'iloc'):
                ld = ld.iloc[-1]
            elif hasattr(ld, '__len__') and not isinstance(ld, str):
                ld = ld[-1]
            ld = _pd.Timestamp(ld)
            if _pd.isna(ld):
                return NEUTRAL_REST
            raw = int((today - ld).days)
            return raw if raw <= BYE_CAP else NEUTRAL_REST
        except Exception:
            return NEUTRAL_REST

    h_rest = calc_rest(hs.get("last_date"))
    a_rest = calc_rest(as_.get("last_date"))

    # Travel × rest fatigue interaction
    h_fatigue = min(h_travel, 3000) / 1000 * max(14 - h_rest, 0)
    a_fatigue = min(a_travel, 3000) / 1000 * max(14 - a_rest, 0)

    # Rolling travel record from enriched historical df
    def get_travel_record(team, min_km):
        """Win rate and avg margin for trips >= min_km from enriched_df."""
        if enriched_df is None or enriched_df.empty:
            return 0.5, 0.0
        # Away games where this team travelled >= min_km
        away_trips = enriched_df[
            (enriched_df["ateam"] == team) &
            (enriched_df["travel_away_km"] >= min_km)
        ]
        # Home games where this team travelled >= min_km (neutral venues)
        home_trips = enriched_df[
            (enriched_df["hteam"] == team) &
            (enriched_df["travel_home_km"] >= min_km)
        ]
        margins = []
        for _, r in away_trips.iterrows():
            margins.append((r.get("ascore",0) or 0) - (r.get("hscore",0) or 0))
        for _, r in home_trips.iterrows():
            margins.append((r.get("hscore",0) or 0) - (r.get("ascore",0) or 0))
        if len(margins) < 3:
            return 0.5, 0.0
        win_rate   = sum(1 for m in margins if m > 0) / len(margins)
        avg_margin = float(np.mean(margins))
        return round(win_rate, 3), round(avg_margin, 2)

    h_twr, h_tam  = get_travel_record(home_team, LONG_TRAVEL_KM)
    a_twr, a_tam  = get_travel_record(away_team, LONG_TRAVEL_KM)
    h_pwr, _      = get_travel_record(home_team, PERTH_TRAVEL_THRESHOLD_KM)
    a_pwr, _      = get_travel_record(away_team, PERTH_TRAVEL_THRESHOLD_KM)

    # Season stats diff
    from datetime import datetime as _dt
    year = _dt.now().year
    def ss(team, stat):
        if season_stats is None or season_stats.empty:
            return 0
        row = season_stats[(season_stats["team"] == team) &
                           (season_stats["year"] == year)]
        if row.empty:
            row = season_stats[(season_stats["team"] == team) &
                               (season_stats["year"] == year - 1)]
        return float(row.iloc[0].get(stat, 0)) if not row.empty else 0

    feats = {
        # Elo
        "elo_diff":               h_elo - a_elo + float(home_advantage),
        # Form
        "form_diff":              h_form - a_form,
        "home_form":              h_form,
        "away_form":              a_form,
        "home_consistency":       h_std,
        "away_consistency":       a_std,
        # Travel raw
        "travel_diff":            h_travel - a_travel,
        "travel_home_km":         h_travel,
        "travel_away_km":         a_travel,
        # Travel × rest fatigue
        "travel_fatigue_diff":    h_fatigue - a_fatigue,
        "home_travel_fatigue":    h_fatigue,
        "away_travel_fatigue":    a_fatigue,
        # Travel record
        "travel_win_rate_diff":   h_twr - a_twr,
        "travel_margin_diff":     h_tam - a_tam,
        "perth_win_rate_diff":    h_pwr - a_pwr,
        "is_perth_game":          is_perth,
        # Rest
        "days_rest_diff":         h_rest - a_rest,
        "days_rest_home":         h_rest,
        "days_rest_away":         a_rest,
        # Streak
        "streak_diff":            hs.get("streak", 0) - as_.get("streak", 0),
        "home_streak":            hs.get("streak", 0),
        "away_streak":            as_.get("streak", 0),
        # Last margins
        "last_margin_diff":       hs.get("last_margin", 0) - as_.get("last_margin", 0),
        "last3_diff":             hs.get("last3_avg", 0)   - as_.get("last3_avg", 0),
        "last5_diff":             h_form - a_form,
        # Season stats
        "cl_diff":      ss(home_team, "avg_clearances")            - ss(away_team, "avg_clearances"),
        "i50_diff":     ss(home_team, "avg_inside_50s")            - ss(away_team, "avg_inside_50s"),
        "cp_diff":      ss(home_team, "avg_contested_possessions") - ss(away_team, "avg_contested_possessions"),
        "tk_diff":      ss(home_team, "avg_tackles")               - ss(away_team, "avg_tackles"),
        "ho_diff":      ss(home_team, "avg_hitouts")               - ss(away_team, "avg_hitouts"),
        "clanger_diff": ss(home_team, "avg_clangers")              - ss(away_team, "avg_clangers"),
        # PAV (0 if lineups not announced)
        "pav_total_diff": 0,
        "pav_off_diff":   0,
        "pav_def_diff":   0,
        "pav_mid_diff":   0,
    }

    # Override PAV if lineup data available
    if lineup_pav:
        from data.lineup import get_lineup_pav_diff
        pav_feats = get_lineup_pav_diff(home_team, away_team, lineup_pav)
        feats.update(pav_feats)

    # Experience features from PAV career data
    feats["exp_avg_diff"]        = 0.0
    feats["exp_veteran_diff"]    = 0.0
    feats["exp_developing_diff"] = 0.0
    if experience_df is not None and not experience_df.empty:
        from datetime import datetime as _dt2
        yr = _dt2.now().year
        def _exp(team, col):
            row = experience_df[(experience_df["team"] == team) &
                                (experience_df["year"] == yr)]
            if row.empty:
                row = experience_df[(experience_df["team"] == team) &
                                    (experience_df["year"] == yr - 1)]
            return float(row.iloc[0][col]) if not row.empty and col in row.columns else 0.0
        feats["exp_avg_diff"]        = _exp(home_team, "avg_career_games")   - _exp(away_team, "avg_career_games")
        feats["exp_veteran_diff"]    = _exp(home_team, "pct_veterans")       - _exp(away_team, "pct_veterans")
        feats["exp_developing_diff"] = _exp(home_team, "pct_developing")     - _exp(away_team, "pct_developing")

    return feats
