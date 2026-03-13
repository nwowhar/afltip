"""
Predictor v5 — adds playing-style matchup features derived from AFL Tables season stats.

Feature groups:
  1. Elo differential
  2. Rolling form (last 5 game avg margin, consistency)
  3. Travel distance + rest days + fatigue interaction
  4. Win/loss streak + last margin rolling averages
  5. Season stats (clearances, inside 50s, contested possessions, tackles, hitouts, clangers)
  6. PAV lineup strength differential (when lineups available)
  7. Player experience differential (career games, veteran %, developing %)
  8. Ladder position + percentage + wins (fade-in by round)
  9. Playing style matchup (kick ratio, tackle rate, hitout rate, mark rate — from season totals)
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
    "travel_diff", "travel_home_km", "travel_away_km",
    "short_rest_diff", "short_rest_home", "short_rest_away",
    "bye_rest_diff",   "bye_rest_home",   "bye_rest_away",
    "days_rest_diff",  "days_rest_home",  "days_rest_away",
    "travel_fatigue_diff", "home_travel_fatigue", "away_travel_fatigue",
    "travel_win_rate_diff", "travel_margin_diff",
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
    "exp_avg_diff",
    "exp_veteran_diff",
    "exp_developing_diff",
]

STANDINGS_FEATURES = [
    "ladder_rank_diff",
    "ladder_pct_diff",
    "ladder_wins_diff",
]

STYLE_FEATURES = [
    "kick_ratio_diff",   # kick-heavy vs handball-heavy style gap
    "tackle_diff",       # pressure / contested game intensity
    "hitout_diff",       # ruck dominance
    "mark_diff",         # aerial vs ground game
    "kick_vs_tackle",    # interaction: kick style vs pressure defence
    "ruck_advantage",    # explicit ruck label (= hitout_diff)
]

ALL_FEATURES = (BASE_FEATURES + FATIGUE_FEATURES +
                CONTEXT_FEATURES + SEASON_STAT_FEATURES +
                PAV_FEATURES + EXPERIENCE_FEATURES +
                STANDINGS_FEATURES + STYLE_FEATURES)

CORE_FEATURES = (BASE_FEATURES + FATIGUE_FEATURES +
                 CONTEXT_FEATURES + SEASON_STAT_FEATURES +
                 EXPERIENCE_FEATURES + STANDINGS_FEATURES +
                 STYLE_FEATURES)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build rolling form features from a sorted games DataFrame."""
    df = df.copy()
    sort_col = "date_parsed" if "date_parsed" in df.columns else "id"
    df = df.sort_values([sort_col, "id"]).reset_index(drop=True)

    df["margin"]   = (df["hscore"].fillna(0) - df["ascore"].fillna(0)).astype(float)
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
    """Merge season-level team stats into game rows as differential features."""
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
            hv = lookup.get((h, year)) or lookup.get((h, year - 1)) or 0
            av = lookup.get((a, year)) or lookup.get((a, year - 1)) or 0
            h_vals.append(float(hv))
            a_vals.append(float(av))

        df[diff_col] = np.array(h_vals) - np.array(a_vals)

    return df


def add_pav_features(df: pd.DataFrame, pav_df: pd.DataFrame) -> pd.DataFrame:
    """Add PAV features to training data using season-level totals."""
    if pav_df is None or pav_df.empty:
        for col in PAV_FEATURES:
            df[col] = 0.0
        return df

    df = df.copy()
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
    """Add team experience differential features to game rows."""
    for col in EXPERIENCE_FEATURES:
        df[col] = 0.0

    if exp_df is None or exp_df.empty:
        return df

    df = df.copy()
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


def add_standings_features(df: pd.DataFrame,
                             standings_df: pd.DataFrame) -> pd.DataFrame:
    """Add ladder position, percentage and wins differential as features."""
    for col in STANDINGS_FEATURES:
        df[col] = 0.0

    if standings_df is None or standings_df.empty:
        return df

    df = df.copy()
    lookup = {}
    pct_col = "percentage" if "percentage" in standings_df.columns else "pct"
    for _, row in standings_df.iterrows():
        lookup[(str(row["team"]), int(row["year"]))] = row.to_dict()

    h_rank, a_rank = [], []
    h_pct,  a_pct  = [], []
    h_wins, a_wins = [], []

    for _, row in df.iterrows():
        year = int(row.get("year", 0))
        h = str(row.get("hteam", ""))
        a = str(row.get("ateam", ""))
        hd = lookup.get((h, year)) or lookup.get((h, year - 1)) or {}
        ad = lookup.get((a, year)) or lookup.get((a, year - 1)) or {}
        h_rank.append(float(hd.get("rank", 9) or 9))
        a_rank.append(float(ad.get("rank", 9) or 9))
        h_pct.append(float(hd.get(pct_col, 100) or 100))
        a_pct.append(float(ad.get(pct_col, 100) or 100))
        h_wins.append(float(hd.get("wins", 0) or 0))
        a_wins.append(float(ad.get("wins", 0) or 0))

    df["ladder_rank_diff"] = np.array(h_rank) - np.array(a_rank)
    df["ladder_pct_diff"]  = np.array(h_pct)  - np.array(a_pct)
    df["ladder_wins_diff"] = np.array(h_wins) - np.array(a_wins)
    return df


def add_style_features(df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add playing-style matchup features to game rows.
    Delegates to data.team_style.attach_style_features() which handles
    leakage-safe year-1 lookups for training data.
    """
    try:
        from data.team_style import attach_style_features
        return attach_style_features(df, style_df)
    except ImportError:
        for f in STYLE_FEATURES:
            df[f] = 0.0
        return df


# ── Model training ────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, feature_set: list = None) -> tuple:
    """
    Train win probability and margin models.
    Returns (win_model, margin_model, metrics, feature_importance_df)
    metrics['features_used'] contains the exact ordered list used for training.
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

    # Elo-only baseline for comparison
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
        "features_used":    available,  # ← exact ordered list the model was trained on
    }

    return win_model, margin_model, metrics, fi_df


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_game(win_model, margin_model,
                 features: dict,
                 feature_set: list = None) -> dict:
    """
    Predict a single game from a feature dict.
    Pass feature_set=metrics['features_used'] for guaranteed compatibility.
    """
    if feature_set is None:
        try:
            n = win_model.named_steps["scaler"].n_features_in_
            feature_set = [f for f in CORE_FEATURES if f in features][:n]
        except Exception:
            feature_set = [f for f in CORE_FEATURES if f in features]

    X = np.array([[float(features.get(f, 0.0)) for f in feature_set]])
    win_prob         = win_model.predict_proba(X)[0][1]
    predicted_margin = margin_model.predict(X)[0]
    return {
        "home_win_prob":    round(win_prob * 100, 1),
        "away_win_prob":    round((1 - win_prob) * 100, 1),
        "predicted_margin": round(predicted_margin, 1),
    }


# ── Feature builder for live predictions ─────────────────────────────────────

def build_prediction_features(home_team: str, away_team: str,
                               venue: str,
                               elo_ratings: dict,
                               team_stats: dict,
                               season_stats: pd.DataFrame = None,
                               lineup_pav: dict = None,
                               enriched_df: pd.DataFrame = None,
                               experience_df: pd.DataFrame = None,
                               standings_df: pd.DataFrame = None,
                               style_df: pd.DataFrame = None,
                               home_advantage: float = 50.0,
                               current_round: int = None) -> dict:
    """Build a full feature dict for an upcoming game prediction."""
    from data.fetcher import (travel_distance_km, PERTH_VENUES,
                               LONG_TRAVEL_KM, PERTH_TRAVEL_THRESHOLD_KM)
    from data.team_style import compute_style_matchup
    import pandas as _pd

    def _f(v, default=0.0):
        try:
            val = v
            if hasattr(val, "iloc"):  val = val.iloc[-1]
            elif hasattr(val, "item"): val = val.item()
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
            ld = last_date
            if hasattr(ld, "iloc"):
                ld = ld.iloc[-1]
            elif hasattr(ld, "__len__") and not isinstance(ld, str):
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

    h_fatigue = min(h_travel, 3000) / 1000 * max(14 - h_rest, 0)
    a_fatigue = min(a_travel, 3000) / 1000 * max(14 - a_rest, 0)

    def get_travel_record(team, min_km):
        if enriched_df is None or enriched_df.empty:
            return 0.5, 0.0
        away_trips = enriched_df[
            (enriched_df["ateam"] == team) &
            (enriched_df["travel_away_km"] >= min_km)
        ]
        home_trips = enriched_df[
            (enriched_df["hteam"] == team) &
            (enriched_df["travel_home_km"] >= min_km)
        ]
        margins = []
        for _, r in away_trips.iterrows():
            margins.append((r.get("ascore", 0) or 0) - (r.get("hscore", 0) or 0))
        for _, r in home_trips.iterrows():
            margins.append((r.get("hscore", 0) or 0) - (r.get("ascore", 0) or 0))
        if len(margins) < 3:
            return 0.5, 0.0
        win_rate   = sum(1 for m in margins if m > 0) / len(margins)
        avg_margin = float(np.mean(margins))
        return round(win_rate, 3), round(avg_margin, 2)

    h_twr, h_tam = get_travel_record(home_team, LONG_TRAVEL_KM)
    a_twr, a_tam = get_travel_record(away_team, LONG_TRAVEL_KM)
    h_pwr, _     = get_travel_record(home_team, PERTH_TRAVEL_THRESHOLD_KM)
    a_pwr, _     = get_travel_record(away_team, PERTH_TRAVEL_THRESHOLD_KM)

    from datetime import datetime as _dt
    year = _dt.now().year

    # Season stats source selection:
    #   Rounds 1-5  → use PREVIOUS year stats only (current season too small a sample)
    #   Round 6+    → use current year stats (enough games to be meaningful)
    # This avoids one blowout game inflating a team's season averages
    _use_prev_stats = (current_round is not None and current_round < 6)

    def ss(team, stat):
        if season_stats is None or season_stats.empty:
            return 0
        if _use_prev_stats:
            row = season_stats[(season_stats["team"] == team) & (season_stats["year"] == year - 1)]
        else:
            row = season_stats[(season_stats["team"] == team) & (season_stats["year"] == year)]
            if row.empty:
                row = season_stats[(season_stats["team"] == team) & (season_stats["year"] == year - 1)]
        return float(row.iloc[0].get(stat, 0)) if not row.empty else 0

    # Form fade-in: early season form data is noisy (only 1-2 games)
    # Round 1=20%, Round 2=40%, Round 3=60%, Round 4=80%, Round 5+=100%
    _form_weight = min(current_round / 5.0, 1.0) if current_round else 1.0

    feats = {
        # Elo
        "elo_diff":               h_elo - a_elo + float(home_advantage),
        # Form — faded in early season so 1 game doesn't override large Elo gaps
        "form_diff":              (h_form - a_form) * _form_weight,
        "home_form":              h_form * _form_weight,
        "away_form":              a_form * _form_weight,
        "home_consistency":       h_std,
        "away_consistency":       a_std,
        # Travel
        "travel_diff":            h_travel - a_travel,
        "travel_home_km":         h_travel,
        "travel_away_km":         a_travel,
        "travel_fatigue_diff":    h_fatigue - a_fatigue,
        "home_travel_fatigue":    h_fatigue,
        "away_travel_fatigue":    a_fatigue,
        "travel_win_rate_diff":   h_twr - a_twr,
        "travel_margin_diff":     h_tam - a_tam,
        "perth_win_rate_diff":    h_pwr - a_pwr,
        "is_perth_game":          is_perth,
        # Rest
        "days_rest_diff":         h_rest - a_rest,
        "days_rest_home":         float(h_rest),
        "days_rest_away":         float(a_rest),
        "short_rest_home":        1.0 if h_rest <= 6 else 0.0,
        "short_rest_away":        1.0 if a_rest <= 6 else 0.0,
        "short_rest_diff":        (1.0 if h_rest <= 6 else 0.0) - (1.0 if a_rest <= 6 else 0.0),
        "bye_rest_home":          1.0 if h_rest >= 14 else 0.0,
        "bye_rest_away":          1.0 if a_rest >= 14 else 0.0,
        "bye_rest_diff":          (1.0 if h_rest >= 14 else 0.0) - (1.0 if a_rest >= 14 else 0.0),
        # Streak — faded same as form
        "streak_diff":            (_f(hs.get("streak", 0)) - _f(as_.get("streak", 0))) * _form_weight,
        "home_streak":            _f(hs.get("streak", 0)) * _form_weight,
        "away_streak":            _f(as_.get("streak", 0)) * _form_weight,
        # Last margins — faded same as form
        "last_margin_diff":       (_f(hs.get("last_margin", 0)) - _f(as_.get("last_margin", 0))) * _form_weight,
        "last3_diff":             (_f(hs.get("last3_avg", 0))   - _f(as_.get("last3_avg", 0))) * _form_weight,
        "last5_diff":             (h_form - a_form) * _form_weight,
        # Season stats
        "cl_diff":      ss(home_team, "avg_clearances")            - ss(away_team, "avg_clearances"),
        "i50_diff":     ss(home_team, "avg_inside_50s")            - ss(away_team, "avg_inside_50s"),
        "cp_diff":      ss(home_team, "avg_contested_possessions") - ss(away_team, "avg_contested_possessions"),
        "tk_diff":      ss(home_team, "avg_tackles")               - ss(away_team, "avg_tackles"),
        "ho_diff":      ss(home_team, "avg_hitouts")               - ss(away_team, "avg_hitouts"),
        "clanger_diff": ss(home_team, "avg_clangers")              - ss(away_team, "avg_clangers"),
        # PAV (0 until lineups announced)
        "pav_total_diff": 0.0,
        "pav_off_diff":   0.0,
        "pav_def_diff":   0.0,
        "pav_mid_diff":   0.0,
        # Style features (0 if style_df unavailable)
        "kick_ratio_diff": 0.0,
        "tackle_diff":     0.0,
        "hitout_diff":     0.0,
        "mark_diff":       0.0,
        "kick_vs_tackle":  0.0,
        "ruck_advantage":  0.0,
    }

    # PAV override if lineups announced
    if lineup_pav:
        from data.lineup import get_lineup_pav_diff
        pav_feats = get_lineup_pav_diff(home_team, away_team, lineup_pav)
        feats.update(pav_feats)

    # Experience features
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
        feats["exp_avg_diff"]        = _exp(home_team, "avg_career_games") - _exp(away_team, "avg_career_games")
        feats["exp_veteran_diff"]    = _exp(home_team, "pct_veterans")     - _exp(away_team, "pct_veterans")
        feats["exp_developing_diff"] = _exp(home_team, "pct_developing")   - _exp(away_team, "pct_developing")

    # Standings / ladder features
    feats["ladder_rank_diff"] = 0.0
    feats["ladder_pct_diff"]  = 0.0
    feats["ladder_wins_diff"] = 0.0
    if standings_df is not None and not standings_df.empty:
        from datetime import datetime as _dt3
        yr = _dt3.now().year
        pct_col = "percentage" if "percentage" in standings_df.columns else "pct"
        def _st(team, col, default):
            row = standings_df[(standings_df["team"] == team) &
                               (standings_df["year"] == yr)]
            if row.empty:
                row = standings_df[(standings_df["team"] == team) &
                                   (standings_df["year"] == yr - 1)]
            return float(row.iloc[0][col]) if not row.empty and col in row.columns else default
        feats["ladder_rank_diff"] = _st(home_team, "rank", 9)     - _st(away_team, "rank", 9)
        feats["ladder_pct_diff"]  = _st(home_team, pct_col, 100)  - _st(away_team, pct_col, 100)
        feats["ladder_wins_diff"] = _st(home_team, "wins", 0)     - _st(away_team, "wins", 0)

    # Ladder fade-in: 0% until R3, 15% R3-R5, 25% R6-R9, 100% R10+
    if current_round is not None:
        if current_round < 3:
            ladder_weight = 0.0
        elif current_round <= 5:
            ladder_weight = 0.15
        elif current_round <= 9:
            ladder_weight = 0.25
        else:
            ladder_weight = 1.0
        feats["ladder_rank_diff"] *= ladder_weight
        feats["ladder_pct_diff"]  *= ladder_weight
        feats["ladder_wins_diff"] *= ladder_weight

    # Style matchup features
    # Use prev-season stats in Rounds 1-3 (not enough current-season data yet)
    if style_df is not None and not style_df.empty:
        try:
            from data.team_style import compute_style_matchup
            use_prev = (current_round is not None and current_round <= 3)
            style_feats = compute_style_matchup(
                home_team, away_team, style_df,
                game_year=year,
                use_prev_season=use_prev,
            )
            feats.update(style_feats)
        except ImportError:
            pass

    return feats
