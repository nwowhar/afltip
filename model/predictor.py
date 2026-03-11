import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Feature sets ──────────────────────────────────────────────────────────────
# Base features (Elo + rolling form)
BASE_FEATURES = [
    "elo_diff",
    "form_diff",
    "home_form",
    "away_form",
    "home_consistency",
    "away_consistency",
]

# Extended features (fatigue + context)
EXTENDED_FEATURES = BASE_FEATURES + [
    "travel_diff",
    "travel_home_km",
    "travel_away_km",
    "days_rest_diff",
    "days_rest_home",
    "days_rest_away",
    "streak_diff",
    "home_streak",
    "away_streak",
    "last_margin_diff",
    "last3_diff",
    "last5_diff",
]

FEATURES = EXTENDED_FEATURES  # Use extended by default

def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build rolling form features. Expects enriched df from fetcher.enrich_games().
    Adds home_form, away_form, home_consistency, away_consistency, form_diff.
    """
    df = df.copy().sort_values(["date_parsed" if "date_parsed" in df.columns else "id", "id"]).reset_index(drop=True)
    df["margin"] = (df["hscore"].fillna(0) - df["ascore"].fillna(0)).astype(float)
    df["home_win"] = (df["margin"] > 0).astype(int)

    team_history = {}
    home_form, away_form = [], []
    home_cons, away_cons = [], []

    for _, row in df.iterrows():
        h, a = row.get("hteam",""), row.get("ateam","")
        hh = team_history.get(h, [])
        ah = team_history.get(a, [])

        home_form.append(np.mean(hh[-window:]) if hh else 0.0)
        away_form.append(np.mean(ah[-window:]) if ah else 0.0)
        home_cons.append(np.std(hh[-window:]) if len(hh)>=2 else 30.0)
        away_cons.append(np.std(ah[-window:]) if len(ah)>=2 else 30.0)

        margin = row["margin"]
        team_history.setdefault(h, []).append(margin)
        team_history.setdefault(a, []).append(-margin)

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["home_consistency"] = home_cons
    df["away_consistency"] = away_cons
    df["form_diff"] = df["home_form"] - df["away_form"]

    return df

def train_models(df: pd.DataFrame, feature_set: list = None) -> tuple:
    """
    Train win + margin models. Returns (win_model, margin_model, metrics, feature_importances).
    Also computes accuracy with BASE_FEATURES only for comparison.
    """
    if feature_set is None:
        feature_set = FEATURES

    # Use only features that exist in df
    available = [f for f in feature_set if f in df.columns]
    base_available = [f for f in BASE_FEATURES if f in df.columns]

    df = df.dropna(subset=available + ["home_win", "margin"])
    X = df[available].values
    y_win = df["home_win"].values
    y_margin = df["margin"].values

    # ── Win model ──
    win_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42))
    ])
    win_cv = cross_val_score(win_model, X, y_win, cv=5, scoring="accuracy")
    win_model.fit(X, y_win)

    # ── Margin model ──
    margin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42))
    ])
    margin_cv = cross_val_score(margin_model, X, y_margin, cv=5, scoring="r2")
    margin_model.fit(X, y_margin)

    # ── Base model for comparison ──
    if base_available and base_available != available:
        X_base = df[base_available].values
        base_win = Pipeline([("scaler", StandardScaler()),
                             ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42))])
        base_cv = cross_val_score(base_win, X_base, y_win, cv=5, scoring="accuracy")
        base_accuracy = base_cv.mean()
    else:
        base_accuracy = win_cv.mean()

    # ── Feature importance ──
    clf = win_model.named_steps["clf"]
    importances = clf.feature_importances_
    feat_importance = pd.DataFrame({
        "feature": available,
        "importance": importances
    }).sort_values("importance", ascending=False)

    metrics = {
        "win_accuracy": win_cv.mean(),
        "win_accuracy_std": win_cv.std(),
        "base_accuracy": base_accuracy,
        "accuracy_gain": win_cv.mean() - base_accuracy,
        "margin_r2": margin_cv.mean(),
        "margin_r2_std": margin_cv.std(),
        "n_games": len(df),
        "n_features": len(available),
        "features_used": available,
    }

    return win_model, margin_model, metrics, feat_importance

def predict_game(win_model, margin_model, features: dict, feature_set: list = None) -> dict:
    """
    Predict a game given a feature dict.
    features should contain keys matching FEATURES list.
    """
    if feature_set is None:
        feature_set = FEATURES
    available = [f for f in feature_set if f in features]
    X = np.array([[features.get(f, 0) for f in available]])

    win_prob = win_model.predict_proba(X)[0][1]
    predicted_margin = margin_model.predict(X)[0]

    return {
        "home_win_prob": round(win_prob * 100, 1),
        "away_win_prob": round((1 - win_prob) * 100, 1),
        "predicted_margin": round(predicted_margin, 1),
    }

def build_prediction_features(home_team: str, away_team: str, venue: str,
                               elo_ratings: dict, team_stats: dict,
                               home_advantage: float = 50.0) -> dict:
    """
    Build a full feature dict for an upcoming game prediction.
    """
    from data.fetcher import travel_distance_km
    import datetime

    h_elo = elo_ratings.get(home_team, 1500)
    a_elo = elo_ratings.get(away_team, 1500)
    elo_diff = h_elo - a_elo + home_advantage

    hs = team_stats.get(home_team, {})
    as_ = team_stats.get(away_team, {})

    h_form = hs.get("last5_avg", 0)
    a_form = as_.get("last5_avg", 0)
    h_std  = hs.get("last5_std", 20)
    a_std  = as_.get("last5_std", 20)

    h_travel = travel_distance_km(home_team, venue)
    a_travel = travel_distance_km(away_team, venue)

    # Days rest: estimate from last game date
    today = pd.Timestamp.now()
    h_rest = int((today - hs["last_date"]).days) if hs.get("last_date") and pd.notna(hs.get("last_date")) else 7
    a_rest = int((today - as_["last_date"]).days) if as_.get("last_date") and pd.notna(as_.get("last_date")) else 7
    h_rest = min(h_rest, 21)
    a_rest = min(a_rest, 21)

    return {
        "elo_diff":         elo_diff,
        "form_diff":        h_form - a_form,
        "home_form":        h_form,
        "away_form":        a_form,
        "home_consistency": h_std,
        "away_consistency": a_std,
        "travel_diff":      h_travel - a_travel,
        "travel_home_km":   h_travel,
        "travel_away_km":   a_travel,
        "days_rest_diff":   h_rest - a_rest,
        "days_rest_home":   h_rest,
        "days_rest_away":   a_rest,
        "streak_diff":      hs.get("streak", 0) - as_.get("streak", 0),
        "home_streak":      hs.get("streak", 0),
        "away_streak":      as_.get("streak", 0),
        "last_margin_diff": hs.get("last_margin", 0) - as_.get("last_margin", 0),
        "last3_diff":       hs.get("last3_avg", 0) - as_.get("last3_avg", 0),
        "last5_diff":       h_form - a_form,
    }
