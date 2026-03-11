import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")

def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Engineer rolling features per team for model training.
    Requires columns: hteam, ateam, hscore, ascore, home_elo_pre, away_elo_pre, year, round
    """
    df = df.copy().sort_values(["year", "round", "id"]).reset_index(drop=True)

    df["margin"] = df["hscore"] - df["ascore"]
    df["home_win"] = (df["margin"] > 0).astype(int)

    # Rolling avg margin per team
    home_rolling = {}
    away_rolling = {}

    for team in pd.concat([df["hteam"], df["ateam"]]).unique():
        home_games = df[df["hteam"] == team]["margin"].values
        away_games = (-df[df["ateam"] == team]["margin"]).values  # Flip for away

    # Simpler: compute rolling avg margin for each row using prior N games
    team_history = {}  # team -> list of margins (from their perspective)

    home_form = []
    away_form = []
    home_consistency = []
    away_consistency = []

    for _, row in df.iterrows():
        h, a = row["hteam"], row["ateam"]

        h_hist = team_history.get(h, [])
        a_hist = team_history.get(a, [])

        h_form = np.mean(h_hist[-window:]) if h_hist else 0.0
        a_form = np.mean(a_hist[-window:]) if a_hist else 0.0
        h_std = np.std(h_hist[-window:]) if len(h_hist) >= 2 else 30.0
        a_std = np.std(a_hist[-window:]) if len(a_hist) >= 2 else 30.0

        home_form.append(h_form)
        away_form.append(a_form)
        home_consistency.append(h_std)
        away_consistency.append(a_std)

        # Update history
        margin = row["hscore"] - row["ascore"]
        team_history.setdefault(h, []).append(margin)
        team_history.setdefault(a, []).append(-margin)

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["home_consistency"] = home_consistency
    df["away_consistency"] = away_consistency
    df["form_diff"] = df["home_form"] - df["away_form"]

    return df

FEATURES = ["elo_diff", "form_diff", "home_form", "away_form", "home_consistency", "away_consistency"]

def train_models(df: pd.DataFrame):
    """Train win probability and margin models. Returns (win_model, margin_model, metrics)."""
    df = df.dropna(subset=FEATURES + ["home_win", "margin"])

    X = df[FEATURES].values
    y_win = df["home_win"].values
    y_margin = df["margin"].values

    # Win probability model
    win_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    ])
    win_scores = cross_val_score(win_model, X, y_win, cv=5, scoring="accuracy")
    win_model.fit(X, y_win)

    # Margin model
    margin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
    ])
    margin_scores = cross_val_score(margin_model, X, y_margin, cv=5, scoring="r2")
    margin_model.fit(X, y_margin)

    metrics = {
        "win_accuracy": win_scores.mean(),
        "win_accuracy_std": win_scores.std(),
        "margin_r2": margin_scores.mean(),
        "margin_r2_std": margin_scores.std(),
        "n_games": len(df)
    }

    return win_model, margin_model, metrics

def predict_game(win_model, margin_model, home_elo: float, away_elo: float,
                 home_form: float, away_form: float,
                 home_consistency: float = 20.0, away_consistency: float = 20.0,
                 home_advantage: float = 50.0) -> dict:
    """Predict a single game given team stats."""
    elo_diff = home_elo - away_elo + home_advantage
    form_diff = home_form - away_form

    X = np.array([[elo_diff, form_diff, home_form, away_form, home_consistency, away_consistency]])

    win_prob = win_model.predict_proba(X)[0][1]
    predicted_margin = margin_model.predict(X)[0]

    return {
        "home_win_prob": round(win_prob * 100, 1),
        "away_win_prob": round((1 - win_prob) * 100, 1),
        "predicted_margin": round(predicted_margin, 1)
    }

def save_models(win_model, margin_model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(win_model, os.path.join(MODEL_DIR, "win_model.pkl"))
    joblib.dump(margin_model, os.path.join(MODEL_DIR, "margin_model.pkl"))

def load_models():
    win_model = joblib.load(os.path.join(MODEL_DIR, "win_model.pkl"))
    margin_model = joblib.load(os.path.join(MODEL_DIR, "margin_model.pkl"))
    return win_model, margin_model
