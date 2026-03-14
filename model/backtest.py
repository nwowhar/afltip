"""
Backtest engine — walk-forward accuracy, ablation testing, calibration, margin MAE.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, brier_score_loss, log_loss
)
from sklearn.inspection import permutation_importance


# ── Feature groups for ablation testing ──────────────────────────────────────
FEATURE_GROUPS = {
    "Elo":               ["elo_diff"],
    "Form (rolling)":    ["form_diff", "home_form", "away_form",
                          "home_consistency", "away_consistency"],
    "Travel":            ["travel_diff", "travel_home_km", "travel_away_km"],
    "Streak":            ["streak_diff", "home_streak", "away_streak"],
    "Last margin":       ["last_margin_diff", "last3_diff", "last5_diff"],
    "Season stats":      ["cl_diff", "i50_diff", "cp_diff", "tk_diff", "ho_diff"],
    "Stats: Clearances": ["cl_diff", "cp_diff"],
    "Stats: Inside 50s": ["i50_diff"],
    "Stats: Tackles":    ["tk_diff"],
    "Stats: Hitouts":    ["ho_diff"],
    "Ladder position":   ["ladder_rank_diff", "ladder_pct_diff", "ladder_wins_diff"],
    "Style: Kick ratio": ["kick_ratio_diff", "kick_vs_tackle"],
    "Style: Hitouts":    ["hitout_diff", "ruck_advantage"],
    # Removed via ablation: Rest days (+0.56%), PAV lineup (+0.19%), Experience (+0.19%)
    # Previously removed: Travel×Rest, Travel record, Clangers, Marks, Style:Tackles
}

ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]


def run_walk_forward_backtest(df: pd.DataFrame,
                               feature_cols: list,
                               min_train_years: int = 3) -> pd.DataFrame:
    """
    Walk-forward backtest: train on years 1..N-1, predict year N.
    Returns a DataFrame with one row per game with predicted prob and actual result.
    """
    df = df.copy()
    years = sorted(df["year"].unique())
    if len(years) < min_train_years + 1:
        return pd.DataFrame()

    available = [f for f in feature_cols if f in df.columns]
    results = []

    for i, test_year in enumerate(years[min_train_years:], start=min_train_years):
        train = df[df["year"].isin(years[:i])].dropna(subset=available + ["home_win"])
        test  = df[df["year"] == test_year].dropna(subset=available + ["home_win"])

        if len(train) < 50 or len(test) < 5:
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=42
            ))
        ])
        model.fit(train[available], train["home_win"])

        probs   = model.predict_proba(test[available])[:, 1]
        preds   = (probs >= 0.5).astype(int)
        margins = test.get("margin", pd.Series([0]*len(test))).values

        for j in range(len(test)):
            results.append({
                "year":          test_year,
                "game_idx":      test.index[j],
                "home_team":     test.iloc[j].get("hteam", ""),
                "away_team":     test.iloc[j].get("ateam", ""),
                "actual":        int(test["home_win"].iloc[j]),
                "predicted":     int(preds[j]),
                "prob":          round(float(probs[j]), 4),
                "correct":       int(preds[j] == test["home_win"].iloc[j]),
                "actual_margin": float(margins[j]) if len(margins) > j else 0,
            })

    return pd.DataFrame(results)


def compute_yearly_accuracy(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, Brier score, and calibration per year."""
    if backtest_df.empty:
        return pd.DataFrame()

    rows = []
    for year, grp in backtest_df.groupby("year"):
        acc   = grp["correct"].mean()
        brier = brier_score_loss(grp["actual"], grp["prob"])
        mean_prob   = grp["prob"].mean()
        actual_rate = grp["actual"].mean()
        rows.append({
            "year":             year,
            "n_games":          len(grp),
            "accuracy":         round(acc * 100, 1),
            "brier_score":      round(brier, 4),
            "mean_pred_prob":   round(mean_prob, 3),
            "actual_win_rate":  round(actual_rate, 3),
            "calibration_err":  round(abs(mean_prob - actual_rate), 3),
        })
    return pd.DataFrame(rows)


def ablation_test(df: pd.DataFrame,
                  feature_groups: dict = None,
                  min_train_years: int = 3) -> pd.DataFrame:
    """
    Leave-one-out ablation: remove each feature group and measure accuracy impact.
    Negative delta = removing that group HURT accuracy (group is useful).
    Positive delta = removing it HELPED (group is noise or harmful).
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    all_feats = [f for feats in feature_groups.values() for f in feats
                 if f in df.columns]

    if not all_feats:
        return pd.DataFrame()

    base_bt = run_walk_forward_backtest(df, all_feats, min_train_years)
    if base_bt.empty:
        return pd.DataFrame()
    base_acc = base_bt["correct"].mean() * 100

    rows = [{
        "group":         "ALL FEATURES (baseline)",
        "accuracy":      round(base_acc, 1),
        "delta":         0.0,
        "n_features":    len(all_feats),
        "interpretation": "",
    }]

    for group_name, group_feats in feature_groups.items():
        available = [f for f in group_feats if f in df.columns]
        if not available:
            continue

        reduced = [f for f in all_feats if f not in available]
        if len(reduced) < 2:
            continue

        bt = run_walk_forward_backtest(df, reduced, min_train_years)
        if bt.empty:
            continue

        acc_without = bt["correct"].mean() * 100
        delta = acc_without - base_acc

        rows.append({
            "group":          group_name,
            "accuracy":       round(acc_without, 1),
            "delta":          round(delta, 2),
            "n_features":     len(available),
            "interpretation": (
                "✅ Helps"   if delta < -0.3
                else ("❌ Hurts" if delta > 0.3 else "➡️ Neutral")
            )
        })

    return pd.DataFrame(rows)


def permutation_importance_analysis(df: pd.DataFrame,
                                     feature_cols: list,
                                     n_repeats: int = 10) -> pd.DataFrame:
    """
    Train on all data, then compute permutation importance.
    Higher drop in accuracy when shuffled = more important feature.
    """
    available = [f for f in feature_cols if f in df.columns]
    clean = df.dropna(subset=available + ["home_win"])

    if len(clean) < 100:
        return pd.DataFrame()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, random_state=42
        ))
    ])
    model.fit(clean[available], clean["home_win"])

    result = permutation_importance(
        model, clean[available].values, clean["home_win"].values,
        n_repeats=n_repeats, random_state=42, scoring="accuracy"
    )

    imp_df = pd.DataFrame({
        "feature":    available,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False)

    group_lookup = {}
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            group_lookup[f] = group
    imp_df["group"] = imp_df["feature"].map(lambda f: group_lookup.get(f, "Other"))

    return imp_df


def optimise_start_year(df: pd.DataFrame,
                         feature_cols: list,
                         candidate_years: list = None,
                         holdout_years: int = 3,
                         min_train_years: int = 2) -> pd.DataFrame:
    """
    For each candidate training start year, run a walk-forward backtest on the
    most recent holdout_years seasons and record out-of-sample accuracy.
    Returns: start_year | n_train_games | accuracy | brier | n_test_games
    """
    all_years = sorted(df["year"].unique())
    if len(all_years) < min_train_years + holdout_years:
        return pd.DataFrame()

    if candidate_years is None:
        max_start = all_years[-(holdout_years + min_train_years)]
        candidate_years = [y for y in all_years if y <= max_start]

    available = [f for f in feature_cols if f in df.columns]
    if not available:
        return pd.DataFrame()

    holdout_window = all_years[-holdout_years:]
    rows = []

    for start_year in candidate_years:
        subset = df[df["year"] >= start_year].copy()
        subset_years = sorted(subset["year"].unique())

        pre_holdout = [y for y in subset_years if y < holdout_window[0]]
        if len(pre_holdout) < min_train_years:
            continue

        results = []
        for test_year in holdout_window:
            train_years = [y for y in subset_years if y < test_year]
            if len(train_years) < min_train_years:
                continue

            train = subset[subset["year"].isin(train_years)].dropna(
                subset=available + ["home_win"])
            test  = subset[subset["year"] == test_year].dropna(
                subset=available + ["home_win"])

            if len(train) < 30 or len(test) < 5:
                continue

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.05, random_state=42
                ))
            ])
            model.fit(train[available], train["home_win"])
            probs = model.predict_proba(test[available])[:, 1]
            preds = (probs >= 0.5).astype(int)

            for j in range(len(test)):
                results.append({
                    "actual":    int(test["home_win"].iloc[j]),
                    "predicted": int(preds[j]),
                    "prob":      float(probs[j]),
                })

        if not results:
            continue

        res_df = pd.DataFrame(results)
        acc   = (res_df["predicted"] == res_df["actual"]).mean() * 100
        brier = brier_score_loss(res_df["actual"], res_df["prob"])
        n_train = len(subset[subset["year"] < holdout_window[0]])

        rows.append({
            "start_year":     start_year,
            "n_train_games":  n_train,
            "accuracy":       round(acc, 2),
            "brier_score":    round(brier, 4),
            "n_test_games":   len(res_df),
            "holdout_seasons": holdout_years,
        })

    return pd.DataFrame(rows).sort_values("start_year")


def margin_prediction_backtest(df: pd.DataFrame,
                                feature_cols: list,
                                min_train_years: int = 3) -> pd.DataFrame:
    """Walk-forward backtest for margin prediction. Returns MAE per year."""
    available = [f for f in feature_cols if f in df.columns]
    years = sorted(df["year"].unique())
    if len(years) < min_train_years + 1:
        return pd.DataFrame()

    rows = []
    for i, test_year in enumerate(years[min_train_years:], start=min_train_years):
        train = df[df["year"].isin(years[:i])].dropna(subset=available + ["margin"])
        test  = df[df["year"] == test_year].dropna(subset=available + ["margin"])

        if len(train) < 50 or len(test) < 5:
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=42
            ))
        ])
        model.fit(train[available], train["margin"])
        preds = model.predict(test[available])
        mae = mean_absolute_error(test["margin"], preds)

        rows.append({
            "year":      test_year,
            "mae_points": round(mae, 1),
            "n_games":   len(test),
        })

    return pd.DataFrame(rows)


def elo_anchor_sweep(df: pd.DataFrame,
                     win_model,
                     margin_model,
                     metrics: dict,
                     min_train_years: int = 3,
                     anchors: list = None) -> pd.DataFrame:
    """
    Sweep a range of Elo anchor weights and measure out-of-sample accuracy.
    For each anchor value, re-run the walk-forward backtest blending GBM
    predictions with pure Elo at that fixed weight.

    anchors: list of floats 0.0-1.0 (default: 0, 0.1, 0.2, ... 1.0)
    Returns DataFrame with columns: elo_anchor, accuracy, brier_score, n_games
    """
    from model.predictor import predict_game as _predict_game

    if anchors is None:
        anchors = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0

    feature_set = metrics.get("features_used", [])
    available   = [f for f in feature_set if f in df.columns]

    years = sorted(df["year"].unique())
    rows  = []

    for anchor in anchors:
        correct_list = []
        brier_list   = []

        for i, test_year in enumerate(years):
            train_years = years[:i]
            if len(train_years) < min_train_years:
                continue

            train = df[df["year"].isin(train_years)]
            test  = df[df["year"] == test_year]
            if len(train) < 50 or test.empty:
                continue

            # Train a fresh model for this fold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import GradientBoostingClassifier

            fold_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    GradientBoostingClassifier(
                    n_estimators=200, max_depth=3,
                    learning_rate=0.05, subsample=0.8,
                    random_state=42
                ))
            ])
            clean_train = train.dropna(subset=available + ["home_win"])
            if len(clean_train) < 50:
                continue
            fold_model.fit(clean_train[available], clean_train["home_win"])

            for _, row in test.iterrows():
                feats = {f: float(row.get(f, 0.0)) for f in available}
                feats["elo_diff"] = float(row.get("elo_diff", 0.0))

                # Use a dummy margin model (we only care about win accuracy here)
                class _DummyMargin:
                    def predict(self, X): return [0.0]

                pred = _predict_game(
                    fold_model, _DummyMargin(),
                    feats, available,
                    elo_anchor=anchor
                )
                prob      = pred["home_win_prob"] / 100.0
                actual    = int(row.get("home_win", 0))
                predicted = 1 if prob > 0.5 else 0
                correct_list.append(int(predicted == actual))
                brier_list.append((prob - actual) ** 2)

        if correct_list:
            rows.append({
                "elo_anchor":  anchor,
                "accuracy":    round(sum(correct_list) / len(correct_list) * 100, 2),
                "brier_score": round(sum(brier_list)   / len(brier_list),  4),
                "n_games":     len(correct_list),
            })

    return pd.DataFrame(rows)
