"""
generate_predictions.py — Pre-bake predictions to JSON for fast mobile loading.

Run by GitHub Actions on schedule:
  - Mon–Thu 8am AWST: refresh ladder/form/Elo (no full prediction run)
  - Fri 8pm AWST: full prediction run for all weekend games
  - Game day: 20 mins before each bounce (caught by the 'pre_bounce' mode)

Output: data/predictions.json (committed back to repo by Actions)
        data/model_meta.json  (model accuracy, last updated, round)

Usage:
  python generate_predictions.py --mode full        # full prediction run
  python generate_predictions.py --mode refresh     # ladder/form/Elo only
  python generate_predictions.py --mode pre_bounce  # re-run predictions only (no retrain)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "AFL-Predictor/1.0 (github.com/nwowhar/afltip)"}
SQUIGGLE  = "https://api.squiggle.com.au/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.json")
META_FILE        = os.path.join(OUTPUT_DIR, "model_meta.json")
START_YEAR = 2016
AEST_OFFSET = timedelta(hours=10)  # AEST (use 11 during daylight saving)


def aest_now():
    return datetime.now(timezone.utc) + AEST_OFFSET


def normalise_team(name: str) -> str:
    """Normalise team name to match our internal naming."""
    MAP = {
        "Greater Western Sydney": "GWS Giants",
        "GWS": "GWS Giants",
        "Brisbane Lions": "Brisbane Lions",
        "Brisbane": "Brisbane Lions",
        "Kangaroos": "North Melbourne",
        "West Coast Eagles": "West Coast",
    }
    return MAP.get(str(name).strip(), str(name).strip())


def fetch_upcoming_games(year: int) -> list[dict]:
    """Fetch all incomplete games for the current year from Squiggle."""
    r = requests.get(f"{SQUIGGLE}?q=games;year={year}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    games = r.json().get("games", [])
    incomplete = [g for g in games if int(g.get("complete", 0)) < 100]
    return incomplete


def fetch_completed_games(year: int) -> list[dict]:
    r = requests.get(f"{SQUIGGLE}?q=games;year={year}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return [g for g in r.json().get("games", []) if int(g.get("complete", 0)) == 100]


def build_and_train():
    """Full model build and training pipeline. Returns all model artefacts."""
    print("Building model from scratch...")

    from data.fetcher import (get_all_games, enrich_games, get_team_current_stats,
                               get_standings_multi_year, normalise_team as _nt)
    from data.afltables import get_all_team_season_stats
    from data.lineup import get_pav_multi_year, get_current_lineups, compute_lineup_strength
    from data.experience import compute_experience_from_pav
    from model.elo import (build_elo_ratings, regress_elos_to_mean,
                           build_odelo_ratings, regress_odelo_to_mean)
    from model.predictor import (build_features, add_season_stat_features,
                                 add_pav_features, add_experience_features,
                                 add_standings_features, train_models,
                                 get_team_current_stats as _gts)

    games_df     = get_all_games(START_YEAR)
    season_stats = get_all_team_season_stats(START_YEAR)
    pav_df       = get_pav_multi_year(2010)
    standings_df = get_standings_multi_year(START_YEAR)

    df = enrich_games(games_df)
    df, elo_history   = build_elo_ratings(df)
    df, odelo_history = build_odelo_ratings(df)
    df = build_features(df)
    df = add_season_stat_features(df, season_stats)
    df = add_pav_features(df, pav_df)
    exp_df = compute_experience_from_pav(pav_df, games_df, year=datetime.now().year)
    df = add_experience_features(df, exp_df)
    df = add_standings_features(df, standings_df)

    win_model, margin_model, metrics, fi_df = train_models(df)

    current_elos = regress_elos_to_mean(elo_history)
    current_elos["_odelo"] = regress_odelo_to_mean(odelo_history)
    team_stats   = get_team_current_stats(df)

    print(f"Model trained: {metrics['win_accuracy']*100:.1f}% accuracy, "
          f"{metrics['n_games']} games, {metrics['n_features']} features")

    return (df, win_model, margin_model, metrics, current_elos,
            team_stats, season_stats, pav_df, fi_df, exp_df, standings_df)


def run_predictions(artefacts, upcoming_games: list[dict]) -> list[dict]:
    """Run predictions for a list of upcoming games."""
    (df, win_model, margin_model, metrics, current_elos,
     team_stats, season_stats, pav_df, fi_df, exp_df, standings_df) = artefacts

    from model.predictor import build_prediction_features, predict_game
    from data.lineup import get_current_lineups, compute_lineup_strength

    lineup_df        = get_current_lineups()
    lineup_strength  = compute_lineup_strength(lineup_df, pav_df) if not lineup_df.empty else {}

    results = []

    if not upcoming_games:
        print("No upcoming games found.")
        return results

    # Group by round to get current_round
    rounds = sorted(set(int(g.get("round", 1)) for g in upcoming_games))
    current_round = min(rounds) if rounds else 1

    print(f"Generating predictions for Round {current_round} "
          f"({len(upcoming_games)} games)...")

    for game in upcoming_games:
        home  = normalise_team(str(game.get("hteam", "")))
        away  = normalise_team(str(game.get("ateam", "")))
        venue = str(game.get("venue", ""))
        rnd   = int(game.get("round", current_round))
        date  = str(game.get("date", ""))
        timestr = str(game.get("timestr", ""))
        game_id = str(game.get("id", ""))

        if home not in current_elos or away not in current_elos:
            print(f"  Skipping {home} vs {away} — Elo not found")
            continue

        try:
            feats = build_prediction_features(
                home, away, venue,
                current_elos, team_stats,
                season_stats, lineup_strength,
                df, exp_df, standings_df,
                current_round=rnd,
            )
            pred = predict_game(win_model, margin_model, feats, metrics["features_used"])

            winner = home if pred["home_win_prob"] > 50 else away
            margin = abs(pred["predicted_margin"])

            # Key factors for display
            h_elo = float(current_elos.get(home, 1500))
            a_elo = float(current_elos.get(away, 1500))
            hs    = team_stats.get(home, {})
            as_   = team_stats.get(away, {})

            results.append({
                "game_id":        game_id,
                "round":          rnd,
                "home":           home,
                "away":           away,
                "venue":          venue,
                "date":           date,
                "timestr":        timestr,
                "home_win_prob":  pred["home_win_prob"],
                "away_win_prob":  pred["away_win_prob"],
                "predicted_margin": pred["predicted_margin"],
                "predicted_winner": winner,
                "predicted_margin_abs": round(margin, 1),
                "home_elo":       round(h_elo, 1),
                "away_elo":       round(a_elo, 1),
                "home_form":      float(hs.get("last5_avg", 0)),
                "away_form":      float(as_.get("last5_avg", 0)),
                "home_streak":    int(hs.get("streak", 0)),
                "away_streak":    int(as_.get("streak", 0)),
                "pav_available":  int(feats.get("lineup_available", 0)),
                "generated_at":   aest_now().isoformat(),
            })
            print(f"  ✓ {home} vs {away}: {winner} by ~{margin:.0f}pts "
                  f"({pred['home_win_prob']}% / {pred['away_win_prob']}%)")

        except Exception as e:
            print(f"  ✗ {home} vs {away}: {e}")
            continue

    return results


def compute_season_tips_record(artefacts, year: int) -> dict:
    """Replay predictions against completed games to get season record."""
    (df, win_model, margin_model, metrics, current_elos,
     team_stats, season_stats, pav_df, fi_df, exp_df, standings_df) = artefacts

    from model.predictor import build_prediction_features, predict_game

    completed = fetch_completed_games(year)
    correct = total = 0

    for game in completed:
        home  = normalise_team(str(game.get("hteam", "")))
        away  = normalise_team(str(game.get("ateam", "")))
        venue = str(game.get("venue", ""))
        rnd   = int(game.get("round", 1))
        hs    = int(game.get("hscore", 0) or 0)
        as_   = int(game.get("ascore", 0) or 0)

        if home not in current_elos or away not in current_elos:
            continue
        if hs == 0 and as_ == 0:
            continue

        try:
            feats = build_prediction_features(
                home, away, venue, current_elos, team_stats,
                season_stats, {}, df, exp_df, standings_df,
                current_round=rnd,
            )
            pred  = predict_game(win_model, margin_model, feats, metrics["features_used"])
            tip   = home if pred["home_win_prob"] > 50 else away
            actual = home if hs > as_ else away
            total += 1
            if tip == actual:
                correct += 1
        except Exception:
            continue

    pct = round(correct / total * 100, 1) if total else 0
    return {"correct": correct, "total": total, "pct": pct}


def save_outputs(predictions: list[dict], meta: dict):
    """Write predictions and meta to JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump({
            "generated_at": aest_now().isoformat(),
            "predictions":  predictions,
        }, f, indent=2, default=str)

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved {len(predictions)} predictions → {PREDICTIONS_FILE}")
    print(f"Saved meta → {META_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "refresh", "pre_bounce"],
                        default="full")
    args = parser.parse_args()

    year = datetime.now().year
    print(f"\n{'='*60}")
    print(f"AFL Predictor — mode={args.mode} — {aest_now().strftime('%a %d %b %Y %H:%M AEST')}")
    print(f"{'='*60}\n")

    artefacts = build_and_train()
    (df, win_model, margin_model, metrics, current_elos,
     team_stats, season_stats, pav_df, fi_df, exp_df, standings_df) = artefacts

    # Always run predictions (even in refresh mode, get upcoming games)
    upcoming = fetch_upcoming_games(year)

    if args.mode == "refresh":
        # Lightweight: just update Elo/form meta, no new predictions unless round changed
        existing = {}
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE) as f:
                existing = json.load(f)
        predictions = existing.get("predictions", [])
        print("Refresh mode — keeping existing predictions, updating meta only")
    else:
        # full or pre_bounce — regenerate all predictions
        predictions = run_predictions(artefacts, upcoming)

    # Season tips record
    tips = compute_season_tips_record(artefacts, year)
    print(f"\n2026 Tips: {tips['correct']}/{tips['total']} ({tips['pct']}%)")

    # Build round info
    rounds = sorted(set(int(g.get("round", 0)) for g in upcoming)) if upcoming else []
    current_round = min(rounds) if rounds else None

    meta = {
        "generated_at":    aest_now().isoformat(),
        "mode":            args.mode,
        "model_accuracy":  round(metrics["win_accuracy"] * 100, 1),
        "model_r2":        round(metrics["margin_r2"], 3),
        "n_games_trained": metrics["n_games"],
        "n_features":      metrics["n_features"],
        "current_round":   current_round,
        "n_upcoming":      len(upcoming),
        "tips_correct":    tips["correct"],
        "tips_total":      tips["total"],
        "tips_pct":        tips["pct"],
        "year":            year,
    }

    save_outputs(predictions, meta)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
