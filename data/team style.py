"""
data/team_style.py
------------------
Derives team playing-style features from AFL Tables season stat TOTALS
(already scraped by afltables.py — no extra HTTP requests needed).

Style features capture HOW teams play, not just whether they win:
  - kick_ratio   : kicks / (kicks + handballs)  — style spectrum (0=handball, 1=kick)
  - tackle_rate  : tackles per game              — pressure intensity
  - hitout_rate  : hitouts per game              — ruck dominance
  - mark_rate    : marks per game                — aerial / corridor game

Matchup diff features added to each game:
  - kick_ratio_diff   : home kick_ratio - away kick_ratio
  - tackle_diff       : home tackle_rate - away tackle_rate
  - hitout_diff       : home hitout_rate - away hitout_rate
  - mark_diff         : home mark_rate - away mark_rate
  - kick_vs_tackle    : kick_ratio_diff × (−tackle_diff)  — pressure vs style interaction
  - ruck_advantage    : hitout_diff  (explicit ruck gap label)

Leakage guard
-------------
Training: attach_style_features() always uses year−1 season stats for year Y games.
Live prediction: build_prediction_features() uses prev-year stats in Rounds 1–3,
  current-year stats from Round 4 onwards (when enough games have been played).
"""

import pandas as pd
import numpy as np

# ── Feature list ──────────────────────────────────────────────────────────────
STYLE_FEATURES = [
    "kick_ratio_diff",   # kick-heavy vs handball-heavy style gap
    "tackle_diff",       # pressure / contested game intensity gap
    "hitout_diff",       # ruck dominance gap
    "mark_diff",         # aerial vs ground game gap
    "kick_vs_tackle",    # interaction: kick style vs pressure defence
    "ruck_advantage",    # explicit ruck label (same as hitout_diff)
]


# ── Build style profile from season stats ─────────────────────────────────────

def build_style_features_from_season_stats(season_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-team per-year style profile from AFL Tables season totals.

    season_stats_df: output of afltables.get_all_team_season_stats()
      expected columns: year, team, games_played,
                        avg_kicks, avg_handballs, avg_marks,
                        avg_tackles, avg_hitouts

    Returns DataFrame: year | team | kick_ratio | tackle_rate | hitout_rate | mark_rate
    One row per team per year.
    """
    if season_stats_df is None or season_stats_df.empty:
        return pd.DataFrame()

    df = season_stats_df.copy()

    # afltables.py stores per-game AVERAGES already (divided by games_played)
    # so avg_kicks, avg_handballs etc. are already per-game rates.
    kicks     = pd.to_numeric(df.get("avg_kicks",     0), errors="coerce").fillna(0)
    handballs = pd.to_numeric(df.get("avg_handballs", 0), errors="coerce").fillna(0)
    marks     = pd.to_numeric(df.get("avg_marks",     0), errors="coerce").fillna(0)
    tackles   = pd.to_numeric(df.get("avg_tackles",   0), errors="coerce").fillna(0)
    hitouts   = pd.to_numeric(df.get("avg_hitouts",   0), errors="coerce").fillna(0)

    total_disp = (kicks + handballs).clip(lower=1)

    result = pd.DataFrame()
    result["year"]        = df["year"] if "year" in df.columns else df.index
    result["team"]        = df["team"]
    result["kick_ratio"]  = kicks / total_disp          # 0–1, higher = kick-heavy
    result["tackle_rate"] = tackles                      # already per-game
    result["hitout_rate"] = hitouts                      # already per-game
    result["mark_rate"]   = marks                        # already per-game

    return result.reset_index(drop=True)


# ── Compute matchup features for one game ────────────────────────────────────

def compute_style_matchup(home_team: str, away_team: str,
                           style_df: pd.DataFrame,
                           game_year: int,
                           use_prev_season: bool = False) -> dict:
    """
    Compute STYLE_FEATURES for a single game.

    style_df: output of build_style_features_from_season_stats()
    use_prev_season: if True, look up year-1 stats (early-season live predictions)

    Returns dict of 6 floats; all 0.0 if data unavailable.
    """
    zeros = {f: 0.0 for f in STYLE_FEATURES}
    if style_df is None or style_df.empty:
        return zeros

    lookup_year = (game_year - 1) if use_prev_season else game_year

    def _get(team, yr):
        row = style_df[(style_df["team"] == team) & (style_df["year"] == yr)]
        if row.empty:
            # Fallback to previous year
            row = style_df[(style_df["team"] == team) & (style_df["year"] == yr - 1)]
        return row.iloc[0] if not row.empty else None

    h = _get(home_team, lookup_year)
    a = _get(away_team, lookup_year)

    if h is None or a is None:
        return zeros

    kick_ratio_diff = float(h["kick_ratio"])  - float(a["kick_ratio"])
    tackle_diff     = float(h["tackle_rate"]) - float(a["tackle_rate"])
    hitout_diff     = float(h["hitout_rate"]) - float(a["hitout_rate"])
    mark_diff       = float(h["mark_rate"])   - float(a["mark_rate"])

    # Interaction: kick-heavy home team vs high-pressure away team
    # +ve = home kicks more AND away tackles less (home style advantage)
    # -ve = away pressure neutralises home's kicking game
    kick_vs_tackle = kick_ratio_diff * (-tackle_diff)

    return {
        "kick_ratio_diff": kick_ratio_diff,
        "tackle_diff":     tackle_diff,
        "hitout_diff":     hitout_diff,
        "mark_diff":       mark_diff,
        "kick_vs_tackle":  kick_vs_tackle,
        "ruck_advantage":  hitout_diff,   # explicit label for the ruck gap
    }


# ── Attach style features to training DataFrame (leakage-safe) ───────────────

def attach_style_features(df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add STYLE_FEATURES columns to a training games DataFrame.

    Always uses year-1 season stats for games in year Y — fully leakage-safe.
    If style_df is empty, all columns are set to 0.0.
    """
    if style_df is None or style_df.empty:
        for f in STYLE_FEATURES:
            df[f] = 0.0
        return df

    rows = []
    for _, game in df.iterrows():
        feat = compute_style_matchup(
            home_team=str(game.get("hteam", "")),
            away_team=str(game.get("ateam", "")),
            style_df=style_df,
            game_year=int(game.get("year", 0)),
            use_prev_season=True,   # always use prior season for training
        )
        rows.append(feat)

    style_game_df = pd.DataFrame(rows, index=df.index)
    for col in style_game_df.columns:
        df[col] = style_game_df[col].fillna(0.0)

    return df
