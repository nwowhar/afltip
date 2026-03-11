import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import (get_all_games, get_upcoming_games, enrich_games, get_team_current_stats,
                          get_squiggle_consensus, get_odds_api, normalise_team)
from data.afltables import get_all_team_season_stats
from data.lineup import get_pav_multi_year, get_current_lineups, compute_lineup_strength
from model.elo import build_elo_ratings, regress_elos_to_mean
from model.predictor import (build_features, add_season_stat_features,
                              add_pav_features, train_models, predict_game,
                              build_prediction_features, CORE_FEATURES, ALL_FEATURES)
from model.backtest import (run_walk_forward_backtest, compute_yearly_accuracy,
                             ablation_test, permutation_importance_analysis,
                             margin_prediction_backtest, FEATURE_GROUPS)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AFL Predictor", page_icon="🏉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1px; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 12px;
    padding: 18px; text-align: center; color: white; margin-bottom: 8px;
}
.metric-card .value { font-size: 1.9rem; font-weight: 700; color: #e94560; }
.metric-card .label { font-size: 0.8rem; color: #aaa; margin-top: 4px; }
.metric-card .sub   { font-size: 0.75rem; color: #2ecc71; margin-top: 4px; }
.team-vs {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    border: 1px solid #e9456033;
}
[data-testid="stSidebar"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏉 AFL Predictor")
    st.markdown("---")
    start_year = st.slider("Training data from", 2010, 2020, 2013,
                           help="Model trains on all completed games from this year to present. Earlier = more data but older game styles. 2013–2015 is a good balance.")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict a Game",
        "📈 Team Form",
        "🏆 Elo Ladder",
        "📋 Team Stats",
        "🔬 Feature Importance",
        "📉 Backtest",
        "👕 Lineup Strength",
        "🤖 AI Analysis",
        "💰 Value Bets",
    ])
    st.markdown("---")
    st.markdown("<small style='color:#666'>Data: Squiggle API + AFL Tables<br>Model: Gradient Boosting + Elo + PAV</small>", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="📡 Fetching game results...")
def load_games(start_year):
    return get_all_games(start_year)

@st.cache_data(ttl=86400, show_spinner="📊 Fetching AFL Tables season stats...")
def load_season_stats(start_year):
    return get_all_team_season_stats(start_year)

@st.cache_data(ttl=86400, show_spinner="⭐ Fetching PAV player ratings...")
def load_pav(start_year):
    return get_pav_multi_year(start_year)

@st.cache_data(ttl=1800, show_spinner="👕 Fetching announced lineups...")
def load_lineups():
    return get_current_lineups()

@st.cache_data(ttl=3601, show_spinner="🤖 Building Elo ratings & training model...")
def build_model(start_year):
    games_df = load_games(start_year)
    if games_df is None or games_df.empty:
        return None, None, None, None, {}, {}, None, None, None

    season_stats = load_season_stats(start_year)
    pav_df       = load_pav(start_year)

    # Enrich with fatigue/context features
    df = enrich_games(games_df)
    # Build Elo
    df, elo_history = build_elo_ratings(df)
    # Build rolling form
    df = build_features(df)
    # Add season stats features
    df = add_season_stat_features(df, season_stats)
    # Add PAV features
    df = add_pav_features(df, pav_df)

    win_model, margin_model, metrics, fi_df = train_models(df)
    current_elos = regress_elos_to_mean(elo_history)
    team_stats   = get_team_current_stats(df)

    return df, win_model, margin_model, metrics, current_elos, team_stats, season_stats, pav_df, fi_df

with st.spinner("Loading model..."):
    result = build_model(start_year)

if result[0] is None:
    st.error("Could not load game data. Check your internet connection.")
    st.stop()

df, win_model, margin_model, metrics, current_elos, team_stats, season_stats, pav_df, fi_df = result
teams = sorted(current_elos.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def mc(value, label, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div>{sub_html}</div>'

def dark_chart(fig, height=350):
    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#0f3460"),
        yaxis=dict(gridcolor="#0f3460"),
        height=height, margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def get_team_form_df(team, n=15):
    home = df[df["hteam"] == team].copy()
    home["margin"]   = home["hscore"] - home["ascore"]
    home["opponent"] = home["ateam"]
    home["venue_type"] = "Home"
    away = df[df["ateam"] == team].copy()
    away["margin"]   = away["ascore"] - away["hscore"]
    away["opponent"] = away["hteam"]
    away["venue_type"] = "Away"
    combined = pd.concat([home, away]).sort_values("date_parsed").tail(n).reset_index(drop=True)
    combined["result"] = combined["margin"].apply(lambda x: "W" if x > 0 else ("L" if x < 0 else "D"))
    combined["game_label"] = combined.apply(
        lambda r: f"R{r['round']} {r['year']} vs {r['opponent']}", axis=1)
    return combined

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# AFL MATCH PREDICTOR")
    st.markdown(f"*{metrics['n_games']:,} games · {metrics['n_features']} features · {start_year}–present*")

    gain = metrics.get("accuracy_gain", 0)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(mc(f"{metrics['win_accuracy']*100:.1f}%", "Win Prediction Accuracy",
                            f"+{gain*100:.1f}% vs Elo-only" if gain > 0 else ""), unsafe_allow_html=True)
    with c2: st.markdown(mc(f"{metrics['base_accuracy']*100:.1f}%", "Elo-Only Baseline"), unsafe_allow_html=True)
    with c3: st.markdown(mc(f"{metrics['margin_r2']:.3f}", "Margin R²"), unsafe_allow_html=True)
    with c4: st.markdown(mc(f"{metrics['n_games']:,}", "Training Games"), unsafe_allow_html=True)

    st.markdown("---")

    try:
        # Fetch all incomplete games for the year so we can offer a round picker
        import requests as _req
        _r = _req.get(f"https://api.squiggle.com.au/?q=games;year={datetime.now().year}",
                      headers={"User-Agent": "AFL-Predictor/1.0"}, timeout=15)
        _all_games = pd.DataFrame(_r.json().get("games", []))
        _incomplete = _all_games[_all_games["complete"] < 100] if not _all_games.empty else pd.DataFrame()

        if not _incomplete.empty:
            available_rounds = sorted(_incomplete["round"].unique())
            default_round    = int(_incomplete["round"].min())
            col_title, col_picker = st.columns([2, 1])
            with col_title:
                st.markdown("## UPCOMING GAMES")
            with col_picker:
                selected_round = st.selectbox("Round", available_rounds,
                                              index=available_rounds.index(default_round),
                                              label_visibility="collapsed")
            upcoming = _incomplete[_incomplete["round"] == selected_round].copy()
            st.markdown(f"*Round {selected_round} — {len(upcoming)} games*")
        else:
            st.markdown("## UPCOMING GAMES")
            upcoming = pd.DataFrame()

        lineup_df = load_lineups()
        lineup_strength = {}
        if not lineup_df.empty and not pav_df.empty:
            lineup_strength = compute_lineup_strength(lineup_df, pav_df)

        if upcoming.empty:
            st.info("No upcoming games right now — check back closer to game day.")
        else:
            for _, game in upcoming.iterrows():
                home  = game.get("hteam", "?")
                away  = game.get("ateam", "?")
                venue = game.get("venue", "")
                if home not in current_elos or away not in current_elos:
                    continue

                try:
                    # Inline feature builder — no dependency on cached modules
                    def _safe_float(v, default=0.0):
                        try:
                            if v is None: return float(default)
                            if hasattr(v, 'iloc'): v = v.iloc[-1]
                            if hasattr(v, 'item'): v = v.item()
                            return float(v)
                        except: return float(default)

                    from data.fetcher import travel_distance_km
                    _PERTH_VENUES = {"Optus Stadium", "Perth Stadium", "Subiaco Oval"}
                    _LONG_KM      = 1000
                    _PERTH_KM     = 2000

                    h_elo   = _safe_float(current_elos.get(home, 1500), 1500)
                    a_elo   = _safe_float(current_elos.get(away, 1500), 1500)
                    hs_     = team_stats.get(home, {})
                    as__    = team_stats.get(away, {})
                    h_form  = _safe_float(hs_.get("last5_avg", 0))
                    a_form  = _safe_float(as__.get("last5_avg", 0))
                    h_std   = _safe_float(hs_.get("last5_std", 20), 20)
                    a_std   = _safe_float(as__.get("last5_std", 20), 20)
                    h_km    = float(travel_distance_km(home, venue))
                    a_km    = float(travel_distance_km(away, venue))

                    import pandas as _pd2
                    today_ = _pd2.Timestamp.now()
                    def _rest(ld):
                        try:
                            if ld is None: return 7
                            if hasattr(ld, 'iloc'): ld = ld.iloc[-1]
                            ld = _pd2.Timestamp(ld)
                            if _pd2.isna(ld): return 7
                            raw = int((today_ - ld).days)
                            return raw if raw <= 21 else 7
                        except: return 7
                    h_rest = _rest(hs_.get("last_date"))
                    a_rest = _rest(as__.get("last_date"))

                    h_fat = min(h_km, 3000) / 1000 * max(14 - h_rest, 0)
                    a_fat = min(a_km, 3000) / 1000 * max(14 - a_rest, 0)

                    cur_yr = datetime.now().year
                    def _ss(team, stat):
                        if season_stats is None or season_stats.empty: return 0.0
                        row_ = season_stats[(season_stats["team"]==team) & (season_stats["year"]==cur_yr)]
                        if row_.empty:
                            row_ = season_stats[(season_stats["team"]==team) & (season_stats["year"]==cur_yr-1)]
                        return float(row_.iloc[0].get(stat, 0)) if not row_.empty else 0.0

                    feats = {
                        "elo_diff":            h_elo - a_elo + 50.0,
                        "form_diff":           h_form - a_form,
                        "home_form":           h_form,
                        "away_form":           a_form,
                        "home_consistency":    h_std,
                        "away_consistency":    a_std,
                        "travel_diff":         h_km - a_km,
                        "travel_home_km":      h_km,
                        "travel_away_km":      a_km,
                        "travel_fatigue_diff": h_fat - a_fat,
                        "home_travel_fatigue": h_fat,
                        "away_travel_fatigue": a_fat,
                        "travel_win_rate_diff": 0.0,
                        "travel_margin_diff":  0.0,
                        "perth_win_rate_diff": 0.0,
                        "is_perth_game":       1.0 if str(venue) in _PERTH_VENUES else 0.0,
                        "days_rest_diff":      float(h_rest - a_rest),
                        "days_rest_home":      float(h_rest),
                        "days_rest_away":      float(a_rest),
                        "streak_diff":         _safe_float(hs_.get("streak",0)) - _safe_float(as__.get("streak",0)),
                        "home_streak":         _safe_float(hs_.get("streak",0)),
                        "away_streak":         _safe_float(as__.get("streak",0)),
                        "last_margin_diff":    _safe_float(hs_.get("last_margin",0)) - _safe_float(as__.get("last_margin",0)),
                        "last3_diff":          _safe_float(hs_.get("last3_avg",0)) - _safe_float(as__.get("last3_avg",0)),
                        "last5_diff":          h_form - a_form,
                        "cl_diff":    _ss(home,"avg_clearances")   - _ss(away,"avg_clearances"),
                        "i50_diff":   _ss(home,"avg_inside_50s")   - _ss(away,"avg_inside_50s"),
                        "cp_diff":    _ss(home,"avg_contested_possessions") - _ss(away,"avg_contested_possessions"),
                        "tk_diff":    _ss(home,"avg_tackles")      - _ss(away,"avg_tackles"),
                        "ho_diff":    _ss(home,"avg_hitouts")      - _ss(away,"avg_hitouts"),
                        "clanger_diff": _ss(home,"avg_clangers")   - _ss(away,"avg_clangers"),
                        "pav_total_diff": 0.0, "pav_off_diff": 0.0,
                        "pav_def_diff":   0.0, "pav_mid_diff": 0.0,
                    }
                except Exception as game_err:
                    import traceback
                    st.error(f"Error for {home} vs {away}: {game_err}")
                    st.code(traceback.format_exc())
                    continue

                pred = predict_game(win_model, margin_model, feats)
                winner = home if pred["home_win_prob"] > 50 else away
                margin = abs(pred["predicted_margin"])
                hs = team_stats.get(home, {})
                as_ = team_stats.get(away, {})

                pav_available = feats.get("lineup_available", 0) == 1
                pav_note = ""
                if pav_available:
                    h_pav = feats.get("home_pav_total", 0)
                    a_pav = feats.get("away_pav_total", 0)
                    pav_note = f"⭐ PAV: {h_pav:.0f} vs {a_pav:.0f}"

                # Build home/away detail strings cleanly (no conditionals inside HTML)
                h_streak = hs.get("streak", 0)
                a_streak = as_.get("streak", 0)
                h_detail = f"✈️ {feats['travel_home_km']:.0f}km  |  💤 {feats['days_rest_home']}d rest"
                a_detail = f"✈️ {feats['travel_away_km']:.0f}km  |  💤 {feats['days_rest_away']}d rest"
                if h_streak > 1:
                    h_detail += f"  |  🔥 {h_streak}W streak"
                if a_streak > 1:
                    a_detail += f"  |  🔥 {a_streak}W streak"
                bar_pct  = int(pred["home_win_prob"])
                venue_str = venue if venue else ""
                footer    = f"{pred['home_win_prob']}%  ·  {venue_str}  ·  {pred['away_win_prob']}%"
                if pav_note:
                    footer += f"  ·  {pav_note}"

                # ── Factor analysis for insight panel ─────────────────────
                factors = [
                    # (label, home_val, away_val, home_is_better_when_higher)
                    ("Elo Rating",          h_elo,                              a_elo,                              True),
                    ("Form (last 5 avg)",   h_form,                             a_form,                             True),
                    ("Current Streak",      _safe_float(hs_.get("streak",0)),   _safe_float(as__.get("streak",0)),  True),
                    ("Last Game Margin",    _safe_float(hs_.get("last_margin",0)), _safe_float(as__.get("last_margin",0)), True),
                    ("Travel to Venue",     h_km,                               a_km,                               False),
                    ("Days Rest",           float(h_rest),                      float(a_rest),                      True),
                    ("Travel Fatigue",      h_fat,                              a_fat,                              False),
                    ("Clearances (season)", _ss(home,"avg_clearances"),         _ss(away,"avg_clearances"),         True),
                    ("Inside 50s (season)", _ss(home,"avg_inside_50s"),         _ss(away,"avg_inside_50s"),         True),
                    ("Contested Poss",      _ss(home,"avg_contested_possessions"), _ss(away,"avg_contested_possessions"), True),
                    ("Tackles (season)",    _ss(home,"avg_tackles"),            _ss(away,"avg_tackles"),            True),
                    ("Clangers (season)",   _ss(home,"avg_clangers"),           _ss(away,"avg_clangers"),           False),
                ]

                # Determine which factors favour each team
                home_edges, away_edges, neutral = [], [], []
                for label, hv, av, higher_better in factors:
                    if hv == 0 and av == 0:
                        continue
                    diff = hv - av if higher_better else av - hv
                    pct  = abs(diff) / (abs(hv) + abs(av) + 0.001) * 100
                    if pct < 3:
                        neutral.append((label, hv, av))
                    elif diff > 0:
                        home_edges.append((label, hv, av, pct, higher_better))
                    else:
                        away_edges.append((label, hv, av, pct, higher_better))

                # Sort by magnitude
                home_edges.sort(key=lambda x: -x[3])
                away_edges.sort(key=lambda x: -x[3])

                # Card
                card_html = (
                    '<div class="team-vs">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
                    f'<div><div style="font-size:1.2rem;font-weight:700;color:white">{home}</div>'
                    f'<div style="color:#aaa;font-size:0.72rem">{h_detail}</div></div>'
                    '<span style="color:#e94560;font-family:\'Bebas Neue\';font-size:1.1rem">VS</span>'
                    f'<div style="text-align:right"><div style="font-size:1.2rem;font-weight:700;color:white">{away}</div>'
                    f'<div style="color:#aaa;font-size:0.72rem">{a_detail}</div></div>'
                    '</div>'
                    '<div style="height:10px;border-radius:5px;background:#0f3460;overflow:hidden">'
                    f'<div style="width:{bar_pct}%;height:100%;background:linear-gradient(90deg,#e94560,#ff6b6b);border-radius:5px"></div>'
                    '</div>'
                    f'<div style="color:#aaa;font-size:0.75rem;margin-top:6px;text-align:center">{footer}</div>'
                    '</div>'
                )

                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(card_html, unsafe_allow_html=True)
                with c2:
                    st.markdown(mc(winner, f"by ~{margin:.0f} pts"), unsafe_allow_html=True)

                # ── Insight expander ───────────────────────────────────────
                with st.expander(f"🔍 Why? — {home} vs {away}"):

                    # Factor metadata: label, home_val, away_val, higher_is_better, unit, explanation
                    factor_meta = [
                        ("Elo Rating",          h_elo,   a_elo,   True,
                         "pts", "Elo is a chess-style rating updated after every game. Each win/loss shifts ratings based on the expected result. Home team gets +50pts advantage baked in. A 100pt gap = roughly 64% win probability."),
                        ("Form (last 5 avg)",    h_form,  a_form,  True,
                         "pts margin", "Average winning/losing margin across the last 5 games. Positive = winning by that many points on average. More responsive to recent form than Elo."),
                        ("Current Streak",       _safe_float(hs_.get("streak",0)), _safe_float(as__.get("streak",0)), True,
                         "games", "Signed win/loss streak. +3 means 3 wins in a row, -2 means 2 losses in a row."),
                        ("Last Game Margin",     _safe_float(hs_.get("last_margin",0)), _safe_float(as__.get("last_margin",0)), True,
                         "pts", "Margin from their most recent completed game. Positive = won by that many points."),
                        ("Travel to Venue",      h_km,    a_km,    False,
                         "km", "Straight-line distance each team travels to reach the venue. Lower is better — home games = ~0km, interstate = 500–900km, Perth = 2,700km from east coast."),
                        ("Days Rest",            float(h_rest), float(a_rest), True,
                         "days", "Days since their last game. More rest = fresher legs. Capped at 21 days — anything longer (summer break, bye) resets to a neutral 7 days so it doesn't skew R1 predictions."),
                        ("Travel Fatigue",       h_fat,   a_fat,   False,
                         "index", "Combined travel+rest stress index: (km travelled ÷ 1000) × max(14 − rest days, 0). A team flying 2,700km to Perth with only 6 days rest scores ~8.1 vs 0 for the home side."),
                        ("Clearances (season)",  _ss(home,"avg_clearances"),  _ss(away,"avg_clearances"),  True,
                         "per game", "Season average clearances per game from AFL Tables. Clearances out of stoppages drive transition and scoring chains — one of the strongest team performance indicators."),
                        ("Inside 50s (season)",  _ss(home,"avg_inside_50s"),  _ss(away,"avg_inside_50s"),  True,
                         "per game", "Season average entries inside the forward 50 per game. More entries = more scoring chances. Strongly correlated with winning margin."),
                        ("Contested Poss",       _ss(home,"avg_contested_possessions"), _ss(away,"avg_contested_possessions"), True,
                         "per game", "Season average contested possessions per game. Reflects contested ball dominance — teams that win this category tend to control the game's tempo."),
                        ("Tackles (season)",     _ss(home,"avg_tackles"),     _ss(away,"avg_tackles"),     True,
                         "per game", "Season average tackles per game. High tackle counts indicate pressure and defensive intensity."),
                        ("Clangers (season)",    _ss(home,"avg_clangers"),    _ss(away,"avg_clangers"),    False,
                         "per game", "Season average clangers (turnovers by hand or foot) per game. Lower is better — clangers directly gift opposition scoring opportunities."),
                    ]

                    # Recompute edges with meta
                    home_edges_m, away_edges_m = [], []
                    for label, hv, av, hib, unit, explanation in factor_meta:
                        if hv == 0 and av == 0:
                            continue
                        diff = hv - av if hib else av - hv
                        pct  = abs(diff) / (abs(hv) + abs(av) + 0.001) * 100
                        if pct < 3:
                            continue
                        entry = (label, hv, av, pct, hib, unit, explanation)
                        if diff > 0:
                            home_edges_m.append(entry)
                        else:
                            away_edges_m.append(entry)
                    home_edges_m.sort(key=lambda x: -x[3])
                    away_edges_m.sort(key=lambda x: -x[3])

                    def edge_html_rich(edges, team, colour):
                        if not edges:
                            return f'<div style="color:#666;font-size:0.8rem;padding:8px">No clear edges detected — closely matched in most areas.</div>'
                        lines = []
                        for label, hv, av, pct, hib, unit, explanation in edges:
                            val  = hv if team == home else av
                            opp  = av if team == home else hv
                            fmt  = f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
                            ofmt = f"{opp:.1f}" if abs(opp) < 100 else f"{opp:.0f}"
                            bar_w = min(int(pct * 2), 100)
                            # Truncate explanation for tooltip
                            tip = explanation.replace('"', "'")
                            lines.append(
                                f'<div style="margin-bottom:10px;padding:8px;background:#0a1628;border-radius:6px;border-left:3px solid {colour}">'
                                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">'
                                f'<span title="{tip}" style="color:white;font-size:0.8rem;cursor:help;border-bottom:1px dotted #555">{label} ℹ</span>'
                                f'<span style="color:{colour};font-weight:700;font-size:0.85rem">{fmt}<span style="color:#666;font-size:0.72rem;font-weight:400"> {unit}</span>'
                                f' <span style="color:#444;font-size:0.72rem">vs {ofmt}</span></span>'
                                f'</div>'
                                f'<div style="height:4px;background:#0f3460;border-radius:2px">'
                                f'<div style="width:{bar_w}%;height:100%;background:{colour};border-radius:2px;opacity:0.8"></div>'
                                f'</div>'
                                f'<div style="color:#555;font-size:0.68rem;margin-top:4px">{explanation[:120]}{"..." if len(explanation)>120 else ""}</div>'
                                f'</div>'
                            )
                        return "".join(lines)

                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"**✅ {home} advantages**")
                        st.markdown(edge_html_rich(home_edges_m, home, "#2ecc71"), unsafe_allow_html=True)
                    with ic2:
                        st.markdown(f"**✅ {away} advantages**")
                        st.markdown(edge_html_rich(away_edges_m, away, "#3498db"), unsafe_allow_html=True)

                    # ── Key numbers table ──────────────────────────────────
                    st.markdown("---")
                    st.markdown("**📊 All factors at a glance**")
                    all_rows = []
                    for label, hv, av, hib, unit, explanation in factor_meta:
                        if hv == 0 and av == 0:
                            continue
                        diff = hv - av if hib else av - hv
                        if diff > 0.5:   edge_tag = f"✅ {home}"
                        elif diff < -0.5: edge_tag = f"✅ {away}"
                        else:             edge_tag = "—  Even"
                        all_rows.append({
                            "Factor": label,
                            home: f"{hv:.1f} {unit}",
                            away: f"{av:.1f} {unit}",
                            "Edge": edge_tag,
                            "What it means": explanation[:80] + "..." if len(explanation) > 80 else explanation,
                        })
                    if all_rows:
                        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

                    # ── Narrative ─────────────────────────────────────────
                    st.markdown("---")
                    confidence = pred["home_win_prob"] if pred["home_win_prob"] > 50 else pred["away_win_prob"]
                    fav = winner
                    if confidence >= 75:   conf_label = "strong favourite"
                    elif confidence >= 62: conf_label = "moderate favourite"
                    else:                  conf_label = "slight favourite"

                    top_h = home_edges_m[0][0] if home_edges_m else None
                    top_a = away_edges_m[0][0] if away_edges_m else None

                    narrative = f"**{fav}** are the {conf_label} at **{confidence:.0f}% win probability**"
                    narrative += f" (predicted margin: ~{margin:.0f} pts). "
                    if fav == home and top_h:
                        narrative += f"Their biggest edge is **{top_h}**. "
                    elif fav == away and top_a:
                        narrative += f"Their biggest edge is **{top_a}**. "
                    if fav == home and top_a:
                        narrative += f"{away}'s best path to an upset is through their **{top_a}** advantage."
                    elif fav == away and top_h:
                        narrative += f"{home}'s best path to an upset is through their **{top_h}** advantage."
                    if not top_h and not top_a:
                        narrative += "Both teams are closely matched across most metrics — this is a genuine toss-up."

                    st.markdown(narrative)
    except Exception as e:
        import traceback
        st.warning(f"Could not load upcoming games: {e}")
        st.code(traceback.format_exc())

    # Accuracy by year chart
    st.markdown("---")
    st.markdown("## MODEL ACCURACY BY YEAR")
    avail = [f for f in CORE_FEATURES if f in df.columns]
    yearly = []
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year].dropna(subset=avail + ["home_win"])
        if len(ydf) < 10: continue
        preds = win_model.predict(ydf[avail].values)
        acc   = (preds == ydf["home_win"].values).mean()
        yearly.append({"year": year, "accuracy": acc * 100})
    if yearly:
        acc_df = pd.DataFrame(yearly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=acc_df["year"], y=acc_df["accuracy"],
            mode="lines+markers", line=dict(color="#e94560", width=3),
            marker=dict(size=8), fill="tozeroy", fillcolor="rgba(233,69,96,0.1)"))
        fig.add_hline(y=50, line_dash="dash", line_color="#555",
                      annotation_text="50% baseline")
        fig.add_hline(y=acc_df["accuracy"].mean(), line_dash="dot",
                      line_color="#2ecc71",
                      annotation_text=f"avg {acc_df['accuracy'].mean():.1f}%")
        dark_chart(fig)
        fig.update_layout(yaxis=dict(range=[40, 85], title="Accuracy %"),
                          xaxis=dict(title="Year"))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT A GAME
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Game":
    st.markdown("# PREDICT A GAME")

    c1, c2, c3 = st.columns(3)
    with c1: home_team = st.selectbox("🏠 Home Team", teams)
    with c2:
        away_opts = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_opts)
    with c3:
        venues = sorted(set(df["venue"].dropna().unique()))
        venue  = st.selectbox("📍 Venue", ["(Auto)"] + venues)
        if venue == "(Auto)": venue = ""

    # Load lineups for PAV
    lineup_df = load_lineups()
    lineup_strength = {}
    if not lineup_df.empty and not pav_df.empty:
        lineup_strength = compute_lineup_strength(lineup_df, pav_df)

    # ── Always-visible stats comparison ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### HEAD-TO-HEAD STATS COMPARISON")
    st.markdown("*Season averages — what each team does per game*")

    cur_year = datetime.now().year

    def get_ss(team, stat):
        if season_stats is None or season_stats.empty:
            return None
        row = season_stats[(season_stats["team"] == team) & (season_stats["year"] == cur_year)]
        if row.empty:
            row = season_stats[(season_stats["team"] == team) & (season_stats["year"] == cur_year - 1)]
        return float(row.iloc[0][stat]) if not row.empty and stat in row.columns else None

    STAT_DISPLAY = [
        ("avg_clearances",   "Clearances",         "🔵", False),
        ("avg_inside_50s",   "Inside 50s",          "🔴", False),
        ("avg_contested_possessions", "Contested Poss", "🟡", False),
        ("avg_tackles",      "Tackles",             "🟢", False),
        ("avg_hitouts",      "Hitouts",             "⚪", False),
        ("avg_disposals",    "Disposals",           "🔷", False),
        ("avg_marks",        "Marks",               "🟣", False),
        ("avg_clangers",     "Clangers",            "🔻", True),   # lower = better
    ]

    stat_rows = []
    for stat_col, label, icon, lower_better in STAT_DISPLAY:
        h_val = get_ss(home_team, stat_col)
        a_val = get_ss(away_team, stat_col)
        if h_val is None or a_val is None:
            continue
        if lower_better:
            h_better = h_val < a_val
            a_better = a_val < h_val
        else:
            h_better = h_val > a_val
            a_better = a_val > h_val
        stat_rows.append((label, icon, h_val, a_val, h_better, a_better, lower_better))

    if stat_rows:
        for label, icon, h_val, a_val, h_better, a_better, lower_better in stat_rows:
            col_l, col_bar, col_r = st.columns([1, 3, 1])
            with col_l:
                colour = "#2ecc71" if h_better else ("#e94560" if a_better else "#aaa")
                st.markdown(
                    f'<div style="text-align:right;font-size:1.1rem;font-weight:600;color:{colour}">{h_val:.1f}</div>'
                    f'<div style="text-align:right;font-size:0.7rem;color:#666">{home_team}</div>',
                    unsafe_allow_html=True
                )
            with col_bar:
                total = h_val + a_val if (h_val + a_val) > 0 else 1
                h_pct = int(h_val / total * 100)
                a_pct = 100 - h_pct
                # Highlight winner side
                h_col = "#2ecc71" if h_better else ("#e94560" if a_better else "#0f3460")
                a_col = "#2ecc71" if a_better else ("#e94560" if h_better else "#1a1a2e")
                bar_html = (
                    f'<div style="margin:4px 0">'
                    f'<div style="font-size:0.72rem;color:#aaa;text-align:center;margin-bottom:3px">{icon} {label}</div>'
                    f'<div style="display:flex;height:12px;border-radius:6px;overflow:hidden">'
                    f'<div style="width:{h_pct}%;background:{h_col}"></div>'
                    f'<div style="width:{a_pct}%;background:{a_col}"></div>'
                    f'</div></div>'
                )
                st.markdown(bar_html, unsafe_allow_html=True)
            with col_r:
                colour = "#2ecc71" if a_better else ("#e94560" if h_better else "#aaa")
                st.markdown(
                    f'<div style="text-align:left;font-size:1.1rem;font-weight:600;color:{colour}">{a_val:.1f}</div>'
                    f'<div style="text-align:left;font-size:0.7rem;color:#666">{away_team}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Season stats not yet available for these teams — AFL Tables data loads on first run.")

    st.markdown("---")

    if st.button("🔮 PREDICT", use_container_width=True):
        feats = build_prediction_features(
            home_team, away_team, venue,
            current_elos, team_stats,
            season_stats, lineup_strength,
            df
        )
        pred = predict_game(win_model, margin_model, feats)
        m    = pred["predicted_margin"]
        winner = home_team if m > 0 else away_team

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(mc(f"{pred['home_win_prob']}%", f"{home_team} Win Prob"), unsafe_allow_html=True)
        with c2: st.markdown(mc(f"{abs(m):.0f} pts", f"Margin ({winner})"), unsafe_allow_html=True)
        with c3: st.markdown(mc(f"{pred['away_win_prob']}%", f"{away_team} Win Prob"), unsafe_allow_html=True)

        # Win prob bar
        fig = go.Figure(go.Bar(
            x=[pred["home_win_prob"], pred["away_win_prob"]],
            y=[home_team, away_team], orientation="h",
            marker=dict(color=["#e94560", "#0f3460"]),
            text=[f"{pred['home_win_prob']}%", f"{pred['away_win_prob']}%"],
            textposition="inside"
        ))
        dark_chart(fig, height=180)
        fig.update_layout(xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

        # Full breakdown table
        hs  = team_stats.get(home_team, {})
        as_ = team_stats.get(away_team, {})

        with st.expander("📊 Full factor breakdown"):
            rows_data = [
                ("Elo Rating",          f"{current_elos.get(home_team,1500):.0f}",    f"{current_elos.get(away_team,1500):.0f}"),
                ("Avg Margin (last 5)", f"{hs.get('last5_avg',0):+.1f}",              f"{as_.get('last5_avg',0):+.1f}"),
                ("Current Streak",      f"{hs.get('streak',0):+d}",                   f"{as_.get('streak',0):+d}"),
                ("Last Game Margin",    f"{hs.get('last_margin',0):+.0f}",            f"{as_.get('last_margin',0):+.0f}"),
                ("Travel to Venue",     f"{feats['travel_home_km']:.0f} km",          f"{feats['travel_away_km']:.0f} km"),
                ("Days Rest",           str(feats['days_rest_home']),                  str(feats['days_rest_away'])),
            ]
            if feats.get("lineup_available"):
                rows_data.append(("PAV Rating (selected 22)",
                                  f"{feats.get('home_pav_total',0):.0f}",
                                  f"{feats.get('away_pav_total',0):.0f}"))
            bd_df = pd.DataFrame(rows_data, columns=["Factor", home_team, away_team])
            st.dataframe(bd_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM FORM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Team Form":
    st.markdown("# TEAM FORM ANALYSIS")
    selected = st.selectbox("Select Team", teams)
    n        = st.slider("Last N games", 5, 30, 15)
    form     = get_team_form_df(selected, n)

    if form.empty:
        st.warning("No data found.")
    else:
        hs   = team_stats.get(selected, {})
        wins = (form["result"] == "W").sum()
        losses = (form["result"] == "L").sum()
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(mc(f"{wins}W {losses}L", f"Last {n} Games"), unsafe_allow_html=True)
        with c2: st.markdown(mc(f"{form['margin'].mean():+.1f}", "Avg Margin"), unsafe_allow_html=True)
        with c3: st.markdown(mc(f"{hs.get('streak',0):+d}", "Current Streak"), unsafe_allow_html=True)
        with c4: st.markdown(mc(f"{current_elos.get(selected,1500):.0f}", "Elo Rating"), unsafe_allow_html=True)

        colors = ["#2ecc71" if m > 0 else "#e94560" for m in form["margin"]]
        fig = go.Figure(go.Bar(x=form["game_label"], y=form["margin"],
            marker_color=colors, text=form["result"], textposition="outside"))
        fig.add_hline(y=0, line_color="white", line_width=1)
        dark_chart(fig, 400)
        fig.update_layout(title=f"{selected} — Game Margins",
                          xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

        form["cumulative"] = form["margin"].cumsum()
        fig2 = go.Figure(go.Scatter(x=form["game_label"], y=form["cumulative"],
            mode="lines+markers", line=dict(color="#e94560", width=2),
            fill="tozeroy", fillcolor="rgba(233,69,96,0.1)"))
        dark_chart(fig2, 280)
        fig2.update_layout(title="Cumulative Margin",
                           xaxis=dict(tickangle=-45))
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ELO LADDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Elo Ladder":
    st.markdown("# ELO LADDER")
    st.markdown("*Rolling measure of true team strength, adjusted for opponent quality*")

    elo_rows = [{"Rank": i+1, "Team": t, "Elo": round(e),
                 "Streak": team_stats.get(t, {}).get("streak", 0),
                 "Avg Margin (L5)": team_stats.get(t, {}).get("last5_avg", 0)}
                for i, (t, e) in enumerate(
                    sorted(current_elos.items(), key=lambda x: -x[1]))]
    elo_df = pd.DataFrame(elo_rows)

    fig = go.Figure(go.Bar(
        x=elo_df["Elo"], y=elo_df["Team"], orientation="h",
        marker=dict(color=elo_df["Elo"],
                    colorscale=[[0, "#0f3460"], [0.5, "#e94560"], [1, "#ff6b6b"]]),
        text=elo_df["Elo"], textposition="inside"
    ))
    dark_chart(fig, 600)
    fig.update_layout(xaxis=dict(range=[1300, 1700]),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(elo_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STATS LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Team Stats":
    st.markdown("# TEAM STATS LEADERBOARD")
    st.markdown("*Season averages per game — all 18 teams ranked*")

    cur_year = datetime.now().year

    if season_stats is None or season_stats.empty:
        st.warning("Season stats not available — AFL Tables data will load on the first full run.")
    else:
        # Get latest available year
        avail_years = sorted(season_stats["year"].unique(), reverse=True)
        sel_year = st.selectbox("Season", avail_years, index=0)
        ss_year = season_stats[season_stats["year"] == sel_year].copy()

        if ss_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:
            LEADERBOARD_STATS = [
                ("avg_clearances",               "Clearances",          False, "#3498db"),
                ("avg_inside_50s",               "Inside 50s",           False, "#e94560"),
                ("avg_contested_possessions",     "Contested Poss",       False, "#f39c12"),
                ("avg_tackles",                  "Tackles",              False, "#2ecc71"),
                ("avg_hitouts",                  "Hitouts",              False, "#9b59b6"),
                ("avg_disposals",                "Disposals",            False, "#1abc9c"),
                ("avg_marks",                    "Marks",                False, "#e67e22"),
                ("avg_marks_inside_50",          "Marks Inside 50",      False, "#e91e63"),
                ("avg_rebound_50s",              "Rebound 50s",          False, "#00bcd4"),
                ("avg_clangers",                 "Clangers",             True,  "#e74c3c"),
                ("avg_frees_for",                "Frees For",            False, "#27ae60"),
                ("avg_frees_against",            "Frees Against",        True,  "#c0392b"),
            ]

            stat_tab_labels = [s[1] for s in LEADERBOARD_STATS
                               if s[0] in ss_year.columns]
            if not stat_tab_labels:
                st.warning("No stat columns found in scraped data.")
            else:
                selected_stat_label = st.selectbox("Rank by stat", stat_tab_labels)
                sel_stat = next(s for s in LEADERBOARD_STATS if s[1] == selected_stat_label)
                stat_col, stat_label, lower_better, bar_colour = sel_stat

                if stat_col not in ss_year.columns:
                    st.warning(f"Stat '{stat_col}' not in data for {sel_year}.")
                else:
                    ranked = ss_year[["team", stat_col]].dropna().copy()
                    ranked[stat_col] = pd.to_numeric(ranked[stat_col], errors="coerce")
                    ranked = ranked.dropna().sort_values(stat_col, ascending=lower_better).reset_index(drop=True)
                    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
                    ranked.columns = ["Rank", "Team", stat_label]

                    fig = go.Figure(go.Bar(
                        x=ranked[stat_label], y=ranked["Team"],
                        orientation="h",
                        marker_color=bar_colour,
                        text=ranked[stat_label].apply(lambda x: f"{x:.1f}"),
                        textposition="outside"
                    ))
                    dark_chart(fig, 560)
                    fig.update_layout(
                        title=f"{sel_year} Season — {stat_label} per game {'(lower = better)' if lower_better else '(higher = better)'}",
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(title=f"Avg {stat_label} per game")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### All Stats Table")
                    display_cols = ["team"] + [s[0] for s in LEADERBOARD_STATS if s[0] in ss_year.columns]
                    rename_map = {s[0]: s[1] for s in LEADERBOARD_STATS}
                    rename_map["team"] = "Team"
                    full_table = ss_year[display_cols].copy()
                    for c in display_cols[1:]:
                        full_table[c] = pd.to_numeric(full_table[c], errors="coerce").round(1)
                    full_table = full_table.sort_values(stat_col, ascending=lower_better).reset_index(drop=True)
                    full_table = full_table.rename(columns=rename_map)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)

                    # Radar chart for team comparison
                    st.markdown("---")
                    st.markdown("### TEAM PROFILE COMPARISON")
                    st.markdown("*Select two teams to compare across all stats*")

                    radar_cols = st.columns(2)
                    with radar_cols[0]:
                        radar_home = st.selectbox("Team A", sorted(ss_year["team"].unique()), key="radar_h")
                    with radar_cols[1]:
                        radar_away_opts = [t for t in sorted(ss_year["team"].unique()) if t != radar_home]
                        radar_away = st.selectbox("Team B", radar_away_opts, key="radar_a")

                    radar_stats = [s for s in LEADERBOARD_STATS
                                   if s[0] in ss_year.columns and not s[3] == "#e74c3c"][:7]

                    h_row = ss_year[ss_year["team"] == radar_home]
                    a_row = ss_year[ss_year["team"] == radar_away]

                    if not h_row.empty and not a_row.empty:
                        categories = [s[1] for s in radar_stats]
                        # Normalise 0–1 across all teams for radar
                        h_vals, a_vals = [], []
                        for s in radar_stats:
                            col_data = pd.to_numeric(ss_year[s[0]], errors="coerce")
                            mn, mx = col_data.min(), col_data.max()
                            rng = mx - mn if mx != mn else 1
                            hv = float(pd.to_numeric(h_row.iloc[0][s[0]], errors="coerce") or 0)
                            av = float(pd.to_numeric(a_row.iloc[0][s[0]], errors="coerce") or 0)
                            # For lower-better stats, invert so "bigger = better" on radar
                            if s[2]:
                                h_vals.append(1 - (hv - mn) / rng)
                                a_vals.append(1 - (av - mn) / rng)
                            else:
                                h_vals.append((hv - mn) / rng)
                                a_vals.append((av - mn) / rng)

                        cats_closed = categories + [categories[0]]
                        h_closed    = h_vals + [h_vals[0]]
                        a_closed    = a_vals + [a_vals[0]]

                        fig_r = go.Figure()
                        fig_r.add_trace(go.Scatterpolar(
                            r=h_closed, theta=cats_closed,
                            fill="toself", name=radar_home,
                            line=dict(color="#e94560"), fillcolor="rgba(233,69,96,0.2)"
                        ))
                        fig_r.add_trace(go.Scatterpolar(
                            r=a_closed, theta=cats_closed,
                            fill="toself", name=radar_away,
                            line=dict(color="#3498db"), fillcolor="rgba(52,152,219,0.2)"
                        ))
                        fig_r.update_layout(
                            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                            font=dict(color="white"),
                            polar=dict(
                                bgcolor="#16213e",
                                radialaxis=dict(visible=True, range=[0, 1],
                                                gridcolor="#0f3460", color="#aaa"),
                                angularaxis=dict(gridcolor="#0f3460", color="white")
                            ),
                            legend=dict(bgcolor="#1a1a2e"),
                            height=420, margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fig_r, use_container_width=True)
                        st.caption("*Normalised 0–1 across all 18 teams. For Clangers/Frees Against, higher = better (inverted).*")

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Importance":
    st.markdown("# FEATURE IMPORTANCE")
    gain = metrics.get("accuracy_gain", 0)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(mc(f"{metrics['win_accuracy']*100:.1f}%", "Full Model Accuracy"), unsafe_allow_html=True)
    with c2: st.markdown(mc(f"{metrics['base_accuracy']*100:.1f}%", "Elo-Only Baseline"), unsafe_allow_html=True)
    with c3:
        col = "#2ecc71" if gain > 0 else "#e94560"
        st.markdown(mc(f'<span style="color:{col}">{gain*100:+.1f}%</span>',
                       "Accuracy Gain from All Features"), unsafe_allow_html=True)

    st.markdown("---")

    GROUP_COLORS = {
        "Elo":            "#e94560",
        "Form (rolling)": "#e67e22",
        "Travel":         "#f1c40f",
        "Rest days":      "#2ecc71",
        "Streak":         "#1abc9c",
        "Last margin":    "#3498db",
        "Season stats":   "#9b59b6",
        "PAV lineup":     "#e91e8c",
        "Other":          "#666",
    }

    fi = fi_df.copy()
    fi["label"] = fi["feature"].str.replace("_", " ").str.title()
    fi["color"] = fi["group"].map(GROUP_COLORS).fillna("#666")

    fig = go.Figure(go.Bar(
        x=fi["importance"], y=fi["label"], orientation="h",
        marker_color=fi["color"],
        text=fi["importance"].apply(lambda x: f"{x:.4f}"),
        textposition="outside"
    ))
    dark_chart(fig, 550)
    fig.update_layout(
        title="Feature Importance (GBM impurity-based)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Importance Score")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    legend_html = " &nbsp; ".join(
        f'<span style="color:{c}">■ {g}</span>'
        for g, c in GROUP_COLORS.items() if g != "Other"
    )
    st.markdown(f"<div style='margin-top:8px'>{legend_html}</div>", unsafe_allow_html=True)

    # Group-level rollup
    st.markdown("### By Feature Group")
    group_imp = fi.groupby("group")["importance"].sum().reset_index()
    group_imp = group_imp.sort_values("importance", ascending=False)
    group_imp["color"] = group_imp["group"].map(GROUP_COLORS).fillna("#666")
    fig2 = go.Figure(go.Bar(
        x=group_imp["importance"], y=group_imp["group"], orientation="h",
        marker_color=group_imp["color"],
        text=group_imp["importance"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))
    dark_chart(fig2, 350)
    fig2.update_layout(yaxis=dict(autorange="reversed"),
                       xaxis=dict(title="Total Group Importance"))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Backtest":
    st.markdown("# WALK-FORWARD BACKTEST")
    st.markdown("*Train on years 1..N-1, predict year N — true out-of-sample performance*")

    min_train = st.slider("Minimum training years before testing", 2, 5, 3)

    with st.spinner("Running walk-forward backtest..."):
        avail_feats = [f for f in CORE_FEATURES if f in df.columns]
        bt_df = run_walk_forward_backtest(df, avail_feats, min_train)

    if bt_df.empty:
        st.warning("Not enough data for backtest with current settings.")
    else:
        yearly_acc = compute_yearly_accuracy(bt_df)
        overall_acc  = bt_df["correct"].mean() * 100
        overall_games = len(bt_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(mc(f"{overall_acc:.1f}%", "Out-of-Sample Accuracy",
                                f"{overall_games} games tested"), unsafe_allow_html=True)
        with c2: st.markdown(mc(f"{yearly_acc['brier_score'].mean():.3f}",
                                "Avg Brier Score", "Lower = better calibrated"), unsafe_allow_html=True)
        with c3:
            # Upset detection: how often did we correctly call upsets?
            upsets = bt_df[bt_df["prob"] < 0.4]
            upset_acc = upsets["correct"].mean() * 100 if len(upsets) > 0 else 0
            st.markdown(mc(f"{upset_acc:.1f}%", "Upset Detection Accuracy",
                           f"{len(upsets)} games < 40% prob"), unsafe_allow_html=True)
        with c4:
            big_fav = bt_df[bt_df["prob"] > 0.7]
            fav_acc = big_fav["correct"].mean() * 100 if len(big_fav) > 0 else 0
            st.markdown(mc(f"{fav_acc:.1f}%", "Big Favourite Accuracy",
                           f"{len(big_fav)} games > 70% prob"), unsafe_allow_html=True)

        st.markdown("---")

        # Accuracy by year
        st.markdown("### Out-of-Sample Accuracy by Year")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly_acc["year"], y=yearly_acc["accuracy"],
            marker_color="#e94560",
            text=yearly_acc["accuracy"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside"))
        fig.add_hline(y=50, line_dash="dash", line_color="#555")
        fig.add_hline(y=overall_acc, line_dash="dot", line_color="#2ecc71",
                      annotation_text=f"avg {overall_acc:.1f}%")
        dark_chart(fig, 350)
        fig.update_layout(yaxis=dict(range=[40, 85], title="Accuracy %"))
        st.plotly_chart(fig, use_container_width=True)

        # Calibration plot
        st.markdown("### Probability Calibration")
        st.markdown("*Are our confidence levels accurate? Bars should follow the diagonal.*")
        bins = pd.cut(bt_df["prob"], bins=10)
        cal  = bt_df.groupby(bins, observed=True).agg(
            mean_prob=("prob", "mean"),
            actual_rate=("actual", "mean"),
            n=("actual", "count")
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=cal["mean_prob"], y=cal["actual_rate"],
            marker_color="#e94560", name="Actual win rate",
            text=cal["n"].apply(lambda x: f"n={x}"), textposition="outside"))
        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#2ecc71", dash="dash", width=2), name="Perfect calibration"))
        dark_chart(fig2, 350)
        fig2.update_layout(
            xaxis=dict(title="Predicted probability", range=[0, 1]),
            yaxis=dict(title="Actual win rate", range=[0, 1]),
            legend=dict(bgcolor="#1a1a2e")
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Ablation test
        st.markdown("---")
        st.markdown("### Feature Group Ablation")
        st.markdown("*Which feature groups actually earn their place? (Accuracy when each group is removed)*")

        if st.button("🧪 Run Ablation Test (takes ~30 seconds)"):
            with st.spinner("Running ablation test..."):
                ablation_df = ablation_test(df, FEATURE_GROUPS, min_train)
            if not ablation_df.empty:
                ablation_df["color"] = ablation_df.get("interpretation", "➡️ Neutral").apply(
                    lambda x: "#2ecc71" if "Helps" in str(x)
                    else ("#e94560" if "Hurts" in str(x) else "#f39c12")
                )
                fig3 = go.Figure(go.Bar(
                    x=ablation_df["delta"],
                    y=ablation_df["group"],
                    orientation="h",
                    marker_color=ablation_df["color"],
                    text=ablation_df["delta"].apply(lambda x: f"{x:+.2f}%"),
                    textposition="outside"
                ))
                fig3.add_vline(x=0, line_color="white", line_width=1)
                dark_chart(fig3, 400)
                fig3.update_layout(
                    title="Accuracy change when feature group is REMOVED (negative = group helps)",
                    xaxis=dict(title="Δ Accuracy %"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.dataframe(ablation_df[["group", "accuracy", "delta",
                                          "interpretation", "n_features"]],
                             use_container_width=True, hide_index=True)

        # Margin backtest
        st.markdown("---")
        st.markdown("### Margin Prediction Error by Year")
        st.markdown("*Mean Absolute Error in points — how far off is our margin prediction?*")
        margin_bt = margin_prediction_backtest(df, avail_feats, min_train)
        if not margin_bt.empty:
            fig4 = go.Figure(go.Bar(
                x=margin_bt["year"], y=margin_bt["mae_points"],
                marker_color="#9b59b6",
                text=margin_bt["mae_points"].apply(lambda x: f"{x:.1f} pts"),
                textposition="outside"
            ))
            dark_chart(fig4, 300)
            fig4.update_layout(yaxis=dict(title="MAE (points)"))
            st.plotly_chart(fig4, use_container_width=True)
            avg_mae = margin_bt["mae_points"].mean()
            st.info(f"📏 Average margin prediction error: **{avg_mae:.1f} points** out-of-sample")

# ═══════════════════════════════════════════════════════════════════════════════
# LINEUP STRENGTH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👕 Lineup Strength":
    st.markdown("# LINEUP STRENGTH (PAV)")
    st.markdown("*Player Approximate Value — rates each player's contribution. Updates when lineups are announced (Thursday).*")

    lineup_df = load_lineups()

    if lineup_df.empty:
        st.info("No lineup data available yet — lineups are typically announced Thursday. Check back then.")
    else:
        if pav_df.empty:
            st.warning("PAV data unavailable.")
        else:
            lineup_strength = compute_lineup_strength(lineup_df, pav_df)

            if not lineup_strength:
                st.warning("Could not match lineup players to PAV ratings.")
            else:
                st.markdown(f"**{len(lineup_strength)} teams with lineup data**")

                # Team strength bar chart
                rows = []
                for team, data in sorted(lineup_strength.items(),
                                          key=lambda x: -x[1].get("PAV_total", 0)):
                    rows.append({
                        "Team":            team,
                        "PAV Total":       round(data.get("PAV_total", 0), 1),
                        "PAV Off":         round(data.get("PAV_off",   0), 1),
                        "PAV Mid":         round(data.get("PAV_mid",   0), 1),
                        "PAV Def":         round(data.get("PAV_def",   0), 1),
                        "Players Matched": data.get("n_players_matched", 0),
                    })
                ls_df = pd.DataFrame(rows)

                fig = go.Figure()
                fig.add_trace(go.Bar(name="Offensive", x=ls_df["Team"],
                    y=ls_df["PAV Off"], marker_color="#e94560"))
                fig.add_trace(go.Bar(name="Midfield",  x=ls_df["Team"],
                    y=ls_df["PAV Mid"], marker_color="#3498db"))
                fig.add_trace(go.Bar(name="Defensive", x=ls_df["Team"],
                    y=ls_df["PAV Def"], marker_color="#2ecc71"))
                fig.update_layout(barmode="stack")
                dark_chart(fig, 420)
                fig.update_layout(
                    title="This Week's Lineup Strength by Component",
                    xaxis=dict(tickangle=-45),
                    legend=dict(bgcolor="#1a1a2e")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ls_df, use_container_width=True, hide_index=True)

    # ── In/Out tracker ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### IN / OUT TRACKER")
    st.markdown("*Who's in, out, or named as emergency vs last round — sorted by PAV rating*")

    if lineup_df.empty:
        st.info("No lineup data available yet.")
    elif pav_df.empty:
        st.warning("PAV data needed for in/out tracker.")
    else:
        # Get previous round lineup for comparison
        cur_year = datetime.now().year
        prev_lineup_df = pd.DataFrame()

        # Find which rounds are in current lineup data
        if "round" in lineup_df.columns:
            rounds_avail = sorted(lineup_df["round"].dropna().unique())
            if len(rounds_avail) >= 2:
                cur_round  = rounds_avail[-1]
                prev_round = rounds_avail[-2]
                prev_lineup_df = lineup_df[lineup_df["round"] == prev_round].copy()
                cur_lineup_df  = lineup_df[lineup_df["round"] == cur_round].copy()
            else:
                cur_lineup_df = lineup_df.copy()
        else:
            cur_lineup_df = lineup_df.copy()

        # Build PAV lookup
        pav_numeric = pav_df.copy()
        for col in ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]:
            if col in pav_numeric.columns:
                pav_numeric[col] = pd.to_numeric(pav_numeric[col], errors="coerce").fillna(0)

        pav_lookup = {}
        if not pav_numeric.empty:
            latest_pav_year = pav_numeric["year"].max() if "year" in pav_numeric.columns else cur_year
            latest_pav = pav_numeric[pav_numeric["year"] == latest_pav_year]
            fn_col = "firstname" if "firstname" in latest_pav.columns else "givenname"
            for _, row in latest_pav.iterrows():
                key = (str(row.get(fn_col, "")).strip().lower(),
                       str(row.get("surname", "")).strip().lower())
                pav_lookup[key] = {
                    "PAV_total": float(row.get("PAV_total", 0) or 0),
                    "team_pav":  str(row.get("team", "")),
                }

        def build_player_set(ldf):
            players = {}
            fn_col = "firstname" if "firstname" in ldf.columns else "givenname"
            team_col = "teamname" if "teamname" in ldf.columns else "team"
            for _, row in ldf.iterrows():
                fn = str(row.get(fn_col, "") or "").strip()
                sn = str(row.get("surname", "") or "").strip()
                team = str(row.get(team_col, "") or "").strip()
                status = str(row.get("status", "") or "").strip()
                key = (fn.lower(), sn.lower())
                pav = pav_lookup.get(key, {}).get("PAV_total", 0)
                players[key] = {
                    "name": f"{fn} {sn}",
                    "team": team,
                    "status": status,
                    "pav": pav,
                }
            return players

        cur_players  = build_player_set(cur_lineup_df)
        prev_players = build_player_set(prev_lineup_df) if not prev_lineup_df.empty else {}

        # Classify changes
        all_teams = sorted(set(v["team"] for v in cur_players.values()))
        sel_inout_team = st.selectbox("Filter by team (or All)", ["All teams"] + all_teams)

        inout_rows = []
        for key, data in cur_players.items():
            if sel_inout_team != "All teams" and data["team"] != sel_inout_team:
                continue
            was_in = key in prev_players
            status = data["status"].lower()
            if "emerg" in status:
                change = "🟡 Emergency"
            elif not was_in and prev_players:
                change = "🟢 IN"
            else:
                change = "✅ Named"
            inout_rows.append({
                "Team": data["team"],
                "Player": data["name"],
                "Status": change,
                "PAV": round(data["pav"], 1),
            })

        # Also find players who dropped out
        for key, data in prev_players.items():
            if key not in cur_players:
                if sel_inout_team != "All teams" and data["team"] != sel_inout_team:
                    continue
                inout_rows.append({
                    "Team": data["team"],
                    "Player": data["name"],
                    "Status": "🔴 OUT",
                    "PAV": round(data["pav"], 1),
                })

        if inout_rows:
            inout_df = pd.DataFrame(inout_rows).sort_values(
                ["Team", "PAV"], ascending=[True, False]
            ).reset_index(drop=True)

            # Show INs and OUTs first for quick scan
            priority = inout_df[inout_df["Status"].isin(["🟢 IN", "🔴 OUT", "🟡 Emergency"])]
            rest     = inout_df[~inout_df["Status"].isin(["🟢 IN", "🔴 OUT", "🟡 Emergency"])]

            if not priority.empty:
                st.markdown("#### Changes this week")
                st.dataframe(priority, use_container_width=True, hide_index=True)
                st.markdown("#### Full named squad")

            st.dataframe(rest if not priority.empty else inout_df,
                         use_container_width=True, hide_index=True)
        else:
            st.info("No player data to display.")

    # ── Top rated players ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### TOP RATED PLAYERS (PAV)")
    st.markdown("*Player Approximate Value — career rating for most recent season*")

    if not pav_df.empty:
        pav_show = pav_df.copy()
        for col in ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]:
            if col in pav_show.columns:
                pav_show[col] = pd.to_numeric(pav_show[col], errors="coerce")

        top_year = int(pav_show["year"].max()) if "year" in pav_show.columns else datetime.now().year
        top = pav_show[pav_show["year"] == top_year].copy()

        pav_team_filter = st.selectbox("Filter by team", ["All"] + sorted(top["team"].dropna().unique().tolist()), key="pav_team")
        if pav_team_filter != "All":
            top = top[top["team"] == pav_team_filter]

        if "PAV_total" in top.columns:
            top = top.sort_values("PAV_total", ascending=False).head(40)
            fn_col = "firstname" if "firstname" in top.columns else "givenname"
            display_cols = [c for c in [fn_col, "surname", "team", "PAV_total",
                                         "PAV_off", "PAV_mid", "PAV_def", "games"]
                            if c in top.columns]
            rename = {fn_col: "First Name", "surname": "Surname", "team": "Team",
                      "PAV_total": "PAV Total", "PAV_off": "Offensive",
                      "PAV_mid": "Midfield", "PAV_def": "Defensive", "games": "Games"}
            st.dataframe(top[display_cols].rename(columns=rename).reset_index(drop=True),
                         use_container_width=True)
    else:
        st.info("PAV data not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Analysis":
    st.markdown("# AI MODEL ANALYSIS")
    st.markdown("*Claude analyses your model's current performance and suggests improvements*")

    # ── Gather all model data to send ────────────────────────────────────────
    fi_summary = ""
    if fi_df is not None and not fi_df.empty:
        top_feats = fi_df.head(10)[["feature", "group", "importance"]].copy()
        top_feats["importance"] = top_feats["importance"].round(4)
        fi_summary = top_feats.to_string(index=False)

    yearly_acc = []
    avail_feats = [f for f in CORE_FEATURES if f in df.columns]
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year].dropna(subset=avail_feats + ["home_win"])
        if len(ydf) < 10: continue
        preds = win_model.predict(ydf[avail_feats].values)
        acc = (preds == ydf["home_win"].values).mean()
        yearly_acc.append(f"  {year}: {acc*100:.1f}%")
    yearly_acc_str = "\n".join(yearly_acc)

    # Team travel records from df
    travel_records = []
    for team in sorted(teams):
        away = df[(df["ateam"] == team) & (df["travel_away_km"] >= 1000)] if "travel_away_km" in df.columns else pd.DataFrame()
        if len(away) >= 5:
            margins = (away["ascore"] - away["hscore"]).dropna()
            wr = (margins > 0).mean()
            travel_records.append(f"  {team}: {len(away)} long trips, {wr*100:.0f}% win rate, avg margin {margins.mean():+.1f}")
    travel_str = "\n".join(travel_records[:10])

    context = f"""
You are analysing an AFL match prediction model. Here is everything we know about it:

## Model Overview
- Algorithm: Gradient Boosting (scikit-learn) for both win probability and margin
- Training data: {start_year} to present ({metrics['n_games']:,} games)
- Features: {metrics['n_features']} total across Elo, form, travel, rest, streak, season stats, PAV lineup

## Performance Metrics
- Win prediction accuracy (5-fold CV): {metrics['win_accuracy']*100:.1f}%
- Elo-only baseline accuracy: {metrics['base_accuracy']*100:.1f}%
- Accuracy gain from all features vs Elo-only: {metrics.get('accuracy_gain',0)*100:+.1f}%
- Margin prediction R²: {metrics['margin_r2']:.3f}
- Margin R² std dev: {metrics['margin_r2_std']:.3f}

## Accuracy by Year (in-sample)
{yearly_acc_str}

## Top 10 Feature Importances (GBM impurity-based)
{fi_summary}

## Feature Groups in Model
- Elo rating differential (home advantage = 50 pts)
- Rolling form: last 3/5 game avg margin, consistency (std dev)
- Travel: raw km, Perth flag, travel×rest fatigue interaction
- Rest days: days since last game, capped at 21 (summer break → neutral 7 days)
- Streak: signed win/loss streak
- Last margin: last game, last 3, last 5 averages
- Season stats: clearances, inside 50s, contested possessions, tackles, hitouts, clangers (from AFL Tables)
- PAV lineup: Player Approximate Value sum for selected 22 (when lineups announced)

## Previous Ablation Test Results (rough)
- Elo: -1.85% when removed (clear positive contribution)
- Form rolling: -0.1% (neutral)
- Travel raw: +0.39% when removed (slightly hurts accuracy — possibly redundant with Elo)
- Rest days: -0.1% (neutral)
- Streak: -0.29% (neutral)
- Season stats: -0.05% (neutral)
- PAV lineup: +0.05% (neutral — limited by season-level proxy, not per-game lineups)

## Team Travel Records (long trips >1000km)
{travel_str}

## Known Limitations
- Season stats are season averages, not rolling — they don't capture mid-season form shifts
- PAV is season-level proxy until Thursday lineup announcements
- No weather/conditions data
- No head-to-head historical matchup data
- No individual player injury data beyond PAV lineup
- Training data mixes pre and post rule-change eras (2019 6-6-6 alignment rule)

Please provide:
1. A frank assessment of model strengths and weaknesses
2. The 3-5 highest-impact improvements we could realistically make
3. Any concerns about the current feature set (overfitting, redundancy, data leakage)
4. Your take on why the season stats and travel features showed neutral ablation results
5. Specific suggestions for new data sources or features worth pursuing
"""

    col_run, col_focus = st.columns([2, 1])
    with col_focus:
        focus = st.selectbox("Analysis focus", [
            "Full overview",
            "Feature engineering only",
            "Travel & fatigue deep dive",
            "How to improve accuracy beyond 67%",
            "Data quality & leakage risks",
        ])
    with col_run:
        run_analysis = st.button("🤖 Run AI Analysis", use_container_width=True)

    if focus != "Full overview":
        context += f"\n\nFocus your analysis specifically on: {focus}"

    if run_analysis:
        with st.spinner("Claude is analysing your model..."):
            try:
                import requests as _req
                # Get API key from Streamlit secrets
                try:
                    api_key = st.secrets["ANTHROPIC_API_KEY"]
                except Exception:
                    st.error("No API key found. Add ANTHROPIC_API_KEY to your Streamlit secrets at share.streamlit.io → App settings → Secrets.")
                    st.stop()

                response = _req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1500,
                        "messages": [{"role": "user", "content": context}]
                    },
                    timeout=60
                )
                data = response.json()
                analysis = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        analysis += block["text"]

                if analysis:
                    st.markdown("---")
                    st.markdown(analysis)
                else:
                    st.warning(f"No response received. API response: {data}")

            except Exception as e:
                st.error(f"API call failed: {e}")

    else:
        st.markdown("---")
        st.markdown("""
**What this does:** Sends your full model stats — feature importances, accuracy by year,
ablation results, travel records, and known limitations — to Claude for analysis.

You'll get back a frank assessment of what's working, what's not, and the highest-impact
things to build next. Pick a focus area if you want to drill into something specific.
""")
        # Show the context that will be sent
        with st.expander("📋 Preview data being sent to Claude"):
            st.text(context)

# ═══════════════════════════════════════════════════════════════════════════════
# VALUE BETS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Value Bets":
    st.markdown("# VALUE BETS")
    st.markdown("*Compares our model's probabilities against bookmaker odds and Squiggle consensus to find edges*")

    # ── Squiggle consensus (free, no key needed) ──────────────────────────────
    st.markdown("## 📡 Squiggle Model Consensus")
    st.markdown("*Aggregated win probabilities from ~15 computer models*")

    try:
        consensus_df = get_squiggle_consensus()
        if consensus_df.empty:
            st.info("No Squiggle consensus data available for current round.")
        else:
            # Match consensus to our predictions
            rows = []
            for _, row in consensus_df.iterrows():
                ht = normalise_team(str(row.get("hteam","")))
                at = normalise_team(str(row.get("ateam","")))
                if ht not in current_elos or at not in current_elos:
                    continue

                sq_prob = float(row["consensus_home_prob"])
                n_models = int(row["n_models"])

                # Get our model's probability using inline builder
                def _sf(v, d=0.0):
                    try:
                        if v is None: return float(d)
                        if hasattr(v,'iloc'): v=v.iloc[-1]
                        if hasattr(v,'item'): v=v.item()
                        return float(v)
                    except: return float(d)

                from data.fetcher import travel_distance_km
                _PV = {"Optus Stadium","Perth Stadium","Subiaco Oval"}
                h_elo = _sf(current_elos.get(ht,1500),1500)
                a_elo = _sf(current_elos.get(at,1500),1500)
                hs_  = team_stats.get(ht,{})
                as__ = team_stats.get(at,{})
                h_form = _sf(hs_.get("last5_avg",0))
                a_form = _sf(as__.get("last5_avg",0))

                our_feats = {
                    "elo_diff": h_elo - a_elo + 50.0,
                    "form_diff": h_form - a_form,
                    "home_form": h_form, "away_form": a_form,
                    "home_consistency": _sf(hs_.get("last5_std",20),20),
                    "away_consistency": _sf(as__.get("last5_std",20),20),
                    "travel_diff": 0.0, "travel_home_km": 0.0, "travel_away_km": 0.0,
                    "travel_fatigue_diff": 0.0, "home_travel_fatigue": 0.0, "away_travel_fatigue": 0.0,
                    "travel_win_rate_diff": 0.0, "travel_margin_diff": 0.0, "perth_win_rate_diff": 0.0,
                    "is_perth_game": 0.0, "days_rest_diff": 0.0, "days_rest_home": 7.0, "days_rest_away": 7.0,
                    "streak_diff": _sf(hs_.get("streak",0))-_sf(as__.get("streak",0)),
                    "home_streak": _sf(hs_.get("streak",0)), "away_streak": _sf(as__.get("streak",0)),
                    "last_margin_diff": _sf(hs_.get("last_margin",0))-_sf(as__.get("last_margin",0)),
                    "last3_diff": _sf(hs_.get("last3_avg",0))-_sf(as__.get("last3_avg",0)),
                    "last5_diff": h_form - a_form,
                    "cl_diff":0.0,"i50_diff":0.0,"cp_diff":0.0,"tk_diff":0.0,"ho_diff":0.0,"clanger_diff":0.0,
                    "pav_total_diff":0.0,"pav_off_diff":0.0,"pav_def_diff":0.0,"pav_mid_diff":0.0,
                }
                pred = predict_game(win_model, margin_model, our_feats)
                our_prob = pred["home_win_prob"] / 100.0
                diff = our_prob - sq_prob

                rows.append({
                    "Home": ht, "Away": at,
                    "Our Model": f"{our_prob*100:.1f}%",
                    "Squiggle Consensus": f"{sq_prob*100:.1f}%",
                    "Difference": f"{diff*100:+.1f}%",
                    "Models": n_models,
                    "_diff": diff,
                    "_our": our_prob,
                    "_sq": sq_prob,
                })

            if rows:
                sq_display = pd.DataFrame(rows)

                def colour_diff(val):
                    try:
                        v = float(val.replace("%",""))
                        if v > 5: return "color: #2ecc71; font-weight: bold"
                        if v < -5: return "color: #e74c3c; font-weight: bold"
                    except: pass
                    return ""

                st.dataframe(
                    sq_display[["Home","Away","Our Model","Squiggle Consensus","Difference","Models"]]
                    .style.applymap(colour_diff, subset=["Difference"]),
                    use_container_width=True, hide_index=True
                )
                st.caption("🟢 Green = our model is more bullish on home team than consensus  🔴 Red = more bearish")
    except Exception as e:
        st.warning(f"Could not load Squiggle consensus: {e}")

    st.markdown("---")

    # ── Live bookmaker odds (requires API key) ────────────────────────────────
    st.markdown("## 🎰 Bookmaker Value Bets")

    odds_key = ""
    try:
        odds_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        pass

    if not odds_key:
        st.info("""
**To enable live bookmaker odds:**
1. Get a free API key at [the-odds-api.com](https://the-odds-api.com) (~500 free requests/month, plenty for weekly use)
2. In Streamlit Cloud → your app → **Settings → Secrets**, add:
```
ODDS_API_KEY = "your_key_here"
```
3. Refresh the app

Free tier covers TAB, Sportsbet, Neds, Ladbrokes and more.
""")
    else:
        with st.spinner("Fetching live odds..."):
            odds_df = get_odds_api(odds_key)

        if odds_df.empty:
            st.info("No AFL odds available right now — check back closer to game day.")
        else:
            # Pick best (highest) odds per game across bookmakers
            best_odds = (
                odds_df.groupby(["home_team","away_team"])
                .agg(best_home_odds=("home_odds","max"), best_away_odds=("away_odds","max"),
                     bookmakers=("bookmaker", lambda x: ", ".join(sorted(set(x)))))
                .reset_index()
            )

            value_rows = []
            for _, row in best_odds.iterrows():
                ht = normalise_team(row["home_team"])
                at = normalise_team(row["away_team"])
                if ht not in current_elos or at not in current_elos:
                    continue

                h_odds = float(row["best_home_odds"])
                a_odds = float(row["best_away_odds"])
                h_implied = 1.0 / h_odds
                a_implied = 1.0 / a_odds
                # Remove vig to get fair implied probs
                total_implied = h_implied + a_implied
                h_fair = h_implied / total_implied
                a_fair = a_implied / total_implied
                vig = (total_implied - 1.0) * 100

                # Our model probability
                hs_ = team_stats.get(ht, {})
                as__ = team_stats.get(at, {})
                def _sf2(v, d=0.0):
                    try:
                        if v is None: return float(d)
                        if hasattr(v,'iloc'): v=v.iloc[-1]
                        if hasattr(v,'item'): v=v.item()
                        return float(v)
                    except: return float(d)
                h_elo2 = _sf2(current_elos.get(ht,1500),1500)
                a_elo2 = _sf2(current_elos.get(at,1500),1500)
                h_form2 = _sf2(hs_.get("last5_avg",0))
                a_form2 = _sf2(as__.get("last5_avg",0))
                our_feats2 = {
                    "elo_diff": h_elo2-a_elo2+50.0, "form_diff": h_form2-a_form2,
                    "home_form": h_form2, "away_form": a_form2,
                    "home_consistency": _sf2(hs_.get("last5_std",20),20),
                    "away_consistency": _sf2(as__.get("last5_std",20),20),
                    **{k:0.0 for k in ["travel_diff","travel_home_km","travel_away_km",
                       "travel_fatigue_diff","home_travel_fatigue","away_travel_fatigue",
                       "travel_win_rate_diff","travel_margin_diff","perth_win_rate_diff",
                       "is_perth_game","days_rest_diff","cl_diff","i50_diff","cp_diff",
                       "tk_diff","ho_diff","clanger_diff","pav_total_diff","pav_off_diff",
                       "pav_def_diff","pav_mid_diff"]},
                    "days_rest_home":7.0,"days_rest_away":7.0,
                    "streak_diff":_sf2(hs_.get("streak",0))-_sf2(as__.get("streak",0)),
                    "home_streak":_sf2(hs_.get("streak",0)),"away_streak":_sf2(as__.get("streak",0)),
                    "last_margin_diff":_sf2(hs_.get("last_margin",0))-_sf2(as__.get("last_margin",0)),
                    "last3_diff":_sf2(hs_.get("last3_avg",0))-_sf2(as__.get("last3_avg",0)),
                    "last5_diff":h_form2-a_form2,
                }
                pred2 = predict_game(win_model, margin_model, our_feats2)
                our_h = pred2["home_win_prob"] / 100.0
                our_a = 1.0 - our_h

                h_edge = our_h - h_fair
                a_edge = our_a - a_fair
                h_kelly = max(h_edge / (h_odds - 1), 0) * 100
                a_kelly = max(a_edge / (a_odds - 1), 0) * 100

                # Flag best value bet for this game
                best_edge = h_edge if h_edge > a_edge else a_edge
                best_team = ht if h_edge > a_edge else at
                best_odds_val = h_odds if h_edge > a_edge else a_odds
                best_kelly = h_kelly if h_edge > a_edge else a_kelly
                best_our_prob = our_h if h_edge > a_edge else our_a
                best_fair = h_fair if h_edge > a_edge else a_fair

                value_rows.append({
                    "Match": f"{ht} vs {at}",
                    "Value Bet": best_team,
                    "Odds": f"${best_odds_val:.2f}",
                    "Our Prob": f"{best_our_prob*100:.1f}%",
                    "Fair Prob": f"{best_fair*100:.1f}%",
                    "Edge": f"{best_edge*100:+.1f}%",
                    "Kelly %": f"{best_kelly:.1f}%",
                    "Vig": f"{vig:.1f}%",
                    "Bookmakers": row["bookmakers"],
                    "_edge": best_edge,
                })

            if value_rows:
                vdf = pd.DataFrame(value_rows).sort_values("_edge", ascending=False)

                # Highlight value bets (edge > 3%)
                value_bets = vdf[vdf["_edge"] > 0.03]
                no_value = vdf[vdf["_edge"] <= 0.03]

                if not value_bets.empty:
                    st.markdown("### 🎯 Value Opportunities (Edge > 3%)")
                    st.dataframe(
                        value_bets[["Match","Value Bet","Odds","Our Prob","Fair Prob","Edge","Kelly %","Bookmakers"]],
                        use_container_width=True, hide_index=True
                    )

                st.markdown("### All Games")
                st.dataframe(
                    vdf[["Match","Value Bet","Odds","Our Prob","Fair Prob","Edge","Kelly %","Vig","Bookmakers"]],
                    use_container_width=True, hide_index=True
                )

                st.markdown("---")
                st.markdown("""
**How to read this:**
- **Edge** = our model probability minus bookmaker's fair (vig-removed) probability. Positive = value.
- **Kelly %** = suggested bet size as % of bankroll using Kelly Criterion. Use ¼ Kelly (divide by 4) for safer sizing.
- **Vig** = bookmaker's margin built into the odds. Lower is better.
- Only bet where edge is consistently positive over many bets — one game means nothing.
""")
