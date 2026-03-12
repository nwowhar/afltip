import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import (get_all_games, get_upcoming_games, enrich_games, get_team_current_stats,
                          get_standings_multi_year)
from data.afltables import get_all_team_season_stats
from data.lineup import get_pav_multi_year, get_current_lineups, compute_lineup_strength
try:
    from data.experience import compute_experience_from_pav, analyse_data_staleness
    _experience_available = True
except ImportError:
    _experience_available = False
    def compute_experience_from_pav(*a, **kw): return __import__('pandas').DataFrame()
    def analyse_data_staleness(*a, **kw): return {}
from model.elo import build_elo_ratings, regress_elos_to_mean
try:
    from model.predictor import (build_features, add_season_stat_features,
                                  add_pav_features, add_experience_features,
                                  add_standings_features,
                                  train_models, predict_game,
                                  build_prediction_features, CORE_FEATURES, ALL_FEATURES)
except ImportError:
    from model.predictor import (build_features, add_season_stat_features,
                                  add_pav_features, train_models, predict_game,
                                  build_prediction_features, CORE_FEATURES, ALL_FEATURES)
    def add_experience_features(df, *a, **kw): return df
    def add_standings_features(df, *a, **kw): return df

# Detect whether the deployed predictor accepts style_df kwarg
import inspect as _inspect
_bpf_params = set(_inspect.signature(build_prediction_features).parameters)
_STYLE_DF_SUPPORTED = "style_df" in _bpf_params

def _build_prediction_features(*args, style_df=None, **kwargs):
    """Wrapper: passes style_df only if the deployed predictor supports it."""
    if _STYLE_DF_SUPPORTED and style_df is not None:
        return build_prediction_features(*args, style_df=style_df, **kwargs)
    return build_prediction_features(*args, **kwargs)
from model.backtest import (run_walk_forward_backtest, compute_yearly_accuracy,
                             ablation_test, permutation_importance_analysis,
                             margin_prediction_backtest, optimise_start_year, FEATURE_GROUPS)
try:
    from data.team_style import (build_style_features_from_season_stats,
                                  compute_style_matchup, STYLE_FEATURES)
    _style_available = True
except ImportError:
    _style_available = False
    def build_style_features_from_season_stats(*a, **kw): return __import__('pandas').DataFrame()
    def compute_style_matchup(*a, **kw): return {}
    STYLE_FEATURES = []

# ── Inline helpers (independent of fetcher.py version) ───────────────────────
import requests as _requests_module

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
_HEADERS = {"User-Agent": "AFL-Predictor/1.0 (nick@example.com)"}

TEAM_NAME_MAP = {
    "Brisbane Lions": "Brisbane Lions", "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney", "Greater Western Sydney Giants": "Greater Western Sydney",
    "GWS": "Greater Western Sydney", "Gold Coast Suns": "Gold Coast",
    "West Coast Eagles": "West Coast", "St Kilda Saints": "St Kilda",
    "North Melbourne Kangaroos": "North Melbourne", "Adelaide Crows": "Adelaide",
    "Geelong Cats": "Geelong", "Sydney Swans": "Sydney",
    "Collingwood Magpies": "Collingwood", "Melbourne Demons": "Melbourne",
    "Hawthorn Hawks": "Hawthorn", "Richmond Tigers": "Richmond",
    "Carlton Blues": "Carlton", "Essendon Bombers": "Essendon",
    "Fremantle Dockers": "Fremantle", "Port Adelaide Power": "Port Adelaide",
    "Port Adelaide": "Port Adelaide", "Western Bulldogs": "Western Bulldogs",
    "Gold Coast": "Gold Coast", "West Coast": "West Coast",
}

def normalise_team(name):
    return TEAM_NAME_MAP.get(str(name), str(name))

def get_squiggle_consensus(year=None, round_num=None):
    if year is None:
        year = datetime.now().year
    url = f"{SQUIGGLE_BASE}?q=tips;year={year}"
    if round_num:
        url += f";round={round_num}"
    try:
        r = _requests_module.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        tips = pd.DataFrame(r.json().get("tips", []))
        if tips.empty or "hconfidence" not in tips.columns:
            return pd.DataFrame()
        tips["hconfidence"] = pd.to_numeric(tips["hconfidence"], errors="coerce")
        if tips["hconfidence"].max() > 1.5:
            tips["hconfidence"] /= 100.0
        tips = tips.dropna(subset=["hconfidence", "gameid"])
        return (tips.groupby("gameid")
                .agg(hteam=("hteam","first"), ateam=("ateam","first"),
                     consensus_home_prob=("hconfidence","mean"),
                     n_models=("hconfidence","count"))
                .reset_index())
    except Exception:
        return pd.DataFrame()

def get_odds_api(api_key):
    try:
        r = _requests_module.get(
            "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
            params={"apiKey": api_key, "regions": "au", "markets": "h2h", "oddsFormat": "decimal"},
            timeout=15)
        r.raise_for_status()
        rows = []
        for game in r.json():
            ht = game.get("home_team","")
            at = game.get("away_team","")
            for bm in game.get("bookmakers",[]):
                for market in bm.get("markets",[]):
                    if market.get("key") != "h2h": continue
                    om = {o["name"]: o["price"] for o in market.get("outcomes",[])}
                    if om.get(ht) and om.get(at):
                        rows.append({"home_team":ht,"away_team":at,
                                     "bookmaker":bm.get("title",""),
                                     "home_odds":float(om[ht]),"away_odds":float(om[at])})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

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
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }
[data-testid="stSidebar"] .stSlider label { color: white !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] small { color: #ccc !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "start_year":      2016,
    "page":            "📊 Dashboard",
    "selected_round":  None,
    "predict_home":    None,
    "predict_away":    None,
    "predict_venue":   "(Auto)",
    "form_team":       None,
    "form_n":          15,
    "stats_year":      None,
    "stats_stat":      None,
    "radar_h":         None,
    "radar_a":         None,
    "backtest_min":    3,
    "inout_team":      "All teams",
    "pav_team":        "All",
    "bankroll":        1000,
    "kelly_fraction":  "Quarter Kelly (recommended)",
    "min_edge":        3,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏉 AFL Predictor")
    st.markdown("---")
    start_year = st.slider("Training data from", 2012, 2020,
                           key="start_year",
                           help="Model trains on all completed games from this year to present. Sweet spot is around 2016 — enough data without stale player pools from retired eras.")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict a Game",
        "📈 Team Form",
        "🏆 Elo Ladder",
        "📋 Team Stats",
        "🔬 Feature Importance",
        "📉 Backtest",
        "🎨 Style Matchup",
        "💰 Value Bets",
        "📖 How It Works",
    ], key="page")
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

@st.cache_data(ttl=3600, show_spinner="🏆 Fetching AFL ladder standings...")
def load_standings(start_year):
    return get_standings_multi_year(start_year)

@st.cache_data(ttl=86400, show_spinner="🎨 Building team style profiles...")
def load_style_features(start_year):
    """Derive style profiles from already-loaded season stats — no extra scraping."""
    ss = load_season_stats(start_year)
    if ss is None or ss.empty:
        return __import__('pandas').DataFrame()
    return build_style_features_from_season_stats(ss)

@st.cache_data(ttl=3601, show_spinner="🤖 Building Elo ratings & training model...")
def build_model(start_year):
    games_df = load_games(start_year)
    if games_df is None or games_df.empty:
        return None, None, None, None, {}, {}, None, None, None, None, None, None

    season_stats  = load_season_stats(start_year)
    pav_df        = load_pav(start_year)
    standings_df  = load_standings(start_year)
    style_df      = load_style_features(start_year)

    df = enrich_games(games_df)
    df, elo_history = build_elo_ratings(df)
    df = build_features(df)
    df = add_season_stat_features(df, season_stats)
    df = add_pav_features(df, pav_df)
    exp_df = compute_experience_from_pav(pav_df, games_df, year=datetime.now().year)
    df = add_experience_features(df, exp_df)
    df = add_standings_features(df, standings_df)
    # Add style features — leakage-safe (uses year-1 season stats)
    try:
        from model.predictor import add_style_features
        df = add_style_features(df, style_df)
    except Exception:
        pass

    win_model, margin_model, metrics, fi_df = train_models(df)
    current_elos = regress_elos_to_mean(elo_history)
    team_stats   = get_team_current_stats(df)

    return df, win_model, margin_model, metrics, current_elos, team_stats, season_stats, pav_df, fi_df, exp_df, standings_df, style_df

with st.spinner("Loading model..."):
    result = build_model(start_year)

if result[0] is None:
    st.error("Could not load game data. Check your internet connection.")
    st.stop()

df, win_model, margin_model, metrics, current_elos, team_stats, season_stats, pav_df, fi_df, exp_df, standings_df, style_df = result
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

    # ── Metric explanations ───────────────────────────────────────────────────
    with st.expander("📊 What do these numbers mean?"):
        acc = metrics['win_accuracy'] * 100
        base = metrics['base_accuracy'] * 100
        r2 = metrics['margin_r2']
        gain_pct = gain * 100

        # Colour-code accuracy contextually
        if acc >= 67:
            acc_colour = "#2ecc71"; acc_verdict = "excellent"
        elif acc >= 65:
            acc_colour = "#f39c12"; acc_verdict = "solid"
        else:
            acc_colour = "#e94560"; acc_verdict = "below target"

        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:8px 0">

  <div style="background:#0a1628;border-radius:10px;padding:16px;border-left:4px solid {acc_colour}">
    <div style="font-size:1.6rem;font-weight:700;color:{acc_colour}">{acc:.1f}%</div>
    <div style="font-size:0.9rem;font-weight:600;color:white;margin:4px 0">Win Prediction Accuracy</div>
    <div style="font-size:0.8rem;color:#aaa;line-height:1.5">
      Measured via <b>5-fold cross-validation</b> — the dataset is split into 5 chunks,
      the model trains on 4 and predicts the 5th, rotating through all combinations.
      This gives a realistic out-of-sample accuracy estimate.<br><br>
      <b>Context:</b> Random guessing = 50%. Tipping the favourite every game ≈ 60–62%.
      Professional tipsters average 65–68%. Our model at <b style="color:{acc_colour}">{acc:.1f}%</b> is {acc_verdict}.<br><br>
      {'⚠️ <b>Why did it drop from ~66%?</b> Adding new features (experience, PAV) increased model complexity. The features are genuinely informative but also add noise when their data is sparse — particularly early-season PAV and experience data that defaults to zero. Accuracy should recover mid-season once PAV and season stats are fully populated.' if acc < 66 else '✅ Model is performing at or above the target benchmark of 66%.'}
    </div>
  </div>

  <div style="background:#0a1628;border-radius:10px;padding:16px;border-left:4px solid #3498db">
    <div style="font-size:1.6rem;font-weight:700;color:#3498db">{base:.1f}%</div>
    <div style="font-size:0.9rem;font-weight:600;color:white;margin:4px 0">Elo-Only Baseline</div>
    <div style="font-size:0.8rem;color:#aaa;line-height:1.5">
      What accuracy you'd get using <b>nothing but Elo ratings</b> — no form, no travel,
      no stats. Just: higher Elo team wins.<br><br>
      Elo alone captures most of the signal because team quality is genuinely the biggest
      predictor of results. Everything else the model adds — form, fatigue, experience —
      explains the remaining variance on top of this base.<br><br>
      The full model adds <b style="color:#2ecc71">{gain_pct:+.1f}%</b> on top of Elo alone.
      That might sound small, but over a full season of 207 games it means ~{int(207 * gain):.0f}
      extra correct tips.
    </div>
  </div>

  <div style="background:#0a1628;border-radius:10px;padding:16px;border-left:4px solid #9b59b6">
    <div style="font-size:1.6rem;font-weight:700;color:#9b59b6">{r2:.3f}</div>
    <div style="font-size:0.9rem;font-weight:600;color:white;margin:4px 0">Margin R²</div>
    <div style="font-size:0.8rem;color:#aaa;line-height:1.5">
      R² (R-squared) measures how well the model predicts the <b>winning margin</b>,
      not just who wins. It ranges from 0 to 1:<br><br>
      • <b>0.0</b> = no better than guessing the average margin every game<br>
      • <b>1.0</b> = perfect margin prediction (impossible in practice)<br>
      • <b style="color:#9b59b6">{r2:.3f}</b> = the model explains {r2*100:.1f}% of the variation in margins<br><br>
      AFL margins are notoriously hard to predict — a 10-point game can easily become
      40 points in the last quarter. R² of 0.20–0.25 is typical for AFL margin models.
      {'✅ On target.' if r2 >= 0.20 else '⚠️ Below target — margin predictions are less reliable this season.'}
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

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
            # If session state has a stale/None round, reset to default
            if st.session_state.get("selected_round") not in available_rounds:
                st.session_state["selected_round"] = default_round
            col_title, col_picker = st.columns([2, 1])
            with col_title:
                st.markdown("## UPCOMING GAMES")
            with col_picker:
                selected_round = st.selectbox("Round", available_rounds,
                                              index=available_rounds.index(st.session_state["selected_round"]),
                                              key="selected_round",
                                              label_visibility="collapsed")
            upcoming = _incomplete[_incomplete["round"] == selected_round].copy()
            _ladder_w = min(selected_round / 8.0, 1.0)
            _ladder_note = f"Ladder weight: {_ladder_w:.0%}" if _ladder_w < 1.0 else "Ladder: full weight"
            st.markdown(f"*Round {selected_round} — {len(upcoming)} games · {_ladder_note}*")
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
                    # Use build_prediction_features for consistency with Predict page
                    feats = _build_prediction_features(
                        home, away, venue,
                        current_elos, team_stats,
                        season_stats, lineup_strength,
                        df, exp_df, standings_df,
                        style_df=style_df,
                        current_round=int(selected_round) if selected_round else None
                    )
                except Exception as game_err:
                    import traceback
                    st.error(f"Error for {home} vs {away}: {game_err}")
                    st.code(traceback.format_exc())
                    continue

                pred = predict_game(win_model, margin_model, feats, metrics["features_used"])
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

                # ── Derive display values from feats dict ──────────────────
                _sf2 = lambda v, d=0.0: float(v) if v is not None else float(d)
                _fv  = lambda k: float(feats.get(k, 0.0))
                hs_  = team_stats.get(home, {})
                as__ = team_stats.get(away, {})
                h_elo_display = _sf2(current_elos.get(home, 1500), 1500)
                a_elo_display = _sf2(current_elos.get(away, 1500), 1500)
                h_form_d = _fv("home_form")
                a_form_d = _fv("away_form")
                h_km_d   = _fv("travel_home_km")
                a_km_d   = _fv("travel_away_km")
                h_rest_d = _fv("days_rest_home")
                a_rest_d = _fv("days_rest_away")
                h_fat_d  = _fv("home_travel_fatigue")
                a_fat_d  = _fv("away_travel_fatigue")
                cur_yr_d = datetime.now().year
                def _ss_d(team, stat):
                    if season_stats is None or season_stats.empty: return 0.0
                    row_ = season_stats[(season_stats["team"]==team) & (season_stats["year"]==cur_yr_d)]
                    if row_.empty:
                        row_ = season_stats[(season_stats["team"]==team) & (season_stats["year"]==cur_yr_d-1)]
                    return float(row_.iloc[0].get(stat, 0)) if not row_.empty else 0.0

                # ── Factor analysis for insight panel ─────────────────────
                factors = [
                    # (label, home_val, away_val, home_is_better_when_higher)
                    ("Elo Rating",          h_elo_display,  a_elo_display,  True),
                    ("Form (last 5 avg)",   h_form_d,       a_form_d,       True),
                    ("Current Streak",      _sf2(hs_.get("streak",0)),   _sf2(as__.get("streak",0)),  True),
                    ("Last Game Margin",    _sf2(hs_.get("last_margin",0)), _sf2(as__.get("last_margin",0)), True),
                    ("Travel to Venue",     h_km_d,         a_km_d,         False),
                    ("Days Rest",           h_rest_d,       a_rest_d,       True),
                    ("Travel Fatigue",      h_fat_d,        a_fat_d,        False),
                    ("Clearances (season)", _ss_d(home,"avg_clearances"),   _ss_d(away,"avg_clearances"),   True),
                    ("Inside 50s (season)", _ss_d(home,"avg_inside_50s"),   _ss_d(away,"avg_inside_50s"),   True),
                    ("Contested Poss",      _ss_d(home,"avg_contested_possessions"), _ss_d(away,"avg_contested_possessions"), True),
                    ("Tackles (season)",    _ss_d(home,"avg_tackles"),      _ss_d(away,"avg_tackles"),      True),
                    ("Clangers (season)",   _ss_d(home,"avg_clangers"),     _ss_d(away,"avg_clangers"),     False),
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
                        ("Elo Rating",          h_elo_display, a_elo_display, True,
                         "pts", "Elo is a chess-style rating updated after every game. Each win/loss shifts ratings based on the expected result. Home team gets +50pts advantage baked in. A 100pt gap = roughly 64% win probability."),
                        ("Form (last 5 avg)",    h_form_d, a_form_d, True,
                         "pts margin", "Average winning/losing margin across the last 5 games. Positive = winning by that many points on average. More responsive to recent form than Elo."),
                        ("Current Streak",       _sf2(hs_.get("streak",0)), _sf2(as__.get("streak",0)), True,
                         "games", "Signed win/loss streak. +3 means 3 wins in a row, -2 means 2 losses in a row."),
                        ("Last Game Margin",     _sf2(hs_.get("last_margin",0)), _sf2(as__.get("last_margin",0)), True,
                         "pts", "Margin from their most recent completed game. Positive = won by that many points."),
                        ("Travel to Venue",      h_km_d, a_km_d, False,
                         "km", "Straight-line distance each team travels to reach the venue. Lower is better — home games = ~0km, interstate = 500–900km, Perth = 2,700km from east coast."),
                        ("Days Rest",            h_rest_d, a_rest_d, True,
                         "days", "Days since their last game. More rest = fresher legs. Capped at 21 days — anything longer (summer break, bye) resets to a neutral 7 days so it doesn't skew R1 predictions."),
                        ("Travel Fatigue",       h_fat_d, a_fat_d, False,
                         "index", "Combined travel+rest stress index: (km travelled ÷ 1000) × max(14 − rest days, 0). A team flying 2,700km to Perth with only 6 days rest scores ~8.1 vs 0 for the home side."),
                        ("Clearances (season)",  _ss_d(home,"avg_clearances"),  _ss_d(away,"avg_clearances"),  True,
                         "per game", "Season average clearances per game from AFL Tables. Clearances out of stoppages drive transition and scoring chains — one of the strongest team performance indicators."),
                        ("Inside 50s (season)",  _ss_d(home,"avg_inside_50s"),  _ss_d(away,"avg_inside_50s"),  True,
                         "per game", "Season average entries inside the forward 50 per game. More entries = more scoring chances. Strongly correlated with winning margin."),
                        ("Contested Poss",       _ss_d(home,"avg_contested_possessions"), _ss_d(away,"avg_contested_possessions"), True,
                         "per game", "Season average contested possessions per game. Reflects contested ball dominance — teams that win this category tend to control the game's tempo."),
                        ("Tackles (season)",     _ss_d(home,"avg_tackles"),     _ss_d(away,"avg_tackles"),     True,
                         "per game", "Season average tackles per game. High tackle counts indicate pressure and defensive intensity."),
                        ("Clangers (season)",    _ss_d(home,"avg_clangers"),    _ss_d(away,"avg_clangers"),    False,
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
    with c1: home_team = st.selectbox("🏠 Home Team", teams, key="predict_home")
    with c2:
        away_opts = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_opts, key="predict_away")
    with c3:
        venues = sorted(set(df["venue"].dropna().unique()))
        venue  = st.selectbox("📍 Venue", ["(Auto)"] + venues, key="predict_venue")
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
        _pred_round = int(df[df["year"] == datetime.now().year]["round"].max()) if not df[df["year"] == datetime.now().year].empty else None
        feats = _build_prediction_features(
            home_team, away_team, venue,
            current_elos, team_stats,
            season_stats, lineup_strength,
            df, exp_df, standings_df,
            style_df=style_df,
            current_round=_pred_round
        )
        pred = predict_game(win_model, margin_model, feats, metrics["features_used"])
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
    selected = st.selectbox("Select Team", teams, key="form_team")
    n        = st.slider("Last N games", 5, 30, 15, key="form_n")
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

    with st.expander("🔧 Debug: season_stats columns"):
        if season_stats is not None and not season_stats.empty:
            st.write(season_stats.columns.tolist())
            st.write(season_stats.head(3))
        else:
            st.write("season_stats is empty")

    cur_year = datetime.now().year

    if season_stats is None or season_stats.empty:
        st.warning("Season stats not available — AFL Tables data will load on the first full run.")
    else:
        # Get latest available year
        avail_years = sorted(season_stats["year"].unique(), reverse=True)
        sel_year = st.selectbox("Season", avail_years, index=0, key="stats_year")
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
                selected_stat_label = st.selectbox("Rank by stat", stat_tab_labels, key="stats_stat")
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

    min_train = st.slider("Minimum training years before testing", 2, 5, 3, key="backtest_min")

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
        st.markdown("*Which feature groups actually earn their place? Accuracy when each group is removed — negative delta = group helps.*")

        # Early-season warning
        current_round = df[df["year"] == datetime.now().year]["round"].max() if not df[df["year"] == datetime.now().year].empty else 0
        if isinstance(current_round, float) and np.isnan(current_round):
            current_round = 0
        if int(current_round) < 5:
            st.warning(f"""
⚠️ **Early-season caveat (Round {int(current_round)}):** Season stats and PAV features are nearly empty right now
— they default to 0.0 for all teams until real data accumulates. This means they'll show as **Neutral** in the
ablation even though they're genuinely useful mid-season. Re-run after Round 5+ for a meaningful result.

Rest days showing ❌ Hurts can also be a pre-season artefact — binary short/bye rest flags have been added
to replace raw rest day counts, which should reduce this noise.
""")

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
                dark_chart(fig3, 420)
                fig3.update_layout(
                    title="Accuracy change when feature group is REMOVED (negative = group helps)",
                    xaxis=dict(title="Δ Accuracy %"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.dataframe(ablation_df[["group", "accuracy", "delta",
                                          "interpretation", "n_features"]],
                             use_container_width=True, hide_index=True)

                # Contextual interpretation
                hurting = ablation_df[ablation_df["interpretation"].str.contains("Hurts", na=False)]
                helping = ablation_df[ablation_df["interpretation"].str.contains("Helps", na=False)]
                neutral_sparse = ablation_df[
                    ablation_df["group"].isin(["Season stats", "PAV lineup", "Experience"]) &
                    ablation_df["interpretation"].str.contains("Neutral", na=False)
                ]
                if not hurting.empty or not neutral_sparse.empty:
                    notes = []
                    for _, row in hurting.iterrows():
                        notes.append(f"• **{row['group']}** is hurting accuracy ({row['delta']:+.2f}%). Consider dropping it or re-engineering the features.")
                    for _, row in neutral_sparse.iterrows():
                        notes.append(f"• **{row['group']}** shows Neutral — likely because data is sparse early-season. Re-test after Round 5.")
                    st.info("**Ablation notes:**\n" + "\n".join(notes))

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

        # ── Start year optimisation ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📅 Optimal Training Start Year")
        st.markdown(
            "*How far back should we train? Each candidate start year is evaluated by its "
            "out-of-sample accuracy on the most recent 3 completed seasons — the sweet spot "
            "balances data volume against the risk of training on a different era of AFL.*"
        )

        with st.spinner("Running start year optimisation (this may take ~30s)..."):
            all_years_avail = sorted(df["year"].unique())
            # Only test start years that give at least 2 pre-holdout seasons
            holdout_n = min(3, len(all_years_avail) - 3)
            if holdout_n < 1:
                st.info("Not enough historical seasons to optimise start year yet.")
            else:
                sy_df = optimise_start_year(
                    df, avail_feats,
                    holdout_years=holdout_n,
                    min_train_years=2
                )

                if sy_df.empty:
                    st.info("Not enough data to run start year optimisation.")
                else:
                    best_row   = sy_df.loc[sy_df["accuracy"].idxmax()]
                    best_year  = int(best_row["start_year"])
                    best_acc   = float(best_row["accuracy"])
                    cur_acc    = sy_df[sy_df["start_year"] == start_year]["accuracy"]
                    cur_acc_val = float(cur_acc.iloc[0]) if not cur_acc.empty else None

                    # Colour bars: highlight best, dim others
                    colours = [
                        "#f39c12" if int(r["start_year"]) == best_year
                        else ("#e94560" if int(r["start_year"]) == start_year else "#3498db")
                        for _, r in sy_df.iterrows()
                    ]

                    fig5 = go.Figure()
                    fig5.add_trace(go.Bar(
                        x=sy_df["start_year"],
                        y=sy_df["accuracy"],
                        marker_color=colours,
                        text=sy_df["accuracy"].apply(lambda v: f"{v:.1f}%"),
                        textposition="outside",
                        customdata=sy_df[["n_train_games", "n_test_games"]].values,
                        hovertemplate=(
                            "<b>Start: %{x}</b><br>"
                            "Accuracy: %{y:.1f}%<br>"
                            "Train games: %{customdata[0]}<br>"
                            "Test games: %{customdata[1]}<extra></extra>"
                        )
                    ))
                    dark_chart(fig5, 380)
                    fig5.update_layout(
                        yaxis=dict(
                            title="Out-of-sample accuracy (%)",
                            range=[
                                max(55, sy_df["accuracy"].min() - 2),
                                min(80, sy_df["accuracy"].max() + 3)
                            ]
                        ),
                        xaxis=dict(title="Training data start year", dtick=1),
                    )
                    # Annotation for best year
                    fig5.add_annotation(
                        x=best_year, y=best_acc + 1.2,
                        text="🏆 Best",
                        showarrow=False,
                        font=dict(color="#f39c12", size=12)
                    )
                    if start_year != best_year:
                        cur_y = float(sy_df[sy_df["start_year"] == start_year]["accuracy"].iloc[0]) if not sy_df[sy_df["start_year"] == start_year].empty else None
                        if cur_y:
                            fig5.add_annotation(
                                x=start_year, y=cur_y + 1.2,
                                text="📍 Current",
                                showarrow=False,
                                font=dict(color="#e94560", size=12)
                            )
                    st.plotly_chart(fig5, use_container_width=True)

                    # Also show n_train_games as a secondary line
                    fig6 = go.Figure()
                    fig6.add_trace(go.Scatter(
                        x=sy_df["start_year"],
                        y=sy_df["n_train_games"],
                        mode="lines+markers",
                        line=dict(color="#2ecc71", width=2),
                        marker=dict(size=6),
                        name="Training games",
                        hovertemplate="Start: %{x}<br>Train games: %{y}<extra></extra>"
                    ))
                    dark_chart(fig6, 200)
                    fig6.update_layout(
                        yaxis=dict(title="Training games available"),
                        xaxis=dict(title="Training data start year", dtick=1),
                        showlegend=False,
                    )
                    st.plotly_chart(fig6, use_container_width=True)

                    # Summary callout
                    if cur_acc_val is not None and start_year != best_year:
                        gap = best_acc - cur_acc_val
                        extra_tips = int(round(gap / 100 * 207))
                        st.warning(
                            f"🏆 **Optimal start year: {best_year}** ({best_acc:.1f}% accuracy on holdout) — "
                            f"your current setting ({start_year}) scores {cur_acc_val:.1f}%, "
                            f"a gap of **{gap:+.1f}%** (~{extra_tips} extra correct tips over a season). "
                            f"Try moving the training slider to {best_year}."
                        )
                    elif start_year == best_year:
                        st.success(
                            f"✅ **{start_year} is already the optimal start year** ({best_acc:.1f}% on holdout). "
                            f"You're training on {int(best_row['n_train_games'])} games."
                        )
                    else:
                        st.info(f"Optimal start year based on holdout accuracy: **{best_year}** ({best_acc:.1f}%)")

                    with st.expander("📋 Full results table"):
                        st.dataframe(
                            sy_df.rename(columns={
                                "start_year": "Start Year",
                                "n_train_games": "Train Games",
                                "accuracy": "Accuracy (%)",
                                "brier_score": "Brier Score",
                                "n_test_games": "Test Games",
                            }).drop(columns=["holdout_seasons"], errors="ignore"),
                            use_container_width=True, hide_index=True
                        )

# ═══════════════════════════════════════════════════════════════════════════════
# LINEUP STRENGTH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Style Matchup":
    st.markdown("# 🎨 STYLE MATCHUP")
    st.markdown("*How do teams' playing styles match up? Derived from AFL Tables season stats.*")

    if style_df is None or style_df.empty:
        st.warning("Style data not available — season stats need to load first.")
    else:
        cur_year = datetime.now().year
        avail_years = sorted(style_df["year"].unique(), reverse=True)
        sel_year_s  = avail_years[0] if avail_years else cur_year

        sm_c1, sm_c2 = st.columns(2)
        with sm_c1:
            sm_home = st.selectbox("🏠 Home Team", teams, key="sm_home")
        with sm_c2:
            sm_away_opts = [t for t in teams if t != sm_home]
            sm_away = st.selectbox("✈️ Away Team", sm_away_opts, key="sm_away")

        # ── Style profile lookup ───────────────────────────────────────────────
        def get_style(team, year):
            row = style_df[(style_df["team"] == team) & (style_df["year"] == year)]
            if row.empty:
                row = style_df[(style_df["team"] == team) & (style_df["year"] == year - 1)]
            return row.iloc[0] if not row.empty else None

        h_style = get_style(sm_home, sel_year_s)
        a_style = get_style(sm_away, sel_year_s)

        if h_style is None or a_style is None:
            st.info(f"Style data not yet available for {sel_year_s} — showing previous season.")
            h_style = get_style(sm_home, sel_year_s - 1)
            a_style = get_style(sm_away, sel_year_s - 1)

        if h_style is not None and a_style is not None:
            # ── KPI cards ─────────────────────────────────────────────────────
            STYLE_DISPLAY = [
                ("kick_ratio",  "Kick Ratio",   "Kicks / (Kicks+HBs). Higher = kick-heavy style.",     False),
                ("tackle_rate", "Tackle Rate",  "Tackles per game. Higher = more pressure.",            False),
                ("hitout_rate", "Hitout Rate",  "Hitouts per game. Higher = ruck dominance.",           False),
                ("mark_rate",   "Mark Rate",    "Marks per game. Higher = aerial/corridor game.",       False),
            ]
            kpi_cols = st.columns(4)
            for i, (col, label, desc, lower_better) in enumerate(STYLE_DISPLAY):
                hv = float(h_style.get(col, 0))
                av = float(a_style.get(col, 0))
                diff = hv - av if not lower_better else av - hv
                edge_team = sm_home if diff > 0.01 else (sm_away if diff < -0.01 else None)
                edge_col  = "#2ecc71" if edge_team == sm_home else ("#3498db" if edge_team == sm_away else "#aaa")
                with kpi_cols[i]:
                    st.markdown(
                        f'<div style="background:#0a1628;border-left:4px solid {edge_col};'
                        f'border-radius:8px;padding:12px;margin-bottom:8px">'
                        f'<div style="font-size:0.75rem;color:#aaa;margin-bottom:4px">{label}</div>'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="color:#e94560;font-weight:700">{hv:.3f}</span>'
                        f'<span style="color:#666;font-size:0.7rem">vs</span>'
                        f'<span style="color:#3498db;font-weight:700">{av:.3f}</span>'
                        f'</div>'
                        f'<div style="font-size:0.68rem;color:#555;margin-top:4px">{desc}</div>'
                        + (f'<div style="font-size:0.7rem;color:{edge_col};margin-top:4px">Edge: {edge_team}</div>' if edge_team else '')
                        + '</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")

            # ── Radar chart ───────────────────────────────────────────────────
            st.markdown("### Style Profile Radar")
            st.markdown("*Normalised across all teams — higher = stronger in that dimension*")

            radar_cats = ["Kick Ratio", "Tackle Rate", "Hitout Rate", "Mark Rate"]
            radar_cols_map = ["kick_ratio", "tackle_rate", "hitout_rate", "mark_rate"]

            # Normalise across all teams for the selected year
            h_vals_r, a_vals_r = [], []
            for rc in radar_cols_map:
                col_data = pd.to_numeric(
                    style_df[style_df["year"] == sel_year_s][rc], errors="coerce"
                ).dropna()
                if col_data.empty:
                    h_vals_r.append(0.5)
                    a_vals_r.append(0.5)
                    continue
                mn, mx = col_data.min(), col_data.max()
                rng = mx - mn if mx != mn else 1
                h_vals_r.append((float(h_style.get(rc, mn)) - mn) / rng)
                a_vals_r.append((float(a_style.get(rc, mn)) - mn) / rng)

            cats_c = radar_cats + [radar_cats[0]]
            h_c    = h_vals_r + [h_vals_r[0]]
            a_c    = a_vals_r + [a_vals_r[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=h_c, theta=cats_c, fill="toself", name=sm_home,
                line=dict(color="#e94560"), fillcolor="rgba(233,69,96,0.2)"
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=a_c, theta=cats_c, fill="toself", name=sm_away,
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
                height=400, margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # ── Matchup callouts ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Matchup Callouts")

            matchup = compute_style_matchup(sm_home, sm_away, style_df, sel_year_s)

            callouts = [
                ("kick_ratio_diff",
                 f"**Kick-style edge:** {sm_home if matchup['kick_ratio_diff'] > 0.01 else (sm_away if matchup['kick_ratio_diff'] < -0.01 else 'Even')}",
                 f"{sm_home} kick ratio {float(h_style['kick_ratio']):.3f} vs {sm_away} {float(a_style['kick_ratio']):.3f}. "
                 f"{'A kick-heavy team playing a handball-heavy team can dictate tempo.' if abs(matchup['kick_ratio_diff']) > 0.02 else 'Both teams use a similar disposal mix.'}"),
                ("tackle_diff",
                 f"**Pressure game edge:** {sm_home if matchup['tackle_diff'] > 1 else (sm_away if matchup['tackle_diff'] < -1 else 'Even')}",
                 f"{sm_home} tackles {float(h_style['tackle_rate']):.1f}/game vs {sm_away} {float(a_style['tackle_rate']):.1f}/game. "
                 f"{'High-pressure team may disrupt the lower-tackling opponent.' if abs(matchup['tackle_diff']) > 2 else 'Similar pressure intensity.'}"),
                ("hitout_diff",
                 f"**Ruck advantage:** {sm_home if matchup['hitout_diff'] > 2 else (sm_away if matchup['hitout_diff'] < -2 else 'Even')}",
                 f"{sm_home} hitouts {float(h_style['hitout_rate']):.1f}/game vs {sm_away} {float(a_style['hitout_rate']):.1f}/game. "
                 f"{'Significant ruck dominance — can win clearances from stoppages.' if abs(matchup['hitout_diff']) > 4 else 'Rucks are closely matched.'}"),
                ("mark_diff",
                 f"**Aerial edge:** {sm_home if matchup['mark_diff'] > 1 else (sm_away if matchup['mark_diff'] < -1 else 'Even')}",
                 f"{sm_home} marks {float(h_style['mark_rate']):.1f}/game vs {sm_away} {float(a_style['mark_rate']):.1f}/game. "
                 f"{'Aerial game advantage — may exploit a ground-ball focused opponent.' if abs(matchup['mark_diff']) > 2 else 'Similar marking rates.'}"),
            ]

            for feat_key, headline, detail in callouts:
                val = matchup.get(feat_key, 0)
                if abs(val) > 0.5:
                    colour = "#2ecc71" if val > 0 else "#3498db"
                else:
                    colour = "#666"
                st.markdown(
                    f'<div style="background:#0a1628;border-left:4px solid {colour};'
                    f'border-radius:8px;padding:12px;margin-bottom:10px">'
                    f'<div style="color:white;font-size:0.9rem">{headline}</div>'
                    f'<div style="color:#aaa;font-size:0.8rem;margin-top:4px">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── Model features ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Style Feature Values (fed to model)")
            feat_rows = [
                {"Feature": k, "Value": round(v, 4),
                 "Favours": sm_home if v > 0.001 else (sm_away if v < -0.001 else "—")}
                for k, v in matchup.items()
            ]
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        # ── All-teams style leaderboard ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### All-Teams Style Rankings")
        st.markdown(f"*{sel_year_s} season — {len(style_df[style_df['year'] == sel_year_s])} teams*")

        yr_style = style_df[style_df["year"] == sel_year_s].copy()
        if not yr_style.empty:
            yr_style = yr_style.sort_values("kick_ratio", ascending=False).reset_index(drop=True)
            yr_style["Kick Ratio"]  = yr_style["kick_ratio"].round(3)
            yr_style["Tackle Rate"] = yr_style["tackle_rate"].round(1)
            yr_style["Hitout Rate"] = yr_style["hitout_rate"].round(1)
            yr_style["Mark Rate"]   = yr_style["mark_rate"].round(1)
            st.dataframe(
                yr_style[["team", "Kick Ratio", "Tackle Rate", "Hitout Rate", "Mark Rate"]]
                .rename(columns={"team": "Team"}),
                use_container_width=True, hide_index=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# VALUE BETS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Value Bets":
    st.markdown("# 💰 VALUE BETS")
    st.markdown("*Find edges between our model's probabilities and bookmaker odds*")

    # ── Helper to get our model prob for any matchup ──────────────────────────
    def get_our_prob(ht, at, venue=""):
        """Use identical feature building to the Predict page."""
        _round = int(df[df["year"] == datetime.now().year]["round"].max()) if not df[df["year"] == datetime.now().year].empty else None
        feats = _build_prediction_features(
            ht, at, venue,
            current_elos, team_stats,
            season_stats, lineup_strength if 'lineup_strength' in dir() else {},
            df, exp_df, standings_df,
            style_df=style_df,
            current_round=_round
        )
        pred = predict_game(win_model, margin_model, feats, metrics["features_used"])
        return pred["home_win_prob"] / 100.0, pred["predicted_margin"]

    # ── Bankroll input ────────────────────────────────────────────────────────
    st.markdown("---")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bankroll = st.number_input("💵 Bankroll ($)", min_value=10, max_value=100000,
                                    value=st.session_state.bankroll, step=50, key="bankroll")
    with bc2:
        kelly_fraction = st.selectbox("Kelly sizing",
                                       ["Full Kelly", "Half Kelly (safer)", "Quarter Kelly (recommended)"],
                                       key="kelly_fraction")
    with bc3:
        min_edge = st.slider("Min edge to show (%)", 0, 15, key="min_edge")

    kelly_divisor = {"Full Kelly": 1, "Half Kelly (safer)": 2, "Quarter Kelly (recommended)": 4}[kelly_fraction]

    st.markdown("---")

    # ── Fetch odds ────────────────────────────────────────────────────────────
    odds_key = ""
    try:
        odds_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        pass

    if not odds_key:
        st.info("Add your `ODDS_API_KEY` to Streamlit secrets to enable live bookmaker odds.")
    else:
        with st.spinner("Fetching live odds from TAB, Sportsbet, Neds, Ladbrokes..."):
            odds_df = get_odds_api(odds_key)

        if odds_df.empty:
            st.info("No AFL odds available right now — check back closer to game day.")
        else:
            # Per-bookmaker odds so user can shop lines
            all_game_rows = []
            games_seen = {}

            for _, row in odds_df.iterrows():
                ht = normalise_team(row["home_team"])
                at = normalise_team(row["away_team"])
                bm = row["bookmaker"]
                h_odds = float(row["home_odds"])
                a_odds = float(row["away_odds"])
                game_key = f"{ht}|{at}"

                if ht not in current_elos or at not in current_elos:
                    continue

                if game_key not in games_seen:
                    our_h, pred_margin = get_our_prob(ht, at)
                    games_seen[game_key] = (our_h, pred_margin)
                else:
                    our_h, pred_margin = games_seen[game_key]

                our_a = 1.0 - our_h
                h_implied = 1.0 / h_odds
                a_implied = 1.0 / a_odds
                total_implied = h_implied + a_implied
                h_fair = h_implied / total_implied
                a_fair = a_implied / total_implied
                vig = (total_implied - 1.0) * 100

                h_edge = (our_h - h_fair) * 100
                a_edge = (our_a - a_fair) * 100
                h_kelly_pct = max(our_h - h_fair, 0) / (h_odds - 1) * 100 / kelly_divisor
                a_kelly_pct = max(our_a - a_fair, 0) / (a_odds - 1) * 100 / kelly_divisor
                h_stake = bankroll * h_kelly_pct / 100
                a_stake = bankroll * a_kelly_pct / 100
                h_return = h_stake * h_odds
                a_return = a_stake * a_odds

                all_game_rows.append({
                    "match_key": game_key,
                    "Home": ht, "Away": at,
                    "Bookmaker": bm,
                    "Home Odds": h_odds, "Away Odds": a_odds,
                    "Our H%": our_h, "Our A%": our_a,
                    "H Fair%": h_fair, "A Fair%": a_fair,
                    "H Edge%": h_edge, "A Edge%": a_edge,
                    "H Kelly%": h_kelly_pct, "A Kelly%": a_kelly_pct,
                    "H Stake": h_stake, "A Stake": a_stake,
                    "H Return": h_return, "A Return": a_return,
                    "Vig%": vig, "Pred Margin": pred_margin,
                })

            if not all_game_rows:
                st.info("No matching teams found in odds data.")
            else:
                full_df = pd.DataFrame(all_game_rows)

                # ── Best odds per game ────────────────────────────────────────
                best = (full_df.groupby("match_key")
                        .apply(lambda g: pd.Series({
                            "Home": g["Home"].iloc[0],
                            "Away": g["Away"].iloc[0],
                            "Best Home Odds": g["Home Odds"].max(),
                            "Best Home Bookie": g.loc[g["Home Odds"].idxmax(), "Bookmaker"],
                            "Best Away Odds": g["Away Odds"].max(),
                            "Best Away Bookie": g.loc[g["Away Odds"].idxmax(), "Bookmaker"],
                            "Our H%": g["Our H%"].iloc[0],
                            "H Fair%": g["H Fair%"].iloc[0],
                            "A Fair%": g["A Fair%"].iloc[0],
                            "H Edge%": g["H Edge%"].max(),
                            "A Edge%": g["A Edge%"].max(),
                            "Pred Margin": g["Pred Margin"].iloc[0],
                        }))
                        .reset_index(drop=True))

                # ── Value bet cards ───────────────────────────────────────────
                best["best_edge"] = best[["H Edge%","A Edge%"]].max(axis=1)
                value = best[best["best_edge"] >= min_edge].sort_values("best_edge", ascending=False)

                if value.empty:
                    st.info(f"No bets found with edge ≥ {min_edge}%. Try lowering the minimum edge slider.")
                else:
                    st.markdown(f"### 🎯 {len(value)} Value Bet{'s' if len(value)>1 else ''} Found (Edge ≥ {min_edge}%)")

                    for _, g in value.iterrows():
                        ht = g["Home"]
                        at = g["Away"]
                        h_edge = g["H Edge%"]
                        a_edge = g["A Edge%"]
                        is_home_value = h_edge >= a_edge
                        val_team  = ht if is_home_value else at
                        val_odds  = g["Best Home Odds"] if is_home_value else g["Best Away Odds"]
                        val_bookie= g["Best Home Bookie"] if is_home_value else g["Best Away Bookie"]
                        val_edge  = h_edge if is_home_value else a_edge
                        val_our   = g["Our H%"] if is_home_value else (1 - g["Our H%"])
                        val_fair  = g["H Fair%"] if is_home_value else g["A Fair%"]
                        kelly_pct = max(val_our - val_fair, 0) / (val_odds - 1) / kelly_divisor * 100
                        stake     = bankroll * kelly_pct / 100
                        exp_return= stake * val_odds
                        exp_profit= exp_return - stake

                        # Colour by edge strength
                        if val_edge >= 10:  card_colour = "#1a4a1a"; badge = "🔥 STRONG VALUE"
                        elif val_edge >= 5: card_colour = "#1a3a1a"; badge = "✅ GOOD VALUE"
                        else:               card_colour = "#1a2a2a"; badge = "👀 MARGINAL VALUE"

                        pred_margin_str = f"{abs(g['Pred Margin']):.0f} pts"
                        winner_str = ht if g["Pred Margin"] > 0 else at

                        card = f"""
<div style="background:{card_colour};border:1px solid #2ecc71;border-radius:10px;padding:16px;margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <div>
      <span style="font-size:1.1rem;font-weight:700;color:white">{ht}</span>
      <span style="color:#e94560;margin:0 8px">vs</span>
      <span style="font-size:1.1rem;font-weight:700;color:white">{at}</span>
    </div>
    <span style="background:#2ecc71;color:#000;font-size:0.72rem;font-weight:700;padding:3px 8px;border-radius:4px">{badge}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem;margin-bottom:4px">BET ON</div>
      <div style="color:#2ecc71;font-weight:700;font-size:1rem">{val_team}</div>
      <div style="color:#666;font-size:0.7rem">@ {val_bookie}</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem;margin-bottom:4px">ODDS</div>
      <div style="color:white;font-weight:700;font-size:1.1rem">${val_odds:.2f}</div>
      <div style="color:#666;font-size:0.7rem">implied {1/val_odds*100:.1f}%</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem;margin-bottom:4px">OUR MODEL</div>
      <div style="color:#3498db;font-weight:700;font-size:1.1rem">{val_our*100:.1f}%</div>
      <div style="color:#2ecc71;font-size:0.7rem">edge: +{val_edge:.1f}%</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem;margin-bottom:4px">KELLY STAKE</div>
      <div style="color:#f39c12;font-weight:700;font-size:1.1rem">${stake:.2f}</div>
      <div style="color:#666;font-size:0.7rem">{kelly_pct:.1f}% of bankroll</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
    <div style="background:#0a1628;border-radius:6px;padding:8px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Expected Return</div>
      <div style="color:#2ecc71;font-weight:600">${exp_return:.2f}</div>
      <div style="color:#666;font-size:0.7rem">profit: ${exp_profit:.2f}</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:8px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Model Prediction</div>
      <div style="color:white;font-weight:600">{winner_str} by ~{pred_margin_str}</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:8px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Best Opp. Odds</div>
      <div style="color:white;font-weight:600">${g["Best Home Odds"] if not is_home_value else g["Best Away Odds"]:.2f} ({g["Best Home Bookie"] if not is_home_value else g["Best Away Bookie"]})</div>
    </div>
  </div>
</div>"""
                        st.markdown(card, unsafe_allow_html=True)

                st.markdown("---")

                # ── Full odds comparison table ────────────────────────────────
                with st.expander("📊 Full odds comparison across all bookmakers"):
                    display_cols = full_df[["Home","Away","Bookmaker",
                                            "Home Odds","Away Odds","H Edge%","A Edge%","Vig%"]].copy()
                    display_cols["H Edge%"] = display_cols["H Edge%"].round(1)
                    display_cols["A Edge%"] = display_cols["A Edge%"].round(1)
                    display_cols["Vig%"]    = display_cols["Vig%"].round(2)
                    st.dataframe(display_cols.sort_values(["Home","Away","Bookmaker"]),
                                 use_container_width=True, hide_index=True)

                # ── Squiggle consensus comparison ─────────────────────────────
                st.markdown("---")
                st.markdown("### 📡 Squiggle Model Consensus")
                st.markdown("*How our model compares to ~15 other computer models*")
                try:
                    consensus_df = get_squiggle_consensus()
                    if not consensus_df.empty:
                        sq_rows = []
                        for _, row in consensus_df.iterrows():
                            ht = normalise_team(str(row.get("hteam","")))
                            at = normalise_team(str(row.get("ateam","")))
                            if ht not in current_elos or at not in current_elos: continue
                            sq_prob = float(row["consensus_home_prob"])
                            our_h, _ = get_our_prob(ht, at)
                            diff = (our_h - sq_prob) * 100
                            sq_rows.append({
                                "Home": ht, "Away": at,
                                "Our Model": f"{our_h*100:.1f}%",
                                "Squiggle Consensus": f"{sq_prob*100:.1f}%",
                                "Difference": f"{diff:+.1f}%",
                                "Models Polled": int(row["n_models"]),
                                "_diff": diff,
                            })
                        if sq_rows:
                            sqdf = pd.DataFrame(sq_rows).sort_values("_diff", key=abs, ascending=False)
                            def _cdiff(val):
                                try:
                                    v = float(str(val).replace("%",""))
                                    if v > 5:  return "color:#2ecc71;font-weight:bold"
                                    if v < -5: return "color:#e74c3c;font-weight:bold"
                                except: pass
                                return ""
                            st.dataframe(
                                sqdf[["Home","Away","Our Model","Squiggle Consensus","Difference","Models Polled"]]
                                .style.applymap(_cdiff, subset=["Difference"]),
                                use_container_width=True, hide_index=True)
                            st.caption("🟢 We're higher than consensus on home team  🔴 We're lower")
                    else:
                        st.info("Squiggle consensus not available for current round.")
                except Exception as e:
                    st.warning(f"Squiggle consensus unavailable: {e}")

                # ── Line betting ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📏 Line Betting (Handicap)")
                st.markdown("*Our model's predicted margin vs the bookmaker's line — find games where the line is off*")
                try:
                    line_df = get_odds_api(odds_key) if odds_key else pd.DataFrame()
                    # Re-fetch with line market
                    try:
                        line_raw = _requests_module.get(
                            "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
                            params={"apiKey": odds_key, "regions": "au",
                                    "markets": "spreads", "oddsFormat": "decimal"},
                            timeout=15)
                        line_raw.raise_for_status()
                        line_rows = []
                        for game in line_raw.json():
                            ht = normalise_team(game.get("home_team",""))
                            at = normalise_team(game.get("away_team",""))
                            if ht not in current_elos or at not in current_elos:
                                continue
                            our_h, pred_margin = get_our_prob(ht, at)
                            for bm in game.get("bookmakers",[]):
                                for market in bm.get("markets",[]):
                                    if market.get("key") != "spreads": continue
                                    for outcome in market.get("outcomes",[]):
                                        team  = normalise_team(outcome.get("name",""))
                                        point = float(outcome.get("point", 0))
                                        price = float(outcome.get("price", 0))
                                        if team not in [ht, at]: continue
                                        # point is the handicap applied TO that team
                                        # +6.5 means they get 6.5 pts start
                                        our_margin = pred_margin if team == ht else -pred_margin
                                        # Does our predicted margin cover this line?
                                        covers = our_margin + point > 0
                                        line_rows.append({
                                            "Match": f"{ht} vs {at}",
                                            "Team": team,
                                            "Line": f"{point:+.1f}",
                                            "Line Odds": f"${price:.2f}",
                                            "Our Pred Margin": f"{our_margin:+.0f}",
                                            "Covers Line?": "✅ Yes" if covers else "❌ No",
                                            "Bookmaker": bm.get("title",""),
                                            "_covers": covers,
                                            "_margin_gap": our_margin + point,
                                        })
                        if line_rows:
                            ldf = pd.DataFrame(line_rows)
                            # Show only where model covers the line
                            covers_df = ldf[ldf["_covers"]].sort_values("_margin_gap", ascending=False)
                            if not covers_df.empty:
                                st.dataframe(
                                    covers_df[["Match","Team","Line","Line Odds","Our Pred Margin","Covers Line?","Bookmaker"]],
                                    use_container_width=True, hide_index=True)
                                st.caption("Shows bets where our model's predicted margin covers the handicap line")
                            else:
                                st.info("No line bets where our model covers the spread this round.")
                        else:
                            st.info("No line/spread markets available from bookmakers right now.")
                    except Exception as le:
                        st.info(f"Line betting data unavailable: {le}")
                except Exception as e:
                    st.warning(f"Could not load line betting: {e}")

                # ── Multi builder ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🎰 Multi Builder")
                st.markdown("*Build a multi from the round's value bets — shows combined odds and expected value*")

                value_games = [(g["Home"], g["Away"], g["H Edge%"], g["A Edge%"],
                                g["Best Home Odds"], g["Best Away Odds"],
                                g["Best Home Bookie"], g["Best Away Bookie"])
                               for _, g in best.iterrows()
                               if g["best_edge"] >= min_edge]

                if not value_games:
                    st.info("No value bets available to build a multi — try lowering the edge threshold.")
                else:
                    st.markdown("**Select legs for your multi:**")
                    multi_legs = []
                    for ht, at, h_edge, a_edge, h_odds, a_odds, h_bookie, a_bookie in value_games:
                        our_h2, _ = get_our_prob(ht, at)
                        our_a2 = 1 - our_h2
                        is_home = h_edge >= a_edge
                        default_team = ht if is_home else at
                        default_odds = h_odds if is_home else a_odds
                        default_bookie = h_bookie if is_home else a_bookie
                        default_prob = our_h2 if is_home else our_a2

                        col_chk, col_sel, col_odds = st.columns([1, 3, 2])
                        with col_chk:
                            include = st.checkbox(f"{ht} vs {at}", key=f"multi_{ht}_{at}", value=True)
                        with col_sel:
                            selection = st.selectbox(
                                "Pick",
                                [f"{ht} (${h_odds:.2f})", f"{at} (${a_odds:.2f})"],
                                index=0 if is_home else 1,
                                key=f"sel_{ht}_{at}",
                                label_visibility="collapsed"
                            )
                        with col_odds:
                            chosen_team = ht if selection.startswith(ht) else at
                            chosen_odds = h_odds if chosen_team == ht else a_odds
                            chosen_prob = our_h2 if chosen_team == ht else our_a2
                            chosen_edge = h_edge if chosen_team == ht else a_edge
                            st.markdown(f"<div style='padding-top:6px;color:#{'2ecc71' if chosen_edge>0 else 'e74c3c'}'>"
                                        f"Edge: {chosen_edge:+.1f}%</div>", unsafe_allow_html=True)

                        if include:
                            multi_legs.append({
                                "match": f"{ht} vs {at}",
                                "team": chosen_team,
                                "odds": chosen_odds,
                                "prob": chosen_prob,
                                "edge": chosen_edge,
                            })

                    if multi_legs:
                        st.markdown("---")
                        combined_odds = 1.0
                        combined_prob = 1.0
                        for leg in multi_legs:
                            combined_odds *= leg["odds"]
                            combined_prob *= leg["prob"]

                        implied_prob = 1.0 / combined_odds
                        multi_edge = (combined_prob - implied_prob) * 100
                        multi_stake = bankroll * max(combined_prob - implied_prob, 0) / (combined_odds - 1) / kelly_divisor * 100 if combined_odds > 1 else 0
                        multi_return = multi_stake * combined_odds

                        # Display summary card
                        leg_colour = "#1a4a1a" if multi_edge > 0 else "#4a1a1a"
                        leg_lines = "".join([
                            f'<div style="color:#aaa;font-size:0.8rem;padding:3px 0;border-bottom:1px solid #1a2a3a">'
                            f'✔ <span style="color:white">{l["team"]}</span> '
                            f'<span style="color:#666">({l["match"]})</span> '
                            f'@ <span style="color:#f39c12">${l["odds"]:.2f}</span> '
                            f'— model: <span style="color:#3498db">{l["prob"]*100:.1f}%</span></div>'
                            for l in multi_legs
                        ])
                        multi_card = f"""
<div style="background:{leg_colour};border:1px solid #{'2ecc71' if multi_edge>0 else 'e74c3c'};border-radius:10px;padding:16px;margin-top:12px">
  <div style="font-size:1rem;font-weight:700;color:white;margin-bottom:10px">
    {len(multi_legs)}-Leg Multi Summary
  </div>
  {leg_lines}
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px">
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Combined Odds</div>
      <div style="color:#f39c12;font-weight:700;font-size:1.1rem">${combined_odds:.2f}</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Our Win Prob</div>
      <div style="color:#3498db;font-weight:700;font-size:1.1rem">{combined_prob*100:.1f}%</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Edge</div>
      <div style="color:{'#2ecc71' if multi_edge>0 else '#e74c3c'};font-weight:700;font-size:1.1rem">{multi_edge:+.1f}%</div>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:10px;text-align:center">
      <div style="color:#aaa;font-size:0.68rem">Kelly Stake → Return</div>
      <div style="color:#{'2ecc71' if multi_edge>0 else '#aaa'};font-weight:700">${multi_stake:.2f} → ${multi_return:.2f}</div>
    </div>
  </div>
  {'<div style="color:#e74c3c;font-size:0.78rem;margin-top:8px">⚠️ Negative edge — this multi has no mathematical value. Consider removing low-edge legs.</div>' if multi_edge <= 0 else '<div style="color:#2ecc71;font-size:0.78rem;margin-top:8px">✅ Positive edge multi — each additional leg multiplies both the odds AND our probability, so value compounds.</div>'}
</div>"""
                        st.markdown(multi_card, unsafe_allow_html=True)
                        st.caption("Note: Multis are high variance — even positive-edge multis lose most of the time. Recommended stake is much smaller than singles.")

                # ── Glossary ──────────────────────────────────────────────────
                st.markdown("---")
                with st.expander("📖 How to read this page"):
                    st.markdown("""
**Edge** — the gap between our model's win probability and the bookmaker's implied probability (after removing their vig/margin). Positive edge means we think the team is more likely to win than the odds suggest.

**Kelly Stake** — mathematically optimal bet size from the Kelly Criterion: `edge / (odds - 1)`. Quarter Kelly is recommended — full Kelly is theoretically optimal but causes huge swings.

**Line Betting** — instead of picking a winner, you bet on a team to win by more than (or lose by less than) a set margin. Our model's predicted margin is compared against the bookmaker's line.

**Multi Builder** — combines multiple bets into one. The combined probability = multiply each leg's probability together. If all legs have positive edge, the multi also has positive edge. But each extra leg also increases variance significantly.

**Expected Return** — stake × odds if the bet wins. Over many bets, positive edge should be profitable — but requires a large sample size to realise.

**Vig** — the bookmaker's built-in margin. Typical AFL vig is 4–7%. Lower vig = better value.

⚠️ *This is a model-based tool, not financial advice. Betting involves real financial risk. Only bet what you can afford to lose.*
""")


# ═══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 How It Works":
    st.markdown("# HOW THE MODEL WORKS")
    st.markdown("*A plain-English explanation of how win probability and margin are calculated*")

    st.markdown("---")

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown("## The Big Picture")
    st.markdown(f"""
The model uses **Gradient Boosting** — a machine learning technique that builds many small decision trees,
each one correcting the mistakes of the last. It's trained on **{metrics['n_games']:,} AFL games from {start_year} to present**.

For each game it outputs two things:
- **Win probability** — the chance the home team wins (e.g. 68%)
- **Predicted margin** — how many points the model expects the game to be won by (e.g. +14 pts)

The model learns which combinations of factors best predict results by seeing thousands of historical games.
It doesn't know anything about footy except what the numbers tell it.
""")

    st.markdown("---")
    st.markdown("## Step by Step: How a Prediction is Made")

    steps = [
        ("1️⃣", "Elo Ratings", "#e94560",
         "Elo is a chess-style rating system applied to AFL teams. Every team starts at 1500. After each game, ratings shift based on the result and how expected it was — beating a strong team gives you more points than beating a weak one. The home team gets a +50 point advantage built in, which reflects the real-world home ground advantage seen in AFL data. The Elo *difference* between the two teams going into a game is the single most predictive feature in the model.",
         f"Current range in our model: {min(current_elos.values()):.0f} – {max(current_elos.values()):.0f} pts"),

        ("2️⃣", "Recent Form", "#f39c12",
         "The model looks at each team's last 5 games and calculates the average winning/losing margin. A team averaging +20 pts over their last 5 is considered in much better form than one averaging -10. It also tracks consistency — a team that wins by 40 one week and loses by 30 the next is treated differently to a team that consistently wins by 15.",
         "Captured as: last5_avg (average margin), last5_std (consistency), last3_avg, last_margin, streak"),

        ("3️⃣", "Travel & Fatigue", "#2ecc71",
         "Interstate travel is genuinely tiring in the AFL, particularly for Perth-based teams or east coast teams flying to Perth. The model calculates straight-line distance each team travels to the venue. This is combined with rest days to create a 'fatigue index': (km ÷ 1000) × max(14 − rest days, 0). A team flying 2,700km to Perth on 6 days rest scores ~8 on this index vs 0 for the home side.",
         "Perth games flagged separately — historical data shows significant away disadvantage at Optus Stadium"),

        ("4️⃣", "Rest Days", "#3498db",
         "Days since each team's last game. More rest generally means fresher legs and more preparation time. The model caps this at 21 days — anything longer (pre-season gap, bye) resets to a neutral 7 days so Round 1 predictions aren't skewed by the entire off-season.",
         "7 days = neutral baseline. <6 days = short turnaround flag. >21 days capped at neutral."),

        ("5️⃣", "Season Stats", "#9b59b6",
         "Season-average statistics from AFL Tables: clearances, inside 50s, contested possessions, tackles, hitouts, and clangers. These capture each team's playing style and execution quality across the whole season. A team that wins clearances tends to control tempo; a team with high inside 50s creates more scoring chances. Clangers are negative — direct turnovers gifting opposition possessions.",
         "Data source: afltables.com — updated once per round"),

        ("6️⃣", "PAV Lineup Strength", "#1abc9c",
         "Player Approximate Value (PAV) is a per-player rating system from Squiggle that estimates each player's total contribution split into offensive, defensive, and midfield value. When lineups are announced (usually Thursday), we sum the PAV of each team's selected 22 to get 'team strength today'. This automatically accounts for injuries, suspensions and selection changes.",
         "Only available after Thursday lineup announcements. Before that, PAV features default to 0."),
    ]

    for icon, title, colour, explanation, detail in steps:
        st.markdown(f"""
<div style="background:#0a1628;border-left:4px solid {colour};border-radius:8px;padding:16px;margin-bottom:12px">
  <div style="font-size:1.05rem;font-weight:700;color:white;margin-bottom:8px">{icon} {title}</div>
  <div style="color:#ccc;font-size:0.85rem;line-height:1.6;margin-bottom:8px">{explanation}</div>
  <div style="color:#666;font-size:0.75rem;font-style:italic">{detail}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## How Win % is Calculated")
    st.markdown("""
The Gradient Boosting model takes all the features above, passes them through hundreds of decision trees,
and outputs a **probability between 0 and 1**. This gets multiplied by 100 to give the win percentage shown on screen.

A probability of **0.68 (68%)** means: *in games with this exact combination of factors, the home team won 68% of the time historically*.

It does **not** mean the home team will definitely win — upsets happen all the time. It means if you played this game 100 times with the same conditions, the home team would win roughly 68 of them.

The margin prediction is separate — it's trained to minimise the average error between predicted and actual margins across all historical games.
""")

    st.markdown("---")
    st.markdown("## Model Performance")

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown(mc(f"{metrics['win_accuracy']*100:.1f}%", "Win Accuracy", "5-fold cross-validation"), unsafe_allow_html=True)
    with pc2:
        st.markdown(mc(f"{metrics['base_accuracy']*100:.1f}%", "Elo-Only Baseline", "without other features"), unsafe_allow_html=True)
    with pc3:
        st.markdown(mc(f"{metrics['margin_r2']:.3f}", "Margin R²", "variance explained"), unsafe_allow_html=True)
    with pc4:
        st.markdown(mc(f"{metrics['n_games']:,}", "Training Games", f"from {start_year}–present"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## What the Model Doesn't Know")
    st.markdown("""
- **Weather and ground conditions** — wet weather significantly impacts game style and scoring
- **In-game events** — momentum shifts, injuries during games, quarter-by-quarter patterns
- **Coaching tactics** — specific match-ups, game plans, rotations
- **Player fitness levels** — PAV captures presence/absence but not fitness percentage
- **Psychological factors** — finals pressure, rivalry games, milestone matches
- **Recent player news** — injuries announced after lineup selection

The model is deliberately simple — it uses only publicly available data and finds patterns in the numbers.
It's roughly as accurate as a well-informed human tipster (~66%), which is genuinely hard to beat consistently.
""")

    st.markdown("---")
    st.markdown("## Comparing to Bookmakers")
    st.markdown("""
Bookmakers employ full teams of analysts with access to far more data. Their implied probabilities
(converted from odds after removing the vig) are typically very well-calibrated.

The **Value Bets page** finds games where our model's probability diverges from the bookmaker's implied probability.
When our model says 65% and the bookmaker prices the team at 55% (implying ~45% after vig removal),
that's a potential edge — but it requires our model to be *more right than the bookmaker* on that specific game,
which is a high bar.

Edge should be assessed over **many bets**, not individual games. The model's long-run accuracy of ~66%
is the ceiling on how often value bets should win.
""")

    # ── Data Staleness & Optimal Training Year ────────────────────────────────
    st.markdown("---")
    st.markdown("## 📅 Optimal Training Start Year")
    st.markdown("""
    One of the trickiest questions is: **how far back should we train?** 

    More data = better statistics. But data from 2013 reflects a completely different player pool —
    most of those players have retired. Training on their patterns means the model is partly learning
    from irrelevant historical noise.

    The chart below shows, for each possible training start year, what percentage of players from that 
    era are **still active today** and how many were in their **prime career stage** (75–149 weighted games).
    A *weighted game* counts finals appearances as **2.5× a regular game** — a Grand Final is worth 
    more formative experience than a Round 5 trip to Hobart.
    """)

    with st.spinner("Analysing data staleness across training years..."):
        staleness = analyse_data_staleness(pav_df, df, start_year=2013)

    if staleness:
        stal_rows = []
        for yr, v in sorted(staleness.items()):
            stal_rows.append({
                "Year":             yr,
                "Players that year": v["n_players"],
                "Still active today": v["n_still_active"],
                "% Still active":   v["pct_still_active"],
                "Prime players":    v["n_prime_players"],
                "Seasons of data":  v["seasons_available"],
                "Relevance score":  v["relevance_score"],
                "Recommended":      "✅" if v.get("recommended") else "",
            })
        stal_df = pd.DataFrame(stal_rows)

        # Chart: % still active + prime players over years
        import plotly.graph_objects as go
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=stal_df["Year"], y=stal_df["% Still active"],
            name="% Players still active today",
            marker_color="#3498db", opacity=0.7
        ))
        fig_s.add_trace(go.Scatter(
            x=stal_df["Year"], y=stal_df["Relevance score"] * 100,
            name="Relevance score (×100)", mode="lines+markers",
            line=dict(color="#e94560", width=2),
            yaxis="y2"
        ))
        # Highlight recommended year
        rec_years = stal_df[stal_df["Recommended"] == "✅"]["Year"].tolist()
        for ry in rec_years:
            fig_s.add_vline(x=ry, line_dash="dash", line_color="#2ecc71",
                            annotation_text=f"Recommended: {ry}", annotation_position="top right")

        fig_s.update_layout(
            paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
            font=dict(color="white"), height=380,
            yaxis=dict(title="% Players still active", color="white"),
            yaxis2=dict(title="Relevance score", overlaying="y", side="right", color="#e94560"),
            xaxis=dict(color="white"),
            legend=dict(bgcolor="#1a1a2e"),
            title="Training Data Relevance by Start Year"
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(stal_df, use_container_width=True, hide_index=True)

        if rec_years:
            best = rec_years[0]
            st.success(f"**Recommended training start year: {best}** — best balance of data volume, player relevance, and prime-career representation. Use the sidebar slider to try different years and compare model accuracy on the Backtest page.")
    else:
        st.info("PAV data needed to run staleness analysis — load a few seasons of data first.")

    # ── Player Experience Feature explainer ───────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎓 Player Experience Features")
    st.markdown("""
    The model now includes **team experience differential** as a feature group. The intuition: 
    a team fielding 18 players with 150+ games each is likely to handle pressure situations 
    better than a team averaging 60 games.

    **How weighted games work:**

    | Career Stage | Regular Games | Finals Multiplier | Notes |
    |---|---|---|---|
    | Developing | 0–24 | ×1.0 | Unproven at elite level |
    | Emerging | 25–74 | ×1.0 | Building consistency |
    | Prime | 75–149 | ×1.0 | Peak athleticism + experience |
    | Veteran | 150–199 | ×1.0 | High experience, declining athleticism |
    | Elite Veteran | 200+ | ×1.0 | Hard-won wisdom |

    Finals games are counted at **×2.5** — a player who plays 10 finals has effectively 
    experienced the equivalent of 25 regular-season pressure games. Grand Finals, Preliminary 
    Finals, and Elimination Finals all count equally in this implementation.

    **The three model features:**
    - `exp_avg_diff`: Home team's average career games minus away team's average. Positive = home team more experienced.
    - `exp_veteran_diff`: Difference in % of veterans (150+ weighted games). High = more battle-hardened side.
    - `exp_developing_diff`: Difference in % of developing players (< 25 games). Negative value is better for home team.
    """)

    # Show current team experience breakdown
    st.markdown("### Current Team Experience (from PAV data)")
    with st.spinner("Computing team experience breakdown..."):
        try:
            exp_table = compute_experience_from_pav(pav_df, df, year=datetime.now().year)
            if not exp_table.empty:
                latest_exp = exp_table[exp_table["year"] == exp_table["year"].max()].copy()
                latest_exp = latest_exp.sort_values("avg_career_games", ascending=False)
                latest_exp["Career Stage Mix"] = latest_exp.apply(
                    lambda r: f"{r['pct_veterans']*100:.0f}% vet / {r['pct_developing']*100:.0f}% dev", axis=1)
                display_exp = latest_exp[["team", "avg_career_games", "med_career_games",
                                          "avg_finals_games", "pct_veterans", "Career Stage Mix"]].copy()
                display_exp.columns = ["Team", "Avg Career Games", "Median Career Games",
                                       "Avg Finals Games", "% Veterans", "Career Mix"]
                display_exp["Avg Career Games"] = display_exp["Avg Career Games"].round(1)
                display_exp["Median Career Games"] = display_exp["Median Career Games"].round(1)
                display_exp["Avg Finals Games"] = display_exp["Avg Finals Games"].round(1)
                display_exp["% Veterans"] = (display_exp["% Veterans"] * 100).round(1)
                st.dataframe(display_exp, use_container_width=True, hide_index=True)
            else:
                st.info("Experience data will populate once PAV data is loaded.")
        except Exception as _e:
            st.warning(f"Experience table error: {_e}")
