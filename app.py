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
                             margin_prediction_backtest,
                             elo_anchor_sweep, FEATURE_GROUPS)
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

@st.cache_data(ttl=1200, show_spinner=False)  # refresh every 20 mins
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
    "bankroll":        100,
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
    start_year = 2016  # Optimal training start year — determined via data staleness analysis
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

def find_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan odds across bookmakers for arbitrage opportunities.
    An arb exists when: 1/best_home_odds + 1/best_away_odds < 1.0
    Returns DataFrame of arb opportunities with profit % and stakes.
    """
    if odds_df.empty:
        return pd.DataFrame()

    rows = []
    for game_key, group in odds_df.groupby(["home_team", "away_team"]):
        ht, at = game_key
        best_h_idx  = group["home_odds"].idxmax()
        best_a_idx  = group["away_odds"].idxmax()
        best_h_odds = group.loc[best_h_idx, "home_odds"]
        best_a_odds = group.loc[best_a_idx, "away_odds"]
        best_h_book = group.loc[best_h_idx, "bookmaker"]
        best_a_book = group.loc[best_a_idx, "bookmaker"]

        arb_pct = (1 / best_h_odds) + (1 / best_a_odds)
        if arb_pct < 1.0:
            profit_pct = round((1 / arb_pct - 1) * 100, 3)
            if profit_pct < 2.5:
                continue  # not worth acting on — need thousands to make meaningful profit
            # Optimal stakes for $100 total outlay
            h_stake = round(100 / (best_h_odds * arb_pct), 2)
            a_stake = round(100 / (best_a_odds * arb_pct), 2)
            rows.append({
                "home":           ht,
                "away":           at,
                "best_home_odds": best_h_odds,
                "best_home_book": best_h_book,
                "best_away_odds": best_a_odds,
                "best_away_book": best_a_book,
                "arb_pct":        round(arb_pct * 100, 2),
                "profit_pct":     profit_pct,
                "h_stake_per100": h_stake,
                "a_stake_per100": a_stake,
            })
    return pd.DataFrame(rows).sort_values("profit_pct", ascending=False) if rows else pd.DataFrame()


def load_games(start_year):
    return get_all_games(start_year)

@st.cache_data(ttl=86400, show_spinner="📊 Fetching AFL Tables season stats...")
def load_season_stats(start_year):
    return get_all_team_season_stats(start_year)

@st.cache_data(ttl=86400, show_spinner="⭐ Fetching PAV player ratings...")
def load_pav(start_year):
    # PAV fetches from 2010 regardless of training start_year —
    # career totals need full history to correctly classify veterans
    return get_pav_multi_year(2010)

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

    # ── Arbitrage alert ───────────────────────────────────────────────────────
    try:
        _odds_key_dash = st.secrets.get("ODDS_API_KEY", "") if hasattr(st, "secrets") else ""
        if _odds_key_dash:
            _arb_odds_dash = get_odds_api(_odds_key_dash)
            if not _arb_odds_dash.empty:
                _arb_found = find_arbitrage(_arb_odds_dash)
                if not _arb_found.empty:
                    for _, _arb in _arb_found.iterrows():
                        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a2a0a,#0d3b0d);border:2px solid #2ecc71;
            border-radius:10px;padding:14px 18px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span style="color:#2ecc71;font-size:1.1rem;font-weight:700">⚡ ARBITRAGE OPPORTUNITY</span>
      <span style="color:white;margin-left:12px;font-size:1rem">{_arb['home']} vs {_arb['away']}</span>
    </div>
    <span style="background:#2ecc71;color:#000;font-weight:700;padding:4px 10px;
                 border-radius:6px;font-size:0.9rem">+{_arb['profit_pct']:.2f}% guaranteed</span>
  </div>
  <div style="color:#aaa;font-size:0.8rem;margin-top:6px">
    Bet <b style="color:white">${_arb['h_stake_per100']:.2f}</b> on {_arb['home']} 
    @ <b style="color:#2ecc71">{_arb['best_home_odds']}</b> ({_arb['best_home_book']})
    &nbsp;+&nbsp;
    <b style="color:white">${_arb['a_stake_per100']:.2f}</b> on {_arb['away']} 
    @ <b style="color:#2ecc71">{_arb['best_away_odds']}</b> ({_arb['best_away_book']})
    &nbsp;·&nbsp; per $100 staked
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    _n_g = _arb_odds_dash.groupby(["home_team","away_team"]).ngroups
                    _n_b = _arb_odds_dash["bookmaker"].nunique()
                    st.caption(f"⚡ Arb scanner: no opportunities ≥2.5% across {_n_g} games / {_n_b} bookmakers — updated every 20 mins")
            else:
                st.caption("⚡ Arb scanner: no odds data yet — check back closer to game day")
    except Exception as _arb_e:
        st.caption(f"⚡ Arb scanner: {_arb_e}")

    try:
        # Fetch ALL games for the year (completed + upcoming)
        import requests as _req
        _r = _req.get(f"https://api.squiggle.com.au/?q=games;year={datetime.now().year}",
                      headers={"User-Agent": "AFL-Predictor/1.0"}, timeout=15)
        _all_games = pd.DataFrame(_r.json().get("games", []))
        _incomplete = _all_games[_all_games["complete"] < 100] if not _all_games.empty else pd.DataFrame()
        _complete   = _all_games[_all_games["complete"] == 100] if not _all_games.empty else pd.DataFrame()

        # Round picker — all rounds that have any game (complete or upcoming)
        if not _all_games.empty:
            _all_rounds = sorted(_all_games["round"].unique())
            # Default to the lowest round with incomplete games (upcoming), else latest completed
            if not _incomplete.empty:
                _default_round = int(_incomplete["round"].min())
            else:
                _default_round = int(_all_rounds[-1])
            if st.session_state.get("selected_round") not in _all_rounds:
                st.session_state["selected_round"] = _default_round

            # ── Compact ladder ────────────────────────────────────────────
            _cur_standings = standings_df[standings_df["year"] == datetime.now().year].copy() if standings_df is not None and not standings_df.empty else pd.DataFrame()
            if not _cur_standings.empty:
                _pct_col = "percentage" if "percentage" in _cur_standings.columns else "pct"
                _cur_standings = _cur_standings.sort_values("rank").head(18)

                st.markdown("### 🏆 LADDER")
                _rows_html = ""
                for _, _lr in _cur_standings.iterrows():
                    _pos   = int(_lr.get("rank", 0))
                    _team  = str(_lr.get("team", ""))
                    _w     = int(_lr.get("wins", 0))
                    _l     = int(_lr.get("losses", 0))
                    _d     = int(_lr.get("draws", 0))
                    _pct   = float(_lr.get(_pct_col, 0))
                    _pts   = int(_lr.get("pts", _w * 4))
                    _elo   = float(current_elos.get(_team, 1500))

                    # Top 8 highlight, top 4 stronger
                    if _pos <= 4:   _bg = "#0d2a3d"; _pos_col = "#3498db"
                    elif _pos <= 8: _bg = "#0a1f2e"; _pos_col = "#aaa"
                    else:           _bg = "#0a1628"; _pos_col = "#555"

                    _rows_html += f"""
<div style="display:grid;grid-template-columns:28px 1fr 36px 36px 36px 64px 64px;
            align-items:center;gap:6px;padding:6px 10px;background:{_bg};
            border-radius:6px;margin-bottom:3px;font-size:0.82rem">
  <div style="color:{_pos_col};font-weight:700;text-align:center">{_pos}</div>
  <div style="color:white;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{_team}</div>
  <div style="color:#2ecc71;text-align:center;font-weight:600">{_w}</div>
  <div style="color:#e74c3c;text-align:center">{_l}</div>
  <div style="color:#aaa;text-align:center">{_d}</div>
  <div style="color:#f39c12;text-align:center;font-weight:600">{_pct:.1f}%</div>
  <div style="color:#3498db;text-align:center">{_elo:.0f}</div>
</div>"""

                _header_html = """
<div style="display:grid;grid-template-columns:28px 1fr 36px 36px 36px 64px 64px;
            align-items:center;gap:6px;padding:4px 10px;margin-bottom:4px;font-size:0.68rem">
  <div style="color:#555;text-align:center">#</div>
  <div style="color:#555">TEAM</div>
  <div style="color:#555;text-align:center">W</div>
  <div style="color:#555;text-align:center">L</div>
  <div style="color:#555;text-align:center">D</div>
  <div style="color:#555;text-align:center">PCT %</div>
  <div style="color:#555;text-align:center">ELO</div>
</div>"""

                st.markdown(
                    f'<div style="background:#0a1628;border-radius:10px;padding:10px 4px">'
                    f'{_header_html}{_rows_html}'
                    f'<div style="color:#333;font-size:0.65rem;text-align:right;padding:4px 10px 0">'
                    f'■■■■ top 4 &nbsp; ■■■■ top 8</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")

            col_title, col_picker = st.columns([2, 1])
            with col_title:
                st.markdown("## ROUND PREVIEW & RESULTS")
            with col_picker:
                selected_round = st.selectbox("Round", _all_rounds,
                                              index=_all_rounds.index(st.session_state["selected_round"]),
                                              key="selected_round",
                                              label_visibility="collapsed")

            if selected_round < 3:
                _ladder_w = 0.0
            elif selected_round <= 5:
                _ladder_w = 0.15
            elif selected_round <= 9:
                _ladder_w = 0.25
            else:
                _ladder_w = 1.0
            _ladder_note = f"Ladder weight: {_ladder_w:.0%}" if _ladder_w < 1.0 else "Ladder: full weight"
            upcoming     = _incomplete[_incomplete["round"] == selected_round].copy()
            _done_this   = _complete[_complete["round"] == selected_round].copy()
            _n_done      = len(_done_this)
            _n_up        = len(upcoming)
            _summary     = f"Round {selected_round} · {_n_done} completed · {_n_up} upcoming · {_ladder_note}"
            st.markdown(f"*{_summary}*")
            # DEBUG — remove once confirmed working
            _yr_games = df[df["year"] == datetime.now().year]
            _last_completed = int(_yr_games["round"].max()) if not _yr_games.empty else 0
            st.caption(f"🔧 Debug: selected_round={selected_round}, last completed in df={_last_completed}, form_weight={'0%' if selected_round<=2 else '40%' if selected_round<=5 else '75%' if selected_round<=9 else '100%'}, ladder_weight={_ladder_w:.0%}")
        else:
            st.markdown("## ROUND PREVIEW & RESULTS")
            upcoming   = pd.DataFrame()
            _done_this = pd.DataFrame()
            selected_round = None

        lineup_df = load_lineups()
        lineup_strength = {}
        if not lineup_df.empty and not pav_df.empty:
            lineup_strength = compute_lineup_strength(lineup_df, pav_df)

        # ── Completed games for this round ─────────────────────────────────
        if not _done_this.empty:
            for _, _cg in _done_this.iterrows():
                _h  = str(_cg.get("hteam", "?"))
                _a  = str(_cg.get("ateam", "?"))
                _hs = int(_cg.get("hscore", 0) or 0)
                _as = int(_cg.get("ascore", 0) or 0)
                _venue = str(_cg.get("venue", ""))
                _actual_winner = _h if _hs > _as else (_a if _as > _hs else "Draw")
                _actual_margin = abs(_hs - _as)

                _pred_winner = None
                _pred_prob   = None
                _pred_margin = None
                _correct     = None
                _missing_elo = _h not in current_elos or _a not in current_elos
                if not _missing_elo:
                    try:
                        _pf = _build_prediction_features(
                            _h, _a, _venue, current_elos, team_stats,
                            season_stats, {}, df, exp_df, standings_df,
                            style_df=style_df, current_round=int(selected_round)
                        )
                        _pp = predict_game(win_model, margin_model, _pf, metrics["features_used"])
                        _pred_winner = _h if _pp["home_win_prob"] > 50 else _a
                        _pred_prob   = _pp["home_win_prob"] if _pred_winner == _h else _pp["away_win_prob"]
                        _pred_margin = abs(_pp["predicted_margin"])
                        _correct     = (_pred_winner == _actual_winner)
                    except Exception as _pred_err:
                        _pred_winner = f"ERR: {_pred_err}"

                _border_col = "#2ecc71" if _correct else ("#e74c3c" if _correct is False else "#444")
                _tick       = "✅" if _correct else ("❌" if _correct is False else "")
                _margin_err_str = ""
                if _pred_margin is not None:
                    _margin_err_str = f" · margin off by {abs(_pred_margin - _actual_margin):.0f} pts"

                # Win probability bar (home = left, away = right)
                _hprob = _pp["home_win_prob"] if (_pred_winner is not None) else 50
                _aprob = 100 - _hprob

                # Actual result bar — home share of total score
                _total_score  = _hs + _as if (_hs + _as) > 0 else 1
                _actual_h_pct = int(_hs / _total_score * 100)
                _actual_a_pct = 100 - _actual_h_pct
                _margin_err_val = abs(_pred_margin - _actual_margin) if _pred_margin is not None else None

                st.markdown(f"""
<div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;margin-bottom:10px;border-left:4px solid {_border_col}">

  <!-- Teams + scoreline -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <div style="font-size:1.1rem;font-weight:700;color:white">{_h}</div>
    <div style="color:#888;font-size:0.75rem;letter-spacing:1px">FINAL</div>
    <div style="font-size:1.1rem;font-weight:700;color:white">{_a}</div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
    <div style="font-size:1.8rem;font-weight:900;color:{'#2ecc71' if _actual_winner==_h else '#aaa'}">{_hs}</div>
    <div style="font-size:0.72rem;color:#555">{_venue}</div>
    <div style="font-size:1.8rem;font-weight:900;color:{'#2ecc71' if _actual_winner==_a else '#aaa'}">{_as}</div>
  </div>

  <!-- Prediction bar -->
  <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-bottom:2px">
    <span>MODEL TIP</span>
    <span>{_tick} {_pred_winner or '—'} by {f'{_pred_margin:.0f}' if _pred_margin else '—'} pts</span>
    <span></span>
  </div>
  <div style="height:8px;border-radius:4px;background:#0f3460;overflow:hidden;margin-bottom:2px">
    <div style="width:{_hprob}%;height:100%;background:linear-gradient(90deg,#e94560,#ff6b6b);border-radius:4px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#666;margin-bottom:8px">
    <span>{_hprob}%</span><span>{_aprob}%</span>
  </div>

  <!-- Actual result bar -->
  <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-bottom:2px">
    <span>ACTUAL</span>
    <span style="color:{'#2ecc71' if _margin_err_val is not None and _margin_err_val <= 12 else '#f39c12' if _margin_err_val is not None and _margin_err_val <= 25 else '#e74c3c' if _margin_err_val is not None else '#888'}">
      {f'margin off by {_margin_err_val:.0f} pts' if _margin_err_val is not None else ''}
    </span>
    <span></span>
  </div>
  <div style="height:8px;border-radius:4px;background:#0f3460;overflow:hidden">
    <div style="width:{_actual_h_pct}%;height:100%;background:linear-gradient(90deg,#2ecc71,#27ae60);border-radius:4px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#666;margin-top:2px">
    <span>{_actual_h_pct}%</span><span>{_actual_a_pct}%</span>
  </div>

</div>
""", unsafe_allow_html=True)

        if upcoming.empty and _done_this.empty:
            st.info("No games found for this round.")
        elif not upcoming.empty:
            st.markdown("### UPCOMING PREDICTIONS")

        if upcoming.empty and not _done_this.empty:
            pass  # all done, no upcoming to show
        elif upcoming.empty:
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

                    # Map display labels to feature names for importance lookup
                    _label_to_feats = {
                        "Elo Rating":           ["elo_diff"],
                        "Form (last 5 avg)":    ["form_diff", "home_form", "away_form"],
                        "Current Streak":       ["streak_diff", "home_streak", "away_streak"],
                        "Last Game Margin":     ["last_margin_diff", "last3_diff"],
                        "Travel to Venue":      ["travel_diff", "travel_home_km", "travel_away_km"],
                        "Days Rest":            ["days_rest_diff"],
                        "Travel Fatigue":       ["travel_fatigue_diff"],
                        "Clearances (season)":  ["cl_diff"],
                        "Inside 50s (season)":  ["i50_diff"],
                        "Contested Poss":       ["cp_diff"],
                        "Tackles (season)":     ["tk_diff"],
                        "Clangers (season)":    ["clanger_diff"],
                    }

                    # Get feature importances from fi_df
                    _fi_lookup = {}
                    if fi_df is not None and not fi_df.empty:
                        for _, _frow in fi_df.iterrows():
                            _fi_lookup[_frow["feature"]] = float(_frow["importance"])

                    # Total importance for normalisation
                    _total_imp = sum(_fi_lookup.values()) if _fi_lookup else 1.0

                    all_rows = []
                    for label, hv, av, hib, unit, explanation in factor_meta:
                        if hv == 0 and av == 0:
                            continue
                        diff = hv - av if hib else av - hv
                        if diff > 0.5:    edge_tag = f"✅ {home}"
                        elif diff < -0.5: edge_tag = f"✅ {away}"
                        else:             edge_tag = "— Even"

                        # Estimate pts impact: feature importance × diff × scaling constant
                        # Feature importance (0-1) × predicted margin gives rough pts contribution
                        _feats = _label_to_feats.get(label, [])
                        _imp = sum(_fi_lookup.get(f, 0) for f in _feats)
                        _imp_share = _imp / _total_imp if _total_imp > 0 else 0
                        # Scale: importance share × total predicted margin × direction sign
                        _pts_impact = _imp_share * margin * (1 if diff > 0 else -1)
                        _pts_str = (f"+{_pts_impact:.1f} pts ({home})"
                                    if _pts_impact > 0.3
                                    else (f"+{abs(_pts_impact):.1f} pts ({away})"
                                          if _pts_impact < -0.3
                                          else "< 0.3 pts"))

                        all_rows.append({
                            "Factor":          label,
                            home:              f"{hv:.1f} {unit}",
                            away:              f"{av:.1f} {unit}",
                            "Edge":            edge_tag,
                            "Est. Pts Impact": _pts_str,
                        })
                    if all_rows:
                        st.dataframe(pd.DataFrame(all_rows), width='stretch', hide_index=True)
                        st.caption("Est. Pts Impact = each factor's share of feature importance × predicted margin. Rough guide only — the GBM combines all features non-linearly.")

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
    avail = [f for f in metrics["features_used"] if f in df.columns]
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
        st.plotly_chart(fig, width='stretch')

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

    if st.button("🔮 PREDICT", width='stretch'):
        # current_round: prefer the round picker selection from session state
        # Fall back to last completed + 1, minimum 1
        if st.session_state.get("selected_round"):
            _pred_round = int(st.session_state["selected_round"])
        else:
            _yr_df = df[df["year"] == datetime.now().year]
            _pred_round = int(_yr_df["round"].max()) + 1 if not _yr_df.empty else 1
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

        # DEBUG — dump raw features fed to model
        with st.expander("🔧 Debug: raw feature values", expanded=True):
            _feat_rows = []
            for k, v in sorted(feats.items()):
                if k in metrics["features_used"]:
                    _feat_rows.append({"feature": k, "value": round(float(v), 4)})
            st.dataframe(pd.DataFrame(_feat_rows), hide_index=True, width="stretch")
            st.write(f"current_round passed: {_pred_round}")

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
        st.plotly_chart(fig, width='stretch')

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
            st.dataframe(bd_df, width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM FORM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Team Form":
    st.markdown("# TEAM FORM")

    _fc1, _fc2 = st.columns([2, 1])
    with _fc1:
        selected = st.selectbox("Select Team", teams, key="form_team")
    with _fc2:
        n = st.slider("Games", 5, 30, 10, key="form_n")

    form = get_team_form_df(selected, n)

    if form.empty:
        st.warning("No data found.")
    else:
        hs     = team_stats.get(selected, {})
        wins   = (form["result"] == "W").sum()
        losses = (form["result"] == "L").sum()
        draws  = (form["result"] == "D").sum()
        avg_margin   = form["margin"].mean()
        biggest_win  = form["margin"].max()
        biggest_loss = form["margin"].min()
        home_form = form[form["venue_type"] == "Home"]
        away_form = form[form["venue_type"] == "Away"]
        home_win_rate = (home_form["result"] == "W").mean() * 100 if len(home_form) else 0
        away_win_rate = (away_form["result"] == "W").mean() * 100 if len(away_form) else 0

        # Top stats
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(mc(
            f"{wins}W {losses}L{f' {draws}D' if draws else ''}",
            f"Last {n} Games",
            f"{wins/n*100:.0f}% win rate"
        ), unsafe_allow_html=True)
        with c2: st.markdown(mc(
            f"{avg_margin:+.1f} pts",
            "Avg Margin",
            "positive = winning by"
        ), unsafe_allow_html=True)
        with c3: st.markdown(mc(
            f"{hs.get('streak', 0):+d}",
            "Current Streak",
            "positive = wins in a row"
        ), unsafe_allow_html=True)
        with c4: st.markdown(mc(
            f"{home_win_rate:.0f}% / {away_win_rate:.0f}%",
            "Home / Away Win %",
            f"{len(home_form)}H {len(away_form)}A games"
        ), unsafe_allow_html=True)
        with c5: st.markdown(mc(
            f"{current_elos.get(selected, 1500):.0f}",
            "Elo Rating",
            f"best win: +{biggest_win:.0f} · worst: {biggest_loss:.0f}"
        ), unsafe_allow_html=True)

        st.markdown("---")

        # Last 10 game result cards
        st.markdown(f"### Last {n} Results")
        _cards = ""
        for _, row in form.iloc[::-1].iterrows():
            _res_col = "#2ecc71" if row["result"] == "W" else ("#e74c3c" if row["result"] == "L" else "#f39c12")
            _margin_str = f"+{row['margin']:.0f}" if row["margin"] > 0 else f"{row['margin']:.0f}"
            _venue_icon = "🏠" if row["venue_type"] == "Home" else "✈️"
            _year_rnd = f"R{int(row['round'])} {int(row['year'])}"
            _cards += f"""
<div style='display:inline-flex;flex-direction:column;align-items:center;
            background:#0a1628;border-radius:8px;padding:8px 10px;margin:3px;
            border-top:3px solid {_res_col};min-width:80px'>
  <div style='color:{_res_col};font-weight:700;font-size:1.1rem'>{row["result"]}</div>
  <div style='color:white;font-size:0.72rem;font-weight:600'>{_margin_str}</div>
  <div style='color:#aaa;font-size:0.65rem'>{_venue_icon} vs {row["opponent"][:3].upper()}</div>
  <div style='color:#555;font-size:0.62rem'>{_year_rnd}</div>
</div>"""
        st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:2px'>{_cards}</div>",
                    unsafe_allow_html=True)

        st.markdown("---")

        # Margin bar chart
        colors = ["#2ecc71" if m > 0 else "#e94560" for m in form["margin"]]
        _labels = [f"R{int(r['round'])} {str(r['year'])[2:]} vs {r['opponent'][:3].upper()}"
                   for _, r in form.iterrows()]
        fig = go.Figure(go.Bar(
            x=_labels, y=form["margin"],
            marker_color=colors,
            text=[f"{m:+.0f}" for m in form["margin"]],
            textposition="outside"
        ))
        fig.add_hline(y=0, line_color="white", line_width=1)
        fig.add_hline(y=avg_margin, line_dash="dot", line_color="#f39c12",
                      annotation_text=f"  avg {avg_margin:+.1f}", annotation_font_color="#f39c12")
        dark_chart(fig, 380)
        fig.update_layout(
            title=f"{selected} — Last {n} Game Margins",
            xaxis=dict(tickangle=-45, title=""),
            yaxis=dict(title="Margin (pts)")
        )
        st.plotly_chart(fig, width='stretch')

        # Cumulative margin trend
        form["cumulative"] = form["margin"].cumsum()
        fig2 = go.Figure(go.Scatter(
            x=_labels, y=form["cumulative"],
            mode="lines+markers",
            line=dict(color="#e94560", width=2),
            marker=dict(size=6, color=colors),
            fill="tozeroy", fillcolor="rgba(233,69,96,0.08)"
        ))
        fig2.add_hline(y=0, line_color="#555", line_width=1)
        dark_chart(fig2, 260)
        fig2.update_layout(
            title="Cumulative Margin Trend",
            xaxis=dict(tickangle=-45, title=""),
            yaxis=dict(title="Cumulative pts")
        )
        st.plotly_chart(fig2, width='stretch')

        # Home vs Away breakdown
        if len(home_form) > 0 and len(away_form) > 0:
            st.markdown("### Home vs Away")
            _hac1, _hac2 = st.columns(2)
            with _hac1:
                h_colors = ["#2ecc71" if m > 0 else "#e94560" for m in home_form["margin"]]
                fig3 = go.Figure(go.Bar(
                    x=[f"R{int(r['round'])} vs {r['opponent'][:3].upper()}"
                       for _, r in home_form.iterrows()],
                    y=home_form["margin"], marker_color=h_colors,
                    text=[f"{m:+.0f}" for m in home_form["margin"]],
                    textposition="outside"
                ))
                fig3.add_hline(y=0, line_color="white", line_width=1)
                dark_chart(fig3, 280)
                fig3.update_layout(title=f"🏠 Home ({home_win_rate:.0f}% wins)",
                                   xaxis=dict(tickangle=-45))
                st.plotly_chart(fig3, width='stretch')
            with _hac2:
                a_colors = ["#2ecc71" if m > 0 else "#e94560" for m in away_form["margin"]]
                fig4 = go.Figure(go.Bar(
                    x=[f"R{int(r['round'])} vs {r['opponent'][:3].upper()}"
                       for _, r in away_form.iterrows()],
                    y=away_form["margin"], marker_color=a_colors,
                    text=[f"{m:+.0f}" for m in away_form["margin"]],
                    textposition="outside"
                ))
                fig4.add_hline(y=0, line_color="white", line_width=1)
                dark_chart(fig4, 280)
                fig4.update_layout(title=f"✈️ Away ({away_win_rate:.0f}% wins)",
                                   xaxis=dict(tickangle=-45))
                st.plotly_chart(fig4, width='stretch')

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
    st.plotly_chart(fig, width='stretch')
    st.dataframe(elo_df, width='stretch', hide_index=True)

    # ── Elo Anchor Tuner ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Elo Anchor Tuner")
    st.markdown(
        "*How much should Elo influence predictions vs the GBM? "
        "Sweep anchor weights 0%→100% to find the accuracy sweet spot. "
        "0% = pure GBM, 100% = pure Elo.*"
    )
    abl_min_train_elo = st.slider("Minimum training years", 2, 5, 3, key="elo_anchor_min_train")

    if st.button("▶ Run Elo Anchor Sweep", type="primary", key="run_elo_sweep"):
        with st.spinner("Sweeping Elo anchor weights 0%→100% — takes ~60 seconds..."):
            sweep_df = elo_anchor_sweep(df, win_model, margin_model, metrics,
                                        min_train_years=abl_min_train_elo)
        st.session_state["elo_sweep_result"] = sweep_df

    sweep_df = st.session_state.get("elo_sweep_result")
    if sweep_df is not None and not sweep_df.empty:
        best_row    = sweep_df.loc[sweep_df["accuracy"].idxmax()]
        best_anchor = best_row["elo_anchor"]
        best_acc    = best_row["accuracy"]

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(mc(f"{best_anchor:.0%}", "Optimal Elo Anchor",
                                f"Peak accuracy {best_acc:.1f}%"), unsafe_allow_html=True)
        with c2:
            v = sweep_df[sweep_df["elo_anchor"]==0.0]["accuracy"].values
            st.markdown(mc(f"{v[0]:.1f}%" if len(v) else "—", "Pure GBM (0% Elo)"), unsafe_allow_html=True)
        with c3:
            v = sweep_df[sweep_df["elo_anchor"]==1.0]["accuracy"].values
            st.markdown(mc(f"{v[0]:.1f}%" if len(v) else "—", "Pure Elo (100%)"), unsafe_allow_html=True)

        st.markdown("---")

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["elo_anchor"]*100, y=sweep_df["accuracy"],
            mode="lines+markers", line=dict(color="#e94560", width=2),
            marker=dict(size=8), name="Accuracy"
        ))
        fig_sweep.add_vline(x=best_anchor*100, line_dash="dot", line_color="#2ecc71",
                            annotation_text=f"  optimal: {best_anchor:.0%}",
                            annotation_font_color="#2ecc71")
        dark_chart(fig_sweep, 320)
        fig_sweep.update_layout(
            xaxis=dict(title="Elo Anchor Weight (%)", ticksuffix="%"),
            yaxis=dict(title="Out-of-Sample Accuracy (%)",
                       range=[sweep_df["accuracy"].min()-1, sweep_df["accuracy"].max()+1])
        )
        st.plotly_chart(fig_sweep, width="stretch")

        fig_brier = go.Figure()
        fig_brier.add_trace(go.Scatter(
            x=sweep_df["elo_anchor"]*100, y=sweep_df["brier_score"],
            mode="lines+markers", line=dict(color="#3498db", width=2),
            marker=dict(size=8), name="Brier Score"
        ))
        best_b = sweep_df.loc[sweep_df["brier_score"].idxmin()]
        fig_brier.add_vline(x=best_b["elo_anchor"]*100, line_dash="dot", line_color="#2ecc71",
                            annotation_text=f"  best calibration: {best_b['elo_anchor']:.0%}",
                            annotation_font_color="#2ecc71")
        dark_chart(fig_brier, 280)
        fig_brier.update_layout(
            xaxis=dict(title="Elo Anchor Weight (%)", ticksuffix="%"),
            yaxis=dict(title="Brier Score (lower = better calibrated)")
        )
        st.plotly_chart(fig_brier, width="stretch")

        disp = sweep_df.copy()
        disp["elo_anchor"] = disp["elo_anchor"].apply(lambda x: f"{x:.0%}")
        disp.columns = ["Elo Anchor", "Accuracy %", "Brier Score", "Games"]
        st.dataframe(disp, hide_index=True, width="stretch")
        st.caption(f"☝️ Paste to Claude: optimal Elo anchor = {best_anchor:.0%} "
                   f"({best_acc:.1f}% accuracy, Brier {best_row['brier_score']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STATS LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Team Stats":
    st.markdown("# TEAM STATS LEADERBOARD")
    st.markdown("*Season averages per game — all 18 teams ranked*")

    with st.expander("🔧 Debug: AFL Tables scraper"):
        import requests as _rq
        from bs4 import BeautifulSoup as _BS
        _year = datetime.now().year
        _url  = f"https://afltables.com/afl/stats/{_year}t.html"
        st.markdown(f"Fetching: `{_url}`")
        try:
            _r = _rq.get(_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            st.markdown(f"Status: `{_r.status_code}` | Length: `{len(_r.text)}`")
            _soup = _BS(_r.text, "lxml")
            _tables = _soup.find_all("table")
            st.markdown(f"Tables found: `{len(_tables)}`")
            if _tables:
                # Show first few rows of first table raw
                _rows = _tables[0].find_all("tr")[:5]
                for _row in _rows:
                    _cells = [td.get_text(strip=True) for td in _row.find_all(["td","th"])]
                    st.write(_cells)
        except Exception as _e:
            st.error(f"Fetch error: {_e}")

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
                    st.plotly_chart(fig, width='stretch')

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
                    st.dataframe(full_table, width='stretch', hide_index=True)

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
                        st.plotly_chart(fig_r, width='stretch')
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
        "Elo":               "#e94560",
        "Form (rolling)":    "#e67e22",
        "Streak":            "#1abc9c",
        "Last margin":       "#3498db",
        "Season stats":      "#9b59b6",
        "Stats: Clearances": "#8e44ad",
        "Stats: Inside 50s": "#7d3c98",
        "Stats: Tackles":    "#6c3483",
        "Stats: Hitouts":    "#5b2c6f",
        "Ladder position":   "#2ecc71",
        "Style: Kick ratio": "#f39c12",
        "Style: Hitouts":    "#e67e22",
        "Other":             "#555",
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
    st.plotly_chart(fig, width='stretch')

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
    st.plotly_chart(fig2, width='stretch')

    # ── Live Ablation ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 Live Ablation")
    st.markdown(
        "*Leave-one-out: remove each feature group and measure accuracy impact. "
        "Run this then paste the results to Claude to tune the model.*"
    )

    abl_min_train = st.slider(
        "Minimum training years", 2, 5, 3, key="fi_abl_min_train"
    )

    if st.button("▶ Run Ablation", type="primary", key="fi_run_ablation"):
        with st.spinner("Running ablation — this takes ~30 seconds..."):
            abl_df = ablation_test(df, FEATURE_GROUPS, abl_min_train)
        st.session_state["fi_ablation_result"] = abl_df

    abl_df = st.session_state.get("fi_ablation_result")

    if abl_df is not None and not abl_df.empty:

        display_df = abl_df[["group", "accuracy", "delta", "n_features", "interpretation"]].copy()
        display_df.columns = ["Feature Group", "Accuracy %", "Δ vs Baseline", "# Features", "Verdict"]
        display_df["Δ vs Baseline"] = display_df["Δ vs Baseline"].apply(
            lambda x: f"{x:+.2f}%" if x != 0 else "—"
        )

        # Colour-code using iloc positions (styler passes renamed columns as index)
        def _abl_row_color(row):
            # row.iloc[0] = "Feature Group" column value
            if row.iloc[0] == "ALL FEATURES (baseline)":
                return ["background-color:#1a1a2e; color:#aaa"] * len(row)
            delta_str = row.iloc[2]  # "Δ vs Baseline" — already formatted string
            try:
                delta_val = float(delta_str.replace("%", "").replace("—", "0"))
            except (ValueError, AttributeError):
                delta_val = 0.0
            c = ("#2ecc71" if delta_val < -0.3
                 else "#e74c3c" if delta_val > 0.3
                 else "#f39c12")
            return ["" if i not in (2, 4) else f"color:{c}" for i in range(len(row))]

        st.dataframe(
            display_df.style.apply(_abl_row_color, axis=1),
            width='stretch',
            hide_index=True,
        )

        # Copy-for-Claude button
        baseline_row = abl_df[abl_df["group"] == "ALL FEATURES (baseline)"].iloc[0]
        lines = [
            f"## Ablation Results — {abl_min_train}yr min training",
            f"Baseline: {baseline_row['accuracy']:.1f}% ({int(baseline_row['n_features'])} features)",
            "",
            f"{'Group':<30} {'Accuracy':>9} {'Delta':>8} {'Verdict'}",
            "-" * 65,
        ]
        for _, row in abl_df.iterrows():
            if row["group"] == "ALL FEATURES (baseline)":
                continue
            delta_str = f"{row['delta']:+.2f}%" if row["delta"] != 0 else "—"
            lines.append(
                f"{row['group']:<30} {row['accuracy']:>8.1f}% {delta_str:>8}  {row['interpretation']}"
            )

        lines += [
            "",
            f"Model: {metrics['win_accuracy']*100:.1f}% CV accuracy, "
            f"{metrics['n_games']} games, "
            f"{metrics['n_features']} features",
            f"Start year: {metrics.get('features_used', ['?'])[0] if metrics else '?'}, "
            f"N games: {metrics.get('n_games','?')}, N features: {metrics.get('n_features','?')}",
        ]

        copy_text = "\n".join(lines)

        st.code(copy_text, language=None)
        st.caption("☝️ Copy the block above and paste it to Claude to tune the model.")

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Backtest":
    st.markdown("# MODEL PERFORMANCE HISTORY")
    st.markdown(
        "*How would our current model have performed on past seasons? "
        "Each season is predicted using only data available at the time — no future information.*"
    )

    with st.spinner("Replaying model on historical seasons..."):
        avail_feats = [f for f in CORE_FEATURES if f in df.columns]
        bt_df = run_walk_forward_backtest(df, avail_feats, 3)

    if bt_df.empty:
        st.warning("Not enough historical data to run backtest.")
    else:
        yearly_acc   = compute_yearly_accuracy(bt_df)
        overall_acc  = bt_df["correct"].mean() * 100
        overall_games = len(bt_df)
        upsets   = bt_df[bt_df["prob"] < 0.4]
        big_favs = bt_df[bt_df["prob"] > 0.7]

        # ── Top stats ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(mc(
            f"{overall_acc:.1f}%", "Overall Accuracy",
            f"{overall_games} games across {len(yearly_acc)} seasons"
        ), unsafe_allow_html=True)
        with c2: st.markdown(mc(
            f"{yearly_acc['brier_score'].mean():.3f}", "Avg Calibration Score",
            "Lower = probabilities are more accurate"
        ), unsafe_allow_html=True)
        with c3: st.markdown(mc(
            f"{upsets['correct'].mean()*100:.1f}%" if len(upsets) else "—",
            "Upset Detection",
            f"{len(upsets)} games where we gave <40% — did we spot them?"
        ), unsafe_allow_html=True)
        with c4: st.markdown(mc(
            f"{big_favs['correct'].mean()*100:.1f}%" if len(big_favs) else "—",
            "Big Favourite Accuracy",
            f"{len(big_favs)} games where we gave >70%"
        ), unsafe_allow_html=True)

        st.markdown("---")

        # ── Accuracy by season ────────────────────────────────────────────────
        st.markdown("### Accuracy by Season")
        st.markdown("*How many games did we tip correctly each year?*")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_acc["year"], y=yearly_acc["accuracy"],
            marker_color="#e94560",
            text=yearly_acc["accuracy"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside"
        ))
        fig.add_hline(y=50,         line_dash="dash", line_color="#555",
                      annotation_text="  coin flip")
        fig.add_hline(y=overall_acc, line_dash="dot", line_color="#2ecc71",
                      annotation_text=f"  avg {overall_acc:.1f}%")
        dark_chart(fig, 350)
        fig.update_layout(yaxis=dict(range=[40, 85], title="Accuracy %"),
                          xaxis=dict(title="Season"))
        st.plotly_chart(fig, width='stretch')

        # ── Margin error by season ────────────────────────────────────────────
        st.markdown("### Margin Error by Season")
        st.markdown("*On average, how many points was our margin prediction off by each year?*")
        margin_bt = margin_prediction_backtest(df, avail_feats, 3)
        if not margin_bt.empty:
            avg_mae = margin_bt["mae_points"].mean()
            fig4 = go.Figure(go.Bar(
                x=margin_bt["year"], y=margin_bt["mae_points"],
                marker_color="#9b59b6",
                text=margin_bt["mae_points"].apply(lambda x: f"{x:.1f} pts"),
                textposition="outside"
            ))
            fig4.add_hline(y=avg_mae, line_dash="dot", line_color="#2ecc71",
                           annotation_text=f"  avg {avg_mae:.1f} pts")
            dark_chart(fig4, 300)
            fig4.update_layout(yaxis=dict(title="Mean Absolute Error (pts)"),
                               xaxis=dict(title="Season"))
            st.plotly_chart(fig4, width='stretch')

        # ── Probability calibration ───────────────────────────────────────────
        st.markdown("### Probability Calibration")
        st.markdown(
            "*When we say 70% — does the team actually win 70% of the time? "
            "Bars should follow the green diagonal.*"
        )
        bins = pd.cut(bt_df["prob"], bins=10)
        cal  = bt_df.groupby(bins, observed=True).agg(
            mean_prob=("prob",   "mean"),
            actual_rate=("actual", "mean"),
            n=("actual", "count")
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=cal["mean_prob"], y=cal["actual_rate"],
            marker_color="#e94560", name="Actual win rate",
            text=cal["n"].apply(lambda x: f"n={x}"), textposition="outside"
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#2ecc71", dash="dash", width=2),
            name="Perfect calibration"
        ))
        dark_chart(fig2, 350)
        fig2.update_layout(
            xaxis=dict(title="Our predicted probability", range=[0, 1]),
            yaxis=dict(title="Actual win rate",           range=[0, 1]),
            legend=dict(bgcolor="#1a1a2e")
        )
        st.plotly_chart(fig2, width='stretch')

        # ── Feature ablation ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Which Features Are Helping?")
        st.markdown(
            "*Remove each feature group and see if accuracy goes up or down. "
            "Green = removing it hurts (it's useful). Red = removing it helps (it's noise).*"
        )

        current_round_bt = df[df["year"] == datetime.now().year]["round"].max()
        if isinstance(current_round_bt, float) and np.isnan(current_round_bt):
            current_round_bt = 0
        if int(current_round_bt) < 5:
            st.warning(
                f"⚠️ **Early-season note (Round {int(current_round_bt)}):** "
                "Season stats are sparse right now so they'll show as Neutral. "
                "Re-run after Round 5+ for meaningful results."
            )

        if st.button("▶ Run Feature Analysis (~30 seconds)", type="primary"):
            with st.spinner("Analysing feature groups..."):
                ablation_df = ablation_test(df, FEATURE_GROUPS, 3)
            st.session_state["bt_ablation"] = ablation_df

        ablation_df = st.session_state.get("bt_ablation")
        if ablation_df is not None and not ablation_df.empty:
            ablation_df["color"] = ablation_df["interpretation"].apply(
                lambda x: "#2ecc71" if "Helps" in str(x)
                else ("#e94560" if "Hurts" in str(x) else "#f39c12")
            )
            fig3 = go.Figure(go.Bar(
                x=ablation_df["delta"], y=ablation_df["group"],
                orientation="h",
                marker_color=ablation_df["color"],
                text=ablation_df["delta"].apply(
                    lambda x: f"{x:+.2f}%" if x != 0 else "baseline"
                ),
                textposition="outside"
            ))
            fig3.add_vline(x=0, line_color="white", line_width=1)
            dark_chart(fig3, 420)
            fig3.update_layout(
                title="Accuracy change when feature group removed  (negative = group helps)",
                xaxis=dict(title="Δ Accuracy %"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig3, width='stretch')

            # Summary callouts
            hurting = ablation_df[ablation_df["interpretation"].str.contains("Hurts", na=False)]
            if not hurting.empty:
                notes = [f"• **{r['group']}** is adding noise ({r['delta']:+.2f}%) — candidate for removal"
                         for _, r in hurting.iterrows()]
                st.info("**Noise detected:**\n" + "\n".join(notes))

        # ── Elo Anchor Tuner ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Elo Anchor Tuner")
        st.markdown(
            "*How much should Elo influence predictions vs the GBM? "
            "Sweep anchor weights 0%→100% to find the accuracy sweet spot. "
            "0% = pure GBM, 100% = pure Elo.*"
        )
        abl_min_train_elo = st.slider("Minimum training years", 2, 5, 3, key="elo_anchor_min_train")

        if st.button("▶ Run Elo Anchor Sweep", type="primary", key="run_elo_sweep"):
            with st.spinner("Sweeping Elo anchor weights 0%→100% — takes ~60 seconds..."):
                sweep_df = elo_anchor_sweep(df, win_model, margin_model, metrics,
                                            min_train_years=abl_min_train_elo)
            st.session_state["elo_sweep_result"] = sweep_df

        sweep_df = st.session_state.get("elo_sweep_result")
        if sweep_df is not None and not sweep_df.empty:
            best_row    = sweep_df.loc[sweep_df["accuracy"].idxmax()]
            best_anchor = best_row["elo_anchor"]
            best_acc    = best_row["accuracy"]

            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(mc(f"{best_anchor:.0%}", "Optimal Elo Anchor",
                                    f"Peak accuracy {best_acc:.1f}%"), unsafe_allow_html=True)
            with c2:
                v = sweep_df[sweep_df["elo_anchor"]==0.0]["accuracy"].values
                st.markdown(mc(f"{v[0]:.1f}%" if len(v) else "—", "Pure GBM (0% Elo)"), unsafe_allow_html=True)
            with c3:
                v = sweep_df[sweep_df["elo_anchor"]==1.0]["accuracy"].values
                st.markdown(mc(f"{v[0]:.1f}%" if len(v) else "—", "Pure Elo (100%)"), unsafe_allow_html=True)

            st.markdown("---")
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=sweep_df["elo_anchor"]*100, y=sweep_df["accuracy"],
                mode="lines+markers", line=dict(color="#e94560", width=2),
                marker=dict(size=8), name="Accuracy"
            ))
            fig_sweep.add_vline(x=best_anchor*100, line_dash="dot", line_color="#2ecc71",
                                annotation_text=f"  optimal: {best_anchor:.0%}",
                                annotation_font_color="#2ecc71")
            dark_chart(fig_sweep, 320)
            fig_sweep.update_layout(
                xaxis=dict(title="Elo Anchor Weight (%)", ticksuffix="%"),
                yaxis=dict(title="Out-of-Sample Accuracy (%)",
                           range=[sweep_df["accuracy"].min()-1, sweep_df["accuracy"].max()+1])
            )
            st.plotly_chart(fig_sweep, width='stretch')

            fig_brier = go.Figure()
            fig_brier.add_trace(go.Scatter(
                x=sweep_df["elo_anchor"]*100, y=sweep_df["brier_score"],
                mode="lines+markers", line=dict(color="#3498db", width=2),
                marker=dict(size=8), name="Brier Score"
            ))
            best_b = sweep_df.loc[sweep_df["brier_score"].idxmin()]
            fig_brier.add_vline(x=best_b["elo_anchor"]*100, line_dash="dot", line_color="#2ecc71",
                                annotation_text=f"  best calibration: {best_b['elo_anchor']:.0%}",
                                annotation_font_color="#2ecc71")
            dark_chart(fig_brier, 280)
            fig_brier.update_layout(
                xaxis=dict(title="Elo Anchor Weight (%)", ticksuffix="%"),
                yaxis=dict(title="Brier Score (lower = better calibrated)")
            )
            st.plotly_chart(fig_brier, width='stretch')

            disp = sweep_df.copy()
            disp["elo_anchor"] = disp["elo_anchor"].apply(lambda x: f"{x:.0%}")
            disp.columns = ["Elo Anchor", "Accuracy %", "Brier Score", "Games"]
            st.dataframe(disp, hide_index=True, width='stretch')
            st.caption(f"☝️ Paste to Claude: optimal Elo anchor = {best_anchor:.0%} "
                       f"({best_acc:.1f}% accuracy, Brier {best_row['brier_score']:.4f})")


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
            st.plotly_chart(fig_r, width='stretch')

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
            st.dataframe(pd.DataFrame(feat_rows), width='stretch', hide_index=True)

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
                width='stretch', hide_index=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# VALUE BETS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Value Bets":
    st.markdown("# 💰 VALUE BETS")
    st.markdown("*Find edges between our model's probabilities and bookmaker odds*")

    # ── Arbitrage Scanner ─────────────────────────────────────────────────────
    st.markdown("## ⚡ Arbitrage Scanner")
    st.markdown(
        "*An arbitrage opportunity exists when the best available odds across all "
        "bookmakers sum to less than 100% implied probability — meaning you can bet "
        "both sides and lock in a guaranteed profit regardless of the result.*"
    )

    _arb_key = ""
    try:
        _arb_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        pass

    if not _arb_key:
        st.info("Add your `ODDS_API_KEY` to Streamlit secrets to enable the arbitrage scanner.")
    else:
        with st.spinner("Scanning for arbitrage opportunities..."):
            _arb_odds_raw = get_odds_api(_arb_key)
            _arb_df = find_arbitrage(_arb_odds_raw) if not _arb_odds_raw.empty else pd.DataFrame()

        if _arb_odds_raw.empty:
            st.warning("⚠️ No odds data returned — bookmakers may not have posted lines for this round yet. Check back closer to game day.")
        elif _arb_df.empty:
            _n_games  = _arb_odds_raw.groupby(["home_team","away_team"]).ngroups
            _n_books  = _arb_odds_raw["bookmaker"].nunique()
            st.success(f"✅ No arbitrage opportunities ≥2.5% right now — scanned {_n_games} games across {_n_books} bookmakers.")
        else:
            st.warning(f"⚡ **{len(_arb_df)} arbitrage opportunit{'y' if len(_arb_df)==1 else 'ies'} found!**")
            for _, arb in _arb_df.iterrows():
                profit_colour = "#2ecc71" if arb["profit_pct"] >= 4.0 else "#f39c12"
                st.markdown(f"""
<div style="background:#0a1628;border:2px solid {profit_colour};border-radius:10px;
            padding:18px;margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div style="font-size:1.15rem;font-weight:700;color:white">
      {arb["home"]} vs {arb["away"]}
    </div>
    <div style="background:{profit_colour};color:#000;font-weight:700;padding:5px 12px;
                border-radius:6px;font-size:1rem">
      +{arb["profit_pct"]:.3f}% guaranteed profit
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
    <div style="background:#0f1f3d;border-radius:8px;padding:12px">
      <div style="color:#aaa;font-size:0.72rem;margin-bottom:4px">BET 1 — {arb["home"]}</div>
      <div style="font-size:1.4rem;font-weight:700;color:{profit_colour}">{arb["best_home_odds"]}</div>
      <div style="color:#aaa;font-size:0.8rem">{arb["best_home_book"]}</div>
      <div style="color:white;font-size:0.85rem;margin-top:6px">
        Stake <b>${arb["h_stake_per100"]:.2f}</b> per $100 outlay
      </div>
    </div>
    <div style="background:#0f1f3d;border-radius:8px;padding:12px">
      <div style="color:#aaa;font-size:0.72rem;margin-bottom:4px">BET 2 — {arb["away"]}</div>
      <div style="font-size:1.4rem;font-weight:700;color:{profit_colour}">{arb["best_away_odds"]}</div>
      <div style="color:#aaa;font-size:0.8rem">{arb["best_away_book"]}</div>
      <div style="color:white;font-size:0.85rem;margin-top:6px">
        Stake <b>${arb["a_stake_per100"]:.2f}</b> per $100 outlay
      </div>
    </div>
  </div>
  <div style="background:#0f1f3d;border-radius:8px;padding:10px;font-size:0.8rem;color:#aaa">
    Combined implied probability: <b style="color:white">{arb["arb_pct"]:.2f}%</b>
    &nbsp;·&nbsp; Market is <b style="color:{profit_colour}">{100 - arb["arb_pct"]:.2f}%</b> overround in your favour
    &nbsp;·&nbsp; Odds refreshed every 20 mins
  </div>
</div>
""", unsafe_allow_html=True)

            st.caption("⚠️ Act fast — arb opportunities close within minutes once bookmakers notice. Always verify odds before placing.")

    st.markdown("---")

    # ── Helper to get our model prob for any matchup ──────────────────────────
    def get_our_prob(ht, at, venue=""):
        """Use identical feature building to the Predict page."""
        _yr_df2 = df[df["year"] == datetime.now().year]
        _round = int(_yr_df2["round"].max()) + 1 if not _yr_df2.empty else 1
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
        min_edge = st.slider("Minimum edge to show (%)", 0, 15, 5, key="min_edge",
                              help="Only show bets where our model gives at least this much edge over the bookmaker's implied probability. 5%+ is a reasonable threshold for genuine value.")

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
                            "Game Time": g["Game Time"].iloc[0] if "Game Time" in g.columns else "",
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
                        }), include_groups=False)
                        .reset_index(drop=True))

                # ── Value bet cards ───────────────────────────────────────────
                best["best_edge"] = best[["H Edge%","A Edge%"]].max(axis=1)
                value = best[best["best_edge"] >= min_edge].sort_values("best_edge", ascending=False)

                if value.empty:
                    st.info(f"No value bets found with ≥{min_edge}% edge. Either the market is efficient this week or try a lower threshold.")
                else:
                    st.markdown(f"### 🎯 {len(value)} Value Bet{'s' if len(value)>1 else ''} Found")
                    st.markdown(f"*Our model gives ≥{min_edge}% edge over the bookmaker's implied probability on these games. Edge = our probability minus the bookie's fair probability.*")

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

                        _gt = str(g.get("Game Time", "") or "")
                        _gt_line = f"<div style='color:#888;font-size:0.75rem;margin-top:2px'>🕐 {_gt}</div>" if _gt else ""

                        # Win/loss scenarios
                        win_payout   = stake * val_odds          # total back if win
                        win_profit   = win_payout - stake        # net profit if win
                        loss_amount  = stake                     # lose stake if loss
                        exp_value    = val_our * win_profit - (1 - val_our) * loss_amount
                        break_even   = 1 / val_odds * 100        # implied prob to break even

                        opp_odds     = g["Best Away Odds"] if is_home_value else g["Best Home Odds"]
                        opp_bookie   = g["Best Away Bookie"] if is_home_value else g["Best Home Bookie"]
                        opp_implied  = 1 / opp_odds * 100 if opp_odds > 0 else 0

                        _time_html = (f"<div style='color:#888;font-size:0.75rem;margin-top:2px'>"
                                      f"&#128336; {_gt}</div>") if _gt else ""
                        card = (
                            f"<div style='background:{card_colour};border:1px solid #2ecc71;"
                            f"border-radius:10px;padding:18px;margin-bottom:14px'>"
                            f"<div style='display:flex;justify-content:space-between;"
                            f"align-items:flex-start;margin-bottom:14px'>"
                            f"<div><div>"
                            f"<span style='font-size:1.2rem;font-weight:700;color:white'>{ht}</span>"
                            f"<span style='color:#e94560;margin:0 10px;font-weight:700;"
                            f"letter-spacing:2px'>VS</span>"
                            f"<span style='font-size:1.2rem;font-weight:700;color:white'>{at}</span>"
                            f"</div>{_time_html}</div>"
                            f"<span style='background:#2ecc71;color:#000;font-size:0.75rem;"
                            f"font-weight:700;padding:4px 10px;border-radius:4px;"
                            f"white-space:nowrap'>{badge}</span>"
                            f"</div>"
                        ) + f"""

  <!-- Bet details row -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px">
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;letter-spacing:0.5px;margin-bottom:6px">BET ON</div>
      <div style="color:#2ecc71;font-weight:700;font-size:1.05rem">{val_team}</div>
      <div style="color:#555;font-size:0.68rem;margin-top:2px">{val_bookie}</div>
    </div>
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;letter-spacing:0.5px;margin-bottom:6px">ODDS</div>
      <div style="color:white;font-weight:700;font-size:1.3rem">${val_odds:.2f}</div>
      <div style="color:#555;font-size:0.68rem;margin-top:2px">bookie implies {break_even:.1f}%</div>
    </div>
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;letter-spacing:0.5px;margin-bottom:6px">OUR PROBABILITY</div>
      <div style="color:#3498db;font-weight:700;font-size:1.3rem">{val_our*100:.1f}%</div>
      <div style="color:#2ecc71;font-size:0.68rem;margin-top:2px">+{val_edge:.1f}% edge</div>
    </div>
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;letter-spacing:0.5px;margin-bottom:6px">KELLY STAKE</div>
      <div style="color:#f39c12;font-weight:700;font-size:1.3rem">${stake:.2f}</div>
      <div style="color:#555;font-size:0.68rem;margin-top:2px">{kelly_pct:.1f}% of ${bankroll:,}</div>
    </div>
  </div>

  <!-- Divider -->
  <div style="border-top:1px solid #1a2a4a;margin:10px 0"></div>

  <!-- Payout calculator -->
  <div style="margin-bottom:8px">
    <div style="color:#aaa;font-size:0.7rem;letter-spacing:0.5px;margin-bottom:8px">PAYOUT CALCULATOR — staking ${stake:.2f}</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
      <div style="background:#0d2a0d;border:1px solid #2ecc71;border-radius:8px;padding:12px;text-align:center">
        <div style="color:#2ecc71;font-size:0.68rem;margin-bottom:4px">✅ IF WIN</div>
        <div style="color:#2ecc71;font-weight:700;font-size:1.4rem">+${win_profit:.2f}</div>
        <div style="color:#aaa;font-size:0.7rem;margin-top:2px">return ${win_payout:.2f} total</div>
      </div>
      <div style="background:#2a0d0d;border:1px solid #e74c3c;border-radius:8px;padding:12px;text-align:center">
        <div style="color:#e74c3c;font-size:0.68rem;margin-bottom:4px">❌ IF LOSS</div>
        <div style="color:#e74c3c;font-weight:700;font-size:1.4rem">-${loss_amount:.2f}</div>
        <div style="color:#aaa;font-size:0.7rem;margin-top:2px">stake forfeited</div>
      </div>
      <div style="background:#0a1628;border:1px solid #3498db;border-radius:8px;padding:12px;text-align:center">
        <div style="color:#3498db;font-size:0.68rem;margin-bottom:4px">📊 EXPECTED VALUE</div>
        <div style="color:#3498db;font-weight:700;font-size:1.4rem">{"+" if exp_value >= 0 else "-"}${abs(exp_value):.2f}</div>
        <div style="color:#aaa;font-size:0.7rem;margin-top:2px">per bet, long run</div>
      </div>
    </div>
  </div>

  <!-- Footer info -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
    <div style="background:#0a1628;border-radius:6px;padding:8px 12px">
      <span style="color:#aaa;font-size:0.68rem">Model tips: </span>
      <span style="color:white;font-size:0.8rem;font-weight:600">{winner_str} by ~{pred_margin_str}</span>
    </div>
    <div style="background:#0a1628;border-radius:6px;padding:8px 12px">
      <span style="color:#aaa;font-size:0.68rem">Opposing best odds: </span>
      <span style="color:white;font-size:0.8rem;font-weight:600">${opp_odds:.2f} ({opp_bookie})</span>
      <span style="color:#555;font-size:0.68rem"> · implies {opp_implied:.1f}%</span>
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
                                 width='stretch', hide_index=True)

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
                                .style.map(_cdiff, subset=["Difference"]),
                                width='stretch', hide_index=True)
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
                                    width='stretch', hide_index=True)
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

                # ── Auto Best Multi ───────────────────────────────────────────
                st.markdown("### 🏆 Recommended Multi")
                st.markdown("*Top 3 highest-confidence picks this round — strong favourites with the best available odds*")

                # Get all upcoming games — fetch fresh since we're on Value Bets page
                try:
                    import requests as _req_m
                    _r_m = _req_m.get(
                        f"https://api.squiggle.com.au/?q=games;year={datetime.now().year}",
                        headers={"User-Agent": "AFL-Predictor/1.0"}, timeout=10
                    )
                    _all_games_m = pd.DataFrame(_r_m.json().get("games", []))
                    _upcoming_m  = _all_games_m[_all_games_m["complete"] < 100].copy() if not _all_games_m.empty else pd.DataFrame()
                    if not _upcoming_m.empty:
                        _next_rnd_m = int(_upcoming_m["round"].min())
                        _upcoming_m = _upcoming_m[_upcoming_m["round"] == _next_rnd_m]
                except Exception:
                    _upcoming_m = pd.DataFrame()

                _all_preds = []
                for _, _mg in _upcoming_m.iterrows():
                    _mh = normalise_team(str(_mg.get("hteam","")))
                    _ma = normalise_team(str(_mg.get("ateam","")))
                    _mv = str(_mg.get("venue",""))
                    if _mh not in current_elos or _ma not in current_elos:
                        continue
                    try:
                        _mf = _build_prediction_features(
                            _mh, _ma, _mv, current_elos, team_stats,
                            season_stats, lineup_strength, df, exp_df, standings_df,
                            style_df=style_df,
                            current_round=int(selected_round) if selected_round else None
                        )
                        _mp = predict_game(win_model, margin_model, _mf, metrics["features_used"])
                        _mconf = max(_mp["home_win_prob"], _mp["away_win_prob"]) / 100
                        _mteam = _mh if _mp["home_win_prob"] > _mp["away_win_prob"] else _ma
                        _mmargin = abs(_mp["predicted_margin"])
                        _all_preds.append({
                            "home": _mh, "away": _ma, "venue": _mv,
                            "tip": _mteam, "conf": _mconf, "margin": _mmargin,
                        })
                    except Exception:
                        continue

                # Sort by confidence, take top 3
                _top3 = sorted(_all_preds, key=lambda x: -x["conf"])[:3]

                if len(_top3) >= 2:
                    # Look up best available odds for each tip from odds_df if available
                    _odds_key_m = ""
                    try:
                        _odds_key_m = st.secrets.get("ODDS_API_KEY", "")
                    except Exception:
                        pass

                    _odds_lookup = {}
                    if _odds_key_m:
                        try:
                            _round_odds = get_odds_api(_odds_key_m)
                            if not _round_odds.empty:
                                for _, _or in _round_odds.iterrows():
                                    _oh = normalise_team(_or["home_team"])
                                    _oa = normalise_team(_or["away_team"])
                                    _key = f"{_oh}|{_oa}"
                                    if _key not in _odds_lookup:
                                        _odds_lookup[_key] = {"home_odds": {}, "away_odds": {}}
                                    _odds_lookup[_key]["home_odds"][_or["bookmaker"]] = float(_or["home_odds"])
                                    _odds_lookup[_key]["away_odds"][_or["bookmaker"]] = float(_or["away_odds"])
                        except Exception:
                            pass

                    _multi_odds = 1.0
                    _multi_prob = 1.0
                    _leg_lines  = ""
                    _multi_stake = 100.0  # fixed $100 stake for recommended multi

                    for _leg in _top3:
                        _lkey = f"{_leg['home']}|{_leg['away']}"
                        _is_home = _leg["tip"] == _leg["home"]
                        _odds_side = "home_odds" if _is_home else "away_odds"

                        # Get best odds from bookmakers if available, else estimate from Elo
                        if _lkey in _odds_lookup and _odds_lookup[_lkey][_odds_side]:
                            _best_bookie = max(_odds_lookup[_lkey][_odds_side], key=_odds_lookup[_lkey][_odds_side].get)
                            _best_leg_odds = _odds_lookup[_lkey][_odds_side][_best_bookie]
                        else:
                            # Estimate fair odds from our probability (no vig)
                            _best_leg_odds = round(1 / max(_leg["conf"], 0.01), 2)
                            _best_bookie = "est."

                        _multi_odds *= _best_leg_odds
                        _multi_prob *= _leg["conf"]

                        _conf_colour = "#2ecc71" if _leg["conf"] >= 0.70 else "#f39c12"
                        _leg_lines += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;background:#0a1628;border-radius:8px;margin-bottom:6px">
  <div>
    <div style="color:white;font-weight:700;font-size:0.95rem">{_leg["tip"]}</div>
    <div style="color:#555;font-size:0.72rem">{_leg["home"]} vs {_leg["away"]}</div>
  </div>
  <div style="text-align:center">
    <div style="color:{_conf_colour};font-weight:700;font-size:1rem">{_leg["conf"]*100:.1f}%</div>
    <div style="color:#555;font-size:0.68rem">confidence</div>
  </div>
  <div style="text-align:center">
    <div style="color:#aaa;font-size:0.72rem">predicted by</div>
    <div style="color:#aaa;font-size:0.72rem">{_leg["margin"]:.0f} pts</div>
  </div>
  <div style="text-align:right">
    <div style="color:#f39c12;font-weight:700;font-size:1.1rem">${_best_leg_odds:.2f}</div>
    <div style="color:#555;font-size:0.68rem">{_best_bookie}</div>
  </div>
</div>"""

                    _multi_return  = _multi_stake * _multi_odds
                    _multi_profit  = _multi_return - _multi_stake

                    st.markdown(f"""
<div style="background:#0f2a1a;border:2px solid #2ecc71;border-radius:12px;padding:18px;margin-bottom:16px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div style="font-size:1.1rem;font-weight:700;color:white">{len(_top3)}-Leg Favourites Multi</div>
    <div style="color:#2ecc71;font-size:0.8rem">{_multi_prob*100:.1f}% combined probability</div>
  </div>
  {_leg_lines}
  <div style="border-top:1px solid #1a3a2a;margin:12px 0"></div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;margin-bottom:4px">COMBINED ODDS</div>
      <div style="color:#f39c12;font-weight:700;font-size:1.4rem">${_multi_odds:.2f}</div>
    </div>
    <div style="background:#0a1628;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#aaa;font-size:0.65rem;margin-bottom:4px">STAKE</div>
      <div style="color:white;font-weight:700;font-size:1.4rem">${_multi_stake:.0f}</div>
    </div>
    <div style="background:#0d2a0d;border:1px solid #2ecc71;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#2ecc71;font-size:0.65rem;margin-bottom:4px">✅ IF WIN</div>
      <div style="color:#2ecc71;font-weight:700;font-size:1.4rem">+${_multi_profit:.2f}</div>
      <div style="color:#aaa;font-size:0.7rem">return ${_multi_return:.2f}</div>
    </div>
    <div style="background:#2a0d0d;border:1px solid #e74c3c;border-radius:8px;padding:12px;text-align:center">
      <div style="color:#e74c3c;font-size:0.65rem;margin-bottom:4px">❌ IF LOSS</div>
      <div style="color:#e74c3c;font-weight:700;font-size:1.4rem">-${_multi_stake:.0f}</div>
      <div style="color:#aaa;font-size:0.7rem">stake forfeited</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
                    st.caption("⚠️ Multis are high variance — even strong favourites lose. Never stake more than you can afford to lose.")
                else:
                    st.info("Not enough upcoming games to build a multi yet — check back closer to game day.")

                st.markdown("---")
                st.markdown("### 🎰 Custom Multi Builder")
                st.markdown("*Manually select legs from this round's value bets*")

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
    st.markdown("*A visual guide to how win probability and margin predictions are calculated*")
    st.markdown("---")
    st.markdown("## The Prediction Pipeline")

    _n_games   = metrics.get("n_games", 0)
    _n_feats   = metrics.get("n_features", 0)
    _acc       = metrics.get("win_accuracy", 0) * 100
    _elo_min   = int(min(current_elos.values())) if current_elos else 1300
    _elo_max   = int(max(current_elos.values())) if current_elos else 1700

    st.markdown(f"""
<style>
.pw {{ font-family:sans-serif; padding:8px 0 }}
.ps {{ color:#888; font-size:0.68rem; letter-spacing:1.5px; text-transform:uppercase; margin:16px 0 6px }}
.pr {{ display:flex; align-items:stretch; gap:0; margin-bottom:4px }}
.pb {{ background:#0a1628; border-radius:8px; padding:10px 12px; flex:1; text-align:center }}
.pt {{ font-size:0.78rem; font-weight:700; color:white; margin-bottom:3px }}
.pd {{ font-size:0.65rem; color:#888; line-height:1.4 }}
.pa {{ display:flex; align-items:center; padding:0 4px; color:#444; font-size:1.2rem }}
.pad {{ text-align:center; color:#444; font-size:1.4rem; margin:3px 0 }}
.pm {{ background:#16213e; border:1px solid #e94560; border-radius:10px; padding:14px 16px; text-align:center; margin:4px 0 }}
.pmt {{ font-size:1rem; font-weight:700; color:#e94560; margin-bottom:4px }}
.pms {{ font-size:0.72rem; color:#aaa }}
.po {{ display:flex; gap:12px; margin-top:4px }}
.pob {{ flex:1; border-radius:8px; padding:12px; text-align:center }}
.pov {{ font-size:1.4rem; font-weight:900; margin-bottom:2px }}
.pol {{ font-size:0.7rem; color:#aaa }}
.bdg {{ display:inline-block; font-size:0.58rem; padding:1px 5px; border-radius:3px; margin-left:4px; vertical-align:middle; font-weight:700; background:#222; color:#888 }}
</style>

<div class="pw">

  <div class="ps">① Data Sources</div>
  <div class="pr">
    <div class="pb" style="border-top:3px solid #3498db">
      <div class="pt">🏉 Squiggle API</div>
      <div class="pd">Game results · Ladder · PAV player ratings · Announced lineups</div>
    </div>
    <div class="pa">→</div>
    <div class="pb" style="border-top:3px solid #9b59b6">
      <div class="pt">📊 AFL Tables</div>
      <div class="pd">Season stats · Clearances · Inside 50s · Tackles · Hitouts</div>
    </div>
    <div class="pa">→</div>
    <div class="pb" style="border-top:3px solid #e67e22">
      <div class="pt">📍 Venue &amp; Odds</div>
      <div class="pd">Ground locations · Travel distances · Bookmaker odds (TAB, Sportsbet)</div>
    </div>
  </div>

  <div class="pad">↓</div>

  <div class="ps">② Feature Engineering — what gets fed to the model</div>
  <div class="pr">
    <div class="pb" style="border-top:3px solid #e94560">
      <div class="pt">⚡ Elo Rating</div>
      <div class="pd">Chess-style team rating · Updated every game · +50pt home advantage<br>
      <span style="color:#666;font-size:0.62rem">Current range: {_elo_min}–{_elo_max} pts</span></div>
    </div>
    <div class="pa"> </div>
    <div class="pb" style="border-top:3px solid #f39c12">
      <div class="pt">📈 Form <span class="bdg">fades in R3+</span></div>
      <div class="pd">Last 5 avg margin · Win streak · Last game margin · Consistency score</div>
    </div>
    <div class="pa"> </div>
    <div class="pb" style="border-top:3px solid #2ecc71">
      <div class="pt">🪜 Ladder <span class="bdg">fades in R6+</span></div>
      <div class="pd">Rank diff · Percentage diff · Wins diff</div>
    </div>
  </div>
  <div style="height:5px"></div>
  <div class="pr">
    <div class="pb" style="border-top:3px solid #9b59b6">
      <div class="pt">📉 Season Stats <span class="bdg">from R3+</span></div>
      <div class="pd">Clearances · Inside 50s · Contested poss · Tackles · Hitouts</div>
    </div>
    <div class="pa"> </div>
    <div class="pb" style="border-top:3px solid #1abc9c">
      <div class="pt">🎨 Playing Style</div>
      <div class="pd">Kick ratio · Ruck dominance · Kick-vs-tackle matchup</div>
    </div>
    <div class="pa"> </div>
    <div class="pb" style="border-top:3px solid #e91e8c">
      <div class="pt">⭐ PAV Lineups <span class="bdg">Thu+</span></div>
      <div class="pd">Selected 22 PAV total · Offence · Defence · Midfield split</div>
    </div>
  </div>

  <div class="pad">↓</div>

  <div style="background:#0a1628;border:1px dashed #555;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.72rem;color:#888">
    <b style="color:#f39c12">⚠️ Early season blend:</b>&nbsp;
    R1–2 = 80% Elo / 20% model &nbsp;·&nbsp;
    R3–5 = 50/50 &nbsp;·&nbsp;
    R6–9 = 25% Elo / 75% model &nbsp;·&nbsp;
    R10+ = 100% model
    <span style="margin-left:8px;color:#555">— prevents one game of data overriding a large Elo gap</span>
  </div>

  <div class="ps">③ Gradient Boosting Model (GBM)</div>
  <div class="pm">
    <div class="pmt">🤖 Gradient Boosting Machine</div>
    <div class="pms">
      200 decision trees &nbsp;·&nbsp; {_n_games:,} training games ({start_year}–present) &nbsp;·&nbsp; {_n_feats} features &nbsp;·&nbsp; <b style="color:#2ecc71">{_acc:.1f}% cross-validated accuracy</b><br>
      Each tree corrects mistakes of the last — the ensemble learns which feature combinations predict wins
    </div>
  </div>

  <div class="pad">↓</div>

  <div class="ps">④ Outputs</div>
  <div class="po">
    <div class="pob" style="background:#1a0a14;border:1px solid #e94560">
      <div class="pov" style="color:#e94560">68%</div>
      <div class="pol">WIN PROBABILITY<br>chance home team wins</div>
    </div>
    <div class="pob" style="background:#0a1620;border:1px solid #3498db">
      <div class="pov" style="color:#3498db">+14 pts</div>
      <div class="pol">PREDICTED MARGIN<br>separate regression model</div>
    </div>
    <div class="pob" style="background:#0a1620;border:1px solid #2ecc71">
      <div class="pov" style="color:#2ecc71">+7%</div>
      <div class="pol">EDGE VS BOOKMAKER<br>when odds are available</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

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

    # ── Player Experience Feature explainer ───────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎓 Player Experience Features")
    st.markdown("""
    The model now includes **team experience differential** as a feature group. The intuition: 
    a team fielding 18 players with 150+ games each is likely to handle pressure situations 
    better than a team averaging 60 games.

    **How weighted games work:**

    | Career Stage | Games Range | Notes |
    |---|---|---|
    | Developing | 0–24 | Unproven at elite level |
    | Emerging | 25–74 | Building consistency |
    | Prime | 75–149 | Peak athleticism + experience |
    | Veteran | 150–199 | High experience, declining athleticism |
    | Elite Veteran | 200+ | Hard-won wisdom |

    Finals games are counted at **×2.5** — a player who plays 10 finals has effectively 
    experienced the equivalent of 25 regular-season pressure games. Grand Finals, Preliminary 
    Finals, and Elimination Finals all count equally in this implementation.

    **The three model features:**
    - `exp_avg_diff`: Home team's average career games minus away team's average. Positive = home team more experienced.
    - `exp_veteran_diff`: Difference in % of veterans (150+ weighted games). High = more battle-hardened side.
    - `exp_developing_diff`: Difference in % of developing players (< 25 games). Negative value is better for home team.

    **Data source & accuracy note:**

    Career game counts are estimated from Squiggle's PAV (Player Approximate Value) data rather than 
    scraped directly from AFL Tables. Since PAV doesn't include a games-played column, we derive career 
    games from cumulative `PAV_total` using a calibration constant of ~0.17 PAV per game.

    This is an approximation — exact career game counts would require scraping ~450 individual player 
    pages from AFL Tables (~2-3 min load time). The proxy gives sensible-looking results and is 
    sufficient for a feature that currently shows as marginal in ablation testing. If ablation after 
    Round 6+ shows experience as a strong signal (>0.5% accuracy improvement), we'll switch to exact 
    AFL Tables counts. If it remains neutral or negative, the feature will be dropped entirely.
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
                                          "pct_veterans", "Career Stage Mix"]].copy()
                display_exp.columns = ["Team", "Avg Career Games", "Median Career Games",
                                       "% Veterans", "Career Mix"]
                display_exp["Avg Career Games"] = display_exp["Avg Career Games"].round(1)
                display_exp["Median Career Games"] = display_exp["Median Career Games"].round(1)
                display_exp["% Veterans"] = (display_exp["% Veterans"] * 100).round(1)
                st.dataframe(display_exp, width='stretch', hide_index=True)
            else:
                st.info("Experience data will populate once PAV data is loaded.")
        except Exception as _e:
            st.warning(f"Experience table error: {_e}")
