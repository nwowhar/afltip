import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import get_all_games, get_upcoming_games, enrich_games, get_team_current_stats
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
    start_year = st.slider("Training data from", 2010, 2020, 2013)
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict a Game",
        "📈 Team Form",
        "🏆 Elo Ladder",
        "🔬 Feature Importance",
        "📉 Backtest",
        "👕 Lineup Strength",
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

@st.cache_data(ttl=3600, show_spinner="🤖 Building Elo ratings & training model...")
def build_model(start_year):
    games_df = load_games(start_year)
    if games_df is None or games_df.empty:
        return None, None, None, None, {}, {}, None, None

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
    st.markdown("## UPCOMING GAMES")

    try:
        upcoming = get_upcoming_games()
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

                feats = build_prediction_features(
                    home, away, venue,
                    current_elos, team_stats,
                    season_stats, lineup_strength
                )
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

                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""<div class="team-vs">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                            <div>
                                <div style="font-size:1.2rem;font-weight:700;color:white">{home}</div>
                                <div style="color:#aaa;font-size:0.72rem">
                                    ✈️ {feats['travel_home_km']:.0f}km &nbsp;|&nbsp; 💤 {feats['days_rest_home']}d rest
                                    {'&nbsp;|&nbsp; 🔥 ' + str(abs(hs.get('streak',0))) + 'W streak' if hs.get('streak',0) > 1 else ''}
                                </div>
                            </div>
                            <span style="color:#e94560;font-family:'Bebas Neue';font-size:1.1rem">VS</span>
                            <div style="text-align:right">
                                <div style="font-size:1.2rem;font-weight:700;color:white">{away}</div>
                                <div style="color:#aaa;font-size:0.72rem">
                                    ✈️ {feats['travel_away_km']:.0f}km &nbsp;|&nbsp; 💤 {feats['days_rest_away']}d rest
                                    {'&nbsp;|&nbsp; 🔥 ' + str(abs(as_.get('streak',0))) + 'W streak' if as_.get('streak',0) > 1 else ''}
                                </div>
                            </div>
                        </div>
                        <div style="height:10px;border-radius:5px;background:#0f3460;overflow:hidden">
                            <div style="width:{pred['home_win_prob']}%;height:100%;background:linear-gradient(90deg,#e94560,#ff6b6b);border-radius:5px"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;color:#aaa;font-size:0.75rem;margin-top:5px">
                            <span>{pred['home_win_prob']}%</span>
                            <span style="color:#555">{venue} {pav_note}</span>
                            <span>{pred['away_win_prob']}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(mc(winner, f"by ~{margin:.0f} pts"), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load upcoming games: {e}")

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

    if st.button("🔮 PREDICT", use_container_width=True):
        feats = build_prediction_features(
            home_team, away_team, venue,
            current_elos, team_stats,
            season_stats, lineup_strength
        )
        pred = predict_game(win_model, margin_model, feats)
        m    = pred["predicted_margin"]
        winner = home_team if m > 0 else away_team

        st.markdown("---")
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

        # Full breakdown
        hs  = team_stats.get(home_team, {})
        as_ = team_stats.get(away_team, {})
        pav_row = ""
        if feats.get("lineup_available"):
            pav_row = f"| PAV Rating (selected 22) | {feats.get('home_pav_total',0):.0f} | {feats.get('away_pav_total',0):.0f} |"

        with st.expander("📊 Full factor breakdown"):
            st.markdown(f"""
| Factor | {home_team} | {away_team} |
|--------|------------|------------|
| Elo Rating | {current_elos.get(home_team,1500):.0f} | {current_elos.get(away_team,1500):.0f} |
| Avg Margin (last 5) | {hs.get('last5_avg',0):+.1f} | {as_.get('last5_avg',0):+.1f} |
| Current Streak | {hs.get('streak',0):+d} | {as_.get('streak',0):+d} |
| Last Game Margin | {hs.get('last_margin',0):+.0f} | {as_.get('last_margin',0):+.0f} |
| Travel to Venue | {feats['travel_home_km']:.0f} km | {feats['travel_away_km']:.0f} km |
| Days Rest | {feats['days_rest_home']} | {feats['days_rest_away']} |
| Clearances (season avg) | {feats['cl_diff']/2+35:.1f} | {35-feats['cl_diff']/2:.1f} |
| Inside 50s (season avg) | {feats['i50_diff']/2+50:.1f} | {50-feats['i50_diff']/2:.1f} |
{pav_row}
""")

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
# FEATURE IMPORTANCE
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
    st.markdown("*Player Approximate Value — rates each player's contribution to their team. Updates when lineups are announced.*")

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
                st.markdown(f"**{len(lineup_strength)} teams with lineup data available**")

                # Summary table
                rows = []
                for team, data in sorted(lineup_strength.items(),
                                          key=lambda x: -x[1].get("PAV_total", 0)):
                    rows.append({
                        "Team":        team,
                        "PAV Total":   round(data.get("PAV_total", 0), 1),
                        "PAV Off":     round(data.get("PAV_off",   0), 1),
                        "PAV Mid":     round(data.get("PAV_mid",   0), 1),
                        "PAV Def":     round(data.get("PAV_def",   0), 1),
                        "Players Matched": data.get("n_players_matched", 0),
                    })
                ls_df = pd.DataFrame(rows)

                # Bar chart
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
                    title="Team Lineup Strength by Component (PAV)",
                    xaxis=dict(tickangle=-45),
                    legend=dict(bgcolor="#1a1a2e")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ls_df, use_container_width=True, hide_index=True)

    # PAV top players
    st.markdown("---")
    st.markdown("### Top Rated Players (Career PAV)")
    if not pav_df.empty:
        pav_show = pav_df.copy()
        for col in ["PAV_total", "PAV_off", "PAV_def", "PAV_mid"]:
            if col in pav_show.columns:
                pav_show[col] = pd.to_numeric(pav_show[col], errors="coerce")

        top_year = pav_show["year"].max() if "year" in pav_show.columns else 2024
        top = pav_show[pav_show["year"] == top_year].copy()

        if "PAV_total" in top.columns:
            top = top.sort_values("PAV_total", ascending=False).head(30)
            display_cols = [c for c in
                ["firstname", "surname", "team", "PAV_total",
                 "PAV_off", "PAV_mid", "PAV_def", "games"]
                if c in top.columns]
            st.dataframe(top[display_cols].reset_index(drop=True),
                         use_container_width=True)
    else:
        st.info("PAV data not available.")
