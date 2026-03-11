import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data.fetcher import get_all_games, get_upcoming_games, enrich_games, get_team_current_stats
from model.elo import build_elo_ratings, regress_elos_to_mean
from model.predictor import build_features, train_models, predict_game, build_prediction_features, FEATURES

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
    padding: 20px; text-align: center; color: white;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-card .label { font-size: 0.82rem; color: #aaa; margin-top: 4px; }
.metric-card .sub { font-size: 0.75rem; color: #2ecc71; margin-top: 4px; }
.team-vs {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 16px; padding: 24px; margin: 12px 0;
    border: 1px solid #e9456033;
}
[data-testid="stSidebar"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏉 AFL Predictor")
    st.markdown("---")
    start_year = st.slider("Training data from", 2010, 2022, 2013)
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict a Game",
        "📈 Team Form",
        "🏆 Elo Ladder",
        "🔬 Feature Importance",
    ])
    st.markdown("---")
    st.markdown("<small style='color:#666'>Data: Squiggle API<br>Model: Gradient Boosting + Elo</small>", unsafe_allow_html=True)

# ── Data pipeline (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching & enriching game data...")
def load_enriched(start_year: int):
    df = get_all_games(start_year)
    if df.empty:
        return None, None, None, None, {}, {}
    df = enrich_games(df)
    df, elo_history = build_elo_ratings(df)
    df = build_features(df)
    win_model, margin_model, metrics, feat_importance = train_models(df)
    current_elos = regress_elos_to_mean(elo_history)
    team_stats = get_team_current_stats(df)
    return df, win_model, margin_model, metrics, current_elos, team_stats, feat_importance

result = load_enriched(start_year)
if len(result) == 7:
    df, win_model, margin_model, metrics, current_elos, team_stats, feat_importance = result
else:
    st.error("Data load failed.")
    st.stop()

if df is None:
    st.error("Could not load game data.")
    st.stop()

teams = sorted(current_elos.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_team_form(team, n=15):
    home = df[df["hteam"]==team][["date_parsed","round","year","hteam","ateam","hscore","ascore","venue"]].copy()
    home["margin"] = home["hscore"] - home["ascore"]
    home["opponent"] = home["ateam"]
    home["venue_type"] = "Home"

    away = df[df["ateam"]==team][["date_parsed","round","year","hteam","ateam","hscore","ascore","venue"]].copy()
    away["margin"] = away["ascore"] - away["hscore"]
    away["opponent"] = away["hteam"]
    away["venue_type"] = "Away"

    combined = pd.concat([home, away]).sort_values("date_parsed").tail(n).reset_index(drop=True)
    combined["result"] = combined["margin"].apply(lambda x: "W" if x>0 else ("L" if x<0 else "D"))
    combined["game_label"] = combined.apply(lambda r: f"R{r['round']} {r['year']} vs {r['opponent']}", axis=1)
    return combined

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# AFL MATCH PREDICTOR")
    st.markdown(f"*{metrics['n_games']:,} games · {metrics['n_features']} features · {start_year}–present*")

    col1, col2, col3, col4 = st.columns(4)
    gain = metrics.get("accuracy_gain", 0)
    gain_str = f"+{gain*100:.1f}% vs base model" if gain > 0 else ""

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['win_accuracy']*100:.1f}%</div>
            <div class="label">Win Prediction Accuracy</div>
            <div class="sub">{gain_str}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['margin_r2']:.3f}</div>
            <div class="label">Margin R²</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['n_games']:,}</div>
            <div class="label">Training Games</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['n_features']}</div>
            <div class="label">Model Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Upcoming games
    st.markdown("## UPCOMING GAMES")
    try:
        upcoming = get_upcoming_games()
        if upcoming.empty:
            st.info("No upcoming games right now — check back closer to game day.")
        else:
            for _, game in upcoming.iterrows():
                home, away = game.get("hteam","?"), game.get("ateam","?")
                venue = game.get("venue","")
                if home not in current_elos or away not in current_elos:
                    continue

                feat = build_prediction_features(home, away, venue, current_elos, team_stats)
                pred = predict_game(win_model, margin_model, feat)

                winner = home if pred["home_win_prob"] > 50 else away
                margin = abs(pred["predicted_margin"])
                h_travel = feat["travel_home_km"]
                a_travel = feat["travel_away_km"]
                h_rest = feat["days_rest_home"]
                a_rest = feat["days_rest_away"]

                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown(f"""<div class="team-vs">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                            <div>
                                <span style="font-size:1.3rem;font-weight:700;color:white">{home}</span>
                                <div style="color:#aaa;font-size:0.75rem;margin-top:2px">
                                    ✈️ {h_travel:.0f}km · 💤 {h_rest}d rest · 
                                    {'🔥 ' + str(abs(team_stats.get(home,{}).get('streak',0))) + ' win streak' if team_stats.get(home,{}).get('streak',0) > 1 else ''}
                                </div>
                            </div>
                            <span style="color:#e94560;font-family:'Bebas Neue';font-size:1.2rem">VS</span>
                            <div style="text-align:right">
                                <span style="font-size:1.3rem;font-weight:700;color:white">{away}</span>
                                <div style="color:#aaa;font-size:0.75rem;margin-top:2px">
                                    ✈️ {a_travel:.0f}km · 💤 {a_rest}d rest ·
                                    {'🔥 ' + str(abs(team_stats.get(away,{}).get('streak',0))) + ' win streak' if team_stats.get(away,{}).get('streak',0) > 1 else ''}
                                </div>
                            </div>
                        </div>
                        <div style="height:10px;border-radius:5px;background:#0f3460;overflow:hidden">
                            <div style="width:{pred['home_win_prob']}%;height:100%;background:linear-gradient(90deg,#e94560,#ff6b6b);border-radius:5px"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;color:#aaa;font-size:0.78rem;margin-top:4px">
                            <span>{pred['home_win_prob']}%</span>
                            <span style="color:#666">{venue}</span>
                            <span>{pred['away_win_prob']}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card" style="margin-top:12px">
                        <div class="value" style="font-size:1.2rem">{winner}</div>
                        <div class="label">by ~{margin:.0f} pts</div>
                    </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load upcoming games: {e}")

    # Accuracy by year
    st.markdown("---")
    st.markdown("## MODEL ACCURACY BY YEAR")
    avail_feats = [f for f in FEATURES if f in df.columns]
    yearly = []
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"]==year].dropna(subset=avail_feats+["home_win"])
        if len(ydf) < 10: continue
        preds = win_model.predict(ydf[avail_feats].values)
        acc = (preds == ydf["home_win"].values).mean()
        yearly.append({"year": year, "accuracy": acc*100})
    if yearly:
        acc_df = pd.DataFrame(yearly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=acc_df["year"], y=acc_df["accuracy"],
            mode="lines+markers", line=dict(color="#e94560", width=3),
            marker=dict(size=8), fill="tozeroy", fillcolor="rgba(233,69,96,0.1)"))
        fig.add_hline(y=50, line_dash="dash", line_color="#666", annotation_text="50% baseline")
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"), height=300,
            yaxis=dict(title="Accuracy %", range=[40,85], gridcolor="#0f3460"),
            xaxis=dict(gridcolor="#0f3460"), margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT A GAME
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Game":
    st.markdown("# PREDICT A GAME")

    col1, col2, col3 = st.columns(3)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_options)
    with col3:
        venues = sorted(set(df["venue"].dropna().unique()))
        venue = st.selectbox("📍 Venue", ["(Auto)"] + venues)
        if venue == "(Auto)":
            venue = ""

    if st.button("🔮 Predict", use_container_width=True):
        feat = build_prediction_features(home_team, away_team, venue, current_elos, team_stats)
        pred = predict_game(win_model, margin_model, feat)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{pred['home_win_prob']}%</div>
                <div class="label">{home_team} win probability</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            m = pred["predicted_margin"]
            winner = home_team if m > 0 else away_team
            st.markdown(f"""<div class="metric-card">
                <div class="value">{abs(m):.0f} pts</div>
                <div class="label">Predicted margin ({winner})</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{pred['away_win_prob']}%</div>
                <div class="label">{away_team} win probability</div>
            </div>""", unsafe_allow_html=True)

        # Prob bar
        fig = go.Figure(go.Bar(
            x=[pred["home_win_prob"], pred["away_win_prob"]],
            y=[home_team, away_team], orientation="h",
            marker=dict(color=["#e94560","#0f3460"]),
            text=[f"{pred['home_win_prob']}%", f"{pred['away_win_prob']}%"],
            textposition="inside"
        ))
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"), height=180,
            xaxis=dict(range=[0,100], gridcolor="#0f3460"),
            yaxis=dict(gridcolor="#0f3460"), margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Supporting stats
        hs = team_stats.get(home_team, {})
        as_ = team_stats.get(away_team, {})
        with st.expander("📊 Full factor breakdown"):
            st.markdown(f"""
| Factor | {home_team} | {away_team} |
|--------|------------|------------|
| Elo Rating | {current_elos.get(home_team,1500):.0f} | {current_elos.get(away_team,1500):.0f} |
| Current Streak | {hs.get('streak',0):+d} | {as_.get('streak',0):+d} |
| Last Game Margin | {hs.get('last_margin',0):+.0f} | {as_.get('last_margin',0):+.0f} |
| Avg Margin (last 5) | {hs.get('last5_avg',0):+.1f} | {as_.get('last5_avg',0):+.1f} |
| Travel to Venue | {feat['travel_home_km']:.0f} km | {feat['travel_away_km']:.0f} km |
| Days Rest | {feat['days_rest_home']} | {feat['days_rest_away']} |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM FORM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Team Form":
    st.markdown("# TEAM FORM ANALYSIS")
    selected = st.selectbox("Select Team", teams)
    n = st.slider("Last N games", 5, 30, 15)
    form = get_team_form(selected, n)

    if form.empty:
        st.warning("No data found.")
    else:
        hs = team_stats.get(selected, {})
        wins = (form["result"]=="W").sum()
        losses = (form["result"]=="L").sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{wins}W {losses}L</div><div class="label">Last {n} Games</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{form['margin'].mean():+.1f}</div><div class="label">Avg Margin</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{hs.get('streak',0):+d}</div><div class="label">Current Streak</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{current_elos.get(selected,1500):.0f}</div><div class="label">Elo Rating</div>
            </div>""", unsafe_allow_html=True)

        colors = ["#2ecc71" if m>0 else "#e94560" for m in form["margin"]]
        fig = go.Figure(go.Bar(x=form["game_label"], y=form["margin"],
            marker_color=colors, text=form["result"], textposition="outside"))
        fig.add_hline(y=0, line_color="white", line_width=1)
        fig.update_layout(title=f"{selected} — Game Margins",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font=dict(color="white"),
            xaxis=dict(tickangle=-45, gridcolor="#0f3460"),
            yaxis=dict(gridcolor="#0f3460"), height=400,
            margin=dict(l=20,r=20,t=40,b=100))
        st.plotly_chart(fig, use_container_width=True)

        form["cumulative"] = form["margin"].cumsum()
        fig2 = go.Figure(go.Scatter(x=form["game_label"], y=form["cumulative"],
            mode="lines+markers", line=dict(color="#e94560", width=2),
            fill="tozeroy", fillcolor="rgba(233,69,96,0.1)"))
        fig2.update_layout(title="Cumulative Margin",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font=dict(color="white"),
            xaxis=dict(tickangle=-45, gridcolor="#0f3460"),
            yaxis=dict(gridcolor="#0f3460"), height=300,
            margin=dict(l=20,r=20,t=40,b=100))
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ELO LADDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Elo Ladder":
    st.markdown("# ELO LADDER")
    st.markdown("*Rolling measure of true team strength, adjusted for opponent quality*")

    elo_df = pd.DataFrame([{
        "Rank": i+1,
        "Team": team,
        "Elo": round(elo),
        "Streak": team_stats.get(team,{}).get("streak",0),
        "Avg Margin (L5)": team_stats.get(team,{}).get("last5_avg",0),
    } for i, (team, elo) in enumerate(sorted(current_elos.items(), key=lambda x: -x[1]))])

    fig = go.Figure(go.Bar(
        x=elo_df["Elo"], y=elo_df["Team"], orientation="h",
        marker=dict(color=elo_df["Elo"], colorscale=[[0,"#0f3460"],[0.5,"#e94560"],[1,"#ff6b6b"]]),
        text=elo_df["Elo"], textposition="inside"
    ))
    fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="white"), height=600,
        xaxis=dict(range=[1300,1700], gridcolor="#0f3460"),
        yaxis=dict(gridcolor="#0f3460", autorange="reversed"),
        margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(elo_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Importance":
    st.markdown("# FEATURE IMPORTANCE")
    st.markdown("*How much each factor contributes to prediction accuracy*")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['win_accuracy']*100:.1f}%</div>
            <div class="label">Full Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['base_accuracy']*100:.1f}%</div>
            <div class="label">Base Model (Elo + Form only)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        gain = metrics.get("accuracy_gain", 0)
        color = "#2ecc71" if gain > 0 else "#e94560"
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:{color}">{gain*100:+.1f}%</div>
            <div class="label">Accuracy Gain from New Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What's actually moving the needle?")

    # Feature importance chart
    fi = feat_importance.copy()
    fi["feature_label"] = fi["feature"].str.replace("_", " ").str.title()
    fi["color"] = fi["feature"].apply(
        lambda f: "#e94560" if f in ["elo_diff","form_diff","home_form","away_form","home_consistency","away_consistency"]
        else ("#f39c12" if "travel" in f or "rest" in f
        else ("#2ecc71" if "streak" in f or "margin" in f else "#3498db"))
    )

    fig = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature_label"],
        orientation="h", marker_color=fi["color"],
        text=fi["importance"].apply(lambda x: f"{x:.3f}"),
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="white"), height=500,
        xaxis=dict(title="Importance Score", gridcolor="#0f3460"),
        yaxis=dict(gridcolor="#0f3460", autorange="reversed"),
        margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px">
        <span style="color:#e94560">■ Elo / Base form</span>
        <span style="color:#f39c12">■ Travel & Rest (fatigue)</span>
        <span style="color:#2ecc71">■ Streak & Last margin</span>
        <span style="color:#3498db">■ Other</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Raw feature importance table")
    st.dataframe(fi[["feature_label","importance"]].rename(columns={"feature_label":"Feature","importance":"Importance"}),
                 use_container_width=True, hide_index=True)
