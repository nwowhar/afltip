import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data.fetcher import get_all_games, get_upcoming_games
from model.elo import build_elo_ratings, win_probability_from_elo, regress_elos_to_mean
from model.predictor import build_features, train_models, predict_game, FEATURES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFL Predictor",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1px; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #e94560; }
.metric-card .label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

.team-vs {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border: 1px solid #e9456033;
}
.win-bar-container { height: 12px; border-radius: 6px; background: #0f3460; overflow: hidden; margin: 8px 0; }
.win-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #e94560, #ff6b6b); }

[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] .css-1d391kg { color: white; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching game data from Squiggle...")
def load_data(start_year: int = 2010):
    df = get_all_games(start_year)
    return df

@st.cache_data(ttl=3600, show_spinner="Building Elo ratings & training model...")
def prepare_model(start_year: int = 2010):
    df = load_data(start_year)
    if df.empty:
        return None, None, None, None, {}

    df, elo_history = build_elo_ratings(df)
    df = build_features(df)
    win_model, margin_model, metrics = train_models(df)

    # Regress Elos for new season predictions
    current_elos = regress_elos_to_mean(elo_history)

    return df, win_model, margin_model, metrics, current_elos

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏉 AFL Predictor")
    st.markdown("---")
    start_year = st.slider("Training data from", 2010, 2022, 2013)
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Predict a Game", "📈 Team Form", "🏆 Elo Ladder"])
    st.markdown("---")
    st.markdown("<small style='color:#666'>Data: Squiggle API<br>Model: Gradient Boosting</small>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    df, win_model, margin_model, metrics, current_elos = prepare_model(start_year)

if df is None or df.empty:
    st.error("Could not load game data. Check your internet connection.")
    st.stop()

teams = sorted(current_elos.keys())

# ── Helper: team rolling form ─────────────────────────────────────────────────
def get_team_form(team: str, n: int = 10) -> pd.DataFrame:
    home = df[df["hteam"] == team][["year", "round", "hteam", "ateam", "hscore", "ascore"]].copy()
    home["team"] = team
    home["opponent"] = home["ateam"]
    home["score"] = home["hscore"]
    home["opp_score"] = home["ascore"]
    home["venue"] = "Home"

    away = df[df["ateam"] == team][["year", "round", "hteam", "ateam", "hscore", "ascore"]].copy()
    away["team"] = team
    away["opponent"] = away["hteam"]
    away["score"] = away["ascore"]
    away["opp_score"] = away["hscore"]
    away["venue"] = "Away"

    combined = pd.concat([home, away]).sort_values(["year", "round"]).tail(n).reset_index(drop=True)
    combined["margin"] = combined["score"] - combined["opp_score"]
    combined["result"] = combined["margin"].apply(lambda x: "W" if x > 0 else ("L" if x < 0 else "D"))
    combined["game_label"] = combined.apply(lambda r: f"R{r['round']} {r['year']} vs {r['opponent']}", axis=1)
    return combined

def get_team_avg_form(team: str, window: int = 5) -> tuple:
    form = get_team_form(team, 20)
    recent = form.tail(window)
    avg_margin = recent["margin"].mean() if not recent.empty else 0
    std_margin = recent["margin"].std() if len(recent) > 1 else 20
    return round(avg_margin, 1), round(std_margin, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# AFL MATCH PREDICTOR")
    st.markdown(f"*Trained on {metrics['n_games']:,} games ({start_year}–present)*")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['win_accuracy']*100:.1f}%</div>
            <div class="label">Win Prediction Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['margin_r2']:.2f}</div>
            <div class="label">Margin R² Score</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{metrics['n_games']:,}</div>
            <div class="label">Games in Training Set</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        current_year = datetime.now().year
        year_games = df[df["year"] == current_year]
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(year_games)}</div>
            <div class="label">{current_year} Games Analysed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Upcoming games
    st.markdown("## UPCOMING GAMES")
    try:
        upcoming = get_upcoming_games()
        if upcoming.empty:
            st.info("No upcoming games found right now — check back closer to game day.")
        else:
            for _, game in upcoming.iterrows():
                home = game.get("hteam", "?")
                away = game.get("ateam", "?")
                if home not in current_elos or away not in current_elos:
                    continue
                h_elo = current_elos[home]
                a_elo = current_elos[away]
                h_form, h_std = get_team_avg_form(home)
                a_form, a_std = get_team_avg_form(away)
                pred = predict_game(win_model, margin_model, h_elo, a_elo, h_form, a_form, h_std, a_std)

                winner = home if pred["home_win_prob"] > 50 else away
                win_pct = max(pred["home_win_prob"], pred["away_win_prob"])
                margin = abs(pred["predicted_margin"])

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""<div class="team-vs">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:1.3rem;font-weight:700;color:white">{home}</span>
                            <span style="color:#e94560;font-family:'Bebas Neue';font-size:1.1rem">VS</span>
                            <span style="font-size:1.3rem;font-weight:700;color:white">{away}</span>
                        </div>
                        <div class="win-bar-container">
                            <div class="win-bar-fill" style="width:{pred['home_win_prob']}%"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;color:#aaa;font-size:0.8rem">
                            <span>{pred['home_win_prob']}%</span>
                            <span>{pred['away_win_prob']}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card" style="margin-top:12px">
                        <div class="value" style="font-size:1.4rem">{winner}</div>
                        <div class="label">Predicted winner</div>
                        <div style="color:#e94560;margin-top:8px;font-weight:600">by ~{margin:.0f} pts</div>
                    </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load upcoming games: {e}")

    # Model accuracy over time
    st.markdown("---")
    st.markdown("## MODEL ACCURACY BY YEAR")
    yearly_acc = []
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year].dropna(subset=FEATURES + ["home_win"])
        if len(ydf) < 10:
            continue
        X = ydf[FEATURES].values
        y = ydf["home_win"].values
        preds = win_model.predict(X)
        acc = (preds == y).mean()
        yearly_acc.append({"year": year, "accuracy": acc * 100})

    if yearly_acc:
        acc_df = pd.DataFrame(yearly_acc)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acc_df["year"], y=acc_df["accuracy"],
            mode="lines+markers",
            line=dict(color="#e94560", width=3),
            marker=dict(size=8, color="#e94560"),
            fill="tozeroy",
            fillcolor="rgba(233,69,96,0.1)"
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#666", annotation_text="50% baseline")
        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"),
            yaxis=dict(title="Accuracy %", range=[40, 80], gridcolor="#0f3460"),
            xaxis=dict(title="Year", gridcolor="#0f3460"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT A GAME
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Game":
    st.markdown("# PREDICT A GAME")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_options, index=min(1, len(away_options)-1))

    if st.button("🔮 Predict", use_container_width=True):
        h_elo = current_elos.get(home_team, 1500)
        a_elo = current_elos.get(away_team, 1500)
        h_form, h_std = get_team_avg_form(home_team)
        a_form, a_std = get_team_avg_form(away_team)

        pred = predict_game(win_model, margin_model, h_elo, a_elo, h_form, a_form, h_std, a_std)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{pred['home_win_prob']}%</div>
                <div class="label">{home_team} Win Probability</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            margin = pred["predicted_margin"]
            winner = home_team if margin > 0 else away_team
            st.markdown(f"""<div class="metric-card">
                <div class="value">{abs(margin):.0f} pts</div>
                <div class="label">Predicted Margin ({winner})</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{pred['away_win_prob']}%</div>
                <div class="label">{away_team} Win Probability</div>
            </div>""", unsafe_allow_html=True)

        # Win probability gauge
        fig = go.Figure(go.Bar(
            x=[pred["home_win_prob"], pred["away_win_prob"]],
            y=[home_team, away_team],
            orientation="h",
            marker=dict(color=["#e94560", "#0f3460"]),
            text=[f"{pred['home_win_prob']}%", f"{pred['away_win_prob']}%"],
            textposition="inside"
        ))
        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"),
            xaxis=dict(range=[0, 100], gridcolor="#0f3460"),
            yaxis=dict(gridcolor="#0f3460"),
            height=200, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show supporting stats
        with st.expander("📊 Supporting Stats"):
            st.markdown(f"""
            | Stat | {home_team} | {away_team} |
            |------|------------|------------|
            | Elo Rating | {h_elo:.0f} | {a_elo:.0f} |
            | Avg Margin (last 5) | {h_form:+.1f} | {a_form:+.1f} |
            | Form Consistency (σ) | {h_std:.1f} | {a_std:.1f} |
            """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TEAM FORM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Team Form":
    st.markdown("# TEAM FORM ANALYSIS")

    selected_team = st.selectbox("Select Team", teams)
    n_games = st.slider("Last N games", 5, 30, 15)

    form = get_team_form(selected_team, n_games)

    if form.empty:
        st.warning("No data found for this team.")
    else:
        # Win/loss record
        wins = (form["result"] == "W").sum()
        losses = (form["result"] == "L").sum()
        avg_margin = form["margin"].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{wins}W {losses}L</div>
                <div class="label">Last {n_games} Games</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            color = "🟢" if avg_margin > 0 else "🔴"
            st.markdown(f"""<div class="metric-card">
                <div class="value">{avg_margin:+.1f}</div>
                <div class="label">Average Margin</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            elo = current_elos.get(selected_team, 1500)
            st.markdown(f"""<div class="metric-card">
                <div class="value">{elo:.0f}</div>
                <div class="label">Current Elo Rating</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Margin chart
        colors = ["#2ecc71" if m > 0 else "#e94560" for m in form["margin"]]
        fig = go.Figure(go.Bar(
            x=form["game_label"],
            y=form["margin"],
            marker_color=colors,
            text=form["result"],
            textposition="outside"
        ))
        fig.add_hline(y=0, line_color="white", line_width=1)
        fig.update_layout(
            title=f"{selected_team} — Game Margins",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"),
            xaxis=dict(tickangle=-45, gridcolor="#0f3460"),
            yaxis=dict(title="Margin (pts)", gridcolor="#0f3460"),
            height=400, margin=dict(l=20, r=20, t=40, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative form
        form["cumulative"] = form["margin"].cumsum()
        fig2 = go.Figure(go.Scatter(
            x=form["game_label"],
            y=form["cumulative"],
            mode="lines+markers",
            line=dict(color="#e94560", width=2),
            fill="tozeroy",
            fillcolor="rgba(233,69,96,0.1)"
        ))
        fig2.update_layout(
            title="Cumulative Margin",
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font=dict(color="white"),
            xaxis=dict(tickangle=-45, gridcolor="#0f3460"),
            yaxis=dict(gridcolor="#0f3460"),
            height=300, margin=dict(l=20, r=20, t=40, b=100)
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ELO LADDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Elo Ladder":
    st.markdown("# ELO LADDER")
    st.markdown("*Teams ranked by current Elo rating — a rolling measure of true team strength*")

    elo_df = pd.DataFrame([
        {"Team": team, "Elo": round(elo), "Form (avg margin, last 5)": get_team_avg_form(team)[0]}
        for team, elo in sorted(current_elos.items(), key=lambda x: -x[1])
    ])
    elo_df["Rank"] = range(1, len(elo_df) + 1)
    elo_df = elo_df[["Rank", "Team", "Elo", "Form (avg margin, last 5)"]]

    fig = go.Figure(go.Bar(
        x=elo_df["Elo"],
        y=elo_df["Team"],
        orientation="h",
        marker=dict(
            color=elo_df["Elo"],
            colorscale=[[0, "#0f3460"], [0.5, "#e94560"], [1, "#ff6b6b"]]
        ),
        text=elo_df["Elo"],
        textposition="inside"
    ))
    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#0f3460", range=[1300, 1700]),
        yaxis=dict(gridcolor="#0f3460", autorange="reversed"),
        height=600, margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(elo_df, use_container_width=True, hide_index=True)
