# AFL Match Predictor

A Streamlit dashboard that predicts AFL game outcomes using Elo ratings and machine learning.

## Features
- 🔮 **Predict any game** — win probability + expected margin
- 📊 **Dashboard** — upcoming games with predictions, model accuracy over time
- 📈 **Team form** — last N game margins, cumulative form chart
- 🏆 **Elo ladder** — teams ranked by true strength (Elo rating)

## How it works
1. **Data** — pulls historical game results from the [Squiggle API](https://api.squiggle.com.au) (free, no key needed)
2. **Elo ratings** — rolling team strength ratings updated after every game, with home ground advantage built in
3. **ML model** — Gradient Boosting classifier (win/loss) and regressor (margin) trained on Elo diff + rolling form features
4. **Dashboard** — built with Streamlit + Plotly

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — done!

## Model performance
Typically achieves ~65–68% win prediction accuracy (5-fold CV), which comfortably beats the 50% baseline. Margin R² is lower (~0.15–0.25) as margins are inherently harder to predict.

## Data source
[Squiggle API](https://api.squiggle.com.au) — free, community-built AFL data API.
