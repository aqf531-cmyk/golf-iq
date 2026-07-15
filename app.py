"""
Golf IQ — FRONTEND (Streamlit UI)
All model/data logic lives in golf_backend.py; this file is purely presentation.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import golf_backend as be

# ----------------------------------------------------------------------
# Page + styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Golf IQ", page_icon="⛳", layout="wide")

GREEN, GREEN_D, INK = "#2E7D46", "#1F5732", "#1A2B22"

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1100px; }}
.hero {{
  background: linear-gradient(120deg, {GREEN_D} 0%, {GREEN} 100%);
  border-radius: 16px; padding: 26px 30px; color: #fff; margin-bottom: 18px;
}}
.hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing:-0.5px; }}
.hero p  {{ margin: 6px 0 0; opacity: .92; font-size: 1.02rem; }}
div[data-testid="stMetric"] {{
  background: #F2F6F3; border: 1px solid #E1EAE4; border-radius: 12px;
  padding: 14px 16px;
}}
div[data-testid="stMetricLabel"] p {{ color:#5b6b62; font-weight:600; }}
.bigcard {{
  background:#F2F6F3; border:1px solid #E1EAE4; border-left:6px solid {GREEN};
  border-radius:12px; padding:20px 24px; margin: 6px 0 4px;
}}
.bigcard .lab {{ color:#5b6b62; font-weight:700; font-size:.9rem; text-transform:uppercase; letter-spacing:.04em; }}
.bigcard .val {{ color:{GREEN_D}; font-size:2.3rem; font-weight:800; line-height:1.1; margin:2px 0; }}
.bigcard .rng {{ color:#41524a; font-size:1rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
</style>
""", unsafe_allow_html=True)


def money(v):
    return f"${v/1e6:.1f}M" if v >= 1e6 else f"${v:,.0f}"

def big_card(label, value, sub=""):
    st.markdown(f"""<div class="bigcard"><div class="lab">{label}</div>
    <div class="val">{value}</div><div class="rng">{sub}</div></div>""",
    unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
players = be.get_players()
with st.sidebar:
    st.markdown("### ⛳ Golf IQ")
    st.caption("PGA Tour performance & earnings predictor")
    default_i = players.index("Scottie Scheffler") if "Scottie Scheffler" in players else 0
    player = st.selectbox("Choose a player", players, index=default_i)
    st.divider()
    st.caption("Built on an XGBoost model trained on 2023–2025 PGA Tour data. "
               "Predictions are estimates, not financial advice.")

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(f"""<div class="hero"><h1>⛳ Golf IQ</h1>
<p>Explore a player's history and project what next season could look like.</p></div>""",
unsafe_allow_html=True)

s = be.summary(player)
st.markdown(f"## {player}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Seasons on record", s["seasons"])
m2.metric(f"{s['latest_year']} earnings", money(s["latest_money"]))
m3.metric("Best season", money(s["best_money"]))
m4.metric("Career wins (in data)", s["career_wins"])

hist = be.get_history(player)

tab_over, tab_fc, tab_wi, tab_cmp = st.tabs(
    ["📈 Overview", "🔮 Forecast", "🎛️ What-if", "📊 Compare"])

# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------
with tab_over:
    st.subheader("Earnings by season")
    base = alt.Chart(hist).encode(
        x=alt.X("Year:O", title=None),
        y=alt.Y("MONEY:Q", title="Earnings ($)", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("MONEY:Q", format="$,.0f", title="Earnings")])
    bars = base.mark_bar(color=GREEN, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    st.altair_chart(bars, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Scoring average")
        line = alt.Chart(hist).mark_line(point=True, color=GREEN_D).encode(
            x=alt.X("Year:O", title=None),
            y=alt.Y("SCORING:Q", scale=alt.Scale(zero=False), title="Strokes/round"),
            tooltip=["Year", "SCORING"])
        st.altair_chart(line, use_container_width=True)
    with c2:
        st.subheader("Strokes gained: tee-to-green")
        line2 = alt.Chart(hist).mark_line(point=True, color="#C77D2E").encode(
            x=alt.X("Year:O", title=None),
            y=alt.Y("SG_TTG:Q", scale=alt.Scale(zero=False), title="SG: TTG"),
            tooltip=["Year", "SG_TTG"])
        st.altair_chart(line2, use_container_width=True)

    with st.expander("See all season-by-season stats"):
        st.dataframe(hist.set_index("Year"), use_container_width=True)

# ----------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------
with tab_fc:
    f = be.forecast_next_season(player)
    st.subheader(f"Projected {f['target_year']} earnings")
    big_card(f"Expected {f['target_year']} earnings", money(f["expected"]),
             f"80% likely range: {money(f['low'])} – {money(f['high'])}")

    st.info(
        "**How this works (in plain terms).** The single most reliable predictor of a "
        "golfer's earnings next season is what they earned *this* season — so this forecast "
        "anchors on the most recent year. We tested a fancier machine-learning model that "
        "uses detailed stats; on a fair forward test it was actually **worse** than this "
        "simple anchor (because golf earnings swing wildly year to year). The wide range is "
        "honest: it reflects real volatility in the sport.")

    with st.expander("Advanced: what the stat-based ML model says (for transparency)"):
        mf = be.model_forecast(player)
        st.write(f"ML stat-model estimate: **{money(mf['point'])}**  "
                 f"(range {money(mf['low'])} – {money(mf['high'])})")
        st.caption("Shown for comparison only. This model underperformed the simple anchor "
                   "above on a true forward test (R² 0.03 vs 0.27), so treat it with caution.")

# ----------------------------------------------------------------------
# What-if
# ----------------------------------------------------------------------
with tab_wi:
    st.subheader("What-if explorer")
    st.caption("Drag the sliders to a performance scenario, then see the earnings that level "
               "of play has historically produced. Sliders start at this player's most recent "
               "season. (Powered by the descriptive model, R² 0.87 on same-season stats.)")

    fields = be.feature_fields()
    last = be.latest_season(player)
    inputs, cols = {}, st.columns(2)
    for i, feat in enumerate(be.features()):
        label, lo, hi, step = fields[feat]
        default = float(np.clip(last[feat], lo, hi))
        inputs[feat] = cols[i % 2].slider(label, float(lo), float(hi), default, step=step)

    if st.button("Estimate earnings", type="primary", use_container_width=True):
        r = be.predict_whatif(inputs)
        big_card("Earnings for this performance", money(r["point"]),
                 f"Likely range (~80%): {money(r['low'])} – {money(r['high'])}")
        st.progress(min(r["point"] / be.all_max_money(), 1.0))
    st.caption("⚠️ This answers “if a player posts these numbers, what do they earn?” — "
               "it is not a standalone year-ahead forecast.")

# ----------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------
with tab_cmp:
    st.subheader("Compare players")
    picks = st.multiselect("Pick 2–4 players", players,
                           default=[p for p in [player, "Rory McIlroy"] if p in players][:2],
                           max_selections=4)
    if len(picks) >= 2:
        rows = []
        for name in picks:
            sm = be.summary(name)
            fc = be.forecast_next_season(name)
            rows.append({"Player": name, "Latest earnings": sm["latest_money"],
                         f"Forecast {fc['target_year']}": fc["expected"]})
        cmp = pd.DataFrame(rows)
        melt = cmp.melt("Player", var_name="Metric", value_name="Earnings")
        chart = alt.Chart(melt).mark_bar().encode(
            x=alt.X("Earnings:Q", title="Earnings ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("Player:N", title=None, sort="-x"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[GREEN, "#C77D2E"])),
            yOffset="Metric:N",
            tooltip=["Player", "Metric", alt.Tooltip("Earnings:Q", format="$,.0f")])
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(cmp.set_index("Player").style.format("${:,.0f}"),
                     use_container_width=True)
    else:
        st.info("Select at least two players to compare.")
