"""
Golf IQ — BACKEND
All data access and model logic lives here, separated from the UI (app.py).
The frontend never touches the model directly; it calls these functions.

Public functions:
    get_players()                      -> list of player names
    get_history(name)                  -> DataFrame of that player's seasons
    summary(name)                      -> dict of headline numbers
    predict_whatif(stats: dict)        -> dict {point, low, high}
    forecast_next_season(name)         -> dict {expected, low, high, ...}
    model_forecast(name)              -> dict {point, low, high}  (stat-based, advanced)
    feature_fields()                   -> dict of slider config for the UI
    record_visit(session_id)           -> log one visit (call once per browser session)
    record_player_view(session_id, name) -> log a player lookup
    visitor_stats()                    -> dict of visit counts + most-viewed players
    visits_by_month()                  -> DataFrame [Month, Visits], one row per month
"""
import json
import functools
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import xgboost as xgb

DATA_FILE = "pgatour_2023_2025.csv"
VISITOR_DB = "visitors.db"   # created on first run; gitignored

# Slider/UI config: feature -> (friendly label, min, max, step)
_FIELDS = {
    "ROUNDS":         ("Rounds played",              40,   130,  1.0),
    "SCORING":        ("Scoring average",            67.0, 73.0, 0.1),
    "DRIVE_DISTANCE": ("Driving distance (yd)",      270.0, 330.0, 0.5),
    "FWY_%":          ("Fairways hit (%)",           40.0, 80.0, 0.5),
    "GIR_%":          ("Greens in regulation (%)",   55.0, 80.0, 0.5),
    "SG_P":           ("Strokes gained: putting",    -1.5, 1.5,  0.05),
    "SG_TTG":         ("Strokes gained: tee-to-green", -2.0, 3.0, 0.05),
    "POINTS":         ("FedEx Cup points",           0,    7500, 10.0),
    "TOP 10":         ("Top-10 finishes",            0,    20,   1.0),
    "1ST":            ("Wins",                       0,    8,    1.0),
}

Z80 = 1.28  # z-score for an ~80% interval


# ---------------------------------------------------------------- loaders
@functools.lru_cache(maxsize=1)
def _data():
    return pd.read_csv(DATA_FILE).drop(columns=["COUNTRY"], errors="ignore")

@functools.lru_cache(maxsize=1)
def _whatif():
    m = xgb.XGBRegressor(); m.load_model("model.json")
    return m, json.load(open("model_meta.json"))

@functools.lru_cache(maxsize=1)
def _forecast_model():
    m = xgb.XGBRegressor(); m.load_model("forecast_model.json")
    return m, json.load(open("forecast_meta.json"))

@functools.lru_cache(maxsize=1)
def _persistence():
    return json.load(open("forecast_persistence_meta.json"))

def features():
    return _whatif()[1]["features"]

def feature_fields():
    return _FIELDS


# ---------------------------------------------------------------- queries
def get_players():
    return sorted(_data()["NAME"].unique().tolist())

def get_history(name):
    return _data()[_data().NAME == name].sort_values("Year").reset_index(drop=True)

def latest_season(name):
    return get_history(name).iloc[-1]

def summary(name):
    h = get_history(name)
    last = h.iloc[-1]
    return {
        "seasons": int(len(h)),
        "latest_year": int(last["Year"]),
        "latest_money": float(last["MONEY"]),
        "best_money": float(h["MONEY"].max()),
        "career_wins": int(h["1ST"].sum()),
        "avg_scoring": float(h["SCORING"].mean()),
    }

def all_max_money():
    return float(_data()["MONEY"].max())


# ---------------------------------------------------------------- predictions
def predict_whatif(stats: dict):
    """Descriptive model: given a season's stats, expected earnings (R^2 0.87)."""
    model, meta = _whatif()
    F = meta["features"]
    x = pd.DataFrame([[float(stats[f]) for f in F]], columns=F)
    lp = float(model.predict(x)[0])
    s = meta["log_resid_std"]
    return {
        "point": float(np.expm1(lp)),
        "low":   float(np.expm1(lp - Z80 * s)),
        "high":  float(np.expm1(lp + Z80 * s)),
    }

def forecast_next_season(name):
    """Persistence-anchored forecast (the most reliable estimator on this data)."""
    p = _persistence()
    anchor = float(latest_season(name)["MONEY"])
    s = p["log_ratio_std"]
    return {
        "expected": anchor,
        "low":  float(anchor * np.exp(-Z80 * s)),
        "high": float(anchor * np.exp(Z80 * s)),
        "source_year": int(p["source_year"]),
        "target_year": int(p["target_year"]),
    }

def model_forecast(name):
    """Advanced/secondary: stat-based ML forecast (weaker than persistence — shown for transparency)."""
    model, meta = _forecast_model()
    F = meta["features"]
    row = latest_season(name)
    x = pd.DataFrame([[float(row[f]) for f in F]], columns=F)
    lp = float(model.predict(x)[0])
    s = meta["log_resid_std"]
    return {
        "point": float(np.expm1(lp)),
        "low":  float(np.expm1(lp - Z80 * s)),
        "high": float(np.expm1(lp + Z80 * s)),
    }


# ---------------------------------------------------------------- visitors
# Lightweight, self-contained traffic counter. A "visit" is one Streamlit
# browser session (a page refresh starts a new one). No IPs or user agents
# are stored — only an opaque session id, timestamps, and player lookups.
_VISITOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS player_views (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    player     TEXT NOT NULL,
    viewed_at  TEXT NOT NULL
);
"""

def _visitor_db():
    con = sqlite3.connect(VISITOR_DB, timeout=5)
    con.executescript(_VISITOR_SCHEMA)
    return con

def _utcnow():
    return datetime.now(timezone.utc)

def record_visit(session_id: str):
    """Log a visit. Idempotent per session_id, so safe to call on every rerun."""
    with _visitor_db() as con:
        con.execute("INSERT OR IGNORE INTO visits VALUES (?, ?)",
                    (session_id, _utcnow().isoformat()))

def record_player_view(session_id: str, name: str):
    """Log that a session looked at a player."""
    with _visitor_db() as con:
        con.execute("INSERT INTO player_views (session_id, player, viewed_at) VALUES (?, ?, ?)",
                    (session_id, name, _utcnow().isoformat()))

def visitor_stats(top_n: int = 3):
    """Headline traffic numbers for the UI (all times UTC)."""
    now = _utcnow()
    today = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")
    week_ago = (now - timedelta(days=7)).isoformat()
    with _visitor_db() as con:
        total = con.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        today_n = con.execute("SELECT COUNT(*) FROM visits WHERE started_at LIKE ?",
                              (today + "%",)).fetchone()[0]
        week_n = con.execute("SELECT COUNT(*) FROM visits WHERE started_at >= ?",
                             (week_ago,)).fetchone()[0]
        month_n = con.execute("SELECT COUNT(*) FROM visits WHERE started_at LIKE ?",
                              (month + "%",)).fetchone()[0]
        top = con.execute(
            "SELECT player, COUNT(DISTINCT session_id) AS n FROM player_views "
            "GROUP BY player ORDER BY n DESC, player LIMIT ?", (top_n,)).fetchall()
    return {
        "total_visits": int(total),
        "visits_today": int(today_n),
        "visits_7d": int(week_n),
        "visits_this_month": int(month_n),
        "top_players": [{"player": str(p), "sessions": int(n)} for p, n in top],
    }

def visits_by_month():
    """Visits per calendar month (UTC) as a DataFrame with columns Month ('YYYY-MM') and Visits."""
    with _visitor_db() as con:
        rows = con.execute(
            "SELECT substr(started_at, 1, 7) AS month, COUNT(*) FROM visits "
            "GROUP BY month ORDER BY month").fetchall()
    return pd.DataFrame([{"Month": str(m), "Visits": int(n)} for m, n in rows],
                        columns=["Month", "Visits"])
