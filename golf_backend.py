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
"""
import json
import functools
import numpy as np
import pandas as pd
import xgboost as xgb

DATA_FILE = "pgatour_2023_2025.csv"

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
