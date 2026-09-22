# Golf IQ

Streamlit app that explores PGA Tour player history (2023–2025) and projects next-season
earnings. Two Python files, three pre-trained estimators, one CSV, plus a small SQLite
visitor counter. No tests yet.

## Commands

```bash
# One-time setup (uv is installed via Homebrew; system python3 is 3.9 and unusable here)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
brew install libomp                    # macOS only: xgboost needs the OpenMP runtime

# Every session
source .venv/bin/activate

# Run — MUST be from the repo root (all data/model paths are relative)
streamlit run app.py

# Headless smoke test of the backend (no UI)
python -c "import golf_backend as be; print(be.forecast_next_season('Scottie Scheffler')); print(be.model_forecast('Scottie Scheffler')); print(be.visitor_stats())"
```

Environment: Python 3.12 in `.venv/` (gitignored), deps pinned in `requirements.txt`.
Never use `/usr/bin/python3` — it's 3.9 with no packages. If `import xgboost` fails with
"libomp.dylib could not be loaded", run `brew install libomp`.

## Architecture — keep the layers separate

- `app.py` — **presentation only**. Streamlit layout, Altair charts, CSS. Calls `be.*`
  functions and formats results. Never load the CSV or a model here.
- `golf_backend.py` — **all data access and model logic**. Public API is listed in its
  module docstring; add new capabilities there first, then wire them into the UI.
- Loaders (`_data`, `_whatif`, `_forecast_model`, `_persistence`) are `functools.lru_cache`d.
  Editing a data/model file on disk requires restarting Streamlit to take effect.
- **Visitor counter** (`record_visit`, `record_player_view`, `visitor_stats`) writes to
  `visitors.db` (SQLite, gitignored, created on first run). A "visit" = one Streamlit
  browser session — `app.py` mints a `uuid4` into `st.session_state.visitor_id` and logs it
  once; a page refresh is a new visit. Only session id, UTC timestamps, and player lookups
  are stored — no IPs/user agents. `_visitor_db()` opens a fresh connection per call
  (deliberately **not** cached). On hosts with an ephemeral filesystem (e.g. Streamlit
  Community Cloud) the counter resets on redeploy.

### The three estimators (don't confuse them — the UI copy depends on this)

| Function | Artifacts | What it answers | Quality |
|---|---|---|---|
| `predict_whatif(stats)` | `model.json` + `model_meta.json` | "If a player posts these same-season stats, what do they earn?" | R² ≈ 0.87 (descriptive, not a forecast) |
| `forecast_next_season(name)` | `forecast_persistence_meta.json` | Year-ahead earnings, anchored on last season's `MONEY` | R² ≈ 0.27 — **the primary forecast shown** |
| `model_forecast(name)` | `forecast_model.json` + `forecast_meta.json` | Year-ahead earnings from last season's stats | R² ≈ 0.03 — shown only in an "advanced" expander |

- Both XGBoost models predict **`log1p(MONEY)`**; always `np.expm1` the output.
- Intervals are ~80% (`Z80 = 1.28`) using `log_resid_std` / `log_ratio_std` from the meta files.
- **Training code is not in this repo.** The `.json` models are opaque artifacts (XGBoost
  3.2.0 format). Feature order at predict time must match `meta["features"]` exactly.
  Don't "retrain" or regenerate them without the user supplying the training pipeline.
- The R² figures quoted in `app.py`'s info/caption text come from that missing training
  run — if models are ever replaced, update that copy too.

## Data: `pgatour_2023_2025.csv`

- 375 rows, 201 unique players. Columns: `NAME, ROUNDS, SCORING, DRIVE_DISTANCE, FWY_%,
  GIR_%, SG_P, SG_TTG, POINTS, TOP 10, 1ST, Year, MONEY, COUNTRY`.
- `COUNTRY` is empty on every row and is dropped on load. No other missing values.
- Coverage is uneven: 100 rows (2023), 98 (2024), **177 (2025)**. Only 64 players appear
  in all three seasons; many have a single season. Code that assumes ≥2 seasons per
  player will break on the long tail.
- Column names contain spaces and `%` (`TOP 10`, `FWY_%`, `GIR_%`) — use `df["TOP 10"]`,
  never attribute access, and quote them in Altair field strings.
- Money is in whole dollars; UI formats via `money()` in `app.py` (`$21.0M` / `$213,335`).

## Conventions

- The what-if tab iterates `be.features()` (= `model_meta.json["features"]`) and looks each
  one up in `_FIELDS` in `golf_backend.py` for its slider (label, min, max, step). Every
  model feature must have a `_FIELDS` entry or the tab raises `KeyError`.
- Palette: `GREEN #2E7D46`, `GREEN_D #1F5732`, accent `#C77D2E`. Cards use the `.bigcard`
  CSS class via `big_card()`. Reuse these rather than inventing new colors/styles.
- Charts are Altair; pass `use_container_width=True`. Year is ordinal (`Year:O`).
- Public backend functions return plain `dict`s/`DataFrame`s with `float`/`int` cast —
  keep that so Streamlit and JSON serialization never see numpy scalars.
- Default demo player is Scottie Scheffler (Rory McIlroy in Compare); both exist in the data.

## Verifying changes

There is no test suite. After backend changes, run the headless smoke test above; after
UI changes, run `streamlit run app.py` and click through all four tabs (Overview,
Forecast, What-if, Compare) with at least one single-season player selected. Expand
**👀 Site stats** in the sidebar and confirm "Visits" incremented. `streamlit.testing.v1.AppTest`
works for driving `app.py` headlessly (`AppTest.from_file("app.py").run()`); delete the
`visitors.db` it creates afterwards.

## Not yet in the repo (consider before adding features)

Tests and the model-training script.
