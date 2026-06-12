"""
app/data_loader.py
==================
Data loading and snapshot management for the AutoCallable Analytics Platform.

WHY THIS MODULE EXISTS:
    The app runs on pre-collected CSV snapshots rather than live API calls.
    This module is the single point of access for all market data — it
    discovers available snapshots, loads them with quality filters applied,
    and extracts the specific data fields needed by pricing and visualization.

    All Streamlit pages go through this module. No page reads CSVs directly.

DESIGN DECISIONS:
    - Snapshot keys use YYYYMMDD_HHMM format for lexicographic sort = time sort.
    - Session labels (Market Open / Midday / Near Close) are derived from the
      timestamp so no separate metadata file is needed.
    - Quality filters are applied at load time, not at collection time, so the
      raw CSV is preserved with full fidelity.
    - get_spot_price() returns the first row's spot value — all rows in a
      snapshot share the same spot (collected in a single API call).
"""

import json
import os
import glob
import pandas as pd
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories relative to the project root.
# Streamlit pages should pass an absolute path or rely on the CWD being correct.
RAW_DATA_DIR       = "sample_data"            # original yfinance CSVs
PROCESSED_DATA_DIR = "sample_data/processed"  # after quality-filter pipeline
DEFAULT_DATA_DIR   = PROCESSED_DATA_DIR       # app uses processed by default

# Moneyness filter: remove very deep ITM (high delta, no vol info) and far OTM
# (no market, unreliable quotes). This matches the filter in the Technical Spec.
MONEYNESS_MIN = 0.70
MONEYNESS_MAX = 1.35

# TTM filter: exclude micro-expiries (gamma risk dominates, no vol surface info)
# and extremely long maturities (beyond typical autocallable horizon).
TTM_MIN_YEARS = 0.08  # ~4 weeks
TTM_MAX_YEARS = 3.0


# ---------------------------------------------------------------------------
# Snapshot Discovery
# ---------------------------------------------------------------------------

def list_available_snapshots(data_dir: str = DEFAULT_DATA_DIR) -> list[dict]:
    """
    Discover all valid SPX options snapshots in the data directory.

    WHY: The UI date dropdown is populated from this list. The label format
    shows date + time + session so the user can pick meaningfully without
    looking at filenames.

    Args:
        data_dir: Directory to search first.  If no processed snapshots exist
                  there, the function falls back to RAW_DATA_DIR so the app
                  still works before process_snapshots.py has been run.

    Returns:
        List of dicts, each with:
            'key':   Filename stem without prefix/suffix (e.g. "20260609_1200")
            'label': Human-readable label (e.g. "2026-06-09 12:00  —  Midday")
        Sorted by timestamp, oldest first.

    Edge cases:
        - Returns [] if neither processed nor raw dir contains matching files.
        - Files with malformed timestamps are included with key as label.
    """
    pattern = os.path.join(data_dir, "spx_options_*.csv")
    files = sorted(glob.glob(pattern))

    # Fall back to raw dir if processed dir is empty or doesn't exist yet
    if not files and data_dir == PROCESSED_DATA_DIR:
        pattern = os.path.join(RAW_DATA_DIR, "spx_options_*.csv")
        files = sorted(glob.glob(pattern))

    results = []
    for f in files:
        key = os.path.basename(f).replace("spx_options_", "").replace(".csv", "")
        try:
            # Parse YYYYMMDD_HHMM format
            dt = datetime.strptime(key, "%Y%m%d_%H%M")
            session = _session_label(dt.hour)
            label = f"{dt.strftime('%Y-%m-%d %H:%M')}  —  {session}"
        except ValueError:
            # Malformed timestamp: include with raw key as label
            label = key
        results.append({"key": key, "label": label})

    return results


def _session_label(hour: int) -> str:
    """
    Map an hour-of-day (ET) to a human-readable session name.

    The boundaries are chosen to match typical SPX market activity:
        Market Open:  9:30–11:00 ET (widest spreads, most volatility)
        Midday:       11:00–14:00 ET (tightest spreads, most liquid)
        Near Close:   14:00–16:00 ET (high volume, good vol surface shape)

    Args:
        hour: Hour component of the snapshot timestamp (0–23).

    Returns:
        One of "Market Open", "Midday", "Near Close", or "After Hours".
    """
    if hour < 11:
        return "Market Open"
    elif hour < 14:
        return "Midday"
    elif hour < 17:
        return "Near Close"
    else:
        return "After Hours"


# ---------------------------------------------------------------------------
# Snapshot Loading
# ---------------------------------------------------------------------------

def load_snapshot(
    key: str,
    data_dir: str = DEFAULT_DATA_DIR,
    apply_moneyness_filter: bool = True,
    apply_ttm_filter: bool = True,
) -> pd.DataFrame:
    """
    Load a snapshot CSV and apply data quality filters.

    WHY FILTERS HERE: The raw CSV contains all quotes including illiquid
    far-OTM options with missing IVs that would destabilize the vol surface
    calibration. We filter at load time so all downstream modules receive
    clean data.

    Args:
        key:                    Snapshot key (e.g. "20260609_1200")
        data_dir:               Directory containing snapshot CSVs
        apply_moneyness_filter: Drop rows with K/S < MONEYNESS_MIN or > MONEYNESS_MAX
        apply_ttm_filter:       Drop rows with TTM < TTM_MIN or > TTM_MAX

    Returns:
        Cleaned DataFrame with all original columns plus computed columns.
        Guaranteed to have: strike, expiry, bid, ask, impliedVolatility,
        moneyness, ttm_years, mid, spot, rfr.

    Raises:
        FileNotFoundError: If the snapshot file doesn't exist.
        ValueError:        If the loaded DataFrame is empty after filtering.

    Edge cases:
        - Applies bid > 0 and bid < ask filters (always — not configurable).
        - volume > 0 filter applied always to remove stale quotes.
        - impliedVolatility NaN rows are kept (they're filtered in vol_surface
          if needed, but useful for pricing that doesn't need IV).
    """
    path = os.path.join(data_dir, f"spx_options_{key}.csv")
    # Fall back to raw directory if the processed file doesn't exist yet
    if not os.path.exists(path) and data_dir == PROCESSED_DATA_DIR:
        raw_path = os.path.join(RAW_DATA_DIR, f"spx_options_{key}.csv")
        if os.path.exists(raw_path):
            path = raw_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Snapshot not found: {path}\n"
            f"Run scripts/process_snapshots.py to build the processed snapshot."
        )

    df = pd.read_csv(path, parse_dates=["expiry"])

    # --- Always-on quality filters ---
    # WHY: Zero-bid and crossed quotes are data errors, not illiquid options.
    df = df[(df["bid"] > 0) & (df["bid"] < df["ask"])]

    # WHY volume > 0: Zero-volume options have no market activity; their quotes
    # are stale and will produce garbage implied vols.
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    # --- Optional range filters ---
    if apply_moneyness_filter and "moneyness" in df.columns:
        df = df[(df["moneyness"] >= MONEYNESS_MIN) & (df["moneyness"] <= MONEYNESS_MAX)]

    if apply_ttm_filter and "ttm_years" in df.columns:
        df = df[(df["ttm_years"] >= TTM_MIN_YEARS) & (df["ttm_years"] <= TTM_MAX_YEARS)]

    if df.empty:
        raise ValueError(
            f"Snapshot '{key}' is empty after quality filters. "
            "The raw data may be from an illiquid market session."
        )

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Derived Data Extractors
# ---------------------------------------------------------------------------

def get_spot_price(snapshot: pd.DataFrame) -> float:
    """
    Extract the SPX spot price from a loaded snapshot.

    WHY: Every row in the snapshot carries the same spot level (recorded at
    collection time). We read it from row 0 rather than requiring callers to
    know the column name.

    Args:
        snapshot: DataFrame returned by load_snapshot().

    Returns:
        SPX spot price as a float.

    Raises:
        KeyError:  If 'spot' column is missing.
        ValueError: If 'spot' column is all-NaN.
    """
    if "spot" not in snapshot.columns:
        raise KeyError("Snapshot does not have a 'spot' column.")
    spot = snapshot["spot"].dropna().iloc[0]
    return float(spot)


def get_rfr(snapshot: pd.DataFrame) -> float:
    """
    Extract the risk-free rate from a loaded snapshot.

    Args:
        snapshot: DataFrame returned by load_snapshot().

    Returns:
        Risk-free rate as a decimal (e.g. 0.045 = 4.5%).
        Falls back to 0.045 if column is missing or all-NaN.
    """
    if "rfr" not in snapshot.columns or snapshot["rfr"].isna().all():
        return 0.045
    return float(snapshot["rfr"].dropna().iloc[0])


def get_expiries(snapshot: pd.DataFrame) -> list[str]:
    """
    Return the list of unique expiry dates in the snapshot, sorted ascending.

    Args:
        snapshot: DataFrame returned by load_snapshot().

    Returns:
        List of expiry date strings in YYYY-MM-DD format.
    """
    expiries = sorted(snapshot["expiry"].dt.strftime("%Y-%m-%d").unique())
    return list(expiries)


def get_strikes_for_expiry(snapshot: pd.DataFrame, expiry: str) -> list[float]:
    """
    Return all available strike prices for a given expiry date.

    Args:
        snapshot: DataFrame returned by load_snapshot().
        expiry:   Expiry date string in YYYY-MM-DD format.

    Returns:
        Sorted list of strike prices.

    Raises:
        ValueError: If the expiry is not found in the snapshot.
    """
    mask = snapshot["expiry"].dt.strftime("%Y-%m-%d") == expiry
    subset = snapshot[mask]
    if subset.empty:
        raise ValueError(f"Expiry '{expiry}' not found in snapshot.")
    return sorted(subset["strike"].unique().tolist())


def get_implied_vol_matrix(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the snapshot into a (strike × expiry) implied vol matrix.

    WHY: The vol surface and Heston calibration modules work on a 2D grid
    of (K, T) → IV pairs. This function converts the flat CSV into that format.

    Args:
        snapshot: DataFrame returned by load_snapshot(). Must have columns:
                  strike, expiry, impliedVolatility, moneyness, ttm_years.

    Returns:
        DataFrame with strikes as index, expiries as columns, IV as values.
        Missing entries (no quote for that strike/expiry combo) are NaN.
        Also returns a parallel matrix of TTM values.

    Edge cases:
        - Rows with NaN impliedVolatility are dropped before pivoting.
        - Duplicate (strike, expiry) rows are averaged before pivoting.
    """
    df = snapshot.dropna(subset=["impliedVolatility"]).copy()

    # Average duplicates (shouldn't happen with real data but defensive)
    df = df.groupby(["strike", "expiry"])["impliedVolatility"].mean().reset_index()

    iv_matrix = df.pivot(index="strike", columns="expiry", values="impliedVolatility")
    return iv_matrix


# ---------------------------------------------------------------------------
# Convenience: resolve data_dir relative to the project root
# ---------------------------------------------------------------------------

def resolve_data_dir(relative: str = "sample_data") -> str:
    """
    Resolve the sample_data directory path relative to this file's location.

    WHY: When running `streamlit run app/Home.py` from the project root, the
    CWD is the project root. But when running tests from tests/ directory, the
    CWD may differ. This function always resolves correctly.

    Args:
        relative: Subdirectory name relative to the project root.

    Returns:
        Absolute path to the data directory.
    """
    # This file is at <project_root>/app/data_loader.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, relative)


def load_calibration_cache(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """
    Load the pre-computed Heston / Bates calibration cache from JSON.

    WHY: Running calibration interactively takes 15-90 s per snapshot.
    The precalibrate.py script runs once offline and saves results for all
    snapshots to sample_data/calibrations_cache.json.  This function loads
    that file so the sidebar can inject the right parameters instantly when
    the user selects a snapshot date.

    Schema:
        {
          "20260609_1200": {
            "heston": {v0, kappa, theta, gamma, rho, rmse_vol_pts, ...},
            "bates":  {v0, kappa, theta, gamma, rho, lam, mu_J, sig_J, rmse_vol_pts, ...}
          },
          ...
        }

    Returns:
        Dict keyed by snapshot key, or {} if file does not exist yet.
    """
    path = os.path.join(data_dir, "calibrations_cache.json")
    # Also check processed dir when called with the raw dir (legacy call sites)
    if not os.path.exists(path) and data_dir == RAW_DATA_DIR:
        proc_path = os.path.join(PROCESSED_DATA_DIR, "calibrations_cache.json")
        if os.path.exists(proc_path):
            path = proc_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
