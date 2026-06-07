"""
scripts/collect_snapshot.py
============================
One-time SPX options chain snapshot collector for the AutoCallable Analytics Platform.

WHY THIS EXISTS:
    The app never makes live API calls at runtime. Instead, all pricing and visualization
    runs on pre-collected CSV snapshots stored in sample_data/. This ensures the demo
    never fails due to market hours, rate limits, or API outages.

USAGE:
    Run 3× per trading day during the collection window (Jun 9–11, 2026):
        python scripts/collect_snapshot.py
    Recommended times (ET): 9:45am, 12:00pm, 3:45pm
    Keep the highest-quality snapshot per day (check the quality score printed).

OUTPUTS:
    sample_data/spx_options_YYYYMMDD_HHMM.csv
    (e.g. sample_data/spx_options_20260609_1200.csv)

DATA SOURCE:
    yfinance — fetches SPX options chain from Yahoo Finance.
    SPX spot: ^SPX  |  Risk-free rate: ^IRX (13-week T-bill annualized)

QUALITY SCORING:
    score_snapshot() returns a dict with metrics and a composite quality score.
    Higher is better. Aim for score > 60 before keeping a snapshot.

EDGE CASES:
    - If yfinance returns no data (market closed, API down), the script exits with a
      clear error rather than saving an empty or corrupt file.
    - After-hours and weekend runs may still return the last trading day's chain.
      Check the 'spot' column to confirm the date matches expectations.
    - Some expiries return empty option chains — these are silently skipped.
"""

import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------

def score_snapshot(df: pd.DataFrame) -> dict:
    """
    Score a snapshot for data quality. Higher score = better data.

    WHY: We collect 3 snapshots per day but only keep the best one. This
    objective score lets us pick without eyeballing thousands of rows.

    Scoring components:
        - avg_strikes_per_expiry × 0.40  → more strikes = richer vol surface
        - pct_tight_spread × 0.40        → tight bid-ask = liquid, reliable quotes
        - pct_missing_iv × (−0.20)       → missing IV = rows we'll have to drop

    Args:
        df: Raw options DataFrame (before quality filters, but after basic cleaning).

    Returns:
        dict with keys: n_expiries, avg_strikes_per_expiry, pct_tight_spread,
                        pct_missing_iv, quality_score.
    """
    if df.empty:
        return {
            "n_expiries": 0,
            "avg_strikes_per_expiry": 0.0,
            "pct_tight_spread": 0.0,
            "pct_missing_iv": 100.0,
            "quality_score": 0.0,
        }

    by_expiry = df.groupby("expiry")
    n_expiries = len(by_expiry)
    avg_strikes = by_expiry.size().mean()

    # Relative spread: (ask - bid) / mid. Tight = < 10% of mid.
    mid = (df["ask"] + df["bid"]) / 2
    rel_spread = (df["ask"] - df["bid"]) / mid.replace(0, float("nan"))
    pct_tight = (rel_spread < 0.10).mean() * 100

    pct_missing_iv = df["impliedVolatility"].isna().mean() * 100

    score = avg_strikes * 0.4 + pct_tight * 0.4 - pct_missing_iv * 0.2

    return {
        "n_expiries": n_expiries,
        "avg_strikes_per_expiry": round(float(avg_strikes), 1),
        "pct_tight_spread": round(float(pct_tight), 1),
        "pct_missing_iv": round(float(pct_missing_iv), 1),
        "quality_score": round(float(score), 1),
    }


# ---------------------------------------------------------------------------
# Snapshot Collection
# ---------------------------------------------------------------------------

def save_snapshot(df: pd.DataFrame, output_dir: str = "sample_data") -> str:
    """
    Save a cleaned options DataFrame as a timestamped CSV.

    WHY: Timestamped filenames let us keep multiple snapshots per day and
    display them in the UI with date + time + session label.

    Args:
        df:          Cleaned options DataFrame ready for saving.
        output_dir:  Directory to write into. Created if it doesn't exist.

    Returns:
        Absolute path of the saved file.

    Edge cases:
        - Creates output_dir if it doesn't exist (e.g. first run).
        - Does NOT overwrite an existing file for the same minute; if you
          run twice in the same minute, the second run will overwrite.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(output_dir, f"spx_options_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def collect_spx_snapshot(output_dir: str = "sample_data") -> None:
    """
    Fetch the SPX options chain from Yahoo Finance and save as a quality-scored CSV.

    WHY: SPX (^SPX) is the reference underlying for the pre-configured autocallable
    securities. We collect calls only — puts are not needed for vol surface construction
    under put-call parity with the known risk-free rate.

    Steps:
        1. Fetch current SPX spot from yfinance.
        2. Fetch 13-week T-bill rate (^IRX) as risk-free rate proxy.
        3. Iterate over all available expiries, downloading the full calls chain.
        4. Apply basic cleaning: remove zero-bid and crossed-market quotes.
        5. Add derived columns: moneyness, TTM, mid, option type.
        6. Score and save.

    Args:
        output_dir: Relative or absolute path to the sample_data directory.

    Edge cases:
        - If SPX history returns empty (market closed / API error), we raise
          immediately with a descriptive error rather than saving a bad file.
        - If an individual expiry chain is empty, we skip it silently.
        - If ^IRX data is unavailable, we fall back to rfr = 0.045 (4.5%)
          and print a warning.
    """
    print("\n" + "=" * 60)
    print("AutoCallable Analytics Platform — SPX Snapshot Collector")
    print(f"Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 60)

    # --- Step 1: Fetch SPX spot ---
    spx = yf.Ticker("^SPX")
    hist = spx.history(period="1d")
    if hist.empty:
        raise RuntimeError(
            "yfinance returned no SPX price data. Market may be closed or API unreachable. "
            "Try running during US market hours (9:30am–4:00pm ET)."
        )
    spot = float(hist["Close"].iloc[-1])
    print(f"  SPX spot: {spot:,.2f}")

    # --- Step 2: Risk-free rate (13-week T-bill) ---
    try:
        irx_hist = yf.Ticker("^IRX").history(period="1d")
        rfr = float(irx_hist["Close"].iloc[-1]) / 100
        print(f"  Risk-free rate (^IRX): {rfr:.4f} ({rfr*100:.2f}%)")
    except Exception:
        rfr = 0.045
        print(f"  WARNING: Could not fetch ^IRX — using fallback rfr = {rfr:.4f}")

    # --- Step 3: Fetch options chains ---
    expiries = spx.options
    if not expiries:
        raise RuntimeError("yfinance returned no option expiries for ^SPX.")
    print(f"  Available expiries: {len(expiries)}")

    collection_date = datetime.now().date()
    frames = []

    for exp in expiries:
        try:
            chain = spx.option_chain(exp)
            calls = chain.calls.copy()
            if calls.empty:
                continue

            # Add metadata
            calls["expiry"] = exp
            calls["spot"] = spot
            calls["rfr"] = rfr
            calls["optionType"] = "call"

            # Derived columns
            calls["moneyness"] = calls["strike"] / spot
            exp_date = pd.to_datetime(exp).date()
            calls["ttm_years"] = (exp_date - collection_date).days / 365.0
            calls["mid"] = (calls["bid"] + calls["ask"]) / 2

            frames.append(calls)
        except Exception as e:
            # Skip silently — some expiries return 404 or malformed data
            print(f"  Skipped expiry {exp}: {e}")
            continue

    if not frames:
        raise RuntimeError("No valid option chains fetched. All expiries returned errors.")

    df = pd.concat(frames, ignore_index=True)

    # --- Step 4: Basic cleaning — remove zero-bid and crossed quotes ---
    # WHY: Zero-bid strikes are stale or illiquid and will produce nonsensical implied
    # vols. Crossed markets (bid >= ask) indicate data errors.
    n_raw = len(df)
    df = df[(df["bid"] > 0) & (df["bid"] < df["ask"])]
    n_clean = len(df)
    print(f"  Rows after basic cleaning: {n_clean} (removed {n_raw - n_clean} bad ticks)")

    # --- Step 5: Score and save ---
    scores = score_snapshot(df)
    path = save_snapshot(df, output_dir)

    print(f"\n  ✅ Snapshot saved: {path}")
    print(f"  Rows:                    {n_clean}")
    print(f"  Expiry dates:            {scores['n_expiries']}")
    print(f"  Avg strikes / expiry:    {scores['avg_strikes_per_expiry']}")
    print(f"  Tight spread (< 10%):    {scores['pct_tight_spread']}%")
    print(f"  Missing IV:              {scores['pct_missing_iv']}%")
    print(f"  Quality score:           {scores['quality_score']}  ← higher is better (aim > 60)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding output directory from command line
    # Usage: python scripts/collect_snapshot.py [output_dir]
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    collect_spx_snapshot(output_dir=output_dir)
