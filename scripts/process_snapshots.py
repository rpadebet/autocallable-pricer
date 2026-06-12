"""
scripts/process_snapshots.py
============================
Build "processed" option-chain snapshots from raw yfinance CSVs.

Raw snapshots are kept untouched in sample_data/.
Processed snapshots are written to sample_data/processed/.

Five filters are applied in order:

1. OTM-only selection
   Keep OTM calls (strike >= spot * 0.99) and OTM puts (strike < spot * 0.99).
   ITM options have prices dominated by intrinsic value — the bid-ask spread
   swamps the vol signal we are trying to calibrate.

2. Bid-ask spread quality  (ask - bid) / mid <= 30%
   A 30 % relative spread means the mid-price has ± 15 % uncertainty, larger
   than the vol signal.  Widening this threshold risks feeding garbage quotes
   to the optimizer.

3. Volume > 0
   Zero-volume options have stale quotes (market-maker auto-fills that may
   not have been updated on the trading day).

4. Calendar-spread no-arbitrage
   For each strike, total implied variance IV² * T must be non-decreasing
   in maturity T.  Violations flag stale short-dated quotes; the shorter-
   dated violating point is dropped.

5. Butterfly no-arbitrage (convexity in strike)
   For each (expiry, optionType) slice with >= 3 strikes, any strike whose
   IV exceeds the linear interpolation of its two neighbours by more than
   2 vol-pts is a convexity violation and is dropped.

USAGE:
    cd <project-root>
    python scripts/process_snapshots.py            # skip existing
    python scripts/process_snapshots.py --force    # overwrite all
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(ROOT, "sample_data")
PROC_DIR = os.path.join(ROOT, "sample_data", "processed")

SPREAD_MAX    = 0.30    # relative bid-ask spread ceiling
BUTTERFLY_TOL = 0.02   # 2 vol-pts convexity-violation threshold
CAL_TOL       = 1e-4   # calendar-spread tolerance (variance units)
TTM_MIN       = 0.02   # ~7 days — removes same-day expiries that break IV math


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def otm_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Keep OTM calls (strike >= spot * 0.99) and OTM puts (strike < spot * 0.99).
    The 1 % ATM band is assigned to the call side by convention.
    Options without an optionType column are passed through unchanged.
    """
    if "optionType" not in df.columns or "spot" not in df.columns:
        return df, 0
    boundary = df["spot"] * 0.99
    call_mask = (df["optionType"] == "call") & (df["strike"] >= boundary)
    put_mask  = (df["optionType"] == "put")  & (df["strike"] <  boundary)
    kept = df[call_mask | put_mask].copy()
    return kept, len(df) - len(kept)


def spread_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop options where (ask - bid) / mid > SPREAD_MAX."""
    if "bid" not in df.columns or "ask" not in df.columns:
        return df, 0
    mid        = (df["bid"] + df["ask"]) / 2.0
    spread_pct = (df["ask"] - df["bid"]) / mid.replace(0, np.nan)
    kept = df[spread_pct.fillna(1.0) <= SPREAD_MAX].copy()
    return kept, len(df) - len(kept)


def volume_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop zero-volume and missing-volume options."""
    if "volume" not in df.columns:
        return df, 0
    kept = df[df["volume"].notna() & (df["volume"] > 0)].copy()
    return kept, len(df) - len(kept)


def calendar_spread_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    For each strike, total implied variance IV² * T must be non-decreasing in T.
    When two adjacent maturities violate this, the shorter-dated point is dropped.
    Rows with NaN or zero IV are excluded from the check (not flagged).
    """
    if "impliedVolatility" not in df.columns or "ttm_years" not in df.columns:
        return df, 0

    valid = df["impliedVolatility"].notna() & (df["impliedVolatility"] > 0) & (df["ttm_years"] > TTM_MIN)
    drop_idx = set()

    for strike, grp in df[valid].groupby("strike"):
        grp      = grp.sort_values("ttm_years")
        tv       = (grp["impliedVolatility"] ** 2 * grp["ttm_years"]).values
        idxs     = grp.index.tolist()
        for i in range(1, len(tv)):
            if tv[i] < tv[i - 1] - CAL_TOL:
                drop_idx.add(idxs[i - 1])   # shorter-dated offender

    kept = df.drop(index=list(drop_idx)).copy()
    return kept, len(drop_idx)


def butterfly_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    For each (expiry, optionType) slice with >= 3 strikes, drop any interior
    strike whose IV is more than BUTTERFLY_TOL above the average of its two
    nearest neighbours (a convexity violation / isolated spike).
    """
    if "impliedVolatility" not in df.columns:
        return df, 0

    drop_idx  = set()
    group_keys = ["expiry"] + (["optionType"] if "optionType" in df.columns else [])

    for _, grp in df.groupby(group_keys):
        valid = grp.dropna(subset=["impliedVolatility"])
        valid = valid[valid["impliedVolatility"] > 0].sort_values("strike")
        if len(valid) < 3:
            continue
        ivs  = valid["impliedVolatility"].values
        idxs = valid.index.tolist()
        for i in range(1, len(ivs) - 1):
            avg = (ivs[i - 1] + ivs[i + 1]) / 2.0
            if ivs[i] > avg + BUTTERFLY_TOL:
                drop_idx.add(idxs[i])

    kept = df.drop(index=list(drop_idx)).copy()
    return kept, len(drop_idx)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_snapshot(raw_path: str, out_path: str) -> dict:
    """
    Apply baseline + five filters to one raw snapshot CSV, write the result,
    and return per-stage row counts for the summary report.
    """
    df = pd.read_csv(raw_path, parse_dates=["expiry"])

    # Baseline: bid > 0, no crossed markets — same as data_loader always applies
    df = df[(df["bid"] > 0) & (df["bid"] < df["ask"])].copy()
    counts = {"raw_clean": len(df)}

    df, n1 = otm_filter(df);              counts["drop_otm"]      = n1
    df, n2 = spread_filter(df);           counts["drop_spread"]   = n2
    df, n3 = volume_filter(df);           counts["drop_volume"]   = n3
    df, n4 = calendar_spread_filter(df);  counts["drop_calendar"] = n4
    df, n5 = butterfly_filter(df);        counts["drop_butterfly"]= n5
    counts["final"] = len(df)

    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Process raw SPX option snapshots.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing processed files.")
    args = parser.parse_args()

    os.makedirs(PROC_DIR, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "spx_options_*.csv")))
    if not raw_files:
        print("No raw snapshots found in", RAW_DIR)
        return

    print(f"Raw snapshots : {RAW_DIR}")
    print(f"Processed dir : {PROC_DIR}")
    print(f"Snapshots     : {len(raw_files)}")
    print()
    hdr = (f"  {'SNAPSHOT':<22}  {'CLEAN':>5}  "
           f"{'OTM':>5}  {'SPREAD':>6}  {'VOL':>5}  "
           f"{'CAL':>5}  {'BFLY':>5}  {'FINAL':>5}  {'KEPT%':>6}")
    print(hdr)
    print("-" * len(hdr))

    for raw_path in raw_files:
        fname   = os.path.basename(raw_path)
        out_path = os.path.join(PROC_DIR, fname)
        key     = fname.replace("spx_options_", "").replace(".csv", "")

        if os.path.exists(out_path) and not args.force:
            print(f"  {key:<22}  skip (already processed — use --force to redo)")
            continue

        try:
            c   = process_snapshot(raw_path, out_path)
            pct = c["final"] / max(c["raw_clean"], 1) * 100
            print(
                f"  {key:<22}  {c['raw_clean']:>5}  "
                f"{c['drop_otm']:>5}  {c['drop_spread']:>6}  {c['drop_volume']:>5}  "
                f"{c['drop_calendar']:>5}  {c['drop_butterfly']:>5}  {c['final']:>5}  {pct:>5.1f}%"
            )
        except Exception as e:
            print(f"  {key:<22}  ERROR: {e}")

    print()
    print("Done.  Run:  python scripts/precalibrate.py --force")
    print("       to regenerate the calibration cache from processed data.")


if __name__ == "__main__":
    main()
