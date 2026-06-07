"""
scripts/generate_synthetic_data.py
====================================
Generate realistic synthetic SPX options snapshots for demo / offline use.

WHY THIS EXISTS:
    The sandbox environment (and Streamlit Cloud) cannot make live API calls to
    Yahoo Finance during a demo. This script generates statistically realistic
    synthetic SPX options data using a Heston-style implied vol surface so the
    app loads, prices, and visualizes correctly even without network access.

    The synthetic data is calibrated to approximate SPX market conditions as of
    June 2026: SPX ~5,300, VIX ~17, moderate vol skew, standard term structure.

USAGE:
    python scripts/generate_synthetic_data.py

OUTPUTS:
    sample_data/spx_options_20260606_1600.csv   (Friday Jun 6 close-ish)
    sample_data/spx_options_20260609_1200.csv   (Monday Jun 9 midday)

    These mimic real yfinance output columns so data_loader.py works identically.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta
import os


# ---------------------------------------------------------------------------
# Black-Scholes Helpers
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes European call price.

    Args:
        S:     Spot price
        K:     Strike
        T:     Time to maturity in years
        r:     Risk-free rate (continuous)
        q:     Dividend yield (continuous)
        sigma: Implied volatility (annual)

    Returns:
        Call price. Returns max(S*exp(-q*T) - K*exp(-r*T), 0) intrinsic value
        when T is very small (avoids division by zero).
    """
    if T < 1e-6:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def synthetic_iv(moneyness: float, ttm: float,
                 atm_vol: float = 0.17,
                 skew: float = -0.12,
                 curvature: float = 0.10,
                 term_slope: float = 0.02) -> float:
    """
    Heuristic implied vol surface calibrated to approximate SPX conditions.

    WHY THIS PARAMETERIZATION:
        - SPX implied vol surface has strong left skew (put buyers for protection).
        - ATM vol term structure slopes upward at short end (typical non-crisis regime).
        - Curvature ensures vol doesn't go negative for far OTM calls.

    Args:
        moneyness:   K/S ratio (1.0 = ATM)
        ttm:         Time to maturity in years
        atm_vol:     ATM vol at 1-year maturity (~17% for SPX in mid-2026)
        skew:        Log-moneyness slope (negative = put skew, typical for equity)
        curvature:   Quadratic term in log-moneyness (smile curvature)
        term_slope:  How much ATM vol decreases per year shorter than 1yr

    Returns:
        Implied volatility as a decimal (e.g. 0.17 = 17%).
    """
    x = np.log(moneyness)  # log-moneyness; 0 = ATM
    ttm_adj = max(ttm, 1 / 52)  # avoid zero-TTM issues

    # Stochastic vol term structure: vol rises with TTM at short end, flattens long
    term_adjustment = term_slope * (1.0 - ttm_adj) / ttm_adj ** 0.3
    base = atm_vol + term_adjustment

    # Skew and smile in log-moneyness space
    iv = base + skew * x + curvature * x ** 2

    # Clamp to realistic range: 5% to 80%
    return float(np.clip(iv, 0.05, 0.80))


# ---------------------------------------------------------------------------
# Snapshot Generator
# ---------------------------------------------------------------------------

def generate_snapshot(
    spot: float,
    rfr: float,
    q: float,
    snapshot_date: date,
    label: str,
    output_dir: str = "sample_data",
) -> str:
    """
    Generate a full SPX-style options snapshot and save to CSV.

    Generates a realistic grid of strikes and expiries matching yfinance output
    format so that data_loader.py can load it without modification.

    Args:
        spot:          SPX spot level (e.g. 5300)
        rfr:           Risk-free rate (e.g. 0.045)
        q:             Dividend yield (e.g. 0.014)
        snapshot_date: The date to label this snapshot
        label:         Time component for filename (e.g. "1600" for 4pm)
        output_dir:    Where to save the CSV

    Returns:
        Path to saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Expiries: weekly for next 4 weeks, then monthly/quarterly out to 3 years
    weekly_expiries = [snapshot_date + timedelta(weeks=i) for i in range(1, 5)]
    monthly_expiries = [
        date(snapshot_date.year + (snapshot_date.month + m - 1) // 12,
             (snapshot_date.month + m - 1) % 12 + 1,
             15)
        for m in range(1, 37)  # monthly out to 3 years
        if date(snapshot_date.year + (snapshot_date.month + m - 1) // 12,
                (snapshot_date.month + m - 1) % 12 + 1,
                15) > snapshot_date + timedelta(weeks=4)
    ]
    expiries = sorted(set(weekly_expiries + monthly_expiries[:20]))  # ~24 expiries

    rows = []
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility

    for exp in expiries:
        ttm = (exp - snapshot_date).days / 365.0
        if ttm < 0.02:
            continue

        # Strike grid: denser ATM, sparser far OTM
        # Weekly: 30 strikes; Monthly: 50–80 strikes
        n_strikes = 30 if ttm < 0.1 else (50 if ttm < 0.5 else 70)
        moneyness_range = 0.25 if ttm < 0.25 else 0.40
        moneyness_grid = np.linspace(1 - moneyness_range, 1 + moneyness_range, n_strikes)
        strikes = np.round(moneyness_grid * spot / 5) * 5  # round to nearest $5

        for K in strikes:
            moneyness = K / spot
            if moneyness < 0.60 or moneyness > 1.45:
                continue

            iv = synthetic_iv(moneyness, ttm)
            price = bs_call_price(spot, K, ttm, rfr, q, iv)

            # Simulate bid-ask spread: tighter ATM, wider OTM/ITM
            spread_factor = 0.03 + 0.15 * abs(np.log(moneyness)) ** 1.5
            spread = max(0.05, price * spread_factor)
            bid = max(0.05, price - spread / 2)
            ask = bid + spread

            # Add small random perturbation to simulate market microstructure
            noise = rng.normal(0, spread * 0.05)
            bid = max(0.05, bid + noise)
            ask = bid + spread

            # Simulate volume and open interest (higher ATM)
            vol_base = 1000 * np.exp(-5 * (np.log(moneyness)) ** 2)
            volume = max(0, int(vol_base * rng.lognormal(0, 0.5)))
            oi = max(0, int(vol_base * 5 * rng.lognormal(0, 0.3)))

            rows.append({
                "contractSymbol": f"SPX{exp.strftime('%y%m%d')}C{int(K):08d}",
                "strike": K,
                "expiry": exp.strftime("%Y-%m-%d"),
                "lastPrice": round((bid + ask) / 2, 2),
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "volume": volume,
                "openInterest": oi,
                "impliedVolatility": round(iv, 6),
                "inTheMoney": bool(K < spot),
                "spot": spot,
                "rfr": rfr,
                "optionType": "call",
                "moneyness": round(moneyness, 6),
                "ttm_years": round(ttm, 6),
                "mid": round((bid + ask) / 2, 2),
            })

    df = pd.DataFrame(rows)

    ts = f"{snapshot_date.strftime('%Y%m%d')}_{label}"
    path = os.path.join(output_dir, f"spx_options_{ts}.csv")
    df.to_csv(path, index=False)

    print(f"  Saved: {path}  ({len(df)} rows, {df['expiry'].nunique()} expiries)")
    return path


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating synthetic SPX options snapshots...")
    print("=" * 60)

    generate_snapshot(
        spot=5312.0,
        rfr=0.0435,
        q=0.014,
        snapshot_date=date(2026, 6, 6),
        label="1600",
        output_dir="sample_data",
    )

    generate_snapshot(
        spot=5328.0,
        rfr=0.0435,
        q=0.014,
        snapshot_date=date(2026, 6, 9),
        label="1200",
        output_dir="sample_data",
    )

    generate_snapshot(
        spot=5295.0,
        rfr=0.0432,
        q=0.014,
        snapshot_date=date(2026, 6, 10),
        label="0945",
        output_dir="sample_data",
    )

    print("\n✅ Synthetic snapshots generated. App can now load data.\n")
