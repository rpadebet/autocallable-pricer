"""
scripts/benchmark_calibration.py
==================================
Quick speed benchmark for Heston, Merton, and Bates calibration.

Run this from the project root:
    python scripts/benchmark_calibration.py

Expected timings on a modern laptop (4-core):
    Heston:   5–20 s
    Merton:   3–10 s
    Bates:   10–25 s
    Total:   18–55 s

If calibration takes >120 s something is wrong — check that
_heston_cf_batch and _bates_cf_batch are being loaded (not a stale .pyc).
Delete __pycache__/ folders and retry if needed.
"""

import sys
import os
import time

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

print("Importing heston module...")
t0 = time.time()
from app.heston import (
    HestonModel, calibrate_merton, calibrate_bates,
    _heston_cf_batch, _bates_cf_batch,
)
print(f"Import: {time.time()-t0:.2f}s")

# Verify vectorized functions are present
for fn_name in ["_heston_cf_batch", "_bates_cf_batch"]:
    assert fn_name in dir(sys.modules["app.heston"]), f"MISSING: {fn_name}"
print("✅ _heston_cf_batch and _bates_cf_batch present\n")

# Quick vectorized CF speed test
phi = np.linspace(0.01, 200, 64)
t0 = time.time()
cf = _heston_cf_batch(phi + 0j, 5000, 1.0, 0.045, 0.015, 0.04, 1.5, 0.04, 0.3, -0.7)
print(f"_heston_cf_batch (64 phi, 1 TTM): {(time.time()-t0)*1000:.2f} ms  shape={cf.shape}")

# Synthetic SPX-like market data (25 quotes across 3 maturities)
np.random.seed(42)
S0, r, q = 5000.0, 0.045, 0.015
mn = np.array([0.85, 0.90, 0.95, 0.97, 1.00, 1.02, 1.05, 1.08, 1.10,
               0.85, 0.90, 0.95, 1.00, 1.05, 1.10,
               0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15,
               0.92, 0.98, 1.03])
tt = np.array([0.25]*9 + [0.50]*6 + [1.0]*7 + [2.0]*3)
base_iv = 0.20 - 0.10*(mn - 1.0) + 0.05*(mn - 1.0)**2
iv = np.clip(base_iv + np.random.normal(0, 0.005, len(mn)), 0.05, 0.60)
df = pd.DataFrame({
    "moneyness": mn,
    "ttm_years": tt,
    "impliedVolatility": iv,
    "strike": mn * S0,
})

class MockSurface:
    raw_df = df

print(f"\nBenchmark with {len(df)} market quotes:")
print("-" * 50)

# Heston
print("Calibrating Heston...")
t0 = time.time()
model = HestonModel(S0=S0, r=r, q=q)
result = model.calibrate(MockSurface())
t_heston = time.time() - t0
status = "✅" if t_heston < 45 else "⚠️ SLOW"
print(f"{status} Heston:  {t_heston:.1f}s  rmse={result['rmse_vol_pts']:.2f} vol-pts  "
      f"feller={result['feller_satisfied']}")
print(f"   v0={result['v0']:.4f}  kappa={result['kappa']:.2f}  "
      f"theta={result['theta']:.4f}  gamma={result['gamma']:.2f}  rho={result['rho']:.2f}")

# Merton
print("Calibrating Merton...")
t0 = time.time()
m_result = calibrate_merton(df, S0, r, q)
t_merton = time.time() - t0
status = "✅" if t_merton < 20 else "⚠️ SLOW"
print(f"{status} Merton:  {t_merton:.1f}s  rmse={m_result['rmse_vol_pts']:.2f} vol-pts")
print(f"   sigma={m_result['sigma']:.3f}  lam={m_result['lam']:.2f}  "
      f"mu_J={m_result['mu_J']:.3f}  sig_J={m_result['sig_J']:.3f}")

# Bates (warm-started from Heston)
print("Calibrating Bates (warm-started from Heston)...")
t0 = time.time()
b_result = calibrate_bates(df, S0, r, q, heston_init=result)
t_bates = time.time() - t0
status = "✅" if t_bates < 40 else "⚠️ SLOW"
print(f"{status} Bates:   {t_bates:.1f}s  rmse={b_result['rmse_vol_pts']:.2f} vol-pts  "
      f"feller={b_result['feller_satisfied']}")

total = t_heston + t_merton + t_bates
status = "✅" if total < 90 else "⚠️ SLOW"
print("-" * 50)
print(f"{status} TOTAL:   {total:.1f}s  (target: <90s for 'Calibrate All Models')")
print()
if total < 90:
    print("🎉 Calibration is fast enough for the demo!")
else:
    print("⚠️  Still slow. Check for stale .pyc files:")
    print("   Delete all __pycache__/ folders in the project and retry.")
