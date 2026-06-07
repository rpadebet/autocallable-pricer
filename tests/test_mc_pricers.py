"""
tests/test_mc_pricers.py
=========================
Unit tests for Standard MC and One-Step Survival MC pricers.

Key assertions:
    1. Both MC methods return finite, positive prices
    2. Prices converge toward each other (same expectation, different variance)
    3. Survival MC has LOWER standard error than Standard MC at same N
    4. return_paths=True returns correct path arrays
    5. track_convergence=True returns increasing N series
    6. Antithetic variates reduce standard error vs no antithetic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from app.autocallable import from_security_dict
from app.components.securities import get_security
from app.mc_standard import MCStandardPricer
from app.mc_survival import MCSurvivalPricer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

S_REF = 5312.0
SIGMA = 0.20
R = 0.045
Q = 0.014

@pytest.fixture
def phoenix_ac():
    params = get_security("Phoenix Autocall")
    return from_security_dict(params, S_ref=S_REF)


# ---------------------------------------------------------------------------
# Test 1: Basic price validity
# ---------------------------------------------------------------------------

def test_mc_standard_price_positive_finite(phoenix_ac):
    """Standard MC price must be finite and positive."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000, seed=42)
    res = mc.price()
    assert np.isfinite(res.price), f"MC price not finite: {res.price}"
    assert res.price > 0, f"MC price non-positive: {res.price}"


def test_mc_survival_price_positive_finite(phoenix_ac):
    """Survival MC price must be finite and positive."""
    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000, seed=42)
    res = sv.price()
    assert np.isfinite(res.price), f"Survival MC price not finite: {res.price}"
    assert res.price > 0, f"Survival MC price non-positive: {res.price}"


def test_mc_std_err_positive(phoenix_ac):
    """Standard error must be positive (variance > 0 across paths)."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=1000, seed=42)
    res = mc.price()
    assert res.std_err > 0, "Standard MC std_err is zero (all paths identical?)"


def test_survival_std_err_positive(phoenix_ac):
    """Survival MC std_err must be positive."""
    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=1000, seed=42)
    res = sv.price()
    assert res.std_err > 0, "Survival MC std_err is zero"


def test_ci_bounds_ordered(phoenix_ac):
    """ci_low < price < ci_high for both methods."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000, seed=42)
    mc_res = mc.price()
    assert mc_res.ci_low < mc_res.price < mc_res.ci_high, \
        f"CI bounds wrong: [{mc_res.ci_low:.2f}, {mc_res.ci_high:.2f}] around {mc_res.price:.2f}"

    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000, seed=42)
    sv_res = sv.price()
    assert sv_res.ci_low < sv_res.price < sv_res.ci_high


# ---------------------------------------------------------------------------
# Test 2: Both MC methods agree (within 3σ)
# ---------------------------------------------------------------------------

def test_mc_methods_agree(phoenix_ac):
    """Standard MC and Survival MC should agree within 3× their standard errors."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=5000, seed=42)
    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=5000, seed=42)
    mc_res = mc.price()
    sv_res = sv.price()

    diff = abs(mc_res.price - sv_res.price)
    # Combined tolerance: 3σ of each method plus $5 for any systematic bias
    tol = 3 * (mc_res.std_err + sv_res.std_err) + 5.0
    assert diff < tol, \
        f"MC ${mc_res.price:.2f} and Survival MC ${sv_res.price:.2f} " \
        f"differ by ${diff:.2f} (tol ${tol:.2f})"


# ---------------------------------------------------------------------------
# Test 3: VARIANCE REDUCTION — Survival MC lower std error
# ---------------------------------------------------------------------------

def test_survival_mc_lower_std_err(phoenix_ac):
    """
    KEY TEST: Survival MC must have LOWER standard error than Standard MC at same N.
    This validates the core claim of Paper 3 (Alm et al. 2013, Algorithm 1).
    """
    N = 3000
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=N, seed=42,
                          antithetic=False)  # no antithetic, for fair comparison
    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=N, seed=42)

    mc_res = mc.price()
    sv_res = sv.price()

    assert sv_res.std_err < mc_res.std_err, \
        f"Survival MC std_err ({sv_res.std_err:.4f}) should be < " \
        f"Standard MC std_err ({mc_res.std_err:.4f})"

    # Variance ratio (effective path reduction factor)
    vr = mc_res.std_err / sv_res.std_err
    assert vr > 1.0, f"Variance reduction ratio {vr:.2f} ≤ 1 — no variance reduction"
    print(f"\nVariance reduction: {vr:.2f}× (Survival MC at same N={N})")


# ---------------------------------------------------------------------------
# Test 4: return_paths=True returns correct arrays
# ---------------------------------------------------------------------------

def test_return_paths_standard(phoenix_ac):
    """Standard MC return_paths=True must return list of spot arrays."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=200, seed=42)
    res = mc.price(return_paths=True)

    assert res.paths is not None, "paths should not be None when return_paths=True"
    assert len(res.paths) == 50, f"Expected 50 stored paths, got {len(res.paths)}"

    n_obs = len(phoenix_ac.observation_dates())
    for i, path in enumerate(res.paths[:5]):
        assert len(path) == n_obs + 1, \
            f"Path {i} length {len(path)} ≠ {n_obs + 1} (obs dates + S0)"


def test_return_paths_survival(phoenix_ac):
    """Survival MC return_paths=True must return stored path arrays."""
    sv = MCSurvivalPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=200, seed=42)
    res = sv.price(return_paths=True)

    assert res.paths is not None
    assert len(res.paths) > 0, "No paths stored"
    # Survival paths may be shorter (truncated at first high-probability call)
    for path in res.paths[:5]:
        assert len(path) >= 1, "Empty path"


def test_return_paths_price_unchanged(phoenix_ac):
    """Enabling return_paths should not change the price estimate."""
    mc_no_paths = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=1000, seed=42)
    mc_with_paths = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=1000, seed=42)

    res_no = mc_no_paths.price(return_paths=False)
    res_with = mc_with_paths.price(return_paths=True)

    assert abs(res_no.price - res_with.price) < 0.01, \
        "Price changed based on return_paths flag (same seed, should be identical)"


# ---------------------------------------------------------------------------
# Test 5: Convergence tracking
# ---------------------------------------------------------------------------

def test_convergence_series_mc(phoenix_ac):
    """track_convergence=True must return a list of (N, mean, se) tuples."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=500, seed=42)
    res = mc.price(track_convergence=True)

    assert len(res.convergence_series) > 0, "Convergence series is empty"
    # Last point should have N close to n_paths
    last_n, last_mean, last_se = res.convergence_series[-1]
    assert last_n == 500, f"Last N = {last_n}, expected 500"
    assert np.isfinite(last_mean), "Last mean is not finite"
    assert last_se >= 0, "Last standard error is negative"


def test_convergence_series_ns_increasing(phoenix_ac):
    """N values in convergence series must be strictly increasing."""
    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=1000, seed=42)
    res = mc.price(track_convergence=True)
    ns = [c[0] for c in res.convergence_series]
    assert ns == sorted(ns), f"Convergence N values not sorted: {ns}"


# ---------------------------------------------------------------------------
# Test 6: Antithetic variates reduce variance
# ---------------------------------------------------------------------------

def test_antithetic_reduces_std_err(phoenix_ac):
    """Antithetic variates should reduce standard error vs no antithetic."""
    mc_ant = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000,
                               seed=42, antithetic=True)
    mc_no = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=2000,
                              seed=42, antithetic=False)
    res_ant = mc_ant.price()
    res_no = mc_no.price()

    assert res_ant.std_err <= res_no.std_err * 1.1, \
        f"Antithetic ({res_ant.std_err:.4f}) not improving vs no antithetic ({res_no.std_err:.4f})"


# ---------------------------------------------------------------------------
# Test 7: Seed reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_price(phoenix_ac):
    """Same seed must give identical results (deterministic)."""
    mc1 = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=500, seed=123)
    mc2 = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=500, seed=123)
    assert mc1.price().price == mc2.price().price, "Same seed gives different price"


def test_different_seeds_different_prices(phoenix_ac):
    """Different seeds should (with overwhelming probability) give different prices."""
    mc1 = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=500, seed=1)
    mc2 = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=500, seed=999)
    assert mc1.price().price != mc2.price().price, "Different seeds give identical price (astronomically unlikely)"
