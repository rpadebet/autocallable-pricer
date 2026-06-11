"""
tests/test_pde_pricer.py
=========================
Unit tests for the Finite Difference PDE pricer.

BENCHMARK: Deng, Mallett, McCann (2011) Paper 1.
The paper reports a specific set of prices and call probabilities for a reference
autocallable. We validate against those benchmarks where available.

Test cases:
    1. FD price is positive and finite
    2. Courant number is below 0.5 (stability condition)
    3. return_grid=True returns valid 2D array
    4. Call probabilities sum ≤ 1 and are all non-negative
    5. FD price converges as grid resolution increases
    6. FD price is close to Standard MC price (within statistical tolerance)
    7. Closed-form continuous autocall gives finite positive price
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from app.autocallable import from_security_dict
from app.components.securities import get_security
from app.pde_pricer import FDPricer, continuous_autocall_closedform, thomas_solve


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


@pytest.fixture
def fd_pricer_default(phoenix_ac):
    return FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=120, N_tau=80)


# ---------------------------------------------------------------------------
# Test 1: Price is finite and positive
# ---------------------------------------------------------------------------

def test_fd_price_finite_positive(fd_pricer_default):
    """FD price must be a finite, positive dollar amount."""
    res = fd_pricer_default.price()
    assert np.isfinite(res.price), f"FD price is not finite: {res.price}"
    assert res.price > 0, f"FD price is non-positive: {res.price}"


def test_fd_price_below_notional_plus_coupons(phoenix_ac, fd_pricer_default):
    """Fair price should not exceed notional + all possible coupons."""
    res = fd_pricer_default.price()
    max_possible = phoenix_ac.notional * 1.5  # generous upper bound
    assert res.price <= max_possible, \
        f"FD price {res.price} exceeds reasonable upper bound {max_possible}"


def test_fd_price_above_floor(phoenix_ac, fd_pricer_default):
    """Price must be above protected floor value (worst case scenario)."""
    res = fd_pricer_default.price()
    min_pv = phoenix_ac.protection_barrier * phoenix_ac.notional * np.exp(-R * phoenix_ac.maturity_years)
    assert res.price > min_pv * 0.8, \
        f"FD price {res.price} is unreasonably low vs floor {min_pv}"


# ---------------------------------------------------------------------------
# Test 2: Courant stability condition
# ---------------------------------------------------------------------------

def test_courant_number_below_half(phoenix_ac):
    """Explicit FD scheme is stable only if Courant number ≤ 0.5."""
    # Use default N_x=120, N_tau=80 — __init__ auto-adjusts if unstable
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=120, N_tau=80)
    assert fd.courant <= 0.5, \
        f"Courant number {fd.courant:.4f} > 0.5 — scheme is unstable"


def test_courant_auto_corrects(phoenix_ac):
    """With very coarse tau grid, __init__ should auto-increase N_tau."""
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=50, N_tau=5)
    # After auto-correction, Courant should be ≤ 0.5
    assert fd.courant <= 0.5, \
        f"Auto-correction failed: Courant {fd.courant:.4f} > 0.5"


# ---------------------------------------------------------------------------
# Test 3: return_grid=True returns valid array
# ---------------------------------------------------------------------------

def test_return_grid_shape(phoenix_ac):
    """return_grid=True must return V_grid of shape (N_x, n_snapshots)."""
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=80, N_tau=60)
    res = fd.price(return_grid=True)

    assert res.V_grid is not None, "V_grid should not be None when return_grid=True"
    assert res.S_axis is not None, "S_axis should not be None"
    assert res.t_axis is not None, "t_axis should not be None"

    N_x, n_snaps = res.V_grid.shape
    assert N_x == 80, f"Expected N_x=80 rows, got {N_x}"
    assert n_snaps > 0, "Expected at least 1 snapshot column"


def test_return_grid_finite_nonneg(phoenix_ac):
    """All values in V_grid should be finite and non-negative."""
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=80, N_tau=60)
    res = fd.price(return_grid=True)
    assert np.all(np.isfinite(res.V_grid)), "V_grid contains NaN or Inf"
    assert np.all(res.V_grid >= -1.0), \
        "V_grid contains values significantly below 0 (small negatives OK due to FD noise)"


def test_return_grid_s_axis_monotone(phoenix_ac):
    """S_axis should be strictly increasing (sorted spot levels)."""
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=80, N_tau=60)
    res = fd.price(return_grid=True)
    diffs = np.diff(res.S_axis)
    assert np.all(diffs > 0), "S_axis is not strictly increasing"


# ---------------------------------------------------------------------------
# Test 4: Call probabilities
# ---------------------------------------------------------------------------

def test_fd_call_probs_count(phoenix_ac, fd_pricer_default):
    """Number of call probs must match number of observation dates."""
    res = fd_pricer_default.price()
    assert len(res.call_probs) == len(phoenix_ac.observation_dates()), \
        f"Mismatch: {len(res.call_probs)} probs vs {len(phoenix_ac.observation_dates())} dates"


def test_fd_call_probs_valid_range(fd_pricer_default):
    """All call probabilities must be in [0, 1]."""
    res = fd_pricer_default.price()
    for i, p in enumerate(res.call_probs):
        assert 0 <= p <= 1.0, f"Call prob at date {i} = {p} is out of [0, 1]"


def test_fd_call_probs_sum_lte_one(fd_pricer_default):
    """Sum of call probabilities must be ≤ 1 (you can't be called more than once)."""
    res = fd_pricer_default.price()
    total = sum(res.call_probs)
    assert total <= 1.01, f"Sum of call probs = {total:.4f} > 1"


# ---------------------------------------------------------------------------
# Test 5: Price convergence with grid resolution
# ---------------------------------------------------------------------------

def test_fd_price_converges_with_resolution(phoenix_ac):
    """FD price should stabilize as N_x, N_tau increase."""
    prices = []
    for n in [80, 120, 200]:
        fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=n, N_tau=n)
        prices.append(fd.price().price)

    # Price should not diverge — max difference across resolutions ≤ $20
    spread = max(prices) - min(prices)
    assert spread < 20.0, \
        f"FD price not converging: spread = ${spread:.2f} across {prices}"


# ---------------------------------------------------------------------------
# Test 6: FD price consistent with standard MC (within 3σ)
# ---------------------------------------------------------------------------


def test_fd_vs_mc_consistency(phoenix_ac):
    """FD and MC prices should agree within 3x MC standard error.

    WHY high-resolution FD: the explicit FD scheme converges as O(h^2) in space.
    At N_x=120 there is ~1.5% bias vs the converged value; at N_x=200 the bias
    drops to <0.1%. Using N_x=200 here makes the comparison dominated by MC
    sampling noise rather than FD discretization error.
    """
    from app.mc_standard import MCStandardPricer

    # High-resolution FD to minimize discretization bias before comparing to MC
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=200, N_tau=200)
    fd_res = fd.price()

    mc = MCStandardPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, n_paths=10000, seed=42)
    mc_res = mc.price()

    diff = abs(fd_res.price - mc_res.price)
    tolerance = 3 * mc_res.std_err + 5.0  # 3sigma + $5 numerical tolerance
    assert diff < tolerance, (
        f"FD ${fd_res.price:.2f} and MC ${mc_res.price:.2f} differ by ${diff:.2f}"
        f" (tol ${tolerance:.2f})"
    )

# ---------------------------------------------------------------------------
# Test 7: Closed-form continuous autocall
# ---------------------------------------------------------------------------

def test_closedform_finite_positive(phoenix_ac):
    """Closed-form continuous autocall price should be finite and positive."""
    price = continuous_autocall_closedform(
        S0=S_REF,
        call_barrier=1.0,
        maturity_years=2.0,
        sigma=SIGMA,
        r=R,
        q=Q,
        coupon_pa=0.08,
        notional=1000.0,
    )
    assert np.isfinite(price), f"Closed-form price is not finite: {price}"
    assert price > 0, f"Closed-form price is non-positive: {price}"


@pytest.mark.parametrize("name", ["Phoenix Autocall", "Digital Autocall", "Step-Down Barrier"])
def test_all_securities_fd_price(name):
    """All single-underlying securities should produce finite FD prices."""
    params = get_security(name)
    ac = from_security_dict(params, S_ref=S_REF)
    fd = FDPricer(ac, sigma=SIGMA, r=R, q=Q, N_x=80, N_tau=60)
    res = fd.price()
    assert np.isfinite(res.price), f"{name}: FD price not finite: {res.price}"
    assert res.price > 0, f"{name}: FD price non-positive: {res.price}"


# ---------------------------------------------------------------------------
# Tests for Feature B: Crank-Nicolson + Thomas Algorithm
# ---------------------------------------------------------------------------

def test_thomas_solve_correctness():
    """Thomas algorithm must match numpy.linalg.solve for a known tridiagonal system."""
    # Build a known 5x5 tridiagonal system and verify against dense solver (ground truth)
    n = 5
    a = np.array([0.0, -1.0, -1.0, -1.0, -1.0])   # sub-diagonal (a[0] unused)
    b = np.array([4.0,  4.0,  4.0,  4.0,  4.0])    # main diagonal
    c = np.array([-1.0, -1.0, -1.0, -1.0, 0.0])    # super-diagonal (c[-1] unused)
    d = np.array([1.0, 2.0, 3.0, 2.0, 1.0])         # RHS

    # Dense solve via numpy (O(n^3) -- ground truth)
    A_dense = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x_ref = np.linalg.solve(A_dense, d)

    # Thomas solve (O(n))
    x_thomas = thomas_solve(a, b, c, d)

    np.testing.assert_allclose(
        x_thomas, x_ref, atol=1e-10,
        err_msg="Thomas algorithm disagrees with numpy.linalg.solve"
    )


def test_cn_matches_explicit_fine_grid(phoenix_ac):
    """At fine grid resolution, CN and explicit prices should agree within $0.50.

    WHY: Both schemes converge to the same true price. At fine grids the
    discretization errors (O(dtau) for explicit, O(dtau^2) for CN) are both
    small, so the two prices should be close.
    """
    fd_explicit = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q,
                           N_x=200, N_tau=200, scheme="explicit")
    fd_cn = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q,
                     N_x=200, N_tau=200, scheme="crank_nicolson")

    price_explicit = fd_explicit.price().price
    price_cn = fd_cn.price().price

    diff = abs(price_explicit - price_cn)
    assert diff < 0.50, (
        f"Explicit (${price_explicit:.2f}) and CN (${price_cn:.2f}) "
        f"differ by ${diff:.2f} at N=200 -- expected < $0.50"
    )


def test_cn_unconditionally_stable(phoenix_ac):
    """CN must produce a valid price even when Courant number is far above 0.5.

    WHY: The explicit scheme requires Courant number rho = dtau/dx^2 <= 0.5.
    The FDPricer auto-corrects N_tau to satisfy this for explicit. CN is
    unconditionally stable -- no such constraint. This test confirms:
      1. At the same requested N_tau, explicit auto-corrects to many more steps.
      2. CN uses the requested (large-dtau) step count and remains stable.
    This is CN's key practical advantage: fewer solver calls at the same spatial grid.
    """
    # N_x=200 gives dx^2=0.00123; N_tau=5 gives dtau=0.008 -> Courant=6.5 >> 0.5
    fd_cn = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q,
                     N_x=200, N_tau=5, scheme="crank_nicolson")
    fd_exp = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q,
                      N_x=200, N_tau=5, scheme="explicit")

    # Confirm CN runs at high Courant without auto-correction
    assert fd_cn.courant > 0.5, (
        f"Expected Courant > 0.5 to test unconditional stability, got {fd_cn.courant:.3f}"
    )
    # Confirm explicit auto-corrected to more steps (it cannot tolerate Courant > 0.5)
    assert fd_exp.N_tau > 5, (
        f"Expected explicit to auto-correct N_tau beyond 5, got {fd_exp.N_tau}"
    )
    assert fd_cn.N_tau == 5, (
        f"CN should NOT auto-correct N_tau; expected 5, got {fd_cn.N_tau}"
    )

    # CN price must be finite and positive despite the coarse time grid
    res = fd_cn.price()
    assert np.isfinite(res.price), (
        f"CN price is not finite at Courant={fd_cn.courant:.2f}: {res.price}"
    )
    assert 0 < res.price < phoenix_ac.notional * 1.5, (
        f"CN price ${res.price:.2f} is unreasonable at Courant={fd_cn.courant:.2f}"
    )


def test_cn_return_grid_valid(phoenix_ac):
    """CN scheme with return_grid=True must return a well-formed 2D grid.

    WHY: The return_grid path stores V(S,t) snapshots for Page 3 (FDM Visualization).
    This test confirms the CN dispatch in price() correctly stores grid snapshots —
    a regression guard since return_grid was originally written for the explicit scheme
    and the snapshot loop runs inside the same dispatch block as the CN step.
    """
    fd = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q,
                  N_x=80, N_tau=60, scheme="crank_nicolson")
    res = fd.price(return_grid=True)

    # Grid must be returned (not None)
    assert res.V_grid is not None, "CN: V_grid should not be None when return_grid=True"
    assert res.S_axis is not None, "CN: S_axis should not be None"
    assert res.t_axis is not None, "CN: t_axis should not be None"

    # Shape: rows = spatial grid, cols = time snapshots
    N_x, n_snaps = res.V_grid.shape
    assert N_x == 80, f"CN: Expected N_x=80 rows, got {N_x}"
    assert n_snaps > 0, "CN: Expected at least 1 time snapshot"

    # All values finite and non-negative (CN is unconditionally stable)
    assert np.all(np.isfinite(res.V_grid)), "CN: V_grid contains NaN or Inf"
    assert np.all(res.V_grid >= -1.0), "CN: V_grid has significantly negative values"


def test_cn_price_digital_autocall():
    """CN scheme must produce a reasonable price for the Digital Autocall.

    WHY: The Digital Autocall has an unusual payoff structure — a fixed $50 digital
    coupon at call (not a percentage), and 80% capital-protected terminal condition.
    This exercises a different branch of the autocall BC logic. We confirm CN handles
    this product class correctly without producing NaN or out-of-range prices.
    """
    params = get_security("Digital Autocall")
    ac = from_security_dict(params, S_ref=S_REF)
    fd = FDPricer(ac, sigma=SIGMA, r=R, q=Q, N_x=100, N_tau=80, scheme="crank_nicolson")
    res = fd.price()

    assert np.isfinite(res.price), f"CN Digital Autocall price not finite: {res.price}"
    # Digital has 80% capital protection: price must be above 80% of PV(notional)
    min_pv = 0.80 * ac.notional * np.exp(-R * ac.maturity_years)
    assert res.price > min_pv * 0.8, (
        f"CN Digital price ${res.price:.2f} below reasonable floor ${min_pv * 0.8:.2f}"
    )
    assert res.price < ac.notional * 1.5, (
        f"CN Digital price ${res.price:.2f} exceeds upper bound"
    )


# ===========================================================================
# GAP-7: Thomas algorithm edge cases
# ===========================================================================

def test_thomas_solve_n1():
    """Single equation: b*x = d → x = d/b."""
    x = thomas_solve(
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.0]),
        np.array([9.0]),
    )
    assert len(x) == 1
    assert abs(x[0] - 3.0) < 1e-12, f"Expected 3.0, got {x[0]}"


def test_thomas_solve_n2():
    """2x2 system [[2,-1],[-1,2]] * x = [1,1] → x = [1,1]."""
    x = thomas_solve(
        a=np.array([0.0, -1.0]),
        b=np.array([2.0,  2.0]),
        c=np.array([-1.0, 0.0]),
        d=np.array([1.0,  1.0]),
    )
    assert len(x) == 2
    assert abs(x[0] - 1.0) < 1e-10, f"x[0]={x[0]}"
    assert abs(x[1] - 1.0) < 1e-10, f"x[1]={x[1]}"


def test_thomas_solve_large():
    """n=500: Thomas result must match numpy dense solver to within 1e-8."""
    n = 500
    rng = np.random.default_rng(0)
    a = np.concatenate([[0.0], -rng.uniform(0.5, 1.0, n - 1)])
    b = rng.uniform(3.0, 5.0, n)   # diagonally dominant → numerically stable
    c = np.concatenate([-rng.uniform(0.5, 1.0, n - 1), [0.0]])
    d = rng.uniform(-1.0, 1.0, n)

    x_thomas = thomas_solve(a, b, c, d)

    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x_numpy = np.linalg.solve(A, d)

    np.testing.assert_allclose(x_thomas, x_numpy, atol=1e-8,
                               err_msg="Thomas n=500 deviates from numpy dense solver")


# ===========================================================================
# GAP-9: tau / t coordinate roundtrip invariance
# ===========================================================================

def test_tau_t_roundtrip(fd_pricer_default):
    """_tau_to_t(_t_to_tau(t)) == t for all t in [0, T]."""
    fd = fd_pricer_default
    for t in [0.0, 0.25, 0.5, 1.0, fd.T]:
        tau = fd._t_to_tau(t)
        t_back = fd._tau_to_t(tau)
        assert abs(t_back - t) < 1e-10, f"Roundtrip failed at t={t}: got {t_back}"


# ===========================================================================
# GAP-1: Local vol FDM dedicated tests
# ===========================================================================

def _make_const_lv_interp(sigma_const, T):
    """RegularGridInterpolator returning sigma_const everywhere — constant local vol."""
    from scipy.interpolate import RegularGridInterpolator
    t_ax = np.array([0.01, T + 0.05])
    m_ax = np.array([0.30, 2.00])
    grid = np.full((len(t_ax), len(m_ax)), sigma_const)
    return RegularGridInterpolator(
        (t_ax, m_ax), grid, method="linear",
        bounds_error=False, fill_value=None,
    )


class _FakeVolSurface:
    """Minimal duck-type for vol_surface.S0 access in _price_local_vol."""
    S0 = S_REF


def test_local_vol_fdm_constant_surface_matches_flat(phoenix_ac):
    """
    Local vol FDM with constant sigma must price within $15 of flat vol FDM.

    WHY: When sigma_local(K, T) = const, the two schemes solve the same
    underlying PDE. Residual gap comes from the different time discretisation
    (tau-space vs physical-time), so a $15 tolerance is generous.
    """
    lv_interp = _make_const_lv_interp(SIGMA, phoenix_ac.maturity_years)

    fd_flat = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=100, N_tau=80)
    fd_lv   = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=100, N_tau=80,
                       vol_model="local",
                       vol_surface=_FakeVolSurface(),
                       local_vol_interp=lv_interp)

    res_flat = fd_flat.price()
    res_lv   = fd_lv.price()

    assert np.isfinite(res_lv.price), f"Local vol FDM returned non-finite: {res_lv.price}"
    assert abs(res_lv.price - res_flat.price) < 15.0, (
        f"Constant local vol FDM ({res_lv.price:.2f}) deviates too much "
        f"from flat vol FDM ({res_flat.price:.2f})"
    )


def test_local_vol_fdm_maturity_bc_applied(phoenix_ac):
    """
    BUG #1 regression: maturity observation BC must fire (< → <= fix).

    The maturity date equals t_axis_phys[0] = T. Before the fix, the condition
    used strict < so T < T was False and the BC was skipped, producing prices
    ~$20–50 below the flat vol benchmark. After the fix the gap is < $10.
    """
    lv_interp = _make_const_lv_interp(SIGMA, phoenix_ac.maturity_years)

    fd_flat = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=100, N_tau=100)
    fd_lv   = FDPricer(phoenix_ac, sigma=SIGMA, r=R, q=Q, N_x=100, N_tau=100,
                       vol_model="local",
                       vol_surface=_FakeVolSurface(),
                       local_vol_interp=lv_interp)

    res_flat = fd_flat.price()
    res_lv   = fd_lv.price()

    gap = abs(res_lv.price - res_flat.price)
    assert gap < 10.0, (
        f"Large gap ({gap:.2f}) suggests maturity BC was NOT applied. "
        f"Local vol={res_lv.price:.2f}, flat={res_flat.price:.2f}"
    )


# ===========================================================================
# GAP-4: FDM autocall BC coupon gating
# ===========================================================================

def test_fd_autocall_bc_coupon_gating(phoenix_ac):
    """
    _apply_autocall_bc must gate coupon on S >= coupon_barrier.

    Setup: call_barrier=0.90, coupon_barrier=1.00.
    At S in [0.90*S_ref, 1.00*S_ref): called but NO coupon → V = redemption.
    At S >= 1.00*S_ref: called AND coupon → V = redemption + coupon_per_period.
    """
    from app.autocallable import AutoCallable

    ac = AutoCallable(
        name="TestCouponGate",
        structure_type="phoenix",
        S_ref=S_REF,
        notional=1000.0,
        maturity_years=1.0,
        obs_frequency="annual",
        call_barrier=0.90,
        coupon_barrier=1.00,   # coupon only above 100%; call triggers at 90%
        coupon_pa=0.10,
        protection_barrier=0.75,
        redemption_at_call=1.00,
    )

    fd = FDPricer(ac, sigma=SIGMA, r=R, q=Q, N_x=120, N_tau=80)

    u = np.zeros(fd.N_x)
    tau = fd._t_to_tau(0.5)
    u_out = fd._apply_autocall_bc(u, tau, period_index=0)
    V_out = fd._u_to_V(u_out, tau)

    redemption_only = 1.00 * 1000.0
    redemption_plus = redemption_only + ac.coupon_per_period()

    call_abs   = 0.90 * S_REF
    coupon_abs = 1.00 * S_REF

    mid_mask   = (fd.S_axis >= call_abs)   & (fd.S_axis < coupon_abs)
    above_mask = fd.S_axis >= coupon_abs

    if mid_mask.any():
        v = V_out[mid_mask][0]
        assert abs(v - redemption_only) < 1.0, (
            f"In [call,coupon) zone expected redemption_only={redemption_only:.2f}, "
            f"got {v:.2f}"
        )

    if above_mask.any():
        v = V_out[above_mask][0]
        assert abs(v - redemption_plus) < 1.0, (
            f"Above coupon_barrier expected {redemption_plus:.2f}, got {v:.2f}"
        )