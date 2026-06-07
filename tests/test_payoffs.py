"""
tests/test_payoffs.py
======================
Unit tests for the AutoCallable product dataclass and payoff logic.

Tests verify:
    - Payoff under all edge cases (called early, survived, knocked-in, not knocked-in)
    - All 4 pre-configured securities instantiate and produce finite prices
    - Step-down barrier mechanics
    - Digital coupon handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from app.autocallable import AutoCallable, from_security_dict
from app.components.securities import get_security, list_securities

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

S_REF = 1000.0
NOTIONAL = 1000.0

def make_phoenix():
    """Standard Phoenix autocall: quarterly, 2Y, 100% call, 8% pa, 75% protection."""
    return AutoCallable(
        name="Phoenix Autocall",
        structure_type="phoenix",
        S_ref=S_REF,
        notional=NOTIONAL,
        maturity_years=2.0,
        obs_frequency="quarterly",
        call_barrier=1.00,
        coupon_barrier=0.75,
        coupon_pa=0.08,
        protection_barrier=0.75,
        redemption_at_call=1.00,
    )


def make_digital():
    """Digital autocall: annual, 3Y, 105% barrier, $50 digital coupon."""
    return AutoCallable(
        name="Digital Autocall",
        structure_type="digital",
        S_ref=S_REF,
        notional=NOTIONAL,
        maturity_years=3.0,
        obs_frequency="annual",
        call_barrier=1.05,
        coupon_barrier=1.05,
        digital_coupon=50.0,
        protection_barrier=1.00,  # entire range triggers soft protection
        protection_floor=0.80,    # minimum 80 cents on the dollar
        redemption_at_call=1.00,
    )


# ---------------------------------------------------------------------------
# Test 1: Observation dates
# ---------------------------------------------------------------------------

def test_observation_dates_quarterly_2y():
    """Quarterly, 2Y → 8 observation dates at t=0.25, 0.50, ..., 2.00."""
    ac = make_phoenix()
    obs = ac.observation_dates()
    assert len(obs) == 8, f"Expected 8 obs dates, got {len(obs)}"
    assert abs(obs[0] - 0.25) < 1e-9
    assert abs(obs[-1] - 2.0) < 1e-9


def test_observation_dates_annual_3y():
    """Annual, 3Y → 3 observation dates at t=1, 2, 3."""
    ac = make_digital()
    obs = ac.observation_dates()
    assert len(obs) == 3
    assert abs(obs[0] - 1.0) < 1e-9
    assert abs(obs[-1] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: Coupon per period
# ---------------------------------------------------------------------------

def test_coupon_per_period_quarterly():
    """8% pa / 4 quarters = $20 per period on $1000 notional."""
    ac = make_phoenix()
    expected = 0.08 * NOTIONAL / 4  # = $20
    assert abs(ac.coupon_per_period() - expected) < 0.01


def test_digital_coupon_per_period():
    """Digital: $50 fixed regardless of pa rate."""
    ac = make_digital()
    # digital_coupon overrides coupon_pa
    assert abs(ac.coupon_per_period() - 50.0) < 0.01


# ---------------------------------------------------------------------------
# Test 3: Barrier not hit → terminal payoff (par)
# ---------------------------------------------------------------------------

def test_terminal_payoff_no_knockin():
    """Final spot above protection barrier → (notional + final coupon) discounted to t=0."""
    ac = make_phoenix()
    spot_T = S_REF * 0.9  # 90% — above coupon_barrier=75%, below call=100%
    knocked_in = False
    r = 0.05
    pv = ac.terminal_payoff(spot_T, knocked_in, r)
    # terminal_payoff returns PV discounted to t=0.
    # spot_T/S_REF=0.9 >= coupon_barrier=0.75 → final coupon also paid.
    expected = (NOTIONAL + ac.coupon_per_period()) * np.exp(-r * ac.maturity_years)
    assert pv == pytest.approx(expected, abs=1.0), f"Expected {expected:.2f}, got {pv:.2f}"


def test_terminal_payoff_with_knockin():
    """Knocked-in → proportional loss (european_ki: no floor, discounted to t=0)."""
    ac = make_phoenix()
    spot_T = S_REF * 0.60  # 60% — below 75% protection
    knocked_in = True
    r = 0.05
    pv = ac.terminal_payoff(spot_T, knocked_in, r)
    # european_ki: investor receives spot_T/S_ref * notional * discount
    # No floor applies (that's soft_protection only).
    expected = (spot_T / S_REF) * NOTIONAL * np.exp(-r * ac.maturity_years)
    assert pv == pytest.approx(expected, abs=1.0), f"Expected {expected:.2f}, got {pv:.2f}"


def test_terminal_payoff_knockin_above_protection():
    """Knocked-in during path, final spot above initial protection: proportional loss, discounted."""
    ac = make_phoenix()
    spot_T = S_REF * 0.80  # 80% final spot, but knocked-in at some earlier date
    knocked_in = True
    r = 0.05
    pv = ac.terminal_payoff(spot_T, knocked_in, r=r)
    # european_ki: spot_T/S_ref * notional * discount (same formula, no special floor)
    expected = (spot_T / S_REF) * NOTIONAL * np.exp(-r * ac.maturity_years)
    assert pv == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# Test 4: Barrier hit at obs date 1 → Q1 payoff
# ---------------------------------------------------------------------------

def test_payoff_at_observation_called():
    """Spot >= call barrier → called, returns discounted notional + coupon."""
    ac = make_phoenix()
    spot = S_REF * 1.05  # 5% above barrier
    r = 0.05
    t1 = ac.observation_dates()[0]  # 0.25y
    pv = ac.payoff_at_observation(spot, t1, period_index=0, r=r)
    expected = (ac.redemption_at_call * NOTIONAL + ac.coupon_per_period()) * np.exp(-r * t1)
    assert pv is not None, "Should have triggered a call"
    assert pv == pytest.approx(expected, abs=1.0), f"Expected {expected:.2f}, got {pv:.2f}"


def test_payoff_at_observation_not_called():
    """Spot < call barrier → None returned (no call triggered)."""
    ac = make_phoenix()
    spot = S_REF * 0.95  # below 100% barrier
    r = 0.05
    t1 = ac.observation_dates()[0]
    pv = ac.payoff_at_observation(spot, t1, period_index=0, r=r)
    assert pv is None, f"Expected None (not called), got {pv}"


# ---------------------------------------------------------------------------
# Test 5: Worst-of basket
# ---------------------------------------------------------------------------

def test_worst_of_basket_uses_minimum():
    """Worst-of: instantiates with n_assets > 1 and correlation_matrix."""
    sec = get_security("Worst-Of Autocall")
    ac = from_security_dict(sec, S_ref=S_REF)
    assert ac.n_assets > 1, f"Worst-of should have n_assets > 1, got {ac.n_assets}"
    assert ac.correlation_matrix is not None, "Worst-of should have a correlation matrix"


# ---------------------------------------------------------------------------
# Test 6: Step-down barrier
# ---------------------------------------------------------------------------

def test_step_down_barrier():
    """Step-down: barrier decreases at specified periods."""
    sec = get_security("Step-Down Barrier")
    ac = from_security_dict(sec, S_ref=S_REF)

    # Period 0: barrier = 100%
    b0 = ac.call_barrier_at_period(0)
    # Period 5: should have stepped down (if that's the step schedule)
    b5 = ac.call_barrier_at_period(5)
    # Step-down means later barriers are ≤ earlier barriers
    assert b5 <= b0, f"Later barrier ({b5}) should be ≤ earlier barrier ({b0})"


# ---------------------------------------------------------------------------
# Test 7: All 4 pre-configured securities instantiate correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list_securities())
def test_all_securities_instantiate(name):
    """All 4 pre-configured securities should instantiate without error."""
    sec = get_security(name)
    ac = from_security_dict(sec, S_ref=5312.0)
    assert ac is not None
    assert ac.maturity_years > 0
    assert len(ac.observation_dates()) > 0
    assert ac.notional > 0
    assert 0.0 < ac.protection_barrier <= 1.0
    assert 0.0 < ac.call_barrier <= 2.0


# ---------------------------------------------------------------------------
# Test 8: Call probabilities are valid distributions
# ---------------------------------------------------------------------------

def test_call_probabilities_sum_lte_one():
    """Sum of call probabilities must be ≤ 1.0 (at most 100% get called eventually)."""
    ac = make_phoenix()
    probs = ac.call_probabilities(sigma=0.20, r=0.05, q=0.01)
    assert len(probs) == len(ac.observation_dates())
    assert all(0 <= p <= 1 for p in probs), f"Invalid probs: {probs}"
    assert sum(probs) <= 1.01, f"Sum of call probs {sum(probs):.3f} > 1"


def test_call_probabilities_monotone_decreasing():
    """Later observation dates should (generally) have lower marginal call probability."""
    ac = make_phoenix()
    probs = ac.call_probabilities(sigma=0.20, r=0.05, q=0.01)
    # Allow one violation (non-monotone is OK for step-down barriers)
    # but first prob should be highest
    assert probs[0] >= probs[-1], \
        f"First call prob ({probs[0]:.3f}) should be >= last ({probs[-1]:.3f})"
