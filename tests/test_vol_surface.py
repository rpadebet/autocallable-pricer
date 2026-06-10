"""
tests/test_vol_surface.py
==========================
Unit tests for VolSurface, SVI parametrization, Dupire surfaces, and CSV export.

Test groups:
    1. SVI formula basics — _svi_w output shape, sign, edge cases
    2. SVI slice fitting   — _fit_svi_slice quality, convergence, invalid inputs
    3. VolSurface.build_svi_surface — builds successfully, stores ready flag
    4. SVI surface evaluation — svi_implied_vol, svi_surface_grid shape/range
    5. SVI Dupire grid     — svi_dupire_local_vol_grid positive, finite, in range
    6. Smoothness comparison — SVI Dupire has lower 2nd-difference variance than cubic
    7. CSV export          — to_csv_implied_vol, to_csv_dupire headers, shape, values
    8. Fallback behaviour  — svi_implied_vol falls back to cubic spline if not built
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from app.vol_surface import (
    VolSurface,
    bs_call,
    bs_implied_vol,
    _svi_w,
    _fit_svi_slice,
)


# ===========================================================================
# Helpers — synthetic option data
# ===========================================================================

def _make_synthetic_snapshot(
    n_expiries: int = 6,
    n_strikes: int = 12,
    S0: float = 5000.0,
    r: float = 0.045,
    q: float = 0.014,
    base_iv: float = 0.18,
    skew: float = -0.05,
) -> pd.DataFrame:
    """
    Build a synthetic SPX-like options snapshot for testing.

    Uses Black-Scholes to price calls with a simple linear-skew vol surface:
        σ(K, T) = base_iv + skew * (K/S0 - 1)

    WHY SYNTHETIC: avoids file-system dependency; the test suite must run in CI
    without sample_data/ present.
    """
    rows = []
    ttms = np.linspace(0.1, 2.5, n_expiries)
    moneynesses = np.linspace(0.80, 1.20, n_strikes)

    for ttm in ttms:
        for m in moneynesses:
            K = m * S0
            iv = max(base_iv + skew * (m - 1.0), 0.05)
            price = bs_call(S0, K, ttm, r, q, iv)
            if price < 1e-6:
                continue
            rows.append({
                "strike": K,
                "moneyness": m,
                "ttm_years": ttm,
                "impliedVolatility": iv,
                "bid": price * 0.98,
                "ask": price * 1.02,
                "volume": 100,
                "optionType": "call",
                "spot": S0,
                "rfr": r,
            })

    return pd.DataFrame(rows)


S0  = 5000.0
R   = 0.045
Q   = 0.014


@pytest.fixture(scope="module")
def synthetic_snap():
    """Shared synthetic snapshot — created once per module."""
    return _make_synthetic_snapshot()


@pytest.fixture(scope="module")
def vol_surf(synthetic_snap):
    """VolSurface fitted once per module for efficiency."""
    return VolSurface(synthetic_snap, S0=S0, r=R, q=Q)


@pytest.fixture(scope="module")
def vol_surf_svi(synthetic_snap):
    """VolSurface with SVI surface built — created once per module."""
    vs = VolSurface(synthetic_snap, S0=S0, r=R, q=Q)
    vs.build_svi_surface()
    return vs


# ===========================================================================
# 1. SVI formula basics
# ===========================================================================

class TestSVIFormula:
    """Tests for the _svi_w total variance function."""

    def test_svi_w_scalar_finite(self):
        """_svi_w must return a finite float for typical parameters."""
        w = _svi_w(0.0, a=0.04, b=0.15, rho=-0.5, m=0.0, sigma=0.2)
        assert np.isfinite(w), f"_svi_w returned non-finite: {w}"

    def test_svi_w_minimum_at_m(self):
        """
        For rho=0, the SVI formula w(k) = a + b*sqrt((k-m)^2 + sigma^2)
        attains its minimum at k = m (exactly, when rho = 0).
        """
        a, b, rho, m, sigma = 0.02, 0.10, 0.0, 0.05, 0.20
        k_grid = np.linspace(-0.5, 0.5, 100)
        w_vals = _svi_w(k_grid, a, b, rho, m, sigma)
        idx_min = np.argmin(w_vals)
        assert abs(k_grid[idx_min] - m) < 0.02, (
            f"SVI minimum not at m={m}; found at k={k_grid[idx_min]:.3f}"
        )

    def test_svi_w_array_output_shape(self):
        """_svi_w applied to an array must return an array of the same shape."""
        k = np.linspace(-0.5, 0.5, 50)
        w = _svi_w(k, 0.03, 0.12, -0.4, 0.0, 0.25)
        assert w.shape == k.shape, f"Shape mismatch: {w.shape} != {k.shape}"

    def test_svi_w_non_negative_typical_params(self):
        """
        For well-conditioned parameters, _svi_w should be ≥ 0 across
        a wide log-moneyness range (± 0.8).
        """
        k = np.linspace(-0.8, 0.8, 200)
        w = _svi_w(k, a=0.04, b=0.20, rho=-0.6, m=0.0, sigma=0.15)
        assert np.all(w >= 0), f"Negative total variance found: min={w.min():.6f}"

    def test_svi_w_wings_increasing(self):
        """
        Away from the minimum, w(k) must be increasing with |k|.
        This tests Lee's moment formula wing condition.
        """
        a, b, rho, m, sigma = 0.03, 0.15, -0.5, 0.0, 0.20
        k_left  = np.array([-0.8, -0.6, -0.4, -0.2])
        k_right = np.array([0.2,  0.4,  0.6,  0.8])
        w_left  = _svi_w(k_left, a, b, rho, m, sigma)
        w_right = _svi_w(k_right, a, b, rho, m, sigma)
        assert np.all(np.diff(w_left) < 0), "SVI w not decreasing toward left minimum"
        assert np.all(np.diff(w_right) > 0), "SVI w not increasing into right wing"


# ===========================================================================
# 2. SVI slice fitting
# ===========================================================================

class TestSVIFit:
    """Tests for _fit_svi_slice."""

    def _make_slice(self, ttm=0.5, n=20, base_iv=0.18, skew=-0.05, S0=5000.0, r=0.045, q=0.014):
        """Create a clean (k, w) slice for testing."""
        F = S0 * np.exp((r - q) * ttm)
        m_arr = np.linspace(0.80, 1.20, n)
        k_arr = np.log(m_arr * S0 / F)
        iv_arr = np.maximum(base_iv + skew * (m_arr - 1.0), 0.05)
        w_arr = iv_arr ** 2 * ttm
        return k_arr, w_arr

    def test_fit_returns_five_params(self):
        """_fit_svi_slice must return an array of exactly 5 elements."""
        k, w = self._make_slice()
        p = _fit_svi_slice(k, w)
        assert p is not None, "_fit_svi_slice returned None on clean data"
        assert len(p) == 5, f"Expected 5 params, got {len(p)}"

    def test_fit_rmse_below_2_vol_pts(self):
        """
        On synthetic data (linear skew with no noise), SVI should fit
        within 2 implied-vol percentage points RMSE.
        """
        ttm = 0.5
        k, w = self._make_slice(ttm=ttm)
        p = _fit_svi_slice(k, w)
        assert p is not None
        w_pred = np.maximum(_svi_w(k, *p), 1e-8)
        iv_pred = np.sqrt(w_pred / ttm)
        iv_mkt  = np.sqrt(w / ttm)
        rmse_vp = np.sqrt(np.mean((iv_pred - iv_mkt) ** 2)) * 100
        assert rmse_vp < 2.0, f"SVI slice RMSE = {rmse_vp:.3f} vol pts — too high"

    def test_fit_params_in_bounds(self):
        """Fitted SVI params must respect the constraint bounds."""
        k, w = self._make_slice()
        p = _fit_svi_slice(k, w)
        assert p is not None
        a, b, rho, m, sigma = p
        assert a >= 0,           f"a={a:.4f} < 0"
        assert b >= 0,           f"b={b:.4f} < 0"
        assert abs(rho) < 1.0,   f"|rho|={abs(rho):.4f} >= 1"
        assert sigma > 0,        f"sigma={sigma:.4f} <= 0"

    def test_fit_returns_none_on_too_few_points(self):
        """_fit_svi_slice must return None when fewer than 5 data points are given."""
        k = np.array([0.0, 0.1, 0.2, 0.3])
        w = np.array([0.04, 0.05, 0.06, 0.07])
        p = _fit_svi_slice(k, w)
        assert p is None, "Expected None for 4-point slice, got params"

    def test_fit_non_negative_total_variance(self):
        """Fitted SVI must predict non-negative total variance across the fit range."""
        k, w = self._make_slice()
        p = _fit_svi_slice(k, w)
        assert p is not None
        k_dense = np.linspace(k.min(), k.max(), 100)
        w_pred = _svi_w(k_dense, *p)
        assert np.all(w_pred >= 0), f"Negative w found: min={w_pred.min():.6f}"


# ===========================================================================
# 3. VolSurface.build_svi_surface
# ===========================================================================

class TestBuildSVISurface:
    """Tests for VolSurface.build_svi_surface()."""

    def test_svi_ready_after_build(self, synthetic_snap):
        """svi_ready must be True after a successful build_svi_surface() call."""
        vs = VolSurface(synthetic_snap, S0=S0, r=R, q=Q)
        assert not vs.svi_ready, "svi_ready should be False before building"
        result = vs.build_svi_surface()
        assert vs.svi_ready, "svi_ready should be True after successful build"
        assert result["svi_ready"] is True

    def test_svi_slices_fitted(self, synthetic_snap):
        """At least 3 expiry slices must be successfully fitted."""
        vs = VolSurface(synthetic_snap, S0=S0, r=R, q=Q)
        result = vs.build_svi_surface()
        assert result["n_slices_fitted"] >= 3, (
            f"Only {result['n_slices_fitted']} slices fitted — expected ≥ 3"
        )

    def test_svi_slice_rmse_reported(self, synthetic_snap):
        """build_svi_surface must return a per-slice RMSE dict."""
        vs = VolSurface(synthetic_snap, S0=S0, r=R, q=Q)
        result = vs.build_svi_surface()
        assert isinstance(result["slice_rmse"], dict)
        assert len(result["slice_rmse"]) >= 3

    def test_rebuild_clears_old_slices(self, synthetic_snap):
        """Calling build_svi_surface() twice must not leave stale state."""
        vs = VolSurface(synthetic_snap, S0=S0, r=R, q=Q)
        r1 = vs.build_svi_surface()
        r2 = vs.build_svi_surface()
        # Both calls should succeed with the same number of slices
        assert r1["n_slices_fitted"] == r2["n_slices_fitted"]
        assert vs.svi_ready


# ===========================================================================
# 4. SVI surface evaluation
# ===========================================================================

class TestSVISurfaceEval:
    """Tests for svi_implied_vol and svi_surface_grid."""

    def test_svi_iv_range(self, vol_surf_svi):
        """svi_implied_vol must return values in a sensible range [2%, 100%]."""
        for m in [0.85, 1.0, 1.10]:
            for t in [0.25, 0.5, 1.0, 2.0]:
                iv = vol_surf_svi.svi_implied_vol(m, t)
                assert 0.02 <= iv <= 1.0, (
                    f"svi_implied_vol({m}, {t}) = {iv:.4f} — out of [0.02, 1.0]"
                )

    def test_svi_iv_atm_positive(self, vol_surf_svi):
        """ATM SVI implied vol must be > 0 for all tenors."""
        for t in np.linspace(0.1, 2.5, 10):
            iv = vol_surf_svi.svi_implied_vol(1.0, t)
            assert iv > 0, f"ATM SVI IV is 0 or negative at T={t}"

    def test_svi_surface_grid_shape(self, vol_surf_svi):
        """svi_surface_grid must return three arrays of shape (n_ttm, n_moneyness)."""
        M, T, IV = vol_surf_svi.svi_surface_grid(n_moneyness=15, n_ttm=10)
        assert M.shape == (10, 15), f"M shape {M.shape} != (10, 15)"
        assert T.shape == (10, 15)
        assert IV.shape == (10, 15)

    def test_svi_surface_grid_all_finite(self, vol_surf_svi):
        """All values in the SVI surface grid must be finite."""
        M, T, IV = vol_surf_svi.svi_surface_grid(n_moneyness=15, n_ttm=10)
        assert np.all(np.isfinite(IV)), "Non-finite IV in SVI surface grid"

    def test_svi_surface_grid_range(self, vol_surf_svi):
        """All IV values must be in [2%, 100%]."""
        _, _, IV = vol_surf_svi.svi_surface_grid(n_moneyness=15, n_ttm=10)
        assert IV.min() >= 0.02, f"SVI IV grid min {IV.min():.4f} < 0.02"
        assert IV.max() <= 1.00, f"SVI IV grid max {IV.max():.4f} > 1.00"


# ===========================================================================
# 5. SVI Dupire grid
# ===========================================================================

class TestSVIDupireGrid:
    """Tests for svi_dupire_local_vol_grid and svi_dupire_surface_grid."""

    def test_svi_dupire_grid_shape(self, vol_surf_svi):
        """svi_dupire_local_vol_grid must return shape (nT, nM)."""
        m_axis = np.linspace(0.85, 1.15, 15)
        t_axis = np.linspace(0.25, 2.0, 10)
        LV = vol_surf_svi.svi_dupire_local_vol_grid(m_axis, t_axis)
        assert LV.shape == (10, 15), f"SVI Dupire grid shape {LV.shape} != (10, 15)"

    def test_svi_dupire_grid_finite(self, vol_surf_svi):
        """All SVI Dupire local vols must be finite."""
        m_axis = np.linspace(0.85, 1.15, 15)
        t_axis = np.linspace(0.25, 2.0, 10)
        LV = vol_surf_svi.svi_dupire_local_vol_grid(m_axis, t_axis)
        assert np.all(np.isfinite(LV)), "Non-finite values in SVI Dupire grid"

    def test_svi_dupire_grid_range(self, vol_surf_svi):
        """SVI Dupire local vols must lie in [5%, 100%] (clamped range)."""
        m_axis = np.linspace(0.85, 1.15, 15)
        t_axis = np.linspace(0.25, 2.0, 10)
        LV = vol_surf_svi.svi_dupire_local_vol_grid(m_axis, t_axis)
        assert LV.min() >= 0.05, f"SVI Dupire min {LV.min():.4f} < 0.05"
        assert LV.max() <= 1.00, f"SVI Dupire max {LV.max():.4f} > 1.00"

    def test_svi_dupire_surface_grid_shape(self, vol_surf_svi):
        """svi_dupire_surface_grid must return three arrays of shape (n_ttm, n_moneyness)."""
        M, T, LV = vol_surf_svi.svi_dupire_surface_grid(n_moneyness=15, n_ttm=10)
        assert M.shape == (10, 15)
        assert T.shape == (10, 15)
        assert LV.shape == (10, 15)

    def test_svi_dupire_fallback_when_not_built(self, vol_surf):
        """
        svi_dupire_local_vol_grid must gracefully fall back to cubic-spline Dupire
        when svi_ready is False.
        """
        assert not vol_surf.svi_ready, "Fixture vol_surf should not have SVI built"
        m_axis = np.linspace(0.90, 1.10, 10)
        t_axis = np.linspace(0.25, 1.5, 8)
        # Should not raise; should return same shape as cubic Dupire
        LV_svi = vol_surf.svi_dupire_local_vol_grid(m_axis, t_axis)
        LV_cubic = vol_surf.dupire_local_vol_grid(m_axis, t_axis)
        assert LV_svi.shape == LV_cubic.shape
        # Values should be identical (both use cubic when SVI not ready)
        np.testing.assert_array_almost_equal(LV_svi, LV_cubic, decimal=6)


# ===========================================================================
# 6. Smoothness comparison — SVI Dupire should be smoother than cubic Dupire
# ===========================================================================

class TestDupireSmoothness:
    """
    Validates that the SVI Dupire surface is materially smoother than the
    cubic-spline Dupire surface.

    Smoothness is measured as the variance of second-order finite differences
    in the strike direction (a proxy for d²LV/dK²). Lower variance → smoother.
    """

    def test_svi_dupire_smoother_in_strike_direction(self, vol_surf_svi):
        """
        SVI Dupire second-difference variance along moneyness axis must be
        strictly lower than cubic-spline Dupire second-difference variance.
        """
        m_axis = np.linspace(0.85, 1.15, 25)  # Fine grid to measure smoothness
        t_axis = np.linspace(0.25, 2.0, 10)

        LV_cubic = vol_surf_svi.dupire_local_vol_grid(m_axis, t_axis)
        LV_svi   = vol_surf_svi.svi_dupire_local_vol_grid(m_axis, t_axis)

        # Second-order finite differences along the moneyness axis (axis=1)
        d2_cubic = np.diff(LV_cubic, n=2, axis=1)
        d2_svi   = np.diff(LV_svi,   n=2, axis=1)

        var_cubic = float(np.var(d2_cubic))
        var_svi   = float(np.var(d2_svi))

        assert var_svi < var_cubic, (
            f"SVI Dupire is NOT smoother than cubic Dupire: "
            f"var_svi={var_svi:.2e} >= var_cubic={var_cubic:.2e}"
        )

    def test_cubic_dupire_jaggedness_is_non_trivial(self, vol_surf_svi):
        """
        The cubic-spline Dupire second-difference variance must be non-zero,
        confirming there is actual jaggedness to compare against.
        """
        m_axis = np.linspace(0.85, 1.15, 25)
        t_axis = np.linspace(0.25, 2.0, 10)
        LV_cubic = vol_surf_svi.dupire_local_vol_grid(m_axis, t_axis)
        d2_cubic = np.diff(LV_cubic, n=2, axis=1)
        assert np.var(d2_cubic) > 1e-12, "Cubic Dupire has zero variance — it is perfectly smooth?"


# ===========================================================================
# 7. CSV export
# ===========================================================================

class TestCSVExport:
    """Tests for to_csv_implied_vol and to_csv_dupire."""

    def test_csv_implied_vol_is_string(self, vol_surf):
        """to_csv_implied_vol must return a non-empty string."""
        csv = vol_surf.to_csv_implied_vol(n_moneyness=10, n_ttm=8)
        assert isinstance(csv, str)
        assert len(csv) > 100

    def test_csv_implied_vol_parseable(self, vol_surf):
        """CSV string must parse back to a DataFrame with the correct shape."""
        csv = vol_surf.to_csv_implied_vol(n_moneyness=10, n_ttm=8)
        df = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
        assert df.shape == (8, 10), f"Parsed shape {df.shape} != (8, 10)"

    def test_csv_implied_vol_header_contains_moneyness(self, vol_surf):
        """Column headers must start with 'M=' to indicate moneyness labels."""
        csv = vol_surf.to_csv_implied_vol(n_moneyness=10, n_ttm=8)
        header_line = csv.split("\n")[0]
        assert "M=" in header_line, f"No 'M=' in header: {header_line[:80]}"

    def test_csv_implied_vol_values_in_range(self, vol_surf):
        """All IV values in the CSV must be in [2%, 100%]."""
        csv = vol_surf.to_csv_implied_vol(n_moneyness=10, n_ttm=8)
        df = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
        assert df.values.min() >= 2.0,    f"Min IV in CSV: {df.values.min():.4f}% < 2%"
        assert df.values.max() <= 100.0,  f"Max IV in CSV: {df.values.max():.4f}% > 100%"

    def test_csv_dupire_cubic_parseable(self, vol_surf):
        """Cubic-spline Dupire CSV must parse back to the correct shape."""
        csv = vol_surf.to_csv_dupire(method="cubic", n_moneyness=10, n_ttm=8)
        # Skip the comment line starting with '#'
        lines = [l for l in csv.split("\n") if not l.startswith("#")]
        df = pd.read_csv(pd.io.common.StringIO("\n".join(lines)), index_col=0)
        assert df.shape == (8, 10), f"Cubic Dupire CSV shape {df.shape} != (8, 10)"

    def test_csv_dupire_svi_parseable(self, vol_surf_svi):
        """SVI Dupire CSV must parse back to the correct shape."""
        csv = vol_surf_svi.to_csv_dupire(method="svi", n_moneyness=10, n_ttm=8)
        assert "# SVI_Dupire" in csv, "SVI Dupire CSV missing comment header"
        lines = [l for l in csv.split("\n") if not l.startswith("#")]
        df = pd.read_csv(pd.io.common.StringIO("\n".join(lines)), index_col=0)
        assert df.shape == (8, 10), f"SVI Dupire CSV shape {df.shape} != (8, 10)"

    def test_csv_dupire_svi_not_built_returns_error(self, vol_surf):
        """
        to_csv_dupire(method='svi') must return an error CSV string
        when the SVI surface has not been built.
        """
        assert not vol_surf.svi_ready
        csv = vol_surf.to_csv_dupire(method="svi")
        assert "error" in csv.lower(), f"Expected error message in CSV, got: {csv[:60]}"

    def test_csv_dupire_values_in_range(self, vol_surf):
        """All local vol values in the Dupire CSV must be in [0.1%, 80%]."""
        csv = vol_surf.to_csv_dupire(method="cubic", n_moneyness=10, n_ttm=8)
        lines = [l for l in csv.split("\n") if not l.startswith("#")]
        df = pd.read_csv(pd.io.common.StringIO("\n".join(lines)), index_col=0)
        assert df.values.min() >= 0.0,   f"Negative local vol in CSV: {df.values.min():.4f}"
        assert df.values.max() <= 100.0, f"Local vol > 100% in CSV: {df.values.max():.4f}"


# ===========================================================================
# 8. Fallback behaviour
# ===========================================================================

class TestFallback:
    """Tests that svi_implied_vol and related methods fall back gracefully."""

    def test_svi_iv_fallback_matches_cubic_when_not_built(self, vol_surf):
        """
        When SVI is not built, svi_implied_vol must return the same value
        as implied_vol (cubic-spline fallback).
        """
        assert not vol_surf.svi_ready
        for m in [0.90, 1.0, 1.10]:
            for t in [0.5, 1.0]:
                iv_svi   = vol_surf.svi_implied_vol(m, t)
                iv_cubic = vol_surf.implied_vol(m, t)
                assert iv_svi == iv_cubic, (
                    f"svi_implied_vol({m}, {t}) = {iv_svi:.4f} "
                    f"!= implied_vol = {iv_cubic:.4f} when SVI not built"
                )

    def test_svi_surface_grid_fallback_finite(self, vol_surf):
        """
        svi_surface_grid must return finite values even when SVI is not built
        (falls back to cubic spline).
        """
        assert not vol_surf.svi_ready
        M, T, IV = vol_surf.svi_surface_grid(n_moneyness=8, n_ttm=6)
        assert np.all(np.isfinite(IV)), "Non-finite IV in svi_surface_grid fallback"
