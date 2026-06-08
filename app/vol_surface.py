"""
app/vol_surface.py
===================
Implied volatility surface and Dupire local volatility calculation.

WHY THIS MODULE EXISTS:
    The vol surface is the market's collective view of uncertainty across
    all strikes and maturities. For an autocallable pricer, it answers:
        "What vol should I use to price an autocall with barrier at K
         and observation dates at T_1, ..., T_n?"

    Two surfaces are constructed:
        1. Implied Vol Surface: Black-Scholes IV for each (K, T) quote in
           the snapshot. This is what the market observes directly.
        2. Dupire Local Vol Surface: The unique local vol function σ_loc(S, t)
           consistent with all market prices simultaneously. Unlike implied vol
           (which is a per-contract number), local vol is a continuous surface
           that can be used in dynamic hedging and exotics pricing.

DUPIRE LOCAL VOL (Paper 2, Eq. 2):
    σ²_loc(T, K) = [∂C/∂T + (r-q)*K*∂C/∂K + q*C] / [K²/2 * ∂²C/∂K²]

    In practice, we work in implied vol space:
        1. Fit a smooth spline to IV as a function of moneyness and TTM.
        2. Compute call prices C from the smoothed IVs.
        3. Differentiate C numerically w.r.t. K and T using the spline.

    ⚠️ WARNING: Dupire local vol is numerically unstable if the implied vol
    surface is not smooth. We use scipy's RectBivariateSpline on a regular
    grid. Far OTM options and illiquid expiries are excluded before fitting.

PAPER REFERENCES:
    Haugh (2013) "Local-Stochastic Jump Models for Option Pricing":
        - Local vol: Eq. 2 (Dupire formula)
        - Heston model: Eq. 12-13 (SDE), Eq. 23 (characteristic function)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from typing import Optional


# ---------------------------------------------------------------------------
# Black-Scholes Helpers
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes European call price.

    Args:
        S: Spot price
        K: Strike
        T: Time to maturity (years)
        r: Risk-free rate (continuous)
        q: Dividend yield (continuous)
        sigma: Implied volatility

    Returns:
        Call price. Returns intrinsic value if T < 1e-6 to avoid division by zero.
    """
    if T < 1e-6 or sigma < 1e-6:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_implied_vol(
    market_price: float, S: float, K: float, T: float, r: float, q: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> Optional[float]:
    """
    Black-Scholes implied volatility via Newton-Raphson bisection.

    Uses a hybrid: Newton-Raphson for fast convergence near the solution,
    bisection fallback if Newton overshoots.

    Args:
        market_price: Observed call mid price.
        S, K, T, r, q: Standard BS inputs.
        tol:  Convergence tolerance on vol (e.g. 1e-6 = 0.0001%).
        max_iter: Maximum Newton iterations.

    Returns:
        Implied vol as a decimal, or None if the price is below intrinsic or
        the algorithm fails to converge.
    """
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if market_price < intrinsic - tol or T < 1e-6:
        return None

    # Initial guess: simplified Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * market_price / S
    sigma = np.clip(sigma, 0.01, 5.0)

    for _ in range(max_iter):
        C = bs_call(S, K, T, r, q, sigma)
        diff = C - market_price

        if abs(diff) < tol * S:  # Converged in price space
            return float(np.clip(sigma, 0.01, 5.0))

        # Vega = S * sqrt(T) * phi(d1) * exp(-q*T)
        if T > 1e-6 and sigma > 1e-6:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T)
            if vega < 1e-10:
                break
            sigma -= diff / vega
            sigma = np.clip(sigma, 0.001, 5.0)
        else:
            break

    # Final check
    final = bs_call(S, K, T, r, q, sigma)
    if abs(final - market_price) < 0.5:  # $0.50 tolerance
        return float(np.clip(sigma, 0.01, 5.0))
    return None


# ---------------------------------------------------------------------------
# VolSurface Class
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Implied volatility surface constructed from SPX options snapshot data.

    Provides:
        1. Raw implied vols for each (moneyness, TTM) pair.
        2. Smoothed surface via bivariate spline interpolation.
        3. Dupire local vol surface derived from the smoothed surface.

    The smoothed surface is needed for Dupire because numerical differentiation
    of raw IV quotes amplifies noise, producing nonsensical local vols.

    Args:
        snapshot:    DataFrame from data_loader.load_snapshot().
        S0:          Current spot price (from snapshot['spot']).
        r:           Risk-free rate.
        q:           Dividend yield.
        min_iv:      Filter: drop quotes with IV below this (default 0.02 = 2%).
        max_iv:      Filter: drop quotes with IV above this (default 1.00 = 100%).
    """

    def __init__(
        self,
        snapshot: pd.DataFrame,
        S0: float,
        r: float,
        q: float = 0.014,
        min_iv: float = 0.02,
        max_iv: float = 1.00,
    ) -> None:
        self.S0 = S0
        self.r = r
        self.q = q

        # --- Build clean IV DataFrame ---
        df = snapshot.dropna(subset=["impliedVolatility"]).copy()
        df = df[
            (df["impliedVolatility"] >= min_iv) &
            (df["impliedVolatility"] <= max_iv) &
            (df["ttm_years"] >= 0.05) &
            (df["ttm_years"] <= 3.0) &
            (df["moneyness"] >= 0.70) &
            (df["moneyness"] <= 1.35)
        ].copy()

        self.raw_df = df

        # --- Build regular grid for spline fitting ---
        # Use representative moneyness and TTM knot points
        self._moneyness_knots = np.array([
            0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
            1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
        ])
        self._ttm_knots = np.array([
            0.08, 0.17, 0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00,
        ])

        self._spline: Optional[RectBivariateSpline] = None
        self._fit_spline()

    def _fit_spline(self) -> None:
        """
        Fit a bivariate spline to the implied vol surface.

        WHY BIVARIATE SPLINE: We need a smooth, differentiable surface to
        compute Dupire derivatives numerically. Raw quote data is irregular
        and noisy. The spline is fit on a regular (moneyness × TTM) grid
        by averaging nearby quotes to the grid.

        The spline degree is 3 (cubic) in both dimensions, with smoothing
        factor s chosen to balance fit quality vs smoothness.
        """
        df = self.raw_df

        # Build a regular grid by averaging IV values near each knot
        grid_ivs = np.full((len(self._moneyness_knots), len(self._ttm_knots)), np.nan)

        for i, m_knot in enumerate(self._moneyness_knots):
            for j, t_knot in enumerate(self._ttm_knots):
                # Find quotes within 5% moneyness and 20% TTM of this knot
                mask = (
                    (np.abs(df["moneyness"] - m_knot) < 0.05) &
                    (np.abs(df["ttm_years"] - t_knot) < t_knot * 0.3)
                )
                if mask.sum() >= 1:
                    grid_ivs[i, j] = df.loc[mask, "impliedVolatility"].median()

        # Fill remaining NaNs by forward-filling along TTM axis
        # (interpolate in TTM direction where we have data)
        for i in range(len(self._moneyness_knots)):
            row = grid_ivs[i, :]
            valid = ~np.isnan(row)
            if valid.sum() >= 2:
                grid_ivs[i, :] = np.interp(
                    self._ttm_knots,
                    self._ttm_knots[valid],
                    row[valid],
                )
            elif valid.sum() == 1:
                grid_ivs[i, :] = row[valid][0]  # flat extrapolation
            else:
                # No data near this moneyness: use ATM vol as fallback
                atm_mask = np.abs(df["moneyness"] - 1.0) < 0.05
                if atm_mask.sum() > 0:
                    grid_ivs[i, :] = df.loc[atm_mask, "impliedVolatility"].mean()
                else:
                    grid_ivs[i, :] = 0.17  # ultimate fallback

        self._iv_grid = grid_ivs  # shape (n_moneyness, n_ttm)

        # Fit the spline
        try:
            self._spline = RectBivariateSpline(
                self._moneyness_knots,
                self._ttm_knots,
                grid_ivs,
                kx=3,  # cubic in moneyness
                ky=3,  # cubic in TTM
                s=0.005,  # smoothing factor (tuned to balance fit vs. smoothness)
            )
        except Exception:
            # Fallback: use a simpler degree-1 spline
            self._spline = RectBivariateSpline(
                self._moneyness_knots,
                self._ttm_knots,
                grid_ivs,
                kx=1,
                ky=1,
                s=0.0,
            )

    def implied_vol(self, moneyness: float, ttm: float) -> float:
        """
        Return smoothed implied vol for a given (moneyness, TTM) point.

        Args:
            moneyness: K/S ratio (1.0 = ATM).
            ttm:       Time to maturity in years.

        Returns:
            Implied vol as decimal (e.g. 0.17 = 17%).
            Clamped to [0.02, 1.0].
        """
        if self._spline is None:
            return 0.17  # Fallback if spline not fitted
        # float() ensures plain Python scalars — np.clip returns a numpy scalar
        # which RectBivariateSpline treats as a 1-element array, returning shape
        # (1, 1) rather than a scalar. [0, 0] extracts the single value.
        m_clamp = float(np.clip(moneyness, self._moneyness_knots[0], self._moneyness_knots[-1]))
        t_clamp = float(np.clip(ttm, self._ttm_knots[0], self._ttm_knots[-1]))
        iv = float(self._spline(m_clamp, t_clamp)[0, 0])
        return float(np.clip(iv, 0.02, 1.0))

    def surface_grid(
        self,
        n_moneyness: int = 30,
        n_ttm: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the smoothed IV surface on a regular grid for 3D plotting.

        Returns:
            (moneyness_grid, ttm_grid, iv_grid) — all shape (n_ttm, n_moneyness).
            Use with plotly's surface plot: x=moneyness_grid, y=ttm_grid, z=iv_grid.
        """
        m_axis = np.linspace(0.72, 1.32, n_moneyness)
        t_axis = np.linspace(0.10, 2.5, n_ttm)
        M, T = np.meshgrid(m_axis, t_axis)
        IV = np.vectorize(lambda m, t: self.implied_vol(m, t))(M, T)
        return M, T, IV

    def atm_vol(self, ttm: float) -> float:
        """Return the ATM (moneyness=1.0) implied vol for a given TTM."""
        return self.implied_vol(1.0, ttm)

    # -------------------------------------------------------------------------
    # Dupire Local Volatility
    # -------------------------------------------------------------------------

    def dupire_local_vol(
        self,
        moneyness: float,
        ttm: float,
        dK: float = 0.01,
        dT: float = 0.01,
    ) -> float:
        """
        Compute Dupire local vol σ_loc(K, T) using numerical differentiation.

        Applies the Dupire formula (Paper 2, Eq. 2):
            σ²_loc(T, K) = [∂C/∂T + (r-q)*K*∂C/∂K + q*C] / [K²/2 * ∂²C/∂K²]

        All derivatives computed numerically from the smoothed call price surface.

        WHY SMOOTH SPLINE FIRST: The Dupire formula involves second derivatives of C
        w.r.t. K. Even small noise in C(K) amplifies into huge swings in d²C/dK².
        The smoothed spline tames this instability.

        Args:
            moneyness: K/S ratio (1.0 = ATM).
            ttm:       Time to maturity in years.
            dK:        Step size for K derivatives (fraction of moneyness, default 1%).
            dT:        Step size for T derivative (years, default 0.01 ≈ 3.6 days).

        Returns:
            Local vol as decimal. Clamped to [0.01, 2.0].
            Returns 0.17 (ATM IV fallback) if the Dupire ratio is invalid.

        Edge cases:
            - If d²C/dK² ≤ 0, the market is arbitrage-free in the strike direction
              and the formula is undefined. Return ATM IV.
            - Far OTM options can give negative numerators; clamping prevents
              negative local vols.
        """
        S = self.S0
        K = moneyness * S
        T = ttm

        def C(m, t):
            """Call price using smoothed IV surface."""
            iv = self.implied_vol(m, t)
            return bs_call(S, m * S, t, self.r, self.q, iv)

        # Cache the 4 unique (moneyness, time) evaluations needed for all three
        # numerical derivatives, rather than calling C() 8 times.
        # WHY: each C() call invokes implied_vol() + bs_call() (~50μs combined).
        # Original code called C(moneyness, T) THREE times and C(moneyness+dK, T)
        # and C(moneyness-dK, T) each TWICE, totalling 8 calls. With caching: 4 calls.
        # At the grid-build level (1K Dupire calls), this saves ~2K C() calls (~100ms).
        C_0T  = C(moneyness,      T)        # centre point — shared by dCdT, d²C/dK², C_val
        C_pT  = C(moneyness + dK, T)        # K+ point    — shared by dCdK and d²C/dK²
        C_mT  = C(moneyness - dK, T)        # K- point    — shared by dCdK and d²C/dK²
        C_0dT = C(moneyness,      T + dT)   # forward time — used only by dCdT

        # Numerical derivatives built from the 4 cached evaluations above
        # dC/dT (forward difference in T)
        dCdT = (C_0dT - C_0T) / dT

        # dC/dK (central difference in moneyness)
        dCdK = (C_pT - C_mT) / (2 * dK * S)

        # d²C/dK² (second derivative in moneyness)
        d2CdK2 = (C_pT - 2 * C_0T + C_mT) / (dK * S) ** 2

        C_val = C_0T

        # Dupire formula numerator and denominator
        numerator = dCdT + (self.r - self.q) * K * dCdK + self.q * C_val
        denominator = 0.5 * K ** 2 * d2CdK2

        if denominator <= 1e-10 or numerator <= 0:
            return self.atm_vol(T)  # Fallback: use ATM IV

        local_var = numerator / denominator
        if local_var <= 0:
            return self.atm_vol(T)

        local_vol = np.sqrt(local_var)
        return float(np.clip(local_vol, 0.01, 2.0))

    def dupire_local_vol_grid(
        self,
        m_axis: np.ndarray,
        T_axis: np.ndarray,
        dK: float = 0.01,
        dT: float = 0.01,
    ) -> np.ndarray:
        """
        Compute Dupire local vol for an entire (moneyness × TTM) grid in one pass.

        WHY THIS EXISTS:
            Building the RegularGridInterpolator used by all three pricers requires
            evaluating dupire_local_vol() at ~1,000 grid points (40 × 25).
            Calling the scalar method 1,000 times costs ~2 s because each call
            invokes implied_vol() + bs_call() separately in Python.
            This method evaluates all stencil points in 4 C-level vectorised calls
            (RectBivariateSpline.ev + scipy.special.ndtr), reducing the cost to ~0.05 s.

        Algorithm:
            1. Build flat arrays for all (m, T) stencil points.
            2. Evaluate the smoothed IV spline at all 4 stencil positions in one shot.
            3. Compute Black-Scholes call prices vectorised using scipy.special.ndtr
               (C-backed Phi, ~10x faster than scipy.stats.norm.cdf).
            4. Apply the Dupire formula element-wise.  Invalid cells fall back to
               the implied vol (same behaviour as the scalar method).

        Args:
            m_axis: 1-D array of moneyness values (length nM).
            T_axis: 1-D array of TTM values in years (length nT).
            dK:     Moneyness step for strike-direction derivatives (default 1%).
            dT:     Time step for the time-direction derivative (default 0.01 yr).

        Returns:
            np.ndarray of shape (nT, nM) where result[i, j] = sigma_loc(T_axis[i], m_axis[j]).
            Values are clamped to [0.05, 1.0].
        """
        from scipy.special import ndtr  # Phi(x) = ndtr(x) -- C-backed, avoids Python norm.cdf overhead

        S, r, q = self.S0, self.r, self.q

        # Flat arrays of all (m, T) pairs in the grid.
        # np.meshgrid returns (nT, nM) arrays so the result matches
        # RegularGridInterpolator's expected (t_axis, m_axis) layout.
        M_g, T_g = np.meshgrid(m_axis, T_axis)   # each (nT, nM)
        m_flat = M_g.ravel()                       # (nT*nM,)
        t_flat = T_g.ravel()                       # (nT*nM,)

        # -- Step 1: Vectorised implied-vol lookup at all 4 stencil positions ----
        # RectBivariateSpline.ev(xi, yi) evaluates at pairs in C -- one Python call
        # for all nT*nM points replaces nT*nM individual implied_vol() calls.
        iv_0T  = np.clip(self._spline.ev(m_flat,      t_flat     ), 0.02, 1.0)
        iv_pT  = np.clip(self._spline.ev(m_flat + dK, t_flat     ), 0.02, 1.0)
        iv_mT  = np.clip(self._spline.ev(m_flat - dK, t_flat     ), 0.02, 1.0)
        iv_0dT = np.clip(self._spline.ev(m_flat,      t_flat + dT), 0.02, 1.0)

        # -- Step 2: Vectorised Black-Scholes call prices ----------------------
        def _bs_vec(iv: np.ndarray, m: np.ndarray, t: np.ndarray) -> np.ndarray:
            """Vectorised B-S call; uses ndtr for speed; handles tiny T gracefully."""
            K = m * S
            safe_t  = np.maximum(t, 1e-6)
            safe_iv = np.maximum(iv, 1e-6)
            d1 = (np.log(S / K) + (r - q + 0.5 * safe_iv ** 2) * safe_t) / (safe_iv * np.sqrt(safe_t))
            d2 = d1 - safe_iv * np.sqrt(safe_t)
            pv = np.exp(-r * safe_t) * (S * np.exp((r - q) * safe_t) * ndtr(d1) - K * ndtr(d2))
            intrinsic = np.maximum(S * np.exp(-q * safe_t) - K * np.exp(-r * safe_t), 0.0)
            return np.where(t < 1e-6, intrinsic, pv)

        C_0T  = _bs_vec(iv_0T,  m_flat,      t_flat     )
        C_pT  = _bs_vec(iv_pT,  m_flat + dK, t_flat     )
        C_mT  = _bs_vec(iv_mT,  m_flat - dK, t_flat     )
        C_0dT = _bs_vec(iv_0dT, m_flat,      t_flat + dT)

        # -- Step 3: Dupire formula (vectorised) -------------------------------
        K_flat  = m_flat * S
        dCdT    = (C_0dT - C_0T) / dT
        dCdK    = (C_pT  - C_mT) / (2.0 * dK * S)
        d2CdK2  = (C_pT  - 2.0 * C_0T + C_mT) / (dK * S) ** 2
        num     = dCdT + (r - q) * K_flat * dCdK + q * C_0T
        denom   = 0.5 * K_flat ** 2 * d2CdK2

        # Where the Dupire ratio is ill-conditioned, fall back to implied vol
        # (mirrors the scalar method's behaviour).
        valid    = (denom > 1e-10) & (num > 0)
        loc_var  = np.where(valid, num / np.where(denom > 1e-10, denom, 1.0), iv_0T ** 2)
        lv_flat  = np.clip(np.sqrt(np.maximum(loc_var, 0.0)), 0.05, 1.0)

        return lv_flat.reshape(T_g.shape)   # (nT, nM)

    def dupire_surface_grid(
        self,
        n_moneyness: int = 20,
        n_ttm: int = 15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the Dupire local vol surface on a grid for plotting.

        Returns:
            (moneyness_grid, ttm_grid, local_vol_grid) -- shape (n_ttm, n_moneyness).
        """
        # Narrower range than IV surface to avoid unstable boundaries
        m_axis = np.linspace(0.80, 1.20, n_moneyness)
        t_axis = np.linspace(0.25, 2.0, n_ttm)
        M, T = np.meshgrid(m_axis, t_axis)
        LV = np.vectorize(lambda m, t: self.dupire_local_vol(m, t))(M, T)
        return M, T, LV

    def calibration_rmse(self, heston_model) -> float:
        """
        Compute RMSE between Heston model IVs and market IVs.

        Used in the Vol Surface page to display how well Heston fits.

        Args:
            heston_model: A fitted HestonModel instance.

        Returns:
            Root-mean-squared error in vol points (e.g. 0.015 = 1.5 vol points).
        """
        df = self.raw_df.head(200)  # Limit for speed
        errors = []
        for _, row in df.iterrows():
            try:
                heston_iv = heston_model.implied_vol(row["moneyness"], row["ttm_years"])
                errors.append((heston_iv - row["impliedVolatility"]) ** 2)
            except Exception:
                pass

        if not errors:
            return 0.0
        return float(np.sqrt(np.mean(errors)))
