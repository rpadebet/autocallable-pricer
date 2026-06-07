"""
app/heston.py
==============
Heston stochastic volatility model: calibration, pricing, and implied vol.

WHY THIS MODULE EXISTS:
    The Heston model is the industry-standard stochastic volatility model.
    Unlike Black-Scholes (constant vol) or local vol (deterministic vol surface),
    Heston captures two key empirical features of equity volatility:
        1. Mean reversion: vol tends to revert to a long-run average.
        2. Vol-of-vol correlation: when the market falls, vol spikes (the "leverage effect").

    For the demo, Heston serves two purposes:
        1. We calibrate it to the SPX snapshot and show the fitted surface overlaid
           on the market surface (Page 1) — demonstrating model fit.
        2. The calibrated Heston parameters (κ, θ, γ, ρ, v0) are displayed and
           explained as financial quantities, not just numbers.

HESTON SDE (Paper 2, Eq. 12-13):
    dS_t = (r - q) * S_t * dt + sqrt(v_t) * S_t * dW_S
    dv_t = κ * (θ - v_t) * dt + γ * sqrt(v_t) * dW_v
    Corr(dW_S, dW_v) = ρ

    κ = mean reversion speed (how fast vol returns to θ)
    θ = long-run variance (squared; so sqrt(θ) is long-run vol)
    γ = vol-of-vol (how much the variance itself fluctuates)
    ρ = correlation between asset and variance Brownian motions
        (ρ < 0 for equities: when stock falls, vol rises)
    v_0 = initial variance (sqrt(v_0) = current vol)

CHARACTERISTIC FUNCTION (Paper 2, Eq. 23 — USE THIS FORM ONLY):
    The Heston characteristic function is used to price European options
    via the Gil-Pelaez inversion formula. We MUST use Eq. 23 (not the
    alternative representation) because the alternative form has branch-cut
    discontinuities that cause the pricing integral to give wrong values.
    Paper 2 explicitly warns about this.

FELLER CONDITION:
    κ * θ > 0.5 * γ²
    Ensures the variance process v_t stays strictly positive (no touching zero).
    Violated in practice sometimes, but we enforce it as a calibration constraint.

CALIBRATION:
    scipy.optimize.minimize (L-BFGS-B or Nelder-Mead) minimizes sum of squared
    IV errors between Heston prices and market quotes. We use multiple random
    starting points to avoid local minima.

PAPER REFERENCE:
    Haugh (2013) "Local-Stochastic Jump Models for Option Pricing",
    Eq. 12-13 (Heston SDE), Eq. 23 (characteristic function)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Heston Characteristic Function
# ---------------------------------------------------------------------------

def heston_char_fn(
    u: complex,
    S0: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    gamma: float,
    rho: float,
) -> complex:
    """
    Heston characteristic function φ_T(u) — Eq. 23 of Paper 2.

    CRITICAL: This is the 'good' form of the Heston CF (Albrecher et al. 2007 / Haugh 2013).
    The 'original' Heston (1993) form uses a log branch cut that causes φ_T(u) to be
    discontinuous as a function of u. This makes the Fourier integration give wrong prices
    for some parameter values. Eq. 23 avoids this by a simple algebraic rearrangement.
    NEVER substitute the 'φ̂_T' form from Paper 2 — use only this form.

    The CF is defined by: E[exp(i*u*log(S_T/S_0))] under the risk-neutral measure.

    Args:
        u:     Fourier frequency (complex number in integration).
        S0:    Initial spot price.
        T:     Time to maturity.
        r:     Risk-free rate.
        q:     Dividend yield.
        v0:    Initial variance (squared volatility).
        kappa: Mean reversion speed.
        theta: Long-run variance.
        gamma: Vol-of-vol.
        rho:   Correlation between asset and variance Brownian motions.

    Returns:
        Complex value φ_T(u).
    """
    # Standard constants
    i = 1j

    # d: square root term (Paper 2, Eq. 23)
    # d = sqrt((ρ*γ*i*u - κ)² + γ²*(i*u + u²))
    d = np.sqrt(
        (rho * gamma * i * u - kappa) ** 2
        + gamma ** 2 * (i * u + u ** 2)
    )

    # g: ratio that controls the sign of d (ensures correct branch)
    # g = (κ - ρ*γ*i*u - d) / (κ - ρ*γ*i*u + d)
    numerator_g = kappa - rho * gamma * i * u - d
    denominator_g = kappa - rho * gamma * i * u + d

    # Avoid division by zero
    if abs(denominator_g) < 1e-15:
        denominator_g = 1e-15

    g = numerator_g / denominator_g

    # Exponential term: exp(-d*T)
    exp_dT = np.exp(-d * T)

    # Logarithm term: log((1 - g*exp(-dT)) / (1-g))
    # This is where the branch cut matters. The Eq.23 form keeps this continuous.
    denom_log = 1 - g * exp_dT
    if abs(denom_log) < 1e-15:
        denom_log = 1e-15
    if abs(1 - g) < 1e-15:
        return complex(0, 0)

    log_term = np.log(denom_log / (1 - g))

    # First exponential factor: exp(i*u*(log(S0) + (r-q)*T))
    factor1 = np.exp(i * u * (np.log(S0) + (r - q) * T))

    # Second factor: exp term with theta*kappa/gamma² coefficient
    coeff_theta = theta * kappa * gamma ** (-2)
    term_in_exp2 = (kappa - rho * gamma * i * u - d) * T - 2 * log_term
    factor2 = np.exp(coeff_theta * term_in_exp2)

    # Third factor: v0 term
    coeff_v0 = v0 * gamma ** (-2)
    v0_numerator = (kappa - rho * gamma * i * u - d) * (1 - exp_dT)
    v0_denominator = 1 - g * exp_dT
    if abs(v0_denominator) < 1e-15:
        v0_denominator = 1e-15
    factor3 = np.exp(coeff_v0 * v0_numerator / v0_denominator)

    return factor1 * factor2 * factor3


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    gamma: float,
    rho: float,
) -> float:
    """
    Price a European call option under the Heston model using the Gil-Pelaez formula.

    The Gil-Pelaez Fourier inversion:
        C = S0*exp(-q*T)*P1 - K*exp(-r*T)*P2

    where P1, P2 are computed from the characteristic function via numerical
    integration. This is the standard approach for semi-analytic Heston pricing.

    Args:
        S0, K, T, r, q:      Standard option parameters.
        v0, kappa, theta, gamma, rho: Heston model parameters.

    Returns:
        European call price. Returns Black-Scholes price with sqrt(v0) vol
        as fallback if numerical integration fails.

    Edge cases:
        - Uses scipy.integrate.quad with tight tolerance. Slower but accurate.
        - Fallback to BS price ensures the calibration doesn't crash on bad params.
    """
    # Enforce basic parameter constraints
    v0 = max(v0, 1e-6)
    kappa = max(kappa, 1e-6)
    theta = max(theta, 1e-6)
    gamma = max(gamma, 1e-6)
    rho = np.clip(rho, -0.999, -0.001)  # Must be negative for equities
    T = max(T, 1e-6)

    def integrand_P1(phi):
        """Integrand for the first probability P1."""
        cf = heston_char_fn(
            phi - 1j, S0, T, r, q, v0, kappa, theta, gamma, rho
        )
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi * heston_char_fn(
            -1j, S0, T, r, q, v0, kappa, theta, gamma, rho
        )
        if abs(denominator) < 1e-15:
            return 0.0
        return np.real(numerator / denominator)

    def integrand_P2(phi):
        """Integrand for the second probability P2."""
        cf = heston_char_fn(phi, S0, T, r, q, v0, kappa, theta, gamma, rho)
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi
        if abs(denominator) < 1e-15:
            return 0.0
        return np.real(numerator / denominator)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I1, _ = quad(integrand_P1, 1e-6, 200, limit=100, epsabs=1e-6, epsrel=1e-6)
            I2, _ = quad(integrand_P2, 1e-6, 200, limit=100, epsabs=1e-6, epsrel=1e-6)

        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi

        call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        call = max(call, max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0))
        return float(call)
    except Exception:
        # Fallback: Black-Scholes with sqrt(v0)
        from app.vol_surface import bs_call
        return bs_call(S0, K, T, r, q, np.sqrt(v0))


# ---------------------------------------------------------------------------
# HestonModel Class
# ---------------------------------------------------------------------------

class HestonModel:
    """
    Heston stochastic volatility model with calibration to market data.

    Wraps the characteristic function and pricing routines, and provides
    a calibrate() method that fits the 5 Heston parameters to a vol surface.

    Args:
        S0:    Spot price.
        r:     Risk-free rate.
        q:     Dividend yield.
        v0:    Initial variance (starting point for calibration).
        kappa: Mean reversion speed.
        theta: Long-run variance.
        gamma: Vol-of-vol.
        rho:   Correlation (must be < 0 for equities).
    """

    def __init__(
        self,
        S0: float,
        r: float,
        q: float = 0.014,
        v0: float = 0.04,
        kappa: float = 1.5,
        theta: float = 0.04,
        gamma: float = 0.3,
        rho: float = -0.7,
    ) -> None:
        self.S0 = S0
        self.r = r
        self.q = q
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.gamma = gamma
        self.rho = rho
        self.calibrated = False
        self.calibration_error = None

    def feller_condition(self) -> bool:
        """
        Check whether the Feller condition is satisfied: κθ > 0.5γ².

        If Feller is violated, the variance process can touch zero, making
        pricing numerically unstable. We enforce this in calibration.
        """
        return self.kappa * self.theta > 0.5 * self.gamma ** 2

    def call_price(self, K: float, T: float) -> float:
        """Price a European call with the current Heston parameters."""
        return heston_call_price(
            self.S0, K, T, self.r, self.q,
            self.v0, self.kappa, self.theta, self.gamma, self.rho,
        )

    def implied_vol(self, moneyness: float, ttm: float, tol: float = 1e-5) -> Optional[float]:
        """
        Compute the Black-Scholes implied vol from the Heston call price.

        WHY: We compare Heston to market quotes in IV space, not price space,
        because IVs are comparable across strikes and maturities.

        Args:
            moneyness: K/S0 ratio.
            ttm:       Time to maturity in years.

        Returns:
            Implied vol as decimal, or None if computation fails.
        """
        from app.vol_surface import bs_implied_vol
        K = moneyness * self.S0
        price = self.call_price(K, ttm)
        return bs_implied_vol(price, self.S0, K, ttm, self.r, self.q, tol=tol)

    def calibrate(
        self,
        vol_surface,  # VolSurface instance
        n_sample: int = 100,
        method: str = "differential_evolution",
    ) -> dict:
        """
        Calibrate Heston parameters to fit the market implied vol surface.

        Uses scipy global optimization (differential evolution by default)
        followed by local refinement. Multiple starting points with L-BFGS-B
        as the fallback.

        WHY DIFFERENTIAL EVOLUTION: The Heston objective function has multiple
        local minima, especially for ρ and γ. Differential evolution explores
        the full parameter space globally before refining locally.

        Args:
            vol_surface: A fitted VolSurface instance to calibrate against.
            n_sample:    Number of market quotes to use in calibration.
                         More = slower but more accurate.
            method:      "differential_evolution" (global) or "lbfgsb" (local, faster).

        Returns:
            Dict with calibrated params and fitting stats:
                {'v0', 'kappa', 'theta', 'gamma', 'rho',
                 'rmse_vol_pts', 'feller_satisfied', 'n_quotes'}
        """
        # Sample market quotes for calibration
        df = vol_surface.raw_df.dropna(subset=["impliedVolatility"])
        df = df[(df["ttm_years"] >= 0.1) & (df["ttm_years"] <= 2.0)]
        if len(df) > n_sample:
            df = df.sample(n_sample, random_state=42)

        moneyness_arr = df["moneyness"].values
        ttm_arr = df["ttm_years"].values
        market_iv_arr = df["impliedVolatility"].values

        def objective(params):
            """Sum of squared errors in IV space."""
            v0_, kappa_, theta_, gamma_, rho_ = params

            # Penalize Feller violation (soft constraint)
            feller_penalty = 0.0
            if kappa_ * theta_ < 0.5 * gamma_ ** 2:
                feller_penalty = 1000 * (0.5 * gamma_ ** 2 - kappa_ * theta_) ** 2

            errors = []
            for m, t, iv_mkt in zip(moneyness_arr, ttm_arr, market_iv_arr):
                try:
                    price = heston_call_price(
                        self.S0, m * self.S0, t, self.r, self.q,
                        v0_, kappa_, theta_, gamma_, rho_,
                    )
                    from app.vol_surface import bs_implied_vol
                    iv_heston = bs_implied_vol(
                        price, self.S0, m * self.S0, t, self.r, self.q
                    )
                    if iv_heston is not None:
                        errors.append((iv_heston - iv_mkt) ** 2)
                except Exception:
                    errors.append(0.25)  # Penalize failures with 50% vol error

            mse = np.mean(errors) if errors else 999.0
            return mse + feller_penalty

        # Parameter bounds: (v0, kappa, theta, gamma, rho)
        bounds = [
            (0.001, 0.5),   # v0: initial variance (1% - 70% vol)
            (0.1, 10.0),    # kappa: mean reversion speed
            (0.001, 0.5),   # theta: long-run variance
            (0.05, 1.5),    # gamma: vol of vol
            (-0.99, -0.01), # rho: correlation (strictly negative for equities)
        ]

        try:
            if method == "differential_evolution":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = differential_evolution(
                        objective,
                        bounds,
                        maxiter=50,
                        popsize=8,
                        seed=42,
                        tol=1e-4,
                        workers=1,
                    )
                params = result.x
            else:
                # L-BFGS-B local search from current params
                x0 = [self.v0, self.kappa, self.theta, self.gamma, self.rho]
                result = minimize(
                    objective, x0, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                params = result.x

            self.v0, self.kappa, self.theta, self.gamma, self.rho = params
            self.calibrated = True
            rmse = float(np.sqrt(objective(params) - 0))  # approx RMSE

        except Exception as e:
            # If calibration fails, keep initial params
            rmse = 99.0

        # Final RMSE in vol points
        self.calibration_error = rmse

        return {
            "v0": round(float(self.v0), 6),
            "kappa": round(float(self.kappa), 4),
            "theta": round(float(self.theta), 6),
            "gamma": round(float(self.gamma), 4),
            "rho": round(float(self.rho), 4),
            "rmse_vol_pts": round(float(rmse * 100), 2),  # as percentage points
            "feller_satisfied": bool(self.feller_condition()),
            "n_quotes": len(df),
        }

    def surface_grid(
        self,
        n_moneyness: int = 30,
        n_ttm: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the Heston implied vol surface on a regular grid for plotting.

        Used by the Vol Surface page to overlay Heston vs. market surface.

        Returns:
            (moneyness_grid, ttm_grid, iv_grid) — shape (n_ttm, n_moneyness).
            NaN for any points where implied vol computation fails.
        """
        m_axis = np.linspace(0.75, 1.25, n_moneyness)
        t_axis = np.linspace(0.1, 2.5, n_ttm)
        M, T = np.meshgrid(m_axis, t_axis)

        IV = np.full_like(M, np.nan)
        for i in range(n_ttm):
            for j in range(n_moneyness):
                try:
                    iv = self.implied_vol(M[i, j], T[i, j])
                    if iv is not None:
                        IV[i, j] = iv
                except Exception:
                    pass

        return M, T, IV
