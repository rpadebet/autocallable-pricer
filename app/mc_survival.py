"""
app/mc_survival.py
One-Step Survival Monte Carlo pricer (Alm et al. 2013, Algorithm 1).

WHY: Standard MC payoff is discontinuous at the barrier -> noisy Greeks.
Survival MC analytically handles barrier crossings -> smooth, stable Greeks.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from app.autocallable import AutoCallable
from app.mc_standard import MCResult


def _ncdf(z: float) -> float:
    """Standard normal CDF using math.erf (always available, no numpy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _nppf(p: float) -> float:
    """
    Standard normal quantile via rational approximation (Acklam 2010).
    Accurate to 1.15e-9 for p in (0, 1).
    """
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


class MCSurvivalPricer:
    """
    One-step survival MC pricer. Same interface as MCStandardPricer.

    Key innovation (Paper 3, Alm et al. 2013, Algorithm 1):
        - At each obs date j, compute p_j = P(S_{t+dt} < barrier | S_t = s)
          using the lognormal formula: Phi((log(B/s) - mu*dt) / (sigma*sqrt(dt)))
        - Add call payoff ANALYTICALLY: payoff += L * (1-p_j) * exp(-r*t) * Q
        - Sample ONLY from the truncated distribution below the barrier
        - Update likelihood: L *= p_j
    This makes the payoff a smooth function of S0 -> stable, reliable Greeks.

    Args:
        autocallable: Product to price.
        sigma: Flat implied vol.
        r:     Risk-free rate.
        q:     Dividend yield.
        n_paths: Number of MC paths.
        seed:  Random seed.
    """

    def __init__(self, autocallable: AutoCallable, sigma: float, r: float,
                 q: float = 0.0, n_paths: int = 10_000,
                 seed: Optional[int] = 42,
                 spot_override: Optional[float] = None) -> None:
        # spot_override: paths START here; barriers remain at call_barrier * S_ref.
        # WHY: enables proper Delta — bump current spot, keep barriers fixed.
        self.ac = autocallable
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)
        self.obs_dates = autocallable.observation_dates()
        self.S_ref = autocallable.S_ref
        self.S0 = spot_override if spot_override is not None else self.S_ref
        # Risk-neutral GBM drift: mu = r - q - sigma^2/2
        self._mu = r - q - 0.5 * sigma * sigma

    def _p_survive(self, s: float, barrier: float, dt: float) -> float:
        """
        P(S_{t+dt} < barrier | S_t = s) under lognormal GBM.

        Uses the exact formula: Phi((log(B/s) - mu*dt) / (sigma*sqrt(dt))).
        Valid for all s including s >= barrier (at initialization, call_barrier=1.0
        means s == barrier, giving p ~= 0.5 which is correct: 50/50 chance).
        """
        if dt < 1e-10:
            return 0.5
        z = (math.log(barrier / s) - self._mu * dt) / (self.sigma * math.sqrt(dt))
        return _ncdf(z)

    def _sample_below(self, s: float, p_j: float, dt: float) -> float:
        """Sample next spot from truncated normal BELOW the barrier."""
        u = float(self.rng.uniform(0.0, 1.0))
        u_t = float(np.clip(u * p_j, 1e-10, p_j - 1e-10))
        z = _nppf(u_t)
        return s * math.exp(self._mu * dt + self.sigma * math.sqrt(dt) * z)

    def _price_one_path(self, store: bool = False) -> tuple:
        """Run one-step survival algorithm for a single path."""
        s = float(self.S0)  # current pricing spot (may differ from S_ref for Delta bumps)
        L = 1.0
        payoff = 0.0
        spots = [s] if store else None
        first_call_idx = None
        t_prev = 0.0

        for i, t_i in enumerate(self.obs_dates):
            dt = t_i - t_prev
            t_prev = t_i
            barrier = self.ac.call_barrier_at_period(i) * self.S_ref

            # Survival probability via lognormal formula
            p_j = self._p_survive(s, barrier, dt)

            # Analytical call contribution from the (1-p_j) fraction that crosses
            call_pv = (self.ac.redemption_at_call * self.ac.notional
                       + self.ac.coupon_per_period())
            payoff += L * (1.0 - p_j) * math.exp(-self.r * t_i) * call_pv

            if store and first_call_idx is None and (1.0 - p_j) > 0.3:
                first_call_idx = i

            # Early termination if weight is negligible
            if p_j < 1e-10 or L < 1e-15:
                L = 0.0
                if store and spots is not None:
                    spots.append(s)
                break

            # Sample next spot from truncated distribution (always below barrier)
            s = self._sample_below(s, p_j, dt)
            L *= p_j
            if store and spots is not None:
                spots.append(s)

        # Terminal payoff (surviving weight)
        if L > 1e-15:
            T = self.ac.maturity_years
            ki = s < self.ac.protection_barrier * self.S_ref
            if ki:
                term_pv = (max(s / self.S_ref, self.ac.protection_floor)
                           * self.ac.notional * math.exp(-self.r * T))
            else:
                term_pv = self.ac.notional * math.exp(-self.r * T)
            payoff += L * term_pv

        return payoff, spots, first_call_idx

    def price(self, return_paths: bool = False,
              track_convergence: bool = False) -> MCResult:
        """
        Price using one-step survival MC. Same interface as MCStandardPricer.

        The 95% CI band is visibly narrower than standard MC at the same N
        (key demo point: variance reduction from eliminating barrier noise).
        """
        payoffs = []
        stored_paths = [] if return_paths else None
        call_indices = [] if return_paths else None
        conv_series = []

        checkpoints = set()
        if track_convergence:
            n = 100
            while n < self.n_paths:
                checkpoints.add(n)
                n = int(n * 1.5)
            checkpoints.add(self.n_paths)

        for idx in range(self.n_paths):
            do_store = return_paths and idx < 50
            pv, sp, ci = self._price_one_path(store=do_store)
            payoffs.append(pv)
            if do_store:
                stored_paths.append(sp)
                call_indices.append(ci)
            if track_convergence and (idx + 1) in checkpoints:
                arr = np.array(payoffs)
                se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                conv_series.append((len(arr), float(arr.mean()), se))

        arr = np.array(payoffs)
        price_est = float(arr.mean())
        n = len(arr)
        se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        return MCResult(
            price=price_est, std_err=se,
            ci_low=price_est - 1.96*se, ci_high=price_est + 1.96*se,
            n_paths=n, paths=stored_paths, call_times=call_indices,
            convergence_series=conv_series,
        )
