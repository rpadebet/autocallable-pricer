"""
app/mc_standard.py
===================
Standard Monte Carlo pricer for autocallable structured products.

WHY THIS MODULE EXISTS:
    Monte Carlo is the industry-standard pricing method for path-dependent
    products like autocallables. The standard GBM simulation provides an
    intuitive baseline. Its price should match the FD pricer to within
    statistical error (a key demo assertion).

    This module is also the source of the path animation on Page 2: when
    return_paths=True, it returns the first 50 paths for visualization.
    Seeing the paths bounce around, trigger the barrier, or survive to
    maturity makes the pricing concept tangible.

ALGORITHM (Paper 3, §2, Eq. 2.3):
    For N independent paths:
        1. Simulate S at each observation date using GBM increments.
        2. At each observation: check call trigger, accumulate call payoff if triggered.
        3. If never called, compute terminal payoff (with knock-in check).
    Price = (1/N) * sum of discounted payoffs

VARIANCE REDUCTION — ANTITHETIC VARIATES:
    For each random Z ~ N(0,1), also simulate -Z. The two paths are
    negatively correlated so their average has lower variance.
    Halves the number of independent simulation runs needed for a given accuracy.
    Enabled by default (antithetic=True).

WEAKNESSES (WHY SURVIVAL MC EXISTS):
    The standard MC payoff is a discontinuous function of S0. If a path barely
    triggers the barrier, a tiny change in S0 flips the payoff from 'called' to
    'not called'. This discontinuity makes numerical Greeks (computed by bumping S0)
    noisy and unreliable — exactly what Paper 3 was written to fix.

PAPER REFERENCE:
    Alm, Harrach, Harrach, Keller (JCF 2013), §2, Eq. 2.3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from app.autocallable import AutoCallable


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """
    Container for Monte Carlo pricing results.

    Attributes:
        price:      Point estimate of the note's fair value.
        std_err:    Standard error of the price estimate (sigma / sqrt(N)).
        ci_low:     Lower 95% confidence interval bound.
        ci_high:    Upper 95% confidence interval bound.
        n_paths:    Number of simulation paths used.
        paths:      List of spot-price arrays (first 50 paths for visualization).
                    None unless price(return_paths=True) was called.
        call_times: For visualization: at which obs date (or None) each stored path called.
        convergence_series: List of (n_paths_so_far, price_estimate) tuples for convergence chart.
    """
    price: float
    std_err: float
    ci_low: float
    ci_high: float
    n_paths: int
    paths: Optional[list] = None
    call_times: Optional[list] = None  # obs date index or None (reached maturity)
    convergence_series: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# MCStandardPricer
# ---------------------------------------------------------------------------

class MCStandardPricer:
    """
    Standard GBM Monte Carlo pricer for autocallable notes.

    Uses geometric Brownian motion (constant vol) with optional antithetic
    variates for variance reduction. Prices a single-underlying autocallable.

    For basket (worst-of) autocallables, use the worst-of path: at each
    observation date, payoff depends on min(S_i/S0_i) across all assets.

    Args:
        autocallable: The AutoCallable product to price.
        sigma:         Flat implied volatility (annualized). For basket, ATM single-asset vol.
        r:             Risk-free rate (continuously compounded).
        q:             Dividend yield.
        n_paths:       Number of Monte Carlo paths. Default 10,000 for speed;
                       use 100,000+ for accuracy.
        seed:          Random seed for reproducibility (set to None for true randomness).
        antithetic:    Enable antithetic variates variance reduction (default True).
    """

    def __init__(
        self,
        autocallable: AutoCallable,
        sigma: float,
        r: float,
        q: float = 0.0,
        n_paths: int = 10_000,
        seed: Optional[int] = 42,
        antithetic: bool = True,
        spot_override: Optional[float] = None,
    ) -> None:
        # spot_override: if set, paths START here instead of S_ref.
        # Barriers remain at call_barrier * S_ref (trade-date reference).
        # WHY: enables proper Delta computation — bump current spot while
        # keeping knock-in/call barriers anchored to the original S_ref.
        self.ac = autocallable
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.spot_override = spot_override

        self.rng = np.random.default_rng(seed)
        self.obs_dates = autocallable.observation_dates()
        self.n_obs = len(self.obs_dates)
        self.S_ref = autocallable.S_ref
        # S0: current pricing spot (may differ from S_ref for seasoned notes)
        self.S0 = spot_override if spot_override is not None else self.S_ref

    def _simulate_paths(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate n GBM paths at all observation dates.

        Uses the exact GBM transition: log-normal increments between obs dates.
        Does NOT simulate every day — only at observation dates. This is exact
        for GBM under constant vol and much faster than daily stepping.

        WHY LOG-NORMAL INCREMENTS: Under GBM, log(S_{t+dt}/S_t) ~ Normal(mu*dt, sigma²*dt).
        Simulating only at obs dates avoids unnecessary computation while remaining exact.

        Args:
            n: Number of paths to simulate.

        Returns:
            Tuple of (S_paths, dts):
                S_paths: Array of shape (n, n_obs) with spot levels at each obs date.
                dts:     Array of time increments between consecutive obs dates.
        """
        # Time increments between consecutive obs dates
        # dts[0] = first obs date, dts[i] = obs_dates[i] - obs_dates[i-1]
        t_prev = np.array([0.0] + list(self.obs_dates[:-1]))
        dts = np.array(self.obs_dates) - t_prev

        # GBM drift and vol per time step
        mu = self.r - self.q  # risk-neutral drift
        drift = (mu - 0.5 * self.sigma ** 2) * dts        # shape (n_obs,)
        vol_dt = self.sigma * np.sqrt(dts)                  # shape (n_obs,)

        # Standard normal draws: shape (n, n_obs)
        # WHY DRAW ALL AT ONCE: Vectorization avoids Python loops over paths.
        Z = self.rng.standard_normal((n, self.n_obs))

        # Antithetic: stack [Z, -Z] so paths come in pairs
        if self.antithetic:
            Z = np.vstack([Z, -Z])  # shape (2n, n_obs)

        # Log-returns at each step
        log_returns = drift[np.newaxis, :] + vol_dt[np.newaxis, :] * Z

        # Cumulative log returns → spot levels
        log_S = np.cumsum(log_returns, axis=1)  # shape (n_paths, n_obs)
        S_paths = self.S0 * np.exp(log_S)   # paths start at S0 (current spot)

        return S_paths, dts

    def _price_paths(
        self,
        S_paths: np.ndarray,
        store_paths: bool = False,
        max_stored: int = 50,
    ) -> tuple[np.ndarray, Optional[list], Optional[list]]:
        """
        Compute discounted payoffs for each simulated path.

        For each path:
            - Check barrier at each observation date in sequence.
            - If called: payoff = (redemption + coupon) * exp(-r * t_call).
            - If never called: compute terminal payoff (with knock-in logic).

        Args:
            S_paths:     Shape (n_paths, n_obs) spot levels.
            store_paths: If True, save spot arrays for visualization (first max_stored paths).
            max_stored:  Maximum number of paths to return for animation.

        Returns:
            Tuple of (payoffs, stored_paths, call_indices):
                payoffs:      Shape (n_paths,) discounted payoff per path.
                stored_paths: List of spot arrays (including S0 prepended).
                call_indices: Obs index where each stored path called, or None.
        """
        n_paths, n_obs = S_paths.shape
        payoffs = np.zeros(n_paths)
        called = np.zeros(n_paths, dtype=bool)

        # Track knock-in: did the spot fall below protection_barrier at any obs?
        ki_barrier = self.ac.protection_barrier * self.S_ref
        knocked_in = np.zeros(n_paths, dtype=bool)

        # Storage for path animation (first max_stored paths)
        stored_paths = [] if store_paths else None
        call_indices = [] if store_paths else None

        for i, t_i in enumerate(self.obs_dates):
            S_i = S_paths[:, i]
            active = ~called  # paths not yet called

            # Knock-in tracking
            knocked_in |= (S_i < ki_barrier)

            # Check autocall condition for active paths
            barrier_i = self.ac.call_barrier_at_period(i) * self.S_ref
            triggered = active & (S_i >= barrier_i)

            # Payoff for triggered paths
            coupon_cond = S_i >= self.ac.coupon_barrier * self.S_ref
            redemption = self.ac.redemption_at_call * self.ac.notional
            coupon = np.where(coupon_cond, self.ac.coupon_per_period(), 0.0)

            payoffs[triggered] = (redemption + coupon[triggered]) * np.exp(-self.r * t_i)
            called |= triggered

        # Terminal payoff for paths that survived to maturity
        survived = ~called
        if survived.any():
            S_T = S_paths[survived, -1]
            ki_surv = knocked_in[survived]
            T = self.ac.maturity_years

            # Knocked-in: proportional loss; not knocked-in: par
            payoffs_surv = np.where(
                ki_surv,
                np.maximum(S_T / self.S_ref, self.ac.protection_floor) * self.ac.notional * np.exp(-self.r * T),
                (self.ac.notional + (self.ac.coupon_per_period() if self.ac.coupon_is_paid(S_T.mean()) else 0.0)) * np.exp(-self.r * T),
            )
            payoffs[survived] = payoffs_surv

        # Build path storage for animation
        if store_paths:
            n_store = min(max_stored, n_paths)
            for p in range(n_store):
                path_spots = np.concatenate([[self.S_ref], S_paths[p, :]])
                stored_paths.append(path_spots)
                # Find the first observation where this path called
                call_idx = None
                for i in range(n_obs):
                    if S_paths[p, i] >= self.ac.call_barrier_at_period(i) * self.S_ref:
                        call_idx = i
                        break
                call_indices.append(call_idx)

        return payoffs, stored_paths, call_indices

    def price(
        self,
        return_paths: bool = False,
        track_convergence: bool = False,
    ) -> MCResult:
        """
        Price the autocallable using standard Monte Carlo simulation.

        Args:
            return_paths:       If True, return first 50 paths for the Page 2
                                animation chart.
            track_convergence:  If True, compute price estimates at
                                [100, 500, 1000, 2000, 5000, 10000, ...]
                                paths for the convergence chart.

        Returns:
            MCResult with price, confidence interval, and optional paths/convergence.

        Edge cases:
            - With antithetic=True, actual paths simulated = n_paths * 2, but
              we estimate std_err using n_paths (conservative, slightly upward-biased SE).
            - Convergence tracking reuses the same random stream — early estimates
              are correlated with the final estimate (this is normal and expected).
        """
        # --- Convergence tracking ---
        conv_series = []
        if track_convergence:
            # Checkpoints at geometrically-spaced path counts
            base_n = self.n_paths
            checkpoints = []
            n = 100
            while n < base_n:
                checkpoints.append(n)
                n = int(n * 1.5)
            checkpoints.append(base_n)

            # Simulate full batch once, then compute running estimates
            S_paths, _ = self._simulate_paths(base_n)
            payoffs, stored_paths, call_idxs = self._price_paths(
                S_paths, store_paths=return_paths
            )

            for cp in checkpoints:
                cp_actual = min(cp, len(payoffs))
                sub = payoffs[:cp_actual]
                mu = sub.mean()
                se = sub.std(ddof=1) / np.sqrt(cp_actual)
                conv_series.append((cp_actual, float(mu), float(se)))
        else:
            # Normal pricing run
            S_paths, _ = self._simulate_paths(self.n_paths)
            payoffs, stored_paths, call_idxs = self._price_paths(
                S_paths, store_paths=return_paths
            )

        # --- Final price estimate ---
        price_est = float(payoffs.mean())
        std_err = float(payoffs.std(ddof=1) / np.sqrt(len(payoffs)))
        z95 = 1.96
        ci_low = price_est - z95 * std_err
        ci_high = price_est + z95 * std_err

        return MCResult(
            price=price_est,
            std_err=std_err,
            ci_low=ci_low,
            ci_high=ci_high,
            n_paths=len(payoffs),
            paths=stored_paths,
            call_times=call_idxs,
            convergence_series=conv_series,
        )
