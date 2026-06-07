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

VOL MODELS:
    vol_model="flat"   (default) — constant sigma everywhere. Fast, exact jumps
                                   from one obs date to the next.
    vol_model="local"  — Dupire local vol surface. sigma = sigma_loc(S_t, t) at each
                         sub-step. Pre-computed on a grid; interpolated during simulation.
    vol_model="heston" — Heston stochastic vol. Simulates variance SDE alongside stock
                         using Euler-Maruyama with weekly sub-steps.
    vol_model="bates"  — Heston stochastic vol + Merton compound-Poisson jumps.
                         Both stochastic vol and fat tails.

PAPER REFERENCES:
    Alm, Harrach, Harrach, Keller (JCF 2013), §2, Eq. 2.3   (standard MC)
    Deng, Mallett, McCann (2011) §2.2                         (call probabilities)
    Haugh (2013) — Heston and Bates models                   (vol dynamics)
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
    GBM Monte Carlo pricer with optional vol-model switching.

    Supports four volatility models via the vol_model parameter:
        "flat"   — constant sigma (default, same as original). Simulates directly
                   from one obs date to the next using exact log-normal increments.
        "local"  — Dupire local vol: sigma(S,t) from a pre-computed surface grid.
                   Requires vol_surface argument (a fitted VolSurface instance).
        "heston" — Heston stochastic variance. Two coupled SDEs simulated with
                   Euler-Maruyama. Requires heston_params dict.
        "bates"  — Heston + Merton jumps. Requires heston_params + jump_params.

    All vol-aware models use weekly sub-stepping (n_steps_per_year=52) to keep
    discretisation error small. The flat model uses exact obs-date increments.

    Args:
        autocallable:    The AutoCallable product to price.
        sigma:           Flat implied vol (annualised). Required when vol_model="flat".
        r:               Risk-free rate (continuously compounded).
        q:               Dividend yield.
        n_paths:         Number of MC paths.
        seed:            RNG seed for reproducibility.
        antithetic:      Antithetic variates for flat vol only (default True).
        spot_override:   Start paths here instead of S_ref (for Delta bumps).
        vol_model:       "flat" | "local" | "heston" | "bates".
        vol_surface:     VolSurface instance. Required for vol_model="local".
        heston_params:   Dict with keys v0, kappa, theta, gamma, rho.
                         Required for vol_model="heston" or "bates".
        jump_params:     Dict with keys lam, mu_J, sig_J.
                         Required for vol_model="bates".
        n_steps_per_year: Sub-steps per year for vol-aware models (default 52).
    """

    def __init__(
        self,
        autocallable: AutoCallable,
        sigma: float = 0.20,
        r: float = 0.045,
        q: float = 0.0,
        n_paths: int = 10_000,
        seed: Optional[int] = 42,
        antithetic: bool = True,
        spot_override: Optional[float] = None,
        vol_model: str = "flat",
        vol_surface=None,
        heston_params: Optional[dict] = None,
        jump_params: Optional[dict] = None,
        n_steps_per_year: int = 52,
    ) -> None:
        self.ac = autocallable
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.spot_override = spot_override
        self.vol_model = vol_model
        self.vol_surface = vol_surface
        self.heston_params = heston_params or {}
        self.jump_params = jump_params or {}
        self.n_steps_per_year = n_steps_per_year

        self.rng = np.random.default_rng(seed)
        self.obs_dates = autocallable.observation_dates()
        self.n_obs = len(self.obs_dates)
        self.S_ref = autocallable.S_ref
        self.S0 = spot_override if spot_override is not None else self.S_ref

        # Pre-build local vol interpolator once (avoids rebuilding during price())
        # WHY PRE-BUILD: dupire_local_vol() is slow per-call; vectorised grid lookup
        # is ~100× faster for the 10K path × 52-step simulation.
        self._local_vol_interp = None
        if vol_model == "local" and vol_surface is not None:
            self._local_vol_interp = self._build_local_vol_interp(vol_surface)

    # -----------------------------------------------------------------------
    # Local vol interpolator
    # -----------------------------------------------------------------------

    def _build_local_vol_interp(self, vol_surface):
        """
        Pre-compute a Dupire local vol grid and build a fast interpolator.

        WHY GRID + INTERPOLATION: Calling vol_surface.dupire_local_vol() for
        every path at every sub-step would require ~500K Dupire evaluations per
        pricing run (10K paths × 52 steps/year × 1 year average). Each evaluation
        involves numerical differentiation of a spline — too slow.

        Instead we evaluate on a 40×25 (moneyness × time) grid once, then use
        scipy RegularGridInterpolator for fast bilinear lookup during simulation.

        Returns:
            Callable interp(pts) where pts.shape = (N, 2) as (t, moneyness).
            Output: local vol array of shape (N,). Extrapolation uses boundary values.
        """
        from scipy.interpolate import RegularGridInterpolator
        # Grid ranges: moneyness 0.40–1.80, time 0.01 to 1.1*maturity
        m_axis = np.linspace(0.40, 1.80, 40)
        max_t = self.ac.maturity_years + 0.05
        t_axis = np.linspace(0.01, max_t, 25)
        M, T_g = np.meshgrid(m_axis, t_axis)
        # Vectorised dupire evaluation on the grid
        LV = np.vectorize(lambda m, t: vol_surface.dupire_local_vol(m, t))(M, T_g)
        # RegularGridInterpolator: axes (t_axis, m_axis), values LV[t, m]
        interp = RegularGridInterpolator(
            (t_axis, m_axis), LV,
            method="linear", bounds_error=False, fill_value=None,
        )
        return interp

    def _query_local_vol(self, S_t: np.ndarray, t_sim: float) -> np.ndarray:
        """
        Query the pre-built local vol surface for all paths at time t_sim.

        Args:
            S_t:    Spot prices for all paths, shape (n_paths,).
            t_sim:  Current simulation time (years from inception).

        Returns:
            Local vol array, shape (n_paths,), clamped to [0.05, 1.0].
        """
        moneyness = np.clip(S_t / self.S0, 0.40, 1.80)
        t_clamped = np.clip(t_sim, 0.01, self.ac.maturity_years + 0.05)
        pts = np.stack([np.full(len(S_t), t_clamped), moneyness], axis=1)
        sigma_v = self._local_vol_interp(pts)
        return np.clip(sigma_v, 0.05, 1.0)

    # -----------------------------------------------------------------------
    # Flat vol simulation (original — exact obs-date jumps)
    # -----------------------------------------------------------------------

    def _simulate_paths(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate n flat-vol GBM paths at all observation dates.

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
        t_prev = np.array([0.0] + list(self.obs_dates[:-1]))
        dts = np.array(self.obs_dates) - t_prev

        mu = self.r - self.q
        drift = (mu - 0.5 * self.sigma ** 2) * dts
        vol_dt = self.sigma * np.sqrt(dts)

        Z = self.rng.standard_normal((n, self.n_obs))
        if self.antithetic:
            Z = np.vstack([Z, -Z])

        log_returns = drift[np.newaxis, :] + vol_dt[np.newaxis, :] * Z
        log_S = np.cumsum(log_returns, axis=1)
        S_paths = self.S0 * np.exp(log_S)
        return S_paths, dts

    # -----------------------------------------------------------------------
    # Vol-aware simulation (local vol, Heston, Bates) — weekly sub-stepping
    # -----------------------------------------------------------------------

    def _simulate_paths_vol_aware(self, n: int) -> np.ndarray:
        """
        Simulate n paths with sub-stepping for vol-aware models.

        Handles local vol, Heston, and Bates. All three use the same weekly
        sub-stepping framework; only the per-step update differs.

        WHY SUB-STEPPING: Euler-Maruyama for stochastic vol and local vol
        requires small time steps for accuracy. Weekly steps (dt ≈ 1/52 yr)
        keep discretisation error below ~0.5% for typical autocallable tenors.

        For Heston/Bates variance process we use the FULL TRUNCATION scheme:
            v_{t+dt} = max(v_t + kappa*(theta-v_t)*dt + gamma*sqrt(max(v_t,0))*sqrt(dt)*Z_v, 0)
        This avoids negative variance without bias. Broadie-Kaya is more accurate
        but far slower — unsuitable for a demo pricer.

        Args:
            n: Number of paths.

        Returns:
            S_paths: Array of shape (n, n_obs) with spot at each observation date.
        """
        # --- Heston/Bates parameter extraction ---
        is_heston = self.vol_model in ("heston", "bates")
        is_bates = self.vol_model == "bates"
        is_local = self.vol_model == "local"

        kappa = self.heston_params.get("kappa", 2.0)
        theta = self.heston_params.get("theta", 0.04)
        gamma = self.heston_params.get("gamma", 0.3)
        rho   = self.heston_params.get("rho", -0.7)
        v0    = self.heston_params.get("v0", 0.04)

        lam   = self.jump_params.get("lam", 0.0)
        mu_J  = self.jump_params.get("mu_J", -0.05)
        sig_J = self.jump_params.get("sig_J", 0.10)
        mu_bar_J = np.exp(mu_J + 0.5 * sig_J ** 2) - 1.0 if lam > 0 else 0.0

        # Risk-neutral drift (already corrected for jumps if Bates)
        r_adj = self.r - self.q - (lam * mu_bar_J if is_bates else 0.0)

        # Path state arrays
        log_S = np.zeros(n)          # log(S_t / S0)
        v_t   = np.full(n, v0)       # variance state (Heston/Bates)

        S_paths = np.zeros((n, self.n_obs))
        t_current = 0.0

        for obs_idx, t_obs in enumerate(self.obs_dates):
            dt_interval = t_obs - t_current
            n_sub = max(1, round(dt_interval * self.n_steps_per_year))
            dt_sub = dt_interval / n_sub

            for sub_step in range(n_sub):
                t_mid = t_current + (sub_step + 0.5) * dt_sub  # midpoint time

                if is_local:
                    # Local vol: query the pre-built Dupire surface
                    S_t = self.S0 * np.exp(log_S)
                    sigma_t = self._query_local_vol(S_t, t_mid)
                    Z = self.rng.standard_normal(n)
                    drift = (self.r - self.q - 0.5 * sigma_t ** 2) * dt_sub
                    log_S += drift + sigma_t * np.sqrt(dt_sub) * Z

                elif is_heston or is_bates:
                    # Heston/Bates: two correlated Brownian motions
                    v_plus = np.maximum(v_t, 0.0)    # floor variance at 0
                    sigma_t = np.sqrt(v_plus)

                    Z1 = self.rng.standard_normal(n)   # drives variance
                    Z2 = self.rng.standard_normal(n)   # independent component
                    Z_v = Z1
                    Z_S = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

                    # Variance update (full truncation)
                    v_t = (v_t
                           + kappa * (theta - v_t) * dt_sub
                           + gamma * sigma_t * np.sqrt(dt_sub) * Z_v)
                    v_t = np.maximum(v_t, 0.0)

                    # Log-stock update
                    log_S += ((r_adj - 0.5 * v_plus) * dt_sub
                              + sigma_t * np.sqrt(dt_sub) * Z_S)

                    # Bates: compound Poisson jumps
                    # Number of jumps per path in this sub-step
                    if is_bates and lam > 0:
                        N_jumps = self.rng.poisson(lam * dt_sub, size=n)
                        # Sum of N_jumps i.i.d. N(mu_J, sig_J²) rv's
                        # = N(N_jumps*mu_J, N_jumps*sig_J²) by convolution
                        # => log_jump = N_jumps*mu_J + sqrt(N_jumps)*sig_J*Z
                        Z_j = self.rng.standard_normal(n)
                        log_S += N_jumps * mu_J + np.sqrt(N_jumps) * sig_J * Z_j

            S_paths[:, obs_idx] = self.S0 * np.exp(log_S)
            t_current = t_obs

        return S_paths

    # -----------------------------------------------------------------------
    # Payoff computation (unchanged — works on any S_paths array)
    # -----------------------------------------------------------------------

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

        ki_barrier = self.ac.protection_barrier * self.S_ref
        knocked_in = np.zeros(n_paths, dtype=bool)

        stored_paths = [] if store_paths else None
        call_indices = [] if store_paths else None

        for i, t_i in enumerate(self.obs_dates):
            S_i = S_paths[:, i]
            active = ~called

            knocked_in |= (S_i < ki_barrier)

            barrier_i = self.ac.call_barrier_at_period(i) * self.S_ref
            triggered = active & (S_i >= barrier_i)

            coupon_cond = S_i >= self.ac.coupon_barrier * self.S_ref
            redemption = self.ac.redemption_at_call * self.ac.notional
            coupon = np.where(coupon_cond, self.ac.coupon_per_period(), 0.0)

            payoffs[triggered] = (redemption + coupon[triggered]) * np.exp(-self.r * t_i)
            called |= triggered

        survived = ~called
        if survived.any():
            S_T = S_paths[survived, -1]
            ki_surv = knocked_in[survived]
            T = self.ac.maturity_years

            payoffs_surv = np.where(
                ki_surv,
                np.maximum(S_T / self.S_ref, self.ac.protection_floor) * self.ac.notional * np.exp(-self.r * T),
                (self.ac.notional + (self.ac.coupon_per_period() if self.ac.coupon_is_paid(S_T.mean()) else 0.0)) * np.exp(-self.r * T),
            )
            payoffs[survived] = payoffs_surv

        if store_paths:
            n_store = min(max_stored, n_paths)
            for p in range(n_store):
                path_spots = np.concatenate([[self.S_ref], S_paths[p, :]])
                stored_paths.append(path_spots)
                call_idx = None
                for i in range(n_obs):
                    if S_paths[p, i] >= self.ac.call_barrier_at_period(i) * self.S_ref:
                        call_idx = i
                        break
                call_indices.append(call_idx)

        return payoffs, stored_paths, call_indices

    # -----------------------------------------------------------------------
    # Main pricing entry point
    # -----------------------------------------------------------------------

    def price(
        self,
        return_paths: bool = False,
        track_convergence: bool = False,
    ) -> MCResult:
        """
        Price the autocallable using standard Monte Carlo simulation.

        Dispatches to the correct simulation method based on vol_model:
            - "flat"   → _simulate_paths() (exact obs-date log-normal increments)
            - "local"  → _simulate_paths_vol_aware() (Dupire local vol, sub-stepped)
            - "heston" → _simulate_paths_vol_aware() (Heston SDE, sub-stepped)
            - "bates"  → _simulate_paths_vol_aware() (Heston + jumps, sub-stepped)

        Args:
            return_paths:       If True, return first 50 paths for animation.
            track_convergence:  If True, compute running estimates for convergence chart.

        Returns:
            MCResult with price, confidence interval, and optional paths/convergence.
        """
        conv_series = []

        if self.vol_model == "flat":
            # --- Flat vol path (original code, supports convergence + antithetic) ---
            if track_convergence:
                base_n = self.n_paths
                checkpoints = []
                n = 100
                while n < base_n:
                    checkpoints.append(n)
                    n = int(n * 1.5)
                checkpoints.append(base_n)

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
                S_paths, _ = self._simulate_paths(self.n_paths)
                payoffs, stored_paths, call_idxs = self._price_paths(
                    S_paths, store_paths=return_paths
                )

        else:
            # --- Vol-aware paths (local, heston, bates) ---
            # Note: antithetic not supported for vol-aware models (variance process
            # is path-dependent; simple sign flip of Z does not give antithetic paths).
            S_paths = self._simulate_paths_vol_aware(self.n_paths)
            payoffs, stored_paths, call_idxs = self._price_paths(
                S_paths, store_paths=return_paths
            )
            if track_convergence:
                n = 100
                while n <= len(payoffs):
                    sub = payoffs[:n]
                    se = sub.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                    conv_series.append((n, float(sub.mean()), float(se)))
                    n = int(n * 1.5)
                conv_series.append((len(payoffs), float(payoffs.mean()),
                                    float(payoffs.std(ddof=1) / np.sqrt(len(payoffs)))))

        price_est = float(payoffs.mean())
        std_err = float(payoffs.std(ddof=1) / np.sqrt(len(payoffs)))
        z95 = 1.96

        return MCResult(
            price=price_est,
            std_err=std_err,
            ci_low=price_est - z95 * std_err,
            ci_high=price_est + z95 * std_err,
            n_paths=len(payoffs),
            paths=stored_paths,
            call_times=call_idxs,
            convergence_series=conv_series,
        )
