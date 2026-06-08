"""
app/mc_survival.py
One-Step Survival Monte Carlo pricer (Alm et al. 2013, Algorithm 1).

WHY: Standard MC payoff is discontinuous at the barrier -> noisy Greeks.
Survival MC analytically handles barrier crossings -> smooth, stable Greeks.

VOL MODELS (same as mc_standard.py):
    vol_model="flat"   — original behaviour, constant sigma. Survival prob uses sigma.
    vol_model="local"  — Dupire local vol: sigma_t = vol_surface.local_vol(S_t, t).
                         Evaluated at each observation date (no sub-stepping needed:
                         survival MC is already obs-date-by-obs-date).
    vol_model="heston" — Heston variance process. Advance v_t alongside s at each obs step
                         using n_sub Euler-Maruyama sub-steps. Use sigma_t = sqrt(v_t) in p_j.
    vol_model="bates"  — Heston + Merton jumps. Jump contribution to p_j approximated
                         analytically (conditional on the Brownian component).
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
    One-step survival MC pricer with optional vol-model switching.

    Same survival-MC algorithm as before (Alm et al. 2013) but extended to
    support local vol, Heston, and Bates dynamics.

    HOW VOL MODELS INTERACT WITH SURVIVAL MC:
        "flat"   — sigma is constant. p_j formula uses self._sigma.
        "local"  — At each obs date, look up sigma_t = local_vol(S_t, t) then use
                   it in the p_j formula. The truncated-normal sampling also uses sigma_t.
        "heston" — Advance the variance state v_t between obs dates using Euler-Maruyama
                   sub-steps. At each obs date, sigma_t = sqrt(v_t). Use sigma_t in p_j.
                   WHY CORRECT: Conditioning on v_t, the stock is locally lognormal with
                   vol sqrt(v_t), so the survival probability formula remains exact given
                   the realised variance.
        "bates"  — Heston variance + Merton jumps. Jump contribution to the survival
                   probability is approximated: p_j ≈ p_continuous × P(no call from jump).
                   P(no call from jump) = exp(-lam * (1-p_jump_call) * dt) where
                   p_jump_call = P(S * exp(jump) >= barrier). This is an approximation
                   that is accurate when jump intensity lam is small (< 2/year).

    Args:
        autocallable:    Product to price.
        sigma:           Flat vol. Required for vol_model="flat".
        r, q:            Risk-free rate, dividend yield.
        n_paths:         MC path count.
        seed:            RNG seed.
        spot_override:   Start spot for Delta bumps (barriers remain at S_ref).
        vol_model:       "flat" | "local" | "heston" | "bates".
        vol_surface:     VolSurface instance (required for "local").
        heston_params:   Dict: v0, kappa, theta, gamma, rho (required for "heston"/"bates").
        jump_params:     Dict: lam, mu_J, sig_J (required for "bates").
        n_steps_per_year: Sub-steps per year for Heston variance advancement (default 52).
    """

    def __init__(self, autocallable: AutoCallable, sigma: float = 0.20, r: float = 0.045,
                 q: float = 0.0, n_paths: int = 10_000,
                 seed: Optional[int] = 42,
                 spot_override: Optional[float] = None,
                 vol_model: str = "flat",
                 vol_surface=None,
                 heston_params: Optional[dict] = None,
                 jump_params: Optional[dict] = None,
                 n_steps_per_year: int = 52) -> None:
        self.ac = autocallable
        self.sigma = sigma
        self.r = r
        self.q = q
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)
        self.obs_dates = autocallable.observation_dates()
        self.S_ref = autocallable.S_ref
        self.S0 = spot_override if spot_override is not None else self.S_ref
        self.vol_model = vol_model
        self.vol_surface = vol_surface
        self.heston_params = heston_params or {}
        self.jump_params = jump_params or {}
        self.n_steps_per_year = n_steps_per_year

        # Risk-neutral GBM drift for flat vol
        self._mu = r - q - 0.5 * sigma * sigma

        # Pre-build local vol interpolator once (same pattern as MCStandardPricer).
        # WHY: dupire_local_vol() costs ~300μs per call (8 C() evaluations with scipy).
        # For N=10K paths × 8 obs dates = 80K calls → ~30s of Python overhead.
        # Building a 40×25 grid once (~1K calls, ~0.3s) then using the C-backed
        # RegularGridInterpolator cuts simulation cost to ~0.5s total.
        self._local_vol_interp = None
        if vol_model == "local" and vol_surface is not None:
            self._local_vol_interp = self._build_local_vol_interp(vol_surface)

    # -----------------------------------------------------------------------
    # Local vol grid builder (mirrors MCStandardPricer._build_local_vol_interp)
    # -----------------------------------------------------------------------

    def _build_local_vol_interp(self, vol_surface):
        """
        Pre-compute a Dupire local vol grid and return a fast RegularGridInterpolator.

        WHY THIS EXISTS HERE TOO:
            MCSurvivalPricer calls _get_sigma_local() once per path per obs date.
            For N=10K paths × 8 obs dates that is 80,000 dupire_local_vol() calls
            at ~300μs each ≈ 30s. Building a 40×25 (moneyness × time) grid once
            (~1,000 calls, ~0.3s) and switching to RegularGridInterpolator lookups
            reduces the simulation cost to ~0.5s — a ~60× speedup.

            The grid dimensions and range are identical to MCStandardPricer so
            both pricers share the same surface representation.

        Returns:
            RegularGridInterpolator keyed on (t_axis, m_axis) → local vol.
            Scalar query: interp([[t, m]]) returns array of shape (1,).
            Extrapolation uses boundary values (fill_value=None).
        """
        from scipy.interpolate import RegularGridInterpolator
        m_axis = np.linspace(0.40, 1.80, 40)
        max_t = self.ac.maturity_years + 0.05
        t_axis = np.linspace(0.01, max_t, 25)
        M, T_g = np.meshgrid(m_axis, t_axis)
        # np.vectorize is still a Python loop, but runs only once at init time.
        # 25×40 = 1,000 Dupire evaluations total.
        LV = np.vectorize(lambda m, t: vol_surface.dupire_local_vol(m, t))(M, T_g)
        interp = RegularGridInterpolator(
            (t_axis, m_axis), LV,
            method="linear", bounds_error=False, fill_value=None,
        )
        return interp

    # -----------------------------------------------------------------------
    # Sigma helpers (dispatch based on vol_model)
    # -----------------------------------------------------------------------

    def _get_sigma_local(self, s: float, t: float) -> float:
        """
        Return local vol sigma(s, t) from the Dupire surface.

        WHY: Local vol captures the strike/term structure of market implied vols.
             Using it here makes the survival MC consistent with the calibrated surface.

        Fast path (preferred): uses the pre-built RegularGridInterpolator built in
        __init__. One scipy C-extension call, ~5μs.

        Slow path (fallback): calls dupire_local_vol() directly, ~300μs. Only reached
        if vol_surface was not provided at construction time.
        """
        moneyness = float(np.clip(s / self.S0, 0.40, 1.80))
        t_clamp = float(np.clip(t, 0.01, self.ac.maturity_years + 0.05))
        if self._local_vol_interp is not None:
            # Fast path: pre-built grid → single C-backed interpolation call
            pts = np.array([[t_clamp, moneyness]])
            return float(np.clip(self._local_vol_interp(pts)[0], 0.05, 1.0))
        # Slow fallback (vol_surface provided but no interp built — should not occur
        # under normal usage since __init__ builds the interp whenever vol_model="local")
        if self.vol_surface is None:
            return self.sigma
        return float(np.clip(
            self.vol_surface.dupire_local_vol(moneyness, t_clamp), 0.05, 1.0
        ))

    def _advance_heston_variance(self, v_t: float, dt: float) -> float:
        """
        Advance the Heston variance v_t over interval dt using Euler-Maruyama
        sub-steps. Returns the new variance state.

        WHY SUB-STEPS: A single Euler-Maruyama step over 0.25 years (quarterly obs)
        introduces significant bias in the variance process (CIR process has mean
        reversion that interacts with the diffusion term). Weekly sub-steps (dt/13
        ≈ 0.02yr) keep discretisation error below 0.5%.

        Uses FULL TRUNCATION: negative variance is reflected at zero. This is the
        standard numerically stable approach (Lord et al. 2010).
        """
        kappa = self.heston_params.get("kappa", 2.0)
        theta = self.heston_params.get("theta", 0.04)
        gamma = self.heston_params.get("gamma", 0.3)
        n_sub = max(1, round(dt * self.n_steps_per_year))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            v_plus = max(v_t, 0.0)
            Z = float(self.rng.standard_normal())
            v_t = v_t + kappa * (theta - v_t) * dt_sub + gamma * math.sqrt(v_plus * dt_sub) * Z
            v_t = max(v_t, 0.0)
        return v_t

    def _jump_survival_correction(self, s: float, barrier: float, dt: float) -> float:
        """
        Approximate the reduction in survival probability due to Merton jumps.

        P(no autocall from jump in [t, t+dt]) ≈ exp(-lam * dt * p_jump_call)
        where p_jump_call = P(s * exp(log_jump) >= barrier)
             = P(log_jump >= log(barrier/s))
             = 1 - Phi((log(barrier/s) - mu_J) / sig_J)

        This is an approximation assuming at most one jump per sub-interval.
        Valid when lam*dt is small (< 0.1). For lam=1, dt=0.25: lam*dt=0.25 →
        small enough that P(>=2 jumps in dt) < 3%.

        Returns the multiplicative correction factor in (0, 1].
        """
        lam = self.jump_params.get("lam", 0.0)
        mu_J = self.jump_params.get("mu_J", -0.05)
        sig_J = self.jump_params.get("sig_J", 0.10)
        if lam < 1e-10 or sig_J < 1e-10:
            return 1.0
        log_ratio = math.log(barrier / s) if barrier > 0 and s > 0 else 0.0
        p_jump_call = 1.0 - _ncdf((log_ratio - mu_J) / sig_J)
        correction = math.exp(-lam * dt * p_jump_call)
        return float(np.clip(correction, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Core single-path algorithm
    # -----------------------------------------------------------------------

    def _p_survive(self, s: float, barrier: float, dt: float, sigma_t: float) -> float:
        """
        P(S_{t+dt} < barrier | S_t = s) under lognormal GBM with instantaneous vol sigma_t.

        Formula: Phi((log(barrier/s) - mu_t*dt) / (sigma_t*sqrt(dt)))
        where mu_t = r - q - sigma_t²/2  (risk-neutral log-drift).

        Args:
            s:       Current spot.
            barrier: Call trigger level (absolute, not relative).
            dt:      Time increment.
            sigma_t: Instantaneous vol at this step (may vary by vol_model).

        Returns float in [0, 1].
        """
        if dt < 1e-10:
            return 0.5
        mu_t = self.r - self.q - 0.5 * sigma_t ** 2
        z = (math.log(barrier / s) - mu_t * dt) / (sigma_t * math.sqrt(dt))
        return _ncdf(z)

    def _sample_below(self, s: float, p_j: float, dt: float, sigma_t: float) -> float:
        """Sample next spot from truncated lognormal BELOW the barrier."""
        mu_t = self.r - self.q - 0.5 * sigma_t ** 2
        u = float(self.rng.uniform(0.0, 1.0))
        u_t = float(np.clip(u * p_j, 1e-10, p_j - 1e-10))
        z = _nppf(u_t)
        return s * math.exp(mu_t * dt + sigma_t * math.sqrt(dt) * z)

    def _price_one_path(self, store: bool = False) -> tuple:
        """
        Run one-step survival algorithm for a single path.

        Vol-model dispatch:
            flat   — sigma_t = self.sigma constant throughout.
            local  — sigma_t = dupire_local_vol(s/S0, t_current) at each obs date.
            heston — v_t advanced between obs dates; sigma_t = sqrt(v_t).
            bates  — heston + jump survival correction applied to p_j.

        Returns (payoff, spots_or_None, first_call_idx_or_None).
        """
        s = float(self.S0)
        L = 1.0
        payoff = 0.0
        spots = [s] if store else None
        first_call_idx = None
        t_prev = 0.0

        # Heston/Bates: initialise variance state
        v_t = self.heston_params.get("v0", 0.04) if self.vol_model in ("heston", "bates") else None

        for i, t_i in enumerate(self.obs_dates):
            dt = t_i - t_prev
            barrier = self.ac.call_barrier_at_period(i) * self.S_ref

            # --- Determine instantaneous vol for this step ---
            if self.vol_model == "flat":
                sigma_t = self.sigma

            elif self.vol_model == "local":
                # Query Dupire surface at current spot and calendar time
                sigma_t = self._get_sigma_local(s, t_prev + 0.5 * dt)

            elif self.vol_model in ("heston", "bates"):
                # Advance variance from t_prev to t_i using Euler-Maruyama sub-steps
                v_t = self._advance_heston_variance(v_t, dt)
                sigma_t = math.sqrt(max(v_t, 0.0))
                # Clamp sigma_t to a reasonable range to avoid numerical issues
                sigma_t = float(np.clip(sigma_t, 0.02, 2.0))

            else:
                sigma_t = self.sigma  # fallback

            # --- Survival probability p_j ---
            p_j = self._p_survive(s, barrier, dt, sigma_t)

            # Bates: further reduce p_j by jump-crossing probability
            if self.vol_model == "bates":
                p_j *= self._jump_survival_correction(s, barrier, dt)
                p_j = float(np.clip(p_j, 0.0, 1.0))

            # --- Analytical call contribution ---
            call_pv = (self.ac.redemption_at_call * self.ac.notional
                       + self.ac.coupon_per_period())
            payoff += L * (1.0 - p_j) * math.exp(-self.r * t_i) * call_pv

            if store and first_call_idx is None and (1.0 - p_j) > 0.3:
                first_call_idx = i

            if p_j < 1e-10 or L < 1e-15:
                L = 0.0
                if store and spots is not None:
                    spots.append(s)
                break

            # --- Sample next spot from truncated distribution ---
            s = self._sample_below(s, p_j, dt, sigma_t)
            L *= p_j
            if store and spots is not None:
                spots.append(s)

            t_prev = t_i

        # Terminal payoff for surviving weight
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

    # -----------------------------------------------------------------------
    # Main pricing entry point
    # -----------------------------------------------------------------------

    def price(self, return_paths: bool = False,
              track_convergence: bool = False) -> MCResult:
        """
        Price using one-step survival MC.

        Works identically to the flat-vol version but dispatches to vol-model-aware
        sigma computation at each observation date step.

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
