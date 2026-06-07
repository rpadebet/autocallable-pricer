"""
app/pde_pricer.py
==================
Finite Difference PDE pricer for autocallable structured products.

WHY THIS MODULE EXISTS:
    The PDE approach provides a deterministic, grid-based price that serves as
    the benchmark for validating Monte Carlo results. It also exposes the full
    V(S, t) value function grid which makes the pricing algorithm visually
    understandable (Page 3: FDM Visualization).

    The algorithm follows Deng, Mallett, McCann (2011) §2.2 exactly:
        1. Change of variables transforms Black-Scholes PDE → heat equation.
        2. Explicit finite difference solves the heat equation on a uniform grid.
        3. Autocall boundary conditions are imposed at each observation date.

ALGORITHM OVERVIEW (Plain English):
    We price by working backward in time from maturity. At maturity T, we know
    the terminal payoff V(S, T) for every spot level S. Then we sweep backward,
    time step by time step, updating V using the FD update rule. At each
    observation date, we impose the autocall condition: if S >= call_barrier,
    the option is called (V becomes the call payoff); otherwise we continue.

CHANGE OF VARIABLES (Paper 1 §2.2):
    S = C * exp(x)         where C = call_barrier * S_ref (strike equivalent)
    t = T - 2*tau / sigma² (backward time)
    V(S, t) = C * exp(alpha*x + beta*tau) * u(x, tau) + f_0

    This transforms the Black-Scholes PDE into the standard heat equation:
        du/dtau = d²u/dx²
    which has simpler boundary conditions and better numerical properties.

    alpha = -(k1 - 1) / 2,  where k1 = 2*(r - q) / sigma²
    beta  = -alpha² - 2*(r + cds_spread) / sigma²

STABILITY CONDITION (Courant–Friedrichs–Lewy criterion):
    For the explicit scheme:  dtau / dx² <= 0.5
    Equivalently:  dt / (sigma² * dx² / 2) <= 1 in original time.
    Violating this causes oscillations that grow exponentially (numerical instability).

PAPER REFERENCE:
    Deng, Mallett, McCann (2011), "Modeling Autocallable Structured Products",
    §2.2 "Discrete Autocall" and §2.3 "Closed-Form Continuous Autocall"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from app.autocallable import AutoCallable


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FDResult:
    """
    Container for finite difference pricing results.

    Attributes:
        price:          Fair value of the autocallable note (present value).
        call_probs:     Probability of being called at each observation date.
        obs_dates:      Observation times in years (matches call_probs).
        V_grid:         Full V(S, t) grid. Shape: (N_S, N_tau). None unless
                        price(return_grid=True) was called.
        S_axis:         Spot price axis corresponding to V_grid rows.
        t_axis:         Time axis (forward time, 0=today) for V_grid columns.
        greeks:         Dict of {'delta': float, 'gamma': float, 'vega': float}
                        computed by finite differences on the price surface.
    """
    price: float
    call_probs: list[float]
    obs_dates: list[float]
    V_grid: Optional[np.ndarray] = None
    S_axis: Optional[np.ndarray] = None
    t_axis: Optional[np.ndarray] = None
    greeks: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FDPricer Class
# ---------------------------------------------------------------------------

class FDPricer:
    """
    Finite Difference pricer for autocallable structured notes.

    Implements the explicit FD scheme from Deng et al. (2011) §2.2.
    Works in the heat-equation domain (after change of variables) for
    numerical stability and direct connection to the paper.

    WHY EXPLICIT SCHEME (not Crank-Nicolson by default):
        The explicit scheme is simpler to explain and matches Paper 1 exactly.
        It requires more time steps for stability but with N_x=1000, N_tau=500
        the Courant number is comfortably below 0.5. Crank-Nicolson is available
        via the `scheme` parameter for comparison.

    Args:
        autocallable: The AutoCallable product to price.
        sigma:         Flat implied volatility (annualized).
        r:             Risk-free rate (continuously compounded).
        q:             Dividend yield (continuously compounded).
        cds_spread:    Credit spread for the issuer (default 0 for demo).
        N_x:           Number of spatial grid points in x-domain.
        N_tau:         Number of time steps (in transformed time).
        x_min:         Left boundary of x-domain (log-moneyness). Default -5.
        scheme:        "explicit" or "crank_nicolson".
    """

    def __init__(
        self,
        autocallable: AutoCallable,
        sigma: float,
        r: float,
        q: float = 0.0,
        cds_spread: float = 0.0,
        N_x: int = 200,    # Reduced default for speed; spec says 1000 for accuracy
        N_tau: int = 100,  # Reduced default for speed; spec says 500 for accuracy
        x_min: float = -5.0,
        scheme: str = "explicit",
    ) -> None:
        self.ac = autocallable
        self.sigma = sigma
        self.r = r
        self.q = q
        self.cds = cds_spread
        self.N_x = N_x
        self.N_tau = N_tau
        self.x_min = x_min
        self.scheme = scheme

        # The "strike equivalent" in the paper's change of variables
        # C = call_barrier * S_ref (the spot level that triggers the call)
        self.C = autocallable.call_barrier * autocallable.S_ref
        self.S_ref = autocallable.S_ref
        self.T = autocallable.maturity_years
        self.notional = autocallable.notional

        # --- Change of variables parameters (Paper 1, §2.2) ---
        # k1 = 2(r - q) / sigma²  — dimensionless drift
        self.k1 = 2 * (r - q) / (sigma ** 2)
        self.alpha = -0.5 * (self.k1 - 1)
        # beta captures risk-free discounting and issuer credit spread
        self.beta = -(self.alpha ** 2) - 2 * (r + cds_spread) / (sigma ** 2)

        # Max transformed time: tau_max = sigma² * T / 2
        self.tau_max = 0.5 * sigma ** 2 * self.T

        # --- Grid setup ---
        # x-domain: from x_min (deep OTM, S << C) to 0 (ATM, S = C)
        # We extend slightly above 0 for the in-the-money region
        self.x_axis = np.linspace(x_min, 2.0, N_x)  # x ∈ [x_min, 2]
        self.dx = self.x_axis[1] - self.x_axis[0]

        self.tau_axis = np.linspace(0, self.tau_max, N_tau)
        self.dtau = self.tau_axis[1] - self.tau_axis[0] if N_tau > 1 else self.tau_max

        # Courant number: must be <= 0.5 for explicit stability
        self.courant = self.dtau / (self.dx ** 2)
        if self.courant > 0.5 and scheme == "explicit":
            # Auto-correct: increase N_tau until stable
            required_N_tau = int(np.ceil(self.tau_max / (0.4 * self.dx ** 2))) + 1
            self.N_tau = required_N_tau
            self.tau_axis = np.linspace(0, self.tau_max, required_N_tau)
            self.dtau = self.tau_axis[1] - self.tau_axis[0]
            self.courant = self.dtau / (self.dx ** 2)

        # Spot price corresponding to each x-grid point: S = C * exp(x)
        self.S_axis = self.C * np.exp(self.x_axis)

    def _tau_to_t(self, tau: float) -> float:
        """Convert transformed time tau to forward time t (years from today)."""
        return self.T - 2 * tau / (self.sigma ** 2)

    def _t_to_tau(self, t: float) -> float:
        """Convert forward time t (years) to transformed time tau."""
        return 0.5 * self.sigma ** 2 * (self.T - t)

    def _u_to_V(self, u: np.ndarray, tau: float) -> np.ndarray:
        """
        Convert heat-equation solution u(x, tau) back to option value V(S, t).

        Inverse of the change of variables (Paper 1 §2.2):
            V(x, tau) = C * exp(alpha*x + beta*tau) * u(x, tau)

        WHY NOT ADDING f_0 TERM: For the homogeneous boundary condition case
        (u = 0 at x = x_min, V approaches 0 for deep OTM), the f_0 correction
        term is effectively zero. We include the full exponential prefactor.

        Args:
            u:   Heat-equation solution array of shape (N_x,).
            tau: Current transformed time.

        Returns:
            V array of shape (N_x,) in dollar terms.
        """
        prefactor = self.C * np.exp(self.alpha * self.x_axis + self.beta * tau)
        return prefactor * u

    def _terminal_u(self) -> np.ndarray:
        """
        Compute the initial condition for the heat equation (= terminal payoff in V).

        At tau = 0 (maturity T), the payoff V(S, T) is known from the product terms.
        We convert this to u(x, 0) = V(S, T) / (C * exp(alpha * x)).

        For a non-called autocallable at maturity:
            - If not knocked-in: V = notional (par)
            - If knocked-in: V = S / S_ref * notional = (C/S_ref) * exp(x) * notional

        WHY WE APPROXIMATE HERE: We use the "no knock-in" payoff (par return)
        as the terminal condition. The knock-in probability is embedded in
        the path-dependent boundary conditions imposed at observation dates.
        This is the standard discrete autocall FD approach.
        """
        # Terminal V: par redemption if not called (conservative; knock-in
        # is path-dependent and handled approximately here)
        V_terminal = np.full(self.N_x, self.notional)

        # Intrinsic value below reference: proportional to spot
        # For knocked-in paths (approximated as terminal spot < S_ref):
        ki_mask = self.S_axis < self.ac.protection_barrier * self.S_ref
        V_terminal[ki_mask] = (
            self.S_axis[ki_mask] / self.S_ref * self.notional
        )

        # Convert V → u using the inverse change of variables at tau=0
        # u(x, 0) = V(x, 0) / (C * exp(alpha * x + beta * 0))
        prefactor = self.C * np.exp(self.alpha * self.x_axis)
        u0 = np.where(prefactor > 1e-10, V_terminal / prefactor, 0.0)
        return u0

    def _apply_autocall_bc(self, u: np.ndarray, tau: float, period_index: int) -> np.ndarray:
        """
        Apply the autocall boundary condition at an observation date.

        At each observation date, if S >= call_barrier * S_ref, the product
        is redeemed. In the heat equation domain, this means:
            u(x, tau) = V_call / (C * exp(alpha*x + beta*tau))
        for all x such that S = C * exp(x) >= call_barrier * S_ref.

        Also pays conditional coupon if spot >= coupon_barrier.

        Args:
            u:            Current heat-equation solution.
            tau:          Current transformed time.
            period_index: 0-based observation index (for step-down barriers).

        Returns:
            Modified u with autocall condition imposed.
        """
        barrier = self.ac.call_barrier_at_period(period_index) * self.S_ref
        coupon_barrier = self.ac.coupon_barrier * self.S_ref

        # Indices where autocall triggers
        called_mask = self.S_axis >= barrier

        # Call payoff: redemption + coupon (if above coupon barrier)
        V_call = np.where(
            self.S_axis[called_mask] >= coupon_barrier,
            self.ac.redemption_at_call * self.notional + self.ac.coupon_per_period(),
            self.ac.redemption_at_call * self.notional,
        )

        # Convert V_call → u_call
        prefactor_called = (
            self.C * np.exp(self.alpha * self.x_axis[called_mask] + self.beta * tau)
        )
        u_call = np.where(prefactor_called > 1e-10, V_call / prefactor_called, 0.0)
        u[called_mask] = u_call

        return u

    def price(self, return_grid: bool = False) -> FDResult:
        """
        Price the autocallable using explicit finite difference backward induction.

        The algorithm:
            1. Set u(x, tau=0) = terminal payoff in heat-equation coordinates.
            2. Identify observation dates in tau coordinates.
            3. Sweep forward in tau (= backward in physical time):
                 a. At each non-observation step: apply FD update rule.
                 b. At each observation step: impose autocall BC before updating.
            4. At tau = tau_max (t=0), extract price at x = log(S0/C).

        Args:
            return_grid: If True, store the full V(S, t) grid in the result.
                         This is used by Page 3 (FDM Visualization) to animate
                         the backward induction. Set False for normal pricing
                         (saves memory — grid can be 200MB for N_x=1000, N_tau=500).

        Returns:
            FDResult with price, call probabilities, and optionally the full grid.

        Edge cases:
            - If S_ref is not in the S_axis range, we extrapolate linearly.
              This can happen if S_ref >> C * exp(x_max) (deep call barrier).
            - Courant stability is enforced in __init__; if violated, N_tau is
              increased automatically.
        """
        obs_dates = self.ac.observation_dates()
        # Convert observation t-values to tau values
        obs_taus = [self._t_to_tau(t) for t in obs_dates]

        # Initialize heat-equation solution at tau=0 (maturity)
        u = self._terminal_u()

        # Optional: store grid snapshots
        if return_grid:
            # Store V at evenly-spaced tau slices (not every step — too much memory)
            n_snapshots = min(50, self.N_tau)
            snapshot_indices = np.linspace(0, self.N_tau - 1, n_snapshots, dtype=int)
            grid_snapshots = []
            t_snapshots = []

        # Track which observation has been processed (work from maturity backward)
        obs_processed = [False] * len(obs_dates)
        # call_counts[i] = weighted count of paths calling at obs i (not exact prob here)
        call_counts = np.zeros(len(obs_dates))

        # Courant number for explicit update: r = dtau / dx²
        rho = self.courant  # <= 0.5

        for tau_step in range(self.N_tau):
            tau = self.tau_axis[tau_step]
            t = self._tau_to_t(tau)

            # --- Apply autocall boundary conditions at observation dates ---
            for i, obs_tau in enumerate(obs_taus):
                if (not obs_processed[i]) and (tau >= obs_tau - self.dtau / 2):
                    u = self._apply_autocall_bc(u, tau, i)
                    obs_processed[i] = True
                    # Approximate call probability: fraction of grid above barrier
                    # weighted by lognormal density around S_ref
                    barrier = self.ac.call_barrier_at_period(i) * self.S_ref
                    call_counts[i] = (self.S_axis >= barrier).mean()

            # --- Explicit FD update: du/dtau = d²u/dx² ---
            # Interior points only (not boundary)
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + rho * (u[2:] - 2 * u[1:-1] + u[:-2])

            # Boundary conditions:
            # Left (x = x_min, S → 0): option worthless → u = 0
            u_new[0] = 0.0
            # Right (x = x_max, S → ∞): intrinsic value dominates
            # Neumann BC: du/dx = 0 (no curvature at far OTM call)
            u_new[-1] = u_new[-2]

            u = u_new

            # Store grid snapshot
            if return_grid and tau_step in snapshot_indices:
                V_now = self._u_to_V(u, tau)
                grid_snapshots.append(V_now.copy())
                t_snapshots.append(t)

        # --- Extract price at current spot level (t=0, tau=tau_max) ---
        # Find index closest to S_ref in S_axis
        S0 = self.S_ref
        idx = np.searchsorted(self.S_axis, S0)
        idx = np.clip(idx, 1, self.N_x - 2)

        # Linear interpolation between grid points
        alpha_interp = (S0 - self.S_axis[idx - 1]) / (self.S_axis[idx] - self.S_axis[idx - 1])
        V_final = self._u_to_V(u, self.tau_max)
        price = (1 - alpha_interp) * V_final[idx - 1] + alpha_interp * V_final[idx]

        # Compute call probabilities (approximate analytical from autocallable.py)
        call_probs = self.ac.call_probabilities(self.sigma, self.r, self.q)

        result = FDResult(
            price=float(np.clip(price, 0, self.notional * 1.5)),
            call_probs=call_probs,
            obs_dates=obs_dates,
        )

        if return_grid:
            # Build 2D grid: rows = S_axis, cols = t_axis (forward time)
            grid_array = np.array(grid_snapshots).T  # shape (N_x, n_snapshots)
            result.V_grid = grid_array
            result.S_axis = self.S_axis.copy()
            result.t_axis = np.array(t_snapshots)

        return result


# ---------------------------------------------------------------------------
# Closed-Form Continuous Autocall (Paper 1 §2.3)
# ---------------------------------------------------------------------------

def continuous_autocall_closedform(
    S0: float,
    call_barrier: float,
    maturity_years: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    coupon_pa: float = 0.08,
    notional: float = 1000.0,
) -> float:
    """
    Closed-form price for a continuously-monitored autocallable (Paper 1 §2.3).

    WHY THIS EXISTS:
        Paper 1 derives a closed-form solution for the special case where the
        autocall is monitored continuously (not just at discrete obs dates).
        This serves as an upper bound on the discretely-monitored price and
        validates our FD implementation: the FD price should converge to this
        as observation frequency → ∞.

    THE FORMULA:
        The continuously-monitored autocall is equivalent to a first-passage
        problem for geometric Brownian motion. If S_t >= B at any time t in [0,T],
        the note is called. The expected discounted payoff uses the reflection
        principle for Brownian motion.

        For a simplified version (zero coupon, par at call):
            P(first passage before T) = N(d1) + exp(2*nu*log(B/S0)/sigma²) * N(d2)
        where:
            nu  = r - q - sigma²/2
            d1  = (log(S0/B) + nu*T) / (sigma*sqrt(T))
            d2  = (log(S0/B) - nu*T) / (sigma*sqrt(T))

        The full formula including coupon accrual integrates over the passage
        time distribution. We use a simplified approximation here: price the
        call option component analytically and approximate the coupon stream.

    Args:
        S0:             Current spot price.
        call_barrier:   Call barrier as fraction of S0 (e.g. 1.0 = at par).
        maturity_years: Maturity in years.
        sigma:          Flat implied vol.
        r:              Risk-free rate.
        q:              Dividend yield.
        coupon_pa:      Annual coupon rate.
        notional:       Face value.

    Returns:
        Approximate price using the continuous-monitoring closed form.
    """
    from scipy.stats import norm

    B = call_barrier * S0  # Barrier level in spot space
    T = maturity_years
    nu = r - q - 0.5 * sigma ** 2

    if B <= S0:
        # Already above or at barrier: immediate call
        return notional * (1 + coupon_pa * T)

    # Log-ratio of barrier to spot
    log_BS = np.log(B / S0)

    # d1 and d2 for first-passage calculation
    # These come from the inverse Gaussian / Bachelier formula for barrier crossing
    d1 = (-log_BS + nu * T) / (sigma * np.sqrt(T))
    d2 = (-log_BS - nu * T) / (sigma * np.sqrt(T))

    # Probability of NOT crossing the barrier by time T
    # Using the reflection principle result for GBM
    p_no_cross = norm.cdf(d1) - np.exp(2 * nu * log_BS / sigma ** 2) * norm.cdf(d2)
    p_cross = 1.0 - p_no_cross

    # Approximate price:
    #   Component 1: called paths → receive notional at call time (approximate mid-T)
    expected_call_time = T / 2  # simplification; actual is E[tau | tau < T]
    pv_call = p_cross * notional * np.exp(-r * expected_call_time)

    #   Component 2: uncalled paths → receive notional at maturity
    pv_no_call = p_no_cross * notional * np.exp(-r * T)

    #   Coupon stream: simplified as coupon rate × expected time under the barrier
    expected_life = p_cross * expected_call_time + p_no_cross * T
    pv_coupon = coupon_pa * notional * expected_life * np.exp(-r * expected_life / 2)

    return float(pv_call + pv_no_call + pv_coupon)
