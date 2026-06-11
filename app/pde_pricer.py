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

# TODO (future): Basket (worst-of) FDM — requires 3D PDE grid (one S dimension
# per asset) or reduced 1D approximation using basket effective vol.
# Currently all pricers treat worst-of as single-asset.


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
# Thomas Algorithm (Tridiagonal Matrix Algorithm) — O(N) Solver
# ---------------------------------------------------------------------------

def thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Solve a tridiagonal linear system A·x = d in O(N) time.

    WHY THIS EXISTS:
        The Crank-Nicolson scheme for the heat equation produces a tridiagonal
        system at each time step: A·u^{n+1} = RHS. The naive dense solver
        (numpy.linalg.solve) is O(N³). The Thomas algorithm exploits the
        tridiagonal structure for an O(N) solve — essential when N_x = 1000.

    ALGORITHM:
        Forward sweep: eliminate the sub-diagonal (a) by row operations.
        Backward substitution: recover the solution from the upper-triangular system.

    Args:
        a: Sub-diagonal (lower diagonal). Length n. a[0] is unused (no row above first).
        b: Main diagonal. Length n.
        c: Super-diagonal (upper diagonal). Length n. c[-1] is unused (no row below last).
        d: Right-hand side vector. Length n.

    Returns:
        x: Solution vector of length n satisfying the tridiagonal system.

    REFERENCE:
        NEXT_SESSION.md §Feature B — Thomas Algorithm pseudocode.
    """
    n = len(d)

    # Work on mutable copies to avoid modifying caller's arrays
    c_ = c.astype(float).copy()
    d_ = d.astype(float).copy()

    # Forward sweep: eliminate sub-diagonal entries
    # After this, the system is upper-triangular.
    c_[0] = c_[0] / b[0]
    d_[0] = d_[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c_[i] / denom
        d_[i] = (d_[i] - a[i] * d_[i - 1]) / denom

    # Backward substitution: recover x from upper-triangular system
    x = np.empty(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


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
        vol_model: str = "flat",
        vol_surface=None,
        local_vol_interp=None,
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
        self.vol_model = vol_model
        self.vol_surface = vol_surface

        # Effective sigma for call probability calculation.
        # The analytical call_probabilities() in autocallable.py assumes
        # flat-vol GBM. Using the vol-model-appropriate sigma here makes
        # the call probability table consistent with the displayed price.
        #   flat / local : self.sigma (ATM implied vol)
        #   heston / bates: self.sigma by default; override via
        #     fd.call_prob_sigma = math.sqrt(v0) after construction
        #     when heston_params are available in the calling page.
        self.call_prob_sigma = self.sigma
        self.local_vol_interp = local_vol_interp

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
        # Terminal V: depends on protection type
        if self.ac.protection_type == "soft_protection":
            # Digital autocall: capital-protected terminal, coupon at 100%
            V_terminal = np.maximum(self.ac.protection_floor, self.S_axis / self.S_ref) * self.notional
            cpn_mask = self.S_axis >= self.S_ref
            V_terminal[cpn_mask] += self.ac.coupon_per_period()
        else:
            # european_ki: par unless knocked in
            V_terminal = np.full(self.N_x, self.notional)
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

        # Per-observation coupon for uncalled nodes above coupon barrier.
        # The coupon is paid at the observation date; the backward sweep
        # handles discounting via the beta term in the change of variables.
        if self.ac.coupon_per_period() > 0:
            cpn_mask = (self.S_axis >= coupon_barrier) & (~called_mask)
            if cpn_mask.any():
                prefactor_cpn = (
                    self.C * np.exp(self.alpha * self.x_axis[cpn_mask] + self.beta * tau)
                )
                u_cpn = np.where(prefactor_cpn > 1e-10,
                                 self.ac.coupon_per_period() / prefactor_cpn, 0.0)
                u[cpn_mask] += u_cpn

        return u

    def _cn_step(self, u: np.ndarray) -> np.ndarray:
        """
        Perform one Crank-Nicolson time step for the heat equation.

        CN SCHEME:
            Averages explicit (forward) and implicit (backward) updates:

                u^{n+1}_j - u^n_j = (ρ/2)(u^{n+1}_{j+1} - 2u^{n+1}_j + u^{n+1}_{j-1})
                                   + (ρ/2)(u^n_{j+1}    - 2u^n_j    + u^n_{j-1})

            where ρ = dtau / dx².  Rearranges to tridiagonal system A·u^{n+1} = d:
                A: main diagonal (1 + ρ), off-diagonals (−ρ/2)
                d: (ρ/2)·u^n_{j-1} + (1 − ρ)·u^n_j + (ρ/2)·u^n_{j+1}

        BOUNDARY CONDITIONS:
            Left  (j=0):   Dirichlet u=0  (deep OTM — option worthless)
            Right (j=N-1): Neumann u[-1]=u[-2]  (no curvature at far end)

        The Neumann BC substitutes u^{n+1}_{N-1} = u^{n+1}_{N-2} into the last
        interior equation, changing the main diagonal entry from (1+ρ) to (1+ρ/2)
        and zeroing the super-diagonal entry there.

        WHY CN IS UNCONDITIONALLY STABLE:
            The explicit scheme requires ρ ≤ 0.5 (CFL condition). CN has no such
            restriction — it is stable for any ρ. This allows larger time steps
            (coarser tau grid) at the same accuracy level. For the same N_tau,
            CN achieves O(dtau²) accuracy vs O(dtau) for explicit.

        Args:
            u: Current heat-equation solution array of shape (N_x,).

        Returns:
            u_new: Solution at next tau step, shape (N_x,).
        """
        N = len(u)
        r = self.courant    # ρ = dtau / dx²; may exceed 0.5 for CN (that's fine)
        n_int = N - 2       # number of interior unknowns (indices 1 to N-2)

        # --- Build tridiagonal system for interior points (j = 1 .. N-2) ---

        # Sub-diagonal: a[i] = −ρ/2 for i > 0; a[0] unused (no row above first)
        a = np.full(n_int, -r / 2.0)

        # Main diagonal: b[i] = 1 + ρ  (modified at last row for Neumann BC)
        b = np.full(n_int, 1.0 + r)

        # Super-diagonal: c[i] = −ρ/2  (set to 0 at last row for Neumann BC)
        c = np.full(n_int, -r / 2.0)

        # RHS: standard CN right-hand side using the previous time step's u
        # d[i] = (ρ/2)·u[j-1] + (1−ρ)·u[j] + (ρ/2)·u[j+1]  where j = i + 1
        # Vectorized: u[:-2] are the j-1 values, u[1:-1] the j values, u[2:] the j+1 values
        d = (r / 2.0) * u[:-2] + (1.0 - r) * u[1:-1] + (r / 2.0) * u[2:]

        # --- Apply boundary condition modifications ---

        # Left Dirichlet BC (j=0, u^{n+1}_0 = 0):
        #   The cross-term a[0]*u^{n+1}_0 = 0, so d[0] needs no modification.
        #   a[0] is never accessed by Thomas (no equation above row 0), so leave as-is.

        # Right Neumann BC (u^{n+1}_{N-1} = u^{n+1}_{N-2}):
        #   Substitute into last interior equation:
        #     -ρ/2·x_{N-3} + (1+ρ)·x_{N-2} − ρ/2·x_{N-2} = d[-1]
        #   → main diagonal becomes (1 + ρ/2), super-diagonal disappears
        b[-1] = 1.0 + r / 2.0
        c[-1] = 0.0
        # d[-1] = (ρ/2)·u[N-3] + (1-ρ)·u[N-2] + (ρ/2)·u[N-1]
        # Since u[N-1] = u[N-2] from the previous step's BC enforcement,
        # this equals (ρ/2)·u[N-3] + (1-ρ/2)·u[N-2], which is exactly right.

        # --- Solve tridiagonal system ---
        x = thomas_solve(a, b, c, d)

        # --- Reconstruct full solution ---
        u_new = np.empty(N)
        u_new[0] = 0.0          # Left Dirichlet BC
        u_new[1:-1] = x         # Interior solution from Thomas
        u_new[-1] = u_new[-2]   # Right Neumann BC (enforce for next step's consistency)

        return u_new

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
        # Dispatch to local vol pricing method if requested
        if self.vol_model == "local" and self.vol_surface is not None:
            return self._price_local_vol(return_grid=return_grid)

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

            # --- FD update: dispatch on scheme ---
            if self.scheme == "crank_nicolson":
                # CN: unconditionally stable, O(dtau²) accuracy
                # Solves tridiagonal system via Thomas algorithm
                u = self._cn_step(u)
            else:
                # Explicit scheme: du/dtau = d²u/dx²  (must have ρ ≤ 0.5)
                u_new = u.copy()
                u_new[1:-1] = u[1:-1] + rho * (u[2:] - 2 * u[1:-1] + u[:-2])
                # Boundary conditions:
                # Left (x = x_min, S → 0): option worthless → u = 0
                u_new[0] = 0.0
                # Right (x = x_max, S → ∞): Neumann BC (no curvature)
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

        # Compute call probabilities — use call_prob_sigma so the table
        # reflects the same effective vol as the price (e.g. sqrt(v0) for Heston)
        call_probs = self.ac.call_probabilities(self.call_prob_sigma, self.r, self.q)

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


    def _price_local_vol(self, return_grid: bool = False) -> "FDResult":
        """
        Price using local vol (Dupire surface) via direct log-space FD.

        WHY A DIFFERENT METHOD:
            The heat-equation change-of-variables assumes constant sigma. With local
            vol, sigma = sigma(x, t) varies across the grid, so the transformation is
            no longer exact. Instead, we work directly with V(x, t) in log-price space:

                PDE: ∂V/∂t + (r-q-σ²/2)·∂V/∂x + (σ²/2)·∂²V/∂x² - r·V = 0

            Converted to backward time τ = T - t (so we sweep τ: 0→T):

                ∂V/∂τ = (σ²_n/2)·(V[n+1]-2V[n]+V[n-1])/dx²
                      + (r-q-σ²_n/2)·(V[n+1]-V[n-1])/(2dx)
                      - r·V[n]

            where σ_n = sigma_loc(S_n, t_current) is the local vol at grid node n.

        STABILITY:
            Explicit scheme stability requires:
                max_n( σ²_n * dt/dx² ) ≤ 0.5
            We compute the maximum sigma in the local vol surface and enforce this
            by auto-adjusting N_tau (same as the flat-vol Courant correction).

        BOUNDARY CONDITIONS (same as heat-equation approach):
            Left  (x = x_min, S → 0): V = 0  (deep OTM, option worthless).
            Right (x = x_max):        Neumann: V[-1] = V[-2].

        AT OBSERVATION DATES:
            Autocall condition imposed identically to the flat-vol method.

        Args:
            return_grid: If True, store V(S, t) snapshots in result.

        Returns:
            FDResult — same structure as the flat-vol price() output.
        """
        import numpy as np

        # --- Pre-compute local vol grid for fast vectorised lookup ---
        # If a pre-built interpolator is provided (shared across pricers), use it.
        # Otherwise build one here (fallback for standalone FDM usage).
        S_ax = self.S_axis                         # shape (N_x,)
        moneyness_ax = S_ax / self.vol_surface.S0  # relative to vol surface spot

        if self.local_vol_interp is not None:
            _lv_interp = self.local_vol_interp
            _LV_g = None  # grid not needed — stability check uses max from shared grid
        else:
            from scipy.interpolate import RegularGridInterpolator as _RGI
            _m_axis = np.linspace(0.40, 1.80, 40)
            _t_axis = np.linspace(0.01, self.ac.maturity_years + 0.05, 25)
            _LV_g = self.vol_surface.dupire_local_vol_grid(_m_axis, _t_axis)
            _lv_interp = _RGI(
                (_t_axis, _m_axis), _LV_g,
                method="linear", bounds_error=False, fill_value=None,
            )

        # --- Stability check based on MAX local vol (not flat vol) ---
        # The Courant condition must hold for the maximum sigma in the local vol
        # surface, not just the flat vol used in __init__. Local vol can reach
        # sigma=1.0 (clamped), which requires much finer time stepping.
        if _LV_g is not None:
            max_sigma = float(np.max(_LV_g))
        else:
            max_sigma = 1.0  # conservative upper bound when grid not available
        dx = self.dx
        dt_phys = self.T / self.N_tau
        courant_local = (max_sigma ** 2) * dt_phys / (dx ** 2)
        if courant_local > 0.5:
            required_N_tau = int(np.ceil((max_sigma ** 2) * self.T / (0.4 * dx ** 2))) + 1
            self.N_tau = required_N_tau
            dt_phys = self.T / self.N_tau

        # Physical backward-time grid: T steps from t=T (maturity) to t=0
        # We match N_tau time steps but in physical time, not tau-space.
        t_axis_phys = np.linspace(self.T, 0.0, self.N_tau + 1)  # T, T-dt, ..., 0

        # --- Terminal payoff at t=T ---
        if self.ac.protection_type == "soft_protection":
            # Digital autocall: capital-protected for all spots
            V = np.maximum(self.ac.protection_floor, S_ax / self.S_ref) * self.ac.notional
            cpn_mask = S_ax >= self.S_ref
            V[cpn_mask] += self.ac.coupon_per_period()
        else:
            # european_ki: par unless knocked in at terminal
            ki_pv = np.maximum(S_ax / self.S_ref, self.ac.protection_floor) * self.ac.notional
            safe_pv = self.ac.notional * np.ones(self.N_x)
            ki_mask = S_ax < self.ac.protection_barrier * self.S_ref
            V = np.where(ki_mask, ki_pv, safe_pv)

        obs_dates = self.ac.observation_dates()
        obs_processed = [False] * len(obs_dates)

        if return_grid:
            n_snapshots = min(50, self.N_tau)
            snapshot_steps = set(np.linspace(0, self.N_tau - 1, n_snapshots, dtype=int))
            grid_snapshots = []
            t_snapshots_list = []

        for step in range(self.N_tau):
            # Current forward time (from today): t decreases as we sweep backward
            t_current = t_axis_phys[step + 1]  # time after this step

            # --- Autocall BC at observation dates ---
            # Apply when we cross an observation date going backward in time.
            # Use non-strict upper bound so maturity (t_obs=T) fires at step 0.
            for i, t_obs in enumerate(obs_dates):
                if (not obs_processed[i]) and (t_current <= t_obs <= t_axis_phys[step]):
                    barrier = self.ac.call_barrier_at_period(i) * self.S_ref
                    call_pv = (self.ac.redemption_at_call * self.ac.notional
                               + self.ac.coupon_per_period())
                    V = np.where(S_ax >= barrier, call_pv, V)

                    # Per-observation coupon for uncalled nodes above coupon barrier
                    cpn_barrier = self.ac.coupon_barrier * self.S_ref
                    cpn_mask = (S_ax >= cpn_barrier) & (S_ax < barrier)
                    V[cpn_mask] += self.ac.coupon_per_period()

                    obs_processed[i] = True

            # --- Local vol at each grid node for this time step ---
            # Clamp moneyness and time to valid surface range, then query all N_x
            # nodes in one vectorised batch via the pre-built interpolator.
            # WHY VECTORISED: the original list comprehension called dupire_local_vol()
            # N_x times per step (N_x × N_tau = 20K total calls at ~300μs each ≈ 6s).
            # One RegularGridInterpolator batch call takes ~50μs regardless of N_x.
            t_q = float(np.clip(t_current, 0.01, self.ac.maturity_years + 0.05))
            m_q = np.clip(moneyness_ax, 0.40, 1.80)
            _pts = np.stack([np.full(self.N_x, t_q), m_q], axis=1)  # (N_x, 2)
            sigma_n = np.clip(_lv_interp(_pts).astype(float), 0.05, 1.0)  # (N_x,)

            # --- Explicit FD update in log-price space ---
            dx = self.dx
            V_new = V.copy()
            # Diffusion coefficient per node: σ²/2
            diff_coeff = 0.5 * sigma_n[1:-1] ** 2
            # Drift coefficient per node: r - q - σ²/2
            drift_coeff = (self.r - self.q) - 0.5 * sigma_n[1:-1] ** 2

            # Second-order central difference (diffusion):
            d2V = (V[2:] - 2 * V[1:-1] + V[:-2]) / dx ** 2
            # First-order central difference (drift):
            dV = (V[2:] - V[:-2]) / (2 * dx)

            V_new[1:-1] = (V[1:-1]
                           + dt_phys * diff_coeff * d2V
                           + dt_phys * drift_coeff * dV
                           - dt_phys * self.r * V[1:-1])
            # BCs: left = 0 (deep OTM), right = Neumann
            V_new[0] = 0.0
            V_new[-1] = V_new[-2]
            V = np.maximum(V_new, 0.0)  # option value non-negative

            if return_grid and step in snapshot_steps:
                grid_snapshots.append(V.copy())
                t_snapshots_list.append(t_current)

        # --- Extract price at S_ref ---
        idx = int(np.searchsorted(S_ax, self.S_ref))
        idx = int(np.clip(idx, 1, self.N_x - 2))
        alpha_i = (self.S_ref - S_ax[idx - 1]) / (S_ax[idx] - S_ax[idx - 1])
        price = (1 - alpha_i) * V[idx - 1] + alpha_i * V[idx]

        # Compute call probabilities — use call_prob_sigma (effective vol for
        # the active vol model) so the table is consistent with the price
        call_probs = self.ac.call_probabilities(self.call_prob_sigma, self.r, self.q)

        result = FDResult(
            price=float(np.clip(price, 0, self.notional * 1.5)),
            call_probs=call_probs,
            obs_dates=obs_dates,
        )
        if return_grid:
            grid_array = np.array(grid_snapshots).T  # (N_x, n_snapshots)
            result.V_grid = grid_array
            result.S_axis = S_ax.copy()
            result.t_axis = np.array(t_snapshots_list)
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

    # d1 and d2 for first-passage calculation (reflection principle for GBM)
    d1 = (log_BS - nu * T) / (sigma * np.sqrt(T))
    d2 = (-log_BS - nu * T) / (sigma * np.sqrt(T))

    # Probability of NOT crossing the barrier by time T
    # Using the reflection principle result for GBM
    p_no_cross = norm.cdf(d1) - np.exp(2 * nu * log_BS / sigma ** 2) * norm.cdf(d2)
    p_cross = 1.0 - p_no_cross

    # Approximate price:
    #   Component 1: called paths -- receive notional at call time (approximate mid-T)
    expected_call_time = T / 2  # simplification; actual is E[tau | tau < T]
    pv_call = p_cross * notional * np.exp(-r * expected_call_time)

    #   Component 2: uncalled paths -- receive notional at maturity
    pv_no_call = p_no_cross * notional * np.exp(-r * T)

    #   Coupon stream: simplified as coupon rate * expected time under the barrier,
    #   discounted at the midpoint. expected_life blends the two exit scenarios.
    expected_life = p_cross * expected_call_time + p_no_cross * T
    pv_coupons = coupon_pa * notional * expected_life * np.exp(-r * expected_life / 2.0)

    price = pv_call + pv_no_call + pv_coupons
    # Clip to sane range: floor at 0, cap at par + all coupons (can't exceed)
    return float(np.clip(price, 0.0, notional * (1.0 + coupon_pa * T)))
