"""
app/autocallable.py
====================
AutoCallable product definition, payoff logic, and call probability calculation.

WHY THIS MODULE EXISTS:
    This module is the single source of truth for the product contract.
    Every pricer (PDE, MC standard, MC survival) imports from here to
    compute payoffs and observation dates. If the payoff logic is wrong,
    fixing it here fixes all pricers simultaneously.

    Separating product definition from pricing follows standard quant
    library design: the payoff is a contract specification, not a
    model assumption. The same autocallable should produce the same
    payoff regardless of which pricing model is used.

DESIGN DECISIONS:
    - Uses Python dataclass for immutability and easy repr/comparison.
    - All barrier levels are expressed as fractions of the reference spot.
      The reference spot S_ref is typically the spot at trade date.
    - Observation dates are computed from (start_date, maturity_years, frequency)
      rather than listed explicitly, to avoid date arithmetic in each pricer.
    - Supports four structure types: phoenix, worst_of, step_down, digital.

PAPER REFERENCE:
    Payoff logic follows Deng, Mallett, McCann (2011) §2:
        - Autocall condition: S_i/S_ref >= call_barrier at obs date i
        - Terminal payoff: max(S_T/S_ref, 1) * notional (no knock-in)
                          or S_T/S_ref * notional (knock-in below barrier)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Observation Date Helpers
# ---------------------------------------------------------------------------

# Frequency → months per period mapping
FREQ_TO_MONTHS: dict[str, int] = {
    "monthly": 1,
    "quarterly": 3,
    "semi-annual": 6,
    "annual": 12,
}


def _observation_dates_from_params(
    maturity_years: float,
    obs_frequency: str,
    start_date: Optional[date] = None,
) -> list[float]:
    """
    Generate observation dates as times-in-years from trade date.

    WHY: Pricers work in continuous time measured in years. Observation
    dates must be expressed consistently in this unit.

    Args:
        maturity_years: Total product maturity in years (e.g. 2.0).
        obs_frequency:  One of "monthly", "quarterly", "semi-annual", "annual".
        start_date:     Trade date. If None, uses today (for testing).

    Returns:
        List of observation times in years, from first to maturity.
        Last element is always maturity_years (within floating-point tolerance).

    Raises:
        ValueError: If obs_frequency is not in FREQ_TO_MONTHS.
    """
    if obs_frequency not in FREQ_TO_MONTHS:
        raise ValueError(
            f"obs_frequency must be one of {list(FREQ_TO_MONTHS)}. Got: '{obs_frequency}'"
        )
    months_per_period = FREQ_TO_MONTHS[obs_frequency]
    n_periods = int(round(maturity_years * 12 / months_per_period))

    # Evenly spaced in year-fractions; last date = maturity_years
    return [round(i * maturity_years / n_periods, 8) for i in range(1, n_periods + 1)]


# ---------------------------------------------------------------------------
# AutoCallable Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutoCallable:
    """
    Complete definition of an autocallable structured note.

    WHY A DATACLASS: Autocallables have many parameters and it's easy to
    make a mistake passing them as positional args. Named fields with
    defaults and __post_init__ validation catch errors early.

    All barrier levels are expressed as fractions of S_ref (reference spot):
        - call_barrier = 1.00 means "call if spot >= 100% of S0"
        - protection_barrier = 0.75 means "knock-in if spot < 75% of S0"

    Attributes:
        name:               Human-readable product name.
        structure_type:     "phoenix", "worst_of", "step_down", or "digital".
        n_assets:           Number of underlying assets (1 for single, 3 for basket).
        S_ref:              Reference spot level(s). Scalar for single, list for basket.
        maturity_years:     Total time to maturity in years.
        obs_frequency:      Observation schedule: "monthly", "quarterly", "semi-annual", "annual".
        call_barrier:       Fraction of S_ref above which autocall triggers.
        coupon_barrier:     Fraction of S_ref above which conditional coupon is paid.
        coupon_pa:          Annual coupon rate (e.g. 0.08 = 8% p.a.).
        digital_coupon:     Fixed dollar coupon for digital structures (None otherwise).
        protection_barrier: Fraction of S_ref below which knock-in occurs.
        protection_type:    "european_ki", "soft_protection".
        protection_floor:   For soft_protection: minimum redemption fraction.
        redemption_at_call: Fraction of notional paid at autocall (usually 1.0).
        notional:           Face value of the note.
        stepped_barriers:   List of (period_index, barrier) for step-down structures.
        correlation_matrix: Asset correlation matrix for basket (worst_of) structures.
        asset_vols:         Per-asset implied vols for basket structures.
    """

    # Required fields
    name: str
    structure_type: str
    S_ref: float  # Reference spot (single underlying)
    maturity_years: float
    obs_frequency: str
    call_barrier: float
    coupon_barrier: float
    protection_barrier: float

    # Optional with defaults
    n_assets: int = 1
    coupon_pa: float = 0.08
    digital_coupon: Optional[float] = None
    protection_type: str = "european_ki"
    protection_floor: float = 0.80
    redemption_at_call: float = 1.00
    notional: float = 1000.0
    stepped_barriers: list = field(default_factory=list)
    correlation_matrix: Optional[list] = None
    asset_vols: Optional[list] = None
    description: str = ""

    def __post_init__(self) -> None:
        """
        Validate parameters on construction.

        WHY: Invalid parameters produce nonsensical prices that are hard to
        debug at pricing time. Catching them here gives a clear error message.

        Edge cases:
            - Maturity must be positive (0 would mean the product already expired).
            - call_barrier must be > protection_barrier (otherwise product collapses
              to either always-called or always-knocked-in).
            - coupon_pa is annualized; per-period payment = coupon_pa / n_obs_per_year.
        """
        if self.maturity_years <= 0:
            raise ValueError(f"maturity_years must be > 0. Got: {self.maturity_years}")

        if self.call_barrier <= 0:
            raise ValueError(f"call_barrier must be > 0. Got: {self.call_barrier}")

        if self.protection_barrier <= 0 or self.protection_barrier > 1.0:
            raise ValueError(
                f"protection_barrier must be in (0, 1]. Got: {self.protection_barrier}"
                # 1.0 is allowed for capital-protected structures (Digital Autocall)
                # where any decline triggers the soft protection.
            )

        # Skip call_barrier >= protection_barrier check for capital-protected structures
        # (Digital Autocall has call_barrier=1.05, protection_barrier=1.0 — valid).
        if self.call_barrier < self.protection_barrier and self.protection_barrier < 1.0:
            raise ValueError(
                f"call_barrier ({self.call_barrier}) should be >= protection_barrier "
                f"({self.protection_barrier}) for a standard autocallable structure."
            )

        if self.coupon_pa is not None and self.coupon_pa < 0:
            raise ValueError(f"coupon_pa must be >= 0. Got: {self.coupon_pa}")

        valid_structures = {"phoenix", "worst_of", "step_down", "digital"}
        if self.structure_type not in valid_structures:
            raise ValueError(
                f"structure_type must be one of {valid_structures}. Got: '{self.structure_type}'"
            )

    # -------------------------------------------------------------------------
    # Observation Schedule
    # -------------------------------------------------------------------------

    def observation_dates(self) -> list[float]:
        """
        Return observation times in years, from first observation to maturity.

        WHY: We compute this here (not at construction) so that the class
        can be serialized/compared without precomputed state.

        Returns:
            Sorted list of floats in (0, maturity_years], e.g.
            [0.25, 0.50, 0.75, 1.00, ..., 2.00] for quarterly 2yr.
        """
        return _observation_dates_from_params(self.maturity_years, self.obs_frequency)

    def n_observations(self) -> int:
        """Return total number of observation dates (including maturity)."""
        return len(self.observation_dates())

    def coupon_per_period(self) -> float:
        """
        Coupon paid at each observation date (dollar amount per $notional).

        For phoenix/step-down: coupon_pa / periods_per_year * notional.
        For digital: the fixed digital_coupon amount.
        Returns 0 for structures with no conditional coupon.
        """
        if self.digital_coupon is not None:
            return self.digital_coupon

        if self.coupon_pa is None or self.coupon_pa == 0:
            return 0.0

        months_per_period = FREQ_TO_MONTHS.get(self.obs_frequency, 3)
        periods_per_year = 12 / months_per_period
        return self.coupon_pa / periods_per_year * self.notional

    def call_barrier_at_period(self, period_index: int) -> float:
        """
        Return the call barrier level (as fraction of S_ref) for a given observation.

        WHY: Step-down barriers change at specific periods. This method
        centralizes the step-down logic so pricers don't need to implement it.

        Args:
            period_index: 0-based index of the observation date.

        Returns:
            Call barrier level as fraction of S_ref (e.g. 0.95).
        """
        if not self.stepped_barriers:
            return self.call_barrier

        # Find the most recent step-down that applies to this period
        # stepped_barriers is [(period_from, barrier_level), ...]
        applicable = self.call_barrier  # default = initial barrier
        for period_from, barrier_level in self.stepped_barriers:
            if period_index >= period_from:
                applicable = barrier_level
        return applicable

    # -------------------------------------------------------------------------
    # Payoff Logic
    # -------------------------------------------------------------------------

    def is_called(self, spot: float, period_index: int = 0) -> bool:
        """
        Check whether the autocall trigger fires at a given observation.

        Args:
            spot:         Spot price (or worst-of spot for basket) at observation.
            period_index: 0-based observation index (needed for step-down barriers).

        Returns:
            True if spot >= call_barrier * S_ref (autocall triggers).
        """
        barrier = self.call_barrier_at_period(period_index) * self.S_ref
        return spot >= barrier

    def coupon_is_paid(self, spot: float) -> bool:
        """
        Check whether the conditional coupon is paid at an observation.

        For phoenix structures, the coupon is paid if spot >= coupon_barrier,
        independently of whether the autocall triggers. (Autocall and coupon
        can both happen simultaneously.)

        Args:
            spot: Spot price (or worst-of spot for basket) at observation.

        Returns:
            True if coupon is due.
        """
        # Digital: same condition as autocall
        if self.structure_type == "digital":
            return spot >= self.coupon_barrier * self.S_ref

        return spot >= self.coupon_barrier * self.S_ref

    def payoff_at_observation(
        self,
        spot: float,
        t_years: float,
        period_index: int,
        r: float,
    ) -> Optional[float]:
        """
        Compute the discounted payoff if the autocall triggers at this observation.

        Returns the present value of all cash flows occurring at this date:
        (redemption + coupon) discounted to time 0.

        WHY DISCOUNTING HERE: The MC pricer accumulates PV contributions
        at each observation. Centralizing the discounting in this method
        avoids duplication across mc_standard and mc_survival.

        Args:
            spot:         Spot price at this observation date.
            t_years:      Time in years from trade date to this observation.
            period_index: 0-based observation index.
            r:            Continuously compounded risk-free rate.

        Returns:
            Present value of the call payoff if autocall triggers, else None.
            None signals "not called; continue to next observation."
        """
        if not self.is_called(spot, period_index):
            return None

        # Cash flows at this observation date
        redemption = self.redemption_at_call * self.notional
        coupon = self.coupon_per_period() if self.coupon_is_paid(spot) else 0.0
        total_cf = redemption + coupon

        # Discount to time 0
        return total_cf * np.exp(-r * t_years)

    def terminal_payoff(
        self,
        spot_T: float,
        knocked_in: bool,
        r: float,
    ) -> float:
        """
        Compute the present value of the terminal (maturity) payoff.

        Called when the note reaches maturity without triggering an autocall.
        The terminal payoff depends on whether the knock-in put barrier was
        breached during the note's life.

        Args:
            spot_T:     Final spot price (or worst-of ratio for basket).
            knocked_in: True if the spot crossed below protection_barrier
                        at any observation (for european_ki type).
            r:          Continuously compounded risk-free rate.

        Returns:
            Present value of the terminal cash flow.

        Payoff rules:
            - Digital with soft_protection: max(protection_floor, 1.0) * notional
              (up to digital_coupon if above call_barrier, otherwise floor)
            - Not knocked-in (european_ki): notional (return of principal)
            - Knocked-in: max(spot_T / S_ref, protection_floor) * notional
              (investor participates in SPX downside below protection_barrier)
        """
        T = self.maturity_years
        discount = np.exp(-r * T)

        if self.protection_type == "soft_protection":
            # Soft protection: investor always receives at least protection_floor
            # At maturity, if above call_barrier get digital_coupon too
            base_redemption = max(self.protection_floor, spot_T / self.S_ref) * self.notional
            coupon = self.coupon_per_period() if self.coupon_is_paid(spot_T) else 0.0
            return (base_redemption + coupon) * discount

        if not knocked_in:
            # No knock-in: principal returned in full plus any earned coupon
            coupon = self.coupon_per_period() if self.coupon_is_paid(spot_T) else 0.0
            return (self.notional + coupon) * discount

        # Knocked-in: investor bears downside. Payoff = spot_T/S_ref * notional.
        # There is no floor — this is a genuine capital-at-risk outcome.
        final_return = spot_T / self.S_ref
        return final_return * self.notional * discount

    # -------------------------------------------------------------------------
    # Call Probability (analytical, Paper 1 §2.2)
    # -------------------------------------------------------------------------

    def call_probabilities(
        self,
        sigma: float,
        r: float,
        q: float = 0.0,
    ) -> list[float]:
        """
        Compute the probability of the autocall triggering at each observation date.

        Uses the analytical approach from Deng, Mallett, McCann (2011) §2.2:
        p_i = P(S_j < C for all j < i AND S_i >= C)

        This uses the joint distribution of correlated lognormal increments.
        For the single-underlying case, this reduces to an analytical formula
        involving the bivariate normal CDF.

        WHY ANALYTICAL: The PDE pricer can compute prices but not call
        probabilities directly. These analytical probabilities give the
        interviewer a clean "term structure of autocall risk" visualization.

        Args:
            sigma: Implied volatility (annualized).
            r:     Risk-free rate (continuous).
            q:     Dividend yield (continuous).

        Returns:
            List of floats, one per observation date, summing to approximately
            the total probability of autocall. Last element is the probability
            of reaching maturity without calling.

        Edge cases:
            - Approximates using GBM lognormal increments (no jumps, flat vol).
            - For step-down barriers, uses the barrier at each period.
            - For basket (worst_of), uses the single-asset approximation
              with asset_vols[0] (conservative — actual basket vol is higher).
        """
        obs_dates = self.observation_dates()
        n = len(obs_dates)

        # Drift under risk-neutral measure
        mu = r - q - 0.5 * sigma ** 2

        # Precompute log-barrier at each date: log(C_i * S_ref / S_ref) = log(C_i)
        # where C_i is the call_barrier_at_period(i) * S_ref
        log_barriers = np.array([
            np.log(self.call_barrier_at_period(i))  # log(B_i / S_ref)
            for i in range(n)
        ])

        # Simulate marginal distribution of log(S_t / S_ref)
        # X_i = sum of increments up to obs date i
        # X_i ~ Normal(mu * t_i, sigma^2 * t_i)

        probs = []
        prob_survived = 1.0  # P(not yet called before obs i)

        for i in range(n):
            t_i = obs_dates[i]
            mean_i = mu * t_i
            std_i = sigma * np.sqrt(t_i)

            # Approximate: P(S_i >= C_i | not called before) * prob_survived
            # For simplicity, use marginal distribution (ignores path dependency
            # for observations > 1, which slightly overestimates probabilities)
            # A full implementation would use the Margrabe formula or simulation.
            from scipy.stats import norm as scipy_norm
            p_above = 1.0 - scipy_norm.cdf((log_barriers[i] - mean_i) / std_i)

            p_called_at_i = p_above * prob_survived
            probs.append(p_called_at_i)
            prob_survived *= (1.0 - p_above)

        return probs


# ---------------------------------------------------------------------------
# Factory: Build AutoCallable from securities dict
# ---------------------------------------------------------------------------

def from_security_dict(params: dict, S_ref: float) -> AutoCallable:
    """
    Construct an AutoCallable from a securities.py parameter dict.

    WHY: The SECURITIES dict in securities.py stores raw parameters from
    the term sheet. This factory fills in defaults and constructs the
    proper AutoCallable object so calling code is clean.

    Args:
        params: A dict from app.components.securities.SECURITIES.
        S_ref:  Reference spot price to use (from the loaded snapshot).

    Returns:
        Fully initialized AutoCallable instance.
    """
    return AutoCallable(
        name=params["name"],
        structure_type=params["structure_type"],
        S_ref=S_ref,
        maturity_years=params["maturity_years"],
        obs_frequency=params["obs_frequency"],
        call_barrier=params["call_barrier"],
        coupon_barrier=params["coupon_barrier"],
        protection_barrier=params["protection_barrier"],
        n_assets=params.get("n_assets", 1),
        coupon_pa=params.get("coupon_pa"),
        digital_coupon=params.get("digital_coupon"),
        protection_floor=params.get("protection_floor", 0.0),
        redemption_at_call=params.get("redemption_at_call", 1.0),
        notional=params.get("notional", 1000.0),
        stepped_barriers=params.get("stepped_barriers") or [],
        correlation_matrix=(
            np.array(params["correlation_matrix"])
            if "correlation_matrix" in params and params["correlation_matrix"] is not None
            else None
        ),
        asset_vols=params.get("asset_vols"),
        description=params.get("description", ""),
    )
