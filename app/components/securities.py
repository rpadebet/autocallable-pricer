"""
app/components/securities.py
=============================
Pre-configured autocallable structured product definitions.

WHY THIS EXISTS:
    The demo uses a fixed set of 4 pre-configured securities rather than a
    parameter-entry form. This serves two purposes:
        1. The demo flow is predictable — we know what price to expect for
           each security, making it easy to verify the pricing engine live.
        2. Each security illustrates a distinct feature (phoenix coupon,
           worst-of basket, step-down barrier, digital payoff), so the
           interviewer sees the breadth of the product space.

DESIGN:
    Each security is stored as a dict of raw parameters. The AutoCallable
    dataclass in app/autocallable.py consumes these dicts. Keeping them
    separate from the dataclass means we can update term sheets without
    touching pricing logic.

    All parameters are realistic and match common bank-issued autocallable
    note structures (EUR/USD denominated, 1-3yr maturity, quarterly obs).

SECURITIES:
    1. Phoenix Autocall   — Standard SPX, quarterly, 8% p.a.
    2. Worst-Of Autocall  — SPX/NDX/RUT basket, quarterly, 12% p.a.
    3. Step-Down Barrier  — Declining call trigger, monthly obs, 10% p.a.
    4. Digital Autocall   — Binary $50 payment, annual obs, 80% capital protection
"""

from typing import Any


# ---------------------------------------------------------------------------
# Security Parameter Registry
# ---------------------------------------------------------------------------

SECURITIES: dict[str, dict[str, Any]] = {
    "Phoenix Autocall": {
        # --- Metadata ---
        "name": "Phoenix Autocall",
        "description": (
            "Standard Phoenix structure on SPX. Pays an 8% p.a. conditional coupon "
            "at each quarterly observation if spot is above the coupon barrier (75%). "
            "Automatically calls (redeems at par) if spot is at or above 100% of initial. "
            "At maturity, if spot fell below 75% at any point (knock-in), investor "
            "receives the proportional SPX return (loses principal). Otherwise, par."
        ),
        "structure_type": "phoenix",  # standard single-underlying phoenix

        # --- Underlying ---
        "underlyings": ["SPX"],
        "n_assets": 1,

        # --- Observation schedule ---
        "obs_frequency": "quarterly",  # quarterly = 4 per year
        "maturity_years": 2.0,
        # Observation dates are computed by autocallable.py from these params

        # --- Barrier levels (as fraction of initial spot) ---
        "call_barrier": 1.00,        # autocall trigger: spot >= 100% of S0
        "coupon_barrier": 0.75,      # conditional coupon: spot >= 75% of S0
        "protection_barrier": 0.75,  # knock-in put: spot < 75% (any obs or daily?)
        "protection_type": "european_ki",  # knock-in is observed at maturity only

        # --- Payoffs ---
        "coupon_pa": 0.08,       # 8% per annum; per observation = 8%/4 = 2%
        "redemption_at_call": 1.00,    # 100% of notional at autocall
        "notional": 1000.0,      # $1,000 face value

        # --- Model params (overridden by sidebar if user changes them) ---
        "default_vol": 0.17,     # ATM vol fallback if no snapshot loaded
        "default_spot": 5300.0,  # SPX reference level if no snapshot loaded
    },

    "Worst-Of Autocall": {
        # --- Metadata ---
        "name": "Worst-Of Autocall",
        "description": (
            "Worst-of basket on SPX, NDX, and RUT. The autocall trigger and all "
            "barrier levels are assessed on the worst-performing asset (minimum "
            "S_i/S0_i ratio). Higher 12% p.a. coupon compensates for the extra "
            "correlation and basket risk. At maturity, loss is linked to the worst "
            "performer if the knock-in barrier (70%) was breached."
        ),
        "structure_type": "worst_of",

        # --- Underlying ---
        "underlyings": ["SPX", "NDX", "RUT"],
        "n_assets": 3,
        # Correlation matrix (SPX-NDX high, SPX-RUT moderate, NDX-RUT moderate)
        "correlation_matrix": [
            [1.00, 0.85, 0.75],
            [0.85, 1.00, 0.70],
            [0.75, 0.70, 1.00],
        ],
        # Individual implied vols (approximate ATM)
        "asset_vols": [0.17, 0.20, 0.22],

        # --- Observation schedule ---
        "obs_frequency": "quarterly",
        "maturity_years": 3.0,

        # --- Barrier levels ---
        "call_barrier": 1.00,
        "coupon_barrier": 0.70,
        "protection_barrier": 0.70,
        "protection_type": "european_ki",

        # --- Payoffs ---
        "coupon_pa": 0.12,
        "redemption_at_call": 1.00,
        "notional": 1000.0,

        # --- Model params ---
        "default_vol": 0.17,
        "default_spot": 5300.0,
    },

    "Step-Down Barrier": {
        # --- Metadata ---
        "name": "Step-Down Barrier",
        "description": (
            "SPX autocall with a declining autocall trigger that steps down quarterly. "
            "Starts at 100%, drops to 95% after year 1, drops again to 90% after year 1.5. "
            "Monthly observation schedule increases call probability over time. "
            "10% p.a. conditional coupon paid monthly if spot >= 75%. "
            "Knock-in put protection at 75%."
        ),
        "structure_type": "step_down",

        # --- Underlying ---
        "underlyings": ["SPX"],
        "n_assets": 1,

        # --- Observation schedule ---
        "obs_frequency": "monthly",
        "maturity_years": 2.0,

        # --- Step-down barrier schedule ---
        # Format: list of (month_number_from_start, barrier_level)
        # The call barrier steps down at these points
        "call_barrier": 1.00,  # initial barrier; stepped_barriers overrides
        "stepped_barriers": [
            (0, 1.00),    # months 1-4: 100% barrier
            (5, 0.95),    # months 5-8 (after 4 quarters): 95% barrier
            (9, 0.90),    # months 9+: 90% barrier
        ],
        "coupon_barrier": 0.75,
        "protection_barrier": 0.75,
        "protection_type": "european_ki",

        # --- Payoffs ---
        "coupon_pa": 0.10,
        "redemption_at_call": 1.00,
        "notional": 1000.0,

        # --- Model params ---
        "default_vol": 0.17,
        "default_spot": 5300.0,
    },

    "Digital Autocall": {
        # --- Metadata ---
        "name": "Digital Autocall",
        "description": (
            "SPX autocall with annual observations and a binary (digital) payoff. "
            "If SPX is above 105% on an observation date, pays a fixed $50 digital "
            "coupon and redeems at par. At maturity: if SPX is at or above initial "
            "level, pays $50 digital + par; if below, 80% capital protection (investor "
            "receives 80 cents on the dollar regardless of SPX return). "
            "Simple structure with capped upside and partial downside protection."
        ),
        "structure_type": "digital",

        # --- Underlying ---
        "underlyings": ["SPX"],
        "n_assets": 1,

        # --- Observation schedule ---
        "obs_frequency": "annual",
        "maturity_years": 3.0,

        # --- Barrier levels ---
        "call_barrier": 1.05,         # 105% trigger (slightly above par)
        "coupon_barrier": 1.05,        # same as call barrier for digital
        "protection_barrier": 1.00,    # capital protection level
        "protection_type": "soft_protection",  # partial guarantee, not knock-in

        # --- Payoffs ---
        "digital_coupon": 50.0,    # fixed dollar amount per observation, not % coupon
        "coupon_pa": None,         # not applicable for digital structure
        "redemption_at_call": 1.00,
        "protection_floor": 0.80,  # 80% capital protection at maturity
        "notional": 1000.0,

        # --- Model params ---
        "default_vol": 0.17,
        "default_spot": 5300.0,
    },
}


# ---------------------------------------------------------------------------
# Access Functions
# ---------------------------------------------------------------------------

def get_security(name: str) -> dict[str, Any]:
    """
    Retrieve a pre-configured security definition by name.

    Args:
        name: Security name string. Must be one of the keys in SECURITIES.

    Returns:
        Dict of security parameters.

    Raises:
        KeyError: If name is not in SECURITIES (with helpful message listing valid names).
    """
    if name not in SECURITIES:
        valid = list(SECURITIES.keys())
        raise KeyError(
            f"Security '{name}' not found. Valid securities: {valid}"
        )
    return SECURITIES[name]


def list_securities() -> list[str]:
    """
    Return the list of available security names in display order.

    WHY: The Streamlit dropdown needs an ordered list. Using dict.keys()
    order preserves the insertion order defined above (Phoenix first —
    it's the simplest and best for introducing the concept).

    Returns:
        Ordered list of security name strings.
    """
    return list(SECURITIES.keys())


def get_security_summary(name: str) -> dict[str, str]:
    """
    Return a compact summary dict suitable for displaying in a Streamlit metric card.

    WHY: The Pricer page shows a "Term Sheet" card with key parameters.
    This function extracts just the display-ready fields so the page code
    doesn't have to know the full parameter schema.

    Args:
        name: Security name.

    Returns:
        Dict with display-friendly key-value pairs.
    """
    sec = get_security(name)

    coupon_str = (
        f"{sec['coupon_pa']*100:.0f}% p.a."
        if sec.get("coupon_pa") is not None
        else f"${sec.get('digital_coupon', 'N/A')} (digital)"
    )

    underlyings = " / ".join(sec["underlyings"])
    obs_freq = sec["obs_frequency"].capitalize()

    summary = {
        "Structure":   sec["structure_type"].replace("_", "-").title(),
        "Underlying":  underlyings,
        "Maturity":    f"{sec['maturity_years']:.0f} years",
        "Observations": obs_freq,
        "Call Barrier": f"{sec['call_barrier']*100:.0f}%",
        "Coupon":       coupon_str,
        "Protection":   f"{sec['protection_barrier']*100:.0f}%",
        "Notional":     f"${sec['notional']:,.0f}",
    }
    return summary
