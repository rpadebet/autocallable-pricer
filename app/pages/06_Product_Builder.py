"""
app/pages/06_Product_Builder.py
================================
Custom Autocallable Product Builder — interactive form for defining a new
autocallable structure and saving it to session_state for use on the Pricer page.

WHY THIS PAGE EXISTS:
    The Pricer page ships with 4 pre-configured securities (Phoenix, Worst-Of,
    Step-Down, Digital). This page lets the user define their own product and
    immediately price it without editing any code.

    Once saved, the custom security appears in the sidebar "② Autocallable Product"
    dropdown across all pages, prefixed with ✏️ to distinguish it from the pre-built
    structures.

DESIGN DECISIONS:
    - All parameters map 1-to-1 to the AutoCallable dataclass fields.
    - A live payoff diagram updates as the user changes parameters, giving immediate
      intuition about how each parameter changes the risk/return profile.
    - Custom securities are stored in st.session_state["custom_securities"] (a dict
      keyed by user-defined name). They persist for the duration of the browser session.
    - The sidebar reads from this dict and shows them in the security dropdown.
    - Step-down barriers: user specifies a list of (observation_index, barrier) pairs.
    - Basket/worst-of: only single-asset is supported in the builder (basket requires
      multi-asset correlation config, deferred to a future enhancement).

LIMITATIONS (future work):
    - Custom securities are not persisted across browser sessions (no file save/load).
    - Basket support (worst-of with multiple underlyings) is not included.
    - Coupons are always continuous (no discrete coupon schedule customization).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.sidebar import render_sidebar
from app.autocallable import AutoCallable

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Product Builder", layout="wide")
st.title("🔧 Custom Product Builder")
st.caption(
    "Design your own autocallable structure. Once saved, it appears in the sidebar "
    "security dropdown and can be priced on the Pricer page."
)

# Render sidebar (provides shared params but we mostly use it for navigation context)
params = render_sidebar("Product Builder")

# Settings-changed banner (shared pattern across all pages)
_last_fp = st.session_state.get("builder_last_save_fp")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Custom security registry in session_state
# ─────────────────────────────────────────────────────────────────────────────
if "custom_securities" not in st.session_state:
    st.session_state["custom_securities"] = {}

custom_secs = st.session_state["custom_securities"]

# ─────────────────────────────────────────────────────────────────────────────
# Product Definition Form
# ─────────────────────────────────────────────────────────────────────────────
col_form, col_preview = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("① Product Terms")

    # Product name
    prod_name = st.text_input(
        "Product Name",
        value="My Custom Autocall",
        help="This name appears in the sidebar dropdown (prefixed with ✏️).",
    )

    # Structure type
    structure_type = st.selectbox(
        "Structure Type",
        options=["phoenix", "step_down", "digital"],
        format_func=lambda x: {
            "phoenix":   "Phoenix Autocall — standard conditional coupon",
            "step_down": "Step-Down Barrier — barrier declines each period",
            "digital":   "Digital Autocall — fixed dollar payoff at call",
        }[x],
        help=(
            "Phoenix: standard autocall with conditional coupon. "
            "Step-Down: call barrier ratchets down each period. "
            "Digital: pays a fixed dollar amount (not a rate) when called."
        ),
    )

    st.divider()
    st.subheader("② Schedule & Maturity")

    col_m, col_f = st.columns(2)
    with col_m:
        maturity_years = st.selectbox(
            "Maturity",
            options=[1, 2, 3, 4, 5],
            index=1,
            format_func=lambda x: f"{x} year{'s' if x > 1 else ''}",
        )
    with col_f:
        obs_frequency = st.selectbox(
            "Observation Frequency",
            options=["monthly", "quarterly", "semi-annual", "annual"],
            index=1,
        )

    # Compute and display observation dates
    from app.autocallable import _observation_dates_from_params, FREQ_TO_MONTHS
    obs_dates = _observation_dates_from_params(maturity_years, obs_frequency)
    st.caption(f"→ {len(obs_dates)} observation dates: {[round(t, 3) for t in obs_dates[:6]]}{'…' if len(obs_dates) > 6 else ''}")

    st.divider()
    st.subheader("③ Barrier Levels")
    st.caption("All barriers are expressed as fractions of the reference spot (1.0 = 100%).")

    col_cb, col_pb = st.columns(2)
    with col_cb:
        call_barrier = st.slider(
            "Call Barrier",
            min_value=0.70, max_value=1.30, value=1.00, step=0.01,
            format="%.2f",
            help="Product autocalls if spot/S₀ ≥ this level at an observation date.",
        )
    with col_pb:
        protection_barrier = st.slider(
            "Knock-In Barrier",
            min_value=0.50, max_value=0.99, value=0.75, step=0.01,
            format="%.2f",
            help="If spot ever closes below this at ANY observation date, "
                 "the protection is breached and the investor bears losses at maturity.",
        )

    coupon_barrier = st.slider(
        "Coupon Barrier (for Phoenix — same as call if not conditional)",
        min_value=0.50, max_value=1.10, value=call_barrier, step=0.01,
        format="%.2f",
        help="Conditional coupon is paid at each observation where spot/S₀ ≥ this level.",
    )

    st.divider()
    st.subheader("④ Coupon / Payoff")

    if structure_type == "digital":
        digital_coupon = st.number_input(
            "Digital Coupon ($, paid at autocall)",
            min_value=1.0, max_value=500.0, value=50.0, step=5.0,
            help="Fixed dollar amount paid when the autocall triggers.",
        )
        coupon_pa = 0.0  # not used for digital
    else:
        digital_coupon = None
        coupon_pa = st.slider(
            "Conditional Coupon (% p.a.)",
            min_value=1, max_value=30, value=8, step=1,
            format="%d%%",
            help="Annual coupon rate. Paid pro-rata at each observation date if coupon barrier is met.",
        ) / 100.0

    notional = st.number_input(
        "Notional ($)",
        min_value=100.0, max_value=10_000.0, value=1_000.0, step=100.0,
        help="Face value of the note.",
    )

    # Step-down barriers for step_down structure
    stepped_barriers = []
    if structure_type == "step_down":
        st.divider()
        st.subheader("⑤ Step-Down Schedule")
        st.caption(
            "Define how the call barrier declines over time. "
            "Each row: observation index (1-based) and the barrier at that date."
        )
        n_steps_for_stepdown = max(1, len(obs_dates) // 3)  # default: 3 step-down points
        n_rows = st.number_input("Number of step-down points", min_value=1, max_value=10,
                                 value=min(3, len(obs_dates)), step=1)
        for i in range(int(n_rows)):
            c1, c2 = st.columns(2)
            with c1:
                obs_idx = st.number_input(
                    f"Obs index #{i+1} (1-based)",
                    min_value=1, max_value=len(obs_dates),
                    value=min(int((i + 1) * len(obs_dates) // int(n_rows)), len(obs_dates)),
                    key=f"sd_idx_{i}",
                )
            with c2:
                barrier_val = st.slider(
                    f"Barrier at obs #{obs_idx}",
                    min_value=0.70, max_value=1.10,
                    value=round(call_barrier - i * 0.05, 2),
                    step=0.01, format="%.2f",
                    key=f"sd_bar_{i}",
                )
            stepped_barriers.append((int(obs_idx), float(barrier_val)))

    # Protection type
    protection_type = st.selectbox(
        "Protection Type",
        options=["european_ki", "soft_protection"],
        format_func=lambda x: {
            "european_ki":     "European knock-in put (most common)",
            "soft_protection": "Soft protection / capital floor",
        }[x],
        help=(
            "European KI put: investor loses S_T/S₀ of notional if barrier was breached. "
            "Soft protection: floor is applied at maturity (investor gets back at least the floor%)."
        ),
    )
    protection_floor = 0.80
    if protection_type == "soft_protection":
        protection_floor = st.slider(
            "Protection Floor",
            min_value=0.50, max_value=1.00, value=0.80, step=0.01, format="%.2f",
            help="Minimum redemption fraction at maturity regardless of spot performance.",
        )

    st.divider()

    # ── Save button ──────────────────────────────────────────────────────────
    save_col, clear_col = st.columns([2, 1])
    with save_col:
        save_clicked = st.button("💾 Save to Sidebar", type="primary", use_container_width=True,
                                 help="Saves this product to the session. It will appear in the sidebar dropdown.")
    with clear_col:
        if custom_secs:
            clear_name = st.selectbox("Delete saved", ["—"] + list(custom_secs.keys()), key="clear_name")
            if st.button("🗑 Delete", use_container_width=True) and clear_name != "—":
                del st.session_state["custom_securities"][clear_name]
                st.success(f"Deleted '{clear_name}'")
                st.rerun()

    if save_clicked:
        if not prod_name.strip():
            st.error("Please enter a product name.")
        elif prod_name in [s for s in custom_secs]:
            st.warning(f"Overwriting existing custom security '{prod_name}'.")

        # Build the sec_params dict (same format as securities.py returns)
        new_sec = {
            "name":               prod_name.strip(),
            "underlying":         "SPX",
            "structure_type":     structure_type,
            "maturity_years":     float(maturity_years),
            "obs_frequency":      obs_frequency,
            "call_barrier":       float(call_barrier),
            "coupon_barrier":     float(coupon_barrier),
            "coupon_pa":          float(coupon_pa),
            "digital_coupon":     digital_coupon,
            "protection_barrier": float(protection_barrier),
            "protection_type":    protection_type,
            "protection_floor":   float(protection_floor),
            "notional":           float(notional),
            "stepped_barriers":   stepped_barriers,
        }
        st.session_state["custom_securities"][prod_name.strip()] = new_sec
        st.session_state["security_name"] = f"✏️ {prod_name.strip()}"
        st.success(
            f"✅ Saved **{prod_name}** — now available in the sidebar dropdown. "
            "Navigate to the **Pricer** page to price it."
        )

    # Show existing custom securities
    if custom_secs:
        st.markdown(f"**Saved custom structures** ({len(custom_secs)}):")
        for nm, sp in custom_secs.items():
            st.caption(
                f"• **{nm}**: {sp['structure_type']} · T={sp['maturity_years']}y · "
                f"{sp['obs_frequency']} · call@{int(sp['call_barrier']*100)}%"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Right column: live payoff diagram
# ─────────────────────────────────────────────────────────────────────────────
with col_preview:
    st.subheader("② Live Payoff Preview")
    st.caption(
        "Payoff at maturity as a function of final spot/S₀. "
        "Assumes the autocall did NOT trigger on any earlier observation date."
    )

    # Build a temporary AutoCallable from current form state
    try:
        tmp_ac = AutoCallable(
            name=prod_name.strip() or "Preview",
            structure_type=structure_type,
            S_ref=params["S0"],
            maturity_years=float(maturity_years),
            obs_frequency=obs_frequency,
            call_barrier=float(call_barrier),
            coupon_barrier=float(coupon_barrier),
            coupon_pa=float(coupon_pa) if not digital_coupon else 0.0,
            digital_coupon=digital_coupon,
            protection_barrier=float(protection_barrier),
            protection_type=protection_type,
            protection_floor=float(protection_floor),
            notional=float(notional),
            stepped_barriers=stepped_barriers,
        )
    except Exception as e:
        st.warning(f"Cannot preview — invalid parameters: {e}")
        tmp_ac = None

    if tmp_ac is not None:
        S0 = params["S0"]
        spot_range = np.linspace(0.40 * S0, 1.60 * S0, 300)
        moneyness  = spot_range / S0

        # ── Compute maturity payoff across spot range ─────────────────────
        # WHY: We use a simplified payoff: no path-dependence (no early call),
        # just the maturity leg. This is what the investor receives if the
        # product runs to full maturity.
        #
        # Payoff logic (mirrors autocallable.py):
        #   1. If spot >= protection_barrier → return notional (capital protected)
        #   2. If spot < protection_barrier  → return spot/S_ref * notional (KI loss)
        #   3. Digital: fixed coupon paid if spot >= call_barrier at maturity

        def _maturity_payoff(m: float, ac: AutoCallable) -> float:
            """
            Terminal payoff at maturity (no early call scenario).
            m = S_T / S_ref.
            """
            if ac.structure_type == "digital":
                if m >= ac.call_barrier:
                    return ac.notional + (ac.digital_coupon or 0.0)
                elif m >= ac.protection_barrier:
                    return ac.notional
                else:
                    return m * ac.notional

            elif ac.protection_type == "soft_protection":
                base = max(m, ac.protection_floor) * ac.notional
                return base

            else:  # european_ki
                if m < ac.protection_barrier:
                    # Knock-in: investor bears the downside
                    return m * ac.notional
                else:
                    # No knock-in: capital returned
                    return ac.notional

        payoffs = [_maturity_payoff(m, tmp_ac) for m in moneyness]

        # ── Also show the autocall payoff (if called at a given level) ────
        # At any observation date: if spot >= call_barrier → receives notional
        # + coupon for the period. We show this as a horizontal band.
        period_coupon = (
            (tmp_ac.coupon_pa / (12 / FREQ_TO_MONTHS[obs_frequency]) * 12)
            * tmp_ac.notional
            if not tmp_ac.digital_coupon
            else (tmp_ac.digital_coupon or 0.0)
        )
        autocall_payoff = tmp_ac.notional + period_coupon

        # ── Build figure ──────────────────────────────────────────────────
        fig = go.Figure()

        # Maturity payoff curve
        fig.add_trace(go.Scatter(
            x=moneyness * 100, y=payoffs,
            mode="lines", name="Maturity Payoff",
            line=dict(color="#2196F3", width=3),
        ))

        # Autocall level (horizontal line): max payoff if called early
        fig.add_hline(
            y=autocall_payoff,
            line=dict(color="#4CAF50", width=1.5, dash="dash"),
            annotation_text=f"Autocall payoff: ${autocall_payoff:,.0f}",
            annotation_position="top right",
        )

        # Notional line
        fig.add_hline(
            y=tmp_ac.notional,
            line=dict(color="#9E9E9E", width=1, dash="dot"),
            annotation_text=f"Notional: ${tmp_ac.notional:,.0f}",
            annotation_position="bottom right",
        )

        # Shade the knock-in zone
        fig.add_vrect(
            x0=40, x1=tmp_ac.protection_barrier * 100,
            fillcolor="#FF5722", opacity=0.07,
            annotation_text="KI zone", annotation_position="top left",
        )

        # Call barrier line
        fig.add_vline(
            x=tmp_ac.call_barrier * 100,
            line=dict(color="#FF9800", width=1.5, dash="dash"),
            annotation_text=f"Call barrier ({int(tmp_ac.call_barrier*100)}%)",
            annotation_position="top",
        )

        # KI barrier line
        fig.add_vline(
            x=tmp_ac.protection_barrier * 100,
            line=dict(color="#FF5722", width=1.5, dash="dash"),
            annotation_text=f"KI ({int(tmp_ac.protection_barrier*100)}%)",
            annotation_position="top left",
        )

        fig.update_layout(
            xaxis_title="Final Spot / S₀ (%)",
            yaxis_title="Payoff ($)",
            height=350,
            margin=dict(t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Key metrics table ─────────────────────────────────────────────
        st.subheader("Key Structure Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Maturity", f"{maturity_years}y")
        col_m2.metric("Observations", str(len(obs_dates)))
        col_m3.metric("Max payoff (autocall)", f"${autocall_payoff:,.0f}")
        col_m4.metric("Max loss (full KI)", f"${0.40 * tmp_ac.notional:,.0f}")

        # ── Observation schedule table ────────────────────────────────────
        with st.expander("📅 Observation Date Schedule", expanded=False):
            import pandas as pd

            # Effective call barrier per date (handles step-down)
            sd_map = dict(stepped_barriers)
            barriers_per_date = []
            for i, t in enumerate(obs_dates):
                b = sd_map.get(i + 1, call_barrier)
                barriers_per_date.append(b)

            schedule_df = pd.DataFrame({
                "Observation": [f"#{i+1}" for i in range(len(obs_dates))],
                "Time (years)": [round(t, 4) for t in obs_dates],
                "Call Barrier": [f"{int(b*100)}%" for b in barriers_per_date],
                "Coupon (if barrier met)": [
                    f"${tmp_ac.notional * tmp_ac.coupon_pa / (12 / FREQ_TO_MONTHS[obs_frequency]) * 12:.2f}"
                    if not tmp_ac.digital_coupon else f"${tmp_ac.digital_coupon:.2f}"
                    for _ in obs_dates
                ],
            })
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        # ── Methodology note ──────────────────────────────────────────────
        with st.expander("📐 Payoff Formula", expanded=False):
            if structure_type == "digital":
                st.markdown("""
**Digital Autocall — Payoff at Maturity (no early call)**

| Condition | Payoff |
|---|---|
| S_T/S₀ ≥ call_barrier | Notional + Digital Coupon |
| S_T/S₀ ≥ KI barrier | Notional (capital protected) |
| S_T/S₀ < KI barrier | (S_T/S₀) × Notional (loss) |
""")
            elif protection_type == "soft_protection":
                st.markdown(f"""
**Soft Protection — Payoff at Maturity**

Payoff = max(S_T/S₀, {protection_floor:.0%}) × Notional

Investor always receives at least {protection_floor:.0%} of notional at maturity,
regardless of how far the spot falls.
""")
            else:
                st.markdown(f"""
**Phoenix / Step-Down — Payoff at Maturity (European KI)**

| Condition | Payoff |
|---|---|
| Never crossed KI barrier | Notional (${notional:,.0f}) |
| Crossed KI barrier (S < {int(protection_barrier*100)}% of S₀ at any obs) | (S_T/S₀) × Notional |

**Conditional coupon** = paid at each obs date where S ≥ coupon_barrier × S₀:

Coupon per period = {coupon_pa*100:.1f}% p.a. × (Δt) × Notional

At maturity, investor receives the period coupon + notional redemption (or KI loss).
""")
