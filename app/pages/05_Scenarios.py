"""
app/pages/05_Scenarios.py
==========================
Scenarios page: payoff intuition, what-if analysis, and product understanding.

PURPOSE:
    The Pricer page (02) is about methodology — three pricing methods, convergence,
    variance reduction. This page is about intuition — what does this product
    actually pay, how does price move with market conditions, and what would have
    happened historically?

    A good test: could a non-quant understand this page? It should be visual,
    labelled clearly, and avoid jargon where possible.

SECTIONS:
    1. Payoff Diagram — shows the three payoff regimes at maturity
    2. What-If Sliders — live repricing as spot/vol/rates change
    3. Call Probability Table — term structure of autocall risk
    4. Value Surface — price as a function of spot × vol (heatmap)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.sidebar import render_sidebar
from app.mc_standard import MCStandardPricer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Scenarios — AutoCallable Analytics",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

params = render_sidebar("Scenarios")
ac = params["autocallable"]
S0 = params["S0"]
sigma = params["sigma"]
r = params["r"]
q = params["q"]

if ac is None:
    st.error("Product initialization failed. Check sidebar parameters.")
    st.stop()

if ac.structure_type == "worst_of":
    st.warning(
        "⚠️ **Worst-of basket pricing is not yet implemented.** "
        "The current pricers treat this as a single-underlying SPX autocallable. "
        "See the Pricer page for details."
    )

# ---------------------------------------------------------------------------
# Helper: fast MC price (low N, for interactive sliders)
# ---------------------------------------------------------------------------

def quick_price(ac, sigma, r, q, spot=None, n_paths=3000, seed=42):
    """Price quickly for interactive use. Returns price as float."""
    return MCStandardPricer(
        ac, sigma=sigma, r=r, q=q,
        n_paths=n_paths, seed=seed, antithetic=True,
        spot_override=spot,
    ).price().price


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🎯 Scenarios & Payoff Analysis")

# ── Settings-changed banner ─────────────────────────────────────────────────
def _param_fingerprint_scenarios(p: dict) -> str:
    return "|".join(str(p.get(k)) for k in (
        "security_name", "vol_model", "S0", "r", "q", "sigma", "n_paths", "seed",
    ))

_cur_fp_scenarios = _param_fingerprint_scenarios(params)
_last_fp_scenarios = st.session_state.get("scenarios_last_run_fp", None)
if _last_fp_scenarios is not None and _last_fp_scenarios != _cur_fp_scenarios:
    st.warning(
        "⚠️ **Settings have changed since the last calculation.** "
        "Re-run the analysis on this page to update results.",
        icon="🔄",
    )

st.markdown(
    "Understand *what this product actually pays* and how price responds "
    "to changes in market conditions."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Payoff Diagram",
    "🎚️ What-If Analysis",
    "📅 Call Probability",
    "🗺️ Value Surface",
])

# ===========================================================================
# TAB 1: PAYOFF DIAGRAM
# ===========================================================================

with tab1:
    st.subheader("Payoff at Maturity — The Three Regimes")
    st.markdown(
        "This chart shows what an investor receives *if the product reaches maturity* "
        "(i.e., was never autocalled early). The final spot determines which regime applies."
    )

    # Compute payoff across a range of final spot levels
    spot_range = np.linspace(S0 * 0.30, S0 * 1.50, 500)
    # knocked_in=True when final spot < protection barrier — standard payoff
    # diagram convention: if you end below the barrier, assume it was hit.
    payoffs = np.array([
        ac.terminal_payoff(s, knocked_in=(s < ac.protection_barrier * ac.S_ref), r=r)
        for s in spot_range
    ])
    moneyness = spot_range / ac.S_ref

    # Identify regime boundaries
    call_level = ac.call_barrier * ac.S_ref       # call trigger
    prot_level = ac.protection_barrier * ac.S_ref  # knock-in threshold

    fig = go.Figure()

    # Shade the three regions
    fig.add_vrect(x0=0, x1=ac.protection_barrier,
                  fillcolor="rgba(239,83,80,0.08)", line_width=0,
                  annotation_text="Loss zone", annotation_position="top left")
    fig.add_vrect(x0=ac.protection_barrier, x1=ac.call_barrier,
                  fillcolor="rgba(255,193,7,0.08)", line_width=0,
                  annotation_text="Protected zone", annotation_position="top left")
    fig.add_vrect(x0=ac.call_barrier, x1=moneyness.max() + 0.1,
                  fillcolor="rgba(76,175,80,0.08)", line_width=0,
                  annotation_text="Called zone", annotation_position="top left")

    # Payoff line
    fig.add_trace(go.Scatter(
        x=moneyness, y=payoffs,
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        name="Maturity payoff ($)",
        hovertemplate="Spot: %{x:.2f}× S_ref<br>Payoff: $%{y:,.2f}<extra></extra>",
    ))

    # Notional line (reference)
    fig.add_hline(y=ac.notional, line_dash="dot", line_color="grey",
                  annotation_text=f"Par (${ac.notional:,.0f})", annotation_position="right")

    # Barrier verticals
    fig.add_vline(x=ac.call_barrier, line_dash="dash", line_color="#4CAF50",
                  annotation_text=f"Call barrier ({ac.call_barrier:.0%})",
                  annotation_position="top")
    fig.add_vline(x=ac.protection_barrier, line_dash="dash", line_color="#EF5350",
                  annotation_text=f"Protection ({ac.protection_barrier:.0%})",
                  annotation_position="top")

    fig.update_layout(
        xaxis_title="Final Spot / S_ref (moneyness at maturity)",
        yaxis_title="Payoff ($)",
        xaxis_tickformat=".0%",
        height=460,
        title=f"Maturity Payoff — {ac.name}",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Regime explanation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        **🔴 Loss zone** (spot < {ac.protection_barrier:.0%})

        Knock-in event occurred. Investor receives proportional SPX return —
        full downside exposure. No coupon buffer applies.
        """)
    with col2:
        st.markdown(f"""
        **🟡 Protected zone** ({ac.protection_barrier:.0%} – {ac.call_barrier:.0%})

        Knock-in did NOT occur. Investor receives par (${ac.notional:,.0f}).
        The protection barrier absorbed the downside.
        """)
    with col3:
        coupon_total = ac.coupon_per_period() * ac.n_observations()
        st.markdown(f"""
        **🟢 Called zone** (spot ≥ {ac.call_barrier:.0%} at any obs. date)

        Autocall triggered. Investor receives par + coupon earned to call date.
        Total max coupon over life: ${coupon_total:,.0f}.
        """)


# ===========================================================================
# TAB 2: WHAT-IF SLIDERS
# ===========================================================================

with tab2:
    st.subheader("What-If Analysis — Live Repricing")
    st.markdown("Adjust market conditions and see how the price moves. Baseline uses sidebar values.")

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        spot_pct = st.slider(
            "Spot level (% of S_ref)",
            min_value=60, max_value=140, value=100, step=1,
            help="Current spot as % of the trade-date reference level"
        )
        vol_pct = st.slider(
            "Implied vol (σ)",
            min_value=5, max_value=60, value=int(sigma * 100), step=1,
            help="Flat vol used for pricing"
        )
        rate_pct = st.slider(
            "Risk-free rate (r)",
            min_value=0, max_value=10, value=int(r * 100), step=1,
            help="Continuously compounded risk-free rate"
        )

        run_whatif = st.button("▶ Reprice", type="primary", use_container_width=True)
        st.caption(f"N=3,000 paths — fast approximation")

    with col_result:
        if run_whatif:
            spot_override = ac.S_ref * (spot_pct / 100)
            vol_override = vol_pct / 100
            r_override = rate_pct / 100

            with st.spinner("Pricing…"):
                base_price = quick_price(ac, sigma, r, q, spot=S0)
                new_price = quick_price(ac, vol_override, r_override, q,
                                        spot=spot_override)

            change = new_price - base_price
            change_pct = (change / base_price) * 100

            st.metric(
                label="Base price (sidebar params)",
                value=f"${base_price:,.2f}",
            )
            st.metric(
                label="What-if price",
                value=f"${new_price:,.2f}",
                delta=f"${change:+,.2f} ({change_pct:+.1f}%)",
                delta_color="normal",
            )

            # Mini breakdown
            st.markdown("---")
            st.markdown(f"""
            **Parameter changes:**
            - Spot: {100}% → {spot_pct}% of S_ref
            - Vol: {sigma:.0%} → {vol_override:.0%}
            - Rate: {r:.1%} → {r_override:.1%}
            """)
        else:
            st.info("Set sliders and click **▶ Reprice**.")

            # Show base price while waiting
            with st.spinner("Loading base price…"):
                base_price = quick_price(ac, sigma, r, q, spot=S0)
            st.metric("Base price (sidebar params)", f"${base_price:,.2f}")


# ===========================================================================
# TAB 3: CALL PROBABILITY TABLE
# ===========================================================================

with tab3:
    st.subheader("Call Probability — Term Structure of Autocall Risk")
    st.markdown(
        "Probability of being called at each observation date, conditional on not "
        "having been called at any prior date. From the analytical formula in Paper 1 §2.2."
    )

    try:
        probs = ac.call_probabilities(sigma=sigma, r=r, q=q)
        obs_dates = ac.observation_dates()

        cum_prob = np.cumsum(probs)
        survival = 1.0 - cum_prob

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Probability of call at each date (conditional)",
                "Cumulative probability of having been called",
            ]
        )

        bar_colors = [f"rgba(31,119,180,{0.4 + 0.6 * p / max(probs)})" for p in probs]

        fig.add_trace(go.Bar(
            x=[f"t={d:.2f}yr" for d in obs_dates],
            y=probs,
            marker_color=bar_colors,
            name="Call prob (conditional)",
            hovertemplate="%{x}<br>P(call here) = %{y:.1%}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[f"t={d:.2f}yr" for d in obs_dates],
            y=cum_prob,
            mode="lines+markers",
            line=dict(color="#4CAF50", width=2),
            marker=dict(size=8),
            name="Cumulative call prob",
            hovertemplate="%{x}<br>P(called by here) = %{y:.1%}<extra></extra>",
        ), row=1, col=2)

        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            height=400,
            showlegend=False,
            title=f"Call Probability Term Structure — {ac.name} (σ={sigma:.0%}, r={r:.1%})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        import pandas as pd
        df = pd.DataFrame({
            "Obs. Date (yr)": [f"{d:.3f}" for d in obs_dates],
            "P(call at this date)": [f"{p:.2%}" for p in probs],
            "P(called by this date)": [f"{c:.2%}" for c in cum_prob],
            "P(still alive)": [f"{s:.2%}" for s in survival],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            f"Expected life: {sum(d * p for d, p in zip(obs_dates, probs)) + obs_dates[-1] * survival[-1]:.2f} years "
            f"(probability-weighted average time to call or maturity)"
        )

    except Exception as e:
        st.warning(f"Could not compute call probabilities: {e}")


# ===========================================================================
# TAB 4: VALUE SURFACE
# ===========================================================================

with tab4:
    st.subheader("Value Surface — Price vs Spot × Volatility")
    st.markdown(
        "How does the autocallable price move as spot and vol change simultaneously? "
        "Each cell is a separate MC price (N=1,000 paths for speed)."
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        run_surface = st.button("▶ Compute Surface", key="btn_surface",
                                 type="primary", use_container_width=True)
        n_spot = st.slider("Spot grid points", 5, 12, 7)
        n_vol = st.slider("Vol grid points", 5, 10, 6)
        st.caption(f"Total: {n_spot * n_vol} MC runs at N=1,000 paths each")
        st.markdown("""
        **Reading the surface:**
        - Brighter = higher price
        - Near the call barrier (spot ≈ 100%): vol has less impact — near-certain call
        - Deep OTM (spot < protection): vol increases probability of knock-in → price falls
        """)

    if run_surface or "scenario_surface" in st.session_state:
        cache_key = f"{params['security_name']}_{S0}_{r}_{q}"
        if run_surface or st.session_state.get("surface_key") != cache_key:
            spot_grid = np.linspace(S0 * 0.70, S0 * 1.20, n_spot)
            vol_grid = np.linspace(0.10, 0.45, n_vol)

            prices = np.zeros((n_vol, n_spot))
            total = n_vol * n_spot

            with col1:
                prog = st.progress(0, text="Computing surface…")
                k = 0
                for i, vol in enumerate(vol_grid):
                    for j, spot in enumerate(spot_grid):
                        prices[i, j] = quick_price(
                            ac, vol, r, q, spot=spot, n_paths=1000, seed=42
                        )
                        k += 1
                        prog.progress(k / total, text=f"Run {k}/{total}…")
                prog.empty()

            st.session_state["scenario_surface"] = prices
            st.session_state["scenario_spots"] = spot_grid
            st.session_state["scenario_vols"] = vol_grid
            st.session_state["surface_key"] = cache_key

        prices = st.session_state["scenario_surface"]
        spot_grid = st.session_state["scenario_spots"]
        vol_grid = st.session_state["scenario_vols"]

        with col1:
            spot_labels = [f"{s/ac.S_ref:.0%}" for s in spot_grid]
            vol_labels = [f"{v:.0%}" for v in vol_grid]

            fig = go.Figure(go.Heatmap(
                z=prices,
                x=spot_labels,
                y=vol_labels,
                colorscale="RdYlGn",
                colorbar=dict(title="Price ($)"),
                hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Price: $%{z:,.2f}<extra></extra>",
                text=[[f"${p:,.0f}" for p in row] for row in prices],
                texttemplate="%{text}",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                xaxis_title="Spot / S_ref",
                yaxis_title="Implied Vol (σ)",
                title=f"Value Surface — {ac.name}",
                height=460,
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        with col1:
            st.info("Click **▶ Compute Surface** to generate the price heatmap.")
            st.caption(
                "Each cell is an independent MC run (N=1,000 paths). "
                "Expect ~30 seconds for a 7×6 grid on Streamlit Cloud."
            )
