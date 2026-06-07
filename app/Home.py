"""
app/Home.py
============
AutoCallable Analytics Platform — Streamlit entry point.

HOW TO RUN:
    streamlit run app/Home.py

    Required working directory: project root (the folder containing app/).
    Streamlit's multi-page app convention: pages/ directory next to Home.py.

WHAT THIS PAGE DOES:
    1. Renders the shared Assumptions sidebar (all model params).
    2. Shows a product summary card for the selected autocallable.
    3. Computes and displays a quick pricing summary using all three methods:
         - FD PDE (deterministic, paper-accurate)
         - Standard MC (industry baseline)
         - Survival MC (Alm et al. 2013 — lower variance)
    4. Shows call probability term structure (bar chart).
    5. Directs users to the specialized pages for deeper analysis.

WHY THREE METHODS ON HOME:
    The headline comparison — three methods agreeing within ~1% — is the key
    validation that the platform is correctly implemented. Showing it on the
    landing page immediately establishes credibility.

NAVIGATION:
    Streamlit renders pages in app/pages/ automatically. The sidebar shows them
    after the widget controls. Pages are:
        01_Vol_Surface.py   — Implied vol surface + Heston calibration
        02_Pricer.py        — Full pricer with MC paths and convergence
        03_FDM_Visualization.py — FD grid visualization
"""

import sys
import os

# Ensure project root is on sys.path so 'from app.xxx import yyy' works
# when Streamlit runs this file directly (it adds app/ to path, not the root).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import traceback

# ── Page config must be the FIRST Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="AutoCallable Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import shared components ───────────────────────────────────────────────────
from app.components.sidebar import render_sidebar
from app.pde_pricer import FDPricer
from app.mc_standard import MCStandardPricer
from app.mc_survival import MCSurvivalPricer


# ==============================================================================
# SIDEBAR — all model assumptions live here
# ==============================================================================
params = render_sidebar(page_name="Home")


# ==============================================================================
# HEADER
# ==============================================================================
st.title("📊 AutoCallable Analytics Platform")
st.caption(
    "Pricing autocallable structured products via Finite Differences, "
    "Standard Monte Carlo, and One-Step Survival MC (Alm et al. 2013)."
)

ac = params["autocallable"]
if ac is None:
    st.error("Failed to initialize product. Check the sidebar parameters.")
    st.stop()


# ==============================================================================
# PRODUCT SUMMARY CARD
# ==============================================================================
st.subheader(f"Selected Product: {params['security_name']}")

sp = params["security_params"]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Maturity", f"{sp.get('maturity_years', 2)}Y")
with col2:
    st.metric("Coupon p.a.", f"{sp.get('coupon_pa', 0)*100:.1f}%")
with col3:
    st.metric("Call Barrier", f"{sp.get('call_barrier', 1.0)*100:.0f}%")
with col4:
    st.metric("Protection Barrier", f"{sp.get('protection_barrier', 0.75)*100:.0f}%")
with col5:
    st.metric("Frequency", sp.get("obs_frequency", "quarterly").title())

# Show description if available
if "description" in sp:
    st.info(sp["description"])

st.divider()


# ==============================================================================
# QUICK PRICING SUMMARY (all three methods)
# ==============================================================================
st.subheader("⚡ Quick Price — Three Methods")
st.caption(
    f"σ = {params['sigma']*100:.1f}%  |  r = {params['r']*100:.2f}%  |  "
    f"q = {params['q']*100:.2f}%  |  S₀ = {params['S0']:,.1f}  |  "
    f"N = {params['n_paths']:,} paths  |  Data: {params['snapshot_label']}"
)

run_col, _ = st.columns([1, 3])
with run_col:
    run_pricing = st.button("🔄 Run Pricing", type="primary", use_container_width=True)

if run_pricing or st.session_state.get("home_priced", False):
    st.session_state["home_priced"] = True

    with st.spinner("Pricing with three methods…"):
        fd_result = mc_result = sv_result = None
        fd_error = mc_error = sv_error = None

        # ── FD Pricer ──
        try:
            fd_pricer = FDPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                N_x=params["N_x"],
                N_tau=params["N_tau"],
                x_min=params["x_min"],
            )
            fd_result = fd_pricer.price(return_grid=False)
        except Exception as e:
            fd_error = str(e)

        # ── Standard MC ──
        try:
            mc_pricer = MCStandardPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                n_paths=params["n_paths"],
                seed=params["seed"],
                antithetic=params["antithetic"],
            )
            mc_result = mc_pricer.price(track_convergence=False)
        except Exception as e:
            mc_error = str(e)

        # ── Survival MC ──
        try:
            sv_pricer = MCSurvivalPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                n_paths=params["n_paths"],
                seed=params["seed"],
            )
            sv_result = sv_pricer.price(track_convergence=False)
        except Exception as e:
            sv_error = str(e)

    # ── Display results ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📐 Finite Difference (PDE)**")
        if fd_result:
            st.metric(
                "FD Price",
                f"${fd_result.price:,.2f}",
                help="Paper 1 — Deng, Mallett, McCann (2011) explicit FD on log-grid",
            )
            # Courant number check
            sigma = params["sigma"]
            T = ac.maturity_years
            dtau = 0.5 * sigma**2 * T / params["N_tau"]
            dx = (abs(params["x_min"]) + 5.0) / params["N_x"]
            courant = dtau / dx**2
            color = "🟢" if courant < 0.5 else "🔴"
            st.caption(f"{color} Courant: {courant:.3f} ({'stable' if courant < 0.5 else 'UNSTABLE'})")
        elif fd_error:
            st.error(f"FD error: {fd_error}")

    with c2:
        st.markdown("**🎲 Standard Monte Carlo**")
        if mc_result:
            st.metric(
                "MC Price",
                f"${mc_result.price:,.2f}",
                delta=f"±${mc_result.std_err:,.2f} (1σ)",
                help="Paper 3 — Alm et al. (2013) §2, Eq. 2.3 — GBM with antithetic variates",
            )
            st.caption(
                f"95% CI: [${mc_result.ci_low:,.2f}, ${mc_result.ci_high:,.2f}]  |  "
                f"N = {mc_result.n_paths:,}"
            )
        elif mc_error:
            st.error(f"MC error: {mc_error}")

    with c3:
        st.markdown("**🎯 Survival MC (Alm et al. 2013)**")
        if sv_result:
            st.metric(
                "Survival Price",
                f"${sv_result.price:,.2f}",
                delta=f"±${sv_result.std_err:,.2f} (1σ)",
                help="Paper 3, Algorithm 1 — analytical barrier handling → smooth payoff → stable Greeks",
            )
            if mc_result and sv_result:
                ratio = mc_result.std_err / sv_result.std_err if sv_result.std_err > 0 else float("nan")
                st.caption(
                    f"95% CI: [${sv_result.ci_low:,.2f}, ${sv_result.ci_high:,.2f}]  |  "
                    f"Variance reduction: {ratio:.2f}×"
                )
        elif sv_error:
            st.error(f"Survival MC error: {sv_error}")

    # ── Method spread ──
    prices_available = [r.price for r in [fd_result, mc_result, sv_result] if r is not None]
    if len(prices_available) >= 2:
        spread = max(prices_available) - min(prices_available)
        spread_pct = spread / np.mean(prices_available) * 100
        spread_color = "🟢" if spread_pct < 2.0 else "🟡" if spread_pct < 5.0 else "🔴"
        st.caption(
            f"{spread_color} Method spread: ${spread:.2f} ({spread_pct:.2f}% of mean)"
            " — within 2% confirms implementation correctness"
        )

    st.divider()

    # ── Call Probability Term Structure ──
    if fd_result and fd_result.call_probs:
        st.subheader("📅 Call Probability Term Structure (FD)")
        obs_dates = fd_result.obs_dates
        call_probs = fd_result.call_probs

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"t={t:.2f}y" for t in obs_dates],
            y=[p * 100 for p in call_probs],
            name="Call Probability",
            marker_color="#1f77b4",
            text=[f"{p*100:.1f}%" for p in call_probs],
            textposition="outside",
        ))
        cumul = np.cumsum(call_probs)
        fig.add_trace(go.Scatter(
            x=[f"t={t:.2f}y" for t in obs_dates],
            y=[c * 100 for c in cumul],
            name="Cumulative Call Prob",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            title="Probability of being called at each observation date",
            xaxis_title="Observation Date",
            yaxis=dict(title="Call Probability (%)", range=[0, max(p * 100 for p in call_probs) * 1.3]),
            yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 110]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=350,
            margin=dict(t=80, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# NAVIGATION LINKS TO ANALYSIS PAGES
# ==============================================================================
st.divider()
st.subheader("📚 Deep-Dive Analysis Pages")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### 📈 Vol Surface")
    st.markdown(
        "3D implied vol surface from snapshot data. "
        "Heston model calibration. Dupire local vol surface. "
        "Model Comparison: Heston vs Merton vs Bates fit quality."
    )
    st.page_link("pages/01_Vol_Surface.py", label="→ Open Vol Surface", icon="📈")

with nav2:
    st.markdown("### 💰 Pricer")
    st.markdown(
        "Full pricing with MC path animation, convergence chart, "
        "barrier overlays, and variance reduction comparison."
    )
    st.page_link("pages/02_Pricer.py", label="→ Open Pricer", icon="💰")

with nav3:
    st.markdown("### 🔲 FDM Grid")
    st.markdown(
        "Finite difference price grid V(S,t) heatmap, "
        "time-step slider, and barrier surface overlay."
    )
    st.page_link("pages/03_FDM_Visualization.py", label="→ Open FDM Viz", icon="🔲")

st.divider()

# ── Additional pages row ─────────────────────────────────────────────────────
nav4, nav5, nav6 = st.columns(3)

with nav4:
    st.markdown("### 📐 Greeks")
    st.markdown(
        "Delta and Vega stability comparison — Standard MC (noisy) "
        "vs Survival MC (smooth) near autocall barriers."
    )
    st.page_link("pages/04_Greeks.py", label="→ Open Greeks", icon="📐")

with nav5:
    st.markdown("### 🎭 Scenarios")
    st.markdown(
        "Payoff diagram, what-if sliders, value surface heatmap, "
        "and call probability by observation date."
    )
    st.page_link("pages/05_Scenarios.py", label="→ Open Scenarios", icon="🎭")

with nav6:
    st.markdown("### 🔧 Product Builder")
    st.markdown(
        "Design a custom autocallable structure with live payoff "
        "preview. Saved products appear in the sidebar."
    )
    st.page_link("pages/06_Product_Builder.py", label="→ Open Builder", icon="🔧")

st.divider()

# ── Concept Guide link ───────────────────────────────────────────────────────
st.markdown("### 📖 Concept Guide")
st.markdown(
    "New to autocallables or any of the models? "
    "The **Concept Guide** explains every feature in plain English — "
    "no formulas, just analogies, what to click, and what to look for."
)
st.info(
    "📄 **AutoCallable_Concept_Guide.html** — open this file in any browser "
    "for a full tutorial covering products, vol models, pricing methods, Greeks, "
    "and a step-by-step demo walkthrough.",
    icon="📖",
)

st.divider()

# ── Technical references footer ──
with st.expander("📄 Technical References"):
    st.markdown("""
**Paper 1** — Deng, Mallett & McCann (2011). *Modeling Autocallable Structured Products.*
Journal of Derivatives. §2.2: Explicit finite difference on log-price grid.

**Paper 2** — Haugh (2013). *The Heston Model and Its Extensions.*
Columbia IEOR lecture notes. Eq. 23: characteristic function (Albrecher form — avoids branch cuts).

**Paper 3** — Alm, Harrach, Harrach & Keller (2013). *A Monte Carlo Pricing Algorithm for Autocallables.*
Journal of Computational Finance. Algorithm 1: One-step survival MC.
    """)
