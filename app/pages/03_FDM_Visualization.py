"""
app/pages/03_FDM_Visualization.py
===================================
Finite Difference Method (PDE) Price Grid Visualization.

WHAT THIS PAGE SHOWS:
    1. V(S,t) heatmap — the full price grid over S and t.
    2. Time-step slider — see how V evolves from maturity backward to t=0.
    3. Barrier overlay — call trigger and protection barrier lines.
    4. Greeks from FD bump — Delta (dV/dS), Gamma (d²V/dS²), Theta (dV/dt).

WHY THIS IS INTERESTING:
    The PDE grid shows HOW the price was computed, not just the final answer.
    At maturity (t=T): the grid matches the product's terminal payoff function.
    At t=0: the single value V(S0, 0) is the fair price.
    The evolution from right to left is backward induction — the pricing algorithm.
    This is the core of Paper 1 (Deng, Mallett, McCann 2011).

TECHNICAL DETAIL:
    FDPricer.price(return_grid=True) returns a V_grid of shape (N_x, n_snapshots)
    where n_snapshots captures V at t=0, each observation date, and t=T.
    The S_axis is the same spatial grid used by the finite difference scheme.
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.sidebar import render_sidebar
from app.pde_pricer import FDPricer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FDM Visualization — AutoCallable Analytics",
    page_icon="🔲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar ─────────────────────────────────────────────────────────────
params = render_sidebar(page_name="FDM Visualization")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔲 Finite Difference PDE Grid Visualization")

# ── Settings-changed banner ─────────────────────────────────────────────────
def _param_fingerprint_fdm(p: dict) -> str:
    return "|".join(str(p.get(k)) for k in (
        "security_name", "vol_model", "S0", "r", "q", "sigma", "n_paths", "seed",
    ))

_cur_fp_fdm = _param_fingerprint_fdm(params)
_last_fp_fdm = st.session_state.get("fdm_last_run_fp", None)
if _last_fp_fdm is not None and _last_fp_fdm != _cur_fp_fdm:
    st.warning(
        "⚠️ **Settings have changed since the last calculation.** "
        "Re-run the analysis on this page to update results.",
        icon="🔄",
    )

st.caption(
    "Price function V(S,t) computed via explicit FD backward induction. "
    "Paper 1 — Deng, Mallett & McCann (2011), §2.2."
)

ac = params["autocallable"]
if ac is None:
    st.error("Product initialization failed. Check sidebar parameters.")
    st.stop()

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 3])
with c1:
    run_fdm = st.button("▶ Run FD + Build Grid", type="primary", use_container_width=True)

st.caption(
    f"Grid: Nₓ = {params['N_x']} | Nτ = {params['N_tau']} | x_min = {params['x_min']} | "
    f"σ = {params['sigma']*100:.1f}% | r = {params['r']*100:.2f}%"
)


# ==============================================================================
# RUN FD PRICER WITH GRID RETURN
# ==============================================================================

if run_fdm:
    with st.spinner("Running FD pricer with full grid return…"):
        try:
            pricer = FDPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                N_x=params["N_x"],
                N_tau=params["N_tau"],
                x_min=params["x_min"],
            )
            fd_result = pricer.price(return_grid=True)
            st.session_state["fdm_result"] = fd_result
            st.session_state["fdm_S_axis"] = pricer._S_axis if hasattr(pricer, "_S_axis") else None
        except Exception as e:
            st.error(f"FD pricer error: {e}")
            import traceback
            st.code(traceback.format_exc())

fd_res = st.session_state.get("fdm_result", None)

if fd_res is None:
    st.info("👆 Click **Run FD + Build Grid** to compute the price surface.")
    st.stop()

# Price summary at top
col_p, col_c, col_g = st.columns(3)
with col_p:
    st.metric("FD Price (V at S₀, t=0)", f"${fd_res.price:,.2f}")
with col_c:
    sigma = params["sigma"]; T = ac.maturity_years
    dtau = 0.5 * sigma**2 * T / params["N_tau"]
    dx = (abs(params["x_min"]) + 5.0) / params["N_x"]
    courant = dtau / dx**2
    st.metric("Courant Number", f"{courant:.4f}", delta="STABLE" if courant < 0.5 else "UNSTABLE")
with col_g:
    if fd_res.greeks:
        st.metric("Delta (Δ)", f"{fd_res.greeks.get('delta', 'N/A'):.4f}",
                  help="dV/dS at S₀")

st.divider()


# ==============================================================================
# TABS
# ==============================================================================

tab1, tab2, tab3 = st.tabs(["🌡️ V(S,t) Heatmap", "📉 Price Slice at t", "📐 Greeks"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — HEATMAP
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Price Grid V(S, t) — Full Heatmap")

    V_grid = fd_res.V_grid
    S_axis = fd_res.S_axis
    t_axis = fd_res.t_axis

    if V_grid is None or S_axis is None or t_axis is None:
        st.warning(
            "Price grid not available. The FDPricer may not have returned the grid. "
            "Check FDResult.V_grid."
        )
    else:
        # V_grid shape: (N_x, n_snapshots) where t_axis has n_snapshots time points
        # Filter to reasonable S range for display (0.5 to 1.5 × S_ref)
        S_ref = ac.S_ref
        s_lo = 0.5 * S_ref
        s_hi = 1.8 * S_ref
        mask_s = (S_axis >= s_lo) & (S_axis <= s_hi)
        S_disp = S_axis[mask_s]
        V_disp = V_grid[mask_s, :]

        fig = go.Figure(data=go.Heatmap(
            x=[f"{t:.2f}y" for t in t_axis],
            y=S_disp,
            z=V_disp,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="V(S,t) $", thickness=15),
            hovertemplate=(
                "t = %{x}<br>"
                "S = %{y:,.0f}<br>"
                "V = $%{z:,.2f}<extra></extra>"
            ),
        ))

        # Call barrier line
        call_barrier_abs = ac.call_barrier_at_period(0) * S_ref
        fig.add_hline(
            y=call_barrier_abs,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Call barrier: ${call_barrier_abs:,.0f}",
            annotation_position="top right",
        )
        # Protection barrier
        prot_barrier_abs = ac.protection_barrier * S_ref
        fig.add_hline(
            y=prot_barrier_abs,
            line=dict(color="orange", dash="dash", width=2),
            annotation_text=f"Protection: ${prot_barrier_abs:,.0f}",
            annotation_position="bottom right",
        )
        # S_ref line
        fig.add_hline(
            y=S_ref,
            line=dict(color="white", dash="dot", width=1),
            annotation_text=f"S₀ = {S_ref:,.0f}",
        )

        fig.update_layout(
            title="V(S, t) — Price function across spot levels and time",
            xaxis=dict(title="Time (backward induction: T → 0)",
                       autorange="reversed"),
            yaxis=dict(title="Spot Level S"),
            height=550,
            margin=dict(t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Reading the heatmap: at t=T (far right), V equals the terminal payoff. "
            "At t=0 (far left), V at S=S₀ is the fair price. "
            "Jumps at observation dates (red vertical bands) show autocall exercise."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — PRICE SLICE SLIDER
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("V(S) Slice at Selected Time")
    st.caption("Use the slider to scrub through time and see how the price function changes.")

    V_grid = fd_res.V_grid
    S_axis = fd_res.S_axis
    t_axis = fd_res.t_axis

    if V_grid is None or S_axis is None or t_axis is None:
        st.warning("Grid not available.")
    else:
        n_snaps = len(t_axis)
        t_idx = st.slider(
            "Time snapshot index",
            min_value=0,
            max_value=n_snaps - 1,
            value=n_snaps - 1,
            format="%d",
            help="Index 0 = maturity (terminal payoff). Max index = t=0 (pricing date, fair value).",
        )
        t_label = t_axis[t_idx] if t_idx < len(t_axis) else "?"
        st.caption(f"Showing V(S, t={t_label:.3f}y)  [snapshot index {t_idx} of {n_snaps-1}]")

        # Filter to display range
        S_ref = ac.S_ref
        mask_s = (S_axis >= 0.4 * S_ref) & (S_axis <= 2.0 * S_ref)
        S_disp = S_axis[mask_s]
        V_slice = V_grid[mask_s, t_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_disp, y=V_slice,
            mode="lines",
            name=f"V(S, t={t_label:.3f}y)",
            line=dict(color="#2196F3", width=2),
        ))

        # Mark S0
        # Find V at S0 by interpolation
        v_at_s0 = float(np.interp(S_ref, S_disp, V_slice))
        fig.add_trace(go.Scatter(
            x=[S_ref], y=[v_at_s0],
            mode="markers",
            marker=dict(color="red", size=10, symbol="circle"),
            name=f"V(S₀) = ${v_at_s0:,.2f}",
        ))

        # Barrier lines
        call_barrier_abs = ac.call_barrier_at_period(0) * S_ref
        prot_barrier_abs = ac.protection_barrier * S_ref
        fig.add_vline(x=call_barrier_abs, line_dash="dash", line_color="red",
                      annotation_text="Call barrier")
        fig.add_vline(x=prot_barrier_abs, line_dash="dash", line_color="orange",
                      annotation_text="Protection barrier")

        fig.update_layout(
            title=f"V(S, t = {t_label:.3f}y) — Price function at this time slice",
            xaxis_title="Spot Level S",
            yaxis_title="V(S, t) — Fair Value ($)",
            height=420,
            margin=dict(t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — GREEKS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Greeks from FD Grid")
    st.caption(
        "Delta and Gamma computed numerically from the price grid at t=0. "
        "Theta from the t=0 vs t=dt time difference."
    )

    V_grid = fd_res.V_grid
    S_axis = fd_res.S_axis
    t_axis = fd_res.t_axis

    if V_grid is None or S_axis is None or len(t_axis) < 2:
        st.info("Greeks require the full grid. Rerun with a larger Nτ if needed.")
    else:
        S_ref = ac.S_ref

        # Delta = dV/dS at t=0 (index 0)
        V_t0 = V_grid[:, 0]
        dV = np.gradient(V_t0, S_axis)  # first derivative
        d2V = np.gradient(dV, S_axis)   # second derivative (Gamma)

        # Theta = (V(t=dt) - V(t=0)) / dt — positive = loses value over time
        dt_step = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 1.0
        theta_grid = (V_grid[:, 1] - V_grid[:, 0]) / dt_step if len(t_axis) > 1 else np.zeros_like(V_t0)

        # Filter to display range
        mask_s = (S_axis >= 0.5 * S_ref) & (S_axis <= 1.8 * S_ref)
        S_disp = S_axis[mask_s]
        V_disp = V_t0[mask_s]
        delta_disp = dV[mask_s]
        gamma_disp = d2V[mask_s]
        theta_disp = theta_grid[mask_s]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Price V(S, t=0)", "Delta Δ = dV/dS",
                            "Gamma Γ = d²V/dS²", "Theta Θ = dV/dt"),
        )

        fig.add_trace(go.Scatter(x=S_disp, y=V_disp, name="V",
                      line=dict(color="#2196F3", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=S_disp, y=delta_disp, name="Δ",
                      line=dict(color="#4CAF50", width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=S_disp, y=gamma_disp, name="Γ",
                      line=dict(color="#FF5722", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=S_disp, y=theta_disp, name="Θ",
                      line=dict(color="#9C27B0", width=2)), row=2, col=2)

        # Mark S0 on all subplots
        for r, c, arr in [(1,1,V_disp), (1,2,delta_disp), (2,1,gamma_disp), (2,2,theta_disp)]:
            v_at_s0 = float(np.interp(S_ref, S_disp, arr))
            fig.add_trace(go.Scatter(
                x=[S_ref], y=[v_at_s0],
                mode="markers",
                marker=dict(color="red", size=8),
                showlegend=False,
            ), row=r, col=c)

        for r in [1, 2]:
            for c in [1, 2]:
                fig.add_vline(x=S_ref, line_dash="dot", line_color="gray", row=r, col=c)

        fig.update_layout(
            title="Greeks at t=0 — from FD Price Grid",
            height=560,
            showlegend=False,
            margin=dict(t=80, b=40),
        )
        fig.update_xaxes(title_text="Spot S", row=2, col=1)
        fig.update_xaxes(title_text="Spot S", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # At S0 summary
        v_s0 = float(np.interp(S_ref, S_disp, V_disp))
        d_s0 = float(np.interp(S_ref, S_disp, delta_disp))
        g_s0 = float(np.interp(S_ref, S_disp, gamma_disp))
        th_s0 = float(np.interp(S_ref, S_disp, theta_disp))

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("V(S₀)", f"${v_s0:,.2f}")
        mc2.metric("Δ at S₀", f"{d_s0:.4f}")
        mc3.metric("Γ at S₀", f"{g_s0:.6f}")
        mc4.metric("Θ at S₀ (per year)", f"${th_s0:,.2f}")

        st.caption(
            "Delta interpretation: a 1-point increase in S changes the price by ~Δ dollars. "
            "Gamma: rate of change of delta (convexity). Theta: time decay."
        )
