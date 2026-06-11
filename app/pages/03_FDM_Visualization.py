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
from app.vol_surface import VolSurface

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
    base = "|".join(str(p.get(k)) for k in (
        "security_name", "S0", "r", "q", "sigma", "n_paths", "seed",
    ))
    fdm_top = st.session_state.get("fdm_vol_top", "Flat (Black-Scholes)")
    fdm_sub = st.session_state.get("fdm_vol_local_sub", "Cubic-Spline")
    return base + f"|{fdm_top}|{fdm_sub}"

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

# ── Vol Model Selection (FDM page) ─────────────────────────────────────────────
# FDPricer supports flat and local vol only. Stochastic vol (Heston/Bates) is not
# implemented for the PDE grid — it requires a 3D or dimension-reduction approach.
# Radio buttons mirror the Pricer page so the user can compare FD grids under
# flat vs Dupire local vol without touching the global sidebar setting.
st.markdown("#### Volatility Model")
_fdm_top_opts = ["Flat (Black-Scholes)", "Local Vol Surface"]
_fdm_top_sel = st.radio(
    "Vol Model",
    options=_fdm_top_opts,
    horizontal=True,
    key="fdm_vol_top",
    label_visibility="collapsed",
)

_fdm_dupire_type = "cubic"
if _fdm_top_sel == "Local Vol Surface":
    _fdm_vs_obj = st.session_state.get("vol_surf_obj")
    _fdm_svi_avail = _fdm_vs_obj is not None and getattr(_fdm_vs_obj, "svi_ready", False)
    if not _fdm_svi_avail and st.session_state.get("fdm_vol_local_sub") == "SVI (smooth)":
        st.session_state["fdm_vol_local_sub"] = "Cubic-Spline"
    _fdm_local_opts = ["Cubic-Spline", "SVI (smooth)"] if _fdm_svi_avail else ["Cubic-Spline"]
    _fc1, _fc2 = st.columns([1, 3])
    with _fc1:
        _fdm_local_sel = st.radio(
            "Dupire surface", options=_fdm_local_opts, horizontal=True, key="fdm_vol_local_sub",
        )
    _fdm_dupire_type = "svi" if "SVI" in _fdm_local_sel else "cubic"
    with _fc2:
        if not _fdm_svi_avail:
            st.caption(
                "SVI surface not built — click **Build Vol Surface** on the **Vol Surface** page "
                "to enable the smooth SVI option."
            )
        elif _fdm_dupire_type == "svi":
            st.caption("**SVI Dupire** — per-slice parametric fitting before differentiation "
                       "produces a smoother local vol surface in the FD grid.")
        else:
            st.caption("**Cubic-Spline Dupire** — differentiates the bicubic IV spline "
                       "directly; may be jagged in sparse maturity/strike regions.")
    _fdm_vol_model = "local"
else:
    st.caption("Constant σ (Black-Scholes PDE). Set σ in **④ Model Parameters** in the sidebar.")
    _fdm_vol_model = "flat"

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 3])
with c1:
    run_fdm = st.button("▶ Run FD + Build Grid", type="primary", use_container_width=True)

_vol_label = "Dupire Local Vol" if _fdm_vol_model == "local" else "Flat"  # set by radio above
_vol_str = (
    f"Local Vol ({_fdm_dupire_type.upper()} Dupire)"
    if _fdm_vol_model == "local"
    else f"Flat \u03c3 = {params['sigma']*100:.1f}%"
)
st.caption(
    f"Grid: N\u2093 = {params['N_x']} | N\u03c4 = {params['N_tau']} | x_min = {params['x_min']} | "
    f"Vol: {_vol_str} | "
    f"r = {params['r']*100:.2f}%"
)


# ==============================================================================
# RUN FD PRICER WITH GRID RETURN
# ==============================================================================

if run_fdm or "fdm_result" not in st.session_state or st.session_state.get("fdm_last_run_fp") != _cur_fp_fdm:
    with st.spinner("Running FD pricer with full grid return…"):
        try:
            from scipy.interpolate import RegularGridInterpolator as _RGI
            vol_surface = None
            local_vol_interp = None

            if _fdm_vol_model == "local":
                # Build a cache key that includes snapshot + params + dupire type so
                # switching Cubic↔SVI invalidates the cached interpolator and rebuilds.
                snap_key = params.get("snapshot_key", "")
                _fdm_cache_key = (
                    f"vs_{snap_key}_{params['S0']}_{params['r']}"
                    f"_{params.get('q', 0.014)}_{_fdm_dupire_type}"
                )
                cached = st.session_state.get("vol_surface_cache")
                if cached and cached.get("key") == _fdm_cache_key:
                    vol_surface = cached["vs"]
                    local_vol_interp = cached.get("local_vol_interp")

                if vol_surface is None:
                    # Reuse VolSurface built on the Vol Surface page if available
                    existing_vs = st.session_state.get("vol_surf_obj")
                    if (
                        existing_vs is not None
                        and existing_vs.S0 == params["S0"]
                        and existing_vs.r == params["r"]
                    ):
                        vol_surface = existing_vs
                    else:
                        snap_df = params["snapshot_df"]
                        vol_surface = VolSurface(
                            snap_df, S0=params["S0"], r=params["r"], q=params.get("q", 0.014)
                        )

                if local_vol_interp is None:
                    # Build RegularGridInterpolator for local vol — shared with FDPricer
                    _m_ax = np.linspace(0.40, 1.80, 40)
                    _t_ax = np.linspace(0.01, ac.maturity_years + 0.05, 25)
                    if _fdm_dupire_type == "svi" and getattr(vol_surface, "svi_ready", False):
                        _LV = vol_surface.svi_dupire_local_vol_grid(_m_ax, _t_ax)
                    else:
                        _LV = vol_surface.dupire_local_vol_grid(_m_ax, _t_ax)
                    local_vol_interp = _RGI(
                        (_t_ax, _m_ax), _LV,
                        method="linear", bounds_error=False, fill_value=None,
                    )
                    st.session_state["vol_surface_cache"] = {
                        "key": _fdm_cache_key,
                        "vs": vol_surface,
                        "local_vol_interp": local_vol_interp,
                    }

            fd_vol_model = _fdm_vol_model

            pricer = FDPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                N_x=params["N_x"],
                N_tau=params["N_tau"],
                x_min=params["x_min"],
                vol_model=fd_vol_model,
                vol_surface=vol_surface,
                local_vol_interp=local_vol_interp,
            )
            fd_result = pricer.price(return_grid=True)
            st.session_state["fdm_result"] = fd_result
            st.session_state["fdm_last_run_fp"] = _cur_fp_fdm
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

tab1, tab2, tab3, tab4 = st.tabs(["🌡️ V(S,t) Heatmap", "📉 Price Slice at t", "📐 Greeks", "📊 Scheme Comparison"])

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

        # Delta = dV/dS at t=0 (last column = today, first column = maturity)
        V_t0 = V_grid[:, -1]
        dV = np.gradient(V_t0, S_axis)  # first derivative
        d2V = np.gradient(dV, S_axis)   # second derivative (Gamma)

        # Theta = (V(t=0) - V(t=dt)) / dt — positive = gains value over time at t=0
        # t_axis runs T→0 (maturity→today), so t[-2] > t[-1]
        dt_step = t_axis[-2] - t_axis[-1] if len(t_axis) > 1 else 1.0
        theta_grid = (V_grid[:, -2] - V_grid[:, -1]) / dt_step if len(t_axis) > 1 else np.zeros_like(V_t0)

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


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — SCHEME COMPARISON (Explicit vs Crank-Nicolson)
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Explicit FD vs Crank-Nicolson: Convergence, Stability & Speed")
    st.caption(
        "Both schemes solve the same heat equation. CN is unconditionally stable and "
        "second-order in time; explicit is simpler but requires small time steps."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    run_compare = st.button(
        "▶ Run Scheme Comparison", type="primary", use_container_width=False,
        key="run_scheme_compare",
        help="Prices the autocallable at multiple grid resolutions using both schemes. "
             "Takes ~10-20 seconds."
    )

    if not run_compare and "scheme_compare_results" not in st.session_state:
        st.info(
            "Click **Run Scheme Comparison** above to compute the three panels. "
            "Uses the product and parameters from the sidebar."
        )
        st.stop()

    # ── Computation ───────────────────────────────────────────────────────────
    if run_compare:
        import time as _time

        _ac = params["autocallable"]
        _sigma = params["sigma"]
        _r = params["r"]
        _q = params["q"]

        # --- Panel 1: Convergence — price vs N_x for both schemes -------
        # Fix N_tau proportional to N_x (so both grids are "balanced").
        # CN uses exactly the requested N_tau; explicit auto-corrects if needed.
        convergence_grid_sizes = [50, 100, 200, 400]
        prices_explicit = []
        prices_cn = []

        with st.spinner("Computing convergence chart (8 pricing runs)…"):
            for nx in convergence_grid_sizes:
                # Explicit scheme
                fd_e = FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                                N_x=nx, N_tau=nx, scheme="explicit")
                prices_explicit.append(fd_e.price().price)

                # Crank-Nicolson scheme
                fd_c = FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                                N_x=nx, N_tau=nx, scheme="crank_nicolson")
                prices_cn.append(fd_c.price().price)

        # --- Panel 2: Stability — vary requested N_tau, fix N_x=100 -----
        # We show the *actual* N_tau each scheme uses after auto-correction.
        # Explicit auto-corrects when Courant > 0.5; CN never does.
        # Price is shown vs requested N_tau to illustrate why CN is efficient.
        stability_n_tau_requested = [5, 10, 20, 50, 100, 200]
        stability_nx = 100
        stability_prices_exp = []
        stability_prices_cn = []
        stability_n_tau_actual_exp = []
        stability_n_tau_actual_cn = []
        stability_courants = []

        with st.spinner("Computing stability demo (12 pricing runs)…"):
            for nt in stability_n_tau_requested:
                fd_e = FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                                N_x=stability_nx, N_tau=nt, scheme="explicit")
                fd_c = FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                                N_x=stability_nx, N_tau=nt, scheme="crank_nicolson")

                stability_prices_exp.append(fd_e.price().price)
                stability_prices_cn.append(fd_c.price().price)
                stability_n_tau_actual_exp.append(fd_e.N_tau)  # may be auto-corrected
                stability_n_tau_actual_cn.append(fd_c.N_tau)   # always equals nt
                stability_courants.append(fd_c.courant)        # same grid geometry for both

        # --- Panel 3: Timing — N_x=200, N_tau=200, repeat 3 times each --
        timing_nx = 200
        timing_ntau = 200
        n_timing_reps = 3
        times_exp_ms = []
        times_cn_ms = []

        with st.spinner("Timing comparison (6 pricing runs)…"):
            for _ in range(n_timing_reps):
                t0 = _time.perf_counter()
                FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                         N_x=timing_nx, N_tau=timing_ntau,
                         scheme="explicit").price()
                times_exp_ms.append((_time.perf_counter() - t0) * 1000)

                t0 = _time.perf_counter()
                FDPricer(_ac, sigma=_sigma, r=_r, q=_q,
                         N_x=timing_nx, N_tau=timing_ntau,
                         scheme="crank_nicolson").price()
                times_cn_ms.append((_time.perf_counter() - t0) * 1000)

        # Cache all results so the tab doesn't re-run on every interaction
        st.session_state["scheme_compare_results"] = {
            "convergence_grid_sizes": convergence_grid_sizes,
            "prices_explicit": prices_explicit,
            "prices_cn": prices_cn,
            "stability_n_tau_requested": stability_n_tau_requested,
            "stability_prices_exp": stability_prices_exp,
            "stability_prices_cn": stability_prices_cn,
            "stability_n_tau_actual_exp": stability_n_tau_actual_exp,
            "stability_n_tau_actual_cn": stability_n_tau_actual_cn,
            "stability_courants": stability_courants,
            "stability_nx": stability_nx,
            "times_exp_ms": times_exp_ms,
            "times_cn_ms": times_cn_ms,
            "timing_nx": timing_nx,
            "timing_ntau": timing_ntau,
        }

    # ── Render from cached results ────────────────────────────────────────────
    cmp = st.session_state.get("scheme_compare_results")
    if cmp is None:
        st.stop()

    import pandas as pd

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — CONVERGENCE CHART
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Panel 1 — Price Convergence vs Grid Resolution")
    st.caption(
        "Both schemes converge to the same true price as N_x (spatial grid points) "
        "increases. N_tau = N_x in each run (balanced grid). "
        "CN converges faster per step because it is O(Δτ²) vs O(Δτ) for explicit."
    )

    fig_conv = go.Figure()

    fig_conv.add_trace(go.Scatter(
        x=cmp["convergence_grid_sizes"],
        y=cmp["prices_explicit"],
        mode="lines+markers",
        name="Explicit FD",
        line=dict(color="#EF5350", width=2, dash="dash"),
        marker=dict(size=8, symbol="circle"),
    ))
    fig_conv.add_trace(go.Scatter(
        x=cmp["convergence_grid_sizes"],
        y=cmp["prices_cn"],
        mode="lines+markers",
        name="Crank-Nicolson",
        line=dict(color="#42A5F5", width=2),
        marker=dict(size=8, symbol="diamond"),
    ))

    # Add a reference line at the fine-grid CN price (best estimate of true price)
    best_price = cmp["prices_cn"][-1]
    fig_conv.add_hline(
        y=best_price,
        line=dict(color="gray", dash="dot", width=1),
        annotation_text=f"Fine-grid estimate: ${best_price:,.2f}",
        annotation_position="bottom right",
    )

    fig_conv.update_layout(
        title="Price Convergence: Explicit vs Crank-Nicolson (N_tau = N_x)",
        xaxis=dict(title="Spatial Grid Points (N_x)", type="log",
                   tickvals=cmp["convergence_grid_sizes"],
                   ticktext=[str(n) for n in cmp["convergence_grid_sizes"]]),
        yaxis=dict(title="Autocallable Price ($)"),
        legend=dict(x=0.02, y=0.98),
        height=380,
        margin=dict(t=60, b=50),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Convergence table
    conv_df = pd.DataFrame({
        "N_x (= N_tau)": cmp["convergence_grid_sizes"],
        "Explicit Price ($)": [f"${p:,.2f}" for p in cmp["prices_explicit"]],
        "CN Price ($)": [f"${p:,.2f}" for p in cmp["prices_cn"]],
        "Difference ($)": [f"${abs(e - c):,.2f}" for e, c in
                           zip(cmp["prices_explicit"], cmp["prices_cn"])],
    })
    st.dataframe(conv_df, hide_index=True, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — STABILITY DEMO
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"### Panel 2 — Stability Demo (N_x = {cmp['stability_nx']}, vary N_tau)")
    st.caption(
        "**Explicit FD** requires Courant number ρ = Δτ/Δx² ≤ 0.5 or it becomes "
        "numerically unstable. FDPricer auto-corrects N_tau upward to stay stable, "
        "so the *actual* step count can far exceed what you requested. "
        "**Crank-Nicolson is unconditionally stable** — it uses exactly the requested "
        "N_tau, regardless of Courant number."
    )

    fig_stab = go.Figure()

    fig_stab.add_trace(go.Scatter(
        x=cmp["stability_n_tau_requested"],
        y=cmp["stability_prices_exp"],
        mode="lines+markers",
        name="Explicit FD (price)",
        line=dict(color="#EF5350", width=2, dash="dash"),
        marker=dict(size=8, symbol="circle"),
    ))
    fig_stab.add_trace(go.Scatter(
        x=cmp["stability_n_tau_requested"],
        y=cmp["stability_prices_cn"],
        mode="lines+markers",
        name="Crank-Nicolson (price)",
        line=dict(color="#42A5F5", width=2),
        marker=dict(size=8, symbol="diamond"),
    ))

    fig_stab.update_layout(
        title=f"Price vs Requested N_tau — N_x={cmp['stability_nx']} fixed",
        xaxis=dict(title="Requested N_tau (time steps)"),
        yaxis=dict(title="Autocallable Price ($)"),
        legend=dict(x=0.02, y=0.98),
        height=360,
        margin=dict(t=60, b=50),
    )
    st.plotly_chart(fig_stab, use_container_width=True)

    # Stability table — key column is "Actual N_tau used" by explicit vs CN
    stab_df = pd.DataFrame({
        "Requested N_tau": cmp["stability_n_tau_requested"],
        "Courant ρ": [f"{c:.2f}" for c in cmp["stability_courants"]],
        "Explicit: Actual N_tau": cmp["stability_n_tau_actual_exp"],
        "CN: Actual N_tau": cmp["stability_n_tau_actual_cn"],
        "Explicit Price ($)": [f"${p:,.2f}" for p in cmp["stability_prices_exp"]],
        "CN Price ($)": [f"${p:,.2f}" for p in cmp["stability_prices_cn"]],
    })
    st.dataframe(stab_df, hide_index=True, use_container_width=True)

    st.caption(
        "When Courant ρ > 0.5, the explicit scheme would be numerically unstable if "
        "used as-is. FDPricer auto-corrects N_tau to ensure ρ ≤ 0.4 — note how "
        "**Explicit: Actual N_tau** jumps well above the requested value at coarse "
        "grids. CN needs no such correction and uses exactly the requested step count."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — TIMING COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"### Panel 3 — Timing Comparison  (N_x = {cmp['timing_nx']}, N_tau = {cmp['timing_ntau']})")
    st.caption(
        "At the same grid resolution, CN solves a tridiagonal system (Thomas algorithm, "
        "O(N_x)) at each time step. Explicit does a single vectorized array operation "
        "per step (also O(N_x), but with lower constant). The tradeoff: CN achieves "
        "O(Δτ²) accuracy vs O(Δτ) for explicit — so CN needs fewer steps for the same "
        "accuracy, making it faster overall at production grid sizes."
    )

    avg_exp = np.mean(cmp["times_exp_ms"])
    avg_cn = np.mean(cmp["times_cn_ms"])
    std_exp = np.std(cmp["times_exp_ms"])
    std_cn = np.std(cmp["times_cn_ms"])

    tc1, tc2 = st.columns(2)
    with tc1:
        st.metric(
            label="Explicit FD — avg time",
            value=f"{avg_exp:.1f} ms",
            delta=f"±{std_exp:.1f} ms std",
            delta_color="off",
        )
    with tc2:
        st.metric(
            label="Crank-Nicolson — avg time",
            value=f"{avg_cn:.1f} ms",
            delta=f"±{std_cn:.1f} ms std",
            delta_color="off",
        )

    timing_df = pd.DataFrame({
        "Run": [f"Run {i+1}" for i in range(len(cmp["times_exp_ms"]))],
        "Explicit (ms)": [f"{t:.1f}" for t in cmp["times_exp_ms"]],
        "Crank-Nicolson (ms)": [f"{t:.1f}" for t in cmp["times_cn_ms"]],
    })
    st.dataframe(timing_df, hide_index=True, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # METHODOLOGY NOTE
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Why Crank-Nicolson vs Explicit — Methodology", expanded=False):
        st.markdown("""
**Explicit Finite Difference (used in Paper 1)**

The explicit update rule advances the heat equation one step at a time using only
information from the *current* time level:

$$u^{n+1}_j = u^n_j + \\rho \\left(u^n_{j+1} - 2u^n_j + u^n_{j-1}\\right), \\quad \\rho = \\frac{\\Delta\\tau}{\\Delta x^2}$$

**Requirement**: ρ ≤ ½ (Courant–Friedrichs–Lewy condition). If violated, rounding
errors grow exponentially each step — the scheme is *conditionally stable*.

**Crank-Nicolson**

CN averages the explicit and implicit updates, using information from *both* the
current and next time levels:

$$u^{n+1}_j - u^n_j = \\frac{\\rho}{2}\\left[(u^{n+1}_{j+1} - 2u^{n+1}_j + u^{n+1}_{j-1}) + (u^n_{j+1} - 2u^n_j + u^n_{j-1})\\right]$$

Rearranging gives a **tridiagonal linear system** $A \\cdot u^{n+1} = d$ per time step,
solved in O(N_x) using the Thomas algorithm.

**CN is unconditionally stable** for any ρ — no CFL restriction. It is also
**second-order accurate in time** (O(Δτ²) vs O(Δτ) for explicit), so it achieves the
same accuracy with far fewer time steps.

| Property | Explicit | Crank-Nicolson |
|----------|----------|----------------|
| Stability | Conditional (ρ ≤ 0.5) | Unconditional |
| Time accuracy | O(Δτ) | O(Δτ²) |
| Space accuracy | O(Δx²) | O(Δx²) |
| Per-step cost | O(N_x) — vector add | O(N_x) — Thomas solve |
| Steps needed for same accuracy | High | Low |

**When to use each**

- **Explicit**: Simpler to implement and debug; direct connection to Paper 1's derivation;
  good for education and validation. Fine for moderate grids (N_x ≤ 200) where the
  CFL constraint is not binding.

- **Crank-Nicolson**: Preferred for production use. Fewer time steps at the same
  accuracy level means faster pricing at large grids. Essential when N_x > 500 or
  when very fine time resolution is needed (e.g., daily observation dates).
""")

    # ── Refresh notice ─────────────────────────────────────────────────────────
    st.caption(
        "Results cached for this session. Click **Run Scheme Comparison** again to "
        "recompute with updated sidebar parameters."
    )
