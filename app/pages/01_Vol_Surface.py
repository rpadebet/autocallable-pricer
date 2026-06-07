"""
app/pages/01_Vol_Surface.py
============================
Implied Volatility Surface + Heston Model Calibration.

WHAT THIS PAGE SHOWS:
    1. 3D implied vol surface from snapshot data (bicubic spline interpolation).
    2. Heston model implied vol overlay — how well the model fits the market.
    3. Dupire local vol surface (forward-looking, path-consistent vol).
    4. Calibration RMSE and parameter summary.

WHY THREE SURFACES:
    - Implied vol: what the market quotes (observable).
    - Heston: a 5-parameter stochastic vol model that fits the market smile.
    - Dupire local vol: derived from implied vol via the Dupire formula — gives
      the unique vol function σ(S,t) consistent with all market quotes.
    The three together tell the story: market → model → local vol.

KEY PAPERS:
    Paper 2 — Haugh (2013): Heston char function (Eq. 23 only) + Gil-Pelaez.
    Paper 2 — Dupire formula (Eq. 2): σ²_loc = [∂C/∂T + (r-q)K∂C/∂K + qC]
                                                / [K²/2 * ∂²C/∂K²]
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
from app.vol_surface import VolSurface
from app.heston import HestonModel

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vol Surface — AutoCallable Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar ─────────────────────────────────────────────────────────────
params = render_sidebar(page_name="Vol Surface")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Implied Volatility Surface")
st.caption(
    f"Data: **{params['snapshot_label']}**  |  "
    f"S₀ = {params['S0']:,.1f}  |  r = {params['r']*100:.2f}%  |  q = {params['q']*100:.2f}%"
)

snap_df = params["snapshot_df"]
if snap_df is None or snap_df.empty:
    st.error("No snapshot data loaded. Check sidebar — Data Date selection.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    run_surf = st.button("📊 Build Vol Surface", type="primary", use_container_width=True)
with c2:
    run_cal = st.button("🔧 Calibrate Heston", use_container_width=True,
                        help="Fits 5 Heston params to snapshot IV data (~15s)")
with c3:
    show_dupire = st.toggle("Show Dupire Local Vol", value=False,
                            help="Dupire forward-looking vol derived from implied surface")

st.divider()


# ==============================================================================
# BUILD VOL SURFACE
# ==============================================================================

if run_surf or st.session_state.get("vol_built", False):
    st.session_state["vol_built"] = True

    with st.spinner("Fitting bicubic spline to implied vol data…"):
        try:
            vol_surf = VolSurface(
                snapshot=snap_df,
                S0=params["S0"],
                r=params["r"],
                q=params["q"],
            )
            st.session_state["vol_surf_obj"] = vol_surf
        except Exception as e:
            st.error(f"VolSurface error: {e}")
            st.stop()

vol_surf = st.session_state.get("vol_surf_obj", None)

if vol_surf is None:
    st.info("👆 Click **Build Vol Surface** to fit the interpolation surface.")
    st.stop()


# ==============================================================================
# 3D SURFACE PLOT
# ==============================================================================

tab1, tab2, tab3 = st.tabs(
    ["🌐 3D Implied Vol", "📉 Heston Overlay", "🔵 Dupire Local Vol"]
)

with tab1:
    st.subheader("Implied Volatility Surface — Market Data (Spline Interpolation)")

    try:
        M, T, IV = vol_surf.surface_grid(n_moneyness=30, n_ttm=20)

        fig = go.Figure(data=[go.Surface(
            x=M, y=T, z=IV * 100,
            colorscale="RdYlGn_r",
            reversescale=False,
            showscale=True,
            colorbar=dict(title="IV (%)", thickness=15),
            hovertemplate=(
                "Moneyness: %{x:.2f}<br>"
                "TTM: %{y:.2f}y<br>"
                "IV: %{z:.1f}%<extra></extra>"
            ),
        )])

        # Overlay raw data points
        raw_calls = snap_df[snap_df["optionType"] == "call"]
        if not raw_calls.empty:
            fig.add_trace(go.Scatter3d(
                x=raw_calls["moneyness"].values,
                y=raw_calls["ttm_years"].values,
                z=raw_calls["impliedVolatility"].values * 100,
                mode="markers",
                marker=dict(size=2, color="black", opacity=0.5),
                name="Raw data",
            ))

        fig.update_layout(
            title="Implied Vol Surface — Bicubic Spline on SPX Options",
            scene=dict(
                xaxis_title="Moneyness (K/S₀)",
                yaxis_title="TTM (years)",
                zaxis_title="Implied Vol (%)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            height=600,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ATM vol term structure
        st.subheader("ATM Implied Vol Term Structure")
        ttm_range = np.linspace(0.1, 3.0, 30)
        atm_vols = [vol_surf.atm_vol(t) * 100 for t in ttm_range]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ttm_range, y=atm_vols,
            mode="lines+markers",
            name="ATM IV",
            line=dict(color="#2196F3", width=2),
        ))
        fig2.add_hline(
            y=params["sigma"] * 100,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Flat σ used for pricing: {params['sigma']*100:.1f}%",
        )
        fig2.update_layout(
            xaxis_title="TTM (years)",
            yaxis_title="ATM Implied Vol (%)",
            height=300,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Surface plot error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Heston Model Overlay")
    st.caption(
        "Compares Heston model implied vols (from sidebar params) "
        "against the market surface. Run calibration to fit optimally."
    )

    heston = HestonModel(
        S0=params["S0"],
        r=params["r"],
        q=params["q"],
        v0=params["v0"],
        kappa=params["kappa"],
        theta=params["theta"],
        gamma=params["gamma"],
        rho=params["rho"],
    )

    # Calibration button handler
    if run_cal:
        with st.spinner("Calibrating Heston model via differential evolution (~15s)…"):
            try:
                cal_result = heston.calibrate(vol_surf, n_sample=80)
                st.session_state["heston_cal"] = cal_result
                st.session_state["heston_model_cal"] = heston
            except Exception as e:
                st.error(f"Calibration error: {e}")

    cal = st.session_state.get("heston_cal", None)
    if cal:
        col_p, col_r = st.columns([2, 1])
        with col_p:
            st.markdown("**Calibrated Parameters:**")
            st.markdown(f"""
| Param | Value |
|---|---|
| v₀ | {heston.v0:.5f} (σ = {heston.v0**0.5*100:.2f}%) |
| κ | {heston.kappa:.4f} |
| θ | {heston.theta:.5f} (σ∞ = {heston.theta**0.5*100:.2f}%) |
| γ | {heston.gamma:.4f} |
| ρ | {heston.rho:.4f} |
""")
        with col_r:
            st.metric("Calibration RMSE", f"{cal['rmse']*100:.4f}%")
            feller = heston.kappa * heston.theta > 0.5 * heston.gamma**2
            st.metric("Feller Condition", "✅ Met" if feller else "❌ Violated")

    # Heston vs market vol comparison (2D smile for each tenor)
    try:
        M_h, T_h, IV_h = heston.surface_grid(n_moneyness=25, n_ttm=10)

        fig = go.Figure()

        # Market surface (spline) — heatmap style
        M_m, T_m, IV_m = vol_surf.surface_grid(n_moneyness=25, n_ttm=10)

        # Show slices at selected tenors
        tenors_to_show = [3, 6, 9]  # indices into TTM grid
        colors_mkt = ["#2196F3", "#4CAF50", "#9C27B0"]
        colors_hes = ["#0D47A1", "#1B5E20", "#4A148C"]

        for j_idx, color_m, color_h in zip(tenors_to_show, colors_mkt, colors_hes):
            if j_idx >= IV_m.shape[1]:
                continue
            ttm_val = T_m[0, j_idx]
            fig.add_trace(go.Scatter(
                x=M_m[:, j_idx],
                y=IV_m[:, j_idx] * 100,
                mode="lines",
                name=f"Market T={ttm_val:.1f}y",
                line=dict(color=color_m, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=M_h[:, j_idx],
                y=IV_h[:, j_idx] * 100,
                mode="lines",
                name=f"Heston T={ttm_val:.1f}y",
                line=dict(color=color_h, width=2, dash="dash"),
            ))

        fig.update_layout(
            title="Heston vs Market Implied Vol Smiles (selected tenors)",
            xaxis_title="Moneyness (K/S₀)",
            yaxis_title="Implied Vol (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420,
            margin=dict(t=80, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Heston overlay error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Dupire Local Vol Surface")
    st.caption(
        "Forward-looking volatility σ(K,T) consistent with all market option prices. "
        "Paper 2, Eq. 2: σ²_loc = [∂C/∂T + (r-q)K∂C/∂K + qC] / [K²/2 * ∂²C/∂K²]"
    )

    if not show_dupire:
        st.info("Toggle **Show Dupire Local Vol** in the control bar above to enable.")
    else:
        with st.spinner("Computing Dupire local vol surface (numerical differentiation)…"):
            try:
                M_d, T_d, LV_d = vol_surf.dupire_surface_grid(n_moneyness=25, n_ttm=15)

                # Clip extreme values (Dupire can blow up near boundaries)
                LV_d_clipped = np.clip(LV_d * 100, 0.1, 80.0)

                fig = go.Figure(data=[go.Surface(
                    x=M_d, y=T_d, z=LV_d_clipped,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Local Vol (%)", thickness=15),
                    hovertemplate=(
                        "Moneyness: %{x:.2f}<br>"
                        "TTM: %{y:.2f}y<br>"
                        "Local Vol: %{z:.1f}%<extra></extra>"
                    ),
                )])

                fig.update_layout(
                    title="Dupire Local Vol Surface σ_loc(K/S₀, T)",
                    scene=dict(
                        xaxis_title="Moneyness (K/S₀)",
                        yaxis_title="TTM (years)",
                        zaxis_title="Local Vol (%)",
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                    ),
                    height=600,
                    margin=dict(t=60, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Compare local vol vs implied vol at ATM
                ttm_range = np.linspace(0.2, 2.5, 20)
                lv_atm = [vol_surf.dupire_local_vol(1.0, t) * 100 for t in ttm_range]
                iv_atm = [vol_surf.atm_vol(t) * 100 for t in ttm_range]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ttm_range, y=lv_atm,
                               mode="lines", name="Dupire Local Vol (ATM)",
                               line=dict(color="#9C27B0", width=2)))
                fig2.add_trace(go.Scatter(x=ttm_range, y=iv_atm,
                               mode="lines", name="Implied Vol (ATM)",
                               line=dict(color="#2196F3", width=2, dash="dash")))
                fig2.update_layout(
                    title="ATM: Dupire Local Vol vs Implied Vol",
                    xaxis_title="TTM (years)",
                    yaxis_title="Vol (%)",
                    height=300,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(
                    "Note: Local vol is generally lower than implied vol at the money "
                    "(due to the smile's curvature), a well-known result in the literature."
                )

            except Exception as e:
                st.error(f"Dupire surface error: {e}")
                st.caption("Dupire differentiation can fail near the boundary of sparse data regions.")
