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
from app.heston import (
    HestonModel,
    calibrate_merton,
    calibrate_bates,
    merton_call_price,
    bates_call_price,
)
from app.vol_surface import bs_implied_vol

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

# ── Settings-changed banner ─────────────────────────────────────────────────
def _param_fingerprint_vol_surface(p: dict) -> str:
    return "|".join(str(p.get(k)) for k in (
        "security_name", "vol_model", "S0", "r", "q", "sigma", "n_paths", "seed",
    ))

_cur_fp_vol_surface = _param_fingerprint_vol_surface(params)
_last_fp_vol_surface = st.session_state.get("vol_surface_last_run_fp", None)
if _last_fp_vol_surface is not None and _last_fp_vol_surface != _cur_fp_vol_surface:
    st.warning(
        "⚠️ **Settings have changed since the last calculation.** "
        "Re-run the analysis on this page to update results.",
        icon="🔄",
    )

st.caption(
    f"Data: **{params['snapshot_label']}**  |  "
    f"S₀ = {params['S0']:,.1f}  |  r = {params['r']*100:.2f}%  |  q = {params['q']*100:.2f}%"
)

snap_df = params["snapshot_df"]
if snap_df is None or snap_df.empty:
    st.error("No snapshot data loaded. Check sidebar — Data Date selection.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    run_surf = st.button("📊 Build Vol Surface", type="primary", use_container_width=True)
with c2:
    run_cal = st.button("🔧 Calibrate Heston", use_container_width=True,
                        help="Fits 5 Heston params to snapshot IV data (~15s)")
with c3:
    run_all_models = st.button("🔬 Calibrate All Models", use_container_width=True,
                               help="Calibrate Heston + Merton + Bates and compare fit quality (~45s total)")
with c4:
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["🌐 3D Implied Vol", "📉 Heston Overlay", "🔵 Dupire Local Vol", "📊 Model Comparison"]
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


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL COMPARISON (Heston vs Merton vs Bates)
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Comparison — Heston vs Merton vs Bates")
    st.markdown(
        "Calibrate all three stochastic vol / jump-diffusion models to the same market data "
        "and compare fit quality. Lower RMSE = better smile fit."
    )
    st.caption(
        "**Heston**: 5 params — κ, θ, γ, ρ, v₀  |  "
        "**Merton**: 4 params — σ, λ, μ_J, σ_J  |  "
        "**Bates**: 8 params — Heston + λ, μ_J, σ_J (warm-started from Heston)"
    )

    # ── Trigger calibration ──────────────────────────────────────────────────
    if run_all_models:
        # Step 1: Heston (reuse cache if already done, else run fresh)
        if "heston_cal" not in st.session_state or st.session_state.get("heston_cal") is None:
            with st.spinner("Calibrating Heston (~15s)…"):
                try:
                    heston_cmp = HestonModel(
                        S0=params["S0"], r=params["r"], q=params["q"],
                        v0=params["v0"], kappa=params["kappa"],
                        theta=params["theta"], gamma=params["gamma"], rho=params["rho"],
                    )
                    cal_h = heston_cmp.calibrate(vol_surf, n_sample=80)
                    st.session_state["heston_cal"] = cal_h
                    st.session_state["heston_model_cal"] = heston_cmp
                    st.session_state["heston_cmp_params"] = dict(
                        v0=heston_cmp.v0, kappa=heston_cmp.kappa,
                        theta=heston_cmp.theta, gamma=heston_cmp.gamma, rho=heston_cmp.rho,
                    )
                except Exception as e:
                    st.error(f"Heston calibration failed: {e}")

        # Step 2: Merton
        with st.spinner("Calibrating Merton jump-diffusion (~15s)…"):
            try:
                # Build a market_df with moneyness / ttm_years / impliedVolatility
                mkt_df = snap_df[snap_df["optionType"] == "call"].copy()
                mkt_df = mkt_df.dropna(subset=["impliedVolatility"])
                mkt_df = mkt_df[mkt_df["impliedVolatility"] > 0]
                cal_m = calibrate_merton(mkt_df, S0=params["S0"], r=params["r"], q=params["q"])
                st.session_state["merton_cal"] = cal_m
            except Exception as e:
                st.error(f"Merton calibration failed: {e}")
                st.session_state["merton_cal"] = None

        # Step 3: Bates (warm-start from Heston)
        with st.spinner("Calibrating Bates (Heston + Jumps, ~15s)…"):
            try:
                heston_init = st.session_state.get("heston_cmp_params") or st.session_state.get("heston_cal")
                mkt_df = snap_df[snap_df["optionType"] == "call"].copy()
                mkt_df = mkt_df.dropna(subset=["impliedVolatility"])
                mkt_df = mkt_df[mkt_df["impliedVolatility"] > 0]
                cal_b = calibrate_bates(mkt_df, S0=params["S0"], r=params["r"], q=params["q"],
                                        heston_init=heston_init)
                st.session_state["bates_cal"] = cal_b
            except Exception as e:
                st.error(f"Bates calibration failed: {e}")
                st.session_state["bates_cal"] = None

        st.session_state["vol_surf_last_run_fp"] = _cur_fp_vol_surface
        st.success("All models calibrated. Scroll down to see the comparison.")

    # ── Display comparison ───────────────────────────────────────────────────
    cal_h = st.session_state.get("heston_cal")
    cal_m = st.session_state.get("merton_cal")
    cal_b = st.session_state.get("bates_cal")

    if not any([cal_h, cal_m, cal_b]):
        st.info("👆 Click **Calibrate All Models** to fit Heston, Merton, and Bates to the market surface.")
        st.stop()

    # ── Fit quality table ────────────────────────────────────────────────────
    st.subheader("Fit Quality Comparison")

    import pandas as pd
    rows = []
    heston_obj = st.session_state.get("heston_model_cal")
    if cal_h and heston_obj:
        rows.append({
            "Model": "Heston",
            "Parameters": 5,
            "RMSE (vol pts)": f"{cal_h['rmse']*100:.4f}%",
            "Key params": (
                f"v₀={heston_obj.v0:.4f} κ={heston_obj.kappa:.3f} "
                f"θ={heston_obj.theta:.4f} γ={heston_obj.gamma:.3f} ρ={heston_obj.rho:.3f}"
            ),
        })
    if cal_m:
        rows.append({
            "Model": "Merton",
            "Parameters": 4,
            "RMSE (vol pts)": f"{cal_m['rmse_vol_pts']:.4f}%",
            "Key params": (
                f"σ={cal_m['sigma']:.4f} λ={cal_m['lam']:.3f} "
                f"μ_J={cal_m['mu_J']:.4f} σ_J={cal_m['sig_J']:.4f}"
            ),
        })
    if cal_b:
        rows.append({
            "Model": "Bates",
            "Parameters": 8,
            "RMSE (vol pts)": f"{cal_b['rmse_vol_pts']:.4f}%",
            "Key params": (
                f"v₀={cal_b['v0']:.4f} λ={cal_b['lam']:.3f} "
                f"μ_J={cal_b['mu_J']:.4f} σ_J={cal_b['sig_J']:.4f}"
            ),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "RMSE = root-mean-square error in implied vol (vol points). "
            "Lower is better. Bates typically wins on fit but at the cost of 8 parameters."
        )

    # ── Smile comparison chart ───────────────────────────────────────────────
    st.subheader("Implied Vol Smile Comparison")
    st.caption("Market vs all three models at selected tenors. Dashed = model; solid = market.")

    # Moneyness grid for smile plots
    m_grid = np.linspace(0.75, 1.25, 40)
    S0 = params["S0"]
    r  = params["r"]
    q  = params["q"]

    # Two representative tenors for comparison
    tenor_choices = {"3 months (0.25y)": 0.25, "6 months (0.50y)": 0.50,
                     "1 year (1.0y)": 1.0, "2 years (2.0y)": 2.0}
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t1_label = st.selectbox("Tenor 1", list(tenor_choices.keys()), index=2, key="cmp_t1")
    with col_t2:
        t2_label = st.selectbox("Tenor 2", list(tenor_choices.keys()), index=3, key="cmp_t2")
    T1 = tenor_choices[t1_label]
    T2 = tenor_choices[t2_label]

    def _model_smile(model_name: str, m_grid, T: float) -> list:
        """
        Compute model implied vols across a moneyness grid at maturity T.

        WHY TRY/EXCEPT PER POINT: Near-boundary strikes can produce invalid prices
        (negative, very large) that cause bs_implied_vol to return None. We clip
        to NaN so the chart shows a gap rather than crashing.
        """
        ivs = []
        for m in m_grid:
            K = m * S0
            try:
                if model_name == "heston" and heston_obj:
                    price = heston_obj.european_call(S0, K, T)
                elif model_name == "merton" and cal_m:
                    price = merton_call_price(
                        S0, K, T, r, q,
                        cal_m["sigma"], cal_m["lam"], cal_m["mu_J"], cal_m["sig_J"]
                    )
                elif model_name == "bates" and cal_b:
                    price = bates_call_price(
                        S0, K, T, r, q,
                        cal_b["v0"], cal_b["kappa"], cal_b["theta"],
                        cal_b["gamma"], cal_b["rho"],
                        cal_b["lam"], cal_b["mu_J"], cal_b["sig_J"]
                    )
                else:
                    ivs.append(None)
                    continue
                iv = bs_implied_vol(price, S0, K, T, r, q)
                ivs.append(iv * 100 if iv else None)
            except Exception:
                ivs.append(None)
        return ivs

    def _market_smile(m_grid, T: float, vol_surf) -> list:
        """Market smile from spline interpolation at given tenor."""
        return [vol_surf.implied_vol(m, T) * 100 for m in m_grid]

    # Build figure with two subplots (one per tenor)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Tenor: {t1_label}", f"Tenor: {t2_label}"],
        shared_yaxes=True,
    )

    MODEL_COLORS = {"market": "#333333", "heston": "#2196F3", "merton": "#4CAF50", "bates": "#FF9800"}
    MODEL_DASH   = {"market": "solid",    "heston": "dash",   "merton": "dot",     "bates": "dashdot"}

    for col_idx, T in enumerate([T1, T2], start=1):
        # Market
        try:
            iv_mkt = _market_smile(m_grid, T, vol_surf)
            fig.add_trace(go.Scatter(
                x=m_grid, y=iv_mkt, mode="lines",
                name="Market" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=MODEL_COLORS["market"], width=2.5, dash="solid"),
            ), row=1, col=col_idx)
        except Exception:
            pass

        # Heston
        if heston_obj:
            with st.spinner(f"Computing Heston smile at T={T}y…") if False else st.empty():
                pass
            iv_h = _model_smile("heston", m_grid, T)
            fig.add_trace(go.Scatter(
                x=m_grid, y=iv_h, mode="lines",
                name="Heston" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=MODEL_COLORS["heston"], width=2, dash=MODEL_DASH["heston"]),
            ), row=1, col=col_idx)

        # Merton
        if cal_m:
            iv_m = _model_smile("merton", m_grid, T)
            fig.add_trace(go.Scatter(
                x=m_grid, y=iv_m, mode="lines",
                name="Merton" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=MODEL_COLORS["merton"], width=2, dash=MODEL_DASH["merton"]),
            ), row=1, col=col_idx)

        # Bates
        if cal_b:
            iv_b = _model_smile("bates", m_grid, T)
            fig.add_trace(go.Scatter(
                x=m_grid, y=iv_b, mode="lines",
                name="Bates" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=MODEL_COLORS["bates"], width=2, dash=MODEL_DASH["bates"]),
            ), row=1, col=col_idx)

        fig.update_xaxes(title_text="Moneyness (K/S₀)", row=1, col=col_idx)

    fig.update_yaxes(title_text="Implied Vol (%)", row=1, col=1)
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(t=60, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Parameter detail tables ──────────────────────────────────────────────
    with st.expander("📋 Full Calibrated Parameters", expanded=False):
        col_h, col_m, col_b = st.columns(3)

        with col_h:
            st.markdown("**Heston**")
            if heston_obj and cal_h:
                st.markdown(f"""
| Param | Value |
|---|---|
| v₀ | {heston_obj.v0:.5f} |
| κ | {heston_obj.kappa:.4f} |
| θ | {heston_obj.theta:.5f} |
| γ | {heston_obj.gamma:.4f} |
| ρ | {heston_obj.rho:.4f} |
| RMSE | {cal_h['rmse']*100:.4f}% |
""")
            else:
                st.caption("Not yet calibrated")

        with col_m:
            st.markdown("**Merton**")
            if cal_m:
                st.markdown(f"""
| Param | Value |
|---|---|
| σ | {cal_m['sigma']:.4f} |
| λ | {cal_m['lam']:.4f} |
| μ_J | {cal_m['mu_J']:.4f} |
| σ_J | {cal_m['sig_J']:.4f} |
| RMSE | {cal_m['rmse_vol_pts']:.4f}% |
""")
            else:
                st.caption("Not yet calibrated")

        with col_b:
            st.markdown("**Bates**")
            if cal_b:
                feller_ok = cal_b.get("feller_satisfied", False)
                st.markdown(f"""
| Param | Value |
|---|---|
| v₀ | {cal_b['v0']:.5f} |
| κ | {cal_b['kappa']:.4f} |
| θ | {cal_b['theta']:.5f} |
| γ | {cal_b['gamma']:.4f} |
| ρ | {cal_b['rho']:.4f} |
| λ | {cal_b['lam']:.4f} |
| μ_J | {cal_b['mu_J']:.4f} |
| σ_J | {cal_b['sig_J']:.4f} |
| Feller | {"✅" if feller_ok else "❌"} |
| RMSE | {cal_b['rmse_vol_pts']:.4f}% |
""")
            else:
                st.caption("Not yet calibrated")

    st.caption(
        "**Interpretation**: Bates nests both Heston (set λ=0) and Merton (set v₀=const). "
        "A significantly lower Bates RMSE vs Heston signals that jump risk matters for this market. "
        "Merton RMSE higher than Heston typically indicates mean-reverting variance is important "
        "beyond simple Gaussian jump diffusion."
    )
