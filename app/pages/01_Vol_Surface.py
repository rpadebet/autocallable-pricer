"""
app/pages/01_Vol_Surface.py
============================
Implied Volatility Surface + Model Calibration + Dupire Local Vol.

PAGE STRUCTURE (3 tabs):
    Tab 1 — Build Vol Surface
        Builds the bicubic-spline implied vol surface from the selected
        snapshot. Shows a 3D interactive plot, ATM term structure, and
        a CSV download of the implied vol surface.

    Tab 2 — Calibrate Models
        Calibrates Heston (5 params), Merton (4 params), and Bates (8 params)
        to the market surface in one run. Displays individual fit metrics,
        the Heston overlay smile, and a cross-model comparison chart.
        (Previously "Calibrate Heston" and "Calibrate All Models" were
        separate buttons; they are merged here because all three models
        share the same vol surface input and the user typically needs
        all three for a complete comparison.)

    Tab 3 — Dupire Vol Surface
        Shows both Dupire surfaces side by side with download buttons:
          • Cubic-Spline Dupire: derived directly from the bicubic spline.
            Numerically differentiating a 2D spline can produce visible
            jaggedness — an honest reflection of the market's data sparsity
            and the limits of polynomial interpolation.
          • SVI Dupire: derived after fitting a per-slice SVI parametric
            model to each expiry. SVI captures only the 5 degrees of freedom
            genuinely present in each smile (level, slope, curvature, etc.),
            discarding bid-ask noise before differentiation, yielding a
            materially smoother Dupire surface.

KEY PAPERS:
    Paper 2 — Haugh (2013): Dupire formula (Eq. 2), Heston char fn (Eq. 23).
    SVI — Gatheral (2004): "A parsimonious arbitrage-free implied volatility
          parametrization with application to the valuation of volatility
          derivatives."
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

# ── Timing guide ────────────────────────────────────────────────────────────
st.caption(
    "⏱ **Expected run times:** "
    "Build Vol Surface ~3 s  ·  "
    "Calibrate All Models ~60–120 s (3 sequential optimizations)  ·  "
    "Build SVI Surface ~10 s  ·  "
    "Dupire Local Vol ~5 s"
)

st.divider()

# ==============================================================================
# TABS
# ==============================================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Build Vol Surface",
    "🔬 Calibrate Models",
    "📐 Dupire Vol Surface",
])

# ==============================================================================
# TAB 1 — BUILD VOL SURFACE
# ==============================================================================

with tab1:
    st.subheader("Build Implied Vol Surface")
    st.caption(
        "Fits a bicubic spline to SPX option implied vols from the selected snapshot. "
        "All pricing on this page and the Pricer page uses this surface."
    )

    run_surf = st.button("📊 Build Vol Surface", type="primary", key="btn_build_surf",
                         use_container_width=False,
                         help="Fits spline to snapshot — ~3 s")

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
    else:
        try:
            M, T, IV = vol_surf.surface_grid(n_moneyness=30, n_ttm=20)

            # ── 3D Surface plot ─────────────────────────────────────────────
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

            # ── ATM term structure ──────────────────────────────────────────
            st.subheader("ATM Implied Vol Term Structure")
            ttm_range = np.linspace(0.1, 3.0, 30)
            atm_vols = [vol_surf.atm_vol(t) * 100 for t in ttm_range]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ttm_range, y=atm_vols,
                mode="lines+markers",
                name="ATM IV (spline)",
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

            # ── Download button ─────────────────────────────────────────────
            st.subheader("Download Implied Vol Surface")
            csv_iv = vol_surf.to_csv_implied_vol(n_moneyness=30, n_ttm=20)
            st.download_button(
                label="⬇ Download Implied Vol Surface (CSV)",
                data=csv_iv,
                file_name="implied_vol_surface.csv",
                mime="text/csv",
                help="Rows = TTM values, columns = moneyness levels, cells = implied vol (%)",
            )
            st.caption(
                "Layout: rows = TTM (years), columns = moneyness (K/S₀), cells = implied vol (%).  "
                "Source: bicubic spline fitted to snapshot data."
            )

        except Exception as e:
            st.error(f"Surface plot error: {e}")


# ==============================================================================
# TAB 2 — CALIBRATE MODELS
# ==============================================================================

with tab2:
    # Guard: need vol surface first
    vol_surf = st.session_state.get("vol_surf_obj", None)
    if vol_surf is None:
        st.info("👈 Go to **Build Vol Surface** tab first to fit the spline, then come back here.")
        st.stop()

    st.subheader("Calibrate Models — Heston · Merton · Bates")
    st.markdown(
        "Calibrates all three stochastic vol / jump-diffusion models to the same market data "
        "and compares fit quality. Lower RMSE = better smile fit."
    )
    st.caption(
        "**Heston**: 5 params — κ, θ, γ, ρ, v₀  |  "
        "**Merton**: 4 params — σ, λ, μ_J, σ_J  |  "
        "**Bates**: 8 params — Heston + λ, μ_J, σ_J (warm-started from Heston)"
    )

    run_all_models = st.button(
        "🔬 Calibrate All Models (Heston + Merton + Bates)",
        type="primary",
        key="btn_cal_all",
        help="Runs 3 sequential calibrations — ~60–120 s total"
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

    # ── Trigger calibration ──────────────────────────────────────────────────
    if run_all_models:
        st.info(
            "🔬 **Calibrating 3 models sequentially** — Heston → Merton → Bates. "
            "Each step runs its own numerical optimizer. "
            "**Total expected time: 60–120 seconds.** "
            "Watch the step labels below — the app is working, not frozen.",
            icon="⏳",
        )

        # Step 1: Heston
        with st.spinner("Step 1 of 3 — Calibrating Heston (5 params: κ, θ, γ, ρ, v₀)… ~15–45 s"):
            try:
                cal_h = heston.calibrate(vol_surf, n_sample=80)
                st.session_state["heston_cal"] = cal_h
                st.session_state["heston_model_cal"] = heston
                st.session_state["heston_cmp_params"] = dict(
                    v0=heston.v0, kappa=heston.kappa,
                    theta=heston.theta, gamma=heston.gamma, rho=heston.rho,
                )
            except Exception as e:
                st.error(f"Heston calibration failed: {e}")

        # Step 2: Merton
        with st.spinner("Step 2 of 3 — Calibrating Merton (4 params: σ, λ, μ_J, σ_J)… ~15–30 s"):
            try:
                mkt_df = snap_df[snap_df["optionType"] == "call"].copy()
                mkt_df = mkt_df.dropna(subset=["impliedVolatility"])
                mkt_df = mkt_df[mkt_df["impliedVolatility"] > 0]
                cal_m = calibrate_merton(mkt_df, S0=params["S0"], r=params["r"], q=params["q"])
                st.session_state["merton_cal"] = cal_m
            except Exception as e:
                st.error(f"Merton calibration failed: {e}")
                st.session_state["merton_cal"] = None

        # Step 3: Bates (warm-start from Heston)
        with st.spinner("Step 3 of 3 — Calibrating Bates (8 params, warm-started)… ~15–30 s"):
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
        st.success("✅ All 3 models calibrated. Scroll down to see results.")

    # ── Display results ───────────────────────────────────────────────────────
    cal_h = st.session_state.get("heston_cal")
    cal_m = st.session_state.get("merton_cal")
    cal_b = st.session_state.get("bates_cal")
    heston_obj = st.session_state.get("heston_model_cal")

    if not any([cal_h, cal_m, cal_b]):
        st.info("👆 Click **Calibrate All Models** to fit Heston, Merton, and Bates.")

    else:
        # ── Heston individual params ────────────────────────────────────────
        if cal_h and heston_obj:
            st.subheader("Heston Parameters")
            col_p, col_r = st.columns([2, 1])
            with col_p:
                st.markdown(f"""
| Param | Value |
|---|---|
| v₀ | {heston_obj.v0:.5f} (σ = {heston_obj.v0**0.5*100:.2f}%) |
| κ | {heston_obj.kappa:.4f} |
| θ | {heston_obj.theta:.5f} (σ∞ = {heston_obj.theta**0.5*100:.2f}%) |
| γ | {heston_obj.gamma:.4f} |
| ρ | {heston_obj.rho:.4f} |
""")
            with col_r:
                st.metric("Heston RMSE", f"{cal_h['rmse']*100:.4f}%")
                feller = heston_obj.kappa * heston_obj.theta > 0.5 * heston_obj.gamma**2
                st.metric("Feller Condition", "✅ Met" if feller else "❌ Violated")

        # ── Fit quality table ───────────────────────────────────────────────
        st.subheader("Fit Quality Comparison")
        import pandas as pd
        rows = []
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
                "Lower is better. Bates typically wins on fit but costs 8 parameters."
            )

        # ── Heston overlay ──────────────────────────────────────────────────
        heston_for_overlay = heston_obj if heston_obj else heston
        st.subheader("Heston vs Market Implied Vol Smiles")
        try:
            M_h, T_h, IV_h = heston_for_overlay.surface_grid(n_moneyness=25, n_ttm=10)
            M_m, T_m, IV_m = vol_surf.surface_grid(n_moneyness=25, n_ttm=10)

            fig = go.Figure()
            tenors_to_show = [3, 6, 9]
            colors_mkt = ["#2196F3", "#4CAF50", "#9C27B0"]
            colors_hes = ["#0D47A1", "#1B5E20", "#4A148C"]

            for j_idx, color_m, color_h in zip(tenors_to_show, colors_mkt, colors_hes):
                if j_idx >= IV_m.shape[1]:
                    continue
                ttm_val = T_m[0, j_idx]
                fig.add_trace(go.Scatter(
                    x=M_m[:, j_idx], y=IV_m[:, j_idx] * 100,
                    mode="lines", name=f"Market T={ttm_val:.1f}y",
                    line=dict(color=color_m, width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=M_h[:, j_idx], y=IV_h[:, j_idx] * 100,
                    mode="lines", name=f"Heston T={ttm_val:.1f}y",
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

        # ── Multi-model smile comparison ────────────────────────────────────
        st.subheader("Implied Vol Smile Comparison — All Models")
        m_grid = np.linspace(0.75, 1.25, 40)
        S0 = params["S0"]
        r  = params["r"]
        q  = params["q"]

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
            (negative, very large) that cause bs_implied_vol to return None.
            """
            ivs = []
            for m in m_grid:
                K = m * S0
                try:
                    if model_name == "heston" and heston_for_overlay:
                        price = heston_for_overlay.european_call(S0, K, T)
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

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"Tenor: {t1_label}", f"Tenor: {t2_label}"],
            shared_yaxes=True,
        )
        MODEL_COLORS = {"market": "#333333", "heston": "#2196F3", "merton": "#4CAF50", "bates": "#FF9800"}
        MODEL_DASH   = {"market": "solid",    "heston": "dash",   "merton": "dot",     "bates": "dashdot"}

        for col_idx, T in enumerate([T1, T2], start=1):
            try:
                iv_mkt = [vol_surf.implied_vol(m, T) * 100 for m in m_grid]
                fig.add_trace(go.Scatter(
                    x=m_grid, y=iv_mkt, mode="lines",
                    name="Market" if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    line=dict(color=MODEL_COLORS["market"], width=2.5, dash="solid"),
                ), row=1, col=col_idx)
            except Exception:
                pass

            if heston_for_overlay:
                iv_h = _model_smile("heston", m_grid, T)
                fig.add_trace(go.Scatter(
                    x=m_grid, y=iv_h, mode="lines",
                    name="Heston" if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    line=dict(color=MODEL_COLORS["heston"], width=2, dash=MODEL_DASH["heston"]),
                ), row=1, col=col_idx)

            if cal_m:
                iv_m = _model_smile("merton", m_grid, T)
                fig.add_trace(go.Scatter(
                    x=m_grid, y=iv_m, mode="lines",
                    name="Merton" if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    line=dict(color=MODEL_COLORS["merton"], width=2, dash=MODEL_DASH["merton"]),
                ), row=1, col=col_idx)

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

        # ── Parameter detail ─────────────────────────────────────────────────
        with st.expander("📋 Full Calibrated Parameters", expanded=False):
            col_h, col_m, col_b_col = st.columns(3)
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

            with col_b_col:
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
            "**Interpretation**: Bates nests both Heston (λ=0) and Merton (v₀=const). "
            "A significantly lower Bates RMSE vs Heston signals that jump risk matters. "
            "Merton RMSE higher than Heston typically indicates mean-reverting variance "
            "is more important than simple Gaussian jump diffusion."
        )


# ==============================================================================
# TAB 3 — DUPIRE VOL SURFACE
# ==============================================================================

with tab3:
    # Guard: need vol surface first
    vol_surf = st.session_state.get("vol_surf_obj", None)
    if vol_surf is None:
        st.info("👈 Go to **Build Vol Surface** tab first, then come back here.")
        st.stop()

    st.subheader("Dupire Local Vol Surface")
    st.markdown(
        "The **Dupire local vol** σ_loc(K, T) is the unique forward-looking volatility "
        "function consistent with all observed option prices simultaneously (Paper 2, Eq. 2):\n\n"
        r"$$\sigma^2_{loc}(T, K) = \frac{\partial C/\partial T + (r-q)K\,\partial C/\partial K + qC}{\frac{K^2}{2}\,\partial^2 C/\partial K^2}$$"
    )
    st.markdown(
        "The **key insight**: how you *interpolate* implied vols between market quotes "
        "determines how smooth the Dupire surface looks — because Dupire differentiates "
        "twice w.r.t. strike K."
    )

    # ── Section 1: Cubic-Spline Dupire ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1️⃣  Cubic-Spline Dupire — Honest interpolation artifact")
    st.caption(
        "Derived by differentiating the bicubic spline fitted directly to raw market quotes. "
        "The spline must balance all data points simultaneously. "
        "Where data is sparse or noisy, d²C/dK² inherits the spline's wiggles — "
        "producing the characteristic 'jagged' local vol surface. "
        "**This is not a bug; it is an honest reflection of data sparsity and interpolation limits.**"
    )

    with st.spinner("Computing cubic-spline Dupire local vol surface…"):
        try:
            M_d, T_d, LV_d = vol_surf.dupire_surface_grid(n_moneyness=25, n_ttm=15)
            LV_d_clipped = np.clip(LV_d * 100, 0.1, 80.0)

            fig_cs = go.Figure(data=[go.Surface(
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
            fig_cs.update_layout(
                title="Cubic-Spline Dupire Local Vol σ_loc(K/S₀, T)",
                scene=dict(
                    xaxis_title="Moneyness (K/S₀)",
                    yaxis_title="TTM (years)",
                    zaxis_title="Local Vol (%)",
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                ),
                height=560,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_cs, use_container_width=True)

            # ATM comparison
            ttm_range = np.linspace(0.2, 2.5, 20)
            lv_atm_cs = [vol_surf.dupire_local_vol(1.0, t) * 100 for t in ttm_range]
            iv_atm    = [vol_surf.atm_vol(t) * 100 for t in ttm_range]

            fig_atm = go.Figure()
            fig_atm.add_trace(go.Scatter(x=ttm_range, y=lv_atm_cs,
                               mode="lines", name="Cubic-Spline Dupire (ATM)",
                               line=dict(color="#9C27B0", width=2)))
            fig_atm.add_trace(go.Scatter(x=ttm_range, y=iv_atm,
                               mode="lines", name="Implied Vol (ATM)",
                               line=dict(color="#2196F3", width=2, dash="dash")))
            fig_atm.update_layout(
                title="ATM: Cubic-Spline Dupire vs Implied Vol",
                xaxis_title="TTM (years)", yaxis_title="Vol (%)",
                height=300, margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_atm, use_container_width=True)

            # Download
            csv_dup_cs = vol_surf.to_csv_dupire(method="cubic", n_moneyness=25, n_ttm=15)
            st.download_button(
                label="⬇ Download Cubic-Spline Dupire Surface (CSV)",
                data=csv_dup_cs,
                file_name="dupire_localvol_cubicspline.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Cubic-spline Dupire error: {e}")

    # ── Section 2: SVI Dupire ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2️⃣  SVI Dupire — Parametric denoising for a smooth surface")

    with st.expander("📖 What is SVI and why does it produce a smoother Dupire?", expanded=False):
        st.markdown(
            "**SVI (Stochastic Volatility Inspired)** parametrizes the *total implied variance* "
            "w(k) = σ²_iv × T at a single expiry T as a function of log-moneyness k = log(K/F):\n\n"
            r"$$w(k;\,a,b,\rho,m,\sigma) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right)$$\n\n"
            "**The 5 parameters have clear economic interpretations:**\n"
            "- **a** — overall variance level (vertical shift)\n"
            "- **b** — wing slope / curvature (larger b → more pronounced smile)\n"
            "- **ρ** — skew direction (-1 = left skew, typical for equities)\n"
            "- **m** — minimum-variance moneyness (usually near ATM)\n"
            "- **σ** — ATM curvature smoothing\n\n"
            "**Why smoother Dupire**: fitting SVI per slice replaces *all* the quote-level noise "
            "(bid-ask spreads, illiquid strikes, stale prints) with a smooth analytic function. "
            "When we differentiate this twice w.r.t. strike, the second derivative is smooth "
            "by construction rather than polluted by interpolation artifacts."
        )
        st.caption(
            "Reference: Gatheral, J. (2004). 'A parsimonious arbitrage-free implied volatility "
            "parametrization with application to the valuation of volatility derivatives.' "
            "Presentation at the Global Derivatives & Risk Management Conference, Madrid."
        )

    # Build SVI button
    svi_cache_key = f"svi_{params.get('snapshot_key', '')}_{params['S0']}_{params['r']}"
    svi_built_key = st.session_state.get("svi_built_for", "")
    svi_already_built = (svi_cache_key == svi_built_key) and vol_surf.svi_ready

    run_svi = st.button(
        "🔧 Build SVI Surface" if not svi_already_built else "🔄 Rebuild SVI Surface",
        key="btn_svi",
        type="primary",
        help="Fits SVI to each expiry slice (~10 s)"
    )

    if run_svi:
        with st.spinner("Fitting SVI parameters to each expiry slice…"):
            try:
                svi_result = vol_surf.build_svi_surface()
                st.session_state["svi_built_for"] = svi_cache_key
                if svi_result["svi_ready"]:
                    st.success(
                        f"✅ SVI surface built — {svi_result['n_slices_fitted']} expiry slices fitted. "
                        f"Avg RMSE: {sum(svi_result['slice_rmse'].values())/max(len(svi_result['slice_rmse']), 1):.2f} vol pts."
                    )
                    # Show per-slice RMSE
                    if svi_result["slice_rmse"]:
                        import pandas as pd
                        rmse_df = pd.DataFrame(
                            [{"TTM": k, "RMSE (vol pts)": f"{v:.3f}%"}
                             for k, v in sorted(svi_result["slice_rmse"].items())]
                        )
                        with st.expander("Per-slice SVI fit quality"):
                            st.dataframe(rmse_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("SVI surface build failed — not enough expiry slices with sufficient data.")
            except Exception as e:
                st.error(f"SVI build error: {e}")

    if vol_surf.svi_ready:
        with st.spinner("Computing SVI-based Dupire local vol surface…"):
            try:
                M_sv, T_sv, LV_sv = vol_surf.svi_dupire_surface_grid(n_moneyness=25, n_ttm=15)
                LV_sv_clipped = np.clip(LV_sv * 100, 0.1, 80.0)

                fig_svi = go.Figure(data=[go.Surface(
                    x=M_sv, y=T_sv, z=LV_sv_clipped,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Local Vol (%)", thickness=15),
                    hovertemplate=(
                        "Moneyness: %{x:.2f}<br>"
                        "TTM: %{y:.2f}y<br>"
                        "SVI Local Vol: %{z:.1f}%<extra></extra>"
                    ),
                )])
                fig_svi.update_layout(
                    title="SVI Dupire Local Vol σ_loc(K/S₀, T) — Parametric Denoising",
                    scene=dict(
                        xaxis_title="Moneyness (K/S₀)",
                        yaxis_title="TTM (years)",
                        zaxis_title="Local Vol (%)",
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                    ),
                    height=560,
                    margin=dict(t=60, b=20),
                )
                st.plotly_chart(fig_svi, use_container_width=True)

                # ATM overlay with cubic for comparison
                ttm_range = np.linspace(0.2, 2.5, 20)
                lv_atm_svi = [vol_surf.svi_dupire_local_vol_grid(
                    np.array([1.0]), np.array([t]))[0, 0] * 100 for t in ttm_range]
                lv_atm_cs  = [vol_surf.dupire_local_vol(1.0, t) * 100 for t in ttm_range]
                iv_atm     = [vol_surf.atm_vol(t) * 100 for t in ttm_range]

                fig_atm2 = go.Figure()
                fig_atm2.add_trace(go.Scatter(x=ttm_range, y=lv_atm_svi,
                                   mode="lines", name="SVI Dupire (ATM)",
                                   line=dict(color="#FF5722", width=2)))
                fig_atm2.add_trace(go.Scatter(x=ttm_range, y=lv_atm_cs,
                                   mode="lines", name="Cubic-Spline Dupire (ATM)",
                                   line=dict(color="#9C27B0", width=2, dash="dot")))
                fig_atm2.add_trace(go.Scatter(x=ttm_range, y=iv_atm,
                                   mode="lines", name="Implied Vol (ATM)",
                                   line=dict(color="#2196F3", width=2, dash="dash")))
                fig_atm2.update_layout(
                    title="ATM: SVI Dupire vs Cubic-Spline Dupire vs Implied Vol",
                    xaxis_title="TTM (years)", yaxis_title="Vol (%)",
                    height=320, margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_atm2, use_container_width=True)

                # SVI IV surface overlay to show smoothing
                st.subheader("SVI Implied Vol Surface (for reference)")
                M_svi_iv, T_svi_iv, IV_svi = vol_surf.svi_surface_grid(n_moneyness=30, n_ttm=20)
                fig_svi_iv = go.Figure(data=[go.Surface(
                    x=M_svi_iv, y=T_svi_iv, z=IV_svi * 100,
                    colorscale="RdYlGn_r", showscale=True,
                    colorbar=dict(title="IV (%)", thickness=15),
                    hovertemplate="Moneyness: %{x:.2f}<br>TTM: %{y:.2f}y<br>SVI IV: %{z:.1f}%<extra></extra>",
                )])
                fig_svi_iv.update_layout(
                    title="SVI-Smoothed Implied Vol Surface (input to SVI Dupire)",
                    scene=dict(
                        xaxis_title="Moneyness (K/S₀)",
                        yaxis_title="TTM (years)",
                        zaxis_title="Implied Vol (%)",
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                    ),
                    height=500, margin=dict(t=60, b=20),
                )
                st.plotly_chart(fig_svi_iv, use_container_width=True)

                # Download
                csv_dup_svi = vol_surf.to_csv_dupire(method="svi", n_moneyness=25, n_ttm=15)
                st.download_button(
                    label="⬇ Download SVI Dupire Surface (CSV)",
                    data=csv_dup_svi,
                    file_name="dupire_localvol_svi.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"SVI Dupire error: {e}")
    else:
        st.info(
            "👆 Click **Build SVI Surface** to fit the parametric SVI model and generate "
            "the smooth Dupire surface."
        )
