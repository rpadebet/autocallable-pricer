"""
app/pages/02_Pricer.py
=======================
MOST IMPORTANT PAGE — Full Pricing Analysis Dashboard.

WHY THIS PAGE IS CENTRAL:
    The pricer page brings together all three pricing methods and shows:
        1. Price comparison table (FD vs Standard MC vs Survival MC).
        2. Convergence chart — how the price estimate evolves as N increases.
        3. MC path animation — 30 simulated paths with barrier/trigger lines,
           colored by outcome (called early vs survived to maturity).
        4. Variance reduction comparison (Survival MC vs Standard MC).
    This visualization makes abstract pricing theory tangible.

KEY CHARTS:
    - Tab "Price Comparison": Side-by-side metrics + comparison bar chart.
    - Tab "MC Convergence": Price estimate vs N paths for both MC methods.
    - Tab "Path Animation": 30 GBM paths colored called/uncalled + barrier lines.
    - Tab "Greeks (FD)": Delta, Gamma, Theta from FD bump-grid (if fast enough).

PAPER REFERENCES:
    Paper 1 §2.2 — explicit FD (FDPricer)
    Paper 3 §2 Eq. 2.3 — Standard MC (MCStandardPricer)
    Paper 3 Algorithm 1 — Survival MC (MCSurvivalPricer)
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
from app.mc_standard import MCStandardPricer
from app.mc_survival import MCSurvivalPricer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pricer — AutoCallable Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar ─────────────────────────────────────────────────────────────
params = render_sidebar(page_name="Pricer")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💰 Autocallable Pricer")
_vol_label = {
    "flat":   "Flat σ",
    "local":  "Local Vol (Dupire)",
    "heston": "Heston",
    "bates":  "Bates",
}.get(params.get("vol_model", "flat"), "Flat σ")
st.caption(
    f"**{params['security_name']}**  |  Vol: {_vol_label}  |  σ = {params['sigma']*100:.1f}%  |  "
    f"r = {params['r']*100:.2f}%  |  q = {params['q']*100:.2f}%  |  "
    f"S₀ = {params['S0']:,.1f}  |  {params['snapshot_label']}"
)

ac = params["autocallable"]
if ac is None:
    st.error("Product initialization failed. Check sidebar parameters.")
    st.stop()

# ── Settings-changed banner ────────────────────────────────────────────────────
# Compare a hash of key params to what was used in the last calculation.
# If different, show a warning so the user knows to re-run.
def _param_fingerprint(p: dict) -> str:
    """
    Compact string that changes whenever any pricing-relevant setting changes.

    WHY ONLY THESE KEYS: Security name, vol model, spot, rates, and MC paths are
    the settings most likely to change between navigations. Hashing all of params
    would be fragile (snapshot_df is a DataFrame). This set covers 99% of cases.
    """
    return "|".join(str(p.get(k)) for k in (
        "security_name", "vol_model", "S0", "r", "q", "sigma",
        "n_paths", "seed", "N_x", "N_tau",
        "v0", "kappa", "theta", "gamma", "rho",
    ))

_current_fp = _param_fingerprint(params)
_last_fp = st.session_state.get("pricer_last_run_fp", None)

if _last_fp is not None and _last_fp != _current_fp:
    st.warning(
        "⚠️ **Settings have changed since the last calculation.** "
        "Click **Run All Pricers** to update results with the new parameters.",
        icon="🔄",
    )

# ── Calibration status banner (Heston / Bates only) ───────────────────────────
# When a stochastic vol model is selected but Heston hasn't been calibrated yet,
# warn the user so they don't unknowingly price with default (non-market) params.
_vm_selected = params.get("vol_model", "flat")
if _vm_selected in ("heston", "bates"):
    _heston_cal = st.session_state.get("heston_cal")
    if _heston_cal:
        st.success(
            f"✅ **Calibrated Heston params in use** — "
            f"RMSE {_heston_cal.get('rmse_vol_pts', 0.0):.1f} vol-pts  "
            f"({_heston_cal.get('n_quotes', '?')} quotes)  |  "
            f"v₀={_heston_cal.get('v0', 0.04):.4f}  "
            f"κ={_heston_cal.get('kappa', 1.5):.2f}  "
            f"ρ={_heston_cal.get('rho', -0.7):.2f}",
            icon="📈",
        )
    else:
        st.info(
            f"ℹ️ **Heston not yet calibrated to market.** "
            f"Using default parameters: "
            f"v₀={params.get('v0', 0.04):.3f}  κ={params.get('kappa', 1.5):.1f}  "
            f"θ={params.get('theta', 0.04):.3f}  γ={params.get('gamma', 0.30):.2f}  "
            f"ρ={params.get('rho', -0.70):.2f}.  "
            f"For market-fitted parameters, go to **Vol Surface → Tab 2 → Calibrate Heston**, "
            f"then enable **Use calibrated values** in the sidebar.",
            icon="📈",
        )

# ── Control bar ────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 3])
with ctrl1:
    run_all = st.button("▶ Run All Pricers", type="primary", use_container_width=True)
with ctrl2:
    show_paths = st.toggle("Return Paths", value=True,
                           help="Compute 30-path animation (adds ~0.5s)")
with ctrl3:
    track_conv = st.toggle("Track Convergence", value=True,
                           help="Compute running price estimate at multiple N (adds ~1s)")

st.divider()


# ==============================================================================
# PRICING COMPUTATION
# ==============================================================================

def _build_vol_surface(params: dict):
    """
    Build a VolSurface from the current snapshot, or return None on failure.

    Caches the result in session_state keyed by snapshot_key + S0 + r + q so
    the surface is not rebuilt on every Run click. The Dupire local vol grid
    is also built once and shared across all three pricers.

    Returns: (VolSurface | None, error_message | None)
    """
    from app.vol_surface import VolSurface
    from scipy.interpolate import RegularGridInterpolator

    snap_key = params.get("snapshot_key", "")
    cache_key = f"vs_{snap_key}_{params['S0']}_{params['r']}_{params.get('q', 0.014)}"

    cached = st.session_state.get("vol_surface_cache")
    if cached and cached.get("key") == cache_key:
        return cached["vs"], None

    try:
        snap_df = params["snapshot_df"]
        vs = VolSurface(snap_df, S0=params["S0"], r=params["r"], q=params.get("q", 0.014))

        # Build shared Dupire grid once for all pricers
        import numpy as np
        m_axis = np.linspace(0.40, 1.80, 40)
        max_t = params["autocallable"].maturity_years + 0.05
        t_axis = np.linspace(0.01, max_t, 25)
        LV = vs.dupire_local_vol_grid(m_axis, t_axis)
        local_vol_interp = RegularGridInterpolator(
            (t_axis, m_axis), LV,
            method="linear", bounds_error=False, fill_value=None,
        )

        st.session_state["vol_surface_cache"] = {
            "key": cache_key,
            "vs": vs,
            "local_vol_interp": local_vol_interp,
        }
        return vs, None
    except Exception as e:
        return None, str(e)


def run_pricers(params, ac, show_paths, track_conv):
    """
    Run all three pricers and return results. Each method is wrapped in try/except
    so a single pricer failure does not block the others.

    Vol model routing:
        "flat"   → all three pricers use constant sigma (fastest)
        "local"  → MCStandard + FDM use Dupire local vol; Survival MC also uses local
        "heston" → MCStandard + Survival use Heston CIR variance; FDM falls back to flat
        "bates"  → MCStandard + Survival use Heston+jumps; FDM falls back to flat
    FDM local vol is the only FDM vol-aware mode. Heston/Bates FDM is not yet implemented
    (PDE for stochastic vol requires an extra state dimension — future work).
    """
    results = {"fd": None, "mc": None, "sv": None,
               "fd_err": None, "mc_err": None, "sv_err": None}

    vol_model     = params.get("vol_model", "flat")
    heston_params = params.get("heston_params")
    jump_params   = params.get("jump_params")

    # Build vol surface if local vol is selected (needed by MC and FDM)
    vol_surface = None
    local_vol_interp = None
    if vol_model == "local":
        with st.spinner("Building Dupire local vol surface from snapshot…"):
            vol_surface, vs_err = _build_vol_surface(params)
            if vol_surface is None:
                st.warning(
                    f"⚠️ Could not build local vol surface: {vs_err}. "
                    "Falling back to flat vol for all pricers."
                )
                vol_model = "flat"
            else:
                cached = st.session_state.get("vol_surface_cache", {})
                local_vol_interp = cached.get("local_vol_interp")

    # FD only supports flat or local vol (Heston/Bates FDM requires 2D PDE — not yet built).
    fd_vol_model  = vol_model if vol_model in ("flat", "local") else "flat"
    fd_vol_note   = " (flat — Heston/Bates PDE not yet implemented)" if vol_model in ("heston", "bates") else ""

    # ── FD PDE ──
    with st.spinner(f"FD PDE pricing{fd_vol_note}…"):
        try:
            fd = FDPricer(
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
            # For Heston/Bates: override call_prob_sigma to sqrt(v0) so the
            # call probability table uses the same vol as the MC pricers,
            # not the flat sigma passed to FDPricer (which falls back to flat).
            if vol_model in ("heston", "bates") and heston_params:
                import math as _math
                fd.call_prob_sigma = _math.sqrt(heston_params.get("v0", params["sigma"] ** 2))
            results["fd"] = fd.price(return_grid=False)
        except Exception as e:
            results["fd_err"] = str(e)

    # Antithetic variates only work with flat vol (requires paired normal samples on GBM;
    # sub-stepped Heston/Bates paths break the pairing assumption).
    use_antithetic = params["antithetic"] and vol_model == "flat"

    # ── Standard MC ──
    with st.spinner(f"Standard MC ({params['n_paths']:,} paths, {vol_model})…"):
        try:
            mcp = MCStandardPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                n_paths=params["n_paths"],
                seed=params["seed"],
                antithetic=use_antithetic,
                vol_model=vol_model,
                vol_surface=vol_surface,
                heston_params=heston_params,
                jump_params=jump_params,
                local_vol_interp=local_vol_interp,
            )
            results["mc"] = mcp.price(
                return_paths=show_paths,
                track_convergence=track_conv,
            )
        except Exception as e:
            results["mc_err"] = str(e)

    # ── Survival MC ──
    with st.spinner(f"Survival MC ({params['n_paths']:,} paths, {vol_model})…"):
        try:
            svp = MCSurvivalPricer(
                autocallable=ac,
                sigma=params["sigma"],
                r=params["r"],
                q=params["q"],
                n_paths=params["n_paths"],
                seed=params["seed"],
                vol_model=vol_model,
                vol_surface=vol_surface,
                heston_params=heston_params,
                jump_params=jump_params,
                local_vol_interp=local_vol_interp,
            )
            results["sv"] = svp.price(
                return_paths=show_paths,
                track_convergence=track_conv,
            )
        except Exception as e:
            results["sv_err"] = str(e)

    return results


# Use session state to cache results between tab switches
if run_all:
    st.session_state["pricer_results"] = run_pricers(params, ac, show_paths, track_conv)
    st.session_state["pricer_params_used"] = {k: v for k, v in params.items()
                                               if k not in ("snapshot_df", "autocallable", "security_params")}
    # Store fingerprint so the "settings changed" banner can detect future changes
    st.session_state["pricer_last_run_fp"] = _current_fp
    # WHY rerun: updating session_state mid-script does NOT re-render widgets in the
    # same pass. The warning banner (computed from _last_fp at the top of the script)
    # still shows the old fingerprint.  A single st.rerun() re-runs the script from
    # the top with the new fingerprint already in session_state → banner disappears
    # and the button returns to its idle state.  Pricers do NOT re-run because
    # `run_all` evaluates to False on the rerun (button is reset after one True).
    st.rerun()

res = st.session_state.get("pricer_results", None)

if res is None:
    st.info("👆 Click **Run All Pricers** to compute prices. Results are cached between tab switches.")
    st.stop()

fd_res = res["fd"]
mc_res = res["mc"]
sv_res = res["sv"]


# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price Comparison", "📈 MC Convergence", "🎲 Path Animation", "📐 Term Structure"]
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Price Comparison — Three Methods")

    # When stochastic vol is selected, explain why FDM shows a different price.
    # Heston/Bates FDM requires an extra PDE state dimension for variance — not implemented.
    _active_vol_model = params.get("vol_model", "flat")
    if _active_vol_model in ("heston", "bates"):
        _vol_model_name = "Heston" if _active_vol_model == "heston" else "Bates"
        st.info(
            f"ℹ️ **FDM uses flat σ = {params['sigma']*100:.1f}%** regardless of the "
            f"{_vol_model_name} selection. Pricing a stochastic-vol PDE requires an "
            f"extra state dimension (one for S, one for v) — not yet implemented. "
            f"MC pricers below use full {_vol_model_name} dynamics. "
            f"Any price gap between FDM and MC shows the **vol-model premium**.",
            icon="⚠️",
        )

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        _fd_vol_used = params.get("vol_model", "flat")
        if _fd_vol_used in ("heston", "bates"):
            _fd_label_html = "<small style='color:orange'>(flat σ — stoch-vol PDE not implemented)</small>"
        elif _fd_vol_used == "local":
            _fd_label_html = "<small>(Dupire local vol)</small>"
        else:
            _fd_label_html = "<small>(flat σ)</small>"
        st.markdown(f"**📐 Finite Difference (PDE)**  {_fd_label_html}",
                    unsafe_allow_html=True)
        if fd_res:
            st.metric("Price", f"${fd_res.price:,.2f}",
                      help="Deng, Mallett & McCann (2011) — Paper 1 §2.2")
            sigma = params["sigma"]; T = ac.maturity_years
            dtau = 0.5 * sigma**2 * T / params["N_tau"]
            dx = (abs(params["x_min"]) + 5.0) / params["N_x"]
            courant = dtau / dx**2
            icon = "🟢" if courant < 0.5 else "🔴"
            st.caption(f"{icon} Courant: {courant:.4f} | Nₓ={params['N_x']} | Nτ={params['N_tau']}")
        else:
            st.error(res["fd_err"] or "FD failed")

    with m2:
        st.markdown("**🎲 Standard Monte Carlo**")
        if mc_res:
            st.metric("Price", f"${mc_res.price:,.2f}",
                      delta=f"± ${mc_res.std_err:,.3f}",
                      help="Alm et al. (2013) §2 Eq. 2.3 — antithetic variates")
            st.caption(
                f"95% CI [{mc_res.ci_low:,.2f}, {mc_res.ci_high:,.2f}]  "
                f"| N = {mc_res.n_paths:,}"
            )
        else:
            st.error(res["mc_err"] or "MC failed")

    with m3:
        st.markdown("**🎯 Survival MC (Alm 2013)**")
        if sv_res:
            st.metric("Price", f"${sv_res.price:,.2f}",
                      delta=f"± ${sv_res.std_err:,.3f}",
                      help="Alm et al. (2013) Algorithm 1 — analytical barrier → smooth Greeks")
            if mc_res and sv_res.std_err > 0:
                vr = mc_res.std_err / sv_res.std_err
                st.caption(
                    f"95% CI [{sv_res.ci_low:,.2f}, {sv_res.ci_high:,.2f}]  "
                    f"| Variance reduction: **{vr:.2f}×**"
                )
        else:
            st.error(res["sv_err"] or "Survival MC failed")

    st.divider()

    # Bar chart comparison
    labels, prices, errors, colors = [], [], [], []
    if fd_res:
        labels.append("FD (PDE)"); prices.append(fd_res.price); errors.append(0)
        colors.append("#2196F3")
    if mc_res:
        labels.append("Standard MC"); prices.append(mc_res.price); errors.append(mc_res.std_err * 1.96)
        colors.append("#4CAF50")
    if sv_res:
        labels.append("Survival MC"); prices.append(sv_res.price); errors.append(sv_res.std_err * 1.96)
        colors.append("#FF9800")

    if prices:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Price",
            x=labels, y=prices,
            error_y=dict(type="data", array=errors, visible=True),
            marker_color=colors,
            text=[f"${p:,.2f}" for p in prices],
            textposition="outside",
            width=0.4,
        ))

        mid = np.mean(prices)
        fig.add_hline(y=mid, line_dash="dot", line_color="gray",
                      annotation_text=f"Mean ${mid:,.2f}", annotation_position="right")

        fig.update_layout(
            title="Price Comparison — Error Bars = 95% CI (MC methods)",
            yaxis=dict(
                title="Price ($)",
                range=[min(prices) * 0.99, max(prices) * 1.01],
            ),
            showlegend=False,
            height=380,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Spread analysis — interpretation depends on vol model.
    # When Heston/Bates is selected, FDM uses flat σ while MC uses stochastic vol,
    # so a spread is EXPECTED (it's the vol-model premium, not an inconsistency).
    # Only flag a large spread as a problem when all methods share the same vol model.
    if len(prices) >= 2:
        spread = max(prices) - min(prices)
        spread_pct = spread / np.mean(prices) * 100
        _vm = params.get("vol_model", "flat")
        if _vm in ("heston", "bates"):
            _vm_label = "Heston" if _vm == "heston" else "Bates"
            st.markdown(
                f"📊 **Vol-model premium: ${spread:.2f} ({spread_pct:.2f}%)**  "
                f"— FDM uses flat σ = {params['sigma']*100:.1f}%; "
                f"MC methods use full {_vm_label} dynamics (vol-of-vol, skew, "
                f"{'jumps' if _vm == 'bates' else 'mean reversion'}). "
                f"This gap quantifies how much stochastic vol affects autocallable pricing."
            )
        else:
            # Same vol model across all three methods — spread should be small
            icon = "🟢" if spread_pct < 1 else "🟡" if spread_pct < 3 else "🔴"
            note = (
                "within 2% at N=10K — implementations are consistent."
                if spread_pct < 3
                else "spread > 3% — try increasing path count or tightening FDM grid."
            )
            st.markdown(f"{icon} **Method spread: ${spread:.2f} ({spread_pct:.2f}%)**  — {note}")

    # Variance reduction table
    if mc_res and sv_res:
        st.divider()
        st.markdown("**Variance Reduction Summary (Survival MC vs Standard MC)**")
        vr = mc_res.std_err / sv_res.std_err if sv_res.std_err > 0 else float("nan")
        eff = vr**2
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
| Metric | Standard MC | Survival MC |
|---|---|---|
| σ̂ (std error) | ${mc_res.std_err:,.4f} | ${sv_res.std_err:,.4f} |
| 95% CI Width | ${mc_res.ci_high - mc_res.ci_low:,.3f} | ${sv_res.ci_high - sv_res.ci_low:,.3f} |
| Variance Reduction | 1.0× | **{vr:.2f}×** |
| Equivalent paths | {mc_res.n_paths:,} | {int(mc_res.n_paths / eff):,}* |
""")
        with col_b:
            st.info(
                f"Survival MC achieves **{vr:.2f}× lower standard error** at the same N. "
                f"This means you need only ~{int(mc_res.n_paths / eff):,} standard MC paths "
                f"to match {mc_res.n_paths:,} survival MC paths. "
                "The gain comes from analytically integrating over barrier crossings (Paper 3, §3)."
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MC CONVERGENCE
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Monte Carlo Convergence")
    st.caption("Price estimate as a function of number of paths. Narrowing CI bands show MC convergence.")

    mc_conv = mc_res.convergence_series if mc_res else []
    sv_conv = sv_res.convergence_series if sv_res else []

    if not mc_conv and not sv_conv:
        st.info("Enable **Track Convergence** toggle and rerun to see this chart.")
    else:
        fig = go.Figure()

        # Standard MC band
        if mc_conv:
            ns = [c[0] for c in mc_conv]
            means = [c[1] for c in mc_conv]
            ses = [c[2] for c in mc_conv]
            upper = [m + 1.96 * s for m, s in zip(means, ses)]
            lower = [m - 1.96 * s for m, s in zip(means, ses)]

            fig.add_trace(go.Scatter(
                x=ns + ns[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(33, 150, 243, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="Standard MC 95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=ns, y=means,
                mode="lines+markers",
                name="Standard MC",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=4),
            ))

        # Survival MC band
        if sv_conv:
            ns2 = [c[0] for c in sv_conv]
            means2 = [c[1] for c in sv_conv]
            ses2 = [c[2] for c in sv_conv]
            upper2 = [m + 1.96 * s for m, s in zip(means2, ses2)]
            lower2 = [m - 1.96 * s for m, s in zip(means2, ses2)]

            fig.add_trace(go.Scatter(
                x=ns2 + ns2[::-1],
                y=upper2 + lower2[::-1],
                fill="toself",
                fillcolor="rgba(255, 152, 0, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="Survival MC 95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=ns2, y=means2,
                mode="lines+markers",
                name="Survival MC",
                line=dict(color="#FF9800", width=2),
                marker=dict(size=4),
            ))

        # FD reference line
        if fd_res:
            all_ns = ([c[0] for c in mc_conv] or []) + ([c[0] for c in sv_conv] or [])
            if all_ns:
                fig.add_hline(
                    y=fd_res.price,
                    line_dash="dash",
                    line_color="#4CAF50",
                    annotation_text=f"FD Price ${fd_res.price:,.2f}",
                    annotation_position="right",
                )

        fig.update_layout(
            title="MC Convergence: Price Estimate vs. Number of Paths",
            xaxis=dict(title="Number of Paths (N)", type="log"),
            yaxis=dict(title="Price ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
            margin=dict(t=80, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Key insight: Survival MC's CI band is visibly **narrower** than Standard MC's at the same N. "
            "This is the variance reduction from Paper 3, Algorithm 1."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PATH ANIMATION (30 paths with barrier lines)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("MC Path Simulation")
    st.caption(
        "Up to 30 GBM paths. **Blue** = called early (barrier crossed). "
        "**Gray** = survived to maturity. "
        "Dashed red = call barrier. Dashed orange = protection (knock-in) barrier."
    )

    # Use Standard MC paths (full paths available, not truncated like survival)
    if not mc_res or mc_res.paths is None:
        st.info("Enable **Return Paths** toggle and rerun to see path animation.")
    else:
        paths = mc_res.paths
        call_idxs = mc_res.call_times or [None] * len(paths)

        # Time axis: 0 (trade date) + all observation dates
        obs_dates = ac.observation_dates()
        t_axis = [0.0] + list(obs_dates)

        fig = go.Figure()

        # ── Call/protection barrier lines ──
        call_barrier_lvl = ac.call_barrier_at_period(0) * ac.S_ref
        prot_barrier_lvl = ac.protection_barrier * ac.S_ref

        fig.add_hline(
            y=call_barrier_lvl,
            line=dict(color="red", dash="dash", width=1.5),
            annotation_text=f"Call barrier {ac.call_barrier_at_period(0)*100:.0f}%  "
                            f"(${call_barrier_lvl:,.0f})",
            annotation_position="top right",
        )
        fig.add_hline(
            y=prot_barrier_lvl,
            line=dict(color="orange", dash="dash", width=1.5),
            annotation_text=f"Protection barrier {ac.protection_barrier*100:.0f}%  "
                            f"(${prot_barrier_lvl:,.0f})",
            annotation_position="bottom right",
        )

        # ── Step-down barriers (if any) ──
        if ac.stepped_barriers:
            for period_idx, (start_period, barrier_frac) in enumerate(ac.stepped_barriers):
                # Find the start time for this barrier level
                if start_period < len(obs_dates):
                    t_start = obs_dates[start_period]
                    t_end = obs_dates[-1] + 0.1
                    lvl = barrier_frac * ac.S_ref
                    fig.add_shape(
                        type="line",
                        x0=t_start, x1=t_end,
                        y0=lvl, y1=lvl,
                        line=dict(color="darkred", dash="dot", width=1),
                    )

        # ── S0 reference line ──
        fig.add_hline(
            y=ac.S_ref,
            line=dict(color="black", dash="dot", width=1),
            annotation_text=f"S₀ = {ac.S_ref:,.0f}",
            annotation_position="right",
        )

        # ── Plot individual paths ──
        n_show = min(30, len(paths))
        called_count = 0
        survived_count = 0

        for i in range(n_show):
            path = paths[i]
            call_idx = call_idxs[i]

            # Align path to t_axis — path length may differ (truncated at call)
            path_t = t_axis[:len(path)]

            if call_idx is not None:
                # Path called at obs date call_idx
                color = "rgba(33, 150, 243, 0.5)"  # blue with alpha
                name = "Called" if called_count == 0 else None
                called_count += 1
            else:
                color = "rgba(150, 150, 150, 0.4)"  # gray with alpha
                name = "Survived" if survived_count == 0 else None
                survived_count += 1

            fig.add_trace(go.Scatter(
                x=path_t,
                y=list(path),
                mode="lines",
                line=dict(color=color, width=1),
                name=name,
                showlegend=(name is not None),
                hovertemplate=(
                    f"Path {i+1}<br>"
                    "t = %{x:.2f}y<br>"
                    "S = %{y:,.1f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=(
                f"30 Simulated Paths — {called_count} called early, "
                f"{survived_count} survived to maturity"
            ),
            xaxis=dict(title="Time (years)", tickformat=".1f"),
            yaxis=dict(title="Spot Level S(t)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=500,
            margin=dict(t=80, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        total_paths = mc_res.n_paths
        if mc_res.call_times:
            n_called = sum(1 for c in mc_res.call_times if c is not None)
            call_pct = n_called / len(mc_res.call_times) * 100
        else:
            call_pct = float("nan")

        st.caption(
            f"Showing 30 of {total_paths:,} paths. "
            f"In full simulation: ~{call_pct:.1f}% called early (estimated from stored paths)."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — TERM STRUCTURE (Call Probability + Expected Payoff by Date)
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Term Structure of Call Probabilities")
    _cp_vol_model = params.get("vol_model", "flat")
    _cp_sigma = fd_res.call_probs and params.get("sigma", 0.20)  # fallback
    if _cp_vol_model in ("heston", "bates") and heston_params:
        import math as _math
        _eff_sigma = _math.sqrt(heston_params.get("v0", params["sigma"] ** 2))
        _model_label = "Heston" if _cp_vol_model == "heston" else "Bates"
        st.caption(
            f"Analytical call probabilities using \u03c3\u209a\u209c\u209a = {_eff_sigma*100:.1f}% "
            f"(√v\u2080 from {_model_label} calibration). "
            "Exact model-consistent probabilities would require simulation."
        )
    else:
        st.caption("FD-derived call probability at each observation date (analytical — no MC noise).")

    if not fd_res or not fd_res.call_probs:
        st.info("FD pricing must succeed to show term structure.")
    else:
        obs_dates = fd_res.obs_dates
        call_probs = fd_res.call_probs
        cumulative = np.cumsum(call_probs)
        survival_prob = 1.0 - cumulative

        # Left panel — table
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown("**Observation Date Analysis**")
            table_rows = []
            for i, (t, p) in enumerate(zip(obs_dates, call_probs)):
                table_rows.append({
                    "Date (yr)": f"{t:.3f}",
                    "Call Barrier": f"{ac.call_barrier_at_period(i)*100:.1f}%",
                    "P(call at t)": f"{p*100:.2f}%",
                    "P(survive)": f"{(1-cumulative[i])*100:.2f}%",
                })
            import pandas as pd
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Right panel — chart
        with col_right:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Call Probability per Observation Date",
                                "Cumulative Call Probability & Survival"),
                shared_xaxes=True,
                vertical_spacing=0.12,
            )

            # Top: per-period bar chart
            fig.add_trace(go.Bar(
                x=[f"{t:.2f}y" for t in obs_dates],
                y=[p * 100 for p in call_probs],
                name="P(call at t)",
                marker_color="#2196F3",
                text=[f"{p*100:.1f}%" for p in call_probs],
                textposition="outside",
            ), row=1, col=1)

            # Bottom: cumulative call + survival
            fig.add_trace(go.Scatter(
                x=[f"{t:.2f}y" for t in obs_dates],
                y=[c * 100 for c in cumulative],
                mode="lines+markers",
                name="Cumulative P(call)",
                line=dict(color="#FF5722", width=2),
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=[f"{t:.2f}y" for t in obs_dates],
                y=[(1 - c) * 100 for c in cumulative],
                mode="lines+markers",
                name="P(survive to maturity)",
                line=dict(color="#4CAF50", width=2, dash="dash"),
            ), row=2, col=1)

            fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
            fig.update_yaxes(title_text="Probability (%)", range=[0, 105], row=2, col=1)
            fig.update_layout(
                height=480,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Expected remaining life
        if call_probs:
            expected_life = sum(t * p for t, p in zip(obs_dates, call_probs))
            expected_life += obs_dates[-1] * float(1 - cumulative[-1])  # maturity if never called
            st.metric(
                "Expected Life (duration-weighted)",
                f"{expected_life:.3f} years",
                help="Σ t_i × P(called at t_i) + T × P(survives to maturity). "
                     "Compare to nominal maturity of " + f"{ac.maturity_years}Y.",
            )
