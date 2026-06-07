"""
app/components/sidebar.py
==========================
Shared Assumptions sidebar rendered on every page of the app.

WHY THIS EXISTS:
    Every Streamlit page calls render_sidebar() to show the same set of model
    assumptions in the left sidebar. Centralizing this in one function ensures:
        1. All pages see the same assumptions — no page has hardcoded params.
        2. Changing a parameter immediately affects all pricing on the next run.
        3. The sidebar layout is consistent across pages.

    The sidebar returns a `params` dict that every page/pricer uses.
    No module in app/ should have hardcoded model parameters.

SECTION STRUCTURE:
    1. Data — snapshot date, risk-free rate, dividend yield
    2. Product — selected security, reference spot
    3. Heston Model — κ, θ, γ, ρ, v₀
    4. Monte Carlo — N paths, time steps, seed, antithetic
    5. FDM/PDE — grid size, domain bounds
"""

import streamlit as st
from typing import Optional
from app.data_loader import list_available_snapshots, load_snapshot, get_spot_price, get_rfr, resolve_data_dir
from app.components.securities import list_securities, get_security


@st.cache_data(show_spinner=False)
def _cached_load_snapshot(key: str, data_dir: str):
    """
    Module-level cached snapshot loader.

    WHY MODULE-LEVEL: Defining @st.cache_data inside render_sidebar() creates a new
    function object on every call, which means Streamlit's cache hash changes and the
    data is reloaded on every page navigation. Moving it here ensures the same cache
    entry is reused across all pages and all render_sidebar() calls.
    """
    return load_snapshot(key, data_dir)


def _ensure_sidebar_defaults(sec_names: list, snaps: list) -> None:
    """
    Initialize all sidebar widget keys in session_state before widgets are rendered.

    WHY THIS MATTERS: In Streamlit, when a widget has both a `key=` and a `value=` /
    `index=` parameter, the behavior differs by version. The safest pattern is:
        1. Initialize session_state manually before creating the widget.
        2. Do NOT pass `index=` or `value=` to widgets whose key is already set.
    This guarantees that user selections persist across page navigation.
    """
    defaults = {
        "snapshot_idx":           len(snaps) - 1,   # most recent snapshot
        "security_name":          sec_names[0],      # first security
        "vol_model_label":        "Flat (Black-Scholes)",
        "r":                      0.045,
        "q":                      0.014,
        "use_calibrated_heston":  False,
        "v0":                     0.04,
        "kappa":                  1.5,
        "theta":                  0.04,
        "gamma":                  0.30,
        "rho":                    -0.70,
        "sigma_flat":             0.20,
        "lam_j":                  0.10,
        "mu_j":                   -0.05,
        "sig_j":                  0.10,
        "n_paths":                10_000,
        "n_steps":                252,
        "seed":                   42,
        "antithetic":             True,
        "N_x":                    150,
        "N_tau":                  100,
        "x_min":                  -5.0,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def render_sidebar(page_name: str = "") -> dict:
    """
    Render the shared Assumptions sidebar and return all parameters as a dict.

    Called at the top of every page. Initializes st.session_state defaults on
    first run, then reads current widget values and returns them.

    Args:
        page_name: Optional page label shown in the sidebar header (unused for now).

    Returns:
        params dict with all model and product parameters. Keys:
            snapshot_key, snapshot_label, S0, r, q,
            security_name, security_params, autocallable,
            sigma, vol_model, heston_params, jump_params,
            v0, kappa, theta, gamma, rho,
            n_paths, n_steps, seed, antithetic,
            N_x, N_tau, x_min,
            snapshot_df   (loaded DataFrame, cached)
    """
    data_dir = resolve_data_dir()

    with st.sidebar:
        st.markdown("## ⚙️ Model Assumptions")
        st.caption("Changes apply on next rerun (⌘R)")
        st.divider()

        # ─────────────────────────────────────
        # Section 1: Data
        # ─────────────────────────────────────
        st.markdown("**① Market Data**")

        snaps = list_available_snapshots(data_dir)
        if not snaps:
            st.error("No snapshots found. Run: python scripts/generate_synthetic_data.py")
            st.stop()

        snap_labels = [s["label"] for s in snaps]
        snap_keys = [s["key"] for s in snaps]

        # Initialize ALL sidebar defaults before any widget is rendered.
        # This guarantees persistence across page navigation (see _ensure_sidebar_defaults).
        _ensure_sidebar_defaults(list_securities(), snaps)

        snap_idx = st.selectbox(
            "Data Date",
            options=list(range(len(snaps))),
            format_func=lambda i: snap_labels[i],
            # No index= — session_state["snapshot_idx"] was initialized above
            key="snapshot_idx",
            help="Pre-collected SPX options snapshot used for vol surface and spot level.",
        )
        selected_key = snap_keys[snap_idx]
        selected_label = snap_labels[snap_idx]

        # Load snapshot via module-level cache (persists across all page navigations)
        snap_df = _cached_load_snapshot(selected_key, data_dir)
        S0_market = get_spot_price(snap_df)
        rfr_market = get_rfr(snap_df)

        if "r" not in st.session_state:
            st.session_state["r"] = float(round(rfr_market, 4))
        r = st.number_input(
            "Risk-Free Rate (r)",
            min_value=0.0, max_value=0.20,
            step=0.001, format="%.3f",
            key="r",
            help="Continuously compounded annual rate. Default from snapshot (^IRX).",
        )
        q = st.number_input(
            "Dividend Yield (q)",
            min_value=0.0, max_value=0.10,
            value=0.014, step=0.001, format="%.3f",
            key="q",
            help="SPX dividend yield. 1.4% is approximate long-run average.",
        )

        st.divider()

        # ─────────────────────────────────────
        # Section 2: Product
        # ─────────────────────────────────────
        st.markdown("**② Autocallable Product**")

        sec_names = list_securities()
        sec_name = st.selectbox(
            "Security",
            options=sec_names,
            # No index= — session_state["security_name"] was initialized by _ensure_sidebar_defaults
            key="security_name",
            help="Choose from 4 pre-configured structures. More details on the Pricer page.",
        )
        sec_params = get_security(sec_name)

        # S0: default to market value only on FIRST load; preserve user overrides after that
        if "S0" not in st.session_state:
            st.session_state["S0"] = float(S0_market)
        S0 = st.number_input(
            "Reference Spot (S₀)",
            min_value=100.0, max_value=20000.0,
            step=10.0, format="%.1f",
            key="S0",
            help="SPX reference level at trade date. Defaults to market close from snapshot.",
        )

        st.divider()

        # ─────────────────────────────────────
        # Section 3: Heston Model
        # ─────────────────────────────────────
        with st.expander("**③ Heston Model**", expanded=False):
            st.caption("σ²-process: dv = κ(θ-v)dt + γ√v dW_v,  Corr(W_S, W_v) = ρ")

            use_calibrated = st.toggle(
                "Use calibrated values",
                value=False,
                key="use_calibrated_heston",
                help="Run Heston calibration against snapshot data (takes ~10s).",
            )

            v0 = st.slider("v₀ (initial variance)", 0.001, 0.30, 0.04, 0.001,
                           format="%.3f", key="v0",
                           help="sqrt(v0) = current vol. 0.04 → 20% vol.")
            kappa = st.slider("κ (mean reversion)", 0.1, 10.0, 1.5, 0.1,
                              format="%.1f", key="kappa")
            theta = st.slider("θ (long-run variance)", 0.001, 0.30, 0.04, 0.001,
                              format="%.3f", key="theta",
                              help="sqrt(theta) = long-run vol. 0.04 → 20%.")
            gamma = st.slider("γ (vol of vol)", 0.05, 1.5, 0.30, 0.01,
                              format="%.2f", key="gamma")
            rho = st.slider("ρ (correlation)", -0.99, -0.01, -0.70, 0.01,
                            format="%.2f", key="rho",
                            help="Negative for equities: when market falls, vol rises.")

            feller = kappa * theta > 0.5 * gamma ** 2
            if feller:
                st.success(f"Feller condition: κθ ({kappa*theta:.4f}) > ½γ² ({0.5*gamma**2:.4f}) ✓")
            else:
                st.warning(f"Feller violated: κθ ({kappa*theta:.4f}) ≤ ½γ² ({0.5*gamma**2:.4f})")

            # ATM vol implied by current params (approximate)
            sigma = float((v0 + theta) / 2) ** 0.5
            st.caption(f"Approx ATM vol: {sigma*100:.1f}%")

        # Flat vol for pricing (from Heston v0 approximation)
        sigma_flat = st.number_input(
            "Flat Vol (σ) for Pricing",
            min_value=0.01, max_value=1.0,
            value=round(float(v0 ** 0.5), 3),
            step=0.01, format="%.3f",
            key="sigma_flat",
            help="Used by FD and MC pricers. Defaults to sqrt(v0).",
        )

        st.divider()

        # ─────────────────────────────────────
        # Section 3b: Volatility Model selector
        #
        # WHY HERE: The vol model is the bridge between the Heston parameters
        # above and the MC/FD sections below. Showing it here makes clear which
        # model the pricers will actually use. The Heston params above are only
        # active when the model is "heston" or "bates".
        # ─────────────────────────────────────
        st.markdown("**③b Volatility Model**")
        st.caption("Selects which vol model all three pricers will use.")

        VOL_MODEL_OPTIONS = {
            "Flat (Black-Scholes)": "flat",
            "Local Vol (Dupire)":   "local",
            "Heston Stochastic Vol": "heston",
            "Bates (Heston + Jumps)": "bates",
        }
        vol_model_label = st.selectbox(
            "Volatility Model",
            options=list(VOL_MODEL_OPTIONS.keys()),
            index=0,
            key="vol_model_label",
            help=(
                "Flat: constant σ (fastest, classic). "
                "Local Vol: σ(S,t) from Dupire formula — needs live vol surface. "
                "Heston: stochastic variance (CIR process). "
                "Bates: Heston + Poisson jumps."
            ),
        )
        vol_model = VOL_MODEL_OPTIONS[vol_model_label]

        # ── Jump parameters (only shown for Bates) ──
        if vol_model == "bates":
            with st.expander("Jump Parameters (Bates)", expanded=True):
                st.caption("Merton jump-diffusion: N(t) ~ Poisson(λ), each jump log-N(μ_J, σ_J²)")
                lam_j = st.slider("λ (jump intensity, pa)", 0.0, 2.0, 0.10, 0.01,
                                  format="%.2f", key="lam_j",
                                  help="Expected number of jumps per year.")
                mu_j = st.slider("μ_J (mean log-jump)", -0.30, 0.10, -0.05, 0.01,
                                 format="%.2f", key="mu_j",
                                 help="Average log-return on jump (negative = downward crash).")
                sig_j = st.slider("σ_J (log-jump vol)", 0.01, 0.50, 0.10, 0.01,
                                  format="%.2f", key="sig_j",
                                  help="Dispersion of jump size.")
            jump_params = dict(lam=lam_j, mu_J=mu_j, sig_J=sig_j)
        else:
            jump_params = None

        if vol_model in ("local",):
            st.caption(
                "ℹ️ Local vol uses the Dupire surface from the selected snapshot. "
                "The flat σ below is used only as FD fallback."
            )
        elif vol_model == "heston":
            st.caption("ℹ️ Heston: uses κ, θ, γ, ρ, v₀ from the Heston section above.")
        elif vol_model == "bates":
            st.caption("ℹ️ Bates: Heston vol process + jump diffusion as configured above.")

        st.divider()

        # ─────────────────────────────────────
        # Section 4: Monte Carlo
        # ─────────────────────────────────────
        with st.expander("**④ Monte Carlo**", expanded=False):
            n_paths = st.select_slider(
                "Paths (N)",
                options=[1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000],
                value=10_000,
                key="n_paths",
                help="More paths → lower error but slower. 10K is interactive; 100K shows warning.",
            )
            if n_paths >= 100_000:
                st.warning("⚠️ 100K+ paths may take >10s on Streamlit Cloud.")

            n_steps = st.slider("Time Steps (M)", 50, 500, 252, 50,
                                key="n_steps",
                                help="Steps per year for path simulation.")
            seed = st.number_input("Random Seed", 0, 9999, 42, 1, key="seed",
                                   help="Fixed seed for reproducible results.")
            antithetic = st.toggle("Antithetic Variates", value=True, key="antithetic",
                                   help="Halves standard error for same path count.")

        st.divider()

        # ─────────────────────────────────────
        # Section 5: FDM / PDE
        # ─────────────────────────────────────
        with st.expander("**⑤ FDM / PDE Grid**", expanded=False):
            st.caption("Explicit FD on log-price grid. Stability requires Δτ/Δx² ≤ 0.5")
            N_x = st.slider("S-grid steps (Nₓ)", 50, 500, 150, 50, key="N_x",
                            help="Spatial resolution. 150 is fast; 500 is paper-accuracy.")
            N_tau = st.slider("τ-grid steps (Nτ)", 50, 500, 100, 50, key="N_tau",
                              help="Time steps in transformed domain.")
            x_min = st.slider("Log-domain lower (x_min)", -10.0, -2.0, -5.0, 0.5,
                              key="x_min", help="Left boundary: S = C*exp(x_min)")

        st.divider()
        st.caption("v0.5.0 — AutoCallable Analytics Platform")

    # ─────────────────────────────────────
    # Build and return the params dict
    # ─────────────────────────────────────
    from app.autocallable import from_security_dict
    try:
        ac = from_security_dict(sec_params, S_ref=S0)
    except Exception as e:
        st.error(f"Product initialization error: {e}")
        ac = None

    params = {
        # Data
        "snapshot_key":   selected_key,
        "snapshot_label": selected_label,
        "snapshot_df":    snap_df,
        "S0":             S0,
        "r":              r,
        "q":              q,
        # Product
        "security_name":   sec_name,
        "security_params": sec_params,
        "autocallable":    ac,
        # Vol model
        "sigma":  sigma_flat,
        "vol_model": vol_model,
        "heston_params": dict(v0=v0, kappa=kappa, theta=theta, gamma=gamma, rho=rho),
        "jump_params":   jump_params,
        "v0":     v0,
        "kappa":  kappa,
        "theta":  theta,
        "gamma":  gamma,
        "rho":    rho,
        "use_calibrated_heston": use_calibrated,
        # Monte Carlo
        "n_paths":   n_paths,
        "n_steps":   n_steps,
        "seed":      seed,
        "antithetic": antithetic,
        # FDM
        "N_x":   N_x,
        "N_tau": N_tau,
        "x_min": x_min,
    }
    return params
