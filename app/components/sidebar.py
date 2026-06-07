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

SECTION STRUCTURE (in order of user-facing importance):
    ① Market Data         — snapshot date, risk-free rate, dividend yield
    ② Autocallable Product — security selector (pre-built + custom), reference spot
    ③ Volatility Model    — flat / local / Heston / Bates selector
    ④ Model Parameters    — conditional: Heston params (Heston/Bates only);
                            jump params (Bates only); flat-σ always shown
    ⑤ Monte Carlo         — N paths, time steps, seed, antithetic
    ⑥ FDM / PDE Grid      — spatial + time grid size, domain bounds

PERSISTENCE DESIGN:
    All keyed widgets follow a strict two-step pattern:
        Step 1: _ensure_sidebar_defaults() pre-populates every session_state key
                BEFORE any widget is rendered. Keys are only set if not already
                present — user selections are never overwritten.
        Step 2: No widget passes `value=`, `index=`, or positional default args
                that could override session_state. The widget reads exclusively
                from session_state.

    WHY NO value= IN WIDGETS: In Streamlit, passing value=X alongside key=K means:
        - If K NOT in session_state → initialise with X  (desired)
        - If K IS in session_state  → X is *supposed* to be ignored, but in
          practice Streamlit 1.x has edge-case bugs where the explicit value=
          wins when the stored value equals a boundary or the options list changes.
        Safe pattern: pre-initialise in session_state → pass NO value/index to widget.

    SNAPSHOT DATE STORAGE:
        We store the date KEY STRING (e.g. "20260606_1200"), not an integer index.
        Reason: if new snapshot files are added between page renders, integer indices
        shift and Streamlit silently resets the selectbox to position 0. String keys
        are immune to list reordering.
"""

import streamlit as st
from app.data_loader import list_available_snapshots, load_snapshot, get_spot_price, get_rfr, resolve_data_dir
from app.components.securities import list_securities, get_security


@st.cache_data(show_spinner=False)
def _cached_load_snapshot(key: str, data_dir: str):
    """
    Module-level cached snapshot loader.

    WHY MODULE-LEVEL: Defining @st.cache_data inside render_sidebar() creates a
    new function object on every call, so Streamlit's cache hash changes on every
    page navigation. Module-level ensures the same cache entry is reused across
    all pages and all render_sidebar() calls.
    """
    return load_snapshot(key, data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Vol model registry (module-level constant — same on every page render)
# ─────────────────────────────────────────────────────────────────────────────
VOL_MODEL_OPTIONS = {
    "Flat (Black-Scholes)":   "flat",
    "Local Vol (Dupire)":     "local",
    "Heston Stochastic Vol":  "heston",
    "Bates (Heston + Jumps)": "bates",
}


def _ensure_sidebar_defaults(snaps: list, pre_built: list) -> None:
    """
    Pre-populate every session_state key used by a sidebar widget.

    WHY: Widgets with key= read from session_state. If the key is absent,
    Streamlit uses the widget's value= arg (or position-default). By setting
    all keys here — BEFORE any widget is created — we guarantee that the
    widget always reads from session_state, and user changes persist through
    page navigation.

    Rule: only sets a key when it is NOT already in session_state. Never
    overwrites a value the user has changed.

    Args:
        snaps:     List of {key, label} dicts from list_available_snapshots().
        pre_built: List of pre-configured security names from list_securities().
    """
    all_sec_names = pre_built + [f"✏️ {n}" for n in st.session_state.get("custom_securities", {})]
    latest_snap_key = snaps[-1]["key"] if snaps else ""

    defaults: dict = {
        # ① Market Data
        "snapshot_key_stored": latest_snap_key,   # STRING key, not integer index
        "r":                   0.045,
        "q":                   0.014,
        # ② Product
        "security_name":       all_sec_names[0] if all_sec_names else "",
        "S0":                  5500.0,             # overridden from market data below if absent
        # ③ Vol model
        "vol_model_label":     "Flat (Black-Scholes)",
        # ④ Model parameters
        "sigma_flat":          0.20,
        "use_calibrated_heston": False,
        "v0":                  0.04,
        "kappa":               1.5,
        "theta":               0.04,
        "gamma":               0.30,
        "rho":                 -0.70,
        "lam_j":               0.10,
        "mu_j":                -0.05,
        "sig_j":               0.10,
        # ⑤ Monte Carlo
        "n_paths":             10_000,
        "n_steps":             250,
        "seed":                42,
        "antithetic":          True,
        # ⑥ FDM
        "N_x":                 150,
        "N_tau":               100,
        "x_min":               -5.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar(page_name: str = "") -> dict:
    """
    Render the shared Assumptions sidebar and return all parameters as a dict.

    Called at the top of every page. Pre-initializes all session_state keys on
    first run, then lets widgets read exclusively from session_state.

    Returns:
        params dict with all model and product parameters.
    """
    data_dir = resolve_data_dir()

    with st.sidebar:
        st.markdown("## ⚙️ Assumptions")
        st.caption("Changes apply on the next run (⌘R / Rerun)")
        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ①: Market Data
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**① Market Data**")

        snaps = list_available_snapshots(data_dir)
        if not snaps:
            st.error("No snapshots found. Run: python scripts/generate_synthetic_data.py")
            st.stop()

        pre_built = list_securities()

        # Pre-populate ALL defaults before any widget is rendered.
        # This is the single call that guarantees persistence.
        _ensure_sidebar_defaults(snaps, pre_built)

        # ── Snapshot selectbox ──────────────────────────────────────────────
        # WHY STRING KEY: we store the date-key string ("20260606_1200") instead
        # of an integer index. Integer indices break when new snapshot files are
        # added (list reorders), causing Streamlit to silently reset to position 0.
        snap_keys_list  = [s["key"]   for s in snaps]
        snap_labels_map = {s["key"]: s["label"] for s in snaps}

        # Guard: if stored key no longer exists (file deleted), reset to latest
        if st.session_state["snapshot_key_stored"] not in snap_keys_list:
            st.session_state["snapshot_key_stored"] = snap_keys_list[-1]

        selected_key = st.selectbox(
            "Data Date",
            options=snap_keys_list,
            format_func=lambda k: snap_labels_map.get(k, k),
            key="snapshot_key_stored",   # stores the date string, e.g. "20260606_1200"
            help="Pre-collected SPX options snapshot. Stored as date key — persists across pages.",
        )
        selected_label = snap_labels_map.get(selected_key, selected_key)

        # Load snapshot via module-level cache (persists across all page navigations)
        snap_df    = _cached_load_snapshot(selected_key, data_dir)
        S0_market  = get_spot_price(snap_df)
        rfr_market = get_rfr(snap_df)

        # r and S0: set from market data the very first time (session_state absent);
        # after that, whatever the user typed is in session_state and we don't touch it.
        if "r_initialized" not in st.session_state:
            st.session_state["r"]            = float(round(rfr_market, 4))
            st.session_state["S0"]           = float(round(S0_market,  1))
            st.session_state["r_initialized"] = True

        col_r, col_q = st.columns(2)
        with col_r:
            # NO value= — widget reads from session_state["r"] set above
            r = st.number_input(
                "r (risk-free)",
                min_value=0.0, max_value=0.20,
                step=0.001, format="%.3f",
                key="r",
                help="Continuously compounded annual rate. Default from snapshot (^IRX).",
            )
        with col_q:
            # NO value= — widget reads from session_state["q"] = 0.014 (set in _ensure_sidebar_defaults)
            q = st.number_input(
                "q (div yield)",
                min_value=0.0, max_value=0.10,
                step=0.001, format="%.3f",
                key="q",
                help="SPX dividend yield. ~1.4% long-run average.",
            )

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ②: Autocallable Product
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**② Autocallable Product**")

        custom_secs  = st.session_state.get("custom_securities", {})
        custom_names = list(custom_secs.keys())
        all_sec_names = pre_built + ([f"✏️ {n}" for n in custom_names] if custom_names else [])

        # Guard: if stored security no longer in options, reset to first
        if st.session_state["security_name"] not in all_sec_names and all_sec_names:
            st.session_state["security_name"] = all_sec_names[0]

        # NO index= — widget reads from session_state["security_name"]
        sec_name_display = st.selectbox(
            "Security",
            options=all_sec_names,
            key="security_name",
            help="Pre-configured structures or your own custom product from the Product Builder.",
        )

        # Resolve to actual lookup key (strip ✏️ prefix for custom securities)
        sec_key = sec_name_display.replace("✏️ ", "", 1) if sec_name_display.startswith("✏️ ") else sec_name_display
        sec_params = custom_secs.get(sec_key) or get_security(sec_key)

        # S0 — NO value= here either; session_state["S0"] was set above from market data
        S0 = st.number_input(
            "Reference Spot (S₀)",
            min_value=100.0, max_value=20000.0,
            step=10.0, format="%.1f",
            key="S0",
            help="SPX reference level at trade date. Set from snapshot on first load.",
        )

        # Quick one-line term sheet
        if sec_params:
            mat  = sec_params.get("maturity_years", "?")
            freq = sec_params.get("obs_frequency",  "?")
            cb   = sec_params.get("call_barrier",   "?")
            cpn  = sec_params.get("coupon_pa",      "?")
            pb   = sec_params.get("protection_barrier", "?")
            st.caption(
                f"T={mat}y · {freq} · call@{int(cb*100) if isinstance(cb,float) else cb}% "
                f"· {int(cpn*100) if isinstance(cpn,float) else cpn}%pa "
                f"· KI@{int(pb*100) if isinstance(pb,float) else pb}%"
            )

        st.page_link("pages/06_Product_Builder.py", label="✏️ Build a custom structure", icon="🔧")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ③: Volatility Model
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**③ Volatility Model**")

        # NO index= — reads from session_state["vol_model_label"]
        vol_model_label = st.selectbox(
            "Model",
            options=list(VOL_MODEL_OPTIONS.keys()),
            key="vol_model_label",
            help=(
                "Flat: constant σ (fastest, classic). "
                "Local Vol: σ(S,t) from Dupire. "
                "Heston: stochastic variance (CIR). "
                "Bates: Heston + Poisson jumps."
            ),
        )
        vol_model = VOL_MODEL_OPTIONS[vol_model_label]

        _vol_notes = {
            "flat":   "Classic Black-Scholes: σ is constant. Fastest pricer.",
            "local":  "Dupire local vol σ(S,t) computed from the market surface.",
            "heston": "Stochastic variance: dv = κ(θ−v)dt + γ√v dW_v. Set params in ④.",
            "bates":  "Heston + Poisson jumps. Set variance & jump params in ④.",
        }
        st.caption(_vol_notes[vol_model])

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ④: Model Parameters (conditional on vol model)
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**④ Model Parameters**")

        # ── ④a: Flat vol — always shown (used by FD as fallback, and flat MC)
        # NO value= — reads from session_state["sigma_flat"]
        sigma_flat = st.number_input(
            "Flat Vol σ (BS / FD fallback)",
            min_value=0.01, max_value=1.0,
            step=0.01, format="%.3f",
            key="sigma_flat",
            help="Used by FD pricer and flat-vol MC. For Heston/Bates MC the variance SDE is used; this acts as fallback.",
        )

        # Pull Heston params from session_state for use in return dict even when
        # the expander is hidden (vol_model = flat / local).
        v0    = st.session_state["v0"]
        kappa = st.session_state["kappa"]
        theta = st.session_state["theta"]
        gamma = st.session_state["gamma"]
        rho   = st.session_state["rho"]
        use_calibrated = st.session_state["use_calibrated_heston"]

        # ── ④b: Heston parameters — only shown for Heston / Bates
        if vol_model in ("heston", "bates"):
            with st.expander("Heston Variance Process", expanded=True):
                st.caption("dv = κ(θ−v)dt + γ√v dW_v,  Corr(W_S, W_v) = ρ")

                # All widgets below: NO value= — reads from session_state pre-set above
                use_calibrated = st.toggle(
                    "Use calibrated values (from Vol Surface page)",
                    key="use_calibrated_heston",
                    help="Pulls calibrated parameters set by Vol Surface → Tab 2 → Calibrate Heston.",
                )

                # If toggle is ON: inject calibrated values into slider session_state keys
                # BEFORE the sliders are rendered. Sliders read exclusively from session_state,
                # so setting the keys here causes them to display the calibrated values.
                _cal = st.session_state.get("heston_cal")
                _sliders_locked = False
                if use_calibrated:
                    if _cal:
                        # Clamp each param to its slider's min/max before injecting
                        st.session_state["v0"]    = float(min(max(_cal.get("v0",    0.04), 0.001), 0.30))
                        st.session_state["kappa"] = float(min(max(_cal.get("kappa", 1.5),  0.1),  10.0))
                        st.session_state["theta"] = float(min(max(_cal.get("theta", 0.04), 0.001), 0.30))
                        st.session_state["gamma"] = float(min(max(_cal.get("gamma", 0.30), 0.05),  1.5))
                        st.session_state["rho"]   = float(min(max(_cal.get("rho",  -0.70), -0.99), -0.01))
                        _sliders_locked = True
                        st.success(
                            f"✅ Calibrated values active — "
                            f"RMSE {_cal.get('rmse_vol_pts', 0.0):.1f} vol-pts  "
                            f"({_cal.get('n_quotes', '?')} quotes fitted)"
                        )
                    else:
                        # No calibration in session — guide user to run it
                        st.warning(
                            "⚠️ No calibration found. "
                            "Go to **Vol Surface → Tab 2** and click **Calibrate Heston** first.",
                            icon="📈",
                        )

                v0 = st.slider(
                    "v₀ (initial variance)",
                    min_value=0.001, max_value=0.30, step=0.001, format="%.3f",
                    key="v0",
                    disabled=_sliders_locked,
                    help="√v₀ = current instantaneous vol. v₀=0.04 → 20% vol.",
                )
                kappa = st.slider(
                    "κ (mean reversion speed)",
                    min_value=0.1, max_value=10.0, step=0.1, format="%.1f",
                    key="kappa",
                    disabled=_sliders_locked,
                )
                theta = st.slider(
                    "θ (long-run variance)",
                    min_value=0.001, max_value=0.30, step=0.001, format="%.3f",
                    key="theta",
                    disabled=_sliders_locked,
                    help="√θ = long-run vol. θ=0.04 → 20%.",
                )
                gamma = st.slider(
                    "γ (vol-of-vol)",
                    min_value=0.05, max_value=1.5, step=0.01, format="%.2f",
                    key="gamma",
                    disabled=_sliders_locked,
                )
                rho = st.slider(
                    "ρ (spot-vol correlation)",
                    min_value=-0.99, max_value=-0.01, step=0.01, format="%.2f",
                    key="rho",
                    disabled=_sliders_locked,
                    help="Negative for equities: falling market → rising vol.",
                )
                feller = kappa * theta > 0.5 * gamma ** 2
                if feller:
                    st.success(f"Feller ✓  κθ={kappa*theta:.4f} > ½γ²={0.5*gamma**2:.4f}")
                else:
                    st.warning(f"Feller ✗  κθ={kappa*theta:.4f} ≤ ½γ²={0.5*gamma**2:.4f}")
                st.caption(f"Implied ATM vol ≈ {((v0+theta)/2)**0.5*100:.1f}%")

        # ── ④c: Jump parameters — only shown for Bates
        lam_j = st.session_state["lam_j"]
        mu_j  = st.session_state["mu_j"]
        sig_j = st.session_state["sig_j"]

        if vol_model == "bates":
            with st.expander("Jump Parameters (Bates)", expanded=True):
                st.caption("N(t) ~ Poisson(λ),  each jump: log-return ~ N(μ_J, σ_J²)")
                lam_j = st.slider(
                    "λ (jumps per year)",
                    min_value=0.0, max_value=2.0, step=0.01, format="%.2f",
                    key="lam_j",
                    help="Expected number of jumps per year.",
                )
                mu_j = st.slider(
                    "μ_J (mean log-jump)",
                    min_value=-0.30, max_value=0.10, step=0.01, format="%.2f",
                    key="mu_j",
                    help="Negative = downward crash bias.",
                )
                sig_j = st.slider(
                    "σ_J (log-jump vol)",
                    min_value=0.01, max_value=0.50, step=0.01, format="%.2f",
                    key="sig_j",
                )
            jump_params = dict(lam=lam_j, mu_J=mu_j, sig_J=sig_j)
        else:
            jump_params = None

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ⑤: Monte Carlo
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("**⑤ Monte Carlo**", expanded=False):
            # select_slider: NO value= — reads from session_state["n_paths"]
            n_paths = st.select_slider(
                "Paths (N)",
                options=[1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000],
                key="n_paths",
                help="More paths → lower MC error, but slower. 10K is interactive.",
            )
            if n_paths >= 100_000:
                st.warning("⚠️ 100K+ paths may take >10s on Streamlit Cloud.")

            # slider: NO positional value arg — reads from session_state["n_steps"]
            n_steps = st.slider(
                "Time Steps (M)",
                min_value=50, max_value=500, step=50,
                key="n_steps",
                help="Discrete steps per year for path simulation.",
            )
            seed = st.number_input(
                "Random Seed", min_value=0, max_value=9999, step=1,
                key="seed",
                help="Fixed seed for reproducible runs.",
            )
            # toggle: NO value= — reads from session_state["antithetic"]
            antithetic = st.toggle(
                "Antithetic Variates",
                key="antithetic",
                help="Pairs each path with its mirror; roughly halves variance.",
            )
            if vol_model != "flat" and antithetic:
                st.caption("ℹ️ Antithetic auto-disabled for non-flat models (variance SDE breaks pairing).")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Section ⑥: FDM / PDE Grid
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("**⑥ FDM / PDE Grid**", expanded=False):
            st.caption("Explicit FD on log-price grid. Stability: Δτ/Δx² ≤ 0.5")
            # All sliders: NO positional value — reads from session_state
            N_x = st.slider(
                "S-grid steps (Nₓ)",
                min_value=50, max_value=500, step=50,
                key="N_x",
                help="Spatial resolution. 150=fast, 500=paper accuracy.",
            )
            N_tau = st.slider(
                "τ-grid steps (Nτ)",
                min_value=50, max_value=500, step=50,
                key="N_tau",
            )
            x_min = st.slider(
                "Log-domain lower (x_min)",
                min_value=-10.0, max_value=-2.0, step=0.5,
                key="x_min",
                help="Left boundary: S = S_ref·exp(x_min)",
            )

        st.divider()
        st.caption("v0.5.1 — AutoCallable Analytics Platform")

    # ─────────────────────────────────────────────────────────────────────────
    # Build and return the params dict
    # ─────────────────────────────────────────────────────────────────────────
    from app.autocallable import from_security_dict
    try:
        ac = from_security_dict(sec_params, S_ref=S0)
    except Exception as e:
        st.sidebar.error(f"Product init error: {e}")
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
        "security_name":   sec_name_display,
        "security_params": sec_params,
        "autocallable":    ac,
        # Vol model
        "sigma":           sigma_flat,
        "vol_model":       vol_model,
        "heston_params":   dict(v0=v0, kappa=kappa, theta=theta, gamma=gamma, rho=rho),
        "jump_params":     jump_params,
        "v0":              v0,
        "kappa":           kappa,
        "theta":           theta,
        "gamma":           gamma,
        "rho":             rho,
        "use_calibrated_heston": use_calibrated,
        # Monte Carlo
        "n_paths":    n_paths,
        "n_steps":    n_steps,
        "seed":       seed,
        "antithetic": antithetic,
        # FDM
        "N_x":   N_x,
        "N_tau": N_tau,
        "x_min": x_min,
    }
    return params
