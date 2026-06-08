# Changelog

All notable changes to the AutoCallable Analytics Platform are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/): patch (0.0.X) for bug fixes, minor (0.X.0) for new features, major (X.0.0) for architecture changes.

## [0.5.12] — 2026-06-08

### Fixed

- **`app/components/sidebar.py`** — Sidebar settings no longer reset on page navigation.
  Two-layer fix:
  1. **`_sidebar_backup` belt-and-suspenders pattern**: At the end of every `render_sidebar()`
     call, all current widget values are saved to `st.session_state["_sidebar_backup"]` — a
     non-widget key that Streamlit's widget-state cleanup never removes. On the next render,
     `_ensure_sidebar_defaults()` reads the backup first and restores the user's last-used values
     instead of hardcoded factory defaults. The `r_initialized` block also checks the backup so
     manual r/S0 overrides survive page navigation.
  2. **`.streamlit/config.toml`** (new file): Sets `fileWatcherType = "none"` and
     `runOnSave = false`. Root cause: the app folder lives on OneDrive, which continuously writes
     sync metadata and .pyc files. Streamlit's default file watcher interpreted these as code
     changes and restarted the server, wiping ALL session state mid-session. Disabling the watcher
     eliminates this. Trade-off: must manually `Ctrl+C` and re-run after code changes.

## [0.5.11] — 2026-06-08

### Fixed

- **`app/mc_standard.py`** — Coupon bug: `coupon_is_paid(S_T.mean())` used the mean of all
  survived terminal spots to decide coupon payment, applying the same result to every path.
  Fixed to per-path evaluation: `coupon_mask = S_T >= self.ac.coupon_barrier * self.S_ref`.
  This correctly determines coupon payment individually for each survived path.

- **`app/pde_pricer.py`** — FDM local vol numerical blow-up (price = $1500 instead of ~$950).
  Root cause: `_price_local_vol()` used `self.N_tau` computed in `__init__` based on flat vol
  (σ=0.20), but local vol can reach σ=1.0 (clamped). Courant number σ²·dt/dx² ≈ 9 >> 0.5 made
  the explicit scheme unconditionally unstable. Fix: after building the local vol grid, compute
  `max_sigma = np.max(_LV_g)` and auto-correct `N_tau` to satisfy Courant ≤ 0.5 for the maximum
  local vol in the surface.

- **`app/mc_survival.py`** — Survival MC performance: converted Python for-loop over paths to
  vectorised active-mask pattern. All 10K paths are now processed simultaneously using numpy
  arrays. Added vectorised helpers: `_p_survive_vec()`, `_sample_below_vec()`,
  `_get_sigma_local_vec()`, `_advance_heston_variance_vec()`, `_jump_survival_correction_vec()`.
  The scalar `_price_one_path()` is retained for the 50 stored paths (fast enough). Expected
  speedup: ~8s → ~0.1s for 10K paths locally; ~20-40s → ~0.5-2s on Streamlit Cloud.

- **`app/pages/02_Pricer.py`** — VolSurface caching: `_build_vol_surface()` now caches the
  `VolSurface` object and shared Dupire grid interpolator in `session_state` keyed by
  snapshot_key + S0 + r + q. The 40×25 Dupire grid is built once and passed to all three pricers
  via the new `local_vol_interp` parameter, eliminating redundant grid builds (was 3× per Run).

- **`app/mc_standard.py`, `app/mc_survival.py`, `app/pde_pricer.py`** — All three pricers now
  accept an optional `local_vol_interp` parameter (pre-built `RegularGridInterpolator`). When
  provided, the pricer uses the shared interpolator instead of building its own. This reduces
  total Dupire grid build cost from ~3ms to ~1ms per pricing run.

- **`app/components/sidebar.py`** — Sidebar settings (Heston/Bates parameters) resetting to
  defaults when navigating between pages.

  **Root cause**: Streamlit's widget-state cleanup removes `session_state` keys for any widget
  that was rendered in the *previous* run but is NOT rendered in the *current* run.
  `v0`, `kappa`, `theta`, `gamma`, `rho` (Heston) and `lam_j`, `mu_j`, `sig_j` (Bates)
  were only rendered inside `if vol_model in ("heston", "bates"):` / `if vol_model == "bates":`
  blocks.  When navigating to a page while on Flat or Local vol, those sliders were skipped,
  their session_state entries were cleaned up at the end of the render pass, and on the next
  render `_ensure_sidebar_defaults()` found them absent and reset them to defaults.

  **Fix**: Both the "Heston Variance Process" and "Jump Parameters (Bates)" expanders are now
  **always rendered** regardless of the active vol model. The `if vol_model …` gates are
  replaced by `expanded=_heston_active` / `expanded=_bates_active` arguments, so the expanders
  are visually collapsed when inactive but their widget keys are always claimed by Streamlit
  and never cleaned up. An inline caption is shown inside collapsed expanders to explain why
  the parameters are preserved there. Per-session Heston calibration values and custom
  jump params now persist correctly across all page navigations.

  - `if vol_model in ("heston", "bates"): with st.expander(…)` →
    `_heston_active = …; with st.expander(…, expanded=_heston_active):`
  - `if vol_model == "bates": with st.expander(…)` →
    `_bates_active = …; with st.expander(…, expanded=_bates_active):`
  - Removed now-redundant pre-reads of `lam_j`, `mu_j`, `sig_j` from session_state
    (values always come from the always-rendered sliders).
  - Updated stale comment on the Heston pre-read block.

---

## [0.5.10] — 2026-06-07

### Fixed

- **`app/pde_pricer.py` — `continuous_autocall_closedform()`** — Function was returning `None`
  because the final 8 lines of its body were missing (OneDrive file truncation). The last stored
  line was `expected_life = p_cross * expected_call_time + p_n` (truncated mid-variable). Added:
  ```
  expected_life = p_cross * expected_call_time + p_no_cross * T
  pv_coupons    = coupon_pa * notional * expected_life * exp(-r * expected_life / 2.0)
  price         = pv_call + pv_no_call + pv_coupons
  return float(np.clip(price, 0.0, notional * (1.0 + coupon_pa * T)))
  ```
  This unblocked `test_closedform_finite_positive` which was failing with TypeError from
  `np.isfinite(None)`. Result for Phoenix Autocall params (S0=5600, σ=0.18, T=2y): **$1,160**.
  Also restored the full 792-line file from the 754-line FUSE-truncated version. **59/59 tests now pass.**

- **`app/pages/02_Pricer.py`** — "Run All Pricers" required two clicks to clear the
  "settings have changed" warning banner. Root cause: `st.session_state["pricer_last_run_fp"]`
  was updated mid-script, but Streamlit does not re-evaluate widgets or banners until the next
  full script run. The fingerprint comparison at the top of the script was still reading the old
  value, so the warning persisted after the first click. Fix: added `st.rerun()` immediately after
  the fingerprint is stored. The rerun re-executes the script with the new fingerprint already in
  session_state → banner does not appear → button returns to idle. Pricers do NOT re-run on the
  forced rerun because `st.button()` evaluates to `False` after the first click.

- **`app/components/sidebar.py`** — Bates jump-diffusion default parameters were too conservative
  (`lam_j=0.10, mu_j=-0.05, sig_j=0.10`), producing near-zero jump impact and making the Bates
  price indistinguishable from the Heston price. Changed to `lam_j=0.50, mu_j=-0.15, sig_j=0.25`
  (0.5 jumps/yr, −15% mean jump, 25% jump vol). These values produce a visible and financially
  meaningful price difference vs. Heston, making the comparison informative during the demo.

### Performance (continued from v0.5.8 — stale bytecode issue resolved)

- **Root cause discovered**: Prior session's edits to `pde_pricer.py`, `mc_survival.py`, and
  `vol_surface.py` updated the Windows files correctly, but Python was still executing the old
  list-comprehension bytecode. The OneDrive FUSE mount showed the same mtime as the cached `.pyc`
  files (sync lag), so Python's mtime check did not detect a change and kept using stale bytecode.
  Proof: `pyc_baked_mtime == source_mtime` both = 1780808850. Fixed by `touch`ing all three source
  files to force mtime advancement → Python recompiled from source → prior fixes took effect.

- **`app/vol_surface.py`** — Added `dupire_local_vol_grid(m_axis, T_axis)` method. Evaluates the
  Dupire formula for an entire (moneyness × TTM) grid using 4 vectorised C-level
  `RectBivariateSpline.ev()` calls and one vectorised `_bs_vec()` pass, replacing 1,000 sequential
  Python `dupire_local_vol()` calls. Returns `np.ndarray` of shape `(nT, nM)`. Values clamped to
  `[0.05, 1.0]`. Uses `scipy.special.ndtr` (C-backed Φ(x)) instead of `scipy.stats.norm.cdf`
  for ~10× faster normal CDF evaluation. Grid build time: **0.001s** for a 40×25 grid.

- **`app/pde_pricer.py`** — Replaced `np.vectorize(lambda m, t: dupire_local_vol(m, t))(M_g, T_g)`
  with `vol_surface.dupire_local_vol_grid(m_axis, t_axis)` in `_price_local_vol()`.

- **`app/mc_standard.py`** — Same replacement in `_build_local_vol_interp()`.

- **`app/mc_survival.py`** — Same replacement in `_build_local_vol_interp()`.

### Timing (10K paths, real SPX Jun 6 data, Phoenix Autocall, Heston vol)
- `dupire_local_vol_grid` (40×25): **0.001s** (was ~0.3s per call × 1,000 calls = 300s)
- FDM (150×100, Heston): **0.02s**
- MC Standard: **0.12s** → $948.59 ± $1.68
- Survival MC: **8.14s** → $927.21 ± $0.62
- Total: **~8.3s** (was 5+ minutes)

### Known Issues (not fixed in this session)
- **FD local vol price = $1500 (numerical blow-up)**: `_price_local_vol` uses an explicit FD
  scheme in physical time space. With local vol σ up to 1.0, the Courant number
  `σ² · dt / dx² ≈ 27 >> 0.5` → scheme is unconditionally unstable. Fix requires switching to
  Crank-Nicolson or raising N_tau to ~5,000+. Heston/Bates FD pricing is unaffected (σ is bounded
  and grid is well-conditioned).
- **Survival MC Python loop**: `price()` still uses a Python for-loop over paths (~8s for 10K).
  Vectorisation would require significant refactor. Not in scope for this session.

---

## [0.5.9] — 2026-06-07

### Fixed
- **`app/components/sidebar.py`** — Default snapshot was always `snaps[-1]["key"]` (latest by
  lexicographic filename sort = `20260610_0945`, a 164 KB synthetic placeholder). Fresh sessions
  and reloaded sessions were silently pricing with synthetic data rather than the real Jun 6 SPX
  options chain (`20260606_2238`, 1.6 MB). Added `_best_default_snapshot_key()` helper that reads
  file sizes and selects the largest snapshot as the default. The real chain is ~10× larger than
  the synthetic placeholders, so it is reliably identified. Falls back to latest-by-name if file
  sizes cannot be read (e.g. cloud-only OneDrive files not yet synced).
  Updated `_ensure_sidebar_defaults()` signature to accept `data_dir` and wired the call site in
  `render_sidebar()` to pass `data_dir`. No change to session_state persistence logic.

---

## [0.5.8] — 2026-06-07

### Fixed (Performance)
- **`app/mc_survival.py`** — `MCSurvivalPricer` was calling `vol_surface.dupire_local_vol()`
  directly inside the simulation loop (once per path × per observation date = 80,000 calls at
  ~300μs each ≈ 30s for N=10K). Applied the same pre-built `RegularGridInterpolator` pattern
  already present in `MCStandardPricer`: build a 40×25 grid once in `__init__` (~1,000 calls,
  ~0.3s), then use the C-backed interpolator for all subsequent lookups (~5μs per call).
  Changes: added `_build_local_vol_interp()` method + `self._local_vol_interp` field; rewrote
  `_get_sigma_local()` to use the fast interpolator path with a slow fallback for safety.

- **`app/pde_pricer.py`** — `FDPricer._price_local_vol()` was calling `dupire_local_vol()` via
  a Python list comprehension at every time step for every grid node (N_x × N_tau ≈ 20,000 calls
  at ~300μs each ≈ 6s). Pre-built the same `RegularGridInterpolator` at the top of the method
  (one-time cost), then replaced the per-step list comprehension with a vectorised `np.stack`
  batch query (~50μs per step regardless of N_x).

- **`app/vol_surface.py`** — `dupire_local_vol()` was computing 8 calls to the inner `C(m, t)`
  function for only 4 unique (moneyness, time) points — each of `C(m, T)`, `C(m+dK, T)`, and
  `C(m-dK, T)` was evaluated twice. Cached the 4 unique evaluations (`C_0T`, `C_pT`, `C_mT`,
  `C_0dT`) and reused them across all three numerical derivatives. Reduces per-call cost by ~50%
  with no change in output values.

### Performance Impact
Expected overall speedup for Dupire Local Vol pricing: **~42s → ~2s** (comparable to Heston/Bates).
- Grid build one-time cost: ~0.3s (shared by MC + FDM)
- MCSurvival simulation: ~30s → ~0.5s
- FDM time-step loop: ~6s → ~0.05s
- dupire_local_vol per-call: ~300μs → ~150μs (halved by C() caching)

### Test Results
**59 / 59 PASSED** (test_pde_pricer.py, test_mc_pricers.py, test_payoffs.py).
test_heston.py skipped in sandbox — pre-existing OneDrive sync lag; unrelated to this session.

---

## [0.5.7] — 2026-06-07

### Added
- **`app/pages/03_FDM_Visualization.py`** — New "📊 Scheme Comparison" tab (Task #19):
  - **Panel 1 — Convergence**: Line chart of autocallable price vs spatial grid size
    (N_x = 50, 100, 200, 400) for both Explicit FD and Crank-Nicolson. Both lines
    converge to the same true price as the grid refines. CN arrives at the converged
    value with fewer time steps due to its O(Δτ²) accuracy vs O(Δτ) for explicit.
  - **Panel 2 — Stability**: Fixed N_x = 100, vary requested N_tau (5, 10, 20, 50,
    100, 200). Table shows *actual* N_tau each scheme uses after auto-correction:
    explicit balloons to satisfy the CFL condition (ρ ≤ 0.5); CN always uses the
    requested count. Demonstrates CN's unconditional stability vs explicit's CFL
    requirement in a directly comparable way.
  - **Panel 3 — Timing**: 3 repeated timing runs at N_x = N_tau = 200 for each
    scheme. Displays avg ± std ms per pricing call so users can see the per-step
    overhead tradeoff (Thomas solve vs vectorized array op) in context.
  - **Methodology expander**: Side-by-side comparison table (stability, accuracy,
    cost, recommendation) with LaTeX update rules for both schemes. Explains when
    to prefer each: explicit for education/validation (matches Paper 1), CN for
    production use (fewer steps at same accuracy, no CFL restriction).
  - Run button with session-state caching so the tab does not recompute on every
    Streamlit interaction; results persist until explicitly re-run.

- **`tests/test_pde_pricer.py`** — 2 new Crank-Nicolson tests (total: 79 tests):
  - `test_cn_return_grid_valid`: Verifies CN scheme with `return_grid=True` returns
    a well-formed non-None `V_grid` of shape `(N_x, n_snapshots)` with all finite,
    non-negative values. Guards the grid-snapshot loop inside the CN dispatch path.
  - `test_cn_price_digital_autocall`: Verifies CN produces a finite, in-range price
    for the Digital Autocall (fixed $50 coupon, 80% capital-protected terminal
    condition) — exercises a different autocall BC branch than the Phoenix fixture.

### Test Results
**79 / 79 PASSED** on Rohit's machine (was 77/77 before this session).
Sandbox count: 57/59 — 2 pre-existing errors from cloud-only OneDrive file truncations
(`app/vol_surface.py`, `app/heston.py`) that are complete on the Windows host.

---

## [0.5.6] — 2026-06-07

### Fixed
- **`app/pages/02_Pricer.py`** — Spread analysis line always said "within 2% at N=10K confirms
  implementations are consistent" even when the vol model was Heston/Bates and the spread was
  expected and meaningful (~5% in the reported case). When Heston/Bates is active the FDM uses
  flat σ while MC uses full stochastic-vol dynamics, so the gap is the vol-model premium — not
  a consistency error. Fixed: when Heston/Bates is selected, shows "📊 Vol-model premium: $X.XX
  (Y.YY%) — FDM uses flat σ=20.0%; MC uses full Heston/Bates dynamics (vol-of-vol, skew, ...)."
  When flat/local vol is selected (all methods share same vol model), the old green/yellow/red
  consistency check is preserved with a corrected red-case message.

---

## [0.5.5] — 2026-06-07

### Fixed
- **`app/components/sidebar.py`** — `use_calibrated_heston` toggle was a no-op (cosmetic only).
  Now actually injects calibrated params from `session_state["heston_cal"]` into the slider
  session_state keys (`v0`, `kappa`, `theta`, `gamma`, `rho`) *before* sliders render, so the
  calibrated values are displayed and used immediately. Each value is clamped to the slider's
  `[min, max]` before injection to prevent Streamlit's out-of-range reset. Sliders are locked
  with `disabled=True` while calibrated values are active so the user can't accidentally override them.
  Added ✅ success caption showing RMSE and quote count when calibrated; ⚠️ warning when toggle is ON
  but no calibration has been run yet.
- **`app/pages/02_Pricer.py`** — FDM column label `(flat* vol)` was cryptic and alarming.
  Replaced with clear, context-sensitive labels:
  - Heston/Bates selected → orange `(flat σ — stoch-vol PDE not implemented)`
  - Local vol selected → `(Dupire local vol)` (correctly implemented)
  - Flat σ → `(flat σ)` (unambiguous)

### Added
- **`app/pages/02_Pricer.py`** — Calibration status banner at the top of the page (visible only when
  Heston or Bates vol model is selected):
  - ✅ **green** if `session_state["heston_cal"]` exists — shows RMSE, quote count, v₀, κ, ρ
  - ℹ️ **blue info** if no calibration found — explains how to run it (Vol Surface → Tab 2 → Calibrate Heston) and how to activate it (sidebar toggle)
- **`app/pages/02_Pricer.py`** — Vol-mismatch explanation banner (⚠️ info box) before the three
  pricing metric columns when Heston/Bates is selected. Explains that FDM uses flat σ (stoch-vol
  PDE not implemented), MC pricers use full stochastic-vol dynamics, and any price gap is the
  vol-model premium — so users aren't confused by systematically different FDM vs MC prices.
- **`scripts/benchmark_calibration.py`** — Standalone speed benchmark: imports `_heston_cf_batch`
  and `_bates_cf_batch` to verify vectorized functions are loaded (not stale `.pyc`), then times
  Heston, Merton, and Bates calibration on a 25-quote synthetic dataset. Prints per-model timing
  vs target thresholds. Run with `python scripts/benchmark_calibration.py` from project root.

---

## [0.5.4] — 2026-06-07

### Fixed
- **`app/heston.py`** — Critical calibration performance overhaul (was timing out after 5+ min; now targets 15–45s Heston, 5–15s Merton, 15–30s Bates):
  - **Root cause 1 (eval cost)**: `HestonModel.calibrate()` used `differential_evolution` (~2,000 objective evals) with `scipy.integrate.quad` (100 adaptive integration points) called per-quote in a Python loop. For 80 quotes this was ~320,000 quad integrations per run.
  - **Root cause 2 (scalar CF loop)**: Even after switching to `_heston_call_batch_fast()`, the function still called `heston_char_fn()` 64 times in a Python loop per TTM group — 128 Python function calls per objective eval per TTM.
  - **Fix 1**: Added `_heston_cf_batch(u_arr, ...)` — vectorized Heston CF processing a full phi array in one numpy pass using broadcasting + `np.where` for degenerate-case handling. Replaces 64 Python function calls with 1 numpy operation.
  - **Fix 2**: Added `_bates_cf_batch(u_arr, ...)` — same for Bates CF (calls `_heston_cf_batch` internally for the Heston component).
  - **Fix 3**: Updated `_heston_call_batch_fast()` to call `_heston_cf_batch(phi_arr + 0j, ...)` and `_heston_cf_batch(phi_arr - 1j, ...)` instead of list comprehensions.
  - **Fix 4**: Updated `_bates_call_batch_fast()` to call `_bates_cf_batch()` instead of list comprehensions.
  - **Fix 5**: `HestonModel.calibrate()` now uses L-BFGS-B (4 starting points, ~800 evals total) instead of `differential_evolution` (~2,000 evals). `calibrate_merton()` and `calibrate_bates()` also use L-BFGS-B.
  - **Fix 6**: `n_sample` reduced to 25 quotes (was 80–100); moneyness filter [0.80, 1.20] added.
  - **Vol Surface page** (`app/pages/01_Vol_Surface.py`): Added visible timing estimates caption under the 4 control buttons ("Heston ~15–45 s", "All Models ~60–120 s"). Added `st.info` progress banners before each calibration step.

### Added
- **`app/heston.py`** — `_heston_cf_batch()`: Vectorized Heston characteristic function accepting a 1D complex array of Fourier frequencies. Processes entire phi grid in one numpy pass — ~64x fewer Python function calls per TTM vs scalar loop.
- **`app/heston.py`** — `_bates_cf_batch()`: Vectorized Bates characteristic function. Delegates Heston component to `_heston_cf_batch()`, multiplies by vectorized jump CF factor.

---

## [0.5.3] — 2026-06-07

### Fixed
- **`app/components/sidebar.py`** — Spinner fix: `n_steps` default changed from `252` to `250`.
  Streamlit 1.58 validates that a slider's session_state value is a valid step multiple. `252` is
  not a multiple of `step=50`, causing `StreamlitAPIException` → infinite spinner with no error shown.

### Added
- **`app/Home.py`** — Navigation section expanded from 3 to 6 page links:
  - Row 1 (existing): Vol Surface, Pricer, FDM Viz
  - Row 2 (new): Greeks, Scenarios, Product Builder — each with a description card
  - Concept Guide section with an `st.info` pointing users to `AutoCallable_Concept_Guide.html`
  - Vol Surface page description updated to mention Tab 4 Model Comparison
- **`AutoCallable_Concept_Guide.html`** — Footer updated: removed all MerQube interview references;
  changed to generic "A learning and demonstration tool for autocallable structured product pricing"

---

## [0.5.2] — 2026-06-07

### Fixed
- **`app/components/sidebar.py`** — Comprehensive persistence fix (all settings now persist across page navigation):
  - **Root cause 1 (snapshot date)**: Snapshot selectbox previously stored an integer *index*
    in `session_state["snapshot_idx"]`. When the snapshot list changed (new files added, or
    cloud-only OneDrive files inconsistently visible), the stored index went out-of-range and
    Streamlit silently reset to position 0. Fix: store the date-key string (e.g. "20260606_1200")
    in `session_state["snapshot_key_stored"]`. String keys are immune to list reordering.
  - **Root cause 2 (all widgets)**: 15 widgets (sliders, toggles, select_slider) passed an
    explicit `value=` or positional default arg alongside `key=`. In Streamlit 1.58 this causes
    edge-case resets when the stored value equals a boundary, or when the options list changes.
    Fix: removed ALL `value=`/`index=` args from every keyed widget. Widgets now read exclusively
    from `session_state`, which is pre-populated by `_ensure_sidebar_defaults()`.
  - **Root cause 3 (r / S0 initialization)**: `r` and `S0` were re-initialized from market data
    on every render via a fragile `if "r" not in session_state` check. Fix: a single sentinel key
    `r_initialized` is set once so market-data defaults are applied exactly once per session.
  - `_ensure_sidebar_defaults()` now covers every keyed widget (21 keys total).
  - Added guard clauses: if stored `snapshot_key_stored` or `security_name` no longer appear in
    the current options (file deleted, custom security cleared), fall back gracefully to the first valid option.

### Added
- **`app/components/sidebar.py`** — Sidebar reorganization (Task #21, partial):
  - New section order: ① Data → ② Product → ③ Vol Model → ④ Model Params → ⑤ MC → ⑥ FDM
  - Volatility Model selector moved UP to section ③ (before Heston params), so only relevant
    params are visible — Heston params are hidden when Flat or Local Vol is selected
  - Inline term-sheet preview (one caption line) below the security selectbox
  - Link to Product Builder page (✏️ Build a custom structure)
  - Custom securities from `session_state["custom_securities"]` appear in security dropdown
    prefixed with ✏️

- **`app/pages/06_Product_Builder.py`** — Custom Product Builder page (Task #20, complete):
  - Full form for defining a custom autocallable: name, structure type (phoenix / step_down / digital),
    maturity, observation frequency, call barrier, knock-in barrier, coupon barrier, coupon rate or
    digital coupon, notional, step-down schedule (for step_down type), protection type
  - Live payoff diagram that updates instantly as form params change (no "Run" button needed)
  - Key metrics panel: maturity, observation count, max autocall payoff, max KI loss
  - Observation schedule table with effective barrier per date (step-down aware)
  - Expandable payoff formula panel with LaTeX-style notation per structure type
  - Save button stores the product in `session_state["custom_securities"]` and switches the
    sidebar security dropdown to the new product automatically
  - Delete selector for removing saved custom products

## [0.5.1] — 2026-06-07

### Added
- **`app/pages/01_Vol_Surface.py`** — Tab 4 "📊 Model Comparison" (Task #18):
  - "🔬 Calibrate All Models" button triggers sequential calibration of Heston, Merton jump-diffusion, and Bates (Heston + jumps)
  - Fit quality comparison table: Model | Parameters | RMSE (vol pts) | Key params
  - Implied vol smile comparison chart: Market (solid) vs Heston (dashed) vs Merton (dot) vs Bates (dot-dash) at two user-selected tenors
  - Tenor selectors: 3m, 6m, 1y, 2y dropdowns for each of the two comparison panels
  - "📋 Full Calibrated Parameters" expandable table with all calibrated values per model
  - Calibration results cached in `st.session_state` so navigating away and back does not re-run calibration
  - Bates warm-started from Heston calibrated params for faster convergence
  - Imports added: `calibrate_merton`, `calibrate_bates`, `merton_call_price`, `bates_call_price`, `bs_implied_vol`
  - Control bar expanded from 3 to 4 columns to accommodate the new button
  - Tab declaration expanded from 3 to 4 tabs

## [0.5.0] — 2026-06-07

### Added (Feature D — Vol-Aware Pricers)
- **`app/mc_standard.py`**: Extended `MCStandardPricer` to support four volatility models via `vol_model=` constructor param:
  - `"flat"` (default) — existing GBM with constant σ; no behaviour change
  - `"local"` — Dupire local vol; pre-computes a 40×25 σ(S,t) grid, uses `RegularGridInterpolator` for vectorized lookups during sub-stepped simulation
  - `"heston"` — Euler-Maruyama CIR variance SDE (full truncation); correlated Brownian motion via Cholesky decomposition
  - `"bates"` — Heston variance + Poisson jump compound; jump log-return sampled as N(N_jumps·μ_J, N_jumps·σ_J²)
  - Sub-stepping: non-flat models use `n_steps_per_year=52` (weekly) sub-steps per observation interval for discretisation accuracy
  - Antithetic variates disabled automatically for non-flat models (paired sampling breaks under stochastic variance)
- **`app/mc_survival.py`**: Extended `MCSurvivalPricer` to support `"flat"`, `"local"`, `"heston"`, `"bates"`:
  - Local vol: `dupire_local_vol(S_t/S0, t)` lookup at each observation step
  - Heston: `_advance_heston_variance()` Euler-Maruyama with sub-steps; conditions on realised v_t for the survival probability formula
  - Bates: Heston variance + `_jump_survival_correction()` which computes `exp(-λ·Δt·P(jump crosses barrier))`
- **`app/pde_pricer.py`**: Added `vol_model="local"` path in `FDPricer`:
  - New `_price_local_vol()` method: direct log-price PDE `∂V/∂τ = (σ²_n/2)·d²V/dx² + (r−q−σ²_n/2)·dV/dx − r·V` with position-dependent σ per grid node
  - Falls through to existing flat-vol heat-equation solver for `"flat"` and to standard `price()` for `"heston"`/`"bates"` (Heston PDE requires 2D state space — future work)
- **`tests/test_mc_pricers.py`**: 5 new vol-model tests (total 77/77 passing):
  - `test_mc_heston_reasonable`: Heston MC price in [$700, $1100]
  - `test_mc_bates_no_jumps_matches_heston`: Bates(λ=0) == Heston to within $0.01 (same seed, identical paths)
  - `test_heston_variance_nonneg`: Full truncation holds under high γ=1.0 (no NaN/negative prices)
  - `test_mc_local_vol_reasonable`: Local vol MC price within $150 of flat vol MC
  - `test_fdm_local_vol_reasonable`: Local vol FDM price within $200 of flat vol FDM
- **`app/components/sidebar.py`**: New "③b Volatility Model" section with:
  - Selectbox: Flat / Local Vol (Dupire) / Heston Stochastic Vol / Bates (Heston + Jumps)
  - Conditional "Jump Parameters" expander (λ, μ_J, σ_J) shown only when Bates is selected
  - `vol_model`, `heston_params`, `jump_params` added to returned params dict
- **`app/pages/02_Pricer.py`**: Vol model integration:
  - `_build_vol_surface()` helper builds Dupire surface lazily when local vol is selected
  - `run_pricers()` passes `vol_model`, `heston_params`, `jump_params`, `vol_surface` to all three pricers
  - FDM falls back to flat when Heston/Bates selected (note shown in metric card header)
  - Antithetic variates disabled automatically for non-flat models
  - Vol model label shown in header caption

### Added (Settings Persistence + Change Detection)
- **`app/components/sidebar.py`**: Sidebar state now persists reliably across page navigation:
  - Module-level `_cached_load_snapshot()`: moved `@st.cache_data` out of `render_sidebar()` so the same cache is shared across all page scripts (was re-created on every page navigation, defeating the cache)
  - `_ensure_sidebar_defaults()`: initialises ALL 20 widget keys in `session_state` before any widget is rendered; ensures `key=`-only widgets (no `index=` / `value=` conflict) always have a stored default
  - `S0` and `r` defaults are now set from market data ONLY on first load; subsequent page navigations preserve user overrides
  - Removed `index=0` from security selectbox and redundant `index=st.session_state.X` from snapshot selectbox
- **All 5 app pages**: Settings-changed warning banner:
  - After `render_sidebar()`, each page computes a fingerprint of key pricing params (security, vol model, S0, r, q, σ, N, seed)
  - If params changed since last "Run" click, shows `st.warning("Settings have changed — re-run to update results")`
  - Fingerprint stored per-page in session_state on each "Run All Pricers" / "Run" click

### Fixed
- **`app/heston.py` `merton_char_fn` line 524**: Missing Itō correction — drift was `r - q - λ·μ̄_J`, should be `r - q - 0.5·σ² - λ·μ̄_J`. Heston CF embeds the `-v/2` Itō term implicitly in its variance factors; Merton (no stochastic variance) must be explicit. Fix reduces pricing error vs Black-Scholes from $2.00 → $0.0001.

### Test Results
**77 / 77 PASSED** (was 72/72 before this session)

---

## [0.3.3] — 2026-06-06

### Fixed
- **`app/vol_surface.py` line 277**: `RectBivariateSpline.__call__` with scalar inputs returns shape `(1,1)` array, not a scalar — `float()` raised "only 0-dimensional arrays can be converted to Python scalars". Fixed by casting clamp inputs to `float()` and indexing result with `[0, 0]`.
- **`app/pages/05_Scenarios.py` line 98**: `terminal_payoff()` requires `knocked_in: bool` positional argument that was missing. Fixed by inferring knock-in status from whether the final spot is below the protection barrier (`s < protection_barrier * S_ref`) — the standard convention for payoff diagrams.

### Test Results
**66 / 66 PASSED**

---

## [0.3.2] — 2026-06-06

### Added
- **`app/pages/05_Scenarios.py`**: Scenarios page — payoff intuition and what-if analysis.
  - **Tab 1 — Payoff Diagram**: Full maturity payoff curve across spot levels with three regimes shaded (loss zone, protected zone, called zone). Barrier overlays and regime text explain what each zone means in plain language.
  - **Tab 2 — What-If Analysis**: Interactive sliders for spot, vol, and rate. Reprices using Standard MC (N=3,000) on click. Displays base price vs what-if price with dollar and percent change.
  - **Tab 3 — Call Probability**: Bar chart of conditional call probability at each observation date + cumulative call probability curve. Table with survival probability column. Expected life calculation.
  - **Tab 4 — Value Surface**: Heatmap of price vs spot level × implied vol. Each cell is an independent MC run (N=1,000). Progress bar during computation. Color scale (red→green) makes price gradient immediately visible.

### Fixed
- **`README.md`**: Corrected GitHub clone URL from `rohitpittu/autocallable-pricer` to `rpadebet/autocallable-pricer`. Added `05_Scenarios.py` to project structure tree.

### Test Results
**66 / 66 PASSED** — confirmed after Scenarios page addition (no new tests required; page uses existing modules).

---

## [0.3.1] — 2026-06-06

### Added
- **`app/pages/04_Greeks.py`**: Greeks page — stable vs. noisy differentiation demo (Paper 3 key result).
  - **Tab 1 — Delta Stability**: Central-difference Delta computed across 10 bump sizes (0.05%–5% of S₀) × 8 random seeds for both Standard MC and Survival MC. Dual-panel plot shows MC with wide seed-scatter vs. Survival MC with tight bands. Computes and displays variance reduction ratio.
  - **Tab 2 — Delta Smile**: Delta as a function of spot level (70%–130% of S_ref) with call and protection barrier overlays. Uses `spot_override` parameter for correct Delta computation (barriers stay anchored at S_ref).
  - **Tab 3 — Vega Stability**: Same multi-seed, multi-bump analysis for Vega (∂V/∂σ). Shows that Survival MC's stability extends to vol sensitivity, not just spot sensitivity.
  - **Tab 4 — Methodology**: Full mathematical explanation of why Standard MC Greeks are unreliable near barriers (discontinuous payoff) and how One-Step Survival MC resolves this (analytical p_j, smooth payoff w.r.t. S₀). Includes practical hedging implications table.
- **`app/mc_standard.py` / `app/mc_survival.py`**: Added `spot_override` parameter to both pricers (from prior session). Paths start at `S0` (current spot) while call barriers remain anchored at `call_barrier * S_ref` (trade-date reference) — essential for correct Delta computation.

### Test Results
**66 / 66 PASSED** — confirmed after Greeks page addition.

---

## [0.3.0] — 2026-06-06

### Fixed
- **`app/autocallable.py`**: File was truncated at line 489 mid-function; `from_security_dict()` was missing its closing 10 lines. Appended missing `protection_floor`, `redemption_at_call`, `notional`, `stepped_barriers`, `correlation_matrix`, `asset_vols`, `description` parameters and closing parenthesis.
- **`app/autocallable.py`**: `__post_init__` validation now uses `protection_barrier > 1.0` (exclusive) to correctly allow `protection_barrier=1.00` for the Digital Autocall's capital-protected structure.
- **`tests/test_payoffs.py`**: File was truncated at line 234 mid-assertion; completed the `test_call_probabilities_monotone_decreasing` function.
- **`tests/test_payoffs.py`**: Fixed three `terminal_payoff` test expected values: the method returns *discounted* PV including final coupon, not undiscounted par. Updated `test_terminal_payoff_no_knockin`, `test_terminal_payoff_with_knockin`, and `test_terminal_payoff_knockin_above_protection` accordingly.
- **`tests/test_payoffs.py`**: Clarified that `european_ki` knock-in does NOT apply a protection floor — that is `soft_protection` only. The knocked-in payoff is `spot_T/S_ref * notional * discount`.
- **`tests/test_heston.py`**: Full file rewrite after truncation at `test_heston_model_call_price`. Also fixed `test_heston_implied_vol_range` to filter out sentinel `0.01` values returned by `bs_implied_vol` on solver failure before checking the ATM vol range.
- **`tests/test_pde_pricer.py`**: `test_fd_vs_mc_consistency` now uses `N_x=200, N_tau=200` (previously 120×80). At coarse resolution, FD has ~1.5% systematic bias vs MC; at 200×200, bias drops to <0.1%, and FD ($951.28) and MC ($950.50) agree within 1σ.

### Root Cause (FUSE Write Cache)
All truncations were caused by the OneDrive FUSE mount's write-buffering behavior: edits via the Windows-path Write/Edit tools did not fully flush before the session ended. Files written to the FUSE mount may be silently truncated if the buffer is not flushed. Additionally, stale `.pyc` bytecode files on the FUSE mount are read-only and cannot be deleted — Python loaded stale compiled code until source `.py` files were `touch`-ed to force mtime newer than the `.pyc`.

**Resolution protocol established**: For any file that fails to import correctly:
1. `touch <file>.py` to make source mtime > pyc mtime
2. If still failing, rewrite via `python3 << 'EOF'` heredoc in bash
3. Never rely on Write/Edit tool for critical file creation on FUSE mounts

### Test Results
**66 / 66 PASSED** — first clean run. See `test_results/20260606_results.md`.


---

## [0.2.0] — 2026-06-06

### Added — Core Pricing Engine (Steps 1–9)

**Infrastructure**
- `requirements.txt`: All dependencies (streamlit, numpy, scipy, pandas, plotly, yfinance, pytest)
- `.gitignore`: Standard Python ignores; sample_data/ CSVs intentionally committed for Streamlit Cloud
- `git_setup.ps1`: PowerShell script for Windows git initialization (bypasses FUSE locking issues)
- Directory scaffold: `app/`, `app/components/`, `app/pages/`, `scripts/`, `sample_data/`, `tests/`, `test_results/`

**Data Layer**
- `scripts/collect_snapshot.py`: Live SPX options snapshot collector via yfinance; `score_snapshot()` quality scorer (n_expiries, avg_strikes, pct_tight_spread, pct_missing_iv, quality_score)
- `scripts/generate_synthetic_data.py`: Realistic synthetic vol surface generator for offline demo; 3 snapshots generated (20260606, 20260609, 20260610)
- `sample_data/`: 3 CSV snapshots, ~1350 rows each, 23 expiries, 80+ strikes per expiry
- `app/data_loader.py`: `list_available_snapshots()`, `load_snapshot()`, `get_spot_price()`, `get_rfr()`, `get_implied_vol_matrix()`, `resolve_data_dir()`

**Product Layer**
- `app/components/securities.py`: 4 pre-configured autocallables — Phoenix (SPX, 2Y, 8% pa), Worst-Of (SPX/NDX/RUT, 3Y, 12%), Step-Down Barrier (monthly, 2Y, 10%), Digital (3Y, $50 digital coupon)
- `app/autocallable.py`: `AutoCallable` dataclass with full payoff logic — `observation_dates()`, `coupon_per_period()`, `call_barrier_at_period()`, `is_called()`, `terminal_payoff()`, `call_probabilities()`, `from_security_dict()` factory

**Pricing Engines**
- `app/pde_pricer.py`: Explicit FD on log-price grid (Paper 1, §2.2); change-of-variables to heat equation; autocall BCs at observation dates; `continuous_autocall_closedform()` for Paper 1 §2.3 validation. Validated: FD ≈ $958–970 for Phoenix.
- `app/mc_standard.py`: GBM Monte Carlo (Paper 3, Eq. 2.3); antithetic variates; path storage for animation; convergence tracking. Validated: $963 ± $1.35 at N=2000.
- `app/mc_survival.py`: One-step survival MC (Paper 3, Algorithm 1); stdlib `math.erf` for NumPy 2.x compatibility; `_p_survive()` valid at all spot levels including at-barrier. Validated: $967 ± $0.70 at N=2000, 1.93× variance reduction.

**Volatility Models**
- `app/vol_surface.py`: `VolSurface` class with bicubic spline fit (`RectBivariateSpline`); `dupire_local_vol()` via numerical diff