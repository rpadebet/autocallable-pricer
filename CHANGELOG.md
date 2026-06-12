# Changelog

All notable changes to the AutoCallable Analytics Platform are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/): patch (0.0.X) for bug fixes, minor (0.X.0) for new features, major (X.0.0) for architecture changes.

## [0.6.9] — 2026-06-12

### Changed

- **`app/vol_surface.py` — `dupire_local_vol_grid()` now clamps spline inputs to the knot domain explicitly**
  The vectorised Dupire grid called `RectBivariateSpline.ev()` without clamping
  inputs to the knot domain (moneyness 0.70–1.35, TTM 0.08–3.0), while all three
  pricers request the grid over moneyness 0.40–1.80 and t down to 0.01
  (`mc_standard.py`, `mc_survival.py`, `pde_pricer.py`, `02_Pricer.py`).
  Code review flagged this as out-of-domain cubic extrapolation feeding the
  knock-in barrier region; empirical verification showed FITPACK's `bispeu`
  in fact clamps out-of-domain coordinates to the boundary internally, so the
  old code already produced flat boundary extension — but that behaviour is
  undocumented in scipy's API and fragile across versions. Inputs are now
  clamped explicitly (verified bit-identical to the old evaluation on scipy
  1.13 across the full 4,000-point pricer stencil; 130/130 tests pass),
  matching the scalar `implied_vol()` and the SVI path's `_ev_svi()`.

## [0.6.8] — 2026-06-11

### Added

- **`app/pages/03_FDM_Visualization.py` — vol model selector on FDM page**
  Added `#### Volatility Model` radio buttons at the top of the FDM Visualization
  page, mirroring the selector on the Pricer page. Users can now choose:
  - **Flat (Black-Scholes)** — constant σ, heat-equation FD (same as before)
  - **Local Vol Surface** — Dupire Σ(S,t) from the market, with Cubic-Spline or
    SVI sub-option (SVI enabled when the surface is built on the Vol Surface page)

  The selection is page-local (session keys `fdm_vol_top` / `fdm_vol_local_sub`)
  and independent of the global sidebar vol model, so switching between Flat and
  Local Vol on this page doesn't affect the Pricer or other pages.

  The local vol path builds a `RegularGridInterpolator` over a 40×25 (moneyness×ttm)
  grid and caches it keyed by snapshot + params + dupire type — hitting SVI↔Cubic
  or snapshot changes invalidates the cache and rebuilds. FDPricer's existing
  `_price_local_vol()` method handles the rest unchanged.

## [0.6.7] — 2026-06-11

### Fixed

- **`app/heston.py` — numpy 2.4+ `np.trapz` removal silently broke Heston/Bates pricing**
  The fast Fourier pricers `_heston_call_batch_fast` and `_bates_call_batch_fast`
  integrated with `np.trapz`, which numpy deprecated in 2.0 and **removed in 2.4**.
  On numpy ≥ 2.4 the call raised `AttributeError`, which the pricers' `try/except`
  silently swallowed — falling back to flat Black-Scholes at √v0 (~8% vol) and
  returning garbage prices with no skew or jumps. This made Bates calibration
  unfittable: every candidate scored ~15 vol-pts, so the optimizer froze at its
  Phase-0 baseline (`lam=0.000, mu_J=-0.05, sig_J=0.10`). Merton was unaffected
  (it uses a closed-form BS series, no Fourier integration), which is why Merton
  fit well while Bates stayed frozen on the same snapshot.

  Fix: bind `_np_trapz` once at import to `numpy.trapezoid` (numpy ≥ 2.0) with a
  fallback to `numpy.trapz` (numpy < 2.0), and use it in all four integration
  sites. Reproduced on the deployment environment (Python 3.11 / numpy 2.4.6 /
  scipy 1.17.1); was invisible in dev (numpy 2.0.0, where `trapz` still exists).
  All 130 tests pass; results unchanged on numpy 2.0.

## [0.6.6] — 2026-06-11

### Fixed

- **`app/heston.py` — Bates jump params no longer freeze at defaults (root-cause fix)**
  Bates calibration previously returned `mu_J=-0.05, sig_J=0.10` (the seed values)
  on every snapshot, with RMSE sometimes worse than Heston. Three compounding causes,
  all fixed:
  1. **Different datasets per model.** Heston fit the smoothed VolSurface (~80 pts),
     Merton a 25-pt random raw sample, Bates a 75-pt random raw sample — so RMSEs
     were never comparable and Bates could not be guaranteed ≤ Merton. New shared
     helper `_calibration_quotes()` gives Merton and Bates an identical, deterministic
     quote set; since Bates nests Merton on the same data, the Merton-mimic seed now
     guarantees Bates fits at least as well as Merton.
  2. **Noisy wings created a ~13 vol-pt RMSE floor.** The old moneyness ∈ [0.80,1.20]
     and ttm ≥ 0.10 filters pulled in noisy deep-OTM and ultra-short quotes that
     neither jumps nor stochastic vol could fit, so the optimizer abandoned jumps.
     Tightened to moneyness ∈ [0.85,1.15], ttm ∈ [0.15,2.0]. RMSEs dropped from
     ~7–24 vol-pts to ~1 vol-pt and jump params now carry real signal.
  3. **0.25 None-penalty cliff.** When `bs_implied_vol` failed to invert a far-OTM
     price the objective added a fixed 0.25 penalty, creating a rounding-sensitive
     RMSE cliff (one bad point shifted reported RMSE by ~5 vol-pts). Both calibrators
     now skip un-invertible quotes (with a ≥50%-valid guard against degenerate fits).

- **`app/heston.py` — Bates `gamma` floor lowered 0.10 → 0.01**
  The v0.6.5 floor of 0.10 prevented Bates from reducing to near-constant vol
  (gamma → 0), i.e. blocked it from nesting Merton. For jump-driven SPX smiles the
  best Bates fit IS the Merton-like constant-vol + jumps solution; the floor forced
  a worse stochastic-vol fit and the optimizer dropped the jumps. (Heston's gamma
  floor is unchanged — its theta cap already fixed the earlier degeneracy.)

### Verified
  Bates RMSE ≤ Merton RMSE and jump params vary meaningfully across snapshots on
  test snapshots 20260610_0945 and 20260608_0945. All 130 tests pass. NOTE: on SPX
  data Bates `gamma` collapses toward the floor — the smile is fully jump-explained,
  so the stochastic-vol component contributes little. This is the honest best fit.
  Calibration covers moneyness [0.85,1.15] only; the 70–80% autocallable barrier
  region is extrapolated by the simulated dynamics, not directly fit.

## [0.6.5] — 2026-06-11

### Fixed

- **`app/heston.py` — Heston calibration bounds tightened**
  `theta` upper bound: `0.5 → 0.20` (caps σ∞ at 44.7%; the old bound allowed 70% long-run
  vol, producing a degenerate near-GBM solution on noisy Market Open snapshots).
  `gamma` lower bound: `0.05 → 0.10` (prevents near-zero vol-of-vol which makes Heston
  degenerate to Black-Scholes). Added a 5th starting point at `(v0=0.05, κ=5, θ=0.05,
  γ=0.60, ρ=-0.65)` to cover the fast mean-reversion regime.

- **`app/heston.py` — Bates calibration data and bounds fixed**
  Sample size increased from 25 → 75 quotes, making the Bates RMSE roughly comparable
  to Heston (which uses ~80 quotes via VolSurface). `theta` and `gamma` bounds matched
  to the new Heston bounds so Bates cannot produce solutions that Heston would reject.
  Added a 4th generic starting point `(v0=0.04, κ=2, θ=0.04, γ=0.40, ρ=-0.70, λ=0.30)`
  independent of the Heston warm-start, so Bates can escape degenerate Heston solutions.

## [0.6.4] — 2026-06-11

### Fixed

- **`app/heston.py` — Bates calibration warm-start fix**
  `calibrate_bates` was starting at `lam=0.5` with Heston diffusion params calibrated
  for `lam=0`. This made the initial point *worse* than plain Heston (visible as 5 of 10
  snapshots having Bates RMSE > Heston RMSE). Changed to start at `lam=0.02` (near-zero
  so Bates ≈ Heston at the initial guess), then try two more restarts at `lam=0.3` and
  `lam=1.0`. Each restart runs its own L-BFGS-B pass; the best result across all three
  is kept. `maxiter` raised from 200 → 300, `ftol` tightened from 1e-6 → 1e-7.

### Added

- **`scripts/precalibrate.py` — Merton calibration added as step 2 of 3**
  The pre-calibration script now runs Heston → Merton → Bates for each snapshot.
  Results saved to cache under `"merton"` key. Skip logic updated to check all three.

- **`app/components/sidebar.py` — Merton auto-loaded from cache on snapshot change**
  `merton_cal` is now populated from the calibration cache alongside Heston and Bates
  when a new snapshot is selected, so the Calibrate Models tab on Vol Surface shows
  pre-loaded Merton params immediately.

## [0.6.3] — 2026-06-11

### Fixed

- **`app/pages/02_Pricer.py` — Calibration banner now shows correct model params**
  When **Bates** is selected, the banner now shows all 8 Bates params (v₀, κ, θ, γ, ρ,
  λ, μ_J, σ_J) with RMSE. If Bates is not yet calibrated but Heston is, a warning
  banner shows the Heston diffusion params + default jump params. Previously the banner
  always said "Calibrated Heston params in use" regardless of which model was selected,
  confusing users who switched to Bates. Also added θ and γ to the Heston banner.

- **`app/pages/01_Vol_Surface.py` — Calibrate tab now shows all 3 models' params**
  Replaced the Heston-only parameter block with a 3-column layout showing Heston, Merton,
  and Bates params side-by-side immediately after calibration completes. Each column shows
  the full parameter table (v₀, σ₀, κ, θ, γ, ρ, jump params, RMSE, Feller). Removed the
  redundant collapsed "Full Calibrated Parameters" expander that duplicated the same data.

### Added

- **`scripts/precalibrate.py` — Offline pre-calibration script for all snapshots**
  New script that runs Heston + Bates calibration on every snapshot in `sample_data/`
  and saves results to `sample_data/calibrations_cache.json`. Supports `--force` flag
  and incremental saves (safe to interrupt and resume). Eliminates interactive wait time
  on the Pricer and Vol Surface pages when pre-calibrated params are available.

- **`app/data_loader.py` — `load_calibration_cache()` function**
  Loads `sample_data/calibrations_cache.json` and returns the dict keyed by snapshot key.
  Returns `{}` gracefully if the file is missing or malformed.

- **`app/components/sidebar.py` — Snapshot auto-populates S0 and r**
  When a new snapshot date is selected, the sidebar now automatically sets S0 and r from
  the snapshot's `spot` and `rfr` columns. Subsequent re-renders with the same snapshot
  preserve any manual overrides. Implemented via a `_snap_key_for_rates` tracker key.

- **`app/components/sidebar.py` — Pre-calibrated params auto-load on snapshot change**
  When a new snapshot is selected, the sidebar auto-loads `heston_cal` and `bates_cal`
  from the calibration cache (if available), so calibrated params are instantly available
  without waiting for a new calibration run. Tracker key `_snap_key_for_cals` prevents
  overwriting manually re-calibrated values on re-render.

- **`app/components/sidebar.py` — `use_calibrated_bates` toggle**
  Added a "Use calibrated Bates values" toggle (default on) in the Bates Jump Parameters
  expander, matching the existing Heston toggle. When on and `bates_cal` is loaded,
  jump params (λ, μ_J, σ_J) are injected from the cache and sliders are disabled.

- **`app/components/sidebar.py` — Active vol model indicator in section ③**
  Added a caption showing the currently selected vol model (e.g. "🔴 Vol model: Bates")
  so users can confirm their Pricer page selection is reflected in the sidebar params.

## [0.6.2] — 2026-06-10

### Changed

- **`app/pages/02_Pricer.py` — Vol model selector migrated from sidebar to Pricer page**
  Replaced the flat selectbox in sidebar section ③ with a three-tier hierarchical radio
  button block at the top of the Pricer page. Top-level choices: **Flat (Black-Scholes)**,
  **Local Vol Surface**, **Stochastic Vol**. Sub-selections appear contextually:
  - Local Vol → Cubic-Spline or SVI (smooth) Dupire surface (SVI greyed-out unless built)
  - Stochastic → Heston or Bates (Heston + Jumps)
  Session keys `pricer_vol_top`, `pricer_vol_local_sub`, `pricer_vol_stoch_sub` are read
  by `render_sidebar()` so `params["vol_model"]` stays correct for all other pages.
  The sub-type selector is also wired into `_param_fingerprint` so switching Cubic↔SVI
  triggers the "settings changed" banner.

- **`app/pages/01_Vol_Surface.py` — Tab order: Dupire moved to Tab 2, Calibrate to Tab 3**
  User-visible tab order is now: Build Vol Surface (1) → Dupire Vol Surface (2) → Calibrate
  Models (3). The Dupire surface is the natural follow-on from building the surface; model
  calibration is the deeper analytical step. Implemented by reassigning `tab2`/`tab3` aliases
  after `st.tabs()` so all existing `with tab2:` / `with tab3:` code blocks are unchanged.

- **`app/components/sidebar.py` — Section ③ vol model widget removed**
  The `st.selectbox` for vol model selection has been removed from the sidebar. The sidebar
  now derives `vol_model` / `vol_model_label` from the three `pricer_vol_*` session state
  keys (set by the Pricer page). Factory defaults for all three keys are registered in
  `_ensure_sidebar_defaults`; all three are persisted in `_sidebar_backup` so the selection
  survives page navigations.

## [0.6.1] — 2026-06-10

### Fixed

- **`app/vol_surface.py` — Dupire local vol `dT` noise amplification**
  Changed default `dT` from `0.01` (3.6 days) to `0.04` (~15 days) in all three Dupire
  methods (`dupire_local_vol`, `dupire_local_vol_grid`, `svi_dupire_local_vol_grid`).
  The time derivative `dCdT = (C(T+dT) - C(T)) / dT` amplified IV surface noise by
  `1/dT = 100×` at the old value. The new value reduces this amplification 4× while
  keeping the derivative accurate. Primary fix for jagged Dupire surfaces.

- **`app/vol_surface.py` — SVI bivariate spline cross-tenor smoothing**
  Changed `s=0.005` → `s=0.02` for the `RectBivariateSpline` fitted to the SVI-evaluated
  IV grid in `build_svi_surface()`. SVI per-slice already handles strike-direction
  smoothness; the higher `s` prevents residual slice-to-slice parameter jumps from
  propagating into `d²C/dK²` via the bivariate spline.

- **`app/vol_surface.py` — SVI per-slice cross-tenor consistency (warm-start)**
  `_fit_svi_slice()` gains an optional `prev_params` argument. `build_svi_surface()`
  now passes the previous slice's fitted parameters as an extra optimizer starting point,
  nudging adjacent expiry slices toward consistent parameter values rather than landing
  on distant local minima independently.

### Changed

- **`app/pages/01_Vol_Surface.py` — SVI auto-built with one click**
  "Build Vol Surface" now runs two steps: (1) bicubic spline + cubic Dupire, then
  (2) SVI + SVI Dupire. All three surfaces are ready after a single button click —
  no separate "Build SVI Surface" step required. The `VolSurface` object stored in
  session state retains `svi_ready=True` across page navigation (a `_surf_key` cache
  check prevents the object from being recreated on every page visit, which previously
  wiped the SVI state).

- **`app/pages/01_Vol_Surface.py` — Tab 3 SVI button demoted to optional rebuild**
  The primary "Build SVI Surface" button is removed. An "🔄 Rebuild SVI surface"
  expander is available for power users who need to force a re-fit.

- **`app/pages/01_Vol_Surface.py` — timing text updated**
  "Build Vol Surface ~3 s" → "~15 s (spline + SVI + all Dupire surfaces auto-built in
  one click)".

## [0.6.0] — 2026-06-10

### Added

- **`app/vol_surface.py` — SVI (Stochastic Volatility Inspired) model**
  Added `_svi_w()` and `_fit_svi_slice()` module-level helpers implementing the Gatheral (2004)
  parametrization `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`. Per-slice fitting uses
  L-BFGS-B with 4 diverse starting points to handle the non-convex objective. Produces a smoother
  Dupire surface than the cubic-spline approach by denoising market IV quotes per expiry before
  computing d2C/dK2.

- **`app/vol_surface.py` — `VolSurface.build_svi_surface()`**
  Fits SVI per expiry slice (>=6 points), re-evaluates on the standard moneyness knot grid, and
  fits a smooth `RectBivariateSpline` (s=0.0005). Returns `{n_slices_fitted, slice_rmse, svi_ready}`.

- **`app/vol_surface.py` — `VolSurface.svi_implied_vol()`, `svi_surface_grid()`**
  SVI-smoothed IV evaluation and 3D grid builder. Falls back to cubic-spline surface if SVI
  not yet built.

- **`app/vol_surface.py` — `VolSurface.svi_dupire_local_vol_grid()`, `svi_dupire_surface_grid()`**
  Vectorised SVI-based Dupire local vol grid. Falls back to cubic-spline Dupire gracefully.

- **`app/vol_surface.py` — CSV export helpers**
  `to_csv_implied_vol()` and `to_csv_dupire(method=["cubic"|"svi"])` produce download-ready
  CSV strings for `st.download_button()`. Dupire CSVs include a label comment header.

- **`app/pages/01_Vol_Surface.py` — 3-tab restructure**
  New tab layout: (1) "Build Vol Surface" -- IV surface + ATM term structure + CSV download;
  (2) "Calibrate Models" -- merged Heston + Merton + Bates in one run;
  (3) "Dupire Vol Surface" -- cubic-spline section + SVI section with per-slice RMSE,
  3D surfaces, ATM 3-way comparison, and CSV downloads.

- **`app/pages/02_Pricer.py` — SVI Dupire surface selector**
  When Local Vol model is active and SVI has been built, a radio button lets the user choose
  between cubic-spline and SVI Dupire surfaces for pricing.

- **`tests/test_vol_surface.py` — 36 new tests (110 total)**
  New test classes: TestSVIFormula (5), TestSVIFit (5), TestBuildSVISurface (4),
  TestSVISurfaceEval (5), TestSVIDupireGrid (5), TestDupireSmoothness (2),
  TestCSVExport (8), TestFallback (2). All 110 tests pass.

### Changed

- **`app/pages/01_Vol_Surface.py`** — "Show Dupire vol" toggle replaced with dedicated
  Dupire tab; "Calibrate Heston" and "Calibrate All Models" buttons merged into single
  "Calibrate All Models (Heston + Merton + Bates)" button.

## [0.5.16] — 2026-06-08

### Reverted

- **`app/mc_survival.py` — Survival MC sub-stepping reverted to paper methodology**
  A prior commit added `n_steps_per_year` sub-stepping inside the survival MC loop for local vol
  and Bates models. This deviated from the paper's one-step-per-observation-date algorithm and
  introduced complexity without a clear accuracy benefit for the demo use case. Reverted to the
  clean Paper 3 formulation: one survival step per observation date, local vol queried at the
  observation midpoint.

### Added

- **`app/pages/02_Pricer.py` — Survival MC methodology limitation note**
  When the active vol model is Local Vol or Bates, the Survival MC result card now displays an
  informational note explaining that the survival probability formula conditions on a
  single effective sigma (local vol at midpoint or Heston √v_t), and that this is an
  approximation relative to the exact continuous-barrier survival probability. Prevents
  the user from mistakenly treating the Survival MC price as exact under these models.

### Fixed

- **`app/pages/02_Pricer.py` — NameError: `heston_params` at module level**
  (`app/pages/02_Pricer.py`)
  `heston_params` and `jump_params` were referenced at module level before being extracted from
  the `params` dict returned by `render_sidebar()`. When Streamlit re-ran the script on page
  navigation, the reference fired before the sidebar had been rendered, causing a `NameError`.
  Fix: moved extraction of `heston_params` and `jump_params` to immediately after the
  `render_sidebar()` call rather than at module top-level.

- **`app/pages/03_FDM_Visualization.py` — FDM Visualization now honors vol model selection**
  (`app/pages/03_FDM_Visualization.py`)
  The FDM Visualization page passed a hardcoded `vol_model="flat"` to `FDPricer` regardless of
  the sidebar selection. When the user switched to Local Vol in the sidebar, the heatmap and
  time-slice animation continued to show flat-vol results with no indication of the mismatch.
  Fix: reads `vol_model` and `vol_surface` from sidebar params and passes them through to
  `FDPricer`. Also added `st.rerun()` on page load when the vol model or snapshot key has
  changed since the last render, so the visualization refreshes automatically on navigation.

---

## [0.5.15] — 2026-06-08

### Fixed

- **DS-1 — Call probability table now consistent with selected vol model**
  (`app/pde_pricer.py`, `app/pages/02_Pricer.py`)

  Previously `call_probabilities()` always received `self.sigma` (flat vol), even when
  the user selected Heston or Bates on the sidebar. The price used the chosen vol model
  but the call probability table used a different vol — a silent inconsistency.

  Fix: `FDPricer.__init__` now sets `self.call_prob_sigma = self.sigma` by default.
  Page 2 overrides it to `√v₀` immediately after FDPricer construction when
  `vol_model in ("heston", "bates")` and `heston_params` is available. Both `price()`
  call sites in `pde_pricer.py` (flat vol path line ~525, local vol path line ~715)
  now pass `self.call_prob_sigma` instead of `self.sigma`.

  The call probability caption on Page 2 is now dynamic: for Heston/Bates it reads
  *"σ_eff = 22.4% (√v₀ from Heston calibration). Exact model-consistent probabilities
  would require simulation."* For flat/local vol the caption is unchanged.

  `call_prob_sigma` is a public attribute — callers can override it further
  (e.g. use `√θ` for long-maturity products) without changing FDPricer's interface.

## [0.5.14] — 2026-06-08

### Added

- **`tests/test_pde_pricer.py` — GAP-7 (Thomas algorithm edge cases)**: Three new tests covering
  `thomas_solve` at n=1 (single equation), n=2 (2×2 hand-verified system), and n=500 (comparison
  against `numpy.linalg.solve` for a diagonally dominant random matrix, tolerance 1e-8).

- **`tests/test_pde_pricer.py` — GAP-9 (tau/t roundtrip)**: Verifies `_tau_to_t(_t_to_tau(t)) == t`
  for t ∈ {0.0, 0.25, 0.5, 1.0, T}, confirming the coordinate transform is lossless.

- **`tests/test_pde_pricer.py` — GAP-1 (local vol FDM)**: Two tests: (1) constant local vol surface
  prices within $15 of flat vol FDM, (2) BUG #1 regression — maturity BC must fire, confirmed by
  gap < $10 (before the `<` → `<=` fix the gap was ~$30+).

- **`tests/test_pde_pricer.py` — GAP-4 (FDM coupon gating)**: Tests `_apply_autocall_bc` with
  `call_barrier=0.90, coupon_barrier=1.00`. Asserts spot in [90%, 100%) receives redemption only,
  spot ≥ 100% receives redemption + coupon. Directly validates the `np.where(S >= coupon_barrier)`
  logic in `_apply_autocall_bc`.

- **`tests/test_payoffs.py` — GAP-3 (soft_protection)**: Three tests covering the Digital Autocall's
  `soft_protection` terminal payoff: (1) far below floor → floor * notional, (2) above floor but
  below call_barrier → spot/S_ref * notional, (3) above call_barrier → base + coupon_per_period.
  All include discount factor verification.

- **`tests/test_payoffs.py` — GAP-8 (step-down call probability)**: Asserts that the Step-Down
  Barrier security's total call probability ≥ Phoenix (fixed 100% barrier) total call probability.
  Validates the economic intuition that a declining barrier makes autocall more likely.

- **`tests/test_mc_pricers.py` — GAP-2 (Bates model)**: Three tests: (1) Bates with λ=0.5 returns
  finite positive price, (2) Bates with λ=1.0 produces meaningfully different price from Heston-only
  (> $0.50 difference — jumps have effect), (3) Bates with λ=0 equals Heston within 4 SE.

- **`tests/test_mc_pricers.py` — GAP-6 (Merton series convergence)**: Asserts that
  `_merton_call_bs_series` with `n_terms=20` matches `n_terms=10` to within $0.01 per option across
  5 strikes, confirming the default truncation is sufficient for λ*T ≤ 2.

### Summary

Test count: 59 → 74 (+15 new tests). All 74 pass; `test_heston.py` still skipped (FUSE truncation).

## [0.5.13] — 2026-06-08

### Fixed

- **`app/pde_pricer.py` — BUG #1 (CRITICAL)**: Local vol FDM never applied the maturity
  observation boundary condition. The backward sweep condition was `t_current <= t_obs < t_axis_phys[step]`
  (strict `<`). At `step=0`, `t_axis_phys[0] = T`, so for any observation at maturity the condition
  evaluated `T < T` = False. Since the last observation date is always `maturity_years` (guaranteed
  by `_observation_dates_from_params`), the maturity autocall BC was silently skipped on every
  local vol run. Fix: change strict `<` to `<=` so the condition fires when the obs date falls
  exactly on the current grid point.
  _File_: `app/pde_pricer.py` line 648. One character change (`<` → `<=`).

- **`app/pde_pricer.py` — BUG #3 (LOW)**: `continuous_autocall_closedform()` returned an
  undiscounted payout when `call_barrier * S0 <= S0` (spot already at or above barrier). Was:
  `notional * (1 + coupon_pa * T)`. Fixed to discount using `expected_call_time = T/2`
  (same approximation used for the called-path PV component), consistent with the rest of the
  closed-form formula.

- **`app/pde_pricer.py` — BUG #4 (LOW)**: `np.clip(NaN, 0, 1500)` behaviour is numpy-version-
  dependent (some versions return NaN, some return 0), silently masking FD instability. Added
  explicit `if not np.isfinite(price): raise RuntimeError(...)` before the clip in both the
  flat vol `price()` and local vol `_price_local_vol()` paths. The error message names the
  likely cause (Courant violation) and the fix (increase N_tau).

- **`app/mc_survival.py` — EC-4 (LOW)**: `_p_survive()` and `_p_survive_vec()` divided by
  `sigma_t * sqrt(dt)` without guarding against near-zero sigma (theoretically possible at
  local vol grid boundaries despite clamping). Added `if sigma_t * sqrt(dt) < 1e-10: return 0.5`
  guard in the scalar path, and `np.where(denom < 1e-10, 0.0, z)` pattern in the vectorised path.

- **`app/pde_pricer.py` — HYG-1**: Duplicate `from scipy.stats import norm` in
  `continuous_autocall_closedform()` (lines 768–769). Removed second import.

- **`app/heston.py` — HYG-2**: Duplicate `from app.vol_surface import bs_implied_vol` in
  `calibrate_bates()` (lines 1240, 1242). Removed second import.

- **`app/components/sidebar.py` / `app/pde_pricer.py` / `app/mc_survival.py`** — OneDrive FUSE
  sync truncates files on the Streamlit Cloud worktree mid-write (truncation at arbitrary byte
  offsets — same pattern as v0.5.10). Files restored from git; subsequent writes use atomic
  `os.replace(tmp, target)` to reduce exposure window.

### Not fixed (assessed, deferred)

- **BUG #2** — Survival MC `call_pv` adds coupon unconditionally without checking
  `coupon_barrier`. Deferred: the spot at barrier crossing is analytically integrated out in
  survival MC; proper fix requires computing conditional P(spot ∈ [call, coupon_barrier))
  at crossing time — non-trivial algorithm change. Zero current impact (all 4 configured
  securities have `call_barrier >= coupon_barrier`).

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
  - New `_price_local_vol()` method: direct log-price PDE `∂V/∂τ = (�