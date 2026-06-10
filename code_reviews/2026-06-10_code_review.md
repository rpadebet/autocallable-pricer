# Code Review — AutoCallable Analytics Platform
**Date:** 2026-06-10 · **Scope:** whole app (all `app/` modules, pages, components, tests) · **Version reviewed:** v0.6.0 (working tree)
**Reviewer:** Claude Code (high-effort review; every core module and page read line-by-line; findings verified against the code)

---

## Executive Summary

The codebase is well-documented and the numerical machinery (Heston CF via Eq. 23, SVI fitting, Dupire grids, Thomas/CN solver, survival MC algorithm) is largely correct. The most important problems are **not in any single pricer** — they are **contract inconsistencies between the three pricers** and between the pricers and the term sheets shown to the user. Because the test suite compares pricers to each other with loose tolerances (±$5 + 3σ), these divergences pass 110/110 tests while producing systematically different product definitions.

**10 primary findings** (4 pricing-correctness, 3 page-logic, 3 product-definition), plus 12 secondary issues. No syntax errors; all files parse.

---

## Primary Findings (ranked by severity)

### 1. Phoenix/step-down coupons before the call date are never paid — all three pricers
**Files:** `app/mc_standard.py:404–431`, `app/mc_survival.py:341–343, 491–493`, `app/pde_pricer.py:340–345`
The Phoenix term sheet (`securities.py:43–48`) says: *"Pays an 8% p.a. conditional coupon at each quarterly observation if spot is above the coupon barrier (75%)"*, and `AutoCallable.coupon_is_paid()` documents the coupon as independent of the autocall. But every pricer pays a coupon **only at the call date** (or maturity). A path sitting at 90% of S0 for four quarters then calling at quarter 5 earns one $20 coupon instead of five.
**Impact:** Phoenix and Step-Down are materially underpriced relative to their displayed term sheets (order of $20–60 per $1,000 notional). Invisible in tests because all three pricers share the omission.
**Fix:** in `_price_paths` accumulate `coupon * exp(-r*t_i)` for every active path with `S_i >= coupon_barrier` at each observation (and the analogous accrual term `L_j * coupon * P(coupon | survive)` in survival MC; for the PDE this needs a coupon source term / dividend-style adjustment at each obs date). Alternatively, if the *intent* is coupon-at-call-only, fix the term sheets and docstrings.

### 2. Knock-in monitoring: Standard MC disagrees with Survival MC and PDE
**Files:** `app/mc_standard.py:408` vs `app/mc_survival.py:365, 514` and `app/pde_pricer.py:304, 644`
Standard MC marks knock-in if spot is below the protection barrier at **any observation date** (`knocked_in |= (S_i < ki_barrier)`). Survival MC and the PDE terminal condition check **only the final spot**. `securities.py` itself is contradictory: `protection_type: "european_ki"` is annotated *"observed at maturity only"* while the Phoenix description says *"if spot fell below 75% at any point"*.
**Impact:** the three pricers price different products. A path that dips to 70% mid-life and recovers to 90% pays ~$900 in Standard MC and $1,000 in Survival MC/PDE. The `test_mc_methods_agree` tolerance (3σ + $5) absorbs the gap.
**Fix:** pick one convention (true European KI = maturity-only is what the field name says), implement it identically in all three pricers, and tighten the cross-pricer test tolerance so a regression would actually fail.

### 3. Survival MC omits the maturity coupon for surviving paths
**Files:** `app/mc_survival.py:362–371, 511–521` vs `app/mc_standard.py:426–430`
For paths ending between the coupon barrier (75%) and call barrier (100%) without knock-in, Standard MC pays `notional + coupon`; Survival MC pays `notional` only. A systematic ~$3–6 downward bias in Survival MC for the Phoenix, again hidden by test tolerance.
**Fix:** add `+ coupon_per_period()` to the survival terminal payoff when `s >= coupon_barrier * S_ref` (both the scalar `_price_one_path` and the vectorised `price()`).

### 4. Sign error in the closed-form continuous autocall first-passage probability
**File:** `app/pde_pricer.py:798–806`
For barrier B > S0 with drift ν, the no-crossing probability is `Φ((b−νT)/(σ√T)) − e^{2νb/σ²}·Φ((−b−νT)/(σ√T))` with `b = log(B/S0)`. The code computes `p_no_cross = Φ(d1) − e^{2νb/σ²}·Φ(d2)` with `d1 = (−b+νT)/(σ√T)` — i.e. `Φ(d1)` where `Φ(−d1)` is required. The function''s own docstring (line 761) states the *crossing* probability as `N(d1) + exp(·)N(d2)`, which the code then contradicts. Separately, `expected_call_time = T/2` and the coupon-stream heuristic make this an ad-hoc approximation, not the Paper 1 §2.3 closed form the spec calls for.
**Impact:** wrong call/no-call split whenever ν ≠ b/T; the function is also benchmarked in the spec at $99.54 ± $0.10, a test which does not exist (only finite/positive is asserted).
**Fix:** replace with the correct reflection-principle formula (use Φ(−d1)) and, ideally, the paper''s actual closed form with proper E[τ | τ<T].

### 5. FDM page "Greeks" tab differentiates the **maturity** slice, not t=0
**File:** `app/pages/03_FDM_Visualization.py:332–339`
`V_grid` columns are ordered by τ ascending, i.e. **t descending**: column 0 is maturity, the last column is t=0. Tab 2''s own slider help text says *"Index 0 = maturity… Max index = t=0"*. Tab 3 nevertheless computes Delta/Gamma from `V_grid[:, 0]` and Theta from `(V_grid[:,1] − V_grid[:,0]) / (t_axis[1] − t_axis[0])` — so the page displays the Greeks of the terminal payoff (step-function Delta), and `dt_step` is negative (t_axis decreases), flipping Theta''s sign on top of the wrong location.
**Fix:** use `V_grid[:, -1]` (t=0) and `dt_step = t_axis[-2] − t_axis[-1]` with `(V_grid[:,-2] − V_grid[:,-1]) / dt_step`.

### 6. Product Builder displays the per-period coupon ×12
**File:** `app/pages/06_Product_Builder.py:357–363, 446–450`
`period_coupon = coupon_pa / (12 / FREQ_TO_MONTHS[freq]) * 12 * notional`. For 8% p.a. quarterly: `0.08/4*12 = 0.24` → **$240** shown as the per-period coupon instead of $20. Affects the autocall-payoff reference line, the "Max payoff (autocall)" metric, and every row of the observation schedule table. `AutoCallable.coupon_per_period()` already computes this correctly — the page should call it instead of re-deriving it.

### 7. Product Builder step-down barriers are off-by-one (and semantically different) vs the pricer
**File:** `app/pages/06_Product_Builder.py:185–202, 436–440` vs `app/autocallable.py:242–251`
The builder saves `(obs_idx, barrier)` with **1-based** indices and its schedule table treats the barrier as applying **only at** that observation (`sd_map.get(i+1, call_barrier)`). `call_barrier_at_period()` treats entries as **0-based** and applies each barrier **from that period onward**. A custom step-down product is therefore priced with a schedule shifted by one observation and with different fill-in semantics than the preview table shows the user.
**Fix:** store 0-based indices (or subtract 1 on save) and render the preview table with the same "from period N onward" rule via `call_barrier_at_period()`.

### 8. Step-Down security''s barrier schedule contradicts its own term sheet
**File:** `app/components/securities.py:124–149`
Description: *"Starts at 100%, drops to 95% after year 1, drops again to 90% after year 1.5"*. With monthly observations over 2 years, that is periods 12 and 18. The code has `(5, 0.95), (9, 0.90)` — i.e. months 6 and 10 — and the inline comments ("months 5–8 (after 4 quarters)") don''t match either reading. The demo will show a call-probability term structure inconsistent with the displayed terms.
**Fix:** `[(12, 0.95), (18, 0.90)]` (0-based: barriers step at the 13th and 19th monthly obs), or rewrite the description.

### 9. Heston curve silently missing from the model-comparison smile chart
**File:** `app/pages/01_Vol_Surface.py:474`
`heston_for_overlay.european_call(S0, K, T)` — `HestonModel` has no `european_call` method (it''s `call_price(K, T)`; verified by grep: the only occurrence in the repo is this call site). The per-point `try/except Exception` swallows the `AttributeError`, so the "All Models" comparison chart simply renders without a Heston line and no error is shown.
**Fix:** `price = heston_for_overlay.call_price(K, T)`.

### 10. Call probabilities use an independence approximation but are presented as the Paper 1 analytical result
**File:** `app/autocallable.py:435–454`; surfaced in `02_Pricer.py` Tab 4 and `05_Scenarios.py` Tab 3
`prob_survived *= (1 − p_above)` multiplies **marginal** probabilities, ignoring the strong positive autocorrelation of overlapping GBM increments (corr = √(t_j/t_i)). Conditional on surviving (spot below barrier at earlier dates), the true call probability at later dates is much lower than the marginal — the docstring admits the approximation, but the UI captions say *"From the analytical formula in Paper 1 §2.2"* and *"analytical — no MC noise"*. The CLAUDE.md test plan (reproduce Table 1: p1=0.3767, p2=0.1435, … ±0.001) would fail; that test is absent from the suite.
**Fix:** either implement the joint-lognormal (multivariate normal CDF / sequential conditioning) computation per the paper, or estimate call probabilities from the MC paths already simulated, and relabel the UI as an approximation until then.

---

## Secondary Findings

| # | Location | Issue |
|---|----------|-------|
| S1 | `mc_standard.py:253–255, 520–529` | Antithetic doubles the path count (request 10,000 → 20,000 reported) and the SE treats the 2N correlated samples as independent. Correct treatment averages each antithetic pair (N independent samples). Current CI is invalid and the antithetic-vs-survival variance comparison is biased. |
| S2 | `components/sidebar.py:527` + `02_Pricer.py` | The "Time Steps (M)" slider (`n_steps`) is collected into `params` but never passed to any pricer (`n_steps_per_year` always defaults to 52). Dead control. |
| S3 | `02_Pricer.py:475–479` | Displayed Courant number uses `dx = (abs(x_min)+5)/N_x`; the actual grid is `[x_min, 2.0]` with `dx = (2−x_min)/(N_x−1)`, and FDPricer auto-corrects N_tau anyway. The 🟢/🔴 indicator can show the wrong state. |
| S4 | `04_Greeks.py:236` | Tab 1 redisplay condition checks `"greeks_delta" in st.session_state.get("cache", {})` — a key never written. Delta-stability results vanish on any rerun/tab-switch, unlike Tabs 2–3 which check the correct keys. |
| S5 | `01_Vol_Surface.py:83 vs 333` | Banner reads `vol_surface_last_run_fp` but calibration writes `vol_surf_last_run_fp` — the "settings changed" banner on this page can never function. |
| S6 | `mc_standard.py:479–495` | With antithetic on, convergence checkpoints stop at `base_n` while the reported price uses `2·base_n` paths — the convergence chart''s endpoint doesn''t match the headline price/SE. The vol-aware branch (510–518) can also append a duplicate final point. |
| S7 | `pde_pricer.py:486–503` | `call_counts` is computed ((S_axis ≥ barrier).mean() — a fraction of grid nodes, not a probability) and never used. Dead/misleading code. |
| S8 | `pde_pricer.py:784–785` | Duplicate `from scipy.stats import norm` import. |
| S9 | `vol_surface.py:509 vs 595` | Scalar `dupire_local_vol` clamps to [0.01, 2.0]; the grid version clamps to [0.05, 1.0]. The slow fallback path in `mc_survival._get_sigma_local` therefore disagrees with the fast path. |
| S10 | `mc_standard.py:217`, `mc_survival.py:191` | Local-vol moneyness is `S_t / self.S0` where `S0` may be a bumped spot (`spot_override`). Delta bumps shift the entire local-vol surface (sticky-moneyness w.r.t. the bump), which slightly contaminates the very Greeks the page demonstrates. Anchor to the surface spot (`vol_surface.S0`) instead. |
| S11 | `heston.py:320–331, 1023–1032` | `integrand_P1` recomputes `CF(−1j)` (a constant) inside the integrand for every quadrature point — hundreds of redundant CF evaluations per price. Hoist it out. |
| S12 | `heston.py:563–566` | When the Feller penalty is active in the best objective value, `rmse` (and the displayed `rmse_vol_pts`) includes the penalty term, overstating the reported fit error. |

---

## Test Coverage Notes

- The canonical benchmarks from CLAUDE.md §7 are **not implemented**: Paper 1 Table 1 call probabilities (±0.001), FD price $98.39 (±$0.10), closed-form $99.54 (±$0.10). The current suite asserts finiteness, positivity, stability, and loose cross-method agreement — which is precisely why findings 1–4 and 10 pass 110/110.
- Recommended new tests once contracts are fixed: (a) hand-computed single-period Phoenix price including an intermediate coupon; (b) a path that breaches KI mid-life and recovers — assert all three pricers agree on its payoff; (c) survival-vs-standard agreement with tolerance 3σ **without** the +$5 fudge; (d) Paper 1 Table 1 probabilities against the corrected `call_probabilities`.

## Not Bugs (reviewed and deliberately excluded)

- Worst-Of basket priced as single-asset: clearly disclosed in the UI and in TODOs.
- FDM falling back to flat vol under Heston/Bates: disclosed with an explanatory banner.
- Survival MC approximations under local vol/Bates: disclosed with a methodology note on the Pricer page.
- Heston CF: correctly uses the Eq. 23 (Albrecher) form in both scalar and batch implementations; Bates/Merton drift corrections check out.
- Sidebar session-state persistence design (defaults + backup dict): unusual but sound.
