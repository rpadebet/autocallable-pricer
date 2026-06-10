# Code Review — AutoCallable Analytics Platform

**Date**: 2026-06-08  
**Version Reviewed**: v0.5.12  
**Reviewer**: OpenCode (deepseek-v4-pro)  
**Artifacts Reviewed**: Full codebase against Deng-Mallett-McCann (2011), Haugh (2013), Alm et al. (2013), and CLAUDE.md Technical Spec.

---

## 🚨 LIVE CRASH: `sidebar.py` truncated in committed version (Streamlit Cloud down)

**Status**: ACTIVE PRODUCTION BUG — all pages fail to load.

**Root cause**: The committed version of `app/components/sidebar.py` is **truncated at byte 588** by OneDrive FUSE sync. The file ends mid-word:

```python
        "securit
```

The missing content (lines 589–649) includes:
- The rest of the `params` dict (`security_name`, `security_params`, `autocallable`, `vol_model`, `heston_params`, `jump_params`, `n_paths`, `N_x`, etc.)
- The `_sidebar_backup` belt-and-suspenders persistence logic
- The `return params` statement

When Python imports this module, it hits `SyntaxError: unexpected EOF while parsing`. This crashes `Home.py:58` (`from app.components.sidebar import render_sidebar`) and every other page that imports sidebar.

**Evidence**:
```bash
$ git diff app/components/sidebar.py
# Shows committed version ends at line 588 with `"securit`
# Local file has full 649 lines with complete dict + return statement
```

**This is the same OneDrive FUSE truncation bug** that previously hit:
- `app/heston.py` (SyntaxError on import — skipped in test suite)
- `app/vol_surface.py` (duped local vol grid — restored in v0.5.10)
- `app/pde_pricer.py` (truncated `continuous_autocall_closedform()` — restored in v0.5.10)

**Fix**:
```bash
git add app/components/sidebar.py
git commit -m "fix: restore truncated sidebar.py (OneDrive FUSE sync truncation)"
git push origin main
```
Streamlit Cloud auto-deploys on push. No code change needed — the local file is complete and correct.

**Prevention**: The `.streamlit/config.toml` file stops Streamlit's dev-mode watcher from triggering on OneDrive sync writes, but **git commits are still vulnerable** — OneDrive can truncate a file between the write and the `git add`. Mitigation: verify file integrity before each commit.

---

## Summary

| Category | Count |
|----------|-------|
| **LIVE CRASH** (production down) | 1 |
| **Bugs** (correctness-affecting) | 6 |
| **Design / Assumption Issues** | 5 |
| **Edge-Case Vulnerabilities** | 4 |
| **Test Coverage Gaps** | 9 |
| **Code Hygiene** | 2 |

---

## Bugs

### BUG #1 — CRITICAL: Local vol FDM misses maturity observation boundary condition

**File**: `app/pde_pricer.py` line 648

The local vol FDM backward sweep (`_price_local_vol`) checks observation dates with:

```python
if (not obs_processed[i]) and (t_current <= t_obs < t_axis_phys[step]):
```

The time grid `t_axis_phys` runs `T → 0` with `t_axis_phys[0] = T` (maturity). For `step=0`, the condition `t_obs < t_axis_phys[0]` evaluates to `T < T` — **strict inequality false**. The last observation date IS maturity (`t_obs = T`), so the maturity autocall boundary condition is **never applied**.

**Consequence**: At maturity, uncalled paths with `S_T >= call_barrier` never receive the call payoff (redemption + coupon). Local vol FDM prices are systematically **lower** than flat vol FDM. This is likely the root cause of unexpectedly large `flat - local` FDM price differences observed in test `test_fdm_local_vol_reasonable`.

**Fix**: Apply the maturity observation BC before the backward sweep loop, or change to non-strict inequality: `t_obs <= t_axis_phys[step]`.

---

### BUG #2 — MEDIUM: Survival MC unconditionally adds coupon at autocall

**File**: `app/mc_survival.py` lines 337–338 (scalar `_price_one_path`) and lines 487–488 (vectorized `price`)

Both paths compute:
```python
call_pv = (self.ac.redemption_at_call * self.ac.notional
           + self.ac.coupon_per_period())
```

The coupon is added **unconditionally** at every autocall, without checking whether `S_t >= coupon_barrier`. If a custom structure has `call_barrier < coupon_barrier`, this overpays.

**Contrast**: The FDM pricer (`_apply_autocall_bc` at line 324–331) correctly gates the coupon on `S_axis >= coupon_barrier`.

**Current impact**: None — all four pre-configured securities have `call_barrier >= coupon_barrier`. But this is a latent bug for custom structures built via the Product Builder.

**Fix**: Add `self.ac.coupon_barrier` check in both scalar and vectorized survival MC call contribution.

---

### BUG #3 — LOW: `continuous_autocall_closedform()` — undiscounted immediate-call return

**File**: `app/pde_pricer.py` lines 775–777

```python
if B <= S0:
    return notional * (1 + coupon_pa * T)
```

When `B <= S0` (spot already at or above the barrier), the function returns the **undiscounted** notional plus full coupon stream. For a 2Y 8% coupon note, this returns `$1,160` instead of the correct PV: `$1,000 * exp(-r * T_avg)` plus discounted coupons.

**Severity**: LOW — the function is documented as an approximation and this condition rarely triggers for realistic SPX (spot is typically below 105% call barrier).

**Fix**: Discount the return value.

---

### BUG #4 — LOW: `np.clip()` silently masks NaN prices

**File**: `app/pde_pricer.py` lines 528 and 702

```python
price=float(np.clip(price, 0, self.notional * 1.5))
```

If the FD scheme produces NaN (numerical instability), `np.clip(NaN, 0, 1500)` behavior is version-dependent — some numpy versions return NaN, others return 0. Either way, the caller never knows the FD computation failed.

**Fix**: Add `if np.isnan(price) or not np.isfinite(price): warn/raise`.

---

### BUG #5 — LOW: FDM `_terminal_u()` knock-in approximation is inconsistent

**File**: `app/pde_pricer.py` lines 289–293

```python
ki_mask = self.S_axis < self.ac.protection_barrier * self.S_ref
V_terminal[ki_mask] = self.S_axis[ki_mask] / self.S_ref * self.notional
```

Treats terminal spot below protection as "knocked-in" and assigns the knock-in payoff. But `european_ki` is **path-dependent** (breached at any observation, not just terminal). The comment acknowledges this as approximate, but the code inconsistently: (a) applies knock-in at terminal, AND (b) claims knock-in is handled through observation BC. Specifically, the terminal condition is conservative (par) but then overridden for ki_mask — the interaction between the two is unclear.

**Severity**: LOW — effect is small, documented as approximate.

**Fix**: Either remove the ki_mask override (use pure par terminal + handle KI through observation dates) or document the exact approximation rigorously.

---

### BUG #6 — LOW: Survival MC vectorized `price()` silently drops `return_paths` when paths don't stay stored alongside vectors

**File**: `app/mc_survival.py` lines 535–541

```python
if return_paths:
    stored_paths = []
    call_indices = []
    for idx in range(min(50, N)):
        _, sp, ci = self._price_one_path(store=True)
```

The vectorized `price()` computes the actual price using numpy arrays (active-mask pattern) but then generates **50 completely independent single-path runs** for visualization via `_price_one_path()`. These paths use a different RNG state than the price computation, so they do not correspond to any of the priced paths.

**Severity**: LOW — the visual paths are still valid GBM realizations, just not the same ones driving the price.

**Fix**: Either save the first 50 rows of the vectorized computation, or document the independence explicitly.

---

## Design / Assumption Issues

### DS-1: Call probabilities use flat vol only

**File**: `app/autocallable.py` lines 382–454

`call_probabilities()` takes a single `sigma` parameter. When the user runs Heston or local vol pricing, the call probability table on Page 2 still uses flat vol. This produces an inconsistent user experience: the price uses the vol surface, but the call probability visualization uses a different model.

**Recommendation**: Add a note in the UI when vol model ≠ flat, or pass the ATM implied vol from the surface.

---

### DS-2: Survival MC stored paths are independent of priced paths

See Bug #6 above. The visualization disconnect applies to all vol models.

---

### DS-3: FDM auto-corrects `N_tau` silently

**File**: `app/pde_pricer.py` lines 228–234

When Courant > 0.5, `__init__` increases `N_tau` to satisfy stability. The sidebar shows the user's requested value (e.g., `N_tau=5`), but internally `self.N_tau` is much higher. This inconsistency could confuse debugging.

**Recommendation**: Return the actual `N_tau` used in the result so the UI can display it.

---

### DS-4: Feller condition is a soft penalty, not a hard constraint

**File**: `app/heston.py` lines 489–491

```python
if kappa_ * theta_ < 0.5 * gamma_ ** 2:
    feller_penalty = 1000 * (0.5 * gamma_ ** 2 - kappa_ * theta_) ** 2
```

The optimizer can return values where the Feller condition is violated. The Heston SDE simulation uses Full Truncation to handle this, but calibrated parameters that violate Feller suggest the model may be a poor fit.

**Recommendation**: Surface a warning in the UI when calibrated values violate Feller.

---

### DS-5: `continuous_autocall_closedform()` `expected_call_time = T/2`

**File**: `app/pde_pricer.py` line 794

Assumes the first-passage time, conditional on passage before T, is uniformly distributed at midpoint. Under GBM with positive drift, the distribution is biased earlier. This biases the closed-form price by a few percent.

**Recommendation**: Document this as a known approximation, or use the exact inverse Gaussian distribution for expected passage time.

---

## Edge-Case Vulnerabilities

### EC-1: `S_ref` = 0 or negative crashes `FDPricer`

**File**: `app/pde_pricer.py` `__init__`

Computes `np.log(S0/C)` and divides by `sigma`. If `S_ref` is set to 0 (e.g., manual override), the pricer crashes with a cryptic divide-by-zero or log(0) error.

**Recommendation**: Add input validation in `FDPricer.__init__` or `AutoCallable.__post_init__`.

---

### EC-2: Zero TTM causes NaN propagation in calibration

**File**: `app/vol_surface.py` `bs_implied_vol()` line 95

Returns `None` for `T < 1e-6`. Callers in the calibration loop (e.g., `objective_fast()` in `heston.py`) may not handle `None`, leading to NaN in RMSE and potentially a failed calibration.

**Recommendation**: Filter out `ttm_years < 0.05` before passing to calibration (already done: `df[df['ttm_years'] >= 0.1]`, but add a safeguard inside `bs_implied_vol` callers).

---

### EC-3: Empty snapshot after quality filters

**File**: `app/data_loader.py` line 184

`load_snapshot()` raises `ValueError` when all rows are filtered. The sidebar catches this and calls `st.stop()`, but any page that calls `load_snapshot()` directly (bypassing sidebar) would crash with an unhandled exception.

**Recommendation**: All snapshot consumers should go through sidebar's `params['snapshot_df']`.

---

### EC-4: `_p_survive()` divide-by-zero when `sigma_t == 0`

**File**: `app/mc_survival.py` lines 255–274

If local vol interpolation returns exactly 0 (theoretically possible at grid boundaries despite clamping), `_p_survive()` computes `1/(sigma_t * sqrt(dt))` → division by zero → NaN.

The local vol is clamped to `[0.05, 1.0]` in most callers, but `_p_survive()` itself has no guard.

**Recommendation**: Add `if sigma_t * sqrt(dt) < 1e-10: return 0.5` guard inside `_p_survive()`.

---

## Test Coverage Gaps

### GAP-1: No dedicated test for `_price_local_vol()` FDM algorithm

**Status**: The local vol FDM path is tested only indirectly via `test_fdm_local_vol_reasonable` in `test_mc_pricers.py`, using coarse grid (N_x=80, N_tau=60) and checking only that the price is "within $200 of flat vol FDM."

**Missing**:  
- A test that runs local vol FDM at fine grid (N_x=200) and verifies convergence  
- A test that verifies the maturity observation BC is applied (would have caught Bug #1)  
- A test with synthetic flat local vol surface (σ(S,t) = constant ∀ S,t) — should match flat vol FDM exactly  
- A test that verifies the Courant auto-correction for local vol properly uses `max_sigma`

**Recommended tests**:
```python
def test_local_vol_fdm_matches_flat_when_vol_is_constant():
    """Local vol FDM with constant local vol surface = flat vol FDM."""
    
def test_local_vol_fdm_applies_maturity_observation_bc():
    """Maturity observation BC must fire at t=T."""
    
def test_local_vol_fdm_courant_respects_max_sigma():
    """Auto-corrected N_tau must satisfy Courant <= 0.5 for max local vol."""
```

---

### GAP-2: No test for Bates MC with λ > 0

**Status**: `test_mc_bates_no_jumps_matches_heston` tests λ=0 only.

**Missing**: A test with λ > 0 (e.g., `lam=0.5, mu_J=-0.15, sig_J=0.25` — the sidebar defaults) that verifies:
- Bates price is finite and positive
- Bates price differs from Heston at the same params (jumps add value to OTM options)
- The jump correction in the survival MC (`_jump_survival_correction`) reduces survival probability

---

### GAP-3: No test for soft_protection terminal payoff

**Status**: The Digital Autocall uses `protection_type="soft_protection"` and `protection_floor=0.80`, but `test_terminal_payoff_*` tests only exercise the `european_ki` path.

**Missing**: Tests for:
```python
def test_terminal_payoff_soft_protection_above_floor():
    """spot_T/S_ref > protection_floor → return spot_T/S_ref * notional * discount"""
    
def test_terminal_payoff_soft_protection_below_floor():
    """spot_T/S_ref < protection_floor → return protection_floor * notional * discount"""
    
def test_terminal_payoff_soft_protection_with_coupon():
    """Final coupon paid when spot >= coupon_barrier under soft_protection"""
```

---

### GAP-4: No test for `_apply_autocall_bc` coupon gating

**Status**: The FDM autocall BC correctly distinguishes between call barrier and coupon barrier (line 324–331), but no test verifies this.

**Missing**: A test where the coupon barrier differs from the call barrier, verifying that:
- At call trigger + below coupon barrier → redemption only, no coupon  
- At call trigger + above coupon barrier → redemption + coupon

---

### GAP-5: No test for Dupire local vol grid extrapolation

**Status**: The grid spans moneyness [0.40, 1.80] × TTM [0.01, T+0.05]. Queries outside this range are clamped. No test verifies that clamped queries return sensible values.

**Missing**: Test `dupire_local_vol()` at moneyness 0.30 (below grid), 2.0 (above grid) and verify return is clamped within [0.05, 1.0].

---

### GAP-6: No test for Merton Poisson series convergence

**Status**: `_merton_call_bs_series` truncates at `n_terms=10`. No test verifies this is sufficient.

**Missing**: Test that increasing `n_terms` from 10 to 20 changes the price by less than $0.01.

---

### GAP-7: Thomas algorithm edge cases

**Status**: `test_thomas_solve_correctness` tests only n=5.

**Missing**: Tests for:
- n=1 (single equation)  
- n=2 (smallest non-trivial tridiagonal)  
- n=1000 (large system, verify O(n) scaling)

---

### GAP-8: Step-down barrier call probability monotonicity

**Status**: `test_step_down_barrier` only checks `call_barrier_at_period()`. No test verifies that `call_probabilities()` increases total call probability as barriers step down.

**Missing**: Compare total call probability for step-down vs. a fixed-barrier equivalent — step-down should have ≥ total call probability.

---

### GAP-9: `_tau_to_t` / `_t_to_tau` roundtrip invariance

**Status**: These coordinate transforms (lines 239–245) are untested.

**Missing**: 
```python
def test_tau_t_roundtrip():
    for t in [0.0, 0.5, 1.0, 2.0]:
        assert abs(fd._tau_to_t(fd._t_to_tau(t)) - t) < 1e-10
```

---

## Code Hygiene

### HYG-1: Double import in `pde_pricer.py`

**File**: `app/pde_pricer.py` lines 768–769

```python
from scipy.stats import norm
from scipy.stats import norm  # duplicate — remove line 769
```

---

### HYG-2: Double import in `heston.py`

**File**: `app/heston.py` lines 1240, 1242

```python
from app.vol_surface import bs_implied_vol
# ... blank line ...
from app.vol_surface import bs_implied_vol  # duplicate — remove line 1242
```

In function `calibrate_bates()`.

---

## Priority Action List

| # | Action | Severity | Effort |
|---|--------|----------|--------|
| **0** | **🛑 FIX LIVE CRASH**: Commit + push full `sidebar.py` (OneDrive truncation) | PRODUCTION DOWN | 2 min |
| **1** | **Fix Bug #1**: Local vol FDM maturity BC | CRITICAL | 5 min |
| **2** | **Fix Bug #2**: Survival MC coupon barrier check | MEDIUM | 10 min |
| **3** | **Add Test GAP-1**: Dedicated local vol FDM tests | HIGH | 30 min |
| **4** | **Add Test GAP-3**: Soft protection payoff tests | HIGH | 20 min |
| **5** | **Fix HYG-1/2**: Remove double imports | LOW | 2 min |
| **6** | **Fix Bug #4**: NaN detection after FD | LOW | 10 min |
| **7** | **Fix Bug #3**: Discount immediate-call return | LOW | 5 min |
| **8** | **Add Test GAP-2**: Bates with λ > 0 | MEDIUM | 15 min |
| **9** | **Add Test GAP-4**: FDM coupon gating test | MEDIUM | 15 min |
| **10** | **Add Test GAP-7/9**: Edge case tests for Thomas + tau/t | LOW | 10 min |
| **11** | **Fix DS-4**: Surface Feller violation warning in UI | LOW | 10 min |

---

## Paper Alignment Notes

### Deng, Mallett, McCann (2011) — Paper 1

| Paper Reference | Implementation | Status |
|-----------------|---------------|--------|
| §2.2 FD explicit scheme + change of variables | `pde_pricer.py` `price()` | OK — matches paper |
| §2.2 Courant stability: δτ/δx² ≤ 0.5 | `pde_pricer.py` auto-correction | OK — enforced |
| §2.2 Autocall BC at obs dates | `_apply_autocall_bc()` | OK — applies correctly |
| §2.3 Closed-form continuous | `continuous_autocall_closedform()` | OK — approximate, documented |
| §2.2 Call probabilities (joint lognormal) | `call_probabilities()` | OK — marginal approximation, documented |

### Haugh (2013) — Paper 2

| Paper Reference | Implementation | Status |
|-----------------|---------------|--------|
| Eq. 2 Dupire local vol | `vol_surface.py` `dupire_local_vol()` | OK — matches formula |
| Eq. 12-13 Heston SDE | `heston.py` / `mc_standard.py` | OK — Full Truncation scheme |
| Eq. 23 Heston CF (branch-cut-safe) | `heston_char_fn()` | OK — uses Albrecher form |
| Eq. 23 warning: never use alternate form | Code comment at line 83 | OK — documented |

### Alm, Harrach, Harrach, Keller (2013) — Paper 3

| Paper Reference | Implementation | Status |
|-----------------|---------------|--------|
| §2 Eq. 2.3 Standard MC | `mc_standard.py` `_simulate_paths()` | OK — GBM exact increments |
| Algorithm 1 One-step survival | `mc_survival.py` `_price_one_path()` | OK — matches paper |
| §2.2 GHK importance sampling (basket) | Not implemented | OK — deferred per CLAUDE.md |
| Stable differentiation claim | `_price_one_path` formula | OK — barrier handled analytically |

---

*Review saved to `code_reviews/2026-06-08_code_review.md`*
