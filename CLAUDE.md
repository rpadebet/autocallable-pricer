# CLAUDE.md — AutoCallable Analytics Platform

> **Last updated**: 2026-06-07
> **Status**: v0.5.14 — 74/74 tests passing (test_heston.py skipped: FUSE truncation) — ALL 5 pages complete + vol-aware pricers + calibration UI connected — 15 new gap-fill tests — deploy pending
> **Author / LLM**: Claude Sonnet 4.6 via Cowork (Rohit's session)
> **Project folder**: `C:\Users\rohit\OneDrive\Documents\Claude Apps\Autocallables\`
> **Linux mount path (bash)**: `/sessions/*/mnt/Documents/Claude Apps/Autocallables/`

---

## 0. Purpose & Framing

This app is a **demonstration tool**, not a production system. Its primary job is to help Rohit Padebettu explain the mechanics of autocallable structured product pricing to a sophisticated interviewer (Keith Loggie, MerQube). Every design decision should optimize for:

1. **Clarity of explanation** — the math must be visible, not hidden inside functions
2. **Visual impact** — every numerical method gets a visualization that shows *what the algorithm is doing*
3. **Reliability during demo** — no API failures, no network dependencies, no surprises

Speed and production-readiness are secondary. A beautiful, explainable app beats a faster but opaque one.

---

## Development Standards — Mandatory Rules for All Sessions

These rules apply to every session, every file, every change. No exceptions. Any LLM working on this project must read and follow these before writing a single line of code.

### 1. Documentation Over Code
- Every function, class, and module must have a docstring explaining: what it does, why it exists, what its inputs/outputs are, and any edge cases or assumptions.
- Inline comments must explain *why* — not *what*. The code says what. The comment says why.
- Complex logic blocks must be preceded by a plain-English explanation of the algorithm before the code.
- Prefer readable, explicit code over clever, compact code. If it needs a comment to understand, it needs better naming first.

### 2. Change Log — Required for Every Session
- The file `CHANGELOG.md` in the project root must be updated at the end of every session.
- Format: `## [vX.Y.Z] — YYYY-MM-DD` followed by `### Added`, `### Changed`, `### Fixed` sections.
- Every meaningful change — even small ones — gets a line. No silent edits.
- Version bump rules: patch (0.0.X) for bug fixes, minor (0.X.0) for new features, major (X.0.0) for architecture changes.

### 3. Version Control — Every Change Committed
- Every logical unit of work gets its own git commit. Do not batch unrelated changes.
- Commit message format: `[type]: short description` where type is `feat`, `fix`, `docs`, `test`, `refactor`, or `chore`.
- Commit body (when non-trivial): explain *why* the change was made, not just what changed.
- After any session that modifies code, check `git status` and commit everything before closing.

### 4. Test Plans — Consistent Execution, Documented Results
- The test plan in the Technical Spec (Section 7) is the canonical test suite. It must be run after every meaningful code change.
- Test results must be logged to `test_results/YYYYMMDD_HHMM_results.md` with: test name, pass/fail, actual vs expected output, and any notes.
- A failing test is never silently ignored. It is either fixed immediately (with explanation) or escalated to Rohit with a clear description of what failed and why.
- New features must come with new tests before the feature is considered complete.

### 5. Surgical Fixes — Touch Only What Is Asked
- When asked to fix a bug or issue: identify the exact lines responsible, fix only those lines, and do not refactor, rename, reformat, or improve anything else in the same commit.
- Before making a fix, state explicitly: "I will change [X] in [file], lines [N–M]. I will not touch anything else."
- If a fix requires touching adjacent code, stop and ask permission first.

### 6. Enhancements — Ask Before Building
- When asked to enhance, add, or extend any feature: do not start writing code.
- First, ask clarifying questions until you are certain of: the exact scope, the expected inputs and outputs, how it integrates with existing modules, and any edge cases.
- Only begin work after Rohit has confirmed the approach.
- State your plan explicitly before writing any code: "Here is what I will build: [plan]. Does this match what you want?"

### 7. No Scope Creep
- Do not improve, optimize, or refactor code you are not explicitly asked to change.
- Do not add features "while you're in there."
- Do not change formatting, variable names, or structure outside the area of the requested change.
- If you notice something worth improving elsewhere, log it as a note but do not act on it.

### 8. Session Start Protocol
Every session working on this project must:
1. Read this CLAUDE.md in full.
2. Read `CHANGELOG.md` to understand current version and recent changes.
3. Read the relevant module(s) before modifying them.
4. State what you are about to do before doing it.
5. After completing work: update CHANGELOG.md, commit to git, run the test suite, log results.

---

## 1. Business Context

Rohit Padebettu is interviewing at **MerQube** (Keith Loggie, ex-SPDJI) for an **Index Researcher — Options specialist** role. MerQube builds financial indexes for bank issuers, including autocallable structured products. Keith asked Rohit to demonstrate:

1. Familiarity with **autocallable structured products** (pricing, features, payoffs)
2. Experience with **Monte Carlo methods** (variance reduction, stable differentiation)
3. Knowledge of **volatility surfaces** (implied vol, local vol, Heston calibration)
4. **Python** proficiency

**The deliverable**: A polished, interactive **Streamlit app** deployed to Streamlit Community Cloud (free, shareable URL) that functions as a professional autocallable pricing and analytics platform. Keith should be able to open the link and play with it immediately — no setup required.

---

## 2. Source Research Papers

All three PDFs are in the project folder:

| File | Authors | Key Concepts |
|------|---------|-------------|
| `modeling-autocallable-structured-products.pdf` | Deng, Mallett, McCann (2011) | PDE framework, finite difference, closed-form continuous autocall, call probabilities |
| `LocalStochasticJumps.pdf` | Martin Haugh, Columbia IEOR (2013) | Local vol (Dupire), Heston stochastic vol, Merton/Kou/Bates jump-diffusion, characteristic functions |
| `StableDiffs.pdf` | Alm, Harrach, Harrach, Keller (JCF 2013) | One-step survival MC, GHK importance sampling, variance reduction, stable Greeks |

---

## 3. Application Architecture

### 3.1 App Name
**AutoCallable Analytics Platform** — `autocallable-pricer` on Streamlit Cloud

### 3.2 Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | Python 3.11+ | |
| UI | Streamlit | Multi-page app |
| Numerics | numpy, scipy, pandas | |
| Market data | **Static snapshots (CSV/JSON)** | Pre-collected; no live API during demo |
| Data collection utility | `yfinance` (scripts only) | Used once to collect snapshots, not at runtime |
| Optimization | scipy.optimize | Heston calibration |
| Plotting | plotly (interactive), matplotlib | 3D surfaces, path animations, convergence charts |
| Deployment | Streamlit Community Cloud | Free, GitHub-connected |
| Repo | GitHub (public) | |

**Critical: No live market data at runtime.** The app loads from pre-collected snapshot files stored in `sample_data/`. This eliminates API failures during the demo and makes every run reproducible.

### 3.3 Directory Structure

```
autocallable-pricer/
├── CLAUDE.md                        # This file (copy to GitHub root)
├── README.md                        # User-facing documentation
├── requirements.txt
├── streamlit_app.py                 # Main entry point + navigation
│
├── sample_data/                     # Pre-collected SPX options snapshots
│   ├── README.md                    # Describes each snapshot date + source
│   ├── spx_options_2025-12-19.csv   # SPX options chain, Dec 19 2025
│   ├── spx_options_2026-01-16.csv   # SPX options chain, Jan 16 2026
│   ├── spx_options_2026-03-20.csv   # SPX options chain, Mar 20 2026
│   └── spx_options_2026-06-06.csv   # SPX options chain, Jun 6 2026 (most recent)
│
├── scripts/
│   └── collect_snapshot.py          # Run once to pull & save a new SPX snapshot
│
├── app/
│   ├── __init__.py
│   ├── pages/
│   │   ├── 01_Vol_Surface.py        # Vol surface + Heston calibration + Dupire
│   │   ├── 02_Pricer.py             # Autocallable pricer (3 methods + path animation)
│   │   ├── 03_FDM_Visualization.py  # Finite difference grid visualization
│   │   ├── 04_Greeks.py             # Greeks + stable differentiation comparison
│   │   └── 05_Scenarios.py          # Payoff diagrams + scenario analysis
│   └── components/
│       ├── sidebar.py               # Shared assumptions sidebar (all model params)
│       └── securities.py            # Pre-configured security definitions
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py               # Loads snapshot CSVs, computes implied vols
│   ├── vol_surface.py               # Implied vol surface, Dupire local vol
│   ├── heston.py                    # Heston model: SDE, characteristic fn, calibration
│   ├── pde_pricer.py                # FD PDE pricer + closed-form continuous autocall
│   ├── mc_standard.py               # Standard Monte Carlo pricer
│   ├── mc_survival.py               # One-step survival MC pricer (Paper 3)
│   └── autocallable.py              # Product definition, payoff, call probabilities
│
├── tests/
│   ├── test_pde_pricer.py
│   ├── test_mc_pricers.py
│   ├── test_heston.py
│   └── test_payoffs.py
│
└── notebooks/
    └── methodology_explainer.ipynb  # Optional: Jupyter notebook walkthrough
```

### 3.4 Application Pages

#### Shared: Assumptions Sidebar (persists across all pages)

A clearly labeled **"Model Assumptions"** section in the Streamlit sidebar, accessible from every page. Changing any value here immediately affects all pricing on re-run. No hardcoded values anywhere in `core/` — all parameters thread through from this sidebar.

**Market / Data section:**
- Data date selector (dropdown: all available snapshot dates)
- Risk-free rate (default: from snapshot metadata or manual override)
- Dividend yield (default: 0.015 for SPX, overridable)

**Heston Model section:**
- Initial variance v₀ (default: calibrated value)
- Mean reversion speed κ
- Long-run variance θ
- Vol-of-vol γ
- Correlation ρ (range: −1 to 0)
- Toggle: "Use calibrated values" (re-runs calibration on demand) vs. manual override

**Monte Carlo section:**
- Number of paths N (slider: 1,000 → 500,000; default: 10,000 for speed)
- Number of time steps M (default: 252 per year)
- Random seed (for reproducibility)
- Antithetic variates on/off
- Control variates on/off (future)

**FDM/PDE section:**
- S-grid steps (default: 1,000)
- τ-grid steps (default: 500)
- x-domain lower bound (default: −5)

**Autocallable Terms section (read-only link):**
- Shows active security name + key terms
- Link to Pricer page to change security selection

---

#### Page 1: Vol Surface (`01_Vol_Surface.py`)
**Purpose**: Demonstrates vol surface construction from real market data and Heston model calibration.

**Features:**
- Load SPX options snapshot from selected date (via data_loader)
- Filter: remove zero-bid quotes, require bid < ask, exclude deep ITM/OTM
- Compute Black-Scholes implied vols for each (K, T) pair
- **Visual 1**: 3D interactive implied vol surface (plotly) — moneyness × maturity × implied vol
- Fit Heston model to the surface via least-squares calibration (scipy)
- **Visual 2**: Heston model surface overlaid on market surface (same 3D plot, two color series) — shows fit quality
- Compute Dupire local vol surface from implied vols (Paper 2, Eq. 2)
- **Visual 3**: Dupire local vol surface side-by-side with implied vol surface — panel layout
- Calibrated Heston parameters displayed: κ, θ, γ, ρ, v₀ and fit error (max abs vol error)
- Expandable "Methodology" section with formula references (Paper 2, Eq. 2, Eq. 12-13, Eq. 23)

**Inputs**: Data date selector (from sidebar), moneyness range filter, maturity range filter

---

#### Page 2: Pricer (`02_Pricer.py`)
**Purpose**: The centerpiece — prices an autocallable using 3 methods and makes all the math visible.

**Security Selector (top of page):**
- Dropdown: choose from pre-configured securities (see Section 3.5)
- On selection: all parameters populate in a read-only "Term Sheet" display card
- No manual configuration required as default UX
- *"Create your own" is a future phase — not built now*

**3 Pricing Methods** (run in parallel, displayed as tabs):

1. **PDE / Finite Difference** (Paper 1)
   - Black-Scholes PDE with autocall boundary conditions
   - Explicit FD scheme; Crank-Nicolson as toggle
   - Discrete and continuous autocall variants
   - Outputs: Price, call probability at each observation date (Table 1 from paper)

2. **Standard Monte Carlo** (baseline)
   - GBM simulation, standard estimator (Paper 3, Eq. 2.3)
   - Configurable: N paths, M steps (from sidebar Assumptions)
   - Outputs: Price ± 95% CI, standard error, convergence chart

3. **One-Step Survival MC** (Paper 3)
   - Glasserman-Staum one-step survival — single underlying
   - GHK importance sampling for basket (Paper 3 §2.2)
   - Outputs: Price ± 95% CI, SE (lower than standard MC), convergence chart

**Core Visualizations (required — not optional polish):**

- **Path Simulation Animation**: Show 20–50 sample GBM paths evolving over time on a chart. Overlay horizontal lines for: call trigger (barrier), protection barrier (knock-in put), spot at time 0. Mark the point where each path triggers an early call (if it does). This is the *"show the students what's happening"* visual.

- **Convergence Chart**: Price estimate vs. number of paths, plotted for both Standard MC and Survival MC on the same chart. X-axis: N paths (log scale). Y-axis: price ± 95% CI band. This visually proves variance reduction — the Survival MC confidence interval is visibly narrower at the same N.

- **Call Probability Table**: From the PDE analytical approach (Paper 1, §2.2) — probability of being called at each observation date. Shows the term structure of autocall risk.

---

#### Page 3: FDM Visualization (`03_FDM_Visualization.py`)
**Purpose**: Make the finite difference algorithm visible. Show the numerical method, not just the output.

**Features:**
- Run the FD pricer with reduced grid (e.g., 100×100) for visualization speed
- **Visual 1**: Heatmap of the value function V(S, t) — x-axis: spot price S, y-axis: time t, color: option value. Shows how value evolves backward from maturity
- **Visual 2**: 3D surface of V(S, t) with the autocall boundary highlighted as a line/plane
- **Visual 3**: Time-slice animation — show V(S) at successive time steps as the FD scheme sweeps backward from T to 0. Renders as a slider or animated loop
- **Visual 4**: Convergence of FD price as grid size increases (N_S × N_τ vs. price) — shows numerical stability
- Expandable "Methodology" section with the heat equation change-of-variables, update rule, and stability condition (all from Paper 1 §2.2)
- Note: this page uses the same `pde_pricer.py` module — just exposes internal grid state for visualization

---

#### Page 4: Greeks (`04_Greeks.py`)
**Purpose**: Demonstrates stable differentiation (key innovation of Paper 3).

**Features:**
- Compute Delta (∂V/∂S) and Vega (∂V/∂σ) using:
  - Standard MC + finite differences → noisy, unstable
  - One-step survival MC + finite differences → smooth, stable
- **Visual 1**: Side-by-side Delta plots as function of bump size ε — Standard MC spiky/sign-flipping, Survival MC smooth
- **Visual 2**: Same for Vega
- **Visual 3**: Delta smile — Delta as function of spot level
- **Visual 4**: Vega surface — Vega as function of spot × time (heatmap)
- **Key message**: Barrier-crossing discontinuity eliminated by construction in one-step survival → reliable numerical Greeks → practical hedging implications

---

#### Page 5: Scenarios (`05_Scenarios.py`)
**Purpose**: Payoff intuition and product understanding.

**Features:**
- Payoff diagram at maturity (x-axis: final spot, y-axis: payoff)
- Value surface V(S, σ) heatmap
- Call probability table by observation date
- What-if sliders: spot, vol, rates, barrier → live price update
- Historical simulation: SPX path from snapshot data overlaid on barrier levels (shows what *would have* happened)

---

### 3.5 Pre-Configured Autocallable Securities

These are built into `app/components/securities.py` as a Python dict. The Pricer page selects from this list. "Create your own" is a future feature.

| Security Name | Structure | Underlying | Obs. Dates | Call Barrier | Coupon | Protection | Maturity |
|---------------|-----------|------------|------------|--------------|--------|------------|---------|
| **Phoenix Autocall** | Standard Phoenix | SPX | Quarterly | 100% | 8% p.a. | 75% (knock-in put) | 2 years |
| **Worst-Of Autocall** | Worst-of basket | SPX / NDX / RUT | Quarterly | 100% | 12% p.a. | 70% (knock-in put) | 3 years |
| **Step-Down Barrier** | Declining barrier | SPX | Monthly | 100% → 95% → 90% (steps down quarterly) | 10% p.a. | 75% | 2 years |
| **Digital Autocall** | Binary payoff | SPX | Annual | 105% | $50 (digital, not coupon) | 80% (capital protected) | 3 years |

All parameters are realistic and match common bank-issued autocallable note structures.

---

## 4. Key Algorithms (Paper → Code Mapping)

### 4.1 PDE Finite Difference (Paper 1, §2.2)

**Change of variables**:
```
S = C·exp(x),  t = T - 2τ/σ²,  V(S,t) = C·exp(αx + βτ)·u(x,τ) + f(0)·exp(...)
k1 = 2(r-q)/σ²,  α = -½(k1-1),  β = -α² - 2(r+cds)/σ²
```

**Heat equation**: ∂u/∂τ = ∂²u/∂x²  for x ∈ (-∞, 0), τ > 0

**Explicit FD update**:
```
u[m+1][n] = u[m][n] + (δτ/δx²) * (u[m][n+1] - 2·u[m][n] + u[m][n-1])
```
Stability condition: δτ/δx² ≤ 1/2

**Grid**: N=1000 (x-steps), M=500 (τ-steps), x ∈ [-5, 0]

**FDM Visualization hook**: `pde_pricer.py` must optionally return the full V(S, t) grid (not just the final price) when `return_grid=True` is passed — used by Page 3.

**Implementation file**: `core/pde_pricer.py`

---

### 4.2 Call Probability (Paper 1, §2.2 probability approach)

```python
# Xi = (r - q - ½σ²)·Δt + σ·√Δt·Wi,  Wi ~ N(0,1)
# S_tc_i = S0 · exp(sum of Xi up to i)
# pi = P(S_j < C for j<i  AND  S_i >= C)
# Uses joint lognormal distribution of increments
```

**Implementation**: `core/autocallable.py`  → `call_probabilities()`

---

### 4.3 Closed-Form Continuous Autocallable (Paper 1, §2.3)

Closed-form solution derived by solving the inhomogeneous heat PDE.

**Implementation**: `core/pde_pricer.py` → `continuous_autocall_closedform()`

---

### 4.4 Heston Model Calibration (Paper 2)

**SDE under risk-neutral measure**:
```
dSt = (r-q)·St·dt + √σt·St·dW_S
dσt = κ(θ - σt)·dt + γ·√σt·dW_vol
Corr(dW_S, dW_vol) = ρ
```

**Characteristic function** (Paper 2, Eq. 23):
```python
d = sqrt((ρ·γ·u·i - κ)² + γ²·(i·u + u²))
g = (κ - ρ·γ·u·i - d) / (κ - ρ·γ·u·i + d)
φT(u) = exp(i·u·(log(S0) + (r-q)·T))
       * exp(θ·κ·γ⁻²·((κ - ρ·γ·u·i - d)·T - 2·log((1 - g·exp(-dT))/(1-g))))
       * exp(σ0·γ⁻²·(κ - ρ·γ·u·i - d)·(1 - exp(-dT))/(1 - g·exp(-dT)))
```

**⚠️ CRITICAL**: Use Eq. 23 ONLY — not the alternative representation. The alternative causes branch-cut discontinuities (Paper 2 explicitly warns). Never use the `φ̂T(u)` form.

**Calibration**: scipy.optimize.minimize (Nelder-Mead or L-BFGS-B), 3-5 random starting points. Bounds: κ,θ,γ,v₀ > 0, -1 < ρ < 0. Enforce Feller condition: κθ > ½γ².

**Implementation**: `core/heston.py`

---

### 4.5 Dupire Local Vol (Paper 2, Eq. 2)

```
σ²_local(T, K) = [∂C/∂T + (r-q)·K·∂C/∂K + q·C] / [K²/2 · ∂²C/∂K²]
```

Build smooth call price surface → interpolate implied vols (SVI or cubic spline) → differentiate numerically.

**⚠️ Warning**: Unstable without smooth interpolation — fit spline in IV space first, then differentiate analytically.

**Implementation**: `core/vol_surface.py` → `dupire_local_vol()`

---

### 4.6 Standard Monte Carlo (Paper 3, §2, Eq. 2.3)

```python
for n in range(N_paths):
    s = S0
    path_history = [S0]   # SAVE for path animation visualization
    for j, t_j in enumerate(obs_dates):
        dt = t_j - t_{j-1}
        z = np.random.normal()
        s = s * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*z)
        path_history.append(s)
        if s/Sref >= B:
            payoff = e^{-r*t_j} * Q_j
            break
    else:
        payoff = e^{-r*T} * q(s/Sref)
```

Save `path_history` for a subset of paths (first 50) for the Page 2 path animation. Return as optional output when `return_paths=True`.

For basket (worst-of): correlated GBM, barrier on min(S_i/Sref_i).

**Implementation**: `core/mc_standard.py`

---

### 4.7 One-Step Survival MC (Paper 3, Algorithm 1)

**Key idea**: Sample from truncated normal, analytically handle barrier crossings.

```python
# At each step j, conditioning on S_j = s_j:
p_j = Phi((log(B*Sref/s_j) - (mu - sigma²/2)*dt) / (sigma*sqrt(dt)))
# p_j = P(S_{j+1}/Sref < B | S_j = s_j)

# Sample from truncated normal:
u = np.random.uniform(0, 1)
z = norm.ppf(p_j * u)    # clip p_j*u to [1e-10, 1-1e-10]
s_{j+1} = s_j * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*z)

payoff_contribution += (1 - p_j) * L * e^{-r*(t_{j+1})*Q_{j+1}}
L *= p_j   # running likelihood weight
```

Final estimator (Eq. 2.7):
```
Q̃ = L_m * e^{-rT} * q(s_m/Sref) + Σ_j L_j * (1-p_j) * e^{-r*t_{j+1}} * Q_{j+1}
```

**Why stable Greeks**: No paths cross the barrier → payoff is continuous w.r.t. S0 → reliable ∂V/∂S.

**For basket**: GHK importance sampling (Paper 3, §2.2) — sequential conditional Cholesky decomposition.

**Implementation**: `core/mc_survival.py`

---

## 5. Data Architecture

### Static Snapshot Approach (replacing live data)

**Rationale**: Makes the app fast, reliable, and demo-safe. No API failures, no rate limiting. The goal is to demonstrate *what we can do with the data*, not to build a data pipeline.

---

### Data Collection Plan (Week of June 6–11, 2026)

**Target: 3-4 high-quality snapshots** spanning Friday June 6 through Wednesday June 11, each from a different day and time of day.

#### Step 1 — Historical data check (Day 1, very first task)

Before running any live collection, check whether Friday June 6, 2026's SPX options chain is available retroactively. Check in this order:

1. **yfinance retroactive** — yfinance often returns the last trading day's chain even after market close. Run `collect_snapshot.py` Saturday morning and check whether Friday's chain is still accessible (spot price and expiry chains will confirm the date).
2. **CBOE DataShop** (https://datashop.cboe.com) — CBOE sells historical SPX options data; a single-day chain costs ~$10–30 depending on the plan. Check availability.
3. **Nasdaq Data Link / Quandl** (https://data.nasdaq.com) — check `OPRA` or `OEX` datasets for free-tier historical options.
4. **Fallback**: if no retroactive data is available for Jun 6, skip it and start with Monday June 9.

Document the outcome in `sample_data/README.md`.

#### Step 2 — Live intraday snapshots (Mon June 9 – Wed June 11)

Run `collect_snapshot.py` at three times each trading day:
- **Market open** (~9:45am ET) — widest spreads but captures opening vol
- **Midday** (~12:00pm ET) — typically tightest spreads, most liquid
- **Near close** (~3:45pm ET) — high volume, best for vol surface shape

Each run saves a file timestamped with date + time: `spx_options_20260609_1200.csv`

Review the quality scores printed by the script and keep the best-scoring file per day.

#### Step 3 — Target snapshot library

| Snapshot | Date | Time | Status |
|----------|------|------|--------|
| Snapshot 1 | Fri Jun 6 | Retroactive (any time) | 🔲 Check historical sources |
| Snapshot 2 | Mon Jun 9 | Best of 3 intraday | 🔲 Collect |
| Snapshot 3 | Tue Jun 10 | Best of 3 intraday | 🔲 Collect |
| Snapshot 4 | Wed Jun 11 | Best of 3 intraday | 🔲 Collect |

---

### `scripts/collect_snapshot.py`

Run at 9:45am, 12pm, and 3:45pm ET on each collection day. Includes a quality scoring function that prints a summary so you can decide which file to keep.

```python
import yfinance as yf
import pandas as pd
from datetime import datetime

def score_snapshot(df: pd.DataFrame) -> dict:
    """Score a snapshot for data quality. Higher is better."""
    by_expiry = df.groupby("expiry")
    n_expiries = len(by_expiry)
    avg_strikes = by_expiry.size().mean()
    rel_spread = (df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2)
    pct_tight = (rel_spread < 0.10).mean() * 100   # % with spread < 10% of mid
    pct_missing_iv = df["impliedVolatility"].isna().mean() * 100
    score = avg_strikes * 0.4 + pct_tight * 0.4 - pct_missing_iv * 0.2
    return {
        "n_expiries": n_expiries,
        "avg_strikes_per_expiry": round(avg_strikes, 1),
        "pct_tight_spread": round(pct_tight, 1),
        "pct_missing_iv": round(pct_missing_iv, 1),
        "quality_score": round(score, 1),
    }

def collect_spx_snapshot(output_dir="sample_data"):
    ticker = yf.Ticker("^SPX")
    spot = ticker.history(period="1d")["Close"].iloc[-1]
    rfr = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100

    expiries = ticker.options
    frames = []
    for exp in expiries:
        chain = ticker.option_chain(exp)
        calls = chain.calls.copy()
        calls["expiry"] = exp
        calls["spot"] = spot
        calls["rfr"] = rfr
        frames.append(calls)

    df = pd.concat(frames)
    df = df[(df["bid"] > 0) & (df["bid"] < df["ask"])]   # remove bad ticks

    ts = datetime.now().strftime("%Y%m%d_%H%M")           # e.g. 20260609_1200
    path = f"{output_dir}/spx_options_{ts}.csv"
    df.to_csv(path, index=False)

    scores = score_snapshot(df)
    print(f"\n{'='*50}")
    print(f"Snapshot saved: {path}")
    print(f"  Rows:                  {len(df)}")
    print(f"  Expiry dates:          {scores['n_expiries']}")
    print(f"  Avg strikes/expiry:    {scores['avg_strikes_per_expiry']}")
    print(f"  Tight spread (< 10%):  {scores['pct_tight_spread']}%")
    print(f"  Missing IV:            {scores['pct_missing_iv']}%")
    print(f"  Quality score:         {scores['quality_score']}  ← higher is better")
    print(f"{'='*50}\n")

collect_spx_snapshot()
```

**Usage**: `python scripts/collect_snapshot.py` — run 3× per trading day, keep the highest-scoring file per day.

---

### `core/data_loader.py`

Filenames use `YYYYMMDD_HHMM` format. The UI date dropdown shows date + time + session label.

```python
import pandas as pd
import glob, os

def list_available_snapshots(data_dir="sample_data") -> list[dict]:
    """Returns list of {key, label} dicts sorted by timestamp."""
    files = sorted(glob.glob(f"{data_dir}/spx_options_*.csv"))
    results = []
    for f in files:
        key = os.path.basename(f).replace("spx_options_", "").replace(".csv", "")
        # key format: YYYYMMDD_HHMM  e.g. "20260609_1200"
        try:
            dt = pd.to_datetime(key, format="%Y%m%d_%H%M")
            session = "Market Open" if dt.hour < 11 else ("Midday" if dt.hour < 14 else "Near Close")
            label = f"{dt.strftime('%Y-%m-%d %H:%M')}  —  {session}"
        except Exception:
            label = key
        results.append({"key": key, "label": label})
    return results

def load_snapshot(key: str, data_dir="sample_data") -> pd.DataFrame:
    path = f"{data_dir}/spx_options_{key}.csv"
    df = pd.read_csv(path, parse_dates=["expiry"])
    df = df[(df["bid"] > 0) & (df["bid"] < df["ask"]) & (df["volume"] > 0)]
    return df
```

**Dropdown label examples**: `"2026-06-09 12:00  —  Midday"`, `"2026-06-10 09:45  —  Market Open"`

**Caching**: Use `@st.cache_data` in Streamlit pages to cache loaded DataFrames.

---

## 6. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| CLAUDE.md | ✅ Written | This document |
| AutoCallable_Project_Plan.docx | ✅ Written | Project plan |
| Historical data check (Jun 6) | 🔲 TODO | First task on Day 1 — check CBOE/Nasdaq/yfinance retroactive for Fri Jun 6 chain |
| `scripts/collect_snapshot.py` | ✅ Done | Includes quality scoring; run 3× per trading day Mon–Wed |
| SPX snapshots (3-4 total) | 🔲 Collect Mon–Wed Jun 9–11 | Target: Jun 6 (retroactive if avail.) + Mon–Wed best intraday snapshots |
| `app/data_loader.py` | ✅ Done | Snapshot loader + filtering; `list_available_snapshots()` with date+time labels |
| `app/components/securities.py` | ✅ Done | 4 pre-configured security definitions |
| `app/autocallable.py` | ✅ Done — fixed truncation v0.3.0 | Product definition, payoff, call probs |
| `app/pde_pricer.py` | ✅ Done | FD + closed-form; `return_grid=True` hook |
| `app/mc_standard.py` | ✅ Done | Standard MC; `return_paths=True` hook |
| `app/mc_survival.py` | ✅ Done | One-step survival MC (single + basket) |
| `app/heston.py` | ✅ Done | SDE, char. fn, calibration |
| `app/vol_surface.py` | ✅ Done | Implied vol, Dupire |
| `app/components/sidebar.py` | ✅ Done | Assumptions sidebar (all params) |
| `app/pages/01_Vol_Surface.py` | ✅ Done | 3D surface, Heston overlay, Dupire side-by-side |
| `app/pages/02_Pricer.py` | ✅ Done | Security dropdown, 3 methods, path animation, convergence |
| `app/pages/03_FDM_Visualization.py` | ✅ Done | FD grid heatmap, time-slice animation |
| `app/pages/04_Greeks.py` | ✅ Done v0.3.1 | Delta stability, Vega stability, Delta smile, Methodology tab |
| `app/pages/05_Scenarios.py` | ✅ Done v0.3.2 | Payoff diagram, what-if sliders, call prob table, value surface |
| `app/Home.py` | ✅ Done (entry point) | Entry point + navigation |
| `tests/` | ✅ Done — 66/66 passing | Unit tests (see Section 7) |
| `README.md` | 🔲 Write before deploy | User-facing docs with screenshots |
| GitHub repo | 🔲 Run git_setup.ps1 first | Create + push |
| Streamlit Cloud | 🔲 After GitHub push | Deploy |

---

## 7. Test Plan

### Unit Tests (pytest)

#### `tests/test_pde_pricer.py`
- **Test 1**: Reproduce Paper 1, Table 1 call probabilities (±0.001 tolerance)
  - S0=100, C=102, r=5%, σ=20%, q=1%, CDS=1%, T=1yr, monthly dates
  - Expected: p1=0.3767, p2=0.1435, …, p12=0.0096
- **Test 2**: Reproduce Paper 1 FD price = $98.39 (tolerance ±$0.10)
- **Test 3**: Reproduce Paper 1 closed-form continuous price = $99.54 (tolerance ±$0.10)
- **Test 4**: FD price converges to closed-form as observation frequency → ∞
- **Test 5**: `return_grid=True` returns 2D array of shape (N_S, N_τ), finite and non-negative

#### `tests/test_mc_pricers.py`
- **Test 1**: Standard MC converges to PDE price (N=1M paths, tolerance ±$0.50)
- **Test 2**: Survival MC converges to same price (N=100K paths, same tolerance)
- **Test 3**: Survival MC SE < Standard MC SE (same N) — variance reduction proof
- **Test 4**: Survival MC Delta is monotone in S0 (no sign flips across ε ∈ [0.01, 1.0])
- **Test 5**: `return_paths=True` returns list of path arrays for first 50 paths
- **Test 6**: Basket MC: two perfectly correlated assets ≈ single asset price

#### `tests/test_heston.py`
- **Test 1**: Heston char. fn → European call price matches Black-Scholes when γ=0
- **Test 2**: Calibrated surface: max abs error vs. market < 2 vol points
- **Test 3**: No branch-cut discontinuities (test continuity across u-range)

#### `tests/test_payoffs.py`
- **Test 1**: Barrier not hit → maturity payoff
- **Test 2**: Barrier hit at obs date 1 → Q1 payoff, discounted
- **Test 3**: Basket worst-of: payoff uses min(S1/Sref1, S2/Sref2)
- **Test 4**: Knock-in put: below protection barrier → S/S0 return
- **Test 5**: All 4 pre-configured securities produce finite, positive prices

---

## 8. Deployment

### Streamlit Community Cloud (Free)
1. Push code to public GitHub repo
2. Go to https://share.streamlit.io → "New app" → connect repo
3. Main file: `streamlit_app.py`
4. Deploy → shareable URL
5. Share URL with Keith

**Note**: `sample_data/` CSV files must be committed to the GitHub repo so Streamlit Cloud can read them. These are not sensitive — just public SPX market data.

**Cost**: $0. Streamlit Cloud free tier: 1 app, 1GB RAM, public repo required.

---

## 9. Key Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Market data | Static snapshots | Demo reliability; no API failures; reproducible |
| Data collection | yfinance scripts (offline) | Free; only runs once per snapshot 