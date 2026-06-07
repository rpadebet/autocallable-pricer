# AutoCallable Analytics Platform

A pricing and analytics platform for autocallable structured products, built as a technical demonstration.

**Live demo**: [https://autocallable-pricer.streamlit.app](https://autocallable-pricer.streamlit.app) *(link active after deploy)*

---

## What This Is

Autocallables are among the most common structured products issued by banks — a barrier option with a conditional coupon and an automatic early redemption feature. Pricing them correctly requires three things this app demonstrates end-to-end:

1. **A realistic volatility surface** — implied vols from real SPX options data, Heston calibration, and Dupire local vol
2. **Multiple pricing methods** — finite difference PDE, standard Monte Carlo, and one-step survival Monte Carlo — with visual proof that they converge
3. **Stable Greeks** — the key practical problem: standard MC gives unreliable Delta and Vega near barriers; one-step survival MC resolves this analytically

The math comes from three papers (in the `docs/` folder):

| Paper | Key Contribution |
|-------|-----------------|
| Deng, Mallett, McCann (2011) | PDE framework, closed-form continuous autocall, call probability term structure |
| Haugh, Columbia IEOR (2013) | Heston SDE + characteristic function (Eq. 23), Dupire local vol |
| Alm, Harrach, Harrach, Keller, JCF (2013) | One-step survival MC, GHK importance sampling, stable Greeks |

---

## App Pages

### 1 · Vol Surface
Loads a pre-collected SPX options snapshot, filters bad ticks, computes Black-Scholes implied vols for each (K, T) pair, and displays an interactive 3D surface. Overlays the calibrated Heston model surface to show fit quality. Computes the Dupire local vol surface side-by-side.

### 2 · Pricer *(start here)*
The centerpiece. Select from four pre-configured autocallable structures and price using all three methods simultaneously:

- **FD/PDE** — explicit finite difference on the log-transformed heat equation (Paper 1 §2.2). Outputs call probability at each observation date.
- **Standard MC** — GBM simulation with antithetic variates. Animated path chart shows how individual trajectories interact with the call and protection barriers.
- **One-Step Survival MC** — no path ever crosses the barrier stochastically; crossings handled via truncated normal sampling. Convergence chart shows tighter confidence intervals at the same path count.

### 3 · FDM Visualization
Makes the finite difference algorithm visible. Heatmap of V(S, t) backward from maturity, 3D value surface with autocall boundary, time-slice animation stepping backward from T → 0.

### 4 · Greeks
Demonstrates the key practical result of Paper 3. Standard MC Delta is noisy and can change sign across random seeds at small bump sizes. Survival MC Delta is stable across all bump sizes and seeds — because the payoff is a smooth function of S₀ by construction (no path stochastically crosses the barrier). Side-by-side plots at 10 bump sizes × 8 seeds make this impossible to miss.

---

## Pre-Configured Securities

| Name | Structure | Underlying | Call Barrier | Coupon | Protection |
|------|-----------|------------|--------------|--------|------------|
| Phoenix Autocall | Standard | SPX | 100% | 8% p.a. | 75% knock-in |
| Worst-Of Autocall | Worst-of basket | SPX / NDX / RUT | 100% | 12% p.a. | 70% knock-in |
| Step-Down Barrier | Declining barrier | SPX | 100→95→90% | 10% p.a. | 75% knock-in |
| Digital Autocall | Binary payoff | SPX | 105% | $50 digital | 80% capital protected |

---

## Market Data

The app uses **pre-collected static snapshots** of SPX options chains (in `sample_data/`). There are no live API calls at runtime — this makes every run reproducible and eliminates demo failures. Snapshots were collected via `scripts/collect_snapshot.py` using yfinance during market hours.

---

## Run Locally

```bash
git clone https://github.com/rpadebet/autocallable-pricer.git
cd autocallable-pricer
pip install -r requirements.txt
streamlit run app/Home.py
```

Open http://localhost:8501. Start on the **Pricer** page.

**Requirements**: Python 3.11+, ~500MB RAM for full MC runs.

---

## Project Structure

```
autocallable-pricer/
├── app/
│   ├── Home.py                    # Entry point + quick pricing
│   ├── pages/
│   │   ├── 01_Vol_Surface.py      # Heston calibration + Dupire
│   │   ├── 02_Pricer.py           # Three pricing methods + path animation
│   │   ├── 03_FDM_Visualization.py
│   │   ├── 04_Greeks.py           # Stable vs noisy differentiation
│   │   └── 05_Scenarios.py        # Payoff diagrams, what-if, value surface
│   ├── components/
│   │   ├── sidebar.py             # Shared assumptions sidebar
│   │   └── securities.py          # Pre-configured product definitions
│   ├── autocallable.py            # Product dataclass + payoff logic
│   ├── pde_pricer.py              # Finite difference pricer
│   ├── mc_standard.py             # Standard Monte Carlo
│   ├── mc_survival.py             # One-step survival MC (Alm et al.)
│   ├── heston.py                  # Heston model + calibration
│   ├── vol_surface.py             # Implied vol + Dupire local vol
│   └── data_loader.py             # Snapshot loader
├── sample_data/                   # Pre-collected SPX options CSVs
├── scripts/collect_snapshot.py    # Run during market hours to add snapshots
├── tests/                         # 66 unit tests (pytest)
└── requirements.txt
```

---

## Tests

```bash
python -m pytest tests/ -v
```

66 tests covering: FD prices vs Paper 1 Table 1, MC convergence, Heston characteristic function (branch-cut safety), payoff logic for all four product structures, variance reduction proof (Survival MC SE < Standard MC SE at same N).

---

## Key Design Decisions

**Static market data** — The app demonstrates what you can do with real options data, not how to build a data pipeline. Pre-collecting snapshots makes the demo reliable and each run fully reproducible.

**Three pricing methods shown together** — FD gives the analytical baseline; Standard MC shows the naive approach; Survival MC shows the improvement. Convergence charts on the same axes make the variance reduction proof visual rather than just claimed.

**No hardcoded parameters** — Every model parameter (σ, r, q, N paths, grid size, Heston params) lives in the shared sidebar and threads through to all pages. Changing one value immediately reprices everything.

---

*Built by Rohit Padebettu — June 2026*
