# Next Session Handoff — AutoCallable Analytics Platform

**Last updated**: 2026-06-06 (end of Session 4)
**Status**: v0.3.1 — **ALL 66 TESTS PASSING** ✅ — Greeks page done

---

## Current State

The full application is written and all tests pass. The only remaining work before demo-ready is:

1. **Git commit** (do this first — requires Windows PowerShell)
2. **GitHub push + Streamlit Cloud deploy**
3. **Collect SPX options snapshots** (Mon June 9 – Wed June 11, during market hours)
4. **Write README.md** (for the GitHub repo / Streamlit app info page)

---

## IMMEDIATE FIRST STEP: Run git_setup.ps1

**From Windows PowerShell:**
```powershell
cd "C:\Users\rohit\OneDrive\Documents\Claude Apps\Autocallables"
.\git_setup.ps1
```

This script handles the broken `.git` state (the FUSE mount corrupted the objects/ directory). It removes the broken .git, inits fresh, and creates 3 logical commits.

After that, create a public GitHub repo named `autocallable-pricer` and push.

---

## Run the App (Local Test)

```bash
cd "C:\Users\rohit\OneDrive\Documents\Claude Apps\Autocallables"
pip install -r requirements.txt
streamlit run app/Home.py
```

The app will open at http://localhost:8501. It will show an error on the Vol Surface page if no snapshots exist — that's expected. All pricing pages work with synthetic parameters.

---

## Session Start Protocol (for next LLM)

1. Read `CLAUDE.md` (project root) — has all architecture decisions and dev standards
2. Read this file
3. Read `CHANGELOG.md` — understand what v0.1.0, v0.2.0, v0.3.0 contain
4. Run `python -m pytest tests/ -v` — should show 66/66 passing
5. State what you will do before doing it

---

## Files Completed (✅) vs Still TODO (🔲)

| Component | Status | Notes |
|-----------|--------|-------|
| `app/autocallable.py` | ✅ | Fixed truncation at line 489 |
| `app/pde_pricer.py` | ✅ | FD + closed-form + return_grid=True |
| `app/mc_standard.py` | ✅ | Standard MC + return_paths + track_convergence |
| `app/mc_survival.py` | ✅ | One-step survival MC (Alm et al. Paper 3) |
| `app/heston.py` | ✅ | Heston CF (Eq. 23 only) + calibration |
| `app/vol_surface.py` | ✅ | Implied vol + Dupire local vol |
| `app/data_loader.py` | ✅ | Snapshot loader + session labels |
| `app/components/securities.py` | ✅ | All 4 pre-configured securities |
| `app/components/sidebar.py` | ✅ | Shared Assumptions sidebar |
| `app/Home.py` | ✅ | Entry point + quick pricing |
| `app/pages/01_Vol_Surface.py` | ✅ | 3D IV + Heston overlay + Dupire |
| `app/pages/02_Pricer.py` | ✅ | MAIN PAGE — 3 methods, path animation, convergence |
| `app/pages/03_FDM_Visualization.py` | ✅ | FD grid heatmap + Greeks |
| `app/pages/04_Greeks.py` | ✅ | Delta stability, Vega stability, Delta smile, Methodology |
| `app/pages/05_Scenarios.py` | 🔲 | Not yet written (lower priority) |
| `tests/` (66 tests) | ✅ | All passing |
| `scripts/collect_snapshot.py` | ✅ | Run during market hours Mon–Wed |
| `sample_data/` | 🔲 | Need 3-4 SPX snapshots (Mon–Wed June 9–11) |
| Git commits | 🔲 | Run git_setup.ps1 from Windows |
| GitHub repo | 🔲 | Create + push after git setup |
| Streamlit Cloud deploy | 🔲 | After GitHub push |
| `README.md` | 🔲 | Optional but good for Keith |

---

## FUSE Mount Warning (CRITICAL for future sessions)

The OneDrive FUSE mount has two dangerous behaviors:
1. **Write truncation**: Large file writes via Write/Edit tool may be silently truncated. Use `python3 -c "open(...).write(...)"` or bash heredoc for critical files.
2. **Stale pyc**: `.pyc` files on the FUSE mount are read-only. After any source file edit, run `touch <file>.py` to force Python to recompile from source.
3. **git objects**: git's `objects/` directory is cloud-only and not downloadable via FUSE. **Do not run git commands from the Linux sandbox** — always use Windows PowerShell for git.

---

## Streamlit Deploy Instructions

After git is set up and pushed to GitHub:

1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect GitHub repo `rohitpittu/autocallable-pricer`
4. Main file: `app/Home.py`
5. Click Deploy → get shareable URL
6. Share URL with Keith Loggie (MerQube)

**Important**: `sample_data/` CSV files must be committed to GitHub (not .gitignored) so Streamlit Cloud can read them. They are public SPX market data — no sensitivity.

---

## Priority Order for Day 7 (June 11)

1. Run `git_setup.ps1` → commit → push to GitHub
2. Collect snapshots Mon/Tue/Wed during market hours (`python scripts/collect_snapshot.py`)
3. Test app locally with real snapshots (`streamlit run app/Home.py`)
4. Deploy to Streamlit Cloud
5. Write README.md (optional but professional)
6. Page 5 (Scenarios) — nice-to-have, not blocking; Greeks is now done

---

## Validated Prices (for demo — know these cold)

| Product | Method | Price | Notes |
|---------|--------|-------|-------|
| Phoenix Autocall | FD (N=200) | $951.28 | S0=5312, σ=20%, r=4.5%, q=1.4% |
| Phoenix Autocall | MC (N=10K) | ~$950.50 ± $0.87 | Agrees with FD within 1σ |
| Phoenix Autocall | Survival MC | Similar price, ~2x lower std_err | Variance reduction confirmed |
| Phoenix call probs | Analytical | 51%, 25%, 12%, ... (sums to 99.7%) | Near-certain call within 2yr at 20% vol |

