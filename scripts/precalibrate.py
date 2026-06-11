"""
scripts/precalibrate.py
========================
Pre-calibrate Heston and Bates parameters for every snapshot in sample_data/
and save the results to sample_data/calibrations_cache.json.

WHY PRE-CALIBRATE:
    Calibration is slow (15–60 s per snapshot for Heston, 30–90 s for Bates).
    With 10 snapshots and two models, interactive calibration on every page load
    is impractical.  Running this script once produces a cache that the sidebar
    auto-loads when a snapshot is selected, giving instant calibrated parameters.

USAGE:
    python scripts/precalibrate.py            # calibrate all snapshots
    python scripts/precalibrate.py --force    # re-calibrate even if in cache

The script saves incrementally after each snapshot so partial results are not
lost if it is interrupted.  Re-running without --force skips already-cached
snapshots so interrupted runs resume rather than restarting.
"""

import argparse, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.data_loader import list_available_snapshots, load_snapshot, get_spot_price, get_rfr
from app.vol_surface import VolSurface
from app.heston import HestonModel, calibrate_bates

DATA_DIR   = os.path.join(ROOT, "sample_data")
CACHE_PATH = os.path.join(DATA_DIR, "calibrations_cache.json")
Q_SPX = 0.014  # SPX long-run dividend yield — not stored in snapshot files


def _calibrate_snapshot(snap_df, S0: float, r: float, q: float) -> dict:
    """
    Run Heston then Bates calibration for one snapshot.

    Returns:
        {"heston": {...}, "bates": {...}}
    """
    vs = VolSurface(snap_df, S0=S0, r=r, q=q)

    t1 = time.time()
    hm = HestonModel(S0=S0, r=r, q=q)
    h  = hm.calibrate(vs)
    print(
        f"    Heston  {time.time()-t1:5.0f}s  "
        f"RMSE={h['rmse_vol_pts']:.2f}vol-pts  "
        f"v0={h['v0']:.4f}  kappa={h['kappa']:.2f}  "
        f"theta={h['theta']:.4f}  gamma={h['gamma']:.2f}  rho={h['rho']:.2f}"
    )

    t2 = time.time()
    b  = calibrate_bates(snap_df, S0=S0, r=r, q=q, heston_init=h)
    print(
        f"    Bates   {time.time()-t2:5.0f}s  "
        f"RMSE={b['rmse_vol_pts']:.2f}vol-pts  "
        f"lam={b['lam']:.2f}  mu_J={b['mu_J']:.3f}  sig_J={b['sig_J']:.3f}"
    )

    return {"heston": h, "bates": b}


def main():
    parser = argparse.ArgumentParser(description="Pre-calibrate Heston/Bates for all snapshots.")
    parser.add_argument("--force", action="store_true", help="Re-calibrate even if already in cache.")
    args = parser.parse_args()

    snaps = list_available_snapshots(DATA_DIR)
    if not snaps:
        print("No snapshot files found. Exiting.")
        return

    # Load existing cache (incremental resume support)
    if os.path.exists(CACHE_PATH) and not args.force:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded existing cache: {len(cache)} entries. Use --force to re-calibrate all.")
    else:
        cache = {}

    total_start = time.time()
    n_done = 0

    for snap in snaps:
        key   = snap["key"]
        label = snap["label"]

        # Skip if already in cache (unless --force)
        if not args.force and key in cache and cache[key].get("heston") and cache[key].get("bates"):
            print(f"  SKIP  {label}  (cached)")
            continue

        print(f"\n{'='*64}")
        print(f"  {label}  [{key}]")
        print(f"{'='*64}")

        try:
            snap_df = load_snapshot(key, DATA_DIR)
            S0      = get_spot_price(snap_df)
            r       = get_rfr(snap_df)
            print(f"    S0={S0:.1f}  r={r*100:.2f}%  q={Q_SPX*100:.1f}%  rows={len(snap_df)}")

            t0           = time.time()
            cache[key]   = _calibrate_snapshot(snap_df, S0=S0, r=r, q=Q_SPX)
            elapsed      = time.time() - t0
            print(f"    Snapshot total: {elapsed:.0f}s")
            n_done      += 1

        except Exception as exc:
            import traceback
            print(f"  ERROR calibrating {key}: {exc}")
            traceback.print_exc()
            cache[key] = {}

        # Incremental save after each snapshot — partial results are preserved
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, default=float)

    total_elapsed = time.time() - total_start
    success_count = sum(1 for v in cache.values() if v.get("heston") and v.get("bates"))

    print(f"\n{'='*64}")
    print(f"Pre-calibration complete: {success_count}/{len(snaps)} snapshots calibrated.")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Cache saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
