"""
test_bcv_capability.py -- probe how well (and how fast) BCV recovers the true
ICQF embedding dimension.

Four axes are swept, one at a time, around a common baseline:

1. **Timing baseline** -- wall-clock for a single BCV sweep, run with all cores
   but one (``n_threads = cpu_count - 1``) for the ADMM solver. One core is left
   free so the machine does not freeze.
2. **n (number of samples)** -- how recovery and runtime scale as rows grow.
3. **true dimension** -- including the hard regime where ``true_dim`` is a large
   fraction of ``min(n, p)`` (i.e. ``n`` close to the number of items).
4. **sigma (noise level)** -- how recovery degrades as observation noise grows.

For every condition we record the wall-clock time, the parallel-analysis guess,
and the BCV estimate under both the naive argmin and the 1-SE rule, then write a
tidy CSV to ``results/bcv_capability.csv``. ``generate_bcv_examples.py`` produces
the richer per-(dimension) curves; this script is the scaling/timing probe.

Run:
    python test_bcv_capability.py                 # full sweep
    python test_bcv_capability.py --quick          # small/fast smoke run
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# This script lives in testing/; add the repo root to sys.path so `import icqf`
# resolves whether run from the repo root or from inside testing/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bcv_simdata import make_lowrank_df
from icqf import default_n_threads, estimate_dimension

RESULTS_DIR = Path(__file__).parent / "results"


# ICQF / BCV settings shared by every condition.
ICQF_KWARGS = dict(W_beta=0.01, Q_beta=0.01, regularizer=1, max_iter=200)
BCV_J = 10
BCV_REPEAT = 2
PA_WINDOW = 4
BASE_SEED = 12345


def run_one(
    *,
    n: int,
    p: int,
    true_dim: int,
    sigma: float,
    n_threads: int,
    seed: int,
    quick: bool,
) -> dict:
    """Generate one matrix, time a full BCV sweep, return a result row."""
    M_df, meta = make_lowrank_df(n, p, true_dim, sigma=sigma, seed=seed)

    j = 5 if quick else BCV_J
    repeat = 1 if quick else BCV_REPEAT
    max_iter = 150 if quick else ICQF_KWARGS["max_iter"]
    icqf_kwargs = {**ICQF_KWARGS, "max_iter": max_iter}

    t0 = time.perf_counter()
    _results_df, summary_df, best_k_1se = estimate_dimension(
        M_df,
        dimension_list=None,            # PA-centered grid
        J=j,
        repeat=repeat,
        random_state=seed,
        icqf_kwargs=icqf_kwargs,
        verbose=False,
        n_threads=n_threads,
        use_parallel_analysis=True,
        pa_window=PA_WINDOW,
        use_1se=True,
    )
    elapsed = time.perf_counter() - t0

    sel = summary_df.attrs.get("selection_info", {})
    pa = summary_df.attrs.get("pa_info", {})
    grid = summary_df["n_components"].tolist()
    n_fits = len(grid) * j * repeat

    return {
        **meta,
        "n_threads": n_threads,
        "J": j,
        "repeat": repeat,
        "max_iter": max_iter,
        "grid_lo": int(min(grid)),
        "grid_hi": int(max(grid)),
        "pa_dim": int(pa.get("pa_dim", -1)),
        "est_dim_argmin": int(sel.get("argmin_k", -1)),
        "est_dim_1se": int(best_k_1se),
        "abs_err_1se": int(abs(best_k_1se - true_dim)),
        "correct_1se": bool(best_k_1se == true_dim),
        "n_fits": n_fits,
        "bcv_time_sec": float(elapsed),
        "time_per_fit_sec": float(elapsed / max(1, n_fits)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small/fast smoke run")
    ap.add_argument("--threads", type=int, default=None,
                    help="BLAS threads for ADMM (default: cpu_count - 1)")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "bcv_capability.csv"))
    args = ap.parse_args()

    n_threads = args.threads if args.threads is not None else default_n_threads()
    quick = args.quick
    print(f"CPU count = {os.cpu_count()}   ADMM threads = {n_threads}   quick = {quick}")

    # Baseline condition (also the center of each sweep). n stays in [200, 500].
    base = dict(n=300, p=80, true_dim=8, sigma=0.05)
    if quick:
        base = dict(n=200, p=60, true_dim=5, sigma=0.05)

    # --- define the sweeps ---
    if quick:
        n_grid = [200, 350, 500]
        dim_grid = [3, 5, 8]
        sigma_grid = [0.02, 0.05, 0.10]
        near_p = 300                              # items count for the n~=p regime
        near_p_n_grid = [200, 350, 500]
    else:
        n_grid = [200, 300, 400, 500]
        dim_grid = [2, 5, 10, 20, 40]             # 40 ~ min(n, p=80) regime
        sigma_grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
        # "n ~= items": p comparable to n, so the crossing happens within [200, 500].
        near_p = 300
        near_p_n_grid = [200, 300, 400, 500]

    conditions = []

    # 1) timing baseline (single point, repeated label)
    conditions.append({"sweep": "baseline", **base})

    # 2) vary n
    for n in n_grid:
        conditions.append({"sweep": "vary_n", **{**base, "n": n}})

    # 3) vary true dimension (p fixed; large dim => true_dim near min(n, p))
    for d in dim_grid:
        if d <= min(base["n"], base["p"]):
            conditions.append({"sweep": "vary_true_dim", **{**base, "true_dim": d}})

    # 3b) the "n ~= item" regime: p is large (== near_p) and n crosses it.
    for n in near_p_n_grid:
        if n >= 2 * base["true_dim"]:
            conditions.append({"sweep": "n_near_p", **{**base, "n": n, "p": near_p}})

    # 4) vary sigma (noise level)
    for s in sigma_grid:
        conditions.append({"sweep": "vary_sigma", **{**base, "sigma": s}})

    rows = []
    for i, cond in enumerate(conditions):
        sweep = cond.pop("sweep")
        seed = BASE_SEED + i
        print(f"[{i + 1:>2}/{len(conditions)}] {sweep:<14} "
              f"n={cond['n']:<5} p={cond['p']:<4} true_dim={cond['true_dim']:<3} "
              f"sigma={cond['sigma']:<5}", end="  ", flush=True)
        row = run_one(n_threads=n_threads, seed=seed, quick=quick, **cond)
        row["sweep"] = sweep
        rows.append(row)
        print(f"-> est_1se={row['est_dim_1se']} (pa={row['pa_dim']}, "
              f"argmin={row['est_dim_argmin']})  {row['bcv_time_sec']:.2f}s")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")

    # quick textual report
    print("\n=== recovery accuracy by sweep ===")
    acc = df.groupby("sweep").agg(
        n_cond=("correct_1se", "size"),
        acc_1se=("correct_1se", "mean"),
        mean_abs_err=("abs_err_1se", "mean"),
        mean_time_s=("bcv_time_sec", "mean"),
    )
    print(acc.to_string())


if __name__ == "__main__":
    main()
