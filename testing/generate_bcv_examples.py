"""
generate_bcv_examples.py -- run BCV dimension estimation across a factorial set
of synthetic examples and persist *all* results for later reporting.

A core factorial is crossed over several seeds, plus a dedicated **Q-sparsity
(Q_beta)** sweep that holds the data fixed and varies ICQF's ``Q_beta`` penalty:

- **n**         number of samples (rows), in the 200-500 range
- **true_dim**  ground-truth embedding dimension
- **sigma**     additive observation-noise level
- **Q_beta**    L1 penalty on the item loadings Q (sparsity knob); larger =>
                sparser Q. Swept separately at a fixed (n, true_dim, sigma).

For every example the script runs the improved BCV (PA-centered search grid +
1-SE selection, ADMM using all cores but one) and saves:

  data/<example_id>.npz                        the matrix M + generative params
                                               (n, p, true_dim, sigma, q_density,
                                               missing_ratio, seed, q_beta, snr)
  results/examples/<example_id>__results.csv   per-(dim, repeat, fold) MSE rows
  results/examples/<example_id>__summary.csv   per-dim mean / std / SE curve
  results/bcv_examples_index.csv               one row per example (config + picks)
  results/bcv_examples_curves.csv              tidy long table of every curve point
  results/bcv_examples.pkl                      everything (configs, curves, PA spectra)

The companion notebook ``report_bcv_examples.ipynb`` loads these and visualizes
them.  Run:

    python generate_bcv_examples.py            # full factorial
    python generate_bcv_examples.py --quick     # tiny/fast smoke run
"""

from __future__ import annotations

import argparse
import itertools
import pickle
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
EXAMPLES_DIR = RESULTS_DIR / "examples"
DATA_DIR = Path(__file__).parent / "data"

ICQF_KWARGS = dict(W_beta=0.01, Q_beta=0.01, regularizer=1, max_iter=200)

# Generative parameters stored alongside each example matrix in its .npz.
NPZ_META_KEYS = ["n", "p", "true_dim", "sigma", "q_density",
                 "missing_ratio", "seed", "q_beta", "snr"]


def save_example_npz(path: Path, M_df: pd.DataFrame, meta: dict) -> None:
    """
    Write one example to ``path`` as a compressed .npz.

    Stores the data matrix ``M`` (NaN = unobserved), its row/column labels, and
    the generative parameters in ``NPZ_META_KEYS``. Reloadable without
    ``allow_pickle`` (labels are saved as unicode arrays, scalars as 0-d arrays)::

        d = np.load(path)
        M = d["M"]; n = int(d["n"]); sigma = float(d["sigma"])
    """
    np.savez_compressed(
        path,
        M=M_df.to_numpy(dtype=np.float64),
        sample_index=np.asarray(M_df.index.astype(str), dtype="U"),
        feature_index=np.asarray(M_df.columns.astype(str), dtype="U"),
        **{k: np.asarray(meta[k]) for k in NPZ_META_KEYS},
    )


def build_design(quick: bool) -> dict:
    """Return the design: a core (n x true_dim x sigma) factorial plus a Q-sparsity
    (Q_beta) sweep, with shared fixed settings."""
    if quick:
        return dict(
            p=40,
            n_values=[200, 350],
            true_dim_values=[3, 6],
            sigma_values=[0.02, 0.10],
            q_beta_default=0.01,
            q_beta_values=[0.001, 0.1],
            qbeta_base=dict(n=200, true_dim=6, sigma=0.05),
            seeds=[0],
            J=5,
            repeat=1,
            max_iter=150,
            pa_window=4,
        )
    return dict(
        p=60,
        n_values=[200, 350, 500],
        true_dim_values=[3, 6, 10],
        sigma_values=[0.02, 0.05, 0.10],
        q_beta_default=0.01,
        # Q-sparsity search: small beta_Q => dense Q, large beta_Q => sparser Q.
        q_beta_values=[0.001, 0.01, 0.05, 0.1, 0.5],
        qbeta_base=dict(n=350, true_dim=6, sigma=0.05),
        seeds=[0, 1],
        J=10,
        repeat=2,
        max_iter=200,
        pa_window=5,
    )


def build_examples(design: dict) -> list:
    """Materialize the list of example configs (dicts) from the design."""
    examples = []
    qd = design["q_beta_default"]
    # Core factorial over (n, true_dim, sigma), all at the default Q_beta.
    for n, d, s, seed in itertools.product(
        design["n_values"], design["true_dim_values"],
        design["sigma_values"], design["seeds"],
    ):
        examples.append(dict(group="factorial", n=n, true_dim=d, sigma=s,
                             q_beta=qd, seed=seed))
    # Q-sparsity sweep: hold (n, true_dim, sigma) at a baseline, vary Q_beta.
    qb = design["qbeta_base"]
    for q_beta, seed in itertools.product(design["q_beta_values"], design["seeds"]):
        examples.append(dict(group="qbeta_sweep", n=qb["n"], true_dim=qb["true_dim"],
                             sigma=qb["sigma"], q_beta=q_beta, seed=seed))
    return examples


def run_example(
    *,
    n: int,
    p: int,
    true_dim: int,
    sigma: float,
    q_beta: float,
    seed: int,
    J: int,
    repeat: int,
    max_iter: int,
    pa_window: int,
    n_threads: int,
) -> dict:
    """Generate one matrix, run BCV, return a bundle dict with all artifacts."""
    M_df, meta = make_lowrank_df(n, p, true_dim, sigma=sigma, seed=seed)
    icqf_kwargs = {**ICQF_KWARGS, "max_iter": max_iter, "Q_beta": q_beta}
    meta["q_beta"] = float(q_beta)

    t0 = time.perf_counter()
    results_df, summary_df, best_k_1se = estimate_dimension(
        M_df,
        dimension_list=None,
        J=J,
        repeat=repeat,
        random_state=1000 + seed,
        icqf_kwargs=icqf_kwargs,
        verbose=False,
        n_threads=n_threads,
        use_parallel_analysis=True,
        pa_window=pa_window,
        use_1se=True,
    )
    elapsed = time.perf_counter() - t0

    sel = dict(summary_df.attrs.get("selection_info", {}))
    pa = dict(summary_df.attrs.get("pa_info", {}))

    return {
        "meta": meta,
        "M_df": M_df,                       # kept for the .npz dump; popped before pickling
        "config": {
            "J": J, "repeat": repeat, "max_iter": max_iter,
            "pa_window": pa_window, "n_threads": n_threads, "seed": seed,
        },
        "results_df": results_df,
        "summary_df": summary_df,
        "selection_info": sel,
        "pa_info": pa,
        "best_k_1se": int(best_k_1se),
        "best_k_argmin": int(sel.get("argmin_k", -1)),
        "pa_dim": int(pa.get("pa_dim", -1)),
        "bcv_time_sec": float(elapsed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny/fast smoke run")
    ap.add_argument("--threads", type=int, default=None,
                    help="BLAS threads for ADMM (default: cpu_count - 1)")
    args = ap.parse_args()

    n_threads = args.threads if args.threads is not None else default_n_threads()
    design = build_design(args.quick)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    examples = build_examples(design)
    print(f"ADMM threads = {n_threads}   examples = {len(examples)}   quick = {args.quick}")

    bundles = []
    index_rows = []
    curve_rows = []

    for idx, ex in enumerate(examples):
        n, true_dim, sigma = ex["n"], ex["true_dim"], ex["sigma"]
        q_beta, seed, group = ex["q_beta"], ex["seed"], ex["group"]
        example_id = (f"ex{idx:03d}_n{n}_d{true_dim}_s{sigma:.2f}"
                      f"_qb{q_beta:g}_seed{seed}")
        print(f"[{idx + 1:>3}/{len(examples)}] {example_id}", end="  ", flush=True)

        bundle = run_example(
            n=n, p=design["p"], true_dim=true_dim, sigma=sigma, q_beta=q_beta,
            seed=seed, J=design["J"], repeat=design["repeat"],
            max_iter=design["max_iter"], pa_window=design["pa_window"],
            n_threads=n_threads,
        )
        bundle["example_id"] = example_id
        bundle["group"] = group

        # Save the example matrix + generative params as a self-describing .npz,
        # then drop M_df from the bundle so the master pickle stays lean.
        save_example_npz(DATA_DIR / f"{example_id}.npz", bundle.pop("M_df"), bundle["meta"])
        bundles.append(bundle)

        # Persist per-example CSVs.
        bundle["results_df"].to_csv(EXAMPLES_DIR / f"{example_id}__results.csv", index=False)
        bundle["summary_df"].to_csv(EXAMPLES_DIR / f"{example_id}__summary.csv", index=False)

        m = bundle["meta"]
        index_rows.append({
            "example_id": example_id, "group": group,
            "n": m["n"], "p": m["p"], "true_dim": m["true_dim"], "sigma": m["sigma"],
            "q_beta": m["q_beta"],
            "missing_ratio": m["missing_ratio"], "snr": m["snr"], "seed": seed,
            "pa_dim": bundle["pa_dim"],
            "best_k_argmin": bundle["best_k_argmin"],
            "best_k_1se": bundle["best_k_1se"],
            "err_argmin": bundle["best_k_argmin"] - m["true_dim"],
            "err_1se": bundle["best_k_1se"] - m["true_dim"],
            "correct_argmin": bundle["best_k_argmin"] == m["true_dim"],
            "correct_1se": bundle["best_k_1se"] == m["true_dim"],
            "bcv_time_sec": bundle["bcv_time_sec"],
        })

        # Tidy long curve table (one row per dimension per example).
        s = bundle["summary_df"]
        for _, srow in s.iterrows():
            curve_rows.append({
                "example_id": example_id, "group": group,
                "n": m["n"], "p": m["p"], "true_dim": m["true_dim"], "sigma": m["sigma"],
                "q_beta": m["q_beta"], "seed": seed,
                "n_components": int(srow["n_components"]),
                "test_mse_mean": srow["test_mse_mean"],
                "test_mse_se": srow["test_mse_se"],
                "test_mse_se_paired": srow.get("test_mse_se_paired", srow["test_mse_se"]),
                "test_mse_se_raw": srow.get("test_mse_se_raw", srow["test_mse_se"]),
                "test_mse_std": srow["test_mse_std"],
                "train_mse_mean": srow["train_mse_mean"],
            })

        print(f"-> true={m['true_dim']} pa={bundle['pa_dim']} "
              f"argmin={bundle['best_k_argmin']} 1se={bundle['best_k_1se']} "
              f"{bundle['bcv_time_sec']:.2f}s")

    # Master tables.
    index_df = pd.DataFrame(index_rows)
    curves_df = pd.DataFrame(curve_rows)
    index_df.to_csv(RESULTS_DIR / "bcv_examples_index.csv", index=False)
    curves_df.to_csv(RESULTS_DIR / "bcv_examples_curves.csv", index=False)

    with open(RESULTS_DIR / "bcv_examples.pkl", "wb") as f:
        pickle.dump({"design": design, "bundles": bundles}, f)

    print(f"\nWrote index ({len(index_df)} examples) and curves "
          f"({len(curves_df)} points) to {RESULTS_DIR}")
    print("\n=== recovery summary ===")
    print(f"argmin correct: {index_df['correct_argmin'].mean():.1%}   "
          f"1-SE correct: {index_df['correct_1se'].mean():.1%}")
    print(f"mean |err| argmin: {index_df['err_argmin'].abs().mean():.2f}   "
          f"1-SE: {index_df['err_1se'].abs().mean():.2f}")


if __name__ == "__main__":
    main()
