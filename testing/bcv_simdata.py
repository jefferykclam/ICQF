"""
bcv_simdata.py -- synthetic low-rank data generators for BCV dimension studies.

Shared by ``test_bcv_capability.py`` and ``generate_bcv_examples.py``. Every
generator returns a ``(samples x features)`` pandas DataFrame in ``[0, 1]`` with a
*known* true rank, matching ICQF's interval convention (NaN = unobserved).

The base model is the same one used in ``demo_ICQF.ipynb``:

    M = W_true @ Q_true.T          # non-negative, low-rank, Q sparse
    M = M / M.max()                # scale into [0, 1]
    M = M + N(0, sigma^2)          # additive observation noise (the "noise level")
    M = clip(M, 0, 1)              # back into the valid interval

``sigma`` is the knob the BCV study calls the *noise level*; ``true_dim`` is the
ground-truth embedding dimension we hope BCV recovers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def make_lowrank_df(
    n: int,
    p: int,
    true_dim: int,
    *,
    sigma: float = 0.05,
    q_density: float = 0.5,
    missing_ratio: float = 0.0,
    seed: int = 0,
    return_factors: bool = False,
):
    """
    Build a non-negative low-rank matrix with a known true rank.

    Parameters
    ----------
    n, p:
        Number of samples (rows) and features/items (columns).
    true_dim:
        Ground-truth embedding dimension (rank of the noiseless signal).
    sigma:
        Std of the additive Gaussian observation noise (the "noise level").
    q_density:
        Fraction of non-zero entries in the sparse item-loading matrix ``Q``.
    missing_ratio:
        Fraction of entries randomly knocked out to NaN (unobserved).
    seed:
        RNG seed.
    return_factors:
        If True, also return the ground-truth ``W_true`` and ``Q_true`` factor
        DataFrames (the noiseless signal is ``W_true @ Q_true.T`` before scaling).

    Returns
    -------
    M_df, meta
        (default) the ``(n x p)`` DataFrame in ``[0, 1]`` (NaN where unobserved)
        and a dict describing the generating parameters.
    M_df, meta, W_df, Q_df
        when ``return_factors=True``: additionally the true factor DataFrames,
        scaled by the same factor applied to the signal so ``W_df @ Q_df.T`` is
        the noiseless, ``[0, 1]``-scaled signal.
    """
    if true_dim < 1:
        raise ValueError("true_dim must be >= 1")
    if true_dim > min(n, p):
        raise ValueError(f"true_dim={true_dim} exceeds min(n, p)={min(n, p)}")

    rng = np.random.RandomState(seed)
    W = rng.rand(n, true_dim)

    # Sparse, non-negative item loadings; guarantee every factor and every item
    # carries at least one non-zero so the effective rank really is ``true_dim``.
    mask = rng.rand(p, true_dim) < q_density
    for k in range(true_dim):
        if not mask[:, k].any():
            mask[rng.randint(p), k] = True
    for j in range(p):
        if not mask[j, :].any():
            mask[j, rng.randint(true_dim)] = True
    Q = mask * rng.rand(p, true_dim)

    signal = W @ Q.T
    scale = signal.max()
    signal = signal / scale
    # Fold the [0, 1] scaling into W so that W_scaled @ Q.T == scaled signal.
    W = W / scale
    M = signal + rng.randn(n, p) * sigma
    M = np.clip(M, 0.0, 1.0)

    if missing_ratio > 0.0:
        n_missing = int(round(missing_ratio * n * p))
        if n_missing > 0:
            flat = rng.choice(n * p, size=n_missing, replace=False)
            ii, jj = np.unravel_index(flat, (n, p))
            M[ii, jj] = np.nan
            # Repair any fully-NaN row/column (ICQF forbids them).
            for i in np.where(np.isnan(M).all(axis=1))[0]:
                M[i, rng.randint(p)] = float(signal[i, rng.randint(p)])
            for jc in np.where(np.isnan(M).all(axis=0))[0]:
                M[rng.randint(n), jc] = float(signal[rng.randint(n), jc])

    M_df = pd.DataFrame(
        M,
        index=[f"subject_{i:04d}" for i in range(n)],
        columns=[f"item_{j:03d}" for j in range(p)],
    )
    meta = {
        "n": int(n),
        "p": int(p),
        "true_dim": int(true_dim),
        "sigma": float(sigma),
        "q_density": float(q_density),
        "missing_ratio": float(missing_ratio),
        "seed": int(seed),
        "snr": float(signal.var() / (sigma ** 2)) if sigma > 0 else float("inf"),
    }
    if return_factors:
        factor_cols = [f"factor_{k + 1}" for k in range(true_dim)]
        W_df = pd.DataFrame(W, index=M_df.index, columns=factor_cols)
        Q_df = pd.DataFrame(Q, index=M_df.columns, columns=factor_cols)
        return M_df, meta, W_df, Q_df
    return M_df, meta
