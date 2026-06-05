"""
bcv.py  -  Bi-cross-validation (BCV) for ICQF embedding-dimension selection.

This is the *standalone, single-process* version of dimension estimation: every
fold is fit inline in the current Python process. There is no cluster / swarm
dispatch here. For a single survey-sized matrix the full sweep finishes in
seconds-to-minutes, so a serial loop is the simplest correct thing.

The ADMM solver inside ICQF leans on multi-threaded BLAS for its dense matrix
products. By default we let it use *all cores but one* (one core left free so
the machine stays responsive); see ``default_n_threads`` and the ``n_threads``
argument of ``estimate_dimension``.

Idea
----
ICQF treats NaN as "unobserved". To score a candidate rank ``k`` we:

1. Split the *observed* entries of ``M_df`` into J folds.
2. For each fold, mask that fold's entries to NaN (the "test" entries), fit
   ICQF with rank ``k`` on the remaining ("train") entries, and measure the
   reconstruction error  ``(M - W Qᵀ)²``  on the held-out test entries.
3. Average the held-out (test) error across folds (and repeats). The rank that
   minimizes held-out error -- or the elbow of the curve -- is the estimate.

Holding out entries the model never saw is what makes the curve informative:
train error keeps falling as ``k`` grows, but test error bottoms out and then
rises once extra components start fitting noise.

Two refinements (both optional, both on by default)
---------------------------------------------------
- **Parallel analysis (PA) to center the search grid.** A permutation parallel
  analysis on the data matrix gives a cheap first guess of the true dimension.
  When no explicit ``dimension_list`` is supplied we center a search *window* on
  that guess instead of sweeping a wide, arbitrary range. See
  ``parallel_analysis`` and ``grid_from_parallel_analysis``.
- **1-SE rule to pick the rank.** The literal argmin of a noisy CV curve jumps
  around in flat regions. The 1-SE rule (Breiman, Friedman, Olshen & Stone 1984,
  *CART*) instead takes the *simplest* model whose mean CV loss is within one
  standard error of the minimum. See ``select_dimension``.

Public API
----------
- make_bcv_masks(M_df, J, ...)           -> list of (train_mask_df, test_mask_df)
- apply_train_mask(M_df, train_mask_df)  -> M_df with test/unobserved -> NaN
- reconstruction_errors(M_df, model, train_mask_df, test_mask_df) -> (train_mse, test_mse)
- parallel_analysis(M_df, ...)           -> (pa_dim, observed_sv, null_threshold)
- grid_from_parallel_analysis(M_df, ...) -> (dimension_list, pa_info)
- select_dimension(summary_df, ...)      -> (best_k, selection_info)
- estimate_dimension(M_df, dimension_list, ...) -> (results_df, summary_df, best_k)
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .ICQF import ICQF
from .utils_data import MatrixData

try:
    from threadpoolctl import threadpool_limits
    _HAVE_THREADPOOLCTL = True
except ImportError:  # pragma: no cover
    _HAVE_THREADPOOLCTL = False


# ---------------------------------------------------------------------------
# Thread control for the ADMM solver
# ---------------------------------------------------------------------------

def default_n_threads() -> int:
    """All cores but one, so the ADMM BLAS calls don't freeze the machine (>=1)."""
    return max(1, (os.cpu_count() or 2) - 1)


@contextmanager
def _blas_threads(n_threads: Optional[int]):
    """
    Cap the BLAS / numpy thread pool to ``n_threads`` for the enclosed block.

    ICQF's ADMM loop is dominated by dense ``W Qᵀ`` / Gram products that hand off
    to a threaded BLAS. ``n_threads=None`` leaves the pool untouched; otherwise we
    pin it (default: all cores but one) via threadpoolctl when available, falling
    back to the OpenMP/MKL/OpenBLAS environment variables otherwise.
    """
    if n_threads is None:
        yield
        return
    n_threads = max(1, int(n_threads))
    if _HAVE_THREADPOOLCTL:
        with threadpool_limits(limits=n_threads):
            yield
        return
    # Fallback: set the common BLAS env vars (only affects pools created after).
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = str(n_threads)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def make_bcv_masks(
    M_df: pd.DataFrame,
    J: int = 10,
    *,
    stratified: bool = True,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Entry-wise K-fold split over the *observed* entries of ``M_df``.

    Each observed entry is assigned to exactly one of ``J`` folds. For fold j,
    that fold's entries become the test set and all other observed entries
    become the train set, so per fold::

        train_mask + test_mask == observed
        train_mask * test_mask == 0

    Parameters
    ----------
    M_df:
        Data matrix (samples x features); NaN means unobserved.
    J:
        Number of folds.
    stratified:
        Balance the proportion of positive vs zero entries across folds when
        possible. Falls back to plain KFold if there are too few of either.
    random_state, shuffle:
        Passed to the underlying scikit-learn splitter.

    Returns
    -------
    list of (train_mask_df, test_mask_df), one tuple per fold. Each mask is a
    0/1 float DataFrame sharing ``M_df``'s index and columns.
    """
    if not isinstance(M_df, pd.DataFrame):
        raise TypeError(f"`M_df` must be a pandas DataFrame, got {type(M_df)}")

    M = M_df.to_numpy(dtype=np.float64)
    observed = ~np.isnan(M)
    obs_idx = np.column_stack(np.where(observed))            # (n_obs, 2)
    if obs_idx.shape[0] < J:
        raise ValueError(f"Only {obs_idx.shape[0]} observed entries for J={J} folds.")

    values = M[obs_idx[:, 0], obs_idx[:, 1]]
    y = (values > 0).astype(int)

    use_stratified = stratified and (np.unique(y).size > 1) and (np.bincount(y).min() >= J)
    if stratified and not use_stratified:
        warnings.warn("make_bcv_masks: too few per-class entries to stratify; using plain KFold.")

    splitter = (
        StratifiedKFold(n_splits=J, shuffle=shuffle, random_state=random_state if shuffle else None)
        if use_stratified
        else KFold(n_splits=J, shuffle=shuffle, random_state=random_state if shuffle else None)
    )
    iter_split = splitter.split(obs_idx, y) if use_stratified else splitter.split(obs_idx)

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_ix, test_ix in iter_split:
        train_mask = np.zeros_like(M, dtype=np.float64)
        test_mask = np.zeros_like(M, dtype=np.float64)
        train_mask[obs_idx[train_ix, 0], obs_idx[train_ix, 1]] = 1.0
        test_mask[obs_idx[test_ix, 0], obs_idx[test_ix, 1]] = 1.0
        folds.append((
            pd.DataFrame(train_mask, index=M_df.index.copy(), columns=M_df.columns.copy()),
            pd.DataFrame(test_mask, index=M_df.index.copy(), columns=M_df.columns.copy()),
        ))
    return folds


def build_bcv_folds(
    M_df: pd.DataFrame,
    *,
    J: int = 10,
    repeat: int = 2,
    random_state: int = 42,
    stratified: bool = True,
) -> List[List[Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Build all BCV fold masks: ``repeat`` independent partitions, each ``J`` folds.

    Returns a nested list ``folds[r][j] == (train_mask_df, test_mask_df)`` where
    repeat ``r`` uses seed ``random_state + r``. The masks depend only on
    ``M_df``'s observed pattern and ``(J, repeat, random_state, stratified)`` --
    never on the candidate dimensions or regularization -- so the same ``folds``
    object can be reused across any dimension / beta sweep for a fair, consistent
    comparison (and to skip rebuilding). Pass it back via the ``folds=`` argument
    of ``estimate_dimension`` / ``estimate_dimension_grid``.
    """
    return [
        make_bcv_masks(M_df, J=J, stratified=stratified,
                       random_state=random_state + r, shuffle=True)
        for r in range(repeat)
    ]


def apply_train_mask(M_df: pd.DataFrame, train_mask_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return ``M_df`` with every entry where ``train_mask == 0`` set to NaN.

    The result is the training view: held-out (test) entries and originally
    unobserved entries are both NaN, matching ICQF's "NaN == unobserved"
    convention. Feed it straight to ``MatrixData.from_frames``.
    """
    if not M_df.index.equals(train_mask_df.index) or not M_df.columns.equals(train_mask_df.columns):
        raise ValueError("M_df and train_mask_df must share the same index and columns.")
    vals = M_df.to_numpy(dtype=np.float64).copy()
    vals[train_mask_df.to_numpy() == 0] = np.nan
    return pd.DataFrame(vals, index=M_df.index.copy(), columns=M_df.columns.copy())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def reconstruction_errors(
    M_df: pd.DataFrame,
    model: ICQF,
    train_mask_df: pd.DataFrame,
    test_mask_df: pd.DataFrame,
    C_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, float]:
    """
    Mean squared reconstruction error of a *fitted* ``model`` on the train and
    test entries.

    We index the reconstruction ``R = W Qᵀ (+ C Qcᵀ)`` with the boolean masks
    rather than multiplying by 0/1, because ``M_df`` carries NaN at structurally
    missing positions and ``0 * NaN = NaN`` would poison the sum.
    """
    if model.state_ is None:
        raise RuntimeError("model must be fitted (call fit_transform) before scoring.")

    st = model.state_
    R = st.W @ st.Q.T
    if C_df is not None and st.Qc is not None:
        R = R + C_df.to_numpy(dtype=np.float64) @ st.Qc.T

    M = M_df.to_numpy(dtype=np.float64)
    tm = train_mask_df.to_numpy().astype(bool)
    vm = test_mask_df.to_numpy().astype(bool)

    train_mse = float(np.mean((M[tm] - R[tm]) ** 2)) if tm.any() else float("nan")
    test_mse = float(np.mean((M[vm] - R[vm]) ** 2)) if vm.any() else float("nan")
    return train_mse, test_mse


# ---------------------------------------------------------------------------
# Parallel analysis (grid centering)
# ---------------------------------------------------------------------------

def parallel_analysis(
    M_df: pd.DataFrame,
    *,
    n_resamples: int = 50,
    percentile: float = 95.0,
    center: bool = True,
    random_state: int = 0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Permutation parallel analysis (Horn 1965) on the data matrix.

    The singular-value spectrum of the (column-centered) data is compared
    against a null built by permuting each column independently -- this destroys
    cross-column structure while preserving every column's marginal
    distribution, so it works for the non-negative, non-Gaussian matrices ICQF
    targets. The estimated dimension is the number of leading singular values
    that exceed the ``percentile``-th percentile of the null spectrum.

    Missing entries (NaN) are mean-imputed per column before the SVD.

    Parameters
    ----------
    M_df:
        Data matrix (samples x features); NaN means unobserved.
    n_resamples:
        Number of permuted null matrices.
    percentile:
        Per-component percentile of the null singular values used as threshold.
    center:
        Subtract each column's mean before the SVD (recommended).
    random_state:
        Seed for the column permutations.

    Returns
    -------
    pa_dim:
        Estimated number of components (>= 1).
    observed_sv:
        Singular values of the observed (centered, imputed) matrix, descending.
    null_threshold:
        Per-component percentile threshold from the permuted null, descending.
    """
    if not isinstance(M_df, pd.DataFrame):
        raise TypeError(f"`M_df` must be a pandas DataFrame, got {type(M_df)}")

    M = M_df.to_numpy(dtype=np.float64)
    n, p = M.shape

    # Per-column mean imputation of missing entries.
    col_mean = np.nanmean(M, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    nan_pos = np.isnan(M)
    if nan_pos.any():
        M = M.copy()
        M[nan_pos] = np.take(col_mean, np.where(nan_pos)[1])

    def _spectrum(A: np.ndarray) -> np.ndarray:
        if center:
            A = A - A.mean(axis=0, keepdims=True)
        return np.linalg.svd(A, compute_uv=False)

    observed_sv = _spectrum(M)
    n_sv = observed_sv.shape[0]

    rng = np.random.default_rng(random_state)
    null = np.empty((n_resamples, n_sv), dtype=np.float64)
    for t in range(n_resamples):
        A = np.empty_like(M)
        for c in range(p):
            A[:, c] = M[rng.permutation(n), c]
        sv = _spectrum(A)
        null[t, : sv.shape[0]] = sv[:n_sv]

    null_threshold = np.percentile(null, percentile, axis=0)
    pa_dim = int(np.sum(observed_sv > null_threshold))
    pa_dim = max(1, pa_dim)
    return pa_dim, observed_sv, null_threshold


def grid_from_parallel_analysis(
    M_df: pd.DataFrame,
    *,
    window: int = 3,
    step: int = 1,
    max_dim: Optional[int] = None,
    pa_kwargs: Optional[dict] = None,
) -> Tuple[List[int], Dict]:
    """
    Build a candidate-rank grid as a window centered on the parallel-analysis guess.

    The grid spans ``[pa_dim - window, pa_dim + window]`` (clipped to ``>= 1`` and
    to ``max_dim``), stepping by ``step``. Returns the grid and a dict of PA
    diagnostics (``pa_dim``, ``observed_sv``, ``null_threshold``, ``window``).
    """
    pa_kwargs = dict(pa_kwargs or {})
    pa_dim, observed_sv, null_threshold = parallel_analysis(M_df, **pa_kwargs)

    n, p = M_df.shape
    hard_cap = min(n, p) - 1
    if max_dim is not None:
        hard_cap = min(hard_cap, int(max_dim))
    hard_cap = max(1, hard_cap)

    lo = max(1, pa_dim - window)
    hi = min(hard_cap, pa_dim + window)
    if hi < lo:
        hi = lo
    grid = list(range(lo, hi + 1, max(1, int(step))))
    if not grid:
        grid = [max(1, pa_dim)]

    info = {
        "pa_dim": pa_dim,
        "observed_sv": observed_sv,
        "null_threshold": null_threshold,
        "window": int(window),
        "grid_lo": lo,
        "grid_hi": hi,
    }
    return grid, info


# ---------------------------------------------------------------------------
# Selection: naive argmin vs 1-SE rule
# ---------------------------------------------------------------------------

def select_dimension(
    summary_df: pd.DataFrame,
    *,
    use_1se: bool = True,
    se_mult: float = 1.0,
    loss_col: str = "test_mse_mean",
    se_col: str = "test_mse_se",
    dim_col: str = "n_components",
) -> Tuple[int, Dict]:
    """
    Pick the embedding dimension from a per-rank summary table.

    With ``use_1se=False`` this is the naive minimizer of the mean CV loss. With
    ``use_1se=True`` (default) it applies the (``se_mult``-)SE rule: let ``r_min``
    be the minimizer and ``SE(r_min)`` its standard error; choose the *smallest*
    rank whose mean loss is within ``loss(r_min) + se_mult * SE(r_min)``. This
    favors the simplest model that is statistically indistinguishable from the
    best, taming the jitter of a flat CV curve.

    ``se_mult`` scales the tolerance band: ``1.0`` is the classic 1-SE rule;
    ``0.5`` is a softer "half-SE" that resists over-simplification; ``0.0``
    recovers the naive argmin. Whichever SE column ``se_col`` names is used --
    ``estimate_dimension`` defaults it to the *fold-paired* SE, which strips the
    fold-difficulty offset and so gives a tighter, fairer band than the raw SE.

    Returns ``(best_k, info)`` where ``info`` records the argmin, the SE pick,
    ``se_mult``, and the threshold used.
    """
    s = summary_df.sort_values(dim_col).reset_index(drop=True)
    dims = s[dim_col].to_numpy()
    loss = s[loss_col].to_numpy(dtype=np.float64)

    argmin_pos = int(np.nanargmin(loss))
    argmin_k = int(dims[argmin_pos])

    info = {
        "argmin_k": argmin_k,
        "argmin_loss": float(loss[argmin_pos]),
        "rule": f"{se_mult:g}se" if use_1se else "argmin",
        "se_mult": float(se_mult),
        "se_col": se_col,
    }

    if not use_1se or se_col not in s.columns:
        info["best_k"] = argmin_k
        info["threshold"] = float(loss[argmin_pos])
        return argmin_k, info

    se_min = float(s[se_col].to_numpy(dtype=np.float64)[argmin_pos])
    se_min = 0.0 if not np.isfinite(se_min) else se_min
    threshold = float(loss[argmin_pos]) + se_mult * se_min

    # Smallest rank within the tolerance band (search only ranks <= argmin).
    within = np.where(loss[: argmin_pos + 1] <= threshold)[0]
    best_pos = int(within[0]) if within.size else argmin_pos
    best_k = int(dims[best_pos])

    info["best_k"] = best_k
    info["se_at_min"] = se_min
    info["threshold"] = threshold
    return best_k, info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def estimate_dimension(
    M_df: pd.DataFrame,
    dimension_list: Optional[Sequence[int]] = None,
    *,
    C_df: Optional[pd.DataFrame] = None,
    J: int = 10,
    repeat: int = 2,
    random_state: int = 42,
    stratified: bool = True,
    icqf_kwargs: Optional[dict] = None,
    verbose: bool = True,
    n_threads: Optional[int] = "auto",
    use_parallel_analysis: bool = True,
    pa_window: int = 3,
    pa_max_dim: Optional[int] = None,
    pa_kwargs: Optional[dict] = None,
    use_1se: bool = True,
    se_mult: float = 1.0,
    paired_se: bool = True,
    folds: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Estimate the ICQF embedding dimension by held-out reconstruction error.

    For every (dimension, repeat, fold) it fits ICQF on the training entries and
    records train/test MSE. Folds are fit serially in this process (no cluster
    dispatch); each ADMM fit may use multiple BLAS threads (see ``n_threads``).

    Parameters
    ----------
    M_df:
        Data matrix (samples x features); NaN means unobserved.
    dimension_list:
        Candidate ranks ``k`` to evaluate, e.g. ``[2, 3, 4, 5, 6, 8, 10]``. If
        ``None`` (the default) and ``use_parallel_analysis`` is True, the grid is
        built automatically as a window centered on the parallel-analysis guess.
    C_df:
        Optional confound design matrix (samples x confounds), fully observed,
        with index exactly equal to ``M_df.index``. When given it is held fixed
        across train/test (only its observed-entry contribution is scored).
    J:
        Number of BCV folds.
    repeat:
        How many independent fold partitions to average over (distinct seeds).
    random_state:
        Base seed; repeat r uses ``random_state + r``.
    stratified:
        Stratify folds by positive/zero entries when possible.
    icqf_kwargs:
        Extra keyword args forwarded to ``ICQF(...)`` (e.g. ``W_beta``,
        ``Q_beta``, ``regularizer``, ``max_iter``, bounds). ``n_components`` and
        ``random_state`` are set automatically per job.
    verbose:
        Print per-job progress.
    n_threads:
        BLAS / numpy thread cap for the ADMM solver. ``"auto"`` (default) uses
        all cores but one so the machine stays responsive; an int pins exactly
        that many; ``None`` leaves the global pool untouched.
    use_parallel_analysis:
        When True and ``dimension_list is None``, center the search grid on a
        permutation parallel-analysis estimate via ``grid_from_parallel_analysis``.
    pa_window:
        Half-width of the PA-centered window (grid is ``pa_dim +/- pa_window``).
    pa_max_dim:
        Optional hard cap on the largest rank in a PA-built grid.
    pa_kwargs:
        Extra keyword args for ``parallel_analysis`` (e.g. ``n_resamples``,
        ``percentile``).
    use_1se:
        Apply the SE rule when selecting ``best_k`` (default). If False, use the
        naive argmin of the mean test MSE.
    se_mult:
        Tolerance multiplier on the SE band (``1.0`` = classic 1-SE; ``0.5`` =
        softer half-SE; ``0.0`` = argmin). Passed to ``select_dimension``.
    paired_se:
        When True (default), the ``test_mse_se`` used for selection is the
        *fold-paired* SE: each (repeat, fold) curve is centered by its own mean
        across ranks before taking the across-fold SE, removing the common
        fold-difficulty offset that otherwise inflates the band. The raw
        (unpaired) SE is still reported as ``test_mse_se_raw``.
    folds:
        Optional precomputed fold masks (``build_bcv_folds`` output:
        ``folds[r][j] == (train_mask_df, test_mask_df)``). When given, ``J`` /
        ``repeat`` / ``stratified`` are taken from it instead of being rebuilt;
        useful to reuse the *same* splits across dimension / beta sweeps. The
        folds actually used are always returned via ``summary_df.attrs["folds"]``.

    Returns
    -------
    results_df:
        One row per (dimension, repeat, fold) with train_mse / test_mse.
    summary_df:
        Per-dimension mean / std / standard-error of test_mse and train_mse,
        sorted by dimension. ``summary_df.attrs`` carries ``selection_info``,
        ``folds`` (the masks used, for reuse), and (when used) ``pa_info``.
    best_k:
        Selected dimension (1-SE rule by default, else argmin of mean test MSE).
    """
    icqf_kwargs = dict(icqf_kwargs or {})
    icqf_kwargs.pop("n_components", None)
    icqf_kwargs.pop("random_state", None)

    if n_threads == "auto":
        n_threads = default_n_threads()

    # --- Build the candidate grid (PA-centered window when none supplied) ---
    pa_info: Optional[Dict] = None
    if dimension_list is None:
        if use_parallel_analysis:
            dimension_list, pa_info = grid_from_parallel_analysis(
                M_df, window=pa_window, max_dim=pa_max_dim, pa_kwargs=pa_kwargs,
            )
            if verbose:
                print(
                    f"[parallel analysis] pa_dim={pa_info['pa_dim']} -> "
                    f"grid {dimension_list}"
                )
        else:
            raise ValueError(
                "dimension_list is None and use_parallel_analysis is False; "
                "supply an explicit dimension_list."
            )
    dimension_list = [int(k) for k in dimension_list]

    # Build the fold masks once (or reuse a passed-in set). They are independent
    # of the candidate dimensions, so they can be reused across sweeps.
    if folds is None:
        folds = build_bcv_folds(
            M_df, J=J, repeat=repeat, random_state=random_state, stratified=stratified,
        )

    rows = []
    with _blas_threads(n_threads):
        for r, folds_r in enumerate(folds):
            for j, (train_mask_df, test_mask_df) in enumerate(folds_r):
                M_train_df = apply_train_mask(M_df, train_mask_df)
                data = MatrixData.from_frames(M_train_df, C_df=C_df)

                for k in dimension_list:
                    model = ICQF(
                        n_components=int(k),
                        random_state=random_state + r,
                        **icqf_kwargs,
                    )
                    model.fit_transform(data)
                    train_mse, test_mse = reconstruction_errors(
                        M_df, model, train_mask_df, test_mask_df, C_df=C_df,
                    )
                    rows.append({
                        "n_components": int(k),
                        "repeat": r,
                        "fold": j,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                    })
                    if verbose:
                        print(f"[k={k:>3} r={r} j={j}]  train_mse={train_mse:.5f}  test_mse={test_mse:.5f}")

    results_df = pd.DataFrame(rows)

    def _se(x: pd.Series) -> float:
        # Across-replicate standard error: std / sqrt(count).
        n = x.count()
        return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    summary_df = (
        results_df
        .groupby("n_components")
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_mse_se_raw=("test_mse", _se),
            train_mse_mean=("train_mse", "mean"),
            train_mse_std=("train_mse", "std"),
            train_mse_se=("train_mse", _se),
            n_replicates=("test_mse", "count"),
        )
        .reset_index()
        .sort_values("n_components")
        .reset_index(drop=True)
    )

    # Fold-paired SE: the same (repeat, fold) blocks score every rank, so the
    # held-out *level* of each block (how hard that fold is) is a shared offset
    # across ranks. Center each block's curve by its own mean before taking the
    # across-block SE, isolating the rank-related variation and yielding a
    # tighter, fairer tolerance band for the SE rule.
    paired_map = _fold_paired_se(results_df, key_cols=["n_components"])
    summary_df["test_mse_se_paired"] = [paired_map[(int(k),)] for k in summary_df["n_components"]]
    summary_df["test_mse_se"] = (
        summary_df["test_mse_se_paired"] if paired_se else summary_df["test_mse_se_raw"]
    )

    best_k, selection_info = select_dimension(
        summary_df, use_1se=use_1se, se_mult=se_mult,
    )
    summary_df.attrs["selection_info"] = selection_info
    summary_df.attrs["folds"] = folds          # emitted by default for reuse
    if pa_info is not None:
        summary_df.attrs["pa_info"] = pa_info
    return results_df, summary_df, best_k


def _fold_paired_se(
    results_df: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("n_components",),
    value_col: str = "test_mse",
) -> Dict[tuple, float]:
    """
    Fold-paired standard error of the held-out loss per model configuration.

    Pivot to (repeat, fold) blocks x configuration, subtract each block's row mean
    (its overall difficulty level), then take std/sqrt(n_blocks) of the centered
    values per configuration. Removes the configuration-constant fold offset that
    inflates the naive across-replicate SE.

    ``key_cols`` names the columns identifying a configuration -- ``["n_components"]``
    for a 1-D rank sweep, or ``["n_components", "W_beta", "Q_beta"]`` for the grid.
    Returns a dict keyed by the configuration **tuple** (always a tuple, even for a
    single key).
    """
    key_cols = list(key_cols)
    piv = results_df.pivot_table(index=["repeat", "fold"], columns=key_cols, values=value_col)
    n_blocks = piv.shape[0]

    def _as_tuple(col) -> tuple:
        return tuple(col) if isinstance(col, tuple) else (col,)

    if n_blocks <= 1:
        return {_as_tuple(c): 0.0 for c in piv.columns}
    centered = piv.sub(piv.mean(axis=1), axis=0)
    se = centered.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    return {_as_tuple(c): float(v) for c, v in se.items()}


# ---------------------------------------------------------------------------
# Joint (dimension x W_beta x Q_beta) grid
# ---------------------------------------------------------------------------

def select_grid(
    summary_df: pd.DataFrame,
    *,
    use_1se: bool = True,
    se_mult: float = 1.0,
    loss_col: str = "test_mse_mean",
    se_col: str = "test_mse_se",
) -> Tuple[Dict, Dict]:
    """
    Pick the best ``(n_components, W_beta, Q_beta)`` from a grid summary table.

    ``use_1se=False`` returns the argmin of the mean held-out loss. ``use_1se=True``
    (default) applies the SE rule on the joint grid: among all configurations whose
    mean loss is within ``loss(best) + se_mult * SE(best)``, pick the *simplest*,
    where "simplest" means smallest ``n_components`` first, then the most
    regularized -- largest ``W_beta``, then largest ``Q_beta``.

    Returns ``(best, info)``: ``best`` is a dict with the chosen
    ``n_components`` / ``W_beta`` / ``Q_beta`` / loss; ``info`` records the argmin
    config, the threshold, and ``se_mult``.
    """
    s = summary_df.reset_index(drop=True)
    loss = s[loss_col].to_numpy(dtype=np.float64)
    argmin_pos = int(np.nanargmin(loss))

    def _row_to_best(pos: int) -> Dict:
        row = s.loc[pos]
        return {
            "n_components": int(row["n_components"]),
            "W_beta": float(row["W_beta"]),
            "Q_beta": float(row["Q_beta"]),
            loss_col: float(row[loss_col]),
        }

    info = {
        "argmin": _row_to_best(argmin_pos),
        "rule": f"{se_mult:g}se" if use_1se else "argmin",
        "se_mult": float(se_mult),
    }

    if not use_1se or se_col not in s.columns:
        info["threshold"] = float(loss[argmin_pos])
        return _row_to_best(argmin_pos), info

    se_min = float(s[se_col].to_numpy(dtype=np.float64)[argmin_pos])
    se_min = 0.0 if not np.isfinite(se_min) else se_min
    threshold = float(loss[argmin_pos]) + se_mult * se_min

    within = s[loss <= threshold]
    # simplest within the band: fewest factors, then most-regularized.
    within = within.sort_values(
        ["n_components", "W_beta", "Q_beta"], ascending=[True, False, False]
    )
    best_pos = int(within.index[0])
    info["threshold"] = threshold
    info["se_at_min"] = se_min
    return _row_to_best(best_pos), info


def estimate_dimension_grid(
    M_df: pd.DataFrame,
    dimension_list: Sequence[int],
    *,
    W_beta_list: Sequence[float],
    Q_beta_list: Sequence[float],
    C_df: Optional[pd.DataFrame] = None,
    J: int = 10,
    repeat: int = 2,
    random_state: int = 42,
    stratified: bool = True,
    icqf_kwargs: Optional[dict] = None,
    verbose: bool = True,
    n_threads: Optional[int] = "auto",
    use_1se: bool = True,
    se_mult: float = 1.0,
    paired_se: bool = True,
    folds: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Jointly select the embedding dimension and the L1 penalties by BCV.

    Sweeps the full 3-D grid ``dimension_list x W_beta_list x Q_beta_list``. The
    fold masks are built once (or reused from ``folds``) and, for each fold, the
    training ``MatrixData`` is built once and shared across every
    ``(k, W_beta, Q_beta)`` -- so the comparison is on identical splits and the
    redundant masking/packing is not repeated.

    Parameters
    ----------
    M_df:
        Data matrix (samples x features); NaN means unobserved. Scale into [0, 1].
    dimension_list:
        Candidate ranks ``k``.
    W_beta_list, Q_beta_list:
        Candidate L1 penalties for ``W`` and ``Q`` (swept independently).
    C_df, J, repeat, random_state, stratified, n_threads, verbose:
        As in ``estimate_dimension``.
    icqf_kwargs:
        Extra ICQF args (e.g. ``regularizer``, ``max_iter``, bounds). ``n_components``,
        ``W_beta``, ``Q_beta`` and ``random_state`` are set per grid point.
    use_1se, se_mult, paired_se:
        Selection controls; see ``select_grid`` and ``estimate_dimension``. The
        paired SE is computed **per ``(k, W_beta, Q_beta)`` configuration**,
        centering each (repeat, fold) block across the whole grid.
    folds:
        Optional precomputed masks (``build_bcv_folds`` output) to reuse the same
        splits; always returned via ``summary_df.attrs["folds"]``.

    Returns
    -------
    results_df:
        One row per (n_components, W_beta, Q_beta, repeat, fold) with train/test MSE.
    summary_df:
        One row per (n_components, W_beta, Q_beta) with mean / std / SE (raw and
        paired) of the MSEs. ``attrs`` carries ``selection_info`` and ``folds``.
    best:
        Dict with the selected ``n_components`` / ``W_beta`` / ``Q_beta``.
    """
    icqf_kwargs = dict(icqf_kwargs or {})
    for key in ("n_components", "W_beta", "Q_beta", "random_state"):
        icqf_kwargs.pop(key, None)

    if n_threads == "auto":
        n_threads = default_n_threads()

    dimension_list = [int(k) for k in dimension_list]
    W_beta_list = [float(b) for b in W_beta_list]
    Q_beta_list = [float(b) for b in Q_beta_list]

    if folds is None:
        folds = build_bcv_folds(
            M_df, J=J, repeat=repeat, random_state=random_state, stratified=stratified,
        )

    rows = []
    with _blas_threads(n_threads):
        for r, folds_r in enumerate(folds):
            for j, (train_mask_df, test_mask_df) in enumerate(folds_r):
                data = MatrixData.from_frames(apply_train_mask(M_df, train_mask_df), C_df=C_df)
                for k in dimension_list:
                    for wb in W_beta_list:
                        for qb in Q_beta_list:
                            model = ICQF(
                                n_components=k, W_beta=wb, Q_beta=qb,
                                random_state=random_state + r, **icqf_kwargs,
                            )
                            model.fit_transform(data)
                            train_mse, test_mse = reconstruction_errors(
                                M_df, model, train_mask_df, test_mask_df, C_df=C_df,
                            )
                            rows.append({
                                "n_components": k, "W_beta": wb, "Q_beta": qb,
                                "repeat": r, "fold": j,
                                "train_mse": train_mse, "test_mse": test_mse,
                            })
                            if verbose:
                                print(f"[k={k:>3} W_beta={wb:g} Q_beta={qb:g} r={r} j={j}]  "
                                      f"test_mse={test_mse:.5f}")

    results_df = pd.DataFrame(rows)

    def _se(x: pd.Series) -> float:
        n = x.count()
        return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    keys = ["n_components", "W_beta", "Q_beta"]
    summary_df = (
        results_df
        .groupby(keys)
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_mse_se_raw=("test_mse", _se),
            train_mse_mean=("train_mse", "mean"),
            n_replicates=("test_mse", "count"),
        )
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )

    paired_map = _fold_paired_se(results_df, key_cols=keys)
    summary_df["test_mse_se_paired"] = [
        paired_map[(int(kk), float(wb), float(qb))]
        for kk, wb, qb in summary_df[keys].itertuples(index=False, name=None)
    ]
    summary_df["test_mse_se"] = (
        summary_df["test_mse_se_paired"] if paired_se else summary_df["test_mse_se_raw"]
    )

    best, selection_info = select_grid(summary_df, use_1se=use_1se, se_mult=se_mult)
    summary_df.attrs["selection_info"] = selection_info
    summary_df.attrs["folds"] = folds          # emitted by default for reuse
    return results_df, summary_df, best
