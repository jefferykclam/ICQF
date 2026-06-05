"""
ICQF / Constrained matrix factorization with an ADMM-style auxiliary variable.

Public API
----------
`ICQF` expects the caller to construct `MatrixData` explicitly before fitting.
This keeps validation and data inspection outside the solver and makes the
optimization entry points small and predictable:
- `fit_transform(X: MatrixData)`
- `transform(X: MatrixData, init_state=None)`

Internal numerical representation
---------------------------------
All iterative updates are performed on C-contiguous `float64` NumPy arrays.
`MatrixData` and `FactorState` carry labels for inspection, but the actual solver
never manipulates pandas objects inside the optimization loop.

Model and optimization problem
------------------------------
Given an observed matrix `M \in R^{n x p}` with observation mask `Omega`, ICQF solves

    min_{W,Q,Qc,Z} 1/2 || Omega ⊙ (M - Z) ||_F^2
                    + beta_W * R(W) + beta_Q * R(Q)
                    + (rho/2) || Z - (W Q^T + C Qc^T) ||_F^2

subject to nonnegativity and optional upper bounds on `W`, `Q`, `Qc`, and `Z`.

Here:
- `W \in R^{n x k}` are subject scores
- `Q \in R^{p x k}` are item loadings
- `C \in R^{n x c}` is an optional fully observed design matrix
- `Qc \in R^{p x c}` are confound loadings
- `Z` is an auxiliary reconstruction variable used to separate the masked loss from
  the low-rank reconstruction term

The ADMM-style loop alternates between updating:
1. `W` with `Q, Qc, Z, aZ` fixed
2. `Q, Qc` with `W, Z, aZ` fixed
3. `Z` in closed form
4. the scaled dual variable `aZ`
"""

from __future__ import annotations
import os, sys
from typing import List, Optional, Tuple



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array
from tqdm.auto import tqdm

# The coordinate-descent kernel `_cdnmf_fast_cy` is a Cython extension. When the
# package is installed (e.g. `uv pip install .`) it is compiled to a real .so at
# build time, so this import just succeeds with no compiler needed at runtime.
# For an in-place/dev checkout where the extension was never built, fall back to
# `pyximport`, which compiles it on first import (needs Cython + a C compiler).
try:
    from ._cdnmf_fast_cy import _update_cdnmf_fast
except ImportError:
    import pyximport
    pyximport.install(
        setup_args={"include_dirs": np.get_include()},
        build_in_temp=False,
        language_level=3,
    )
    from ._cdnmf_fast_cy import _update_cdnmf_fast

from .utils_SVD import _svd_initialize
from .utils_data import FactorState, MatrixData

Array = np.ndarray


def _require_c_float64_writeable(a: np.ndarray) -> np.ndarray:
    """Return a writeable C-order float64 array suitable for the Cython CD kernel."""
    if a.dtype != np.float64:
        a = a.astype(np.float64, copy=False)
    a = np.require(a, requirements=["C", "ALIGNED", "WRITEABLE"])
    # A 2-D array with a singleton dimension (e.g. an (n, 1) rank-1 factor) is
    # flagged *both* C- and F-contiguous, so `np.require`/`ascontiguousarray` may
    # leave Fortran column strides in place. The Cython kernel requires the last
    # dim to be unit-strided, so force a true C-order copy in that case.
    if a.ndim == 2 and a.strides[-1] != a.itemsize:
        a = a.copy(order="C")
    return a


def _clip_inplace(x: Array, lo: float = 0.0, hi: Optional[float] = None) -> None:
    """Project `x` onto the box `[lo, hi]` in place."""
    np.maximum(x, lo, out=x)
    if hi is not None:
        np.minimum(x, hi, out=x)


def _update_coordinate_descent(
    X: Array,
    W: Array,
    Ht: Array,
    *,
    regularizer: int,
    beta: float,
    upperbd: float,
    shuffle: bool,
    rng: np.random.RandomState,
):
    """
    Solve a nonnegative coordinate-descent subproblem using the project Cython kernel.

    The kernel updates `W` in the penalized least-squares problem

        min_{W >= 0} 1/2 || X - W H^T ||_F^2 + beta * R(W)

    where `R` is either an L1 or L2 row-wise penalty depending on `regularizer`.
    `Ht` stores `H` rather than `H^T` to match the existing project kernel contract.
    """
    n_components = Ht.shape[1]
    HHt = Ht.T @ Ht
    XHt = safe_sparse_dot(X, Ht)

    if regularizer == 2:
        HHt.flat[:: n_components + 1] += beta
    elif regularizer == 1:
        XHt -= beta
    else:
        raise ValueError("regularizer must be 1 (L1) or 2 (L2).")

    perm = rng.permutation(n_components) if shuffle else np.arange(n_components)
    perm = np.asarray(perm, dtype=np.intp)

    if not (W.flags["C_CONTIGUOUS"] and W.flags["WRITEABLE"]):
        raise ValueError("W must be C-contiguous and writeable.")
    if W.strides[1] != W.itemsize:
        raise ValueError(f"W last-dim stride must be {W.itemsize}, got {W.strides[1]}")

    out = _update_cdnmf_fast(W, HHt, XHt, upperbd, perm)

    if out is None or isinstance(out, (float, np.floating, int, np.integer)):
        return out
    if isinstance(out, np.ndarray):
        if out.shape != W.shape:
            raise TypeError(f"_update_cdnmf_fast returned {out.shape}, expected {W.shape}")
        W[...] = out
        return None
    if isinstance(out, tuple):
        for x in out:
            if isinstance(x, np.ndarray) and x.shape == W.shape:
                W[...] = x
        for x in out:
            if isinstance(x, (float, np.floating, int, np.integer)):
                return x
        return None
    raise TypeError(f"Unexpected return type from _update_cdnmf_fast: {type(out)}")


class ICQF(TransformerMixin, BaseEstimator):
    """
    Constrained matrix factorization with optional confound adjustment.

    Input scaling
    -------------
    **Normalize ``M`` into ``[0, 1]`` before fitting.** The defaults assume it:
    the reconstruction ``Z`` is clipped to ``[0, 1]`` (``M_upperbd=(True, 1.0)``)
    and the item loadings ``Q`` are likewise capped at ``1`` (``Q_upperbd=(True,
    1.0)``). If ``M`` is not in ``[0, 1]`` these bounds will distort the fit (the
    reconstruction cannot exceed 1 and the loadings cannot grow past 1), so either
    rescale ``M`` (e.g. divide by its max) or relax/raise the bounds to match the
    data's range.

    Parameters
    ----------
    n_components:
        Rank `k` of the factorization `M \approx W Q^T`.
    W_beta, Q_beta:
        Regularization weights on `W` and `Q`.
    regularizer:
        `1` for L1 and `2` for L2 row-wise regularization.
    rho:
        Penalty parameter in the ADMM-style splitting.
    W_upperbd, Q_upperbd, M_upperbd:
        Tuples `(enabled, value)` controlling optional upper bounds. Defaults:
        `W_upperbd=(False, 1.0)` (W unbounded above), `Q_upperbd=(True, 1.0)` and
        `M_upperbd=(True, 1.0)` (Q and the reconstruction capped at 1, matching an
        `M` rescaled into `[0, 1]`). Set the flag to `False` to disable a bound.
    init_state:
        Optional `FactorState` used to initialize `(W, Q, Qc)`.
    min_iter, max_iter, tol:
        Iteration controls for the outer optimization loop.
    transform_knn_neighbors:
        Number of neighbors used to initialize `W` in `transform` when no
        explicit initialization is supplied.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        W_beta: float = 0.1,
        Q_beta: float = 0.1,
        regularizer: int = 1,
        rho: float = 3.0,
        W_upperbd: Tuple[bool, float] = (False, 1.0),
        Q_upperbd: Tuple[bool, float] = (True, 1.0),
        M_upperbd: Tuple[bool, float] = (True, 1.0),
        init_state: Optional[FactorState] = None,
        min_iter: int = 10,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state=None,
        verbose: bool = False,
        transform_knn_neighbors: int = 5,
    ):
        self.n_components = n_components
        self.W_beta = W_beta
        self.Q_beta = Q_beta
        self.regularizer = regularizer
        self.rho = rho
        self.W_upperbd = W_upperbd
        self.Q_upperbd = Q_upperbd
        self.M_upperbd = M_upperbd
        self.init_state = init_state
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.transform_knn_neighbors = transform_knn_neighbors

        self.state_: Optional[FactorState] = None
        self.loss_history_: Optional[List[Tuple[float, float, float]]] = None

        self._rng = check_random_state(self.random_state)
        self.Mmin = 0.0
        self.Mmax = self.M_upperbd[1] if self.M_upperbd[0] else None
        self._train_data_M_: Optional[Array] = None
        self._train_W_: Optional[Array] = None

    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _upper_or_minus1(self, flag_value: Tuple[bool, float]) -> float:
        return flag_value[1] if flag_value[0] else -1.0

    def _require_matrixdata(self, X) -> MatrixData:
        """Require a prevalidated `MatrixData` object as solver input."""
        if isinstance(X, MatrixData):
            return X
        raise TypeError(
            "ICQF expects `X` to be a MatrixData instance. "
            "Construct it explicitly with MatrixData.from_frames(...) before fitting. "
            f"Got {type(X)}."
        )

    def _attach_dataframe_views(self, data: MatrixData, state: FactorState) -> FactorState:
        """Attach label-aware DataFrame views to a fitted `FactorState`."""
        factor_cols = [f"factor_{j + 1}" for j in range(self.n_components)]
        sample_index = pd.Index(data.sample_index.astype(str), name="sample_id")
        feature_index = pd.Index(data.feature_index.astype(str), name="feature")

        state.W_df = pd.DataFrame(np.asarray(state.W, dtype=np.float64), index=sample_index, columns=factor_cols)
        state.Q_df = pd.DataFrame(np.asarray(state.Q, dtype=np.float64), index=feature_index, columns=factor_cols)
        state.Z_df = pd.DataFrame(np.asarray(state.Z, dtype=np.float64), index=sample_index, columns=feature_index)
        state.aZ_df = pd.DataFrame(np.asarray(state.aZ, dtype=np.float64), index=sample_index, columns=feature_index)

        if data.C is not None and state.Qc is not None:
            confound_index = pd.Index(data.confound_feature_index.astype(str), name="confound")
            state.Qc_df = pd.DataFrame(
                np.asarray(state.Qc, dtype=np.float64),
                index=feature_index,
                columns=confound_index,
            )
        else:
            state.Qc_df = None

        return state

    def _ensure_C_and_Qc(self, data: MatrixData, state: FactorState) -> None:
        """Enforce the contract that `Qc` is present iff `C` is present."""
        if data.C is not None:
            C = check_array(data.C, order="C", copy=False, dtype=np.float64)
            p = data.M.shape[1]
            c = C.shape[1]
            if state.Qc is None:
                raise ValueError("data.C is present but init_state.Qc is None. Provide Qc with shape (p, c).")
            Qc = check_array(state.Qc, order="C", copy=False, dtype=np.float64)
            if Qc.shape != (p, c):
                raise ValueError(f"Qc shape mismatch: expected {(p, c)}, got {Qc.shape}")
            state.Qc = Qc
        else:
            if state.Qc is not None:
                raise ValueError("data.C is None but init_state.Qc was provided. Set Qc=None.")
            state.Qc = None

    def _validate_and_copy_init_state(self, data: MatrixData, init_state: FactorState) -> FactorState:
        """
        Validate a user-supplied initialization and rebuild `Z` from the supplied factors.

        We allow the user to initialize only the low-rank factors. The auxiliary variable
        `Z` is then reset to the bounded reconstruction

            Z <- clip(W Q^T + C Qc^T)

        and the scaled dual variable is reset to zero for stability.
        """
        M = check_array(data.M, order="C", copy=False, dtype=np.float64)
        n, p = M.shape
        k = self.n_components

        W = np.asarray(init_state.W, dtype=np.float64, order="C").copy()
        Q = np.asarray(init_state.Q, dtype=np.float64, order="C").copy()
        if W.shape != (n, k):
            raise ValueError(f"init_state.W shape mismatch: expected {(n, k)}, got {W.shape}")
        if Q.shape != (p, k):
            raise ValueError(f"init_state.Q shape mismatch: expected {(p, k)}, got {Q.shape}")

        Qc = None if init_state.Qc is None else np.asarray(init_state.Qc, dtype=np.float64, order="C").copy()
        state = FactorState(W=W, Q=Q, Qc=Qc, Z=np.zeros_like(M), aZ=np.zeros_like(M))
        self._ensure_C_and_Qc(data, state)

        _clip_inplace(state.W, 0.0, self.W_upperbd[1] if self.W_upperbd[0] else None)
        _clip_inplace(state.Q, 0.0, self.Q_upperbd[1] if self.Q_upperbd[0] else None)
        if state.Qc is not None:
            _clip_inplace(state.Qc, 0.0, self.Q_upperbd[1] if self.Q_upperbd[0] else None)

        Z = state.W @ state.Q.T
        if data.C is not None:
            Z = Z + check_array(data.C, order="C", copy=False, dtype=np.float64) @ state.Qc.T
        _clip_inplace(Z, self.Mmin, self.Mmax)
        state.Z = np.asarray(Z, dtype=np.float64, order="C")
        state.aZ = np.zeros_like(M, dtype=np.float64, order="C")
        return state

    def _init_state(self, data: MatrixData, svd_components=None) -> FactorState:
        """Initialize `(W, Q, Qc, Z, aZ)` either from `init_state` or from NNDSVD."""
        if self.init_state is not None:
            return self._validate_and_copy_init_state(data, self.init_state)

        M = check_array(data.M, order="C", copy=False, dtype=np.float64)
        k = self.n_components
        W, Q = _svd_initialize(M, k, svd_components=svd_components, random_state=self.random_state)
        W = np.asarray(W, dtype=np.float64, order="C")
        Q = np.asarray(Q, dtype=np.float64, order="C")

        _clip_inplace(W, 0.0, self.W_upperbd[1] if self.W_upperbd[0] else None)
        _clip_inplace(Q, 0.0, self.Q_upperbd[1] if self.Q_upperbd[0] else None)

        Qc = None
        if data.C is not None:
            C = check_array(data.C, order="C", copy=False, dtype=np.float64)
            Qc = np.zeros((M.shape[1], C.shape[1]), dtype=np.float64, order="C")

        Z = W @ Q.T
        if data.C is not None and Qc is not None:
            Z = Z + data.C @ Qc.T
        _clip_inplace(Z, self.Mmin, self.Mmax)

        state = FactorState(
            W=W,
            Q=Q,
            Qc=Qc,
            Z=np.asarray(Z, dtype=np.float64, order="C"),
            aZ=np.zeros_like(M, dtype=np.float64, order="C"),
        )
        self._ensure_C_and_Qc(data, state)
        return state

    def _update_W_inplace(self, state: FactorState, B: Array, gamma: float) -> None:
        """
        Update `W` by solving

            W <- argmin_{W >= 0}  (rho/2) || B - W Q^T ||_F^2 + gamma * R(W)

        where `B = Z + aZ / rho - C Qc^T` when confounds are present.
        """
        state.W = _require_c_float64_writeable(state.W)
        state.Q = _require_c_float64_writeable(state.Q)
        B = _require_c_float64_writeable(B)

        _update_coordinate_descent(
            X=B,
            W=state.W,
            Ht=state.Q,
            regularizer=self.regularizer,
            beta=gamma,
            upperbd=self._upper_or_minus1(self.W_upperbd),
            shuffle=False,
            rng=self._rng,
        )

    def _update_Q_inplace(self, state: FactorState, B: Array, C: Optional[Array], gamma: float) -> None:
        """
        Update `Q` (and `Qc` when `C` is present) by solving

            min_{Q, Qc >= 0} (rho/2) || B - W Q^T - C Qc^T ||_F^2 + gamma * R(Q)

        through a single block coordinate-descent call on the concatenated variable
        `[Q | Qc]` and concatenated design `[W | C]`.
        """
        if C is not None:
            if state.Qc is None:
                raise ValueError("C provided but state.Qc is None.")
            Q_comp = np.column_stack((state.Q, state.Qc))
            W_comp = np.column_stack((state.W, C))
        else:
            Q_comp = state.Q
            W_comp = state.W

        Q_comp = _require_c_float64_writeable(Q_comp)
        W_comp = _require_c_float64_writeable(W_comp)
        Bt = _require_c_float64_writeable(B.T)

        _update_coordinate_descent(
            X=Bt,
            W=Q_comp,
            Ht=W_comp,
            regularizer=self.regularizer,
            beta=gamma,
            upperbd=self._upper_or_minus1(self.Q_upperbd),
            shuffle=False,
            rng=self._rng,
        )

        k = state.Q.shape[1]
        state.Q = Q_comp[:, :k]
        if C is not None:
            state.Qc = Q_comp[:, k:]

    def _update_Z_inplace(self, data: MatrixData, state: FactorState, R: Array) -> None:
        r"""
        Closed-form update of the auxiliary variable `Z`.

        With `R = W Q^T + C Qc^T - aZ / rho`, the `Z` subproblem is

            min_Z  1/2 || Omega ⊙ (M - Z) ||_F^2 + (rho/2) || Z - R ||_F^2

        which decouples entrywise and has the solution

            Z_{ij} = (Omega_{ij} M_{ij} + rho R_{ij}) / (Omega_{ij} + rho).
        """
        Z = (data.nan_mask * data.M + self.rho * R) / (data.nan_mask + self.rho)
        _clip_inplace(Z, self.Mmin, self.Mmax)
        state.Z = np.asarray(Z, dtype=np.float64, order="C")

    def _obj_terms(self, data: MatrixData, state: FactorState, betaW: float, betaQ: float) -> Tuple[float, float]:
        """Evaluate the masked reconstruction loss and regularization terms."""
        R = state.W @ state.Q.T
        if data.C is not None:
            R = R + check_array(data.C, order="C", copy=False, dtype=np.float64) @ state.Qc.T

        mismatch = 0.5 * np.sum(data.nan_mask * (data.M - R) ** 2)
        W_reg = betaW * np.sum(np.linalg.norm(state.W, ord=self.regularizer, axis=1))
        Q_reg = betaQ * np.sum(np.linalg.norm(state.Q, ord=self.regularizer, axis=1))
        return float(mismatch), float(W_reg + Q_reg)

    def _fit_core(self, data: MatrixData, state: FactorState, update_Q: bool):
        """
        Run the main ICQF optimization loop.

        At each iteration we apply:
        1. `W` update with `Q, Qc, Z, aZ` fixed
        2. `Q, Qc` update with `W, Z, aZ` fixed (optional during `transform`)
        3. closed-form `Z` update
        4. scaled dual update `aZ <- aZ + rho (Z - B)`

        where `B = W Q^T + C Qc^T` is the current reconstruction.
        """
        data.M = check_array(data.M, order="C", copy=False, dtype=np.float64)
        data.nan_mask = check_array(data.nan_mask, order="C", copy=False, dtype=np.float64)
        C = None if data.C is None else check_array(data.C, order="C", copy=False, dtype=np.float64)

        state.W = check_array(state.W, order="C", copy=False, dtype=np.float64)
        state.Q = check_array(state.Q, order="C", copy=False, dtype=np.float64)
        state.Z = check_array(state.Z, order="C", copy=False, dtype=np.float64)
        state.aZ = check_array(state.aZ, order="C", copy=False, dtype=np.float64)
        if C is not None:
            state.Qc = check_array(state.Qc, order="C", copy=False, dtype=np.float64)

        self._ensure_C_and_Qc(data, state)

        betaW = float(self.W_beta)
        if self.W_upperbd[0]:
            magnitude_ratio = (self.Q_upperbd[1] / self.W_upperbd[1]) if self.Q_upperbd[0] else (1.0 / self.W_upperbd[1])
        else:
            magnitude_ratio = 1.0
        betaQ = float(self.Q_beta) * state.W.shape[0] / state.Q.shape[0] * magnitude_ratio

        history: List[Tuple[float, float, float]] = []
        pbar = tqdm(range(self.max_iter), disable=not self.verbose, desc="ICQF", leave=True)

        for it in pbar:
            B = state.Z + state.aZ / self.rho
            if C is not None:
                B = B - C @ state.Qc.T
            W_prev = state.W.copy() if self.verbose else None
            self._update_W_inplace(state, B, betaW / self.rho)

            B = state.Z + state.aZ / self.rho
            Q_prev = state.Q.copy() if (self.verbose and update_Q) else None
            if update_Q:
                self._update_Q_inplace(state, B, C, betaQ / self.rho)

            reconstruction = state.W @ state.Q.T
            if C is not None:
                reconstruction = reconstruction + C @ state.Qc.T

            Z_prev = state.Z.copy() if self.verbose else None
            self._update_Z_inplace(data, state, reconstruction - state.aZ / self.rho)
            state.aZ += self.rho * (state.Z - reconstruction)

            mismatch, reg = self._obj_terms(data, state, betaW, betaQ)
            total = mismatch + reg
            history.append((total, mismatch, reg))

            relchg = (
                abs(history[-1][0] - history[-2][0]) / (abs(history[-1][0]) + 1e-12)
                if it > 0 else np.nan
            )

            if self.verbose:
                postfix = {
                    "loss": f"{total:.2e}",
                    "rel": f"{relchg:.1e}" if np.isfinite(relchg) else "nan",
                }
                if W_prev is not None:
                    postfix["dW"] = f"{np.linalg.norm(state.W - W_prev):.1e}"
                if Z_prev is not None:
                    postfix["dZ"] = f"{np.linalg.norm(state.Z - Z_prev):.1e}"
                if update_Q and Q_prev is not None:
                    postfix["dQ"] = f"{np.linalg.norm(state.Q - Q_prev):.1e}"
                pbar.set_postfix(postfix, refresh=False)

            if it > self.min_iter and relchg < self.tol:
                break

        if self.verbose:
            pbar.close()

        return state, history

    def _fit_transform_data(self, data: MatrixData, svd_components=None):
        state = self._init_state(data, svd_components=svd_components)
        state, history = self._fit_core(data, state, update_Q=True)
        state = self._attach_dataframe_views(data, state)
        self.state_ = state
        self.loss_history_ = history
        self._cache_training_for_transform(data, state)
        return state, history

    def fit_transform(
        self,
        X: MatrixData,
        *,
        svd_components=None,
    ):
        """
        Fit ICQF on a preconstructed `MatrixData` object.

        Keeping construction outside the solver makes it easy to inspect row/column labels,
        missingness patterns, and confounds before optimization begins.
        """
        data = self._require_matrixdata(X)
        return self._fit_transform_data(data, svd_components=svd_components)

    def fit(self, X: MatrixData, y=None, **kwargs):
        self.fit_transform(X, **kwargs)
        return self

    def _init_transform_state(self, data: MatrixData, init_state: Optional[FactorState]) -> FactorState:
        if self.state_ is None:
            raise RuntimeError("ICQF.transform() called before fit().")

        Q = np.asarray(self.state_.Q, dtype=np.float64, order="C").copy()
        Qc = None if self.state_.Qc is None else np.asarray(self.state_.Qc, dtype=np.float64, order="C").copy()
        _clip_inplace(Q, 0.0, self.Q_upperbd[1] if self.Q_upperbd[0] else None)

        C = None if data.C is None else check_array(data.C, order="C", copy=False, dtype=np.float64)
        if C is not None:
            if Qc is None:
                raise ValueError("New data has C, but trained model has no Qc.")
            if Qc.shape[1] != C.shape[1]:
                raise ValueError(f"New data C has {C.shape[1]} columns but trained Qc has {Qc.shape[1]}.")
            _clip_inplace(Qc, 0.0, self.Q_upperbd[1] if self.Q_upperbd[0] else None)
        else:
            Qc = None

        if init_state is not None:
            W = np.asarray(init_state.W, dtype=np.float64, order="C").copy()
            if W.shape != (data.M.shape[0], self.n_components):
                raise ValueError(f"transform init_state.W shape mismatch: expected {(data.M.shape[0], self.n_components)}, got {W.shape}")
            Z = None if init_state.Z is None else np.asarray(init_state.Z, dtype=np.float64, order="C").copy()
        else:
            W = None
            Z = None

        if W is None:
            if self._train_data_M_ is None or self._train_W_ is None:
                raise RuntimeError("Missing cached training data for transform initialization.")
            self._vprint("KNN init for W (transform)")
            imputer = KNNImputer(n_neighbors=self.transform_knn_neighbors, weights="uniform")
            imputer.fit(self._train_data_M_)
            M_new = data.M.copy()
            M_new[data.nan_mask == 0] = np.nan
            M_new_imp = imputer.transform(M_new)
            dist = pairwise_distances(self._train_data_M_, M_new_imp, metric="nan_euclidean")
            nn = np.nanargmin(dist, axis=0)
            W = self._train_W_[nn, :].copy()

        _clip_inplace(W, 0.0, self.W_upperbd[1] if self.W_upperbd[0] else None)

        if Z is None:
            Z = W @ Q.T
            if C is not None and Qc is not None:
                Z = Z + C @ Qc.T
        _clip_inplace(Z, self.Mmin, self.Mmax)

        return FactorState(
            W=np.asarray(W, dtype=np.float64, order="C"),
            Q=Q,
            Qc=Qc,
            Z=np.asarray(Z, dtype=np.float64, order="C"),
            aZ=np.zeros_like(data.M, dtype=np.float64, order="C"),
        )

    def _transform_data(self, data: MatrixData, init_state: Optional[FactorState] = None):
        state = self._init_transform_state(data, init_state)
        state, history = self._fit_core(data, state, update_Q=False)
        state = self._attach_dataframe_views(data, state)
        return state, history

    def transform(
        self,
        X: MatrixData,
        *,
        init_state: Optional[FactorState] = None,
    ):
        """
        Infer `W` and `Z` for a new `MatrixData` object while keeping `Q` and `Qc` fixed.

        The caller may optionally provide a `FactorState` carrying an initial `W` and/or `Z`.
        """
        data = self._require_matrixdata(X)
        return self._transform_data(data, init_state=init_state)

    def _cache_training_for_transform(self, data: MatrixData, state: FactorState) -> None:
        """Cache training data with NaNs restored for KNN-based transform initialization."""
        M_train = data.M.copy()
        M_train[data.nan_mask == 0] = np.nan
        self._train_data_M_ = M_train
        self._train_W_ = np.asarray(state.W, dtype=np.float64, order="C").copy()
