from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import numpy as np
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot


def _nndsvd_init(
    M: np.ndarray,
    n_components: int,
    *,
    svd_components: Optional[Sequence[np.ndarray]] = None,
    random_state: Optional[int] = None,
    n_iter: int = 5,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    NNDSVD initialization (as used in sklearn NMF), producing nonnegative factors.

    This method computes (or reuses) a truncated SVD:
        M ≈ U diag(S) Vt
    and then constructs nonnegative matrices W and H such that:
        M ≈ W H

    Parameters
    ----------
    M:
        Nonnegative (or approximately nonnegative) data matrix of shape (n_samples, n_features).
        This function does not clip negatives; do that upstream if needed.
    n_components:
        Target rank / number of components (k).
    svd_components:
        Optional cached SVD pieces to reuse. Expected as (S, Vt) where:
          - S has shape (k,)
          - Vt has shape (k, n_features)
        If provided, U is recomputed as M V / S[0] (consistent with your original approach).
    random_state:
        Seed for randomized SVD when `svd_components` is None.
    n_iter:
        Power iterations for randomized SVD. Small is fine for initialization.
    eps:
        Values smaller than eps are set to 0 for numerical stability.

    Returns
    -------
    W:
        Nonnegative matrix of shape (n_samples, k).
    H:
        Nonnegative matrix of shape (k, n_features).
    cache:
        A list [S, Vt] that can be reused as `svd_components` in later calls.

    Notes
    -----
    - Leading singular value S[0] is nonnegative.
    - For each component j>0, we split singular vectors into positive/negative parts
      and keep the part with larger product of norms, following NNDSVD.
    """
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_components > min(M.shape):
        raise ValueError(
            f"n_components={n_components} cannot exceed min(M.shape)={min(M.shape)}."
        )

    # Compute or reuse truncated SVD pieces.
    if svd_components is None:
        U, S, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
        )
    else:
        if len(svd_components) != 2:
            raise ValueError("svd_components must be a sequence (S, Vt).")
        S, Vt = svd_components
        S = np.asarray(S)
        Vt = np.asarray(Vt)

        if S.shape != (n_components,):
            raise ValueError(f"S must have shape ({n_components},), got {S.shape}.")
        if Vt.shape[0] != n_components or Vt.shape[1] != M.shape[1]:
            raise ValueError(
                f"Vt must have shape ({n_components}, {M.shape[1]}), got {Vt.shape}."
            )
        if S[0] == 0:
            raise ValueError("Leading singular value S[0] is zero; cannot form U.")

        # Match your original reuse strategy.
        U = safe_sparse_dot(M, Vt.T) / S[0]

    n_samples, n_features = M.shape
    k = n_components

    W = np.zeros((n_samples, k), dtype=M.dtype)
    H = np.zeros((k, n_features), dtype=M.dtype)

    # First component: absolute values (guaranteed nonnegative).
    s0 = np.sqrt(S[0])
    W[:, 0] = s0 * np.abs(U[:, 0])
    H[0, :] = s0 * np.abs(Vt[0, :])

    # Remaining components: choose positive or negative part with larger "energy".
    for j in range(1, k):
        u = U[:, j]
        v = Vt[j, :]

        u_pos = np.maximum(u, 0.0)
        v_pos = np.maximum(v, 0.0)
        u_neg = np.maximum(-u, 0.0)  # = abs(min(u,0))
        v_neg = np.maximum(-v, 0.0)

        u_pos_nrm = np.linalg.norm(u_pos)
        v_pos_nrm = np.linalg.norm(v_pos)
        u_neg_nrm = np.linalg.norm(u_neg)
        v_neg_nrm = np.linalg.norm(v_neg)

        m_pos = u_pos_nrm * v_pos_nrm
        m_neg = u_neg_nrm * v_neg_nrm

        # Handle degenerate cases safely (avoid division by zero).
        if m_pos == 0 and m_neg == 0:
            continue

        if m_pos >= m_neg:
            u_chosen = u_pos / (u_pos_nrm + 1e-20)
            v_chosen = v_pos / (v_pos_nrm + 1e-20)
            sigma = m_pos
        else:
            u_chosen = u_neg / (u_neg_nrm + 1e-20)
            v_chosen = v_neg / (v_neg_nrm + 1e-20)
            sigma = m_neg

        scale = np.sqrt(S[j] * sigma)
        W[:, j] = scale * u_chosen
        H[j, :] = scale * v_chosen

    # Sparsify tiny numerical noise.
    if eps > 0:
        W[W < eps] = 0.0
        H[H < eps] = 0.0

    return W, H, [S, Vt]


def _svd_initialize(
    M: np.ndarray,
    n_components: int,
    *,
    svd_components: Optional[Sequence[np.ndarray]] = None,
    random_state: Optional[int] = None,
    clip_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience initializer returning (W, Q) where Q is the transpose of NNDSVD's H.

    Parameters
    ----------
    M:
        Input matrix, shape (n_samples, n_features). Must not contain NaNs.
    n_components:
        Rank / number of components.
    svd_components:
        Optional cached SVD pieces (S, Vt) for reuse.
    random_state:
        Seed for randomized SVD when `svd_components` is None.
    clip_negative:
        If True, negatives are clipped to 0 before initializing.

    Returns
    -------
    W:
        Nonnegative left factor of shape (n_samples, k).
    Q:
        Nonnegative right factor of shape (n_features, k) such that M ≈ W Q^T.
    """
    if np.isnan(M).any():
        raise ValueError("M contains NaNs; initialization requires finite values.")

    X = np.asarray(M)
    if clip_negative:
        # Do NOT mutate caller array.
        X = X.copy()
        X[X < 0] = 0.0

    W, H, _ = _nndsvd_init(
        X,
        n_components,
        svd_components=svd_components,
        random_state=random_state,
    )
    Q = H.T
    return W, Q



# import os
# import pickle
# import copy
# import time

# import numpy as np
# import itertools

# import pandas as pd

# import sklearn
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
# from sklearn.utils import check_random_state, check_array

# from sklearn.model_selection import KFold, StratifiedKFold

# import warnings
# from tqdm import tqdm, trange

# from .data_class import matrix_class


# ### Initialization

# # we adopted the NNDSVD initialization as in sklearn NMF
# def _NNDSVD(M,
#             n_components,
#             svd_components=None,
#             random_state=None,
#             n_iter=5, # could be smaller as it is just for initialization
#             eps=1e-6):
    
#     if svd_components is None:
#         U, S, VT = randomized_svd(M, n_components,
#                                  n_iter=n_iter,
#                                  random_state=random_state)
#     else:
#         assert len(svd_components) == 2
#         S = svd_components[0]
#         VT = svd_components[1]
#         U = safe_sparse_dot(M, VT.T) / S[0] # leading singular value is non-negative
    
#     W, H = np.zeros(U.shape), np.zeros(VT.shape)

#     W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
#     H[0, :] = np.sqrt(S[0]) * np.abs(VT[0, :])
#     for j in range(1, n_components):
#         x, y = U[:, j], VT[j, :]
#         x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
#         x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
#         x_p_nrm, y_p_nrm = np.linalg.norm(x_p), np.linalg.norm(y_p)
#         x_n_nrm, y_n_nrm = np.linalg.norm(x_n), np.linalg.norm(y_n)
#         m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
#         if m_p > m_n:
#             u = x_p / x_p_nrm
#             v = y_p / y_p_nrm
#             sigma = m_p
#         else:
#             u = x_n / x_n_nrm
#             v = y_n / y_n_nrm
#             sigma = m_n
#         lbd = np.sqrt(S[j] * sigma)
#         W[:, j] = lbd * u
#         H[j, :] = lbd * v
#     W[W < eps] = 0
#     H[H < eps] = 0
#     return W, H, [S, VT]

# def _SVD_initialize(M,
#                     n_components,
#                     svd_components=None,
#                     random_state=None):
    
#     assert np.sum(np.isnan(M)) == 0
#     M[M < 0] = 0
#     W, QT, svd_components = _NNDSVD(M,
#                                     n_components,
#                                     svd_components=svd_components,
#                                     random_state=random_state,
#                                     )
#     Q = QT.T
#     return W, Q
