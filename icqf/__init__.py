"""
icqf -- standalone, DataFrame-friendly Interval-Constrained Quantile Factorization.

Quick start
-----------
    import pandas as pd
    from icqf import ICQF, MatrixData

    M_df = pd.read_csv("my_data.csv", index_col=0)   # samples x features, NaN = missing
    data = MatrixData.from_frames(M_df)

    model = ICQF(n_components=5, random_state=0)
    state, history = model.fit_transform(data)

    state.W_df   # (samples x factors) subject scores
    state.Q_df   # (features x factors) item loadings
    state.Z_df   # (samples x features) bounded reconstruction

Dimension selection (bi-cross-validation, single process -- no cluster):
    from icqf import estimate_dimension
    results, summary, best_k = estimate_dimension(M_df, [2, 3, 4, 5, 6, 8])
"""

from .ICQF import ICQF
from .utils_data import MatrixData, FactorState
from .bcv import (
    make_bcv_masks,
    build_bcv_folds,
    apply_train_mask,
    reconstruction_errors,
    parallel_analysis,
    grid_from_parallel_analysis,
    select_dimension,
    select_grid,
    estimate_dimension,
    estimate_dimension_grid,
    default_n_threads,
)

__all__ = [
    "ICQF",
    "MatrixData",
    "FactorState",
    "make_bcv_masks",
    "build_bcv_folds",
    "apply_train_mask",
    "reconstruction_errors",
    "parallel_analysis",
    "grid_from_parallel_analysis",
    "select_dimension",
    "select_grid",
    "estimate_dimension",
    "estimate_dimension_grid",
    "default_n_threads",
]
