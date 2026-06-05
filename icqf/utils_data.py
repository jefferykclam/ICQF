from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

Array = np.ndarray


def _index_to_str_object_array(idx: pd.Index) -> np.ndarray:
    """Convert pandas Index to numpy array of Python strings (dtype=object)."""
    return idx.astype(str).to_numpy(dtype=object)


@dataclass
class MatrixData:
    """
    Validated container holding the data ICQF factorizes.

    Contract:
    - M_df must be a pandas DataFrame (NaNs allowed; NaN means "unobserved").
      Normalize its values into [0, 1] before fitting: ICQF's defaults bound the
      reconstruction and the Q loadings at 1 (M_upperbd / Q_upperbd), which assume
      an [0, 1]-scaled M.
    - C_df is optional; if provided, must be a pandas DataFrame, already including
      any intercept column, and must be fully observed (no NaNs).
    - sample_index / feature_index are always present (strings, from M_df).
    - confound_feature_index present only if C_df is provided.

    Stored arrays:
    - M: float64 (n, p) with NaNs replaced by 0
    - nan_mask: float64 (n, p) in {0,1} indicating observed entries in M_df
    - C: float64 (n, c) if provided, else None
    """
    M: Array
    nan_mask: Array
    sample_index: np.ndarray
    feature_index: np.ndarray

    C: Optional[Array] = None
    confound_feature_index: Optional[np.ndarray] = None

    @staticmethod
    def from_frames(
        M_df: pd.DataFrame,
        C_df: Optional[pd.DataFrame] = None,
        *,
        ensure_numeric: bool = True,
    ) -> "MatrixData":
        if not isinstance(M_df, pd.DataFrame):
            raise TypeError(f"`M_df` must be a pandas DataFrame, got {type(M_df)}")

        # ---- uniqueness checks on M labels ----
        if not M_df.index.is_unique:
            dup = _index_to_str_object_array(M_df.index[M_df.index.duplicated()])[:10].tolist()
            raise ValueError(
                "M_df.index must be unique (no duplicate sample IDs). "
                f"Example duplicates (up to 10): {dup}"
            )
        if not M_df.columns.is_unique:
            dup = _index_to_str_object_array(M_df.columns[M_df.columns.duplicated()])[:10].tolist()
            raise ValueError(
                "M_df.columns must be unique (no duplicate feature names). "
                f"Example duplicates (up to 10): {dup}"
            )

        # labels (always stored)
        sample_index = _index_to_str_object_array(M_df.index)
        feature_index = _index_to_str_object_array(M_df.columns)

        # numeric coercion
        M_work = M_df.apply(pd.to_numeric, errors="coerce") if ensure_numeric else M_df

        # ---- M checks: no fully-NaN rows or columns ----
        row_all_nan = M_work.isna().all(axis=1)
        if row_all_nan.any():
            bad = _index_to_str_object_array(M_work.index[row_all_nan])[:10].tolist()
            raise ValueError(
                "M_df has one or more rows that are entirely NaN after numeric coercion. "
                f"Example rows (up to 10): {bad}"
            )

        col_all_nan = M_work.isna().all(axis=0)
        if col_all_nan.any():
            bad = _index_to_str_object_array(M_work.columns[col_all_nan])[:10].tolist()
            raise ValueError(
                "M_df has one or more columns that are entirely NaN after numeric coercion. "
                f"Example columns (up to 10): {bad}"
            )

        # ---- build M + nan_mask ----
        M_raw = M_work.to_numpy(dtype=np.float64, copy=False)
        nan_mask = (~np.isnan(M_raw)).astype(np.float64, copy=False)
        M = np.nan_to_num(M_raw, nan=0.0).astype(np.float64, copy=False)

        # enforce C-order for downstream cython calls
        M = np.asarray(M, order="C", dtype=np.float64)
        nan_mask = np.asarray(nan_mask, order="C", dtype=np.float64)

        # ---- optional C ----
        C = None
        confound_feature_index = None

        if C_df is not None:
            if not isinstance(C_df, pd.DataFrame):
                raise TypeError(f"`C_df` must be a pandas DataFrame when provided, got {type(C_df)}")

            # uniqueness checks on C labels
            if not C_df.index.is_unique:
                dup = _index_to_str_object_array(C_df.index[C_df.index.duplicated()])[:10].tolist()
                raise ValueError(
                    "C_df.index must be unique (no duplicate sample IDs). "
                    f"Example duplicates (up to 10): {dup}"
                )
            if not C_df.columns.is_unique:
                dup = _index_to_str_object_array(C_df.columns[C_df.columns.duplicated()])[:10].tolist()
                raise ValueError(
                    "C_df.columns must be unique (no duplicate confound names). "
                    f"Example duplicates (up to 10): {dup}"
                )

            # strict sample index match (labels + order)
            if not C_df.index.equals(M_df.index):
                raise ValueError("C_df.index must exactly equal M_df.index (same labels and order).")

            confound_feature_index = _index_to_str_object_array(C_df.columns)

            C_work = C_df.apply(pd.to_numeric, errors="coerce") if ensure_numeric else C_df
            C_raw = C_work.to_numpy(dtype=np.float64, copy=False)

            # fully observed check
            if np.isnan(C_raw).any():
                na_pos = np.argwhere(np.isnan(C_raw))
                examples = []
                for r, c in na_pos[:10]:
                    examples.append((str(M_df.index[r]), str(C_df.columns[c])))
                raise ValueError(
                    "C_df must be fully observed (no NaNs) after numeric coercion. "
                    f"Example missing entries (row_id, confound_name) up to 10: {examples}"
                )

            C = np.asarray(C_raw, order="C", dtype=np.float64)

        return MatrixData(
            M=M,
            nan_mask=nan_mask,
            sample_index=sample_index,
            feature_index=feature_index,
            C=C,
            confound_feature_index=confound_feature_index,
        )


@dataclass
class FactorState:
    """
    Factorization state for a single matrix factorization problem.

    Core NumPy arrays used by the optimizer
    --------------------------------------
    W  : (n, k) subject scores / left factor
    Q  : (p, k) item loadings / right factor
    Qc : (p, c') confound loadings when a design matrix C is present
    Z  : (n, p) auxiliary reconstruction variable in the ADMM splitting
    aZ : (n, p) scaled dual variable associated with the constraint Z = W Q^T (+ C Qc^T)

    Optional convenience views populated by ICQF after fitting
    ---------------------------------------------------------
    W_df  : DataFrame view of W indexed by sample IDs
    Q_df  : DataFrame view of Q indexed by feature IDs
    Qc_df : DataFrame view of Qc indexed by feature IDs
    Z_df  : DataFrame view of Z aligned to M_df
    aZ_df : DataFrame view of aZ aligned to M_df
    """
    W: Array
    Q: Array
    Qc: Optional[Array]
    Z: Array
    aZ: Array

    W_df: Optional[pd.DataFrame] = None
    Q_df: Optional[pd.DataFrame] = None
    Qc_df: Optional[pd.DataFrame] = None
    Z_df: Optional[pd.DataFrame] = None
    aZ_df: Optional[pd.DataFrame] = None
