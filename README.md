# ICQF

ICQF factorizes an observed matrix `M ≈ W Qᵀ` (optionally adjusting for fully observed confounds `C` via `M ≈ W Qᵀ + C Qcᵀ`) under nonnegativity and optional upper-bound (interval) constraints, using an ADMM-style solver. Missing entries are handled natively: `NaN` means "unobserved" and is simply skipped in the loss.

## Install / requirements

The package is pure-Python except for one small Cython coordinate-descent kernel (`_cdnmf_fast_cy`). It is **compiled to a native extension at install time**, so the *runtime* needs neither Cython nor a C compiler — only a C compiler at *install* time (already standard on most systems; `gcc`/`clang`).

Install with [uv](https://docs.astral.sh/uv/):

```bash
# from this folder
uv pip install .

# or, for an editable/dev install
uv pip install -e .

# optional extras for the demo notebooks (scipy, matplotlib, jupyter)
uv pip install ".[examples]"
```

Runtime dependencies (`numpy`, `pandas`, `scikit-learn`, `tqdm`) are resolved
automatically; Cython and NumPy headers are pulled into an isolated build
environment by uv and are *not* installed into your project. After installing:

```python
from icqf import ICQF, MatrixData, estimate_dimension
```

> **Note on platforms:** the compiled `.so`/`.pyd` is platform- and
> Python-version-specific, not universal. That's fine — building from source
> (the command above) recompiles the kernel for whatever machine you install
> on. The `.pyx` source ships in the sdist so each install builds its own.
> If a checkout has *not* been built (e.g. running straight from the source
> tree), the import falls back to `pyximport`, which compiles the kernel on
> first import and does require Cython + a compiler at runtime.

## Usage

```python
import pandas as pd
from icqf import ICQF, MatrixData

# samples x features; NaN = unobserved
M_df = pd.read_csv("my_data.csv", index_col=0)

data  = MatrixData.from_frames(M_df)          # validate + package
model = ICQF(n_components=5, random_state=0)
state, history = model.fit_transform(data)

state.W_df   # (samples  x factors)  subject scores
state.Q_df   # (features x factors)  item loadings
state.Z_df   # (samples  x features) bounded reconstruction

# apply the learned Q to new rows (Q stays fixed, only W/Z are inferred)
new_state, _ = model.transform(MatrixData.from_frames(M_new_df))
```

### Estimating the embedding dimension (BCV)

Dimension is chosen by **bi-cross-validation**: held-out observed entries are
masked to `NaN`, ICQF is fit on the rest, and reconstruction error is measured
on the held-out entries. The rank that minimizes held-out error (or the elbow of
the curve) is the estimate.

```python
from icqf import estimate_dimension

# Auto: parallel analysis centers the rank-search grid, the SE rule picks best_k.
results_df, summary_df, best_k = estimate_dimension(
    M_df,
    dimension_list=None,        # None -> grid centered on the PA estimate
    J=10, repeat=2, random_state=42,
    icqf_kwargs=dict(W_beta=0.1, Q_beta=0.1, max_iter=200),
)
print(best_k)
```

You can still pass an explicit `dimension_list=[2, 3, 4, 5, 6, 8, 10]` to sweep a
fixed range.

**Two refinements (both optional, both on by default):**

- **Parallel analysis to center the search.** A permutation parallel analysis on
  the data matrix gives a cheap first guess of the rank; when
  `dimension_list is None` the search grid is a window `pa_dim ± pa_window`
  centered on it, instead of an arbitrary wide sweep. (`parallel_analysis`,
  `grid_from_parallel_analysis`.)
- **The SE rule for selection.** The literal argmin of a noisy CV curve jumps
  around in flat regions. The 1-SE rule (Breiman et al. 1984, *CART*) takes the
  *simplest* rank whose mean CV loss is within one standard error of the
  minimum. Because the same folds score every rank, the SE is computed
  **fold-paired** (each fold's curve is centered before taking the SE), which
  removes the fold-difficulty offset and gives a tighter, fairer band. The
  tolerance scales with `se_mult` (`1.0` = classic 1-SE, `0.5` = softer,
  `0.0` = argmin). Toggle with `use_1se`, `paired_se`, `se_mult`.
  (`select_dimension`.)

**Threads.** Each ICQF fit leans on multi-threaded BLAS. `n_threads="auto"` (default) caps it at *all cores but one* so the machine stays responsive; pass an int to pin a specific count, or `None` to leave the global pool untouched.

This runs **serially over folds in the current process** — there is no cluster/swarm dispatch. For a single survey-sized matrix the full sweep is fast enough to run inline; large-scale parallel dispatch is intentionally left out of this public standalone.

#### Also tuning the regularization (`W_beta`, `Q_beta`) by CV

`estimate_dimension` fixes the penalties (from `icqf_kwargs`) and sweeps only the rank. To choose the dimension **and** the L1 penalties together, use `estimate_dimension_grid`, which sweeps the full grid `dimension_list × W_beta_list × Q_beta_list` on shared folds:

```python
from icqf import estimate_dimension_grid

results_df, summary_df, best = estimate_dimension_grid(
    M_df,
    dimension_list=[3, 4, 5, 6, 8],
    W_beta_list=[0.001, 0.01, 0.1],
    Q_beta_list=[0.001, 0.01, 0.1, 0.5],
    J=10, repeat=2, random_state=42,
    icqf_kwargs=dict(regularizer=1, max_iter=200),
)
print(best)   # {"n_components": ..., "W_beta": ..., "Q_beta": ...}
# summary_df: one row per (n_components, W_beta, Q_beta) with paired-SE columns
```

The fold masks are built **once** and reused across every `(k, W_beta, Q_beta)`, so the comparison is on identical splits. The same 1-SE machinery applies, with the standard error computed **per configuration**; on the joint grid "simplest" means smallest rank first, then most-regularized (largest `W_beta`, then `Q_beta`). The masks depend only on `(M_df, J, repeat, random_state)` — never on the betas or ranks — so both functions return them via `summary_df.attrs["folds"]` (also buildable with `build_bcv_folds`); pass them back through `folds=` to extend a sweep on the *same* splits:

```python
folds = summary_df.attrs["folds"]            # reuse identical folds
_, more, _ = estimate_dimension_grid(M_df, [7, 9], W_beta_list=[0.01], Q_beta_list=[0.01], folds=folds)
```

### Confounds: additive baseline vs. regressing out

If you pass `C_df`, ICQF fits `M ≈ W Qᵀ + C Qcᵀ` and treats the confound as a
**non-negative additive baseline** `C Qcᵀ ≥ 0`, so the part the factors explain,
`W Qᵀ = Z − C Qcᵀ ≥ 0`, stays inside the data's non-negative range. This differs
from the conventional "regress the confound out first, then factorize the
residuals" approach, whose residuals are mean-centered and routinely go negative.

Because the confound loadings are non-negative, a single `age` column can only
push the baseline *up* with age. To allow a *decreasing* trend too, encode the
complement: use `C = [intercept, age, 1 − age]`. Non-negative combinations then
span any straight-line baseline (constant, increasing, or decreasing). Note also
that the non-negative baseline behaves like a **lower envelope** (≈ per-item
minimum), not a conditional mean — a few zero entries pull it toward 0; a richer
basis and lighter `Q_beta` loosen this. Section 6 of the demo notebook builds a
confounded example (via `_generate_examples.py`) and visualizes the contrast.

See `demo_ICQF.ipynb` for a worked, runnable walkthrough.

## Layout

```
ICQF_df/
├── README.md
├── demo_ICQF.ipynb           # training, transforming, BCV, confound handling
├── _generate_examples.py     # synthetic-data helpers (used by the notebook's confound demo)
├── testing/                  # verification scripts & BCV studies (not part of the library)
└── icqf/
    ├── __init__.py
    ├── ICQF.py              # the model (fit_transform / transform)
    ├── utils_data.py        # MatrixData (input) + FactorState (output)
    ├── utils_SVD.py         # NNDSVD initialization
    ├── bcv.py               # single-process BCV: PA grid + fold-paired SE rule
    └── _cdnmf_fast_cy.pyx   # Cython coordinate-descent kernel (compiled on import)
```
