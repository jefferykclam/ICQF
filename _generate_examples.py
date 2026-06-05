import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


# ---------- Utilities ----------
def _rbf_kernel(X, bw):
    D2 = cdist(X, X, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * bw**2))

def _self_tuned_bandwidths(X, k=7):
    # Zelnik-Manor & Perona self-tuning: sigma_i = distance to k-th NN
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    sig = np.sort(D, axis=1)[:, k-1]
    sig[sig == 0] = np.median(sig[sig > 0])
    return sig

def _symmetrize(S):
    S = 0.5*(S + S.T)
    # small numerical clip
    S[S < 0] = 0.0
    return S


# simpliest step patten

def simulation(nrow, ncol, ndict=10, overlap=None, density=0.3,
               Q_upperbd=100, lowerbd=0,
               noise=True, delta=0.3,
               missing_ratio = 0.0,
               confound=False,
               visualize=True):
    
    if overlap is None:
        overlap = int(nrow//40)
    
    D = np.zeros((nrow, ndict))
    for k in range(ndict):
        for j in range(nrow):
            if (j >= (k)*( (nrow//ndict) )) and (j <= (nrow//ndict) + overlap + (k)*((nrow//ndict))):
                D[j,k] += np.random.uniform(0.5, 1.0)*np.random.binomial(1, p=0.9)
            else:
                D[j,k] = 0
    D[D > 1] = 1
    return D


    
# ===========================================================
# 1) Degree-Corrected SBM (DCSBM) with RBF edge weights
#    - Few blocks (low latent dimension), but heavy-tailed degrees and hubs
#      make the spectrum slowly decaying -> AIC/BIC over-select.
# ===========================================================

"""
- using a degree-corrected SBM (DCSBM) with continuous RBF edge wegihts
- n nodes are assigned into K latent communities, each represented by a 2D latent center with Gaussian perturbations.
- Base affinities are computed via an RBF kernel on latent positions and modulated by multiplicative Pareto-distribution degree weights theta_i with tail index parameter tau, to induce hubs and heavy-tailed degree heterogenity, i.e. S_ij = theta_i k(x_i, x_j) theta_j
- then symmetrizzed and rw-normalized to enforce graph-like stochasticity.
- slow spectral decay from hubs makes AIC, BIC prone to over-estimaton
"""

def make_similarity_dcsbm(n=300, K=3, block_sep=3.0, tail_strength=1.5, seed=0):
    """
    n: nodes, K: communities (true low latent dim ~ K),
    block_sep: cluster separation in latent space,
    tail_strength: Pareto tail for degree heterogeneity (>=1 is heavier)
    """
    rng = np.random.default_rng(seed)
    # Assign communities
    sizes = rng.multinomial(n, np.ones(K)/K)
    z = np.hstack([np.full(s, k) for k, s in enumerate(sizes)])

    # Community centers in 2D for realism
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    centers = np.c_[block_sep*np.cos(angles), block_sep*np.sin(angles)]

    # Node latent positions with within-block scatter
    X = np.zeros((n, 2))
    for k in range(K):
        idx = np.where(z == k)[0]
        X[idx] = centers[k] + 0.6*rng.standard_normal((len(idx), 2))

    # Degree heterogeneity: Pareto-ish multipliers
    theta = (1 + rng.pareto(tail_strength, size=n))
    theta = theta / np.mean(theta)

    # Base RBF similarity then degree-correct
    S0 = _rbf_kernel(X, bw=1.0)
    S = (theta[:, None] * S0) * theta[None, :]  # degree correction
    S = _symmetrize(S)

    # Row-stochastic smoothing + symmetrize again (more graph-like)
    D = np.maximum(S.sum(axis=1, keepdims=True), 1e-12)
    S = S / D
    S = _symmetrize(S)

    true_dim = K  # communities -> low “intrinsic” dimension
    return S, true_dim, {"story": "DCSBM with degree heterogeneity (hubs) and smooth weights"}

# ===========================================================
# 2) Mixture of manifolds with mismatched local scales + self-tuned kernels
#    - Two low-d manifolds (lines/curves) with very different densities.
#      Self-tuning bandwidths stabilize local geometry but create multi-scale
#      spectra with many near-signal eigenvalues.
# ===========================================================
"""
- similarity matrices from two low-dimensional manifolds with mismatched local scales: a dense, curved “S”-shaped manifold and a sparse, elongated line 
- Nodes are embedded in R2 with Gaussian noise, and affinities are computed using a self-tuned Gaussian kernel with locally adaptive bandwidths
- preserves local geometry across density regimes but induces a multi-scale spectrum
- with several near-signal eigenvalues beyond the true intrinsic dimension of two
- the kernel is sparsified using k-nearest-neighbor graph 
- aim to creates spectral ambiguity that misleads standard rank-selection criteria.

"""

def make_similarity_multiscale_manifolds(n1=200, n2=200, seed=1):
    rng = np.random.default_rng(seed)

    # Manifold A: dense, curved "S" in 2D
    t1 = rng.uniform(-3, 3, size=n1)
    A = np.c_[t1, np.tanh(t1) + 0.05*rng.standard_normal(n1)] # 0.15

    # Manifold B: sparse, elongated line at different scale
    t2 = rng.uniform(-2, 2, size=n2)
    B = np.c_[5 + 2.5*t2, 3 + 0.4*t2 + 0.05*rng.standard_normal(n2)] # 0.3

    X = np.vstack([A, B])

    # Self-tuned kernel (local bandwidths), creates multi-scale spectrum
    sig = _self_tuned_bandwidths(X, k=10)
    D2 = cdist(X, X, metric="sqeuclidean")
    S = np.exp(-D2 / (sig[:, None]*sig[None, :] + 1e-12))
    S = _symmetrize(S)

    # Optional: sparsify by kNN graph to be more realistic (comment out if undesired)
    # k = 12
    # D = cdist(X, X)
    # nn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    # mask = np.zeros_like(S, dtype=bool)
    # rows = np.arange(S.shape[0])[:, None]
    # mask[rows, nn_idx] = True
    # S = np.where(mask | mask.T, S, 0.0)
    # S = _symmetrize(S)

    true_dim = 2  # two manifolds / factors; low intrinsic structure
    return S, true_dim, {"story": "Two manifolds, different densities and scales; self-tuned kernel + kNN"}

# ===========================================================
# 3) Temporal process with local correlations + a few regime shifts
#    - Think: time series similarities with banded (Toeplitz-like) structure
#      plus sparse change-point “patches”. Low latent regimes (few), but
#      banded+patch structure yields long spectral tails.
# ===========================================================

"""
- similarity matrices that mimic time-series structure by combining local correlations with a small number of latent regimes
- kernel encodes short-range temporal dependencies, while regime shifts are modeled via piecewise-constant latent states with low intrinsic rank
- also introduce sparse localized “patches” resembling transient events or anomalies.
- resultant matirx exhibits banded structure with block perturbations
- the true latent dimension equals the number of regimes, banded correlations and patches yields slowly decaying spectra, producing long spectral tails that confound conventional dimension selection.

"""

def make_similarity_temporal(n=400, regimes=3, band=6, patch_strength=0.6, seed=2):
    rng = np.random.default_rng(seed)

    # Base Toeplitz-like kernel for local time correlation
    t = np.arange(n)
    base = np.exp(-cdist(t[:, None], t[:, None], metric="cityblock") / band)

    # Regime shifts: piecewise constant latent states (low #regimes)
    cuts = np.sort(rng.choice(np.arange(50, n-50), size=regimes-1, replace=False))
    segs = np.split(np.arange(n), cuts)
    L = np.zeros((n, regimes))
    for k, seg in enumerate(segs):
        L[seg, k] = 1.0

    # Low-rank regime similarity
    R = L @ L.T

    # Sparse patches mimicking events/anomalies (not a single vector; localized blocks)
    S = 0.5*base + 0.5*R
    for _ in range(8):
        i = rng.integers(0, n-20)
        j = rng.integers(0, n-20)
        h = rng.integers(6, 18)
        patch = np.outer(np.hanning(h), np.hanning(h))
        S[i:i+h, j:j+h] += patch_strength * patch
        S[j:j+h, i:i+h] += patch_strength * patch.T

    S = _symmetrize(S)
    S = S / (S.max() + 1e-12)
    true_dim = regimes
    return S, true_dim, {"story": "Toeplitz-like local correlations + few regimes + sparse event patches"}


"""

- combine multiple structured components with heterogeneous strengths
- a weak global low-rank factor ensures limited dominance of the leading eigenvalues
- numerous small, overlapping clusters with variable strengths introduce heterogeneous mid-spectrum signals
- a smooth AR(1)-type correlation kernel enforces temporal-like structure
- include heterogeneous uniqueness variances to add noise diversity 
- spectrum exhibits a fat middle bulk with many moderate eigenvalues, causing parallel analysis to overestimate dimension

"""

def make_correlation_horn_fail_challenging(
    n=100,
    r_true=3,
    n_clusters=20,           # more clusters
    cluster_size=5,           # smaller, more numerous
    cluster_strength_range=(0.3, 0.8),  # heterogeneous cluster strengths
    ar_rho=0.5,
    global_factor_strength=2.0,  # weaker global factor
    hetero_psi_scale=0.5,
    seed=0
):
    rng = np.random.default_rng(seed)

    # 1) Global low-rank factor (weaker)
    B = np.abs(rng.normal(loc=0.3, scale=0.4, size=(n, r_true)))
    B *= global_factor_strength / np.sqrt(r_true)
    Sigma_global = B @ B.T

    # 2) Heterogeneous, possibly overlapping clusters
    Sigma_clusters = np.zeros((n, n))
    for _ in range(n_clusters):
        size = rng.integers(max(2, cluster_size-1), cluster_size+2)
        start = rng.integers(0, n - size)
        idx = np.arange(start, start + size)
        v = np.exp(-((np.arange(size) - size//2)**2) / (2.0*(size/3.0)**2))
        block = np.outer(v, v)
        block /= block.max()
        strength = rng.uniform(*cluster_strength_range)
        Sigma_clusters[np.ix_(idx, idx)] += strength * block
    Sigma_clusters = 0.5 * (Sigma_clusters + Sigma_clusters.T)

    # 3) Smooth AR(1) kernel
    i = np.arange(n)
    j = i[:, None]
    Sigma_ar = ar_rho ** np.abs(i - j)

    # 4) Heterogeneous uniqueness
    psi = 0.1 + hetero_psi_scale * np.abs(rng.standard_normal(n))
    Sigma_psi = np.diag(psi)

    # Combine: reduce global dominance
    Sigma = 0.7 * Sigma_global + 0.9 * Sigma_clusters + 0.6 * Sigma_ar + Sigma_psi
    Sigma = 0.5 * (Sigma + Sigma.T)
    vals, vecs = eigh(Sigma)
    vals[vals < 1e-12] = 1e-12
    Sigma = (vecs * vals) @ vecs.T

    # Convert to correlation
    D = np.sqrt(np.diag(Sigma))
    D[D == 0] = 1e-12
    R = Sigma / (D[:, None] * D[None, :])
    R[R < 0] = 0.0

    # Project to PSD and rescale to correlation
    vals, vecs = eigh(R)
    vals[vals < 1e-10] = 0.0
    R_psd = (vecs * vals) @ vecs.T
    D2 = np.sqrt(np.diag(R_psd))
    D2[D2 == 0] = 1e-12
    R_psd = R_psd / (D2[:, None] * D2[None, :])
    R_psd[R_psd < 0] = 0.0
    np.fill_diagonal(R_psd, 1.0)

    meta = {
        "true_rank": int(r_true),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "cluster_size": int(cluster_size),
        "cluster_strength_range": cluster_strength_range,
        "ar_rho": float(ar_rho),
        "global_factor_strength": float(global_factor_strength),
        "hetero_psi_scale": float(hetero_psi_scale),
        "seed": int(seed),
        "notes": "Overlapping heterogeneous clusters + weaker global factors create fat mid-spectrum, challenging Horn's PA"
    }
    return R_psd, meta



# -----------------------------
# Horn's parallel analysis helper
# -----------------------------
def horn_parallel_analysis(R, n_obs=200, n_resamples=200, percentile=95, rng_seed=1):
    """
    A simple Horn-style parallel analysis for a correlation matrix R (p x p).
    Since Horn's original procedure operates on sample correlation matrices
    from observed data with a given sample size, we simulate 'n_resamples'
    random normal datasets of shape (n_obs, p), compute their correlation
    eigenvalues, and take the percentile-th percentile per component as the
    parallel-analysis threshold.

    Returns:
      pa_thresh : ndarray (p,) thresholds per component sorted descending
      eigvals_R : ndarray (p,) eigenvalues of R sorted descending
      guessed_dim : int, number of eigenvalues of R greater than thresh
    """
    rng = np.random.default_rng(rng_seed)
    p = R.shape[0]
    
    # eigenvalues of R
    eigvals_R = np.linalg.eigvalsh(R)[::-1]
    
    # collect resampled eigenvalues
    res_eigs = np.zeros((n_resamples, p))
    for t in range(n_resamples):
        # generate iid normal data with no structure (null) and compute sample correlation
        X = rng.normal(size=(n_obs, p))
        # compute correlation matrix
        Xc = X - X.mean(axis=0, keepdims=True)
        C = np.corrcoef(Xc, rowvar=False)
        e = np.linalg.eigvalsh(C)[::-1]
        res_eigs[t, :] = e
    
    pa_thresh = np.percentile(res_eigs, percentile, axis=0)
    guessed = np.sum(eigvals_R > pa_thresh)
    return pa_thresh, eigvals_R, int(guessed)
