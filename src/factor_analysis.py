import os
import pickle
import copy

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd

from cvxopt import matrix, spmatrix, solvers, sparse, spdiag
import quadprog

import warnings
from tqdm import tqdm, trange

# from utils.matrix_class import matrix_class

from dataclasses import dataclass

from sklearn.decomposition import FactorAnalysis


from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent import futures

from itertools import repeat, starmap
from functools import partial

import numpy as np
from numpy.random import SeedSequence

@dataclass
class matrix_class:

    M : np.ndarray # (column)-normalized data matrix
    M_raw : np.ndarray # raw data matrix
    confound : np.ndarray # normalized confounder matrix
    confound_raw : np.ndarray # raw confounder matrix
    nan_mask : np.ndarray # mask matrix for missing entires (0=missing, 1=available)
    row_idx : np.ndarray # global row index (for multiple data matrices)
    col_idx : np.ndarray # global column index (for multiple data matrices)
    mask : np.ndarray # global mask (for multiple data matrices)
    dataname : str # dataname
    subjlist : list # information on subjects (row information)
    itemlist : list # information on items (column information)
    W : np.ndarray # subject embedding (recall M = [W, C]Q^T)
    Q : np.ndarray # item embedding (recall M = [W, C]Q^T)
    C : np.ndarray # confounder matrix
    Qc : np.ndarray # confounders' loadings (recall Q = [RQ, CQ])
    Z : np.ndarray # auxiliary Z=WQ^T (ADMM)
    aZ : np.ndarray # auxiliary variables (ADMM)
    

class factor_analysis(TransformerMixin, BaseEstimator):
    
    def __init__(
        self,
        n_components=None,
        *,
        rotation='promax',
        random_state=None,
        verbose=False):
        
        self.n_components = n_components
        self.rotation = rotation
        self.random_state = None
        self.verbose = verbose

        self.clf =  FactorAnalysis(n_components=self.n_components, random_state=self.random_state, rotation=None)
        
        
    def initialize(self, matrix_class):
        
        MF_data = copy.deepcopy(matrix_class)
        assert np.sum(np.isnan(MF_data.M)) == 0
        
        if matrix_class.confound is not None:
            C = np.hstack((matrix_class.confound, 1.0-matrix_class.confound))
            C = np.hstack((C, np.ones((C.shape[0], 1))))
            MF_data.C = C
        else:
            MF_data.C = None
        # return matrix_class obj
        return MF_data
    

    def _obj_func(self, MF_data, betaW, betaQ):
    
        R = MF_data.W @ (MF_data.Q.T)

        mismatch_loss = 0.5 * np.sum(MF_data.nan_mask * (MF_data.M - R)**2)
        W_norm = betaW*np.sum(np.linalg.norm(MF_data.W, ord=self.regularizer, axis=1))
        Q_norm = betaQ*np.sum(np.linalg.norm(MF_data.Q, ord=self.regularizer, axis=1))
        reg_loss = W_norm+Q_norm
        return mismatch_loss, reg_loss
    
    def _fit_transform(self, MF_data, update_Q=True):
        
        if update_Q:
            W = self.clf.fit_transform(MF_data.M)
            if self.rotation == 'promax':
                MF_data.Q, Rotate, _ = _promax(self.clf.components_.T)
            elif self.rotation == 'varimax':
                MF_data.Q, Rotate = _varimax(Q)
            else:
                raise ValueError('Unknown rotation/Not implemented')
            MF_data.W = W@Rotate.T
            self.Rotate = Rotate

        else:
            MF_data.W = self.clf.transform(MF_data.M) @ (self.Rotate.T)
        return MF_data

    def fit_transform(self, matrix_class):
        MF_data = self.initialize(matrix_class)
        self.MF_init = copy.deepcopy(MF_data)
        self.MF = self._fit_transform(MF_data, update_Q=True)
        return self.MF

    def fit(self, matrix_class):
        _ = self.fit_transform(matrix_class, update_Q=True)
        
    def transform(self, matrix_class):
        # optimize W with given Q, C, M (and mask)
        
        # make a copy of the trained model (to get Q, C)
        MF_init = copy.deepcopy(self.MF)
        
        MF_init.M = matrix_class.M
        MF_init.M_raw = matrix_class.M_raw
        MF_init.confound = matrix_class.confound
        MF_init.confound_raw = matrix_class.confound_raw
        MF_init.nan_mask = matrix_class.nan_mask
        
        MF_init.row_idx = matrix_class.row_idx
        MF_init.col_idx = matrix_class.col_idx
        MF_init.mask = matrix_class.mask
        
        MF_init.dataname = matrix_class.dataname
        MF_init.subjlist = matrix_class.subjlist
        MF_init.itemlist = matrix_class.itemlist
        
        # add confounds corresponding new data matrix, if applicable
        if matrix_class.confound is not None:
            assert self.MF.confound is not None
            C = np.hstack((matrix_class.confound, 1.0-matrix_class.confound))
            C = np.hstack((C, np.ones((C.shape[0], 1))))
            
            MF_init.C = C
        else:
            assert self.MF.confound is None
            MF_init.C = None

        self.MF_init = copy.deepcopy(MF_init)
        
        # optimize for W
        MF_data = self._fit_transform(MF_init, update_Q=False)
        
        return MF_data


def _varimax(loadings, normalize=False, max_iter=500, tol=1e-5):
    """
    Perform varimax (orthogonal) rotation, with optional
    Kaiser normalization.
    This implementation is adatped from
    https://github.com/EducationalTestingService/factor_analyzer/tree/main

    Parameters
    ----------
    loadings : array-like
        The loading matrix

    Returns
    -------
    loadings : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_mtx : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    """
    X = loadings.copy()
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X, np.array([[1]])

    # normalize the loadings matrix
    # using sqrt of the sum of squares (Kaiser)
    if normalize:
        normalized_mtx = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, X.copy())
        X = (X.T / normalized_mtx).T

    # initialize the rotation matrix
    # to N x N identity matrix
    rotation_mtx = np.eye(n_cols)

    d = 0
    for _ in range(max_iter):

        old_d = d

        # take inner product of loading matrix
        # and rotation matrix
        basis = np.dot(X, rotation_mtx)

        # transform data for singular value decomposition
        transformed = np.dot(X.T, basis**3 - (1.0 / n_rows) *
                             np.dot(basis, np.diag(np.diag(np.dot(basis.T, basis)))))

        # perform SVD on
        # the transformed matrix
        U, S, V = np.linalg.svd(transformed)

        # take inner product of U and V, and sum of S
        rotation_mtx = np.dot(U, V)
        d = np.sum(S)

        # check convergence
        if old_d != 0 and d / old_d < 1 + tol:
            break

    # take inner product of loading matrix
    # and rotation matrix
    X = np.dot(X, rotation_mtx)

    # de-normalize the data
    if normalize:
        X = X.T * normalized_mtx
    else:
        X = X.T

    # convert loadings matrix to data frame
    loadings = X.T.copy()
    return loadings, rotation_mtx

def _promax(loadings, normalize=False, power=4):
    """
    Perform promax (oblique) rotation, with optional
    Kaiser normalization.
    
    This implementation is adatped from
    https://github.com/EducationalTestingService/factor_analyzer/tree/main

    Parameters
    ----------
    loadings : array-like
        The loading matrix

    Returns
    -------
    loadings : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_mtx : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    psi : numpy array or None, shape (n_factors, n_factors)
        The factor correlations
        matrix. This only exists
        if the rotation is oblique.
    """
    X = loadings.copy()
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X, np.array([[1]]), 1

    if normalize:
        # pre-normalization is done in R's
        # `kaiser()` function when rotate='Promax'.
        array = X.copy()
        h2 = np.diag(np.dot(array, array.T))
        h2 = np.reshape(h2, (h2.shape[0], 1))
        weights = array / np.sqrt(h2)

    else:
        weights = X.copy()

    # first get varimax rotation
    X, rotation_mtx = _varimax(weights)
    Y = X * np.abs(X)**(power - 1)

    # fit linear regression model
    coef = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))

    # calculate diagonal of inverse square
    try:
        diag_inv = np.diag(sp.linalg.inv(np.dot(coef.T, coef)))
    except np.linalg.LinAlgError:
        diag_inv = np.diag(sp.linalg.pinv(np.dot(coef.T, coef)))

    # transform and calculate inner products
    coef = np.dot(coef, np.diag(np.sqrt(diag_inv)))
    z = np.dot(X, coef)

    if normalize:
        # post-normalization is done in R's
        # `kaiser()` function when rotate='Promax'
        z = z * np.sqrt(h2)

    rotation_mtx = np.dot(rotation_mtx, coef)

    coef_inv = np.linalg.pinv(coef)
    phi = np.dot(coef_inv, coef_inv.T)

    # convert loadings matrix to data frame
    loadings = z.copy()
    return loadings, rotation_mtx, phi


__all__ = ["parallel_analysis", "parallel_analysis_serial"]


def _get_correlation_function(method):
    """Returns the correlation function. """
    func = np.corrcoef
    return func

def parallel_analysis_serial(raw_data, n_iterations, correlation=('pearsons',), seed=None):
    """Estimate dimensionality from random data permutations.
    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data
        seed:  (integer) Random number generator seed value
    Returns
        eigs: mean eigenvalues
        sigma: standard deviation of eigenvalues
    """
    # Get Seeds for repeatablity
    random_seeds = SeedSequence(seed).spawn(n_iterations)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)

    correlation_method = _get_correlation_function(correlation)

    eigenvalue_array = np.zeros((n_iterations, n_items))

    for ndx, rseed in enumerate(random_seeds):
        rng_local = np.random.default_rng(rseed)

        new_data = rng_local.permutation(raw_data, axis=1).reshape(n_items, -1)
        local_correlation = correlation_method(new_data)

        eigenvals = np.linalg.eigvalsh(local_correlation)[::-1]

        eigenvalue_array[ndx] = eigenvals
    
    return eigenvalue_array.mean(0), eigenvalue_array.std(0, ddof=1)


def parallel_analysis(raw_data, n_iterations, correlation=('pearsons',), 
                      seed=None, num_processors=2):
    """Estimate dimensionality from random data permutations.
    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data
        seed:  (integer) Random number generator seed value
        num_processors: number of processors on a multi-core cpu to use
    Returns
        eigs: mean eigenvalues
        sigma: standard deviation of eigenvalues
    """
    if num_processors == 1:
        return parallel_analysis_serial(raw_data, n_iterations, correlation, seed)
    
    # Get Seeds for repeatablity
    random_seeds = SeedSequence(seed).spawn(n_iterations)
    chunk_seeds = np.array_split(random_seeds, num_processors)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)
    
    # Do the parallel calculation
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=raw_data.nbytes)
        shared_buff = np.ndarray(raw_data.shape, 
                                 dtype=raw_data.dtype, buffer=shm.buf)
        shared_buff[:] = raw_data[:]

        with futures.ThreadPoolExecutor(max_workers=num_processors) as pool:
            results = pool.map(_pa_engine, repeat(shm.name), repeat(correlation),
                               repeat(n_items), repeat(raw_data.dtype), 
                               repeat(raw_data.shape), chunk_seeds)

    eigenvalue_array = np.concatenate(list(results), axis=0)

    return eigenvalue_array.mean(0), eigenvalue_array.std(0, ddof=1)


def _pa_engine(name, correlation, n_items, dtype, shape, subset):
    """Parallel analysis engine for distributed computing."""
    correlation_method = _get_correlation_function(correlation)
    eigenvalue_array = np.zeros((subset.shape[0], n_items))
    
    # Read the shared memory buffer
    existing_shm = shared_memory.SharedMemory(name=name)    
    raw_data = np.ndarray(shape, dtype=dtype, 
                          buffer=existing_shm.buf)

    for ndx, rseed in enumerate(subset):
        rng_local = np.random.default_rng(rseed)

        new_data = rng_local.permutation(raw_data, axis=1).reshape(n_items, -1)

        local_correlation = correlation_method(new_data)

        eigenvals = np.linalg.eigvalsh(local_correlation)[::-1]

        eigenvalue_array[ndx] = eigenvals       
        
    return eigenvalue_array   
