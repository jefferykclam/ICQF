import os
import pickle
import copy
import time

import pyximport
pyximport.install()

import numpy as np
import itertools

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from sklearn.utils import check_random_state, check_array

from sklearn.model_selection import KFold, StratifiedKFold

from cvxopt import matrix, spmatrix, solvers, sparse, spdiag
import quadprog

import warnings
from tqdm import tqdm, trange

from ._cdnmf_fast import _update_cdnmf_fast
from .factor_analysis import factor_analysis, parallel_analysis_serial
from .data_class import matrix_class

from kneed import KneeLocator

from matplotlib import pyplot
import seaborn as sns
    
# we adopted the NNDSVD initialization as in sklearn NMF
def NNDSVD(M, n_components,
           svd_components=None,
           random_state=None,
           n_iter=4, # could be smaller as it is just for initialization
           eps=1e-6):
    
    if svd_components is None:
        U, S, VT = randomized_svd(M, n_components,
                                 n_iter=n_iter,
                                 random_state=random_state)
    else:
        assert len(svd_components) == 2
        S = svd_components[0]
        VT = svd_components[1]
        U = safe_sparse_dot(M, VT.T) / S[0] # leading singular value is non-negative
    
    W, H = np.zeros(U.shape), np.zeros(VT.shape)

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(VT[0, :])
    for j in range(1, n_components):
        x, y = U[:, j], VT[j, :]
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
        x_p_nrm, y_p_nrm = np.linalg.norm(x_p), np.linalg.norm(y_p)
        x_n_nrm, y_n_nrm = np.linalg.norm(x_n), np.linalg.norm(y_n)
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    W[W < eps] = 0
    H[H < eps] = 0
    return W, H, [S, VT]

    
def soft_shrinkage(X, l):
    return np.sign(X) * np.maximum(np.abs(X) - l, 0.)

def _multiFISTA(X, tau, eta, A, B, V, operator, lips_const, t0=1.0, max_iter=200, tol=1e-3):
    # row-ise FISTA algorithm
    # arg min_x eta ||x||_1 + 1/2 * f(x)
    
    f = lambda X : np.sum((X@A-B)**2, axis=1) + tau*np.sum((X-V)**2, axis=1)
    grad_f = lambda X : X@operator - 2*(B@A.T + tau*V) 
    loss_fcn = lambda X : 0.5*f(X) + eta*np.linalg.norm(X, ord=1, axis=1)

    loss = loss_fcn(X)
    _row_idx = np.arange(X.shape[0])

    for iteration in range(max_iter):
        _X = X.copy()
        _X[_row_idx] = soft_shrinkage(X[_row_idx] - grad_f(X)[_row_idx]/lips_const,
                                           eta/lips_const)
        t = (1 + np.sqrt(1 + 4*t0**2))/2
        X[_row_idx,:] = _X[_row_idx,:] + ((t0 - 1)/t) * (_X[_row_idx,:] - X[_row_idx,:])

        new_loss = loss_fcn(X)
        _ratio = np.abs(loss - new_loss)/(np.abs(new_loss)+1e-12)
        _row_idx = np.where(_ratio > tol)[0]
        loss = new_loss

        if len(_row_idx) == 0:
            return X
    return X

def _projection(Y, uppbd):
    Y = np.maximum(Y, 0)
    if uppbd[0]:
        Y = np.minimum(Y, uppbd[1])
    return Y

def _multiregularized_loss(X, A, B, gamma):
    # loss function for _admm_constrained_multilasso
    mismatch_loss = 0.5 * np.sum((X@A - B)**2, axis=1)
    reg_loss = gamma * np.linalg.norm(X, ord=1, axis=1)
    return mismatch_loss + reg_loss

def _admm_constrained_multilasso(X, A, B, gamma, uppbd, tau, tol=1e-3,
                                 min_iter=1, max_iter=200):

    # solve row-wise constrained Lasso with ADMM
    # 1/2 * || B - XA ||^2_F + gamma * sum_i || X[i,:] ||_1
    # where the L1-norm is computed row-wise

    # Intuition: sparsity control on every single subject/question representation

    # gamma = beta / rho

    # initialize _mu
    _mu = np.zeros_like(X)
    # initialize _Y
    _Y = _projection(X, uppbd)

    loss = _multiregularized_loss(X, A, B, gamma)
    row_idx = np.arange(X.shape[0])

    loss_list = [loss]

    operator = 2*(A@A.T + tau * np.eye(A.shape[0]))
    lipschitz_const = np.linalg.norm(operator, ord=2)

    for iteration in range(max_iter):

        _V = _Y - _mu/tau
        X[row_idx] = _multiFISTA(X[row_idx], tau, gamma, 
                                 A, # for defining f and grad_f
                                 B[row_idx], # for defining f and grad_f
                                 _V[row_idx], # for defining f and grad_f
                                 operator, # for defining f and grad_f
                                 lipschitz_const, # for defining f and grad_f
                                )

        _Y[row_idx] = _projection(X[row_idx] + _mu[row_idx]/tau, uppbd)
        _mu[row_idx] += tau * (X[row_idx] - _Y[row_idx])

        new_loss = _multiregularized_loss(X, A, B, gamma)
        _ratio = np.abs(loss_list[-1] - new_loss)/(np.abs(new_loss)+1e-12)
        row_idx = np.where(_ratio > tol)[0]
        loss_list.append(new_loss)

        if iteration > min_iter:
            converged = True
            if len(row_idx) > 0:
                converged = False
            if converged:
                return X, _Y

    return X, _Y 

def _nnQPsolver_CVXOPT(H, f, upperbound):

    solvers.options['show_progress'] = False

    n = H.shape[0]
    P = matrix(H)
    q = matrix(f)
    if upperbound[0] == False:
        G = matrix( -np.eye(n) )
        h = matrix( np.zeros(n) )
    elif upperbound[0] == True:
        G = matrix( np.vstack((-np.eye(n), np.eye(n))) )
        h = matrix( np.hstack((np.zeros(n), upperbound[1]*np.ones(n) )) )
    sol = solvers.qp(P,q,G,h)
    return np.array(sol['x']).flatten()

def _nnQPsolver_quadprog(H, f, upperbound):

    solvers.options['show_progress'] = False

    n = H.shape[0]
    if upperbound[0] == False:
        qp_C = np.eye(n)
        qp_b = np.zeros(n)
    elif upperbound[0] == True:
        qp_C = np.vstack((np.eye(n), -np.eye(n))).T
        qp_b = np.hstack((np.zeros(n), -upperbound[1]*np.ones(n)))
    sol = quadprog.solve_qp(H, -f, qp_C, qp_b)[0]
    return sol


def _constrained_multiquadratic(X, A, B, gamma, uppbd, solver='quadprog'):

    # solve row-wise constrained quadratic programming with ADMM
    # 1/2 * || B - XA ||^2_F + gamma * sum_i || X[i,:] ||_2
    # where the L2-norm is computed row-wise

    H = A@A.T + 2*gamma*np.eye(A.shape[0])
    f = -B@A.T

    # quadprog package cannot handle semi-positive definite matrix
    if gamma < 1e-8:
        solver = 'cvxopt'

    for idx in range(X.shape[0]):
        if solver == 'quadprog':
            X[idx] = _nnQPsolver_quadprog(H, f[idx], uppbd)
        elif solver == 'cvxopt':
            X[idx] = _nnQPsolver_CVXOPT(H, f[idx], uppbd)

    return X

def _update_coordinate_descent(X, W, Ht, regularizer, beta, upperbd, shuffle, random_state):
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).

    """
    
#     Ht = check_array(Ht, order="C")
#     # W = check_array(W, order="C")
#     # X = check_array(X, order="C")
#     X = check_array(X, accept_sparse="csr")
    
#     print(Ht.flags)
#     print(W.flags)
#     print(X.flags)
    
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if regularizer == 2:
        # adds l2_reg only on the diagonal
        HHt.flat[:: n_components + 1] += beta
    # L1 regularization corresponds to decrease of each element of XHt
    if regularizer == 1:
        XHt -= beta

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, upperbd, permutation)





def kfold_select(P, Krow, Kcol, shuffling=True):
    block_list = []
    if (Krow > 1) & (Kcol > 1):
        krowf = KFold(n_splits=Krow, shuffle=shuffling)
        kcolf = KFold(n_splits=Kcol, shuffle=shuffling)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
                uHO = np.zeros(P.shape[0])
                uHO[HOrow_idx] = 1 # row hold out
                vHO = np.zeros(P.shape[1])
                vHO[HOcol_idx] = 1 # column hold out
                block = [uHO, vHO]
                block_list.append(block)
    if (Krow > 1) & (Kcol == 1):
        krowf = KFold(n_splits=Krow, shuffle=shuffling)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            uHO = np.zeros(P.shape[0])
            uHO[HOrow_idx] = 1
            vHO = np.ones(P.shape[1])
            block = [uHO, vHO]
            block_list.append(block)
    if (Krow == 1) & (Kcol > 1):
        kcolf = KFold(n_splits=Kcol, shuffle=shuffling)
        for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
            uHO = np.ones(P.shape[0])
            vHO = np.zeros(P.shape[1])
            vHO[HOcol_idx] = 1
            block = [uHO, vHO]
            block_list.append(block)
    return block_list

def block_CV_mask(P, Krow, Kcol, J=10):
    block_list = kfold_select(P, Krow, Kcol)
    idx = np.arange(len(block_list))
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    if n % J != 0:
        warnings.warn('nblocks is not divisible!')
        
    np.random.shuffle(idx)
    
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        block_idx = idx[j*nc:(j+1)*nc]
        mask_valid = np.zeros_like(P)
        for k in block_idx:
            block_vectors = block_list[k]
            mask_valid += np.outer(block_vectors[0], block_vectors[1])
        mask_valid[mask_valid > 1] == 1
        mask_train = 1 - mask_valid
        mask_train_list.append(mask_train)
        mask_valid_list.append(mask_valid)
    return mask_train_list, mask_valid_list
    
def random_CV_mask(M, J=10):
    Mp = M>0
    Mz = M==0
    
    idx = np.arange(M.shape[0]*M.shape[1])
    y = Mp.reshape(-1)
    
    mask_train_list = []
    mask_valid_list = []
    skf = StratifiedKFold(n_splits=J, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(idx, y)):
        mask_train = np.zeros_like(y)
        mask_valid = np.zeros_like(y)
        mask_train[train_index] = 1
        mask_valid[test_index] = 1
        mask_train_list.append(mask_train.reshape(M.shape[0], M.shape[1]))
        mask_valid_list.append(mask_valid.reshape(M.shape[0], M.shape[1]))
        
    return mask_train_list, mask_valid_list








class ICQF(TransformerMixin, BaseEstimator):
    

    def __init__(
        self,
        n_components=None,
        *,
        method='cd',
        W_beta=0.1,
        Q_beta=0.1,
        regularizer=1,
        rho=3.0,
        tau=3.0,
        W_upperbd=(False, 1.0),
        Q_upperbd=(False, 1.0),
        M_upperbd=(True, 1.0),
        min_iter=10,
        max_iter=200,
        admm_iter=5,
        tol=1e-4,
        random_state=None,
        verbose=False):
        
        self.n_components = n_components
        self.method = method
        self.W_beta = W_beta
        self.Q_beta = Q_beta
        self.regularizer = regularizer
        self.rho = rho
        self.tau = tau
        self.W_upperbd = W_upperbd
        self.Q_upperbd = Q_upperbd
        self.M_upperbd = M_upperbd
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.admm_iter = admm_iter
        self.tol = tol
        self.random_state = None
        self.verbose = verbose
        
        self.iteration = 0
        
        # since M is min-max normalized, 
        # min(M) = 0.0, and
        # max(M) = 1.0 are assumed: M_upperbd=(True, 1.0)
        self.Mmin = 0.0
        if self.M_upperbd[0] == True:
            self.Mmax = self.M_upperbd[1]
            
            
        self.rng = check_random_state(self.random_state)
        
    def vprint(self, str):
        if self.verbose==True:
            print(str)

    def initialize(self, matrix_class, svd_components=None):
        
        MF_data = copy.deepcopy(matrix_class)
        assert np.sum(np.isnan(MF_data.M)) == 0
        
        MF_data.M[MF_data.M < 0] = 0
        
        W, QT, svd_components = NNDSVD(matrix_class.M, 
                                       self.n_components,
                                       random_state=self.random_state,
                                       svd_components=svd_components)
        Q = QT.T

        # initialize W satisfying the bounded constraints
        W[W < 0] = 0
        if self.W_upperbd[0] == True:
            W[W > self.W_upperbd[1]] = self.W_upperbd[1]
        MF_data.W = W
        
        # initialize Q satisfying the bounded constraints
        Q[Q < 0] = 0
        if self.Q_upperbd[0] == True:
            Q[Q > self.Q_upperbd[1]] = self.Q_upperbd[1]
        MF_data.Q = Q

        # initialize Z satisfying the bounded constraints
        Z = W@(Q.T)
        Z[Z < self.Mmin] = self.Mmin
        if self.M_upperbd[0] == True:
            Z[Z > self.Mmax] = self.Mmax
        MF_data.Z = Z
        MF_data.aZ = np.zeros_like(matrix_class.M)
        
        # include confounder matrix if provided
        # assumed rescaled but not mirrored
        if matrix_class.confound is not None:
            C = np.hstack((matrix_class.confound, 1.0-matrix_class.confound))
            C = np.hstack((C, np.ones((C.shape[0], 1))))
            MF_data.C = C
            
            # initialization on Qc (simply zero matrix as an initialization)
            Qc = np.zeros((matrix_class.M.shape[1], C.shape[1]))
            MF_data.Qc = Qc
        else:
            MF_data.C = None
            MF_data.Qc = None
        
        # return matrix_class obj
        return MF_data
        
                    
    def _update_Z(self, MF_data, R):
        
        # update Z element-wise
        # min || mask * (M - X) ||^2_F + rho * || X - R ||^2_F
        
        Z = (MF_data.M * MF_data.nan_mask + self.rho * R) / (self.rho + MF_data.nan_mask * 1.0)
        Z[Z < self.Mmin] = self.Mmin
        if self.M_upperbd[0] == True:
            Z = np.minimum(Z, self.Mmax)
        return Z
    
    def _update_W(self, MF_data, B, gamma):
        
        if self.method == 'admm':
            if self.regularizer == 1:
                _, MF_data.W = _admm_constrained_multilasso(MF_data.W,
                                                            MF_data.Q.T,
                                                            B,
                                                            gamma,
                                                            self.W_upperbd,
                                                            self.tau)
            if self.regularizer == 2:
                MF_data.W = _constrained_multiquadratic(MF_data.W,
                                                        MF_data.Q.T,
                                                        B,
                                                        gamma,
                                                        self.W_upperbd)
                
        elif self.method == 'cd':
            MF_data.W = check_array(MF_data.W, order='C')
            MF_data.Q = check_array(MF_data.Q, order='C')
            B = check_array(B, order='C')
            if self.W_upperbd[0] is True:
                violation = _update_coordinate_descent(B, MF_data.W, MF_data.Q,
                                                       self.regularizer,
                                                       gamma,
                                                       self.W_upperbd[1],
                                                       False, self.rng)
            else:
                violation = _update_coordinate_descent(B, MF_data.W, MF_data.Q,
                                                       self.regularizer,
                                                       gamma, -1,
                                                       False, self.rng)
        elif self.method == 'hybrid':
            if self.regularizer == 1:
                _, MF_data.W = _admm_constrained_multilasso(MF_data.W,
                                                            MF_data.Q.T,
                                                            B,
                                                            gamma,
                                                            self.W_upperbd,
                                                            self.tau,
                                                            max_iter=self.admm_iter)
                MF_data.W = check_array(MF_data.W, order='C')
                MF_data.Q = check_array(MF_data.Q, order='C')
                B = check_array(B, order='C')
                if self.W_upperbd[0] is True:
                    violation = _update_coordinate_descent(B, MF_data.W, MF_data.Q,
                                                           self.regularizer,
                                                           gamma,
                                                           self.W_upperbd[1],
                                                           False, self.rng)
                else:
                    violation = _update_coordinate_descent(B, MF_data.W, MF_data.Q,
                                                           self.regularizer,
                                                           gamma, -1,
                                                           False, self.rng)
            if self.regularizer == 2:
                MF_data.W = _constrained_multiquadratic(MF_data.W,
                                                        MF_data.Q.T,
                                                        B,
                                                        gamma,
                                                        self.W_upperbd)
                
        return MF_data
    
    def _update_Q(self, MF_data, B, gamma):
        
        if MF_data.C is not None:
            QComp = np.column_stack((MF_data.Q, MF_data.Qc))
            WComp = np.column_stack((MF_data.W, MF_data.C))
        else:
            QComp = MF_data.Q
            WComp = MF_data.W
        
        if self.method == 'admm':
            if self.regularizer == 1:
                _, QComp = _admm_constrained_multilasso(QComp, WComp.T, B.T, 
                                                        gamma,
                                                        self.Q_upperbd,
                                                        self.tau)
            if self.regularizer == 2:
                QComp = self._constrained_multiquadratic(QComp, WComp.T, B.T,
                                                         gamma,
                                                         self.Q_upperbd)
                        
        elif self.method == 'cd':
            
            QComp = check_array(QComp, order='C')
            WComp = check_array(WComp, order='C')
            B = check_array(B, order='C')
            if self.Q_upperbd[0] == True:
                violation = _update_coordinate_descent(B.T, QComp, WComp,
                                                        self.regularizer,
                                                        gamma,
                                                        self.Q_upperbd[1],
                                                        False, self.rng)
            else:
                violation = _update_coordinate_descent(B.T, QComp, WComp,
                                                        self.regularizer,
                                                        gamma,
                                                        -1,
                                                        False, self.rng)
        elif self.method == 'hybrid':
            if self.regularizer == 1:
                _, QComp = _admm_constrained_multilasso(QComp, WComp.T, B.T, 
                                                        gamma,
                                                        self.Q_upperbd,
                                                        self.tau,
                                                        max_iter=self.admm_iter)
                
                QComp = check_array(QComp, order='C')
                WComp = check_array(WComp, order='C')
                B = check_array(B, order='C')
                if self.Q_upperbd[0] == True:
                    violation = _update_coordinate_descent(B.T, QComp, WComp,
                                                            self.regularizer,
                                                            gamma,
                                                            self.Q_upperbd[1],
                                                            False, self.rng)
                else:
                    violation = _update_coordinate_descent(B.T, QComp, WComp,
                                                            self.regularizer,
                                                            gamma,
                                                            -1,
                                                            False, self.rng)

            if self.regularizer == 2:
                QComp = self._constrained_multiquadratic(QComp, WComp.T, B.T,
                                                         gamma,
                                                         self.Q_upperbd,
                                                         max_iter=self.admm_iter)

        
        else:
            raise ValueError("Unknown method.")
                

        MF_data.Q = QComp[:, :MF_data.Q.shape[1]]
        if MF_data.C is not None:
            MF_data.Qc = QComp[:, MF_data.Q.shape[1]:]
        return MF_data
        
    
    def _obj_func(self, MF_data, betaW, betaQ):
        R = MF_data.W @ (MF_data.Q.T)
        if MF_data.C is not None:
            R += MF_data.C @ (MF_data.Qc.T)
            
        mismatch_loss = 0.5 * np.sum(MF_data.nan_mask * (MF_data.M - R)**2)
        W_norm = betaW*np.sum(np.linalg.norm(MF_data.W, ord=self.regularizer, axis=1))
        Q_norm = betaQ*np.sum(np.linalg.norm(MF_data.Q, ord=self.regularizer, axis=1))
        reg_loss = W_norm+Q_norm
        return mismatch_loss, reg_loss

    def _missmatch_loss(self, MF_data, extra_mask):
        if MF_data.C is None:
            M_approx = MF_data.W@MF_data.Q.T
        else:
            M_approx = MF_data.W@MF_data.Q.T + MF_data.C@MF_data.Qc.T
        
        nentry = np.sum(extra_mask*MF_data.nan_mask)
        
        # normalized reconstruction error
        return np.sum(extra_mask*MF_data.nan_mask*(MF_data.M - M_approx) ** 2) / nentry
        
    
    def fit_transform(self, matrix_class, svd_components=None):
        
        tic = time.perf_counter()
        MF_data = self.initialize(matrix_class, svd_components=svd_components)
        self.MF_init = copy.deepcopy(MF_data)
        toc = time.perf_counter()
        # print(f"Initialization time: {toc-tic:0.4f}s")
        self.MF_data_, self.loss_history_ = self._fit_transform(MF_data, update_Q=True)
        
        return self.MF_data_, self.loss_history_

    def fit(self, matrix_class):
        _ = self.fit_transform(matrix_class)
        
    def transform(self, matrix_class):
        # optimize W with given Q, C, M (and mask)
        
        # make a copy of the trained model (to get Q, C)
        MF_init = copy.deepcopy(self.MF_data_)
        
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
        
        # perform NNDSVD on new data matrix to intialize W
        W, _ = self.NNDSVD(matrix_class.M)
        W[W < 0] = 0
        if self.W_upperbd[0] == True:
            W[W > self.W_upperbd[1]] = self.W_upperbd[1]
        MF_init.W = W
            
        # Q is not updated, just redo the projection in case the upperbound is not satisfied
        # could be omitted
        MF_init.Q[MF_init.Q < 0] = 0
        if self.Q_upperbd[0] == True:
            MF_init.Q[MF_init.Q > self.Q_upperbd[1]] = self.Q_upperbd[1]
        
        # add confounds corresponding new data matrix, if applicable
        if matrix_class.confound is not None:
            assert self.MF_data_.confound is not None
            C = np.hstack((matrix_class.confound, 1.0-matrix_class.confound))
            C = np.hstack((C, np.ones((C.shape[0], 1))))
            
            MF_init.C = C
            MF_init.Qc[MF_init.Qc < 0] = 0
            if self.Q_upperbd[0] == True:
                MF_init.Qc[MF_init.Qc > self.Q_upperbd[1]] = self.Q_upperbd[1]
        else:
            assert self.MF_data_.confound is None
            MF_init.C = None
            MF_init.Qc = None
            
        # simply initialize Z with WQ^T, where W is estimated from NNDSVD
        Z = MF_init.W@(MF_init.Q.T)
        if MF_init.C is not None:
            Z += MF_init.C@(MF_init.Qc.T)
        Z[Z < self.Mmin] = self.Mmin
        if self.M_upperbd[0] == True:
            Z[Z > self.Mmax] = self.Mmax
        MF_init.Z = Z
        MF_init.aZ = np.zeros_like(matrix_class.M)
        
        self.MF_init = copy.deepcopy(MF_init)
        
        # optimize for W
        MF_data, loss_history = self._fit_transform(MF_init, update_Q=False)
        
        return MF_data, loss_history
    
    
    def _fit_transform(self, MF_data, update_Q=True):
        
        MF_data.W = check_array(MF_data.W, order='C')
        MF_data.Q = check_array(MF_data.Q, order='C')
        MF_data.M = check_array(MF_data.M, order='C')
        
        self.n_components_ = self.n_components
        
        # initial distance value
        loss_history = []

        # tqdm setting
        tqdm_iterator = trange(self.max_iter, desc='ICQF', leave=True, disable=not self.verbose)

        betaW = self.W_beta
        if self.W_upperbd[0] == True:
            if self.Q_upperbd[0] == True:
                magnitude_ratio = self.Q_upperbd[1]/self.W_upperbd[1]
            else:
                magnitude_ratio = 1/self.W_upperbd[1]
        else:
            magnitude_ratio = 1.0
        
        betaQ = self.Q_beta * MF_data.W.shape[0]/MF_data.Q.shape[0] * magnitude_ratio

        # Main iteration
        for i in tqdm_iterator:
            tic = time.perf_counter()
            
            B = MF_data.Z + MF_data.aZ/self.rho
            if MF_data.C is not None:
                B -= MF_data.C@(MF_data.Qc.T)
            
            # subproblem 1
            tic_s1 = time.perf_counter()
            MF_data = self._update_W(MF_data, B, betaW/self.rho)            
            toc_s1 = time.perf_counter()
            s1_time = tic_s1 - toc_s1
                    
            B = MF_data.Z + MF_data.aZ/self.rho
                
            # subproblem 2     
            if update_Q:
                tic_s2 = time.perf_counter()       
                MF_data = self._update_Q(MF_data, B, betaQ/self.rho)
                toc_s2 = time.perf_counter()
            s2_time = tic_s2 - toc_s2
            
            B = MF_data.W@(MF_data.Q.T)
            if MF_data.C is not None:
                B += MF_data.C@(MF_data.Qc.T)

            # subproblem 3
            tic_s3 = time.perf_counter()
            MF_data.Z = self._update_Z(MF_data, B-MF_data.aZ/self.rho)
            toc_s3 = time.perf_counter()
            s3_time = tic_s3 - toc_s3
            
            # auxiliary varibles update        
            MF_data.aZ += self.rho*(MF_data.Z - B)
            
            # Iteration info
            mismatch_loss, reg_loss = self._obj_func(MF_data, betaW, betaQ)
            loss_history.append((mismatch_loss+reg_loss, mismatch_loss, reg_loss))
            toc = time.perf_counter()
            if i > 0:
                err_ratio = np.abs(loss_history[-1][0]-loss_history[-2][0])/np.abs(loss_history[-1][0])
            else:
                err_ratio = np.nan
            message = f"loss={loss_history[-1][0]:0.3e}, tol={err_ratio:0.3e}, "
            message += f"time/iter={toc-tic:0.2f}s ({s1_time:0.2f}s,{s2_time:0.2f}s,{s3_time:0.2f})"
            tqdm_iterator.set_description(message)
            tqdm_iterator.refresh()

            # Check convergence
            if i > self.min_iter:
                converged = True
                if err_ratio < self.tol:
                    self.vprint('Algorithm converged with relative error < {}.'.format(self.tol))
                    return MF_data, loss_history
                else:
                    converged = False

        return MF_data, loss_history
    
    def embed_holdout(self, MF_data, mask_train, mask_valid):

        # store data nan_mask
        _nan_mask = copy.deepcopy(MF_data.nan_mask)
        
        # multiply nan_mask with training mask
        MF_data.nan_mask *= mask_train
        
        # factorization
        MF_data, loss_history = self.fit_transform(MF_data)
        
        # recovery nan_mask
        MF_data.nan_mask = _nan_mask
        
        train_error = self._missmatch_loss(MF_data, mask_train)
        valid_error = self._missmatch_loss(MF_data, mask_valid)

        embedding_stat = [self.n_components, self.W_beta, self.Q_beta, train_error, valid_error]
        
        return embedding_stat, MF_data, loss_history
    
    
    def detect_dimension(self,
                         MF_data,
                         dimension_list=None,
                         W_beta_list=None,
                         Q_beta_list=None,
                         separate_beta=False,
                         mask_type='random',
                         repeat=1,
                         nfold=5,
                         random_fold=True,
                         detection='kneed',
                         nrow=10,
                         ncol=10,
                         show_profile=True):

        self.verbose = False
        
        if (W_beta_list is None) and (Q_beta_list is None):
            W_beta_list = [0.001, 0.01, 0.1, 0.2, 0.5]
            Q_beta_list = [0.001, 0.01, 0.1, 0.2, 0.5]
        elif (W_beta_list is not None) and (Q_beta_list is None):
            if separate_beta:
                assert isinstance(W_beta_list, list)
                Q_beta_list = [0.001, 0.01, 0.1, 0.2, 0.5]
            else:
                Q_beta_list = W_beta_list
        elif (W_beta_list is None) and (Q_beta_list is not None):
            if separate_beta:
                assert isinstance(Q_beta_list, list)
                W_beta_list = [0.001, 0.01, 0.1, 0.2, 0.5]
            else:
                W_beta_list = Q_beta_list
        else:
            assert isinstance(W_beta_list, list)
            assert isinstance(Q_beta_list, list)
        
        if separate_beta:
            print('W_beta search space : ', W_beta_list)
            print('Q_beta search space : ', Q_beta_list)
        else:
            print('shared beta search space : ', W_beta_list)
                
        if separate_beta: 
            config_list = list(itertools.product(W_beta_list, Q_beta_list))
        else:    
            config_list = list(zip(W_beta_list, Q_beta_list))
        
        if dimension_list is not None:
            assert isinstance(dimension_list, (list, tuple, np.ndarray))
            print('dimension detection range: {} - {}'.format(dimension_list[0], dimension_list[-1]))
        else:
            correlation = np.corrcoef(MF_data.M.T)
            # correction when some columns are completely zeros
            correlation = np.nan_to_num(correlation)
            eigs, vects = np.linalg.eigh(correlation)
            result = parallel_analysis_serial(MF_data.M.T, 30, correlation=('pearsons',),
                                              seed=None)
            better = (eigs - result[0][::-1])
            horn_dim = np.sum(better > 0)

            dimension_list = np.arange(np.maximum(2,int(horn_dim)-10), int(horn_dim)+11) #-5,+6
            print('dimension detection range: {} - {} ({})'.format(dimension_list[0], dimension_list[-1], horn_dim))

        config_list = itertools.product(dimension_list, config_list)
        config_list = [(d, *betas) for d, betas in config_list]
        
        optimal_stat = None
        optimal_MF_data = None
        optimal_loss_history = None
        embed_stat_pd = pd.DataFrame(columns=['repeat', 'fold', 'dimension', 'W_beta', 'Q_beta', 'train_error', 'valid_error'])
        
        # create masks
        tqdm_range = trange(repeat, desc='dimension detection', leave=True)
        for r in tqdm_range:
            # np.random.seed()
            # set random seed
            
            if mask_type == 'block':
                mask_train_list, mask_valid_list = block_CV_mask(MF_data.M, Krow=nrow, Kcol=ncol, J=nfold)
            if mask_type == 'random':
                mask_train_list, mask_valid_list = random_CV_mask(MF_data.M, J=nfold)

                    
            for config in config_list:
                
                (dim, W_beta, Q_beta) = config
                self.n_components = dim
                self.W_beta = W_beta
                self.Q_beta = Q_beta
                
                if random_fold:
                    rfold = np.random.randint(nfold)
                    
                for fold in range(nfold):
                    if random_fold:
                        if fold != rfold:
                            continue
                    
                    mask_train = mask_train_list[fold]
                    mask_valid = mask_valid_list[fold]

                    embed_stat, MF_data, loss_history = self.embed_holdout(MF_data, mask_train, mask_valid)
                    embed_stat_pd.loc[len(embed_stat_pd)] = [r, fold] + embed_stat

                    message = f"repeat-[{r+1:2.0f}]: config-[{dim:2.0f},{W_beta:1.3f},{Q_beta:1.3f}], "
                    if random_fold:
                        message += f"fold-[{fold+1:2.0f}], "
                    else:
                        message += f"fold-[{fold+1:2.0f}/{nfold:2.0f}]"

                    tqdm_range.set_description(message)
                    tqdm_range.refresh()
                    
            avg_stat = embed_stat_pd.groupby(['dimension', 'W_beta', 'Q_beta'])['valid_error'].mean().reset_index()
                    
            if detection == 'kneed':
                optimal_config = avg_stat.loc[avg_stat['valid_error'].idxmin()].values
                optimal_W_beta = optimal_config[1]
                optimal_Q_beta = optimal_config[2]
                optimal_stat = embed_stat_pd.loc[(embed_stat_pd['W_beta']==optimal_W_beta) & (embed_stat_pd['Q_beta']==optimal_Q_beta)]
                optimal_valid_error = optimal_stat.groupby(['dimension'])['valid_error'].mean().reset_index()
                search_range = optimal_valid_error['dimension'].values
                reconst_err = optimal_valid_error['valid_error'].values
                try: 
                    kn = KneeLocator(search_range, reconst_err, curve='convex', direction='decreasing')
                    optimal_config[0] = int(kn.knee)
                except:
                    self.vprint('optimal kneed point cannot be detected, report lowest point instead.')
                    
            elif detection == 'lowest':
                optimal_config = avg_stat.loc[avg_stat['valid_error'].idxmin()].values
                
            message = f"repeat-[{r+1:2.0f}]: "
            message += f"optimal config-[{optimal_config[0]:2.0f},{optimal_config[1]:1.3f},{optimal_config[2]:1.3f}]"

            tqdm_range.set_description(message)
            tqdm_range.refresh()
        
        self.n_components = int(optimal_config[0])
        self.W_beta = optimal_config[1]
        self.Q_beta = optimal_config[2]
        
        optimal_MF_data, optimal_loss_history = self.fit_transform(MF_data)
        
        self.n_components_ = int(optimal_config[0])
        self.MF_data_ = optimal_MF_data
        self.loss_history_ = optimal_loss_history
        
        self.detection_profile = embed_stat_pd
        
        if show_profile:
            
            cmap = sns.color_palette("tab10")
            fig, ax = pyplot.subplots(figsize=(6, 6))
            for idx, config in enumerate(config_list):
                (dim, W_beta, Q_beta) = config
                config_err = embed_stat_pd.loc[(embed_stat_pd['W_beta']==W_beta) & (embed_stat_pd['Q_beta'] == Q_beta)]
                mean_err = config_err.groupby(['dimension'])['valid_error'].mean().reset_index()
                search_range = mean_err['dimension'].values
                reconst_err = mean_err['valid_error'].values
                if (config == optimal_config[:3]).all():
                    ax.plot(search_range, reconst_err, c=cmap[np.mod(idx,10)],
                            alpha=1, linewidth=3, label=f"({W_beta:1.3f},{Q_beta:1.3f})")
                    sns.lineplot(data=config_err, x="dimension", y="valid_error", alpha=0.1, ax=ax, 
                             color=cmap[np.mod(idx,10)], linestyle='', errorbar=('ci',100))
                    # try: 
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    try:
                        kn = KneeLocator(search_range, reconst_err, curve='convex', direction='decreasing')
                        ax.vlines(int(kn.knee), ymin, ymax, linestyle='--', colors='b', label='elbow')
                        ax.hlines(reconst_err[np.where(search_range==kn.knee)[0]], xmin, xmax, linestyle='--', colors='b')
                        ax.scatter(int(kn.knee), reconst_err[np.where(search_range==kn.knee)[0]], marker='o', c='b')
                    except:
                        pass
                    
                    lowest_err = reconst_err[np.argmin(reconst_err)]
                    lowest_dim = search_range[np.argmin(reconst_err)]
                    ax.vlines(lowest_dim, ymin, ymax, linestyle='--', colors='k', label='lowest')
                    ax.hlines(lowest_err, xmin, xmax, linestyle='--', colors='k')
                    ax.scatter(lowest_dim, lowest_err, marker='o', c='k')
                      
                else:
                    ax.plot(search_range, reconst_err, c=cmap[np.mod(idx,10)],
                            alpha=0.1)
                    sns.lineplot(data=config_err, x="dimension", y="valid_error", alpha=0.1, ax=ax, 
                             color='grey', linestyle='', errorbar=('ci',100))
            ax.set_xlim(search_range[0], search_range[-1])
            pyplot.show()
            
                                    
        return optimal_MF_data, optimal_stat, embed_stat_pd