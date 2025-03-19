import os
import pickle
import copy
import time

import json

import numpy as np
import itertools
import numbers

import pandas as pd

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from sklearn.utils import check_random_state, check_array

from sklearn.model_selection import KFold, StratifiedKFold

import warnings
from tqdm import tqdm, trange

from .data_class import matrix_class


### Initialization

# we adopted the NNDSVD initialization as in sklearn NMF
def _NNDSVD(M,
            n_components,
            svd_components=None,
            random_state=None,
            n_iter=5, # could be smaller as it is just for initialization
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

def _SVD_initialize(M,
                    n_components,
                    svd_components=None,
                    random_state=None):
    
    assert np.sum(np.isnan(M)) == 0
    M[M < 0] = 0
    W, QT, svd_components = _NNDSVD(M,
                                    n_components,
                                    svd_components=svd_components,
                                    random_state=random_state,
                                    )
    Q = QT.T
    return W, Q


def tsv_to_matrix_class(path_tsv,
                        path_json,
                        exclude_itemlist=[],
                        confoundlist=['Age', 'Sex'],
                        confoundintercept=True,
                        response_rescale=True,
                        response_shift=False,
                        confound_rescale=True,
                        confound_shift=False
                       ):


    tsv_data = pd.read_csv(path_tsv, sep ='\t', low_memory=False)
    # json_data = pd.read_json(path_json)
    json_info = json.load(open(path_json, 'r'))

    if (exclude_itemlist == None) or (len(exclude_itemlist) == 0):
        exclude_columns = ['Age', 'Sex', 'EID']
    else:
        exclude_columns = ['Age', 'Sex', 'EID'] + exclude_itemlist
    
    itemlist = [col for col in tsv_data.columns if col not in exclude_columns]

    # remove dropped out subjects
    tsv_data = tsv_data.dropna(subset=itemlist, axis=0, how='all')

    # check if any comlumn is full NaN
    all_nan_columns = tsv_data.isna().all()
    assert all(check is False for check in all_nan_columns)

    # EID
    subjlist = tsv_data['EID'].values

    # Extract min and max age
    # currently it is questionnaire specific
    # age_min, age_max = np.min(tsv_data['Age']), np.max(tsv_data['Age'])
    age_min = json_info['entire_meta_info']['min_age_entire']
    age_max = json_info['entire_meta_info']['max_age_entire']

    # Extract min and max response values
    min_values = [json_info['item_info'][item]['min'] for item in json_info['item_info']]
    max_values = [json_info['item_info'][item]['max'] for item in json_info['item_info']]

    # Save M_raw
    M_raw = tsv_data[itemlist].values
    
    # missing entries (1=available, 0=missing)
    nan_mask = 1 - np.isnan(M_raw)

    # Extract age and sex information, whether they will be confounds or not
    # confound Age
    age = tsv_data['Age'].values
    if confound_shift:
        age = age - age_min
    if confound_rescale:
        age = (age - age_min) / (age_max - age_min)
    age_old = age
    age_young = 1 - age_old

    # confound Sex
    # Sex code : Male = 0, Female = 1
    sex = tsv_data['Sex'].values
    sex_female = sex
    sex_male = 1 - sex_female

                           
    # Start generating the confound matrix C
    C = []
    Clist = []
    if len(confoundlist) > 0:
        # if there are confounds (Age or Sex)
        for confound in confoundlist:
            assert confound in tsv_data.columns
            assert confound in ['Age', 'Sex'] # current only support age and sex

        # if confoundintercept:
        #     C = np.zeros((len(subjlist), 2*len(confoundlist)+1))
        # else:
        #     C = np.zeros((len(subjlist), 2*len(confoundlist)))
                        
        if 'Age' in confoundlist:
            # age
            C.append(age_young)
            C.append(age_old)
            Clist.append('Young')
            Clist.append('Old')
        if 'Sex' in confoundlist:
            # sex
            C.append(sex_female)
            C.append(sex_male)
            Clist.append('Female')
            Clist.append('Male')
            
        ## ======== Current only support age and sex ========
        # other_confounds = [c for c in confoundlist if c not in ['Age', 'Sex']]
        # if len(other_confounds) > 0:
        #     for idx, item in enumerate(other_confounds):
        #         C[:,4+idx*2] = tsv_data[item].values
        #         C[:,4+idx*2+1] = 1 - tsv_data[item].values

        if confoundintercept:
            # intercept
            C.append(np.ones(M_raw.shape[0]))
            Clist.append('Intercept')
        C = np.vstack(C).T
        
    else:
        # if no confounds are considered
        if confoundintercept:
            C = np.ones((M_raw.shape[0], 1))
            Clist.append('Intercept')
        else:
            # by setting C = None, we make sure that no confounds and no intercept will be modeled
            C = None
                           
    M = copy.deepcopy(M_raw)
    # M[np.isnan(M)] = np.min(min_values)
    for i in range(M.shape[1]):
        M[np.isnan(M[:,i]),i] = np.min(min_values[i])
        # shift minimum to zero for sparsity
        if response_shift:
            M[:,i] -= np.min(min_values[i])

    if response_rescale:
        for idx, max_response in enumerate(max_values):
            min_response = min_values[idx]
            if isinstance(max_response, numbers.Number):
                M[:,idx] = M[:,idx] / (max_response - min_response)
            else:
                raise ValueError("Input is of unknown type")

    matrix = matrix_class(M=M,
                          M_raw=M_raw,
                          nan_mask=nan_mask,
                          IDlist=tsv_data['EID'].values,
                          itemlist=itemlist,
                          W=None,
                          Q=None,
                          C=C,
                          Qc=None,
                          Z=None,
                          aZ=None,
                          verbose=True
                         )

    M_df = pd.DataFrame(M, index=tsv_data['EID'].values, columns=itemlist)
    M_raw_df = pd.DataFrame(M_raw, index=tsv_data['EID'].values, columns=itemlist)
    nan_mask_df = pd.DataFrame(nan_mask, index=tsv_data['EID'].values, columns=itemlist)
    if len(Clist) > 0:
        C_df = pd.DataFrame(C, index=tsv_data['EID'].values, columns=Clist)
    else:
        C_df = None

    return matrix, M_df, M_raw_df, nan_mask_df, C_df