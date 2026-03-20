import os
import pickle
import pandas as pd
import glob
from tqdm import tqdm, trange

# math imports
import numpy as np
import scipy
import sklearn

import copy
from src.data_class import matrix_class


def generate_class(_MF, n_age=5):

    gender_split = _MF.confound[:,1]
    age_split = np.zeros(_MF.M.shape[0])
    ages = _MF.confound[:,0]

    n_age_split = n_age
    for i in range(n_age_split):
        age_group = np.where(np.logical_and(ages >= i/n_age_split, ages < (i+1)/n_age_split))[0]
        age_split[age_group] += i
    
    classes = age_split
    classes[gender_split == 1] *= 2

    return classes


def reduce_class(split, iteration):
    new_split = split.copy()
    n_class = np.unique(split)
    assert iteration < n_class.shape[0]
    class_size = [ np.where(split == c)[0].shape[0] for c in n_class ]
    for i in range(iteration):
        smallest_class = np.argsort(class_size)[i]
        new_split[ new_split == smallest_class ] = np.argsort(class_size)[i+1]
        
    return new_split

def combine_MF(MF1,idx1, MF2,idx2):
    
    MF_combine = copy.deepcopy(MF1)
    
    MF_combine.subjlist = list(np.array(MF1.subjlist)[idx1])+list(np.array(MF2.subjlist)[idx2])
    MF_combine.M = np.vstack((MF1.M[idx1], MF2.M[idx2]))
    MF_combine.M_raw = np.vstack((MF1.M_raw[idx1], MF2.M_raw[idx2]))
    MF_combine.confound = np.vstack((MF1.confound[idx1], MF2.confound[idx2]))
    MF_combine.confound_raw = np.vstack((MF1.confound_raw[idx1], MF2.confound_raw[idx2]))

    MF_combine.row_idx = list(np.array(MF1.row_idx)[idx1])+list(np.array(MF2.row_idx)[idx2])
    MF_combine.nan_mask = np.vstack((MF1.nan_mask[idx1], MF2.nan_mask[idx2]))
    
    return MF_combine



def extract_MF(matrix, split_idx):
    split_matrix = matrix_class(matrix.M[split_idx],
                                matrix.M_raw[split_idx],
                                matrix.confound[split_idx],
                                matrix.confound_raw[split_idx],
                                matrix.nan_mask[split_idx],
                                None,
                                None,
                                None,
                                matrix.dataname,
                                list(np.array(matrix.subjlist)[split_idx]),
                                matrix.itemlist,
                                     None, None,
                                     None, None,
                                     None, None)
    
    return split_matrix



def extract_feature_label(matrix, diagnosis_df, diagnoses_subset, confound=True):

    labels = []
    features = []
    for ID in matrix.subjlist:
        if ID in diagnosis_df["EID"].values:
            labels.append(diagnosis_df.loc[diagnosis_df["EID"]==ID])
            row_idx = list(matrix.subjlist).index(ID)
            if confound:
                features.append( np.concatenate((matrix.W[row_idx], matrix.confound[row_idx])) )
            else:
                features.append( matrix.W[row_idx] )
    labels_df = pd.concat(labels)

    labels = []
    for d in diagnoses_subset:
        labels_df[d] = pd.to_numeric(labels_df[d])
        labels.append(labels_df[d].values)
    labels = np.vstack(labels).T
    
    return np.vstack(features), labels


def bestmatch(ref, source, distance='pearsonr', colwise=True):
    
    assert ref.shape[1] <= source.shape[1]
  
    if colwise == False:
        ref = ref.T
        source = source.T
        
    k = ref.shape[1]
    
    cost = np.zeros((ref.shape[1], source.shape[1]))
    for i in range(ref.shape[1]):
        for j in range(source.shape[1]):

            if distance == 'euclidean':
                dist = scipy.spatial.distance.euclidean(ref[:,i], source[:,j])
                cost[i,j] = dist
            if distance == 'peasonr':
                pcoef, _ = scipy.stats.peasonr(ref[:,i], source[:,j])
                cost[i,j] = 1 - np.abs(pcoef)
            if distance == 'spearman':
                scoef, _ = scipy.stats.spearmanr(ref[:,i], source[:,j])
                cost[i,j] = 1 - np.abs(scoef)
            if distance == 'kendall':
                kcoef, _ = scipy.stats.kendalltau(ref[:,i], source[:,j])
                cost[i,j] = 1 - np.abs(kcoef)
            if distance == 'weightedtau':
                wkcoef, _ = scipy.stats.weightedtau(ref[:,i], source[:,j])
                cost[i,j] = 1 - np.abs(wkcoef)
                
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    
    reg = np.zeros((source.shape[0], k))
    for (i, ind) in enumerate(col_ind[:k]):
        reg[:,i] = source[:,ind]
    
    if colwise == False:
        reg = reg.T
        
    return reg, col_ind[:k]