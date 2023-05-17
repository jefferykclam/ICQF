import os
import pickle
import pandas as pd
import glob
from tqdm import tqdm, trange
import joblib

# math imports
import numpy as np
import scipy
import sklearn

import warnings
import matplotlib
from matplotlib import pyplot

import copy

from src.data_class import matrix_class
from src.ICQF import ICQF

from utils.util import extract_MF


### train ICQF on CBCL-HBN with different sample size

MF_data = pickle.load(open('./data/HBN/processed/dataclass/CBCL.pickle', 'rb'))
MF_data.row_idx = np.arange(MF_data.M.shape[0])
MF_data.col_idx = np.arange(MF_data.M.shape[1])

clf = ICQF(8, rho=3.0, tau=3.0, regularizer=1,
           W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
           M_upperbd=(True, 1),
           W_beta=0.5, Q_beta=0.5, random_state=0)

MF_data, _ = clf.fit_transform(MF_data)
full_filepath = './output/CBCL-HBN_split//ICQF-full.pickle'
with open(full_filepath, 'wb') as handle:
    pickle.dump(MF_data , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
repeat_list = np.arange(30)
portion_list = [100, 80, 60, 40, 20]

# these hyperparameters are estimated using Blockwise cross validation.
bw_list = [0.1, 0.1, 0, 0, 0]
bq_list = [0.1, 0.1, 0.2, 0.2, 0.5]
dim_list = [8, 6, 5, 5, 4]

for repeat in repeat_list:
    dimensions = []
    for index, portion in enumerate(portion_list):

        matrix = pickle.load(open('./output/CBCL-HBN_split/repeat_{}/portion_{}.pickle'.format(repeat+1, portion), 'rb'))
        matrix_heldout = pickle.load(open('./output/CBCL-HBN_split/repeat_{}/heldout.pickle'.format(repeat+1), 'rb'))
        
        clf = ICQF(int(dim_list[index]), rho=3.0, tau=3.0, regularizer=1,
                   W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
                   M_upperbd=(True, 1), max_iter=500,
                   W_beta=bw_list[index], Q_beta=bq_list[index], random_state=0)
        
        matrix, _ = clf.fit_transform(matrix)
        matrix_heldout, _ = clf.transform( matrix_heldout )
        
        heldin_filepath = './output/CBCL-HBN_split/repeat_{}/ICQF-heldin_portion-{}.pickle'.format(repeat+1, portion)
        with open(heldin_filepath, 'wb') as handle:
            pickle.dump(matrix , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        heldout_filepath = './output/CBCL-HBN_split/repeat_{}/ICQF-heldout_portion-{}.pickle'.format(repeat+1, portion)
        with open(heldout_filepath, 'wb') as handle:
            pickle.dump(matrix_heldout , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
            
### train ICQF on full-HBN dataset


repeat_list = np.arange(30)

for repeat in repeat_list:
    
    split = np.load('./output/full-HBN_split/repeat_{}/split.npz'.format(repeat+1))
    train_idx = list(split['train'])
    valid_idx = list(split['valid'])
    test_idx = list(split['test'])

    matrix_merge = pickle.load( open('./output/full-HBN_split/merge.pickle', 'rb') )
    matrix_train = extract_MF(matrix_merge, train_idx)
    matrix_valid = extract_MF(matrix_merge, valid_idx)
    matrix_test = extract_MF(matrix_merge, test_idx)
    
    clf = ICQF(22, rho=3.0, tau=3.0, regularizer=1,
               W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
               M_upperbd=(True, 1),
               W_beta=0.2, Q_beta=0.2, random_state=0)
    
    matrix_train, _ = clf.fit_transform(matrix_train)
    matrix_valid, _ = clf.transform(matrix_valid)
    matrix_test, _ = clf.transform(matrix_test)

    train_filepath = './output/full-HBN_split/repeat_{}/ICQF-train.pickle'.format(repeat+1)
    with open(train_filepath, 'wb') as handle:
        pickle.dump(matrix_train , handle, protocol=pickle.HIGHEST_PROTOCOL)

    valid_filepath = './output/full-HBN_split/repeat_{}/ICQF-valid.pickle'.format(repeat+1, portion)
    with open(valid_filepath, 'wb') as handle:
        pickle.dump(matrix_valid , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    test_filepath = './output/full-HBN_split/repeat_{}/ICQF-test.pickle'.format(repeat+1, portion)
    with open(test_filepath, 'wb') as handle:
        pickle.dump(matrix_test , handle, protocol=pickle.HIGHEST_PROTOCOL)