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


### train ICQF on CBCL-ABCD with different sample size

MF_data = pickle.load(open('./data/ABCD/processed/dataclass/CBCL.pickle', 'rb'))
MF_data.row_idx = np.arange(MF_data.M.shape[0])
MF_data.col_idx = np.arange(MF_data.M.shape[1])

clf = ICQF(7, rho=3.0, tau=3.0, regularizer=1,
           W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
           M_upperbd=(True, 1),
           W_beta=0.0, Q_beta=0.1, random_state=0)

MF_data, _ = clf.fit_transform(MF_data)
full_filepath = './output/CBCL-ABCD_split/ICQF-full.pickle'
with open(full_filepath, 'wb') as handle:
    pickle.dump(MF_data , handle, protocol=pickle.HIGHEST_PROTOCOL)
    

repeat_list = np.arange(1, 30)
portion_list = [100, 80, 60, 40, 20]

# these hyperparameters are estimated using Blockwise cross validation.
bw_list = [0.0, 0.0, 0.0, 0.0, 0.0]
bq_list = [0.1, 0.1, 0.2, 0.2, 0.5]
dim_list = [7, 7, 7, 7, 7]

for repeat in repeat_list:
    dimensions = []
    for index, portion in enumerate(portion_list):

        matrix = pickle.load(open('./output/CBCL-ABCD_split/repeat_{}/portion_{}.pickle'.format(repeat+1, portion), 'rb'))
        matrix_heldout = pickle.load(open('./output/CBCL-ABCD_split/repeat_{}/heldout.pickle'.format(repeat+1), 'rb'))
        
        clf = ICQF(int(dim_list[index]), rho=3.0, tau=3.0, regularizer=1,
                   W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
                   M_upperbd=(True, 1),
                   W_beta=bw_list[index], Q_beta=bq_list[index], random_state=0)
        
        matrix, _ = clf.fit_transform(matrix)
        matrix_heldout, _ = clf.transform( matrix_heldout )
        
        heldin_filepath = './output/CBCL-ABCD_split/repeat_{}/ICQF-heldin_portion-{}.pickle'.format(repeat+1, portion)
        with open(heldin_filepath, 'wb') as handle:
            pickle.dump(matrix , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        heldout_filepath = './output/CBCL-ABCD_split/repeat_{}/ICQF-heldout_portion-{}.pickle'.format(repeat+1, portion)
        with open(heldout_filepath, 'wb') as handle:
            pickle.dump(matrix_heldout , handle, protocol=pickle.HIGHEST_PROTOCOL)