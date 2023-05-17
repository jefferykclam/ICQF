import os
import pickle
import pandas as pd
import glob
from tqdm import tqdm, trange

# math imports
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import warnings
import matplotlib
from matplotlib import pyplot

import copy
from functools import reduce

from src.data_class import matrix_class
from utils.util import generate_class, combine_MF


MF_data = pickle.load(open('./data/ABCD/processed/dataclass/CBCL.pickle', 'rb'))
MF_data.row_idx = np.arange(MF_data.M.shape[0])
MF_data.col_idx = np.arange(MF_data.M.shape[1])


unique_EID, unique_idx = np.unique(MF_data.subjlist, return_index=True)
duplicates = list(set(np.arange(len(MF_data.subjlist))) - set(unique_idx))
if len(duplicates) > 0: print(duplicates)
assert len(duplicates) == 0

print('Unique subjects : {}'.format(len(MF_data.subjlist)))
min_age = np.min(MF_data.confound_raw[:,0])
max_age = np.max(MF_data.confound_raw[:,0])
print('Age range : {:.2f} -- {:.2f}'.format(min_age, max_age))

diagnosis_df = pickle.load(open('./data/ABCD/processed/diagnosis.pickle','rb'))
diagnosis_df.insert(0, 'EID', diagnosis_df.pop('EID'))

print('Subjects with diagnostic information : {}'.format(diagnosis_df.shape[0]))

for repeat in range(30):
    os.makedirs('./output/CBCL-ABCD_split/repeat_{}'.format(repeat+1), exist_ok=True)
    
    wd_idx = []
    wod_idx = []
    for row, ID in enumerate(MF_data.subjlist):
        if ID in diagnosis_df["EID"].values:
            wd_idx.append(row)
        else:
            wod_idx.append(row)
    print(len(wod_idx))
    
    MF_wd = copy.deepcopy(MF_data)
    MF_wd.subjlist = list(np.array(MF_wd.subjlist)[wd_idx])
    MF_wd.M = MF_wd.M[wd_idx]
    MF_wd.M_raw = MF_wd.M_raw[wd_idx]
    MF_wd.confound = MF_wd.confound[wd_idx]
    MF_wd.confound_raw = MF_wd.confound_raw[wd_idx]
    MF_wd.row_idx = list(np.array(MF_wd.row_idx)[wd_idx])
    MF_wd.nan_mask = MF_wd.nan_mask[wd_idx]
    
    wd_class = generate_class(MF_wd, n_age=2)
    
    ### create held out on subjects with labels only (20%)
    nsplit = 5
    wd_heldin = []
    wd_idx = np.arange(MF_wd.M.shape[0])
    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=repeat)
    for i, (_, wd_index) in enumerate(skf.split(wd_idx, wd_class)):
        if i == 0:
            wd_heldout = wd_index
        else:
            wd_heldin.append(wd_index)

    MF_heldout = copy.deepcopy(MF_wd)
    MF_heldout.subjlist = list(np.array(MF_wd.subjlist)[wd_heldout])
    MF_heldout.M = MF_wd.M[wd_heldout]
    MF_heldout.M_raw = MF_wd.M_raw[wd_heldout]
    MF_heldout.confound = MF_wd.confound[wd_heldout]
    MF_heldout.confound_raw = MF_wd.confound_raw[wd_heldout]
    MF_heldout.row_idx = list(np.array(MF_wd.row_idx)[wd_heldout])
    MF_heldout.nan_mask = MF_wd.nan_mask[wd_heldout]

    wd_heldin = [item for sublist in wd_heldin for item in sublist]
    print(len(wd_heldin))
    # MF_heldin = combine_MF(MF_wd, wd_heldin, MF_wod, wod_heldin)
    
    MF_heldin = copy.deepcopy(MF_wd)
    MF_heldin.subjlist = list(np.array(MF_wd.subjlist)[wd_heldin])
    MF_heldin.M = MF_wd.M[wd_heldin]
    MF_heldin.M_raw = MF_wd.M_raw[wd_heldin]
    MF_heldin.confound = MF_wd.confound[wd_heldin]
    MF_heldin.confound_raw = MF_wd.confound_raw[wd_heldin]
    MF_heldin.row_idx = list(np.array(MF_wd.row_idx)[wd_heldin])
    MF_heldin.nan_mask = MF_wd.nan_mask[wd_heldin]

    print(MF_heldin.M.shape, MF_heldout.M.shape)
    
    np.savez('./output/CBCL-ABCD_split/repeat_{}/heldout.npz'.format(repeat+1),
         M=MF_heldout.M,
         M_raw=MF_heldout.M_raw,
         nan_mask=MF_heldout.nan_mask,
         confound=MF_heldout.confound,
         confound_raw=MF_heldout.confound_raw)
    
    with open('./output/CBCL-ABCD_split/repeat_{}/heldout.pickle'.format(repeat+1), 'wb') as handle:
        pickle.dump(MF_heldout , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    wd_idx = []
    wod_idx = []
    for row, ID in enumerate(MF_heldin.subjlist):
        if ID in diagnosis_df["EID"].values:
            wd_idx.append(row)
        else:
            wod_idx.append(row)
    
    
    MF_wd = copy.deepcopy(MF_heldin)
    MF_wd.subjlist = list(np.array(MF_wd.subjlist)[wd_idx])
    MF_wd.M = MF_wd.M[wd_idx]
    MF_wd.M_raw = MF_wd.M_raw[wd_idx]
    MF_wd.confound = MF_wd.confound[wd_idx]
    MF_wd.confound_raw = MF_wd.confound_raw[wd_idx]
    MF_wd.row_idx = list(np.array(MF_wd.row_idx)[wd_idx])
    MF_wd.nan_mask = MF_wd.nan_mask[wd_idx]
    
    wd_class = generate_class(MF_wd)

    nsplit = 10
    wd_list = []
    wd_idx = np.arange(MF_wd.M.shape[0])
    skf = StratifiedKFold(n_splits=nsplit)
    for i, (_, wd_index) in enumerate(skf.split(wd_idx, wd_class)):
        wd_list.append(wd_index)

    acc_wd = []
    for i in range(nsplit):
        temp = []
        for j in range(i, nsplit):
            temp += list(wd_list[j])
        acc_wd.append(temp)


    for k in range(nsplit):
        
        MF_k = copy.deepcopy(MF_wd)
        MF_k.subjlist = list(np.array(MF_wd.subjlist)[acc_wd[k]])
        MF_k.M = MF_wd.M[acc_wd[k]]
        MF_k.M_raw = MF_wd.M_raw[acc_wd[k]]
        MF_k.confound = MF_wd.confound[acc_wd[k]]
        MF_k.confound_raw = MF_wd.confound_raw[acc_wd[k]]
        MF_k.row_idx = list(np.array(MF_wd.row_idx)[acc_wd[k]])
        MF_k.nan_mask = MF_wd.nan_mask[acc_wd[k]]

        print(k, MF_k.M.shape)

        np.savez('./output/CBCL-ABCD_split/repeat_{}/portion_{}.npz'.format(repeat+1, int(100-k*nsplit)),
             M=MF_k.M,
             M_raw=MF_k.M_raw,
             nan_mask=MF_k.nan_mask,
             confound=MF_k.confound,
             confound_raw=MF_k.confound_raw)
        
        with open('./output/CBCL-ABCD_split/repeat_{}/portion_{}.pickle'.format(repeat+1, int(100-k*nsplit)), 'wb') as handle:
            pickle.dump(MF_k , handle, protocol=pickle.HIGHEST_PROTOCOL)