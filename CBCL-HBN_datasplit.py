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


MF_data = pickle.load(open('./data/HBN/processed/dataclass/CBCL.pickle', 'rb'))
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

diagnosis_df = pd.read_csv('./data/HBN/HBN_participantDx.csv')
diagnosis_df = diagnosis_df.iloc[1:, :]
basic_demos_df = pd.read_csv('./data/HBN/9994_Basic_Demos_20210310.csv')
basic_demos_df = basic_demos_df.iloc[1:,:]

EID_list = []
for AID in diagnosis_df['anonymized_id'].values:
    EID = basic_demos_df.loc[basic_demos_df['Anonymized ID'] == AID]['EID'].values[0]
    EID_list.append(str(EID))
diagnosis_df['EID'] = EID_list
diagnosis_df = diagnosis_df.iloc[:,1:]
diagnosis_df.insert(0, 'EID', diagnosis_df.pop('EID'))

print('Subjects with diagnostic information : {}'.format(diagnosis_df.shape[0]))

for repeat in range(30):
    os.makedirs('./output/CBCL-HBN_split/repeat_{}'.format(repeat+1), exist_ok=True)
    
    wd_idx = []
    wod_idx = []
    for row, ID in enumerate(MF_data.subjlist):
        if ID in diagnosis_df["EID"].values:
            wd_idx.append(row)
        else:
            wod_idx.append(row)
            
    MF_wod = copy.deepcopy(MF_data)
    MF_wod.subjlist = list(np.array(MF_wod.subjlist)[wod_idx])
    MF_wod.M = MF_wod.M[wod_idx]
    MF_wod.M_raw = MF_wod.M_raw[wod_idx]
    MF_wod.confound = MF_wod.confound[wod_idx]
    MF_wod.confound_raw = MF_wod.confound_raw[wod_idx]
    MF_wod.row_idx = list(np.array(MF_wod.row_idx)[wod_idx])
    MF_wod.nan_mask = MF_wod.nan_mask[wod_idx]
    
    MF_wd = copy.deepcopy(MF_data)
    MF_wd.subjlist = list(np.array(MF_wd.subjlist)[wd_idx])
    MF_wd.M = MF_wd.M[wd_idx]
    MF_wd.M_raw = MF_wd.M_raw[wd_idx]
    MF_wd.confound = MF_wd.confound[wd_idx]
    MF_wd.confound_raw = MF_wd.confound_raw[wd_idx]
    MF_wd.row_idx = list(np.array(MF_wd.row_idx)[wd_idx])
    MF_wd.nan_mask = MF_wd.nan_mask[wd_idx]
    
    wd_class = generate_class(MF_wd)
    wod_class = generate_class(MF_wod)
    
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

    wod_heldin = []
    wod_idx = np.arange(MF_wod.M.shape[0])
    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=repeat)
    for i, (_, wod_index) in enumerate(skf.split(wod_idx, wod_class)):
        wod_heldin.append(wod_index)

    MF_heldout = copy.deepcopy(MF_wd)

    MF_heldout.subjlist = list(np.array(MF_wd.subjlist)[wd_heldout])
    MF_heldout.M = MF_wd.M[wd_heldout]
    MF_heldout.M_raw = MF_wd.M_raw[wd_heldout]
    MF_heldout.confound = MF_wd.confound[wd_heldout]
    MF_heldout.confound_raw = MF_wd.confound_raw[wd_heldout]

    MF_heldout.row_idx = list(np.array(MF_wd.row_idx)[wd_heldout])
    MF_heldout.nan_mask = MF_wd.nan_mask[wd_heldout]


    wd_heldin = [item for sublist in wd_heldin for item in sublist]
    wod_heldin = [item for sublist in wod_heldin for item in sublist]
    print(len(wd_heldin), len(wod_heldin))

    MF_heldin = combine_MF(MF_wd, wd_heldin, MF_wod, wod_heldin)

    print(MF_heldin.M.shape, MF_heldout.M.shape)
    
    np.savez('./output/CBCL-HBN_split/repeat_{}/heldout.npz'.format(repeat+1),
         M=MF_heldout.M,
         M_raw=MF_heldout.M_raw,
         nan_mask=MF_heldout.nan_mask,
         confound=MF_heldout.confound,
         confound_raw=MF_heldout.confound_raw)
    
    with open('./output/CBCL-HBN_split/repeat_{}/heldout.pickle'.format(repeat+1), 'wb') as handle:
        pickle.dump(MF_heldout , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    ### split by percentage
    wd_idx = []
    wod_idx = []
    for row, ID in enumerate(MF_heldin.subjlist):
        if ID in diagnosis_df["EID"].values:
            wd_idx.append(row)
        else:
            wod_idx.append(row)
            
    MF_wod = copy.deepcopy(MF_heldin)
    MF_wod.subjlist = list(np.array(MF_wod.subjlist)[wod_idx])
    MF_wod.M = MF_wod.M[wod_idx]
    MF_wod.M_raw = MF_wod.M_raw[wod_idx]
    MF_wod.confound = MF_wod.confound[wod_idx]
    MF_wod.confound_raw = MF_wod.confound_raw[wod_idx]
    MF_wod.row_idx = list(np.array(MF_wod.row_idx)[wod_idx])
    MF_wod.nan_mask = MF_wod.nan_mask[wod_idx]

    MF_wd = copy.deepcopy(MF_heldin)
    MF_wd.subjlist = list(np.array(MF_wd.subjlist)[wd_idx])
    MF_wd.M = MF_wd.M[wd_idx]
    MF_wd.M_raw = MF_wd.M_raw[wd_idx]
    MF_wd.confound = MF_wd.confound[wd_idx]
    MF_wd.confound_raw = MF_wd.confound_raw[wd_idx]
    MF_wd.row_idx = list(np.array(MF_wd.row_idx)[wd_idx])
    MF_wd.nan_mask = MF_wd.nan_mask[wd_idx]

    wd_class = generate_class(MF_wd)
    wod_class = generate_class(MF_wod)

    
    nsplit = 10
    wd_list = []
    wd_idx = np.arange(MF_wd.M.shape[0])
    skf = StratifiedKFold(n_splits=nsplit)
    for i, (_, wd_index) in enumerate(skf.split(wd_idx, wd_class)):
        wd_list.append(wd_index)

    wod_list = []
    wod_idx = np.arange(MF_wod.M.shape[0])
    skf = StratifiedKFold(n_splits=nsplit)
    for i, (_, wod_index) in enumerate(skf.split(wod_idx, wod_class)):
        wod_list.append(wod_index)

    acc_wd = []
    for i in range(nsplit):
        temp = []
        for j in range(i, nsplit):
            temp += list(wd_list[j])
        acc_wd.append(temp)

    acc_wod = []
    for i in range(nsplit):
        temp = []
        for j in range(i, nsplit):
            temp += list(wod_list[j])
        acc_wod.append(temp)

    for k in range(nsplit):
        MF_k = combine_MF(MF_wd,acc_wd[k], MF_wod,acc_wod[k])

        print(k, MF_k.M.shape)

        np.savez('./output/CBCL-HBN_split/repeat_{}/portion_{}.npz'.format(repeat+1, int(100-k*nsplit)),
             M=MF_k.M,
             M_raw=MF_k.M_raw,
             nan_mask=MF_k.nan_mask,
             confound=MF_k.confound,
             confound_raw=MF_k.confound_raw)
        
        with open('./output/CBCL-HBN_split/repeat_{}/portion_{}.pickle'.format(repeat+1, int(100-k*nsplit)), 'wb') as handle:
            pickle.dump(MF_k , handle, protocol=pickle.HIGHEST_PROTOCOL)
