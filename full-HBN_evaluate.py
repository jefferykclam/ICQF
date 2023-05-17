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

import matplotlib
from matplotlib import pyplot
import seaborn as sns

import copy

from utils.util import extract_MF, bestmatch
from utils.util_classifier import *

from src.ICQF import ICQF
from src.data_class import matrix_class

import warnings
warnings.filterwarnings('ignore')




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


top_diagnoses = ['Depression', 'BPD', 'Schizophrenia_Psychosis',
       'Panic_Agoraphobia_SeparationAnx_SocialAnx', 'Specific_Phobia',
       'GenAnxiety', 'OCD', 'Eating_Disorder', 'ADHD',
       'ODD_ConductDis', 'Suspected_ASD', 'Substance_Issue',
       'PTSD_Trauma', 'Sleep_Probs', 'Suicidal_SelfHarm_Homicidal']


def extract_matrix_label(matrix, diagnosis_df):
    row_idx = []
    labels = []
    ID_list = []
    for ID in diagnosis_df["EID"].values:
        if ID in matrix.subjlist:
            row_idx.append(matrix.subjlist.index(ID))
            labels.append(diagnosis_df.loc[diagnosis_df["EID"]==ID])
            ID_list.append(ID)
    row_idx = np.stack(row_idx).squeeze()
    
    matrix_select = matrix_class(matrix.M[row_idx],
                                 matrix.M_raw[row_idx],
                                 matrix.confound[row_idx],
                                 matrix.confound_raw[row_idx],
                                 matrix.nan_mask[row_idx],
                                 row_idx,
                                 matrix.col_idx,
                                 None,
                                 matrix.dataname,
                                 list(np.array(matrix.subjlist)[row_idx]),
                                 matrix.itemlist,
                                 matrix.W[row_idx], matrix.Q,
                                 matrix.C[row_idx], matrix.Qc,
                                 matrix.Z[row_idx], matrix.aZ)

    labels_df = pd.concat(labels)
    labels = labels_df.iloc[:,2:].values # ignore 'EID' and 'anonymized ID'
    variable = labels_df.columns[2:]
    
    return matrix_select, labels, variable




repeat_list = np.arange(30)

n_list = [40, 35, 30, 25, 20, 15, 10, 5, 3, 1]

for repeat in repeat_list:

    train_filename = open('./output/full-HBN_split/repeat_{}/ICQF-train.pickle'.format(repeat+1), 'rb')
    matrix_train = pickle.load( train_filename )

    valid_filename = open('./output/full-HBN_split/repeat_{}/ICQF-valid.pickle'.format(repeat+1), 'rb')
    matrix_valid = pickle.load( valid_filename )
    
    test_filename = open('./output/full-HBN_split/repeat_{}/ICQF-test.pickle'.format(repeat+1), 'rb')
    matrix_test = pickle.load( test_filename )
    
    matrix_train, train_label, variable = extract_matrix_label(matrix_train, diagnosis_df)
    matrix_valid, valid_label, _ = extract_matrix_label(matrix_valid, diagnosis_df)
    matrix_test, test_label, _ = extract_matrix_label(matrix_test, diagnosis_df)

    train_Q = matrix_train.Q
    
    for n in n_list:
        
        metric_results_list = []
        select_dir = './output/variable_reduction/repeat_{}/select_{}'.format(repeat+1, n)
        os.makedirs(select_dir, exist_ok=True)

        
        variable_idx = []
        for k in range(train_Q.shape[1]):
            variable_idx += list(np.argsort(np.abs(train_Q[:,k]))[::-1][:n])
        variable_idx = list(np.unique(variable_idx))

        select_train = copy.deepcopy(matrix_train)
        select_valid = copy.deepcopy(matrix_valid)
        select_test = copy.deepcopy(matrix_test)

        select_train.M = select_train.M[:, variable_idx]
        select_valid.M = select_valid.M[:, variable_idx]
        select_test.M = select_test.M[:, variable_idx]
        
        select_train.M_raw = select_train.M_raw[:, variable_idx]
        select_valid.M_raw = select_valid.M_raw[:, variable_idx]
        select_test.M_raw = select_test.M_raw[:, variable_idx]
        
        select_train.nan_mask = select_train.nan_mask[:, variable_idx]
        select_valid.nan_mask = select_valid.nan_mask[:, variable_idx]
        select_test.nan_mask = select_test.nan_mask[:, variable_idx]
        
        select_model = ICQF(22, rho=3.0, tau=3.0, regularizer=1,
                       W_upperbd=(True, 1.0), Q_upperbd=(True, 1),
                       M_upperbd=(True, 1),
                       W_beta=0.2, Q_beta=0.2, random_state=0)
        
        select_train, _ = select_model.fit_transform(select_train)
        select_valid, _ = select_model.transform(select_valid)
        select_test, _ = select_model.transform(select_test)
        
        train_feature = np.hstack((select_train.W, select_train.C))
        valid_feature = np.hstack((select_valid.W, select_valid.C))
        test_feature = np.hstack((select_test.W, select_test.C))

        for (idx, v) in enumerate(variable):
            try:
                model = optimize_LR(train_feature, train_label[:,idx], random_state=0)


                metric_results, prediction_results = inference_LR(test_feature, test_label[:,idx], model)
                metric_results['diagnosis'] = v
                mean_acc = np.round(metric_results['accuracy'].mean(), 2)
                mean_auc = np.round(metric_results['auc'].mean(), 2)
                print('{} -- accuracy : {}, AUC : {}'.format(v, mean_acc, mean_auc))
                metric_results_list.append(metric_results)
            except:
                pass


        metric_results_df = pd.concat(metric_results_list)
        metric_results_df['select'] = n
        metric_results_df['nvariables'] = len(variable_idx)
        
metric_filename = './output/full-HBN_metric_importance.pickle'
with open(metric_filename, 'wb') as handle:
    pickle.dump(metric_results_df, handle, protocol=4)