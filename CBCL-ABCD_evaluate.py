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

from utils.util import extract_feature_label, bestmatch
from utils.util_classifier import *

import warnings
warnings.filterwarnings('ignore')


diagnosis_df = pickle.load(open('./data/ABCD/processed/diagnosis.pickle','rb'))
diagnosis_df.insert(0, 'EID', diagnosis_df.pop('EID'))


top_diagnoses = ['Depression', 'BPD', 'Schizophrenia_Psychosis',
       'Panic_Agoraphobia_SeparationAnx_SocialAnx', 'Specific_Phobia',
       'GenAnxiety', 'OCD', 'Eating_Disorder', 'ADHD',
       'ODD_ConductDis', 'Suspected_ASD', 'Substance_Issue',
       'PTSD_Trauma', 'Sleep_Probs', 'Suicidal_SelfHarm_Homicidal']


repeat_list = np.arange(1, 30)
portion_list = [80, 60, 40, 20]

metric_results_list = []

for repeat in repeat_list:
    for index, portion in enumerate(portion_list):

        heldin_filepath = './output/CBCL-ABCD_split/repeat_{}/ICQF-heldin_portion-{}.pickle'.format(repeat+1, portion)
        matrix = pickle.load(open(heldin_filepath, 'rb'))
        
        heldout_filepath = './output/CBCL-ABCD_split/repeat_{}/ICQF-heldout_portion-{}.pickle'.format(repeat+1, portion)
        matrix_heldout = pickle.load(open(heldout_filepath, 'rb'))
        
        x_train, y_train = extract_feature_label(matrix, diagnosis_df, top_diagnoses)
        x_test, y_test = extract_feature_label(matrix_heldout, diagnosis_df, top_diagnoses)
        
        for idx in range(len(top_diagnoses)):
            try:
                model = optimize_LR(x_train, y_train[:,idx], random_state=0, regularizer=2)

                metric_results, prediction_results = inference_LR(x_test, y_test[:,idx], model)
                metric_results['diagnosis'] = top_diagnoses[idx]
                metric_results['portion'] = portion
                metric_results['repeat'] = repeat
                mean_acc = metric_results['accuracy'].mean()
                mean_auc = metric_results['auc'].mean()
                print('{} -- accuracy : {:.3f}, AUC : {:.3f}'.format(top_diagnoses[idx], mean_acc, mean_auc))
                metric_results_list.append(metric_results)

            except:
                pass

        metric_results = pd.concat(metric_results_list)
        metric_results.reset_index(drop=True)
        
metric_filename = './output/CBCL-ABCD_metric_decreasing_samples.pickle'
with open(metric_filename, 'wb') as handle:
    pickle.dump(metric_results, handle, protocol=4)