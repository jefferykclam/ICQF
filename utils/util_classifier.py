import os
import pickle
import pandas as pd
import glob

# math imports
import numpy as np
import scipy
import sklearn

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, r2_score, median_absolute_error


def fit_LRCV(feature, y, random_state=None, regularizer=2):
    if regularizer == 1:
        solver = 'saga'
    else:
        solver = 'lbfgs'
    model = LogisticRegressionCV(penalty='l{}'.format(regularizer),
                                 Cs=10,
                                 scoring='roc_auc',
                                 random_state=random_state, solver=solver,
                                 fit_intercept=True,
                                 class_weight='balanced',
                                 cv=5, max_iter=10000)
    model.fit(feature, y)
    
    return model

def optimize_LR(feature, y, random_state=None, regularizer=1):
    model = fit_LRCV(feature, y, random_state=random_state, regularizer=regularizer)
    return model

def inference_LR(feature, y, model, n_repeats=1):
    
    metric_results = []
    prediction_results = []

    for rep in range(n_repeats):        
        
        metric = {}
        # metric['diagnosis'] = variable
        
        predictions = model.predict_proba(feature)[:,1]

        tn, fp, fn, tp = confusion_matrix(y, 1*(predictions >= 0.5)).ravel()
        f1 = f1_score(y, 1*(predictions >= 0.5))

        metric['repeat'] = rep
        metric['tp'] = tp
        metric['tn'] = tn
        metric['fp'] = fp
        metric['fn'] = fn
        metric['f1'] = f1
        metric['specificity'] = tn / (tn+fp)
        metric['auc'] = roc_auc_score(y, predictions)
        metric['precision'] = precision_score(y, 1*(predictions >= 0.5))
        metric['recall/sensitivity'] = recall_score(y, 1*(predictions >= 0.5))
        metric['accuracy'] = accuracy_score(y, 1*(predictions >= 0.5))

        metric_results.append(pd.DataFrame(metric, index=[0]))
        
        prediction_results.append([predictions, y])
            
    metric_results = pd.concat(metric_results)
    return metric_results, prediction_results
