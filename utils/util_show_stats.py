
import os
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot
from matplotlib import cm
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns

from scipy.stats import f_oneway
import scikit_posthocs as sp

from kneed import KneeLocator


def load_errlist(csv_filepath):
    stats = pd.read_csv(csv_filepath)
    
    repeat_idx = np.unique(stats['repeat'].values)
    d_list = np.unique(stats['dimension'].values)
    W_beta_list = np.unique(stats['W_beta'].values)
    Q_beta_list = np.unique(stats['Q_beta'].values)

    train_err_list = []
    valid_err_list = []

    for d in d_list:
        for bw in W_beta_list:
            for bq in Q_beta_list:
                for r in repeat_idx:
                    per_stats = stats.loc[ (stats['repeat'] == r) &
                                           (stats['dimension'] == d) &
                                           (stats['W_beta'] == bw) &
                                           (stats['Q_beta'] == bq) ]

                    if per_stats.shape[0] > 0:
                        per_stats = per_stats.mean()
                        train_err_list.append([d, bw, bq, per_stats['train_error']]) 
                        valid_err_list.append([d, bw, bq, per_stats['valid_error']])
                    # else:
                    #     print(d, bw, bq, r)

    train_err_list = np.stack(train_err_list)
    valid_err_list = np.stack(valid_err_list)
    train_err_list = pd.DataFrame(train_err_list, columns=['dimension','bw','bq','error'])
    valid_err_list = pd.DataFrame(valid_err_list, columns=['dimension','bw','bq','error'])
    return stats, train_err_list, valid_err_list

def show_stats(train_err_list, valid_err_list):
    
    knee_list = []
    
    for bw in np.unique(valid_err_list['bw'].values):
        nq = len(np.unique(valid_err_list['bq'].values))
        fig, ax = pyplot.subplots(nrows=1, ncols=nq, figsize=(10, 2), sharey=True)

        for (idx,bq) in enumerate(np.unique(valid_err_list['bq'].values)):
            sns.lineplot(data=valid_err_list.loc[(valid_err_list['bw'] == bw) & (valid_err_list['bq'] == bq)],
                         x="dimension", y="error", hue='bq', palette="tab10", markers=True, ax=ax[idx], legend=False)
            sns.lineplot(data=train_err_list.loc[(train_err_list['bw'] == bw) & (train_err_list['bq'] == bq)],
                         x="dimension", y="error", hue='bq', palette=sns.color_palette("hls", 1), markers=True, ax=ax[idx], legend=False)
            
            err = valid_err_list.loc[(valid_err_list['bw'] == bw) & (valid_err_list['bq'] == bq)]
            err_mean = err.groupby('dimension', as_index=False)['error'].mean()
            search_range = err_mean['dimension'].values
            reconst_err = err_mean['error'].values
            try: 
                kn = KneeLocator(search_range, reconst_err, curve='convex', direction='decreasing')
                min_index = int(kn.knee)

                min_dimension = search_range[min_index]
                min_err = reconst_err[min_index]

                # print(bw, bq, min_dimension, min_err)

                knee_list.append([bw, bq, min_dimension, min_err])

                ax[idx].scatter(min_dimension, min_err, marker='o')
            except:
                pass
            
            if idx == 0:
                ax[idx].set_title('beta W={}\n beta Q={}'.format(bw, bq))
            else:
                ax[idx].set_title('beta Q={}'.format(bq))
            ax[idx].set_xticks(np.unique(valid_err_list['dimension'].values))
        pyplot.show()
        
    return np.vstack(knee_list)


def optimal_stats(valid_err_list):
    
    knee_list = []
    
    for bw in np.unique(valid_err_list['bw'].values):
        nq = len(np.unique(valid_err_list['bq'].values))

        for (idx,bq) in enumerate(np.unique(valid_err_list['bq'].values)):
            
            err = valid_err_list.loc[(valid_err_list['bw'] == bw) & (valid_err_list['bq'] == bq)]
            err_mean = err.groupby('dimension', as_index=False)['error'].mean()
            search_range = err_mean['dimension'].values
            reconst_err = err_mean['error'].values
            
            try:
                kn = KneeLocator(search_range, reconst_err, curve='convex', direction='decreasing')
                min_index = int(kn.knee)
                min_dimension = search_range[min_index]
                min_err = reconst_err[min_index]

                knee_list.append([bw, bq, min_dimension, min_err])
 
            except:
                pass
    
    knee_list = np.vstack(knee_list)
    
    optimal_setting = knee_list[np.argmin(knee_list[:,-1])]
    return optimal_setting

        
def show_distribution(stats):
    
    repeat_idx = np.unique(stats['repeat'].values)
    d_list = np.unique(stats['dimension'].values)
    W_beta_list = np.unique(stats['W_beta'].values)
    Q_beta_list = np.unique(stats['Q_beta'].values)
    
    output_validation = []

    for d in d_list:
        train_err_list = []
        valid_err_list = []
        for r in repeat_idx:
            for bw in W_beta_list:
                for bq in Q_beta_list:
                    per_stats = stats.loc[ (stats['repeat'] == r) &
                                           (stats['dimension'] == d) &
                                           (stats['W_beta'] == bw) &
                                           (stats['Q_beta'] == bq) ]

                    if len(per_stats) > 0:
                        per_stats = per_stats.mean()
                        train_err_list.append([d, bw, bq, per_stats['train_error']])
                        valid_err_list.append([d, bw, bq, per_stats['valid_error']])

        train_err_list = np.stack(train_err_list)
        valid_err_list = np.stack(valid_err_list)

        for i in range(len(W_beta_list)):
            distribution_list = []
            setting_list = []

            fig, ax = pyplot.subplots(nrows=1, ncols=len(Q_beta_list), figsize=(10, 1), sharey=True)

            for j in range(len(Q_beta_list)):
                bw = W_beta_list[i]
                bq = Q_beta_list[j]
                idx = np.where( (valid_err_list[:,1] == bw) & (valid_err_list[:,2] == bq) )[0]

                sns.histplot(valid_err_list[idx,3], ax=ax[j], kde=True, bins=20)
                ax[j].set_title(str(bq)+', '+str(np.round(np.mean(valid_err_list[idx,3]),3)))
                if j == 0:
                    ax[j].set_ylabel('d={}\n beta W={}'.format(d, bw))
                ax[j].set_yticks([])
                
                # if d == 2 and j == 2:
                if d == 2:
                    output_validation.append(valid_err_list[idx,3])
                
            pyplot.show()
    return output_validation
                       
            
