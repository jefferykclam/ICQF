import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import *

import seaborn as sns
import copy

from scipy.spatial.distance import cosine

def simulation(nrow, ncol, ndict, overlap=None, density=0.3,
               Q_upperbd=100, lowerbd=0,
               noise=False, delta=0.1,
               missing_ratio = 0.0,
               confound=False,
               visualize=True):
    
    if overlap is None:
        overlap = int(nrow//40)
    
    D = np.zeros((nrow, ndict))
    for k in range(ndict):
        for j in range(nrow):
            if (j >= (k)*( (nrow//ndict) )) and (j <= (nrow//ndict) + overlap + (k)*((nrow//ndict))):
                D[j,k] += np.random.uniform(0.5, 1.0)*np.random.binomial(1, p=0.9)
            else:
                D[j,k] = 0
    D[D > 1] = 1
                
    A = np.random.binomial(size=(ndict,ncol), n=1, p=density)
    A = 1.0*(A > 0)
    S = np.random.uniform(low=lowerbd, high=Q_upperbd, size=(ndict,ncol))
    A = A*S
    
    if confound:
        
        confound_sex = np.zeros((nrow, 1))
        confound_sex[nrow//2:,0] = 1
        confound_age = np.linspace(0,1,nrow)[:,None]
        confound = np.hstack((confound_sex, confound_age))
        C = np.hstack((confound, 1.0-confound))
        C = np.hstack((C, np.ones((C.shape[0], 1))))
        
        # C = np.zeros((nrow, 2))
        # C[:nrow//2,0] = 1
        # C[nrow//2:,1] = 1

        # W2 = np.vstack((C[:,0]*np.random.binomial(size=(nrow), n=1, p=0.9),
        #                 C[:,1]*np.random.binomial(size=(nrow), n=1, p=0.9))).T
        
        W2 = C
        # A21 = np.ones(ncol)*np.random.binomial(size=(ncol), n=1, p=0.3)
        # A22 = np.ones(ncol)*np.random.binomial(size=(ncol), n=1, p=0.3)
        # A2 = np.vstack((A21, A22))
        
        A2 = np.random.binomial(size=(C.shape[1],ncol), n=1, p=0.1)
        # A2 = 1.0*(A2 > 0)
        S2 = np.random.uniform(low=lowerbd, high=Q_upperbd, size=(C.shape[1],ncol))
        A2 = A2*S2
        
        M = D@A + W2@A2
        true_W = np.hstack((D, W2))
        true_Q = np.vstack((A, A2)).T

    else:
        confound = None
        C = None
        M = D@A
        true_W = D
        true_Q = A.T
    
    M_clean = copy.deepcopy(M)
    if noise:
        noise_indicator = np.random.choice((0, 1), size=(nrow, ncol), p=[(1.0 - delta), delta])
        noise_matrix = (noise_indicator == 1) * np.random.uniform(-np.max(M), np.max(M), size=(nrow, ncol))
        M +=noise_matrix
        M[M < 0] = 0
        M[M > np.max(M)] = np.max(M)
        
    if missing_ratio > 0:
        nan_mask = np.random.choice((0, 1), size=(nrow, ncol), p=[missing_ratio, (1-missing_ratio)])
        M[nan_mask] = np.nan
    else:
        nan_mask = np.ones_like(M)
        
    if visualize:
        fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3, figsize=(12, 3))
        
        if noise:
            sns.heatmap(M, ax=ax1, cmap="Blues", mask=1-nan_mask, cbar=False)
            ax1.set_title('$M, \delta$={}, range:[{:.2f}, {:.2f}]'.format(delta, np.nanmin(M), np.nanmax(M)))
            bottom, top = ax1.get_ylim()
            ax1.set_ylim(bottom + 1.5, top - 0.5)   
        else:
            sns.heatmap(M_clean, ax=ax1, cmap="Blues", mask=1-nan_mask, cbar=False)
            ax1.set_title('$M, range:[{:.2f}, {:.2f}]'.format(np.nanmin(M), np.nanmax(M)))
            bottom, top = ax1.get_ylim()
            ax1.set_ylim(bottom + 1.5, top - 0.5)

        sns.heatmap(true_W, ax=ax2, cmap="Blues", cbar=False)
        if confound is not None:
            ax2.set_title('$[W, C]$, range:[{:.2f}, {:.2f}]'.format(np.min(true_W), np.max(true_W)))
        else:
            ax2.set_title('$W$, range:[{:.2f}, {:.2f}]'.format(np.min(true_W), np.max(true_W)))
        bottom, top = ax2.get_ylim()
        ax2.set_ylim(bottom + 1.5, top - 0.5)

        sns.heatmap(true_Q, ax=ax3, cmap="Blues", cbar=False)
        if confound is not None:
            ax3.set_title('$[Q, Q_c]$, range:[{:.2f}, {:.2f}]'.format(np.min(true_Q), np.max(true_Q)))
        else:
            ax3.set_title('$Q$, range:[{:.2f}, {:.2f}]'.format(np.min(true_Q), np.max(true_Q)))
        bottom, top = ax3.get_ylim()
        ax3.set_ylim(bottom + 1.5, top - 0.5)

        pyplot.tight_layout()
        pyplot.show()
    
    return true_W, true_Q, confound, M_clean, M, nan_mask



def greedy_sort(W, true_W):
    bestmatch_ordering = np.array([])
    bestmatch_cor = np.array([])
    tempidx = np.arange(W.shape[1])
    temp_W = W
    match_W = np.zeros_like(W)
    
    for k in range(W.shape[1]):
        pc = np.array([])
        pcorrelation=np.array([])

        if k < true_W.shape[1]:

            for d in range(temp_W.shape[1]):
                pc = np.append(pc, 1 - cosine(true_W[:,k], temp_W[:,d]))
            sort_idx = np.argsort(-pc)
            match_W[:,k] = temp_W[:, sort_idx[0]]
            bestmatch_ordering = np.append(bestmatch_ordering, tempidx[sort_idx[0]])
            bestmatch_cor = np.append(bestmatch_cor, pc[sort_idx[0]])

            temp_W = np.delete(temp_W, sort_idx[0], axis=1)
            tempidx = np.delete(tempidx, sort_idx[0])

        else:
            if temp_W.shape[1] > 0:
                sort_idx = np.argsort(-np.max(temp_W,axis=0))
                temp_W = temp_W[:,sort_idx]
                match_W[:,k] = temp_W[:,0]
                temp_W = np.delete(temp_W, sort_idx[0], axis=1)
                
    return match_W, bestmatch_ordering, bestmatch_cor


def show_synthetic_result(MF_data, true_W, true_Q):

    match_W, orders, corr = greedy_sort(MF_data.W, true_W)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = pyplot.subplots(2, 3, figsize=(12, 5))
        

    sns.heatmap(MF_data.M, ax=ax1, cmap="Blues", mask=1-MF_data.nan_mask, cbar=False)
    ax1.set_title('$M$, range:[{:.2f}, {:.2f}]'.format(np.nanmin(MF_data.M), np.nanmax(MF_data.M)))
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 1.5, top - 0.5)

    sns.heatmap(true_W, ax=ax2, cmap="Blues", cbar=False)
    # ax2.set_title('W, range:[{:.2f}, {:.2f}]'.format(np.min(true_W), np.max(true_W)))
    if MF_data.C is not None:
        ax2.set_title('$[W, C]$, range:[{:.2f}, {:.2f}]'.format(np.min(true_W), np.max(true_W)))
    else:
        ax2.set_title('$W$, range:[{:.2f}, {:.2f}]'.format(np.min(true_W), np.max(true_W)))
    bottom, top = ax2.get_ylim()
    ax2.set_ylim(bottom + 1.5, top - 0.5)

    sns.heatmap(true_Q, ax=ax3, cmap="Blues", cbar=False)
    # ax3.set_title('Q, range:[{:.2f}, {:.2f}]'.format(np.min(true_Q), np.max(true_Q)))
    if MF_data.C is not None:
        ax3.set_title('$[Q, Q_c]$, range:[{:.2f}, {:.2f}]'.format(np.min(true_Q), np.max(true_Q)))
    else:
        ax3.set_title('$Q$, range:[{:.2f}, {:.2f}]'.format(np.min(true_Q), np.max(true_Q)))
    bottom, top = ax3.get_ylim()
    ax3.set_ylim(bottom + 1.5, top - 0.5)

    
    sns.heatmap(MF_data.W@(MF_data.Q.T), ax=ax4, cmap="Blues", cbar=False)
    ax4.set_title('$WQ^T$, range:[{:.2f}, {:.2f}]'.format(np.min(MF_data.W@(MF_data.Q.T)), np.max(MF_data.W@(MF_data.Q.T))))
    bottom, top = ax4.get_ylim()
    ax4.set_ylim(bottom + 1.5, top - 0.5)

    if MF_data.C is None:
        sns.heatmap(match_W, ax=ax5, cmap="Blues", cbar=False)
        ax5.set_title('W, range:[{:.2f}, {:.2f}]'.format(np.min(MF_data.W), np.max(MF_data.W)))
        bottom, top = ax5.get_ylim()
        ax5.set_ylim(bottom + 1.5, top - 0.5)
        
        sns.heatmap(MF_data.Q[:, orders.astype('int')], ax=ax6, cmap="Blues", cbar=False)
        ax6.set_title('Q, range:[{:.2f}, {:.2f}]'.format(np.min(MF_data.Q), np.max(MF_data.Q)))
        bottom, top = ax6.get_ylim()
        ax6.set_ylim(bottom + 1.5, top - 0.5)
    else:
        sns.heatmap(np.hstack((match_W, MF_data.C)), ax=ax5, cmap="Blues", cbar=False)
        ax5.set_title('[W, C], range:[{:.2f}, {:.2f}]'.format(np.min(MF_data.W), np.max(MF_data.W)))
        bottom, top = ax5.get_ylim()
        ax5.set_ylim(bottom + 1.5, top - 0.5)
    
        sns.heatmap(np.hstack((MF_data.Q[:, orders.astype('int')], MF_data.Qc)), ax=ax6, cmap="Blues", cbar=False)
        ax6.set_title('[Q, Qc], range:[{:.2f}, {:.2f}]'.format(np.min(MF_data.Q), np.max(MF_data.Q)))
        bottom, top = ax6.get_ylim()
        ax6.set_ylim(bottom + 1.5, top - 0.5)
    
    pyplot.tight_layout()
    pyplot.show()
    