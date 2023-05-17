import sys
# sys.path.append( os.path.abspath(os.path.join('./')) )

import os
import argparse

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot
import seaborn as sns

import time
from tqdm import tqdm

from kneed import KneeLocator
from sklearn.impute import SimpleImputer
from src.ICQF import ICQF
from src.data_class import matrix_class

class MF_model:
    
    def __init__(self,
                 data_matrix,
                 data_mask,
                 n_components,
                 regularizer=1,
                 W_beta = 0.1,
                 Q_beta = 0.1,
                 rho = 3.0,
                 tau = 3.0,
                 W_upperbd=(False, 1.0),
                 Q_upperbd=(False, 1.0),
                 M_upperbd=(True, 1.0),
                 max_iter=200,
                 confound=None,
                 random_state=0):
        
        # Algorithm to decomposition data matrix M into M = W Q'
        
        # ===input===
        # data_matrix : the data matrix
        # data_mask : binary matrix with same shape as the data matrix, 0=data not avaialble, 1=data available
        #             if all data entries are available, simply set data_mask = np.ones(data_matrix)
        # n_components : latent dimension of the representation
        # regularizer : 1 (sparsity) or 2 (smoothness)
        # W_beta : hyperparameter to control regularizer of W. Zero corresponds to no regularization.
        # Q_beta : hyperparameter to control regularizer of Q. Zero corresponds to no regularization.
        # rho : hyperparameters used in ADMM, the default setting guarantees convergence
        # tau : hyperparameters used in ADMM, the default setting guarantees convergence
        # W_upperbd : tuple for setting an upper bound of the entries in W.
        #             E.g. (True, 3.0) represents constraining entries in W to be within [0.0, 3.0]
        # Q_upperbd : Similar to W_upperbd
        # M_upperbd : Similar to W_upperbd, to constrain the reconstructed matrix WQ'.
        # max_iter : Maximum iteration number
        # confound : Normalized cofounder matrix. [Assume data_matrix is also Normalized]
        
        self.data_matrix = data_matrix
        self.data_mask = data_mask
        
        self.n_components = n_components
        self.regularizer = regularizer
        self.W_beta = W_beta
        self.Q_beta = Q_beta
        self.rho = rho
        self.tau = tau
        self.W_upperbd = W_upperbd
        self.Q_upperbd = Q_upperbd
        self.M_upperbd = M_upperbd
        self.max_iter = max_iter
        
        self.confound = confound
        
        
        self.M = self.data_matrix.copy()
        
        self.MF_data = matrix_class(self.M,
                                    None, # M_raw
                                    self.confound,
                                    None, # confound_raw
                                    self.data_mask,
                                    None, # row_idx
                                    None, # col_idx
                                    None, # mask
                                    None, # dataname
                                    None, # subjlist
                                    None, # itemlist
                                    None, # W
                                    None, # Q
                                    None, # C
                                    None, # Qc
                                    None, # Z
                                    None, # aZ
                                   )
         
        
        
        self.random_state = random_state
        
    def decomposition(self, verbose=True):
                
        clf = ICQF(self.n_components,
                    rho=self.rho,
                    tau=self.tau,
                    regularizer=self.regularizer,
                    W_upperbd=self.W_upperbd,
                    Q_upperbd=self.Q_upperbd,
                    M_upperbd=self.M_upperbd,
                    W_beta=self.W_beta,
                    Q_beta=self.Q_beta,
                    max_iter=self.max_iter)
        self.MF_data, loss_list = clf.fit_transform(self.MF_data)
        
        return self.MF_data

    
    def obj_func(self, MF_data, extra_mask):
        if MF_data.C is None:
            M_approx = MF_data.W@MF_data.Q.T
        else:
            M_approx = MF_data.W@MF_data.Q.T + MF_data.C@MF_data.Qc.T
        
        nentry = np.sum(extra_mask*MF_data.nan_mask)
        
        # normalized reconstruction error
        return np.sum(extra_mask*MF_data.nan_mask*(MF_data.M - M_approx) ** 2) / nentry
        
    def embed_holdout(self, mask_train, mask_valid):
        start = time.time()
        
        # temporary store nan_mask locally
        data_mask = self.data_mask.copy()
        
        # multiply nan_mask with training mask
        self.MF_data.nan_mask *= mask_train
        # perform decomposition
        MF_data = self.decomposition()
        
        # recovery nan_mask
        self.MF_data.nan_mask = data_mask
        
        train_error = self.obj_func(MF_data, mask_train)
        valid_error = self.obj_func(MF_data, mask_valid)

        embedding_stat = [self.n_components, self.W_beta, self.Q_beta, train_error, valid_error]
        
        end = time.time()
        return embedding_stat, MF_data
    

                
                
                
        
def main(args):
    
    data = np.load(args.filepath, allow_pickle=True)
    data_matrix = data['M']
    data_mask = data['nan_mask']
    confound = data['confound']
    if confound is not None:
        if confound.ndim == 0:
            confound = None
    
    if args.CV_mask != "":
        extra_mask = np.load(args.CV_mask)
        mask_train = extra_mask['mask_train']
        mask_valid = extra_mask['mask_valid']
    else:
        mask_train = np.ones_like(data_matrix)
        mask_valid = np.ones_like(data_matrix)
        
    if args.W_upperbd > 0.0:
        W_upperbd = (True, args.W_upperbd)
    else:
        W_upperbd = (False, 1.0)
        
    if args.Q_upperbd > 0.0:
        Q_upperbd = (True, args.Q_upperbd)
    else:
        Q_upperbd = (False, 1.0)
        
    if args.M_upperbd == -1:
        M_upperbd = (True, np.max(data_matrix))
    elif args.M_upperbd > 0.0:
        M_upperbd = (True, args.M_upperbd)
    elif args.M_upperbd == 0:
        M_upperbd = (False, 1.0)
    else:
        raise ValueError('Unknown upperbound for M')
    
    clf = MF_model(data_matrix,
                   data_mask,
                   args.n_components,
                   method=args.method,
                   W_beta=args.W_beta,
                   Q_beta=args.Q_beta,
                   regularizer=args.regularizer,
                   rho=args.rho,
                   tau=args.tau,
                   W_upperbd=W_upperbd,
                   Q_upperbd=Q_upperbd,
                   M_upperbd=M_upperbd,
                   confound=confound)
    
    embedding_stat, MF_data = clf.embed_holdout(mask_train, mask_valid)
    
    return embedding_stat, MF_data
    
        
        
def options(argv=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help="npz file with key 'M' and 'nan_mask'.")
    parser.add_argument('-o', '--output_filepath', type=str, required=True,
                        help="Resulting npy file with embedding stat.")
    
    parser.add_argument('--n_components', type=int, required=True,
                        help="embedding dimension.")
    parser.add_argument('--W_beta', type=float, required=True,
                        help="regularizer parameter for W.")
    parser.add_argument('--Q_beta', type=float, required=True,
                        help="regularizer parameter for Q.")
        
    parser.add_argument('--method', choices={'ICQF'}, default='ICQF', 
                        help="Choice of algorithm for factorization.")

    parser.add_argument('-l', '--regularizer', type=int, choices={1, 2}, default=1,
                        help="Types of regularizer, 1:sparsity, 2:smoothness.")
    parser.add_argument('--rho', type=float, default=3.0, 
                        help="Lagrangian multiplier's parameter.")
    parser.add_argument('--tau', type=float, default=3.0, 
                        help="time step for FISTA.")
    parser.add_argument('--W_upperbd', type=float, default=1.0, 
                        help="upper bounds for Left matrix W. No bound enforced when Wbound=0.")
    parser.add_argument('--Q_upperbd', type=float, default=0.0, 
                        help="upper bounds for Right matrix Q. No bound enforced when Qbound=0.")
    parser.add_argument('--M_upperbd', type=float, default=-1, 
                        help="upper bounds for data matrix M. No bound enforced when Mbound=0.")
    
    parser.add_argument('--CV_mask', type=str, default='',
                        help="extra npz filepath for training and validation mask (used for CV)")
    
    return parser.parse_args()
        
if __name__ == '__main__':
    
    args = options()
    embedding_stat, MF_data = main(args)
    np.save( args.output_filepath, embedding_stat)
    