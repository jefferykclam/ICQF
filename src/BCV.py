import os
import numpy as np
import time
import argparse
import pickle
import glob
import json
import tqdm
from tqdm import trange

import scipy
import sklearn
from sklearn.model_selection import KFold

from datetime import datetime
import warnings
import sys


### General flow
# 1. read arguments
# 2a. load npz file, check data format
# 2b. create folders for masks, results, logs
# 3. create matrix masks for blockwise / random cross validation
# 4. create swarm files and generate commands for job submission

### Parser
parser = argparse.ArgumentParser()

# IO
parser.add_argument('-f', '--filepath', type=str, required=True,
                    help="npz file with key 'M' and 'nan_mask'.")
parser.add_argument('--output_dir', type=str, required=True,
                    help="Output directory containing masks, results, log.")
parser.add_argument('--custom_name', type=str, default='',
                    help='Adding description at the end of the output foldername')

# Algorithm parameters
parser.add_argument('-d', '--d_list', type=str, required=True,
                    help="embedding dimension list. Comma separates dimension for grid search.")
parser.add_argument('-b', '--beta_list', type=str, required=True,
                    help="regularizer parameter list for W. Comma separates beta for grid search.")
parser.add_argument('-s', '--beta_same', type=str, choices={'same','separate'}, default='same',
                    help="separate:detect optimal beta for Q separately. same:same beta with automatic weighting")
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
                    help="upper bounds for data matrix M. No bound enforced when Mbound=0. Default=-1, max(M) will be used.")


# CV parameters
parser.add_argument('--repeat', type=int, default=20,
                    help="numbers of repeat for each setting.")
parser.add_argument('--nrow', type=int, default=20,
                    help="split number (in row). Warning appears if #rows in M is not divisible by nrow. ")
parser.add_argument('--ncol', type=int, default=10,
                    help="split number (in column). Warning appears if #columns in M is not divisible by ncol.")
parser.add_argument('--mask_type', choices={'block', 'random'}, default="block",
                    help="held-out type: block or random.")
parser.add_argument('--nfold', type=int, default=10,
                    help="number of folds in CV. ")


### Define function for masks

def kfold_select(P, Krow, Kcol, shuffling=True):
    block_list = []
    if (Krow > 1) & (Kcol > 1):
        krowf = KFold(n_splits=Krow, shuffle=shuffling)
        kcolf = KFold(n_splits=Kcol, shuffle=shuffling)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
                uHO = np.zeros(P.shape[0])
                uHO[HOrow_idx] = 1 # row hold out
                vHO = np.zeros(P.shape[1])
                vHO[HOcol_idx] = 1 # column hold out
                block = [uHO, vHO]
                block_list.append(block)
    if (Krow > 1) & (Kcol == 1):
        krowf = KFold(n_splits=Krow, shuffle=shuffling)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            uHO = np.zeros(P.shape[0])
            uHO[HOrow_idx] = 1
            vHO = np.ones(P.shape[1])
            block = [uHO, vHO]
            block_list.append(block)
    if (Krow == 1) & (Kcol > 1):
        kcolf = KFold(n_splits=Kcol, shuffle=shuffling)
        for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
            uHO = np.ones(P.shape[0])
            vHO = np.zeros(P.shape[1])
            vHO[HOcol_idx] = 1
            block = [uHO, vHO]
            block_list.append(block)
    return block_list

def block_CV_mask(P, Krow, Kcol, J=10):
    block_list = kfold_select(P, Krow, Kcol)
    idx = np.arange(len(block_list))
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    if n % J != 0:
        warnings.warn('nblocks is not divisible!')
        
    np.random.shuffle(idx)
    
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        block_idx = idx[j*nc:(j+1)*nc]
        mask_valid = np.zeros_like(P)
        for k in block_idx:
            block_vectors = block_list[k]
            mask_valid += np.outer(block_vectors[0], block_vectors[1])
        mask_valid[mask_valid > 1] == 1
        mask_train = 1 - mask_valid
        mask_train_list.append(mask_train)
        mask_valid_list.append(mask_valid)
    return mask_train_list, mask_valid_list
    
def random_CV_mask(P, J=10):
    idx = np.arange(P.shape[0]*P.shape[1])
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    np.random.shuffle(idx)
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        mask_train = np.ones_like(P).reshape(-1)
        mask_train[idx[j*nc:(j+1)*nc]] = 0
        mask_valid = 1 - mask_train
        mask_train_list.append(mask_train.reshape(P.shape[0], P.shape[1]))
        mask_valid_list.append(mask_valid.reshape(P.shape[0], P.shape[1]))
    return mask_train_list, mask_valid_list



args = parser.parse_args()
config = args.__dict__

config['d_list'] = [int(d) for d in args.d_list.split(',')]
config['beta_list'] = [float(b) for b in args.beta_list.split(',')]

now = datetime.now()
time_string = now.strftime("%b-%d-%Y_%H%M%S-%f")
output_dir = config['output_dir'] + '-' + config['custom_name'] +'-'  + time_string

mask_dir = os.path.join(output_dir, 'mask')
result_dir = os.path.join(output_dir, 'result')
log_dir = os.path.join(output_dir, 'log')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

with open(os.path.join(output_dir, 'options.json'), 'wt') as option_file:
    json.dump(vars(args), option_file, indent=4)
    

### Create masks
data = np.load(args.filepath)

M = data['M']
nan_mask = data['nan_mask']
    
tqdm_range = trange(args.repeat, desc='Bar desc', leave=True)
for r in tqdm_range:
    np.random.seed()
    if args.mask_type == 'block':
        mask_train_list, mask_valid_list = block_CV_mask(M,
                                                         Krow=args.nrow,
                                                         Kcol=args.ncol,
                                                         J=args.nfold)
    if args.mask_type == 'random':
        mask_train_list, mask_valid_list = random_CV_mask(M,
                                                          J=args.nfold)
    
    for j in range(len(mask_train_list)):
        mask_filename = os.path.join(mask_dir, "./masks-{}-{}.npz".format(r, j))
        if not os.path.exists(mask_filename):
            np.savez(mask_filename, mask_train=mask_train_list[j], mask_valid=mask_valid_list[j])
            tqdm_range.set_description(mask_filename.split('/')[-1])
            tqdm_range.refresh()


### Create swarm

_nfolds = len(glob.glob(os.path.join(mask_dir, "*-0-*.npz")))
_nmasks = len(glob.glob(os.path.join(mask_dir, "*.npz")))
_repeat = int(_nmasks/_nfolds)

assert _repeat == args.repeat

swarm_filename = os.path.join(output_dir, "swarmfile" + '-' + time_string + ".swarm")
swarm_file = open(swarm_filename, "w")

for d in config['d_list']:
    for bW in config['beta_list']:
        if config['beta_same'] == 'same':
            bQ = bW
            for r in range(_repeat):
                for j in range(_nfolds):
                    mask_filename = os.path.join(mask_dir, "masks-{}-{}.npz".format(r, j))

                    filename = os.path.join(result_dir, "cv-d{}-bW{}-bQ{}-r{}-j{}.npy".format(d, bW, bQ, r, j))

                    MF_arguments  = ' --filepath {}'.format(args.filepath)
                    MF_arguments += ' --output_filepath {}'.format(filename)
                    MF_arguments += ' --n_components {}'.format(d)
                    MF_arguments += ' --W_beta {}'.format(bW)
                    MF_arguments += ' --Q_beta {}'.format(bQ)
                    MF_arguments += ' --regularizer {}'.format(args.regularizer)
                    MF_arguments += ' --rho {}'.format(args.rho)
                    MF_arguments += ' --tau {}'.format(args.tau)
                    MF_arguments += ' --W_upperbd {}'.format(args.W_upperbd)
                    MF_arguments += ' --Q_upperbd {}'.format(args.Q_upperbd)
                    MF_arguments += ' --M_upperbd {}'.format(args.M_upperbd)
                    MF_arguments += ' --CV_mask {}'.format(mask_filename)
                    MF_arguments += ' \n'

                    swarm_file.write("cd ./; python ./MF_algorithm.py" + MF_arguments)
                    
        elif config['beta_same'] == 'separate':
            for bQ in config['beta_list']:
                for r in range(_repeat):
                    for j in range(_nfolds):
                        mask_filename = os.path.join(mask_dir, "masks-{}-{}.npz".format(r, j))

                        filename = os.path.join(result_dir, "cv-d{}-bW{}-bQ{}-r{}-j{}.npy".format(d, bW, bQ, r, j))

                        MF_arguments  = ' --filepath {}'.format(args.filepath)
                        MF_arguments += ' --output_filepath {}'.format(filename)
                        MF_arguments += ' --n_components {}'.format(d)
                        MF_arguments += ' --W_beta {}'.format(bW)
                        MF_arguments += ' --Q_beta {}'.format(bQ)
                        MF_arguments += ' --regularizer {}'.format(args.regularizer)
                        MF_arguments += ' --rho {}'.format(args.rho)
                        MF_arguments += ' --tau {}'.format(args.tau)
                        MF_arguments += ' --W_upperbd {}'.format(args.W_upperbd)
                        MF_arguments += ' --Q_upperbd {}'.format(args.Q_upperbd)
                        MF_arguments += ' --M_upperbd {}'.format(args.M_upperbd)
                        MF_arguments += ' --CV_mask {}'.format(mask_filename)
                        MF_arguments += ' \n'

                        swarm_file.write("cd ./; python ./MF_algorithm.py" + MF_arguments)
                
swarm_file.close()


swarm_runcode = ' swarm -f '
swarm_runcode += ' ' + swarm_filename + ' '
swarm_runcode += ' --logdir ' + log_dir
swarm_runcode += ' --noht '

swarm_runcode += ' --partition quick '
swarm_runcode += ' --gb-per-process 8 '
swarm_runcode += ' --time 00:10:00 '
swarm_runcode += ' --gres=lscratch:10 '

swarm_runcode += ' -b 20'
 
print('=============Sample swarm command==============')

print(swarm_runcode)

print('=============Sample swarm command==============')

# os.system(swarm_runcode)