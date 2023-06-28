import numpy as np
import copy
import warnings
from sklearn.impute import SimpleImputer, KNNImputer

class matrix_class:
    
    def __init__(self,
                 M:np.ndarray, # (column)-normalized data matrix
                 M_raw:np.ndarray=None, # raw data matrix
                 confound:np.ndarray=None, # normalized confounder matrix
                 confound_raw:np.ndarray=None, # raw confounder matrix
                 nan_mask:np.ndarray=None, # mask matrix for missing entires (0=missing, 1=available)
                 row_idx:np.ndarray=None, # global row index (for multiple data matrices)
                 col_idx:np.ndarray=None, # global column index (for multiple data matrices)
                 mask:np.ndarray=None, # global mask (for multiple data matrices)
                 dataname:str=None, # dataname
                 subjlist:list=None, # information on subjects (row information)
                 itemlist:list=None, # information on items (column information)
                 W:np.ndarray=None, # subject embedding (recall M = [W, C]Q^T)
                 Q:np.ndarray=None, # item embedding (recall M = [W, C]Q^T)
                 C:np.ndarray=None, # confounder matrix
                 Qc:np.ndarray=None, # confounders' loadings (recall Q = [RQ, CQ])
                 Z:np.ndarray=None, # auxiliary Z=WQ^T (ADMM)
                 aZ:np.ndarray=None, # auxiliary variables (ADMM)
                 verbose=True
                ):
        
        self.M = M
        self.M_raw = M_raw
        self.confound = confound
        self.confound_raw = confound_raw
        self.nan_mask = nan_mask
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.mask = mask
        self.dataname = dataname
        self.subjlist = subjlist
        self.itemlist = itemlist
        self.W = W
        self.Q = Q
        self.C = C
        self.Qc = Qc
        self.Z = Z
        self.aZ = aZ
        
        self.verbose = verbose
        
    def vprint(self, str):
        if self.verbose==True:
            print(str)
        
    def check_input(self, verbose=True):
        
        if self.M_raw is None:
            self.vprint('No M_raw[matrix] provided')
            self.vprint('Make a copy of M[matrix] as M_raw[matrix]')
            self.M_raw = copy.deepcopy(self.M)
            
        if self.confound_raw is None:
            if self.confound is not None:
                self.vprint('No conofund_raw[matrix] provided')
                self.vprint('Make a copy of confound[matrix] as confound_raw[matrix]')
                self.confound_raw = copy.deepcopy(self.confound)
        else:
            if self.confound is None:
                self.vprint('Make a copy of confound_raw[matrix] as confound[matrix]')
                self.confound = copy.deepcopy(self.confound_raw)
        
        # check if M contains nan elements
        num_nan = np.sum(np.isnan(self.M))
        if num_nan > 0:
            warnings.warn('Input M[matrix] contains {} NaN elements'.format(num_nan))
            self.vprint('Map NaN entries to zero in M[matrix]')
            nan_mask = 1 - np.isnan(self.M)
            if self.nan_mask is None:
                self.M[nan_mask] = 0
                self.nan_mask = nan_mask
            else:
                assert self.nan_mask == nan_mask
                self.M[self.nan_mask] = 0
        else:
            if self.nan_mask is None:
                self.nan_mask = np.ones_like(self.M)

        # check if M contains negative entries
        M_max = np.max(self.M)
        M_min = np.min(self.M)
        if M_min < 0:
            warnings.warn('Input M[matrix] contains negative entries')
            self.vprint('Project negative entries onto 0')
            self.M[self.M < 0] = 0
        if M_max > 1:
            self.vprint('M[matrix] is not normalized, contains entries > 1')
            # self.vprint('Rescale the whole matrix by 1/max(M)')
            # self.M = self.M / M_max
            column_max = np.max(self.M, axis=1)
            self.vprint(f"Column-wise maximum of M are ranging {np.min(column_max):.3f} --  {np.max(column_max):.3f}")
            
        # check if confound contains nan elements
        
        if self.confound is not None:
            num_nan = np.sum(np.isnan(self.confound))
            if num_nan > 0:
                warnings.warn('Input confound[matrix] contains {} NaN elements'.format(num_nan))
                self.vprint('Impute NaN entries in confound')
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
                self.confound = imp_mean.fit_transform(self.confound)
            
            #check the range of confound and perform shifting if needed
            confound_max = np.max(self.confound)
            confound_min = np.min(self.confound)
            if confound_min < 0:
                warnings.warn('Input confound[matrix] contains negative elements')
                self.vprint('Shift confound[matrix] such that it is non-negative')
                self.confound = self.confound - confound_min
            
        # assert row_idx and col_idx length
        if self.row_idx is not None:
            if len(self.row_idx) != self.M.shape[0]:
                warnings.warn('row_idx[vector] size is different from row number of M[matrix]')
        if self.col_idx is not None:
            if len(self.col_idx) != self.M.shape[1]:
                warnings.warn('col_idx[vector] size is different from column number of M[matrix]')
                
        # check subjlist and itemlist length
        if self.subjlist is not None:
            if len(self.subjlist) != self.M.shape[0]:
                warnings.warn('Subject list[list] length is different from row number of M[matrix]')
            if len(self.itemlist) != self.M.shape[1]:
                warnings.warn('Item list[list] length is different from column number of M[matrix]')
    