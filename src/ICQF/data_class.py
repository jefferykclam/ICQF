import numpy as np
import copy
import warnings
from sklearn.impute import SimpleImputer, KNNImputer

class matrix_class:
    
    def __init__(self,
                 # data
                 M:np.ndarray, # (column)-rescaled data matrix
                 M_raw:np.ndarray=None, # raw data matrix
                 nan_mask:np.ndarray=None, # mask matrix for missing entires (0=missing, 1=available)

                 # meta info
                 dataname:str=None, # dataname
                 IDlist:list=None, # unique sample identification (row)
                 itemlist:list=None, # unique item abbreviation (column)
                 confoundlist:list=None, # unique confound abbreviation (column)

                 # factorization (M = [W, C][Q^T, Qc^T])
                 W:np.ndarray=None, # sample embedding
                 Q:np.ndarray=None, # item embedding
                 C:np.ndarray=None, # confound matrix
                 Qc:np.ndarray=None, # confound loadings

                 # internal optimization
                 Z:np.ndarray=None, # auxiliary (ADMM)
                 aZ:np.ndarray=None, # auxiliary variables (ADMM)

                 # misc
                 verbose=True
                ):
        
        self.M = M
        self.M_raw = M_raw
        self.nan_mask = nan_mask
        
        self.dataname = dataname
        self.IDlist = IDlist
        self.itemlist = itemlist
        self.confoundlist = confoundlist

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

    def plot_heatmap(self, cmap="Blues", nan_color="grey"):
        fig, ax = pyplot.subplots(figsize=(8,4))
        sns.heatmap(self.M, ax=ax, cmap="Blues", cbar=False)
        ax.set_xlabel('item')
        ax.set_ylabel('sample')
        ax.set_title(f"{self.dataname}")
        ax.set_facecolor(nan_color)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 1.5, top - 0.5)
        
        
        

def check_matrix(matrix):

    if matrix.M is None:
        raise ValueError('No M[matrix] provided')
    if matrix.M_raw is None:
        matrix.vprint('No M_raw[matrix] provided')
    if matrix.confoundlist is None:
        matrix.vprint('No confound[list] provided for C[matrix]')
        
    # check if M contains nan elements
    num_nan = np.sum(np.isnan(matrix.M))
    if num_nan > 0:
        matrix.vprint('[[[Warning]]] Input M[matrix] contains {} NaN elements'.format(num_nan))
        matrix.vprint('Map NaN entries to zero in M[matrix]')
        nan_mask = 1 - np.isnan(matrix.M)
        if matrix.nan_mask is None:
            matrix.M[np.isnan(matrix.M)] = 0
            matrix.nan_mask = nan_mask
        else:
            assert (matrix.nan_mask == nan_mask).all()
            matrix.M[~matrix.nan_mask.astype(dtype=bool)] = 0
    else:
        if matrix.nan_mask is None:
            matrix.nan_mask = np.ones_like(matrix.M)

    # check if M contains negative entries
    M_max = np.max(matrix.M)
    M_min = np.min(matrix.M)
    if M_min < 0:
        matrix.vprint('[[[Warning]]] Input M[matrix] contains negative entries')
        matrix.vprint('Project negative entries onto 0')
        matrix.M[matrix.M < 0] = 0
    if M_max > 1:
        matrix.vprint('M[matrix] is not scaled, contains entries > 1')
        # self.vprint('Rescale the whole matrix by 1/max(M)')
        # self.M = self.M / M_max
        column_max = np.max(matrix.M, axis=1)
        matrix.vprint(f"Column-wise maximum of M are ranging {np.min(column_max):.3f} --  {np.max(column_max):.3f}")

    # check if confound matrix C contains nan elements
    if matrix.C is not None:
        num_nan = np.sum(np.isnan(matrix.C))
        if num_nan > 0:
            matrix.vprint('[[[Warning]]] Input confound[matrix] C contains {} NaN elements'.format(num_nan))
            matrix.vprint('Impute NaN entries in confound')
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            matrix.C = imp_mean.fit_transform(matrix.C)

        #check the range of confound matrix C and perform shifting if needed
        C_max = np.max(matrix.C)
        C_min = np.min(matrix.C)
        if C_min < 0:
            warnings.warn('Input confound[matrix] C contains negative elements')
            matrix.vprint('Shift confound[matrix] C such that it is non-negative')
            matrix.C = matrix.C - C_min
   
    # check subjlist and itemlist length
    if matrix.IDlist is not None:
        if len(matrix.IDlist) != matrix.M.shape[0]:
            warnings.warn('Sample ID list[list] length is different from row number of M[matrix]')
        if len(matrix.itemlist) != matrix.M.shape[1]:
            warnings.warn('Item list[list] length is different from column number of M[matrix]')

