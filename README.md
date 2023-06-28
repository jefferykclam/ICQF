# ICQF_beta
closed beta version for ICQF package

### Getting start

##### Installation



### Quick Example

Given a data matrix `M`, we first construct the data class

```MF_data = matrix_class(M=M)``` 

and initialize the ICQF model:

```python
clf = ICQF(n_components=10,
           W_beta=0.1,
           Q_beta=0.1,
           W_upperbd=(True, 1.0),
           M_upperbd=(True, np.max(MF_data.M)),
           Q_upperbd=(True, 100),
           method='admm', max_iter=200, min_iter=10, tol=1e-4)
```

ICQF is performed by

```MF_data, loss = clf.fit_transform(MF_data)```

The matrix factorization results can be accessed via

```python
W = MF_data.W
Q = MF_data.Q
```



### ICQF

##### Main inputs

Consider a questionnaire data matrix $M \in \mathbb{R}_{\geq 0}^{n \times m}$ with $n$ participants and $m$ questions, where entry $(i,j)$ is the answer given by participant $i$ to question $j$. 

As questionnaires often have missing data, we also have a mask matrix  $\mathcal{M} \in \{0, 1\}^{n \times m}$ of the same dimensionality as $M$, indicating whether each entry is available $(=1)$ or not $(=0)$. 

We store the data matrix as a customized `matrix_class` with the following input attributes:

- `M_raw` : the input data matrix.
- `M` : the processed data matrix
- `nan_mask` : an indicator matrix representing the availability of entries (=1) and missing entries (=0)
- `confound_raw` : [optional] the input confound matrix 
- `confound` : [optional] the processed confound matrix, columns are either one-hot encoded / rescaled to $[0, 1]$
- `dataname` : dataset name
- `itemlist` : unique keys for item (column ID)
- `subjlist` : unique keys for subject (row ID)

Given a data matrix `M_raw` with indicator mask matrix `nan_mask` and confound matrix `confound_raw` , we can run to initialize the data class:

```python
MF_data = matrix_class(M_raw=M_raw, confound_raw=confound_raw, nan_mask=nan_mask)
```

##### Confounds

Optionally, we may have a confounder matrix  $C \in \mathbb{R}_{\geq 0}^{n \times c}$, encoding $c$ known variables for each participant that could account for correlations across questions (e.g. age or sex). 

- If the $j^{th}$ confound $C_{[:,j]}$ is categorical, we convert it to indicator columns for each value. 
- If it is continuous, we first rescale it into $[0, 1]$ (where 0 and 1 are the minimum and maximum in the dataset), and replace it with two new columns, $C_{[:,j]}$ and $1 - C_{[:,j]}$. This mirroring procedure ensures that both directions of the confounding variables are considered (e.g. answer patterns more common the younger or the older the participants are). 

Lastly, we incorporate a vector of ones into $C$ to facilitate intercept modeling of dataset wide answer patterns.

##### Factorization model

We seek to factorize the questionnaire matrix $M$ as the product of a $n \times k$ factor matrix $W \in [0,1]$, with the confound matrix $C \in [0,1]$ as optional additional columns, and a $m \times (k+c)$ loading matrix $Q := [{}^{R}\!Q, {}^{C}\!Q]$, with a loading pattern ${}^{R}\!Q$ over $m$ questions for each of the $k$ factors (and ${}^{C}\!Q$ for optional confounds). Our optimization problem minimizes the squared error of this factorization
$$
\underset{W \in \mathcal{W}, Q \in \mathcal{Q}, Z \in \mathcal{Z}}{\text{minimize}} \quad \quad 1/2 \left\Vert \mathcal{M}\odot (M - Z) \right\Vert_F^2 + \beta \cdot R(W, Q) 
$$
such that

- $ [W, C] Q^{T} = Z, \ \mathcal{Z} = \{ Z | \ \min(M) \leq Z_{ij} \leq \max(M)\}$
- $\mathcal{Q} = \{ Q | \ 0 \leq Q_{ij} \leq Q_{bd}\}$ 
- $\mathcal{W} = \{ W | \ 0 \leq W_{ij} \leq W_{bd} \}$

for some optional upper bounds $Q_{bd}, W_{bd}$ for $Q$ and $W$ respectively, where

$$R(W, Q) := \Vert W \Vert_{p, q} + \gamma \Vert Q \Vert_{p, q}$$

with $\gamma = \frac{n}{m}\max(M)$. The matrix norm is defined as $\Vert A \Vert_{p, q} := ( \sum^m_{i=1} ( \sum^n_{j=1} |A_{ij}|^p )^{q/p} )^{1/q}$. 

In this package, $p=q=\{1,2\}$ for the $L_1$ and $L_2$ regularization.



#### Parameters

- `n_components` : **int, default=2** latent dimension. To estimate, see `detect_dim` in **Methods** section.
- `method` : **{'admm', 'cd', 'hybrid'}, default='admm'**. Method used to optimize `W` and `Q` subproblems.
- `W_beta` : **float, default=0.0**. Strength of regularizer on `W`. Set it to zero to have no regularization.
- `Q_beta` : **float, default=0.0**. Strength of regularizer on `Q`. Set it to zero to have no regularization
- `regularizer` : **{1, 2}, default=1**. Type of regularizer. For L-1 (sparsity) choose `1`. For L-2 (smoothness), choose `2`.
- `rho` : **float, default=3.0**. Penalty parameter. The larger the `rho`, the faster it converges to local minimum. A smaller `rho` often gives a better local minimum. Theoretically, $\rho \geq \sqrt{2}$ to guarantee convergnence.
- `tau` : **float, default=3.0**. Penalty parameter for **`method`='admm'**.
- `W_upperbd` : **tuple, default=(True, 1.0)**. Used for constraining the upper bound of entries of `W`. If the first entry of the tuple is `True`, the optimized `W` is bounded above by the second entry of the tuple. Default upper bound for `W` is 1.0. If the first entry of the tuple is `False`, no upper bound on `W` is introduced and the second entry is ignored.
- `Q_upperbd` : **tuple, default=(False, 1.0)**. Used for constraining the upper bound of entries of `Q`. If the first entry of the tuple is `True`, the optimized `Q` is bounded above by the second entry of the tuple. If the first entry of the tuple is `False`, no upper bound on `Q` is introduced and the second entry is ignored.
- `M_upperbd` : **tuple, default=(True, 1.0)**. Used for constraining the upper bound of entries of `M`. If the first entry of the tuple is `True`, the optimized `M` is bounded above by the second entry of the tuple. If the first entry of the tuple is `False`, no upper bound on `M` is introduced and the second entry is ignored.
- `min_iter` : **int, default=10**. Minimum number of iteration..
- `max_iter` : **int, default=200**. Maximum number of itererations.
- `admm_iter` : **int, default=5**. Number of ADMM iterations if `method`='hybrid' is used. For more details on the 'hybrid' solver, see **Methods** section.
- `tol` : **float, default 1e-4**. Tolerance of the stopping criterion (relative error between successive iteration ).
- `verbose` : **int, default=0**. Whether to be verbose.

#### Attributes

- `n_components_` : **int**. Latent dimension. If it was given, it is the same as the n`_components`. otherwise, it will be the dimension detected using `detect_dim` method.
- `loss_history_` : **list**. Trend of the objective loss during the iteration.

#### Methods


#### Solvers:

Numerical solver to use for solving the subproblems. For `regularizer`=1, the `W` and `Q` subproblems become (constrained) LASSO problems. For `regularizer`=2, the `W` and `Q` subproblems become (constrained) quadratic programming.

- ‘admm’ : ADMM algorithm to solve the subproblems.
- ‘cd’ : coordinate descent to solve the subproblems.
- ‘hybrid’ : it utilizes a hybrid approach to address the subproblems. Initially, the solver operates similarly to the "admm" solver for the first 'admm_iter' iterations. Subsequently, the 'cd' algorithm is executed until the stopping criterion is met. This hybrid approach aims to prevent convergence to local minima while ensuring computational efficiency.

#### Supports



### Examples



### Reference


### Current updates
