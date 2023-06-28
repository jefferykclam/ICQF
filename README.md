# ICQF_beta
closed beta version for ICQF package

### Getting start


### Quick Example



### ICQF

#### Factorization model



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

#### Attributes

#### Methods

#### Supports



### Examples



### Reference


### Current updates
