import warnings

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy import stats

from sklearn.utils.extmath import svd_flip
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                      FLOAT_DTYPES)

BOUNDS_THRESHOLD = 1e-7


class QuantileTransformer(TransformerMixin, BaseEstimator):
    """Transform features using quantiles information.
    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.
    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.
    Read more in the :ref:`User Guide <preprocessing_transformer>`.
    .. versionadded:: 0.19
    Parameters
    ----------
    n_quantiles : int, optional (default=1000 or n_samples)
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.
    output_distribution : str, optional (default='uniform')
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.
    ignore_implicit_zeros : bool, optional (default=False)
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.
    subsample : int, optional (default=1e5)
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Note that this is used by subsampling and smoothing
        noise.
    copy : boolean, optional, (default=True)
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).
    Attributes
    ----------
    n_quantiles_ : integer
        The actual number of quantiles used to discretize the cumulative
        distribution function.
    quantiles_ : ndarray, shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.
    references_ : ndarray, shape(n_quantiles, )
        Quantiles of references.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X)
    array([...])
    See also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """

    def __init__(self, n_quantiles=1000, output_distribution='uniform',
                 ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn("'ignore_implicit_zeros' takes effect only with"
                          " sparse matrix. This parameter has no effect.")

        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(n_samples,
                                                    size=self.subsample,
                                                    replace=False)
                col = col.take(subsample_idx, mode='clip')
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.
        Parameters
        ----------
        X : sparse matrix CSC, shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative.
        """
        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for feature_idx in range(n_features):
            column_nnz_data = X.data[X.indptr[feature_idx]:
                                     X.indptr[feature_idx + 1]]
            if len(column_nnz_data) > self.subsample:
                column_subsample = (self.subsample * len(column_nnz_data) //
                                    n_samples)
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample,
                                           dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                column_data[:column_subsample] = random_state.choice(
                    column_nnz_data, size=column_subsample, replace=False)
            else:
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data),
                                           dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                column_data[:len(column_nnz_data)] = column_nnz_data

            if not column_data.size:
                # if no nnz, an error will be raised for computing the
                # quantiles. Force the quantiles to be zeros.
                self.quantiles_.append([0] * len(references))
            else:
                self.quantiles_.append(
                    np.nanpercentile(column_data, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # due to floating-point precision error in `np.nanpercentile`,
        # make sure the quantiles are monotonically increasing
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.
        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.
        Returns
        -------
        self : object
        """
        if self.n_quantiles <= 0:
            raise ValueError("Invalid value for 'n_quantiles': %d. "
                             "The number of quantiles must be at least one."
                             % self.n_quantiles)

        if self.subsample <= 0:
            raise ValueError("Invalid value for 'subsample': %d. "
                             "The number of subsamples must be at least one."
                             % self.subsample)

        if self.n_quantiles > self.subsample:
            raise ValueError("The number of quantiles cannot be greater than"
                             " the number of samples used. Got {} quantiles"
                             " and {} samples.".format(self.n_quantiles,
                                                       self.subsample))

        X = self._check_inputs(X, copy=False)
        n_samples = X.shape[0]

        if self.n_quantiles > n_samples:
            warnings.warn("n_quantiles (%s) is greater than the total number "
                          "of samples (%s). n_quantiles is set to "
                          "n_samples."
                          % (self.n_quantiles, n_samples))
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_,
                                       endpoint=True)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature"""

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if output_distribution == 'normal':
                lower_bounds_idx = (X_col - BOUNDS_THRESHOLD <
                                    lower_bound_x)
                upper_bounds_idx = (X_col + BOUNDS_THRESHOLD >
                                    upper_bound_x)
            if output_distribution == 'uniform':
                lower_bounds_idx = (X_col == lower_bound_x)
                upper_bounds_idx = (X_col == upper_bound_x)

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = .5 * (
                    np.interp(X_col_finite, quantiles, self.references_)
                    - np.interp(-X_col_finite, -quantiles[::-1],
                                -self.references_[::-1]))
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite,
                                             self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD -
                                                   np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col

    def _check_inputs(self, X, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform"""
        X = check_array(X, accept_sparse='csc', copy=copy,
                        dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if (not accept_sparse_negative and not self.ignore_implicit_zeros
                    and (sparse.issparse(X) and np.any(X.data < 0))):
                raise ValueError('QuantileTransformer only accepts'
                                 ' non-negative sparse matrices.')

        # check the output distribution
        if self.output_distribution not in ('normal', 'uniform'):
            raise ValueError("'output_distribution' has to be either 'normal'"
                             " or 'uniform'. Got '{}' instead.".format(
                self.output_distribution))

        return X

    def _check_is_fitted(self, X):
        """Check the inputs before transforming"""
        check_is_fitted(self, 'quantiles_')
        # check that the dimension of X are adequate with the fitted data
        if X.shape[1] != self.quantiles_.shape[1]:
            raise ValueError('X does not have the same number of features as'
                             ' the previously fitted data. Got {} instead of'
                             ' {}.'.format(X.shape[1],
                                           self.quantiles_.shape[1]))

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.
        inverse : bool, optional (default=False)
            If False, apply forward transform. If True, apply
            inverse transform.
        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            Projected data
        """

        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx],
                                     X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx],
                    inverse)
        else:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx],
                    inverse)

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.
        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.
        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X, copy=self.copy)
        self._check_is_fitted(X)

        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.
        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.
        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X, accept_sparse_negative=True, copy=self.copy)
        self._check_is_fitted(X)

        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {'allow_nan': True}


class KernelPCA(TransformerMixin, BaseEstimator):
    """Kernel Principal component analysis (KPCA)
    Non-linear dimensionality reduction through the use of kernels (see
    :ref:`metrics`).
    Read more in the :ref:`User Guide <kernel_PCA>`.
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.
    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".
    gamma : float, default=1/n_features
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.
    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
    alpha : int, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).
    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)
    eigen_solver : string ['auto'|'dense'|'arpack'], default='auto'
        Select eigensolver to use. If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.
    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
    max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.
    remove_zero_eig : boolean, default=False
        If True, then all components with zero eigenvalues are removed, so
        that the number of components in the output may be < n_components
        (and sometimes even zero due to numerical instability).
        When n_components is None, this parameter is ignored and components
        with zero eigenvalues are removed regardless.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``eigen_solver`` == 'arpack'.
        .. versionadded:: 0.18
    copy_X : boolean, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.
        .. versionadded:: 0.18
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.18
    Attributes
    ----------
    lambdas_ : array, (n_components,)
        Eigenvalues of the centered kernel matrix in decreasing order.
        If `n_components` and `remove_zero_eig` are not set,
        then all values are stored.
    alphas_ : array, (n_samples, n_components)
        Eigenvectors of the centered kernel matrix. If `n_components` and
        `remove_zero_eig` are not set, then all components are stored.
    dual_coef_ : array, (n_samples, n_features)
        Inverse transform matrix. Only available when
        ``fit_inverse_transform`` is True.
    X_transformed_fit_ : array, (n_samples, n_components)
        Projection of the fitted data on the kernel principal components.
        Only available when ``fit_inverse_transform`` is True.
    X_fit_ : (n_samples, n_features)
        The data used to fit the model. If `copy_X=False`, then `X_fit_` is
        a reference. This attribute is used for the calls to transform.
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import KernelPCA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = KernelPCA(n_components=7, kernel='linear')
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    References
    ----------
    Kernel PCA was introduced in:
        Bernhard Schoelkopf, Alexander J. Smola,
        and Klaus-Robert Mueller. 1999. Kernel principal
        component analysis. In Advances in kernel methods,
        MIT Press, Cambridge, MA, USA 327-352.
    """

    def __init__(self, n_components=None, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False,
                 random_state=None, copy_X=True, n_jobs=None):
        if fit_inverse_transform and kernel == 'precomputed':
            raise ValueError(
                "Cannot fit_inverse_transform with a precomputed kernel.")
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, n_jobs=self.n_jobs,
                                **params)

    def _fit_transform(self, K):
        """ Fit's using kernel K"""
        # center kernel
        K = self._centerer.fit_transform(K)

        if self.n_components is None:
            n_components = K.shape[0]
        else:
            n_components = min(K.shape[0], self.n_components)

        # compute eigenvectors
        if self.eigen_solver == 'auto':
            if K.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(
                K, eigvals=(K.shape[0] - n_components, K.shape[0] - 1))
        elif eigen_solver == 'arpack':
            random_state = check_random_state(self.random_state)
            # initialize with [-1,1] as in ARPACK
            v0 = random_state.uniform(-1, 1, K.shape[0])
            self.lambdas_, self.alphas_ = eigsh(K, n_components,
                                                which="LA",
                                                tol=self.tol,
                                                maxiter=self.max_iter,
                                                v0=v0)

        # make sure that the eigenvalues are ok and fix numerical issues
        self.lambdas_ = _check_psd_eigenvalues(self.lambdas_,
                                               enable_warnings=False)

        # flip eigenvectors' sign to enforce deterministic output
        self.alphas_, _ = svd_flip(self.alphas_,
                                   np.empty_like(self.alphas_).T)

        # sort eigenvectors in descending order
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        # remove eigenvectors with a zero eigenvalue (null space) if required
        if self.remove_zero_eig or self.n_components is None:
            self.alphas_ = self.alphas_[:, self.lambdas_ > 0]
            self.lambdas_ = self.lambdas_[self.lambdas_ > 0]

        # Maintenance note on Eigenvectors normalization
        # ----------------------------------------------
        # there is a link between
        # the eigenvectors of K=Phi(X)'Phi(X) and the ones of Phi(X)Phi(X)'
        # if v is an eigenvector of K
        #     then Phi(X)v  is an eigenvector of Phi(X)Phi(X)'
        # if u is an eigenvector of Phi(X)Phi(X)'
        #     then Phi(X)'u is an eigenvector of Phi(X)Phi(X)'
        #
        # At this stage our self.alphas_ (the v) have norm 1, we need to scale
        # them so that eigenvectors in kernel feature space (the u) have norm=1
        # instead
        #
        # We COULD scale them here:
        #       self.alphas_ = self.alphas_ / np.sqrt(self.lambdas_)
        #
        # But choose to perform that LATER when needed, in `fit()` and in
        # `transform()`.

        return K

    def _fit_inverse_transform(self, X_transformed, X):
        if hasattr(X, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for "
                                      "sparse matrices!")

        n_samples = X_transformed.shape[0]
        K = self._get_kernel(X_transformed)
        K.flat[::n_samples + 1] += self.alpha
        self.dual_coef_ = linalg.solve(K, X, sym_pos=True, overwrite_a=True)
        self.X_transformed_fit_ = X_transformed

    def fit(self, X, y=None):
        """Fit the model from data in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, accept_sparse='csr', copy=self.copy_X)
        self._centerer = KernelCenterer()
        K = self._get_kernel(X)
        self._fit_transform(K)

        if self.fit_inverse_transform:
            # no need to use the kernel to transform X, use shortcut expression
            X_transformed = self.alphas_ * np.sqrt(self.lambdas_)

            self._fit_inverse_transform(X_transformed, X)

        self.X_fit_ = X
        return self

    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X, **params)

        # no need to use the kernel to transform X, use shortcut expression
        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)

        if self.fit_inverse_transform:
            self._fit_inverse_transform(X_transformed, X)

        return X_transformed

    def transform(self, X):
        """Transform X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'X_fit_')

        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))

        # scale eigenvectors (properly account for null-space for dot product)
        non_zeros = np.flatnonzero(self.lambdas_)
        scaled_alphas = np.zeros_like(self.alphas_)
        scaled_alphas[:, non_zeros] = (self.alphas_[:, non_zeros]
                                       / np.sqrt(self.lambdas_[non_zeros]))

        # Project with a scalar product between K and the scaled eigenvectors
        return np.dot(K, scaled_alphas)

    def inverse_transform(self, X):
        """Transform X back to original space.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        if not self.fit_inverse_transform:
            raise NotFittedError("The fit_inverse_transform parameter was not"
                                 " set to True when instantiating and hence "
                                 "the inverse transform is not available.")

        K = self._get_kernel(X, self.X_transformed_fit_)

        return np.dot(K, self.dual_coef_)


def _check_psd_eigenvalues(lambdas, enable_warnings=False):
    """Check the eigenvalues of a positive semidefinite (PSD) matrix.

    Checks the provided array of PSD matrix eigenvalues for numerical or
    conditioning issues and returns a fixed validated version. This method
    should typically be used if the PSD matrix is user-provided (e.g. a
    Gram matrix) or computed using a user-provided dissimilarity metric
    (e.g. kernel function), or if the decomposition process uses approximation
    methods (randomized SVD, etc.).

    It checks for three things:

    - that there are no significant imaginary parts in eigenvalues (more than
      1e-5 times the maximum real part). If this check fails, it raises a
      ``ValueError``. Otherwise all non-significant imaginary parts that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    - that eigenvalues are not all negative. If this check fails, it raises a
      ``ValueError``

    - that there are no significant negative eigenvalues with absolute value
      more than 1e-10 (1e-6) and more than 1e-5 (5e-3) times the largest
      positive eigenvalue in double (simple) precision. If this check fails,
      it raises a ``ValueError``. Otherwise all negative eigenvalues that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    Finally, all the positive eigenvalues that are too small (with a value
    smaller than the maximum eigenvalue divided by 1e12) are set to zero.
    This operation is traced with a ``PositiveSpectrumWarning`` when
    ``enable_warnings=True``.

    Parameters
    ----------
    lambdas : array-like of shape (n_eigenvalues,)
        Array of eigenvalues to check / fix.

    enable_warnings : bool, default=False
        When this is set to ``True``, a ``PositiveSpectrumWarning`` will be
        raised when there are imaginary parts, negative eigenvalues, or
        extremely small non-zero eigenvalues. Otherwise no warning will be
        raised. In both cases, imaginary parts, negative eigenvalues, and
        extremely small non-zero eigenvalues will be set to zero.

    Returns
    -------
    lambdas_fixed : ndarray of shape (n_eigenvalues,)
        A fixed validated copy of the array of eigenvalues.

    Examples
    --------
    >>> _check_psd_eigenvalues([1, 2])      # nominal case
    array([1, 2])
    >>> _check_psd_eigenvalues([5, 5j])     # significant imag part
    Traceback (most recent call last):
        ...
    ValueError: There are significant imaginary parts in eigenvalues (1
        of the maximum real part). Either the matrix is not PSD, or there was
        an issue while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, 5e-5j])  # insignificant imag part
    array([5., 0.])
    >>> _check_psd_eigenvalues([-5, -1])    # all negative
    Traceback (most recent call last):
        ...
    ValueError: All eigenvalues are negative (maximum is -1). Either the
        matrix is not PSD, or there was an issue while computing the
        eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -1])     # significant negative
    Traceback (most recent call last):
        ...
    ValueError: There are significant negative eigenvalues (0.2 of the
        maximum positive). Either the matrix is not PSD, or there was an issue
        while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -5e-5])  # insignificant negative
    array([5., 0.])
    >>> _check_psd_eigenvalues([5, 4e-12])  # bad conditioning (too small)
    array([5., 0.])

    """

    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64

    # note: the minimum value available is
    #  - single-precision: np.finfo('float32').eps = 1.2e-07
    #  - double-precision: np.finfo('float64').eps = 2.2e-16

    # the various thresholds used for validation
    # we may wish to change the value according to precision.
    significant_imag_ratio = 1e-5
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6
    small_pos_ratio = 1e-12

    # Check that there are no significant imaginary parts
    if not np.isreal(lambdas).all():
        max_imag_abs = np.abs(np.imag(lambdas)).max()
        max_real_abs = np.abs(np.real(lambdas)).max()
        if max_imag_abs > significant_imag_ratio * max_real_abs:
            raise ValueError(
                "There are significant imaginary parts in eigenvalues (%g "
                "of the maximum real part). Either the matrix is not PSD, or "
                "there was an issue while computing the eigendecomposition "
                "of the matrix."
                % (max_imag_abs / max_real_abs))

        # warn about imaginary parts being removed
        if enable_warnings:
            warnings.warn("There are imaginary parts in eigenvalues (%g "
                          "of the maximum real part). Either the matrix is not"
                          " PSD, or there was an issue while computing the "
                          "eigendecomposition of the matrix. Only the real "
                          "parts will be kept."
                          % (max_imag_abs / max_real_abs),
                          PositiveSpectrumWarning)

    # Remove all imaginary parts (even if zero)
    lambdas = np.real(lambdas)

    # Check that there are no significant negative eigenvalues
    max_eig = lambdas.max()
    if max_eig < 0:
        raise ValueError("All eigenvalues are negative (maximum is %g). "
                         "Either the matrix is not PSD, or there was an "
                         "issue while computing the eigendecomposition of "
                         "the matrix." % max_eig)

    else:
        min_eig = lambdas.min()
        if (min_eig < -significant_neg_ratio * max_eig
                and min_eig < -significant_neg_value):
            raise ValueError("There are significant negative eigenvalues (%g"
                             " of the maximum positive). Either the matrix is "
                             "not PSD, or there was an issue while computing "
                             "the eigendecomposition of the matrix."
                             % (-min_eig / max_eig))
        elif min_eig < 0:
            # Remove all negative values and warn about it
            if enable_warnings:
                warnings.warn("There are negative eigenvalues (%g of the "
                              "maximum positive). Either the matrix is not "
                              "PSD, or there was an issue while computing the"
                              " eigendecomposition of the matrix. Negative "
                              "eigenvalues will be replaced with 0."
                              % (-min_eig / max_eig),
                              PositiveSpectrumWarning)
            lambdas[lambdas < 0] = 0

    # Check for conditioning (small positive non-zeros)
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_eig)
    if too_small_lambdas.any():
        if enable_warnings:
            warnings.warn("Badly conditioned PSD matrix spectrum: the largest "
                          "eigenvalue is more than %g times the smallest. "
                          "Small eigenvalues will be replaced with 0."
                          "" % (1 / small_pos_ratio),
                          PositiveSpectrumWarning)
        lambdas[too_small_lambdas] = 0

    return lambdas


class PositiveSpectrumWarning(UserWarning):
    """Warning raised when the eigenvalues of a PSD matrix have issues

    This warning is typically raised by ``_check_psd_eigenvalues`` when the
    eigenvalues of a positive semidefinite (PSD) matrix such as a gram matrix
    (kernel) present significant negative eigenvalues, or bad conditioning i.e.
    very small non-zero eigenvalues compared to the largest eigenvalue.

    .. versionadded:: 0.22
    """
