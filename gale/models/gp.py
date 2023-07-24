"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Gaussian Process Regression
"""
import warnings
from typing import Callable, Tuple, Union

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from skopt.learning import GaussianProcessRegressor

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.models.kernel_gradients import gradient_matern15, gradient_rbf
from gale.models.surrogate import SurrogateRegressionModel


def _gradient_gp(X: np.ndarray, gp_model: GaussianProcessRegressor) -> np.ndarray:
    """
    Calculate the gradient

    Parameters
    ----------
    X : np.ndarray
        Points at which to evaluate the gradient
    gp_model : GaussianProcessRegressor
        gp model

    Returns
    -------
    df_dx : np.ndarray
        The gradient of the GP wrt. x aka. Diagonal entries of its Jacobi-Matrix.
    """
    x_train = gp_model.X_train_
    X = np.array(X, dtype=np.float64)
    y_train_std = gp_model.y_train_std_

    scale_kern = gp_model.kernel_.k1
    prod_kern = gp_model.kernel_.k2

    assert isinstance(
        scale_kern, ConstantKernel
    ), "Only implemented if first kernel is ConstantKernel"
    assert isinstance(
        prod_kern, (Matern, RBF)
    ), "Only implemented if second kernel is Matérn or RBF Kernel"

    length_scale = prod_kern.length_scale.copy()
    if not np.iterable(length_scale) or len(length_scale) < X.shape[1]:
        length_scale = np.ones(X.shape[1]) * length_scale
    else:
        length_scale = np.array(length_scale).ravel()
    sigma2 = scale_kern.constant_value
    kern_str = str(gp_model.kernel)

    dists = (
        X[:, np.newaxis, :] / length_scale - x_train[np.newaxis, :, :] / length_scale
    )

    d_trans = sigma2 / length_scale
    if "RBF" in kern_str:
        K = gradient_rbf(dists)
    elif "Matern" in kern_str:
        nu = gp_model.kernel_.k2.nu

        if nu == 1.5:
            K = gradient_matern15(dists)
        else:
            raise NotImplementedError(
                f"Graident for kernel {kern_str} is not available"
            )
    else:
        raise NotImplementedError(f"Graident for kernel {kern_str} is not available")

    y_grad_mu = np.tensordot(
        d_trans * K * y_train_std, gp_model.alpha_, axes=((1), (0))
    ).reshape((X.shape))

    return y_grad_mu


class GPRegressor(GaussianProcessRegressor, SurrogateRegressionModel):
    """
    Add additional functionality to sklearn GP
    """

    model_name = "Sklearn - GP Regression"
    is_multioutput = False

    def predict(
        self,
        X,
        return_std=False,
        return_cov=False,
        return_mean_grad=False,
        return_std_grad=False,
    ):
        return super(GPRegressor, self).predict(X, return_std=return_std)

    def predict_grad(self, X: ARRAY_LIKE_2D):
        """
        The gradient of the mean with respect to X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        Returns
        -------
        y_mean_grad : shape = (n_samples, n_features)
            The gradient of the predicted mean
        """
        # y_mean_grad = gradient_fd(X, self.predict)
        return _gradient_gp(X, self)

    def _update(self, X, y):
        """
        Condition model on new observations without refitting
        """
        if self.kernel_.requires_vector_input:
            X, y = self._validate_data(
                X, y, multi_output=True, y_numeric=True, ensure_2d=True, dtype="numeric"
            )
        else:
            X, y = self._validate_data(
                X, y, multi_output=True, y_numeric=True, ensure_2d=False, dtype=None
            )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel, %s, is not returning a "
                "positive definite matrix. Try gradually "
                "increasing the 'alpha' parameter of your "
                "GaussianProcessRegressor estimator." % self.kernel_,
            ) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)

        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        self.K_inv_ = L_inv.dot(L_inv.T)

        return self

    def fit(self, X, y, filter_warning: bool = True):

        with warnings.catch_warnings():

            if filter_warning:
                warnings.simplefilter("ignore", category=ConvergenceWarning)

            return super(GPRegressor, self).fit(X, y)

    def loo(
        self,
        X=None,
        y=None,
        method: str = "apprx",
        return_var: bool = False,
        squared: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Calculate Leave one out Cross validation error (LOO), e_Loo

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features), optional (default=None)
            Query points where the loo error is evaluated, if None -> loo is returned for
            the trainings samples
        y: array-like, shape = (n_samples), optional (default=None)
            Query points where the loo error is evaluated, if None -> loo is returned for
            the trainings samples
        method: str, optional (default="apprx")
            method used to calculate loo error
        return_var: bool, optional (default=False)
            if True -> return predicted variance from model in loo iteration i
        squared: bool, optional (default=False)
            return the squared LOO error, e_loo^2

        Returns
        -------
        loo_error_mean: np.ndarray
            mean of loo_error for input data
        loo_error_var: np.ndarray, only if return_var=True
            predicted variance from model in loo iteration i
        """
        if X is None and y is None:
            y = self.y_train_
        elif X is not None and y is not None:
            y = y
        else:
            raise ValueError("X and y should be given.")

        if method == "apprx":
            loo_error, loo_error_var = self._loo_apprx(y)
        else:
            raise NotImplementedError("Method not implemented")

        if squared:
            loo_error = loo_error**2

        if return_var and loo_error_var is None:
            raise NotImplementedError("Variance is not implemented for given method.")
        elif return_var:
            return loo_error, loo_error_var
        else:
            return loo_error

    def _loo_apprx(self, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast approximation for the Leave one out Cross validation error (LOO)
        for gp with zero mean_fun

        Parameters
        ----------
        y: array-like, shape = (n_samples), optional (default=None)
            Query points where the loo error is evaluated

        Returns
        -------
        loo_error_mean: np.ndarray
            mean of loo_error for input data
        loo_error_var: np.ndarray
            predicted variance from model in loo iteration i

        Literature
        ----------
        [1] Sundararajan S, Keerthi S (2000) Predictive approaches for choosing
            hyperparameters in Gaussian processes. In: Advances in neural information
            processing systems, pp 631–637
        [2] Liu H, Cai J, Ong Y (2017) An adaptive sampling approach for kriging
            metamodeling by maximizing expected prediction error. Comput Chem Eng 106:171–182
        [3] Rasmussen, Williams - Gaussian Processes for Machine Learning, 2006
        """
        K_inv = self.K_inv_

        loo_error = list()
        loo_error_var = list()
        for i, _ in enumerate(y):

            loo_error_i = (K_inv[i, :] @ y.flatten()) / K_inv[
                i, i
            ]  # see formula (17) from [2] and (11) from [1]

            loo_error.append(np.abs(loo_error_i))
            loo_error_var.append(K_inv[i, i])  # predicted variance

        loo_error_mean = np.array(loo_error).flatten()

        loo_error_var = 1 / np.array(loo_error_var).flatten()

        return loo_error_mean, loo_error_var


def gradient_fd(
    X: np.ndarray,
    func: Callable[[np.ndarray], ARRAY_LIKE_1D],
    eps=1.4901161193847656e-08,
) -> np.ndarray:
    """
    Computes gradient with forward differences in an optimized
    fashion for n_samples < n_dim where X.shape = (n_samples, n_dim)

    Note that this function assumes a scalar prediction from the
    passed model

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, n_features)
         Point at witch to approximate the gradient. x.shape = (n_samples, n_dim)
    func : fun
        model with predict function, should take an input X and return y as array
    eps : float, optional
        Step size for the numerical approximation of the gradient.
        The default is 1.4901161193847656e-08.

    Returns
    -------
    gradient : np.ndarray, shape=(n_samples, n_features)
        The estimated gradient with the same shape as x.
    """
    X = np.array(X)
    if X.ndim < 2:
        X = X.reshape((1, -1))

    n_samp, n_dim = X.shape
    diffs = np.eye(n_dim) * eps
    diffs = np.append(np.zeros((1, n_dim)), diffs, axis=0)
    points = np.repeat(X, (n_dim + 1), axis=0) + np.tile(diffs, (n_samp, 1))

    # Order f to have each column represent predictions for a sample
    y_pred = np.array(func(points))
    y_pred = y_pred.reshape((n_dim + 1, -1), order="F")
    y_grad = (y_pred[1:, :] - y_pred[0, :]) / eps
    return y_grad.T
