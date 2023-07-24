"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Sparse Variational Gaussian Process Regression
"""
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from GPy.core import SVGP
from GPy.kern import Matern32
from GPy.likelihoods import Gaussian

from gale._typing import ARRAY_LIKE_2D
from gale.models.surrogate import SurrogateRegressionModel


def get_inducing_points(
    X: np.ndarray,
    kernel,
    num_inducing: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    r"""
    A pivoted Cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. This method returns a greedy
    approximation of the MAP estimate of the specified DPP, i.e. its returns a
    set of points that are highly diverse (according to the provided kernel_matrix)
    and have high quality (according to the provided quality_scores).

    Adapted from BoTorch:
    https://github.com/pytorch/botorch/blob/e2caef4439f0a145eea463e13f7aed8206fcb0a6/
    botorch/models/utils/inducing_point_allocators.py#L264

    Args:
        X: training inputs (of shape n x d)
        kernel: kernel matrix on the training inputs
        num_inducing: number of inducing points to initialize
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """
    kernel_matrix = kernel.K(X)
    quality_scores = np.ones(X.shape[0], dtype=X.dtype)

    item_size = kernel_matrix.shape[-2]
    cis = np.zeros((num_inducing, item_size), dtype=kernel_matrix.dtype)
    di2s = np.diag(kernel_matrix)
    scores = di2s * (quality_scores**2)
    selected_items = []
    selected_item = np.argmax(scores)
    selected_items.append(selected_item)

    while len(selected_items) < num_inducing:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - np.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis**2
        di2s[selected_item] = -np.inf
        scores = di2s * (quality_scores**2)
        selected_item = np.argmax(scores)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = X[np.stack(selected_items)]

    return ind_points[:num_inducing, :]


class SparseGPRegressor(SurrogateRegressionModel):
    """
    Sparse variational Gaussian Process Regression

    References
    ----------
    James Hensman and Nicolo Fusi and Neil D. Lawrence, Gaussian Processes
    for Big Data, Proceedings of the 29th Conference on Uncertainty in
    Artificial Intelligence, 2013, https://arxiv.org/abs/1309.6835.
    """

    model_name = "Sklearn - GP Regression"
    is_multioutput = False

    def __init__(self, kernel=None, num_inducing: int = 256, normalize_y: bool = True):

        self.model: Optional[SVGP] = None
        if kernel is None:
            kernel = Matern32
        self.kernel = kernel
        self.num_inducing = max([10, int(num_inducing)])

        self.normalize_y: bool = normalize_y
        self._y_mu = 0.0
        self._y_std = 1.0
        self._y_std_pw = 1.0

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        self._y_mu = np.mean(y)
        self._y_std = np.std(y)
        self._y_std_pw = self._y_std**2
        return (y - self._y_mu) / self._y_std

    def _inv_normalize(
        self, y: np.ndarray, y_var: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not self.normalize_y:
            return y, y_var

        y = y * self._y_std + self._y_mu

        if y_var is not None:
            y_var = y_var * self._y_std_pw
        return y, y_var

    def predict(
        self,
        X: ARRAY_LIKE_2D,
        return_std: bool = False,
        return_cov: bool = False,
        return_grad: bool = False,
    ):
        y_mu, y_var = self.model.predict_noiseless(X)
        if return_std:
            y_mu, y_var = self._inv_normalize(y_mu.flatten(), y_var.flatten())
            return y_mu, np.sqrt(y_var)
        return self._inv_normalize(y_mu.flatten())[0]

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
        X = np.array(X)
        return self.model.predictive_gradients(X)[0][..., 0] * self._y_std

    def _update(self, X, y):
        """
        Condition model on new observations without refitting
        """
        if len(y.shape) == 1:
            y = y[:, None]
        if self.normalize_y:
            y = self._normalize(y)

        self.model.set_XY(X, y)

        return self

    def fit(self, X, y, filter_warning: bool = True):

        if len(y.shape) == 1:
            y = y[:, None]
        dim = X.shape[-1]

        if self.normalize_y:
            y = self._normalize(y)

        kernel = self.kernel(input_dim=dim, ARD=True)

        Z = get_inducing_points(X, kernel, self.num_inducing)

        likelihood = Gaussian(variance=1e-4)

        self.model = SVGP(X, y, Z=Z, kernel=kernel, likelihood=likelihood)

        self.model.kern.variance.constrain_fixed(1.0)
        self.model.Gaussian_noise.variance.constrain_fixed(1e-4)

        with warnings.catch_warnings():

            if filter_warning:
                warnings.simplefilter("ignore", category=RuntimeWarning)
            self.model.optimize_restarts(
                5, optimizer="lbfgsb", verbose=False, max_iters=2000, robust=True
            )

        return self

    def loo(
        self,
        X=None,
        y=None,
        method: str = "apprx",
        return_var: bool = False,
        squared: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        raise NotImplementedError("")

    def _loo_apprx(self, y) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("")
