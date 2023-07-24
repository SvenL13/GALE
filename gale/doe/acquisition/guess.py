"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import numpy as np
from scipy.spatial.distance import cdist

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.models import SurrogateRegressionModel

from .eigf import EIGF


class GUESS(EIGF):
    """
    Gradient and Uncertainty Enhanced Sequential Sampling (GUESS)

    Literature
    ----------
    [1] S. Lämmle, C. Bogoclu, K. Cremanns, D. Roos - Gradient and Uncertainty enhanced
    Sequential Sampling for Global Fit. Computer Methods in applied Mechanics and
    Engineering, Volume 415, 2023. https://doi.org/10.1016/j.cma.2023.116226
    """

    # class information
    name = "Gradient and Uncertainty Enhanced Sequential Sampling"
    short_name = "GUESS"
    optimizer_used = False

    def _acquisition_fun(
        self,
        X: ARRAY_LIKE_2D,
        X_obs: ARRAY_LIKE_2D,
        y_obs: ARRAY_LIKE_1D,
        model: SurrogateRegressionModel,
    ) -> np.ndarray:
        """
        Evaluate the acquisition function

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features)
            points where to evaluate acquisition function
        X_obs: arr_like_2d
            observed points
        y_obs: arr_like_1d
            true response at observed points
        model: 'SurrogateRegressionModel'
            model that approximates the function through the method "predict". It should
            have a "return_std" parameter that returns the standard deviation.

        Returns
        -------
        aq: np.ndarray, shape=(n_samples)
            acquisition value at given samples
        """
        X, X_obs, y_obs = self._check_aq_input(X, X_obs, y_obs)

        y_mu, y_std = model.predict(X, return_std=True)
        y_mu = y_mu.flatten()
        y_std = y_std.flatten()
        y_grad_mu = model.predict_grad(X_obs)

        d = cdist(X_obs, X)
        index = np.argmin(d, axis=0).flatten().astype(int)

        # compute first order taylor expansion
        delta_x = X - X_obs[index]

        y_taylor = np.zeros(delta_x.shape[0], dtype=self.dtype)
        for i in range(X.shape[0]):
            # prediction based on first-order taylor exp., -> 1D
            y_taylor[i] = (
                y_obs[index[i]].flatten() + delta_x[i] @ y_grad_mu[index[i]]
            ).item()

        # calculate acquisition function (Exploitation + Exploration)
        aq = np.abs(y_mu - y_taylor) * y_std + y_std

        return np.array(aq, dtype=self.dtype).flatten()

    def _return_aq(self, x):
        """
        Return acquisition fun
        """
        return [self.X_cand, self.aq_cand]

    def ask_(self):
        """
        Propose new point to sample from based on optimizing aq fun
        """
        X_obs = np.array(self.X_observed, dtype=self.dtype)
        y_obs = np.array(self.y_observed, dtype=self.dtype)

        # generate candidate points
        self.X_cand = self._gen_cand_points(
            "lhs",
            n_samples=min([80000, 5000 * self._design_space.n_dims]),
            lhs_crit=None,
        )

        # evaluate acquisition fun for candidate points
        self.aq_cand = self._acquisition_fun(self.X_cand, X_obs, y_obs, self.surr_model)

        # find maximum
        max_cand_idx = np.argmax(self.aq_cand)

        # propose new point
        x_new = self.X_cand[max_cand_idx]

        return x_new
