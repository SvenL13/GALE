"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
from typing import Tuple

import numpy as np
from scipy.spatial import KDTree

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.models import SurrogateRegressionModel

from .adaptive_sampling import AdaptiveSampling


class EIGF(AdaptiveSampling):
    """
    Expected Improvement for Global Fit (EIGF)

    Literature
    ----------
    [1] Lam C (2008) - Sequential adaptive designs in computer experiments for response surface model fit,
        Ph.D. thesis, The Ohio State University.
    """

    # class information
    name = "Expected Improvement for Global Fit"
    short_name = "EIGF"
    optimizer_used = True

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
            model that approximates the function through the method "predict". It should have a
            "return_std" parameter that returns the standard deviation.

        Returns
        -------
        aq: np.ndarray, shape=(n_samples)
            negative acquisition value at given samples
        """
        X, X_obs, y_obs = self._check_aq_input(X, X_obs, y_obs)

        mu, std = model.predict(X, return_std=True)
        mu, std = (
            np.ascontiguousarray(mu).flatten(),
            np.ascontiguousarray(std).flatten(),
        )

        # find closest neighbor based on euclidean distance (Exploitation)
        kdt = KDTree(X_obs)
        _, index = kdt.query(X, workers=-1)

        # calculate acquisition
        aq = np.power(mu - y_obs[index], 2) + std**2

        return -aq.flatten()

    def _check_aq_input(
        self, X: ARRAY_LIKE_2D, X_obs: ARRAY_LIKE_2D, y_obs: ARRAY_LIKE_1D
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        X = np.ascontiguousarray(X, dtype=self.dtype)
        y_obs = np.ascontiguousarray(y_obs, dtype=self.dtype).ravel()
        X_obs = np.ascontiguousarray(X_obs, dtype=self.dtype)

        if len(X.shape) == 1:  # single point
            X = X.reshape(1, -1)  # reshape to 2d (input, features)
        elif len(X.shape) > 2:
            raise ValueError("Array shape should be 2D")

        if len(X_obs.shape) == 1:
            X_obs = X_obs.reshape(1, -1)
        elif len(X_obs.shape) > 2:
            raise ValueError("Array shape should be 2D")

        # check shapes
        assert len(X.shape) == len(X_obs.shape) == 2
        assert X.shape[1] == self._design_space.n_dims
        assert X_obs.shape[1] == self._design_space.n_dims
        assert y_obs.shape[0] == X_obs.shape[0]

        # return shapes x: (n_input, n_feat), X_obs: (n_input_obs, n_feat), y_obs: (n_input_obs)
        return X, X_obs, y_obs

    def _return_aq(self, x: ARRAY_LIKE_2D) -> np.ndarray:
        """
        eval acquisition fun at x
        """
        return -self._acquisition_fun(
            x, self.X_observed, self.y_observed, self.surr_model
        )

    def ask_(self):
        """
        Propose new point to sample from based on optimizing aq fun
        """
        # optimize acquisition function and return next proposal point
        next_x = self._optimize(
            self._acquisition_fun,
            args=(self.X_observed, self.y_observed, self.surr_model),
            optimizer=self.optimizer_name,
        )
        return next_x
