"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
from typing import List, Literal

import numpy as np
from scipy.spatial.distance import cdist

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D, BOUNDS
from gale.models import SurrogateRegressionModel

from .adaptive_sampling import AdaptiveSampling


class GGESS(AdaptiveSampling):
    """
    Gradient and Geometry Enhanced Sequential Sampling (GGESS)

    Literature
    ----------
    [1] X. Chen, Y. Zhang, W. Zhou, W. Yao - An effective gradient and geometry enhanced
        sequential sampling approach for Kriging modeling. Structural and Multidisciplinary
        Optimization, 64, 2021.
    """

    # class information
    name: str = "Gradient and Geometry Enhanced Sequential Sampling"
    short_name: str = "GGESS"
    optimizer_used: bool = False

    def __init__(
        self,
        bounds: BOUNDS,
        n_init: int = None,
        rnd_state: int = None,
        verbose: bool = False,
        model: Literal["GP", "BNN"] = "GP",
        alpha: float = 1,
        **kwargs
    ):

        super(GGESS, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            model=model,
            **kwargs
        )

        # reduction of candidate points, if alpha=1 -> all candidate points are used
        if 0 < alpha <= 1:
            self._alpha = alpha
        else:
            raise ValueError("alpha out of bounds.")

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
            acquisition value at given samples
        """
        # predict mean and variance
        mu, std = model.predict(X, return_std=True)
        mu = mu.flatten()
        var = std.flatten() ** 2  # Exploration term

        # calc approx gradient
        y_grad_mu = model.predict_grad(X)  # y_dot with shape (n_samples, n_features)

        # find closest point based on euclidean distance (get Voronoi subdomain)
        d = cdist(X_obs, X)
        index = np.argmin(d, axis=0).flatten().astype(int)

        delta_x = X_obs[index] - X

        y_taylor = np.zeros(X.shape[0], dtype=self.dtype)
        for i in range(X.shape[0]):
            y_taylor[i] = (mu[i].flatten() - delta_x[i] @ y_grad_mu[index[i]]).item()

        aq_values = np.power(y_obs[index].flatten() - y_taylor, 2) + var

        return aq_values

    def _return_aq(self, x) -> List[np.ndarray]:
        """
        Return acquisition fun
        """
        return [self.X_cand, self.aq_cand]

    def ask_(self) -> np.ndarray:
        """
        Propose new point to sample from based on optimizing aq fun
        """
        X_obs = np.array(self.X_observed, dtype=self.dtype)
        y_obs = np.array(self.y_observed, dtype=self.dtype)

        # step 3: generate candidate points
        self.X_cand = self._gen_cand_points(
            "lhs",
            n_samples=5000 * self._design_space.n_dims,
            lhs_crit=None,
        )

        # step 4: partition design domain into voronoi subdomains (not implemented)
        if self._alpha < 1:
            raise NotImplementedError()

        # step 5 & 6: Compute the approximate gradients and improvement function
        self.aq_cand = self._acquisition_fun(self.X_cand, X_obs, y_obs, self.surr_model)

        # step 6: Maximize the improvement function
        max_cand_idx = np.argmax(self.aq_cand)
        next_x = self.X_cand[max_cand_idx]

        return next_x
