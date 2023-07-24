"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
from typing import Union

import numpy as np
from scipy.stats import norm

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.models import SurrogateRegressionModel

from .adaptive_sampling import AdaptiveSampling


class EI(AdaptiveSampling):
    """
    Expected Improvement (EI)

    Literature
    ----------
    [1] Jones, Donald R., Matthias Schonlau, and William J. Welch. "Efficient global optimization of
        expensive black-box functions." Journal of Global optimization 13.4 (1998): 455-492.
    """

    # class information
    name = "Expected Improvement"
    short_name = "EI"
    optimizer_used = True

    def _acquisition_fun(
        self,
        x: ARRAY_LIKE_2D,
        model: SurrogateRegressionModel,
        y_opt: float = 0.0,
        xi: float = 0.01,
        minimize: bool = True,
    ) -> np.ndarray:
        """
        Use the expected improvement to calculate the acquisition values.

        The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
        mean and standard deviation approximated by the model.

        X : array-like, shape=(n_samples, n_features)
            Values where the acquisition function should be computed.

        model : sklearn estimator that implements predict with ``return_std``
            The fit estimator that approximates the function through the method ``predict``. It should have a
            ``return_std`` parameter that returns the standard deviation.

        y_opt : float, default 0
            Previous minimum value which we would like to improve upon.

        xi : float, default=0.01
            Controls how much improvement one wants over the previous best values. Useful only when ``method`` is set
            to "EI"

        Implementation based on skopt
        code: https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/acquisition.py#L232
        """
        # check input
        x = self._check_x(x)

        mu, std = model.predict(x, return_std=True)
        mu = np.array(mu).flatten()
        std = np.array(std).flatten()

        # for xi=0 -> Formula (15)
        values = np.zeros_like(mu)
        mask = std > 0

        improve = y_opt - xi - mu[mask]
        z = improve / std[mask]

        cdf = norm.cdf(z)
        pdf = norm.pdf(z)

        exploit = improve * cdf
        explore = std[mask] * pdf

        values[mask] = exploit + explore

        if minimize:
            return -values.flatten()
        else:
            return values.flatten()

    def _check_x(self, x: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D]) -> np.ndarray:
        """
        check input
        """
        if len(x.shape) == 1:  # single point
            x = x.reshape(1, -1)  # reshape to 2d (input, features)
        elif len(x.shape) > 2:
            raise ValueError("Array shape should be 2D")
        assert x.shape[1] == self._design_space.n_dims

        return x

    def _return_aq(self, x):
        """
        Return acquisition fun
        """
        x_min = np.amin(np.array(self.X_observed), axis=0)

        return -self._acquisition_fun(
            x, self.surr_model, y_opt=x_min, xi=0, minimize=False
        )

    def ask_(self):
        """
        Propose new point to sample from based on optimizing aq fun
        """
        x_min = np.amin(np.array(self.X_observed), axis=0)

        # optimize acquisition function and return next proposal point
        next_x = self._optimize(
            self._acquisition_fun,
            args=(self.surr_model, x_min, 0, True),
            optimizer=self.optimizer_name,
        )
        return next_x
