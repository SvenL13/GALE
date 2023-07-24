"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import numpy as np

from .ei import EI


class MMSE(EI):
    """
    Maximizing Mean Squared Error (MMSE)
    """

    # class information
    name = "Maximizing Mean Squared Error"
    short_name = "MMSE"
    optimizer_used = True

    def _acquisition_fun(
        self, x, model, y_opt=None, xi=None, minimize=None
    ) -> np.ndarray:
        """
        Return global model uncertainty at given values

        X : array-like, shape=(n_samples, n_features)
            Values where the acquisition function should be computed.

        model : sklearn estimator that implements predict with ``return_std``
            The fit estimator that approximates the function through the method
            ``predict``. It should have a ``return_std`` parameter that returns the
            standard deviation.

        y_opt : float, default 0
            Not used

        xi : float
            Not used

        Returns
        -------
        aq: np.ndarray, shape=(n_samples)
            negative acquisition value at given samples
        """
        _, std = model.predict(
            np.array(x, dtype=self.dtype).reshape(1, -1), return_std=True
        )
        var = np.array(std, dtype=self.dtype).flatten() ** 2

        return -var
