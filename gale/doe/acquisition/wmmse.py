"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import numpy as np
from scipy.spatial.distance import cdist

from .adaptive_sampling import AdaptiveSampling


class wMMSE(AdaptiveSampling):
    """
    Weighted Maximum Mean Squared Error (wMMSE)

    Literature
    ----------
    [1] Kyprioti A., Zhang J. and Taflanidis A. - Adaptive design of experiments for
        global Kriging metamodeling through cross-validation information. Springer
        Structural and Multidisciplinary Optimization 62. 2020.
    """

    # class information
    name = "Weighted Maximum Mean Squared Error"
    short_name = "wMMSE"
    optimizer_used = False

    def __init__(
        self,
        bounds,
        n_init=None,
        optimizer=None,
        rnd_state=None,
        verbose=False,
        model="GP",
        **kwargs
    ):
        super(wMMSE, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            optimizer=optimizer,
            model=model,
            **kwargs
        )

        # additional params
        self._rho: float = 1  # tradeoff between exploration and exploitation
        self._eCV = 0
        self._phi = None

    def _nearest_point(self, X, X_obs, e_CV):
        """
        Utility function to find the leave-one-out error of the nearest point

        Parameters
        ----------
        X: float or arr_like
            Input value
        X_obs: arr_like
            observed points
        e_CV: arr_like
            Array of leave-one-out errors

        Returns
        -------
        e_CV_nearestPoint:
            Leave-one-out error at position x
        """
        d = cdist(X_obs, X)
        index = np.argmin(d, axis=0).flatten().astype(int)

        # choose e_CV from closest experiment x, corresponding to the natural neighbor
        e_CV_nearestPoint = e_CV[index] ** 2

        return np.array(e_CV_nearestPoint, dtype=self.dtype).flatten()

    def _acquisition_fun(self, X, X_obs, e_CV) -> np.ndarray:

        # get var from model prediction
        _, x_std = self.surr_model.predict(X, return_std=True)
        var_norm = x_std**2

        # calc phi
        self._phi = self._nearest_point(X, X_obs, e_CV)

        assert len(self._phi) == len(var_norm)
        # calc acquisition fun
        val = self._phi**self._rho * var_norm

        return np.array(val, dtype=self.dtype)

    def _return_aq(self, x):

        return [self.X_cand, self.aq_cand]

    def ask_(self):

        X_obs = np.array(self.X_observed, dtype=self.dtype)

        self._eCV = self.surr_model.loo(X=None, y=None, return_var=False)

        self.X_cand = self._gen_cand_points(
            "lhs",
            n_samples=5000 * self._design_space.n_dims,
            lhs_crit=None,
        )

        self.aq_cand = self._acquisition_fun(self.X_cand, X_obs=X_obs, e_CV=self._eCV)

        # find next proposal point
        next_index = np.argmax(self.aq_cand)
        next_x = self.X_cand[next_index]

        # track information
        self.tracking_i["alpha"] = self._phi

        return next_x
