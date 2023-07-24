"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - LHS
"""
import numpy as np
from scipy.spatial.distance import cdist

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.doe.utils import sampling
from gale.models import SurrogateRegressionModel

from .eigf import EIGF
from .._lhs import find_empty_bins, optimize_doe, inherit_lhs
from ...utils import split_bounds


class SLHS(EIGF):
    """
    Sequential Latin Hypercube Sampling Baseline (LHS)

    Sample sequentially from a set of candidate points based on the maximum distance
    (Euclidean).
    """

    name = "Latin Hypercube Sampling Baseline"
    short_name = "LHS"
    optimizer_used = False

    def __init__(
        self, bounds, n_init=None, rnd_state=None, verbose=False, model="GP", **kwargs
    ):
        super(SLHS, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            model=model,
            **kwargs
        )

        # use initial samples to create new optimized lhs
        X_lhs = self._lhs_sampling(self.X_init, self.n_max)
        self.X_init = X_lhs[: self.n_init, :]
        self.X_cand: np.ndarray = X_lhs[self.n_init:, :]

    def _lhs_sampling(self, X_init: np.ndarray, n_max: int) -> np.ndarray:

        if X_init.shape[0] > n_max:
            return X_init

        if X_init is not None:  # create optimized LHS based on old samples
            # split bounds
            lower_bounds, upper_bounds = split_bounds(self.bounds)

            # num. of new samples that have to be generated with LHS
            n_init = X_init.shape[0]
            n_new_samples = n_max - n_init

            empty_bins = self._get_empty_bins(
                n_new_samples,
                X_init,
                lower_bounds,
                upper_bounds,
            )

            # create new samples
            X_new = inherit_lhs(n_new_samples, empty_bins, lower_bounds, upper_bounds)

            # shift new samples such that they create with the old samples the new
            # optimized LHS
            X_new_opt = optimize_doe(X_new, doe_old=X_init)

            # optimized LHS samples
            X_lhs = np.append(X_init, X_new_opt, axis=0)

        else:
            X_lhs = sampling(
                self._design_space.bounds,
                n_max,
                lhs_crit="maximin",
            )

        return X_lhs

    @staticmethod
    def _get_empty_bins(n_new, old_doe, lower_bounds, upper_bounds):

        n_bins = n_new
        n_empty = 0
        empty_bins = None

        while n_empty < n_new:
            empty_bins = find_empty_bins(old_doe, n_bins, lower_bounds, upper_bounds)
            n_empty = np.max(empty_bins.sum(0))
            n_bins += 1

        return empty_bins

    def _acquisition_fun(
        self,
        X: ARRAY_LIKE_2D,
        X_obs: ARRAY_LIKE_2D,
        y_obs: ARRAY_LIKE_1D,
        model: SurrogateRegressionModel,
    ) -> np.ndarray:
        X, X_obs, y_obs = self._check_aq_input(X, X_obs, y_obs)

        dist = np.min(cdist(X_obs, X), axis=0)

        return np.array(dist, dtype=self.dtype).flatten()

    def _return_aq(self, x):
        return [self.X_cand, self.aq_cand]

    def ask_(self):

        X_obs = np.array(self.X_observed, dtype=self.dtype)
        y_obs = np.array(self.y_observed, dtype=self.dtype)

        if self.X_cand is None:
            raise ValueError("Number of asked points exceeds n_max.")

        # evaluate acquisition fun for candidate points
        self.aq_cand = self._acquisition_fun(self.X_cand, X_obs, y_obs, self.surr_model)

        # find maximum
        next_x_index = np.argmax(self.aq_cand)

        # propose new point
        x_new = self.X_cand[next_x_index]

        # del used x_next from X_cand
        if len(self.X_cand) > 1:
            self.X_cand = np.delete(self.X_cand, next_x_index, axis=0)
        else:
            self.X_cand = None

        return x_new
