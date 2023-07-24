"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from gale.models.utils import cook_model

from .adaptive_sampling import AdaptiveSampling


class MASA(AdaptiveSampling):
    """
    Mixed Adaptive Sampling Algorithm (MASA)

    Literature
    ----------
    [1] Eason J, Cremaschi S (2014) - Adaptive sequential sampling for surrogate model
        generation with artificial neural networks. Comput Chem Eng 68:220–232
    """

    # class information
    name = "Mixed Adaptive Sampling Algorithm"
    short_name = "MASA"
    optimizer_used = False

    def __init__(
        self, bounds, n_init=None, rnd_state=None, verbose=False, model="GP", **kwargs
    ):
        super(MASA, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            model=model,
            **kwargs
        )

        # additional param
        self._committee = self._create_committee()

    def _acquisition_fun(self, X_obs, X_cand, committee) -> np.ndarray:

        # prediction with default model
        y_pred: np.ndarray = self.surr_model.predict(X_cand)
        n_committee = len(committee)

        y_committee = list()
        # prediction with committee
        for memb in committee:
            y_committee.append(memb.predict(X_cand).flatten())
        y_committee = np.array(y_committee).T

        if y_pred.ndim == 1:
            y_pred = np.atleast_2d(y_pred).T

        y_tilde = (
            n_committee * np.tile(y_pred, len(committee))
            - (n_committee - 1) * y_committee
        )

        # avg committee
        y_committee_avg = np.mean(y_tilde, axis=1)
        y_committee_avg_stacked = np.tile(
            np.expand_dims(y_committee_avg, axis=1), len(committee)
        )

        # calc fluctuation based on committee
        fluc_score = (1 / (n_committee * (n_committee - 1))) * np.sum(
            (y_committee - y_committee_avg_stacked) ** 2, axis=1
        )

        # calc distance between cand. points and obs. points based on euclidean distance
        d = cdist(X_obs, X_cand)
        D_min = np.min(d, axis=0)

        # combine exploration and exploitation
        aq_cand = D_min / np.max(D_min) + fluc_score / np.max(fluc_score)

        return aq_cand

    def _create_committee(self):
        """
        Create committee of GP models based on different kernels

        Committee:
            Matern (nu=1.5), Matern (nu=2.5), RBF, DotProduct, RationalQuadratic
        """
        kernels = list()

        # create different kernels
        kernels.append(
            ConstantKernel(1, constant_value_bounds=(0.4, 10))
            * Matern(1, length_scale_bounds=(0.4, 10), nu=1.5)
        )
        kernels.append(Matern(1, length_scale_bounds=(0.4, 10), nu=2.5))
        kernels.append(RBF())
        # add small noise to make this numerically stable
        kernels.append(
            DotProduct() + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        )
        kernels.append(RationalQuadratic())

        committee = list()
        for kernel in kernels:  # create committee
            committee.append(
                cook_model("GP", gp_kernel=kernel, rnd_state=self._rnd_state)
            )
        return committee

    @staticmethod
    def _update_committee(committee, X_train, y_train):
        """
        Train committee members
        """
        for i, memb in enumerate(committee):
            memb = memb.fit(X_train, y_train)
            committee[i] = memb

        return committee

    def _return_aq(self, x):

        # sort candidates
        X_cand_sorted_idx = np.argsort(self.X_cand.flatten())

        X_cand_sorted = self.X_cand[X_cand_sorted_idx]
        aq_cand_sorted = self.aq_cand[X_cand_sorted_idx]

        return [X_cand_sorted, aq_cand_sorted]

    def ask_(self):

        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)

        # generate candidate points
        self.X_cand = self._gen_cand_points(
            "lhs",
            n_samples=5000 * self._design_space.n_dims,
            lhs_crit=None,
        )

        self._committee = self._update_committee(self._committee, X_obs, y_obs)

        self.aq_cand = self._acquisition_fun(X_obs, self.X_cand, self._committee)

        cand_ind = np.argmax(self.aq_cand)
        next_x = self.X_cand[cand_ind]

        return next_x
