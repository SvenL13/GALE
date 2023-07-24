"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import numpy as np
from scipy.spatial.distance import cdist

from ...models import SurrogateRegressionModel
from .adaptive_sampling import AdaptiveSampling


class TEAD(AdaptiveSampling):
    """
    Taylor-Expansion Based Adaptive Design (TEAD)

    Literature
    ----------
    [1] Mo S., Lu D., Shi X., Zhang G., Ye M., Wu J., Wu J. (2017) - A Taylor
        expansion-based adaptive design strategy for global surrogate modeling
        with applications in groundwater modeling. Water Resources Res 53(12).
        Code: https://github.com/njujinchun/Codes-of-TEAD
    """

    # class information
    name = "Taylor-Expansion Based Adaptive Design"
    short_name = "TEAD"
    optimizer_used = False

    def _acquisition_fun(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        X_cand: np.ndarray,
        model: SurrogateRegressionModel,
    ) -> np.ndarray:
        """
        Calculate acquisition function
        """
        # calc distance between cand. points and obs. points based on euclidean distance
        d = cdist(X_obs, X_cand)
        index = np.argmin(d, axis=0).flatten().astype(int)
        # minimum distance between candidate point and observed points
        D_min = np.min(d, axis=0)

        # calculate exploration score
        J_exploration = D_min / np.max(D_min)

        # calc gradient (M_delta)
        y_grad_mu = model.predict_grad(X_obs)

        # compute first order taylor expansion
        delta_x = X_cand - X_obs[index]  # Equation (6), step size

        y_taylor_list = list()
        for i in range(X_cand.shape[0]):

            # prediction based on first-order taylor exp., -> 1D
            y_taylor_i = (
                y_obs[index[i]].flatten() + delta_x[i] @ y_grad_mu[index[i]]
            )  # Equation (6)

            y_taylor_list.append(y_taylor_i)

        y_taylor = np.array(y_taylor_list, dtype=self.dtype).flatten()

        # prediction at candidate points
        y_hat_cand = self.surr_model.predict(X_cand).flatten()

        # higher order remainders
        R = np.abs(y_hat_cand - y_taylor.flatten())  # Equation (7)

        # calculate exploitation score
        J_exploitation = R / np.max(R)

        # obtain weight
        L_max = np.max(
            cdist(X_obs, X_obs)
        )  # apprx. to maximum distance of any two points in design space
        weight = 1 - D_min / L_max  # Equation (9)

        # compute J
        aq = J_exploration + weight * J_exploitation  # Equation (8)

        return aq

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
            n_samples=5000 * self._design_space.n_dims,
            lhs_crit=None,
        )

        # evaluate acquisition fun for candidate points
        self.aq_cand = self._acquisition_fun(X_obs, y_obs, self.X_cand, self.surr_model)

        # find maximum
        max_cand_idx = np.argmax(self.aq_cand)

        # propose new point
        x_new = self.X_cand[max_cand_idx]

        return x_new
