"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
import os
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

from gale._typing import ARRAY_LIKE_2D, BOUNDS
from gale.models import SurrogateRegressionModel

from .adaptive_sampling import AdaptiveSampling


class MEPE(AdaptiveSampling):
    """
    Maximizing Expected Prediction Error (MEPE)

    Literature
    ----------
    [1] Liu Haitao, Cai Jianfei, Ong Yew-Soon - An adaptive sampling approach for
        Sklearn_Kriging metamodeling by maximizing expected prediction error.
        Computers and Chemical Engineering 106, 2017.
    """

    name = "Maximizing Expected Prediction Error"
    short_name = "MEPE"
    optimizer_used = True

    def __init__(
        self,
        bounds: BOUNDS,
        n_init: int = None,
        optimizer: Literal["pygmo", "lbfgs", "diff_evo"] = None,
        rnd_state: int = None,
        verbose: bool = False,
        model: Literal["GP", "BNN"] = "GP",
        **kwargs
    ):
        super(MEPE, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            optimizer=optimizer,
            model=model,
            **kwargs
        )

        # additional params
        self._alpha: float = 0.5
        self._eCV: Optional[np.ndarray] = None

        self._prev_model: Optional[SurrogateRegressionModel] = None
        self._path_prev_model: str = ""

    @staticmethod
    def _nearest_point(
        X: np.ndarray, model: SurrogateRegressionModel, e_CV: np.ndarray
    ) -> np.ndarray:
        """
        Utility function to find the leave-one-out error of the nearest point

        Parameters
        ----------
        X: np.ndarray, shape(n_input, n_features)
            Input value
        model: SurrogateRegressionModel
            trained model
        e_CV: np.ndarray, shape(n_input_observed)
            Array of leave-one-out errors for the observed points

        Returns
        -------
        e_cv_nearest_point: np.ndarray, shape(n_input)
            Leave-one-out error at position x, e_Loo
        """
        d = cdist(model.X_train_, X)
        index = np.argmin(d, axis=0).flatten().astype(int)

        e_cv_nearest_point = e_CV[index]  # formula 21

        return e_cv_nearest_point.flatten()

    @staticmethod
    def _update_weight(e_true: np.ndarray, e_CV: np.ndarray) -> float:
        """
        Adaptively update weights

        Parameters
        ----------
        e_true: np.ndarray
            true error
        e_CV: np.ndarray
            cross validation error

        Returns
        -------
        weight: float
            updated weight
        """
        val = 0.5 * ((e_true**2) / e_CV**2)
        weight = 0.99 * np.min([np.min(val), 1])  # formula 24
        return weight

    def _acquisition_fun(
        self,
        X: ARRAY_LIKE_2D,
        model: SurrogateRegressionModel,
        alpha: float,
        e_CV: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the acquisition function

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features)
            points where to evaluate acquisition function
        alpha: float
            weight between exploration and exploitation
        e_CV: arr_like_1d
            LOOCV error
        model: 'SurrogateRegressionModel'
            model that approximates the function through the method "predict". It should have a
            "return_std" parameter that returns the standard deviation.

        Returns
        -------
        aq: np.ndarray, shape=(n_samples)
            negative acquisition value at given samples
        """
        X_ = self._check_x(X)

        # get predicted std
        _, y_std = model.predict(X_, return_std=True)
        y_std = np.ascontiguousarray(y_std, dtype=self.dtype).flatten()

        e_cv_near = self._nearest_point(X_, model, e_CV)

        val = -(alpha * e_cv_near**2 + (1 - alpha) * y_std**2)

        return np.ascontiguousarray(val, dtype=self.dtype)

    def _check_x(self, X: ARRAY_LIKE_2D) -> np.ndarray:

        if len(X.shape) == 1:  # single point
            X = X.reshape(1, -1)  # reshape to 2d (input, features)
        elif len(X.shape) > 2:
            raise ValueError("Array shape should be 2D")
        assert X.shape[1] == self._design_space.n_dims

        return X

    def _return_aq(self, X: ARRAY_LIKE_2D) -> np.ndarray:
        """
        eval acquisition fun at x
        """
        aq_value = self._acquisition_fun(X, self.surr_model, self._alpha, self._eCV)
        return -aq_value

    def ask_(self):

        X_obs = np.ascontiguousarray(self.X_observed, dtype=self.dtype)
        y_obs = np.ascontiguousarray(self.y_observed, dtype=self.dtype)

        # LOOCV error
        self._eCV = self.surr_model.loo(X=None, y=None)

        # update weights
        if self._prev_model is not None or self._path_prev_model != "":

            if self.surrogate_name == "BNN":
                # load prev model from file
                prev_model = self.model_class()
                self._prev_model = prev_model.load(self._path_prev_model)

                # delete prev model
                os.remove(self._path_prev_model + ".dat")
                self._path_prev_model = ""

            # get prediction for last sample point
            y_pred = self._prev_model.predict(X_obs[-1, :].reshape(1, -1)).flatten()

            e_true = np.abs(y_pred[-1] - y_obs.flatten()[-1])
            self._alpha = self._update_weight(e_true, self._eCV[-1])

        # optimize acquisition function and return next proposal point
        next_x = self._optimize(
            self._acquisition_fun,
            args=(self.surr_model, self._alpha, self._eCV),
            optimizer=self.optimizer_name,
        )

        # update previous model
        if self.surrogate_name == "BNN":
            self._path_prev_model = "./mepe_prev_%s_model_%i_%i" % (
                self.surrogate_name,
                self.iter_count,
                int(np.random.uniform(0, 1000000)),
            )
            self._path_prev_model = os.path.abspath(self._path_prev_model)
            self.surr_model.save(self._path_prev_model)
        else:
            self._prev_model = deepcopy(self.surr_model)

        # track information
        self.tracking_i["alpha"] = self._alpha

        return next_x
