"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Surrogate base class
"""
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D


class SurrogateRegressionModel(ABC):
    """
    A regression model class meant for subclassing
    """

    model_name: str
    is_multioutput: bool

    def check_input(
        self,
        X: ARRAY_LIKE_2D = None,
        y: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D] = None,
        **check_params
    ):
        """
        Validate input data based on sklearn validation

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features), optional (default=None)
            input samples to check
        y: arr_like_1d, shape=(n_samples) or arr_like_2d, shape=(n_samples, n_outputs), optional (default=None)
            targets to check
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        if X is None and y is None:
            raise ValueError("Validation should be done on X, y or both.")
        if X is not None and y is None:
            X = check_array(X, estimator=self, **check_params)
            return X
        if X is None and y is not None:
            p = {"ensure_2d": self.is_multioutput}
            p.update(check_params)
            y = check_array(y, estimator=self, **p)
            return y
        else:
            p = {"y_numeric": True, "multi_output": self.is_multioutput}
            p.update(check_params)
            X, y = check_X_y(X, y, estimator=self, **p)
            return X, y

    @abstractmethod
    def fit(self, X: ARRAY_LIKE_2D, y: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D], **kwargs):
        """
        Placeholder for the surrogate model fit method

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features)
            input for training the surrogate model
        y: arr_like_1d, shape=(n_samples) or arr_like_2d, shape=(n_samples, n_outputs)

        Returns
        -------
        model: SurrogateRegressionModel
            fitted model
        """
        return self

    def update(self, X: ARRAY_LIKE_2D, y: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D]):
        """
        Condition model on new observations without refitting

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features)
            input for conditioning the surrogate model
        y: arr_like_1d, shape=(n_samples) or arr_like_2d, shape=(n_samples, n_outputs)
            targets for conditioning the surrogate model

        Returns
        -------
        model: SurrogateRegressionModel
            conditioned model
        """
        if hasattr(self, "_update"):
            return self._update(X, y)
        raise ValueError("Model doesn't implement update method.")

    @abstractmethod
    def predict(
        self,
        X: ARRAY_LIKE_2D,
        return_std: bool = False,
        return_cov: bool = False,
        return_grad: bool = False,
    ):
        """
        Placeholder for the surrogate model prediction method

        Parameters
        ----------
        X: arr_like_2d, shape=(n_samples, n_features)
            points to predict
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.
        return_grad : bool, default: False
            Whether or not to return the gradient of the mean.

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
            Optional for some models
        y_mean_grad : shape = (n_samples, n_features)
            The gradient of the predicted mean
        """
        pass

    @abstractmethod
    def predict_grad(self, X: ARRAY_LIKE_2D) -> np.ndarray:
        """
        Placeholder for the surrogate model predict_grad method.
        Should return gradient at X.

        Parameters
        ----------
        X: arr_like, shape=(n_samples, n_features)
            predict gradient of y for given X, dy/dx

        Returns
        -------
        y_mean_grad : shape = (n_samples, n_features)
            gradient of y, dy/dx
        """
        pass
