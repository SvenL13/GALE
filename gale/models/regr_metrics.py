"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Additional Regression Metrics
"""
from typing import Literal, Optional, Union

import numpy as np
from sklearn.metrics._regression import _check_reg_targets, check_consistent_length

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D

__ALL__ = [
    "mean_relative_error",
]


def mean_relative_error(
    y_true: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D],
    y_pred: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D],
    y_min: Union[ARRAY_LIKE_1D, ARRAY_LIKE_2D],
    *,
    sample_weight: Optional[ARRAY_LIKE_1D] = None,
    multioutput: Union[
        Literal["uniform_average", "raw_values"], ARRAY_LIKE_1D
    ] = "uniform_average",
    squared: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate the mean relative error (MRE)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    y_min: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Minimum of target function.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    sample_weight : array-like of shape (n_samples,), optional (default=None)
        Sample weights.
    squared: bool, optional (default=False)
        output the squared MRE

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )

    check_consistent_length(y_true, y_pred)

    shift = y_min + 1
    output_errors = np.abs((y_pred - y_true) ** 2 / (y_true + shift))

    if not squared:
        output_errors = np.sqrt(output_errors)

    output_errors = np.average(output_errors, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)
