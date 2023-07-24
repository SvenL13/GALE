"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Gaussian Process Regression
"""
import numpy as np

__all__ = [
    "gradient_rbf",
    "gradient_matern15",
    "gradient_matern25",
    "gradient_exp",
    "gradient_rq",
]


def gradient_rbf(dists: np.ndarray) -> np.ndarray:
    """
    Gradient of the rbf (squared exponential) kernel

    Parameters
    ----------
    dists : np.ndarray
        A tensor of distances with shape (n_test_samp, n_train_samp, n_input_dims)

    Returns
    -------
    dK_dx_test_dot_k_x_test : np.ndarray
        The left side of the gradient multiplied with the covariance vector of
        test points with the same shape as dists.
    """
    K = np.exp(-0.5 * np.sum((dists**2), axis=-1, keepdims=True))  # shape = N1, N2, 1
    return -dists * K


def gradient_matern15(dists: np.ndarray) -> np.ndarray:
    """
    Gradient of the Matern1.5 kernel

    Parameters
    ----------
    dists : np.ndarray
        A tensor of distances with shape (n_test_samp, n_train_samp, n_input_dims)

    Returns
    -------
    dK_dx_test_dot_k_x_test : np.ndarray
        The left side of the gradient multiplied with the covariance vector of
        test points with the same shape as dists.
    """
    K = np.exp(
        -np.sqrt(3) * np.sqrt(np.sum(dists**2, axis=-1, keepdims=True))
    )  # shape = N1, N2, 1
    return -3.0 * dists * K


def gradient_matern25(dists: np.ndarray) -> np.ndarray:
    """
    Gradient of the Matern(nu=)2.5 kernel

    Parameters
    ----------
    dists : np.ndarray
        A tensor of distances with shape (n_test_samp, n_train_samp, n_input_dims)

    Returns
    -------
    dK_dx_test_dot_k_x_test : np.ndarray
        The left side of the gradient multiplied with the covariance vector of
        test points with the same shape as dists.
    """

    K = np.sqrt(5) * np.sqrt(np.sum(dists**2, axis=-1, keepdims=True))
    K = np.exp(-K) * (1 + K)  # shape = N1, N2, 1
    return -5.0 * dists * (K / 3.0)


def gradient_exp(dists: np.ndarray) -> np.ndarray:
    """
    Gradient of the exp (absolute exponential) kernel

    Parameters
    ----------
    dists : np.ndarray
        A tensor of distances with shape (n_test_samp, n_train_samp, n_input_dims)

    Returns
    -------
    dK_dx_test_dot_k_x_test : np.ndarray
        The left side of the gradient multiplied with the covariance vector of
        test points with the same shape as dists.
    """
    K = np.exp(np.sum(dists, axis=-1, keepdims=True))  # shape = N1, N2, 1
    return -np.sqrt(dists) * K


def gradient_rq(dists, alpha: float = 2.0) -> np.ndarray:
    """
    Gradient of the RationalQuadratic kernel

    Parameters
    ----------
    dists : np.ndarray
        A tensor of distances with shape (n_test_samp, n_train_samp, n_input_dims)

    Returns
    -------
    dK_dx_test_dot_k_x_test : np.ndarray
        The left side of the gradient multiplied with the covariance vector of
        test points with the same shape as dists.

    """
    coeff = 1.0 / (2 * alpha)
    K = 1 + coeff * np.sum(dists**2, axis=-1, keepdims=True)  # shape = N1, N2, 1
    return -dists * (K ** (-alpha - 1))
