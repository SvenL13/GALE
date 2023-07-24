"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Utilities
"""
import numpy as np
import itertools

from typing import Optional, Literal, Callable, Union, Tuple
from skopt.space import Real, Integer, Categorical
from skopt.sampler import Lhs

from gale._typing import BOUNDS, ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.doe.space import Space


def edge_sampling(bounds: BOUNDS, n_dim: int) -> np.ndarray:
    """
    Get the edges in design space (only implemented for real space)

    Returns
    -------
    X_edge: np.ndarray, shape=(n_samples, n_features)
        edges of design space
    """
    assert isinstance(n_dim, int) and n_dim > 0

    def recursion(bounds_: list, edges_: list, edge_: list):

        for b0 in bounds_[0]:
            edge_.append(b0)

            if len(bounds_) >= 2:  # go further

                new_bounds = bounds_.copy()
                new_bounds.pop(0)  # delete first bound

                edges_, edge = recursion(new_bounds, edges_, edge_)
                edge_.pop(-1)

            else:  # is leaf
                edges_.append(edge_.copy())
                edge_.pop(-1)

        return edges_, edge_

    if n_dim <= 8:  # use bit indexing for up to 8 dimensions
        bounds = np.array(bounds)
        n_edges = 2**n_dim

        # create index array
        ind_arr = np.atleast_2d(np.arange(n_edges, dtype=np.uint8)).T
        ind_arr = np.unpackbits(ind_arr, axis=1)[:, -n_dim:]

        # get edges
        rows = np.arange(bounds.shape[0])
        new_edges = bounds[rows, ind_arr]

    else:  # get edges from binary tree
        new_edges, _ = recursion(bounds, [], [])
        new_edges = np.array(new_edges)

    return new_edges


def sampling(
    bounds: BOUNDS,
    n_samples: int,
    fun: Optional[Callable[[ARRAY_LIKE_2D], ARRAY_LIKE_1D]] = None,
    method: Literal["lhs", "rnd"] = "lhs",
    lhs_crit: str = None,
    seed: int = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Create samples within bounds, additionally if fun is given then fun is directly sampled

    If fun is given, then fun is evaluated at the created samples

    Parameters
    ----------
    bounds: Bounds
        design bounds
    n_samples: int
        number of samples created from fun
    fun: function, optional (default=None)
        function should take an input X (n, n_features) and return y
    method: str, one of "lhs" or "rnd", optional (default="lhs")
        sampling method used
    lhs_crit: str or None, optional (default=None)
        When set to None, the LHS is not optimized, options: "correlation", "maximin" or "ratio"
    seed: int, optional (default=None)
        rnd seed

    Returns
    -------
    y_samples: np.ndarray, only if fun is given
        fun evaluated at X_samples
    X_samples: np.ndarray, shape=(n_samples, n_features)
        input samples created with sampling method
    """
    # create space
    space = Space(bounds)

    if method == "rnd":  # draw random samples
        X_samples = space.rvs(n_samples=n_samples, random_state=seed)

    elif method == "lhs":  # use latin hypercube sampling
        lhs = Lhs(criterion=lhs_crit, iterations=1000)
        X_samples = lhs.generate(space.bounds, n_samples, random_state=seed)

    else:
        raise ValueError("Unknown sampling method, should be one of: 'rnd', 'lhs'")

    X_samples = np.array(X_samples)

    if fun:
        # sample from fun
        y_samples = fun(X_samples)

        y_samples = np.array(y_samples)

        return y_samples, X_samples
    else:
        return X_samples
