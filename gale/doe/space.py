"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Modified version of skopts space class
"""
import numbers

import numpy as np
from skopt.space import Categorical, Dimension, Integer, Real
from skopt.space import Space as SK_Space

from gale.utils import ensure_2d


def _transpose_list_array(x):
    """Transposes a list matrix"""
    n_dims = len(x)
    assert n_dims > 0
    n_samples = len(x[0])
    rows = [None] * n_samples
    for i in range(n_samples):
        r = [None] * n_dims
        for j in range(n_dims):
            r[j] = x[j][i]
        rows[i] = r
    return rows


class Space(SK_Space):
    def __init__(self, dimensions):
        self.dimensions = [check_dimension(dim) for dim in dimensions]

    def transform(self, X) -> np.ndarray:
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        X : list of lists, shape=(n_samples, n_dims) or shape=(n_dims)
            The samples to transform.

        Returns
        -------
        Xt : array of floats, shape=(n_samples, transformed_n_dims)
            The transformed samples.
        """
        X = ensure_2d(X)

        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        Xt : array of floats, shape=(n_samples, transformed_n_dims) or shape=(transformed_n_dims)
            The samples to inverse transform.

        Returns
        -------
        X : list of lists, shape=(n_samples, n_dims)
            The original samples.
        """
        Xt = ensure_2d(Xt)

        # Inverse transform
        columns = []
        start = 0
        Xt = np.asarray(Xt)
        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(dim.inverse_transform(Xt[:, start : start + offset]))

            start += offset

        # Transpose
        return _transpose_list_array(columns)


def check_dimension(dimension, transform=None):
    """Turn a provided dimension description into a dimension object.
    Checks that the provided dimension falls into one of the
    supported types. For a list of supported types, look at
    the documentation of ``dimension`` below.
    If ``dimension`` is already a ``Dimension`` instance, return it.
    Parameters
    ----------
    dimension : Dimension
        Search space Dimension.
        Each search dimension can be defined either as
        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).
    transform : "identity", "normalize", "string", "label", "onehot" optional
        - For `Categorical` dimensions, the following transformations are
          supported.
          - "onehot" (default) one-hot transformation of the original space.
          - "label" integer transformation of the original space
          - "string" string transformation of the original space.
          - "identity" same as the original space.
        - For `Real` and `Integer` dimensions, the following transformations
          are supported.
          - "identity", (default) the transformed space is the same as the
            original space.
          - "normalize", the transformed space is scaled to be between 0 and 1.
    Returns
    -------
    dimension : Dimension
        Dimension instance.
    """
    if isinstance(dimension, Dimension):
        return dimension

    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple.")

    # A `Dimension` described by a single value is assumed to be
    # a `Categorical` dimension. This can be used in `BayesSearchCV`
    # to define subspaces that fix one value, e.g. to choose the
    # model type, see "sklearn-gridsearchcv-replacement.py"
    # for examples.

    if len(dimension) == 1:
        return Categorical(dimension, transform=transform)

    if len(dimension) == 2:  # alternative specification
        if (
            dimension[0] == "disc"
            or dimension[0] == "discrete"
            or dimension[0] == "categorical"
        ):
            return Categorical(dimension[1], transform=transform)

        if dimension[0] == "bounds":
            if all([isinstance(dim, numbers.Integral) for dim in dimension[1]]):
                return Integer(*dimension[1], transform=transform)
            elif any([isinstance(dim, numbers.Real) for dim in dimension[1]]):
                return Real(*dimension[1], transform=transform)
            else:
                ValueError(
                    "Invalid dimension {}. Read the documentation for"
                    " supported types.".format(dimension)
                )

    if len(dimension) == 2:
        if any(
            [isinstance(d, (str, bool)) or isinstance(d, np.bool_) for d in dimension]
        ):
            return Categorical(dimension, transform=transform)
        elif all([isinstance(dim, numbers.Integral) for dim in dimension]):
            return Integer(*dimension, transform=transform)
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            return Real(*dimension, transform=transform)
        else:
            raise ValueError(
                "Invalid dimension {}. Read the documentation for"
                " supported types.".format(dimension)
            )

    if len(dimension) == 3:
        if any([isinstance(dim, int) for dim in dimension[:2]]) and dimension[2] in [
            "uniform",
            "log-uniform",
        ]:
            return Integer(*dimension, transform=transform)
        elif any(
            [isinstance(dim, (float, int)) for dim in dimension[:2]]
        ) and dimension[2] in ["uniform", "log-uniform"]:
            return Real(*dimension, transform=transform)
        else:
            return Categorical(dimension, transform=transform)

    if len(dimension) == 4:
        if (
            any([isinstance(dim, int) for dim in dimension[:2]])
            and dimension[2] == "log-uniform"
            and isinstance(dimension[3], int)
        ):
            return Integer(*dimension, transform=transform)
        elif (
            any([isinstance(dim, (float, int)) for dim in dimension[:2]])
            and dimension[2] == "log-uniform"
            and isinstance(dimension[3], int)
        ):
            return Real(*dimension, transform=transform)

    if len(dimension) > 3:
        return Categorical(dimension, transform=transform)

    raise ValueError(
        "Invalid dimension {}. Read the documentation for "
        "supported types.".format(dimension)
    )


def normalize_dimensions(dimensions):
    """Create a ``Space`` where all dimensions are normalized to unit range.

    This is particularly useful for Gaussian process based regressors and is
    used internally by ``gp_minimize``.

    Parameters
    ----------
    dimensions : list, shape (n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

         NOTE: The upper and lower bounds are inclusive for `Integer`
         dimensions.
    """
    space = Space(dimensions)
    transformed_dimensions = []
    for dimension in space.dimensions:
        if isinstance(dimension, Categorical):
            transformed_dimensions.append(
                Categorical(
                    dimension.categories,
                    dimension.prior,
                    name=dimension.name,
                    transform="normalize",
                )
            )
        # To make sure that GP operates in the [0, 1] space
        elif isinstance(dimension, Real):
            transformed_dimensions.append(
                Real(
                    dimension.low,
                    dimension.high,
                    dimension.prior,
                    name=dimension.name,
                    transform="normalize",
                    dtype=dimension.dtype,
                )
            )
        elif isinstance(dimension, Integer):
            transformed_dimensions.append(
                Integer(
                    dimension.low,
                    dimension.high,
                    name=dimension.name,
                    transform="normalize",
                    dtype=dimension.dtype,
                )
            )
        else:
            raise RuntimeError("Unknown dimension type " "(%s)" % type(dimension))

    return Space(transformed_dimensions)
