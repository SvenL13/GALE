"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Benchmark functions
"""
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pygmo.core as pymgo

from gale._typing import ARRAY_LIKE_2D, BOUNDS, FUNCTION
from gale.doe import sampling
from gale.doe.acquisition.adaptive_sampling import OptProb
from gale.utils import rescale_from_unity


class BenchmarkFun:
    """
    Default benchmark problem
    """

    def __init__(
        self,
        fun: FUNCTION,
        bounds: BOUNDS,
        fun_param: Dict[str, float] = None,
        normalize_x: bool = False,
        scale_y: bool = False,
        y_min: float = None,
        y_max: float = None,
    ):

        self._fun: FUNCTION = fun
        self._bounds: BOUNDS = bounds
        self._dim = len(bounds)
        self._normalize_x = normalize_x
        self._norm_bounds = [[0, 1]] * self._dim
        self._scale_y = scale_y

        self._ymin = y_min
        self._ymax = y_max

        if isinstance(fun_param, dict):
            self._fun_param = (
                fun_param  # dict with additional parameter to change bench. fun
            )
        else:
            self._fun_param = dict()

    @property
    def bounds(self) -> BOUNDS:
        if self._normalize_x:
            bounds = np.array(self._norm_bounds).astype(float).tolist()
            return bounds
        else:
            bounds = np.array(self._bounds).astype(float).tolist()
            return bounds

    @property
    def dim(self) -> int:
        return int(self._dim)

    @property
    def ymin(self) -> float:
        """
        Return value of global minimum.
        """
        if self._ymin is None:  # find minima if not given
            return self.find_opt()
        else:
            return self._ymin

    @property
    def ymax(self) -> float:
        """
        Return value of global maximum.
        """
        if self._ymax is None:
            return self.find_opt("max")
        else:
            return self._ymax

    def update_param(self, **param):
        self._fun_param.update(param)

    def find_opt(
        self,
        opt: Literal["min", "max"] = "min",
        n_restarts: int = 15,
        bounds: Optional[BOUNDS] = None,
        _scaled_y: bool = True,
    ) -> float:
        """
        Find optimum of benchmark function based on (N+1)-ES Simple Evolutionary Algorithm (pygmo implementation)

        Parameters
        ----------
        opt: str, optional (default="min")
            find minimum or maximum
        n_restarts: int, optional (default=10)
            number of restarts for optimizer
        bounds: list, optional (default=None)
            bounds where to search for optimum, if None then default bounds are used

        Returns
        -------
        optimum: float
            found optimum, min or max depending on "opt" parameter
        """
        # used candidates in evolutionary algo
        n_cand = 1000 * self.dim

        if bounds is None:
            bounds = self.bounds

        if opt == "min":
            fun = lambda x: self.run(x, _scaled_y=_scaled_y)
        elif opt == "max":
            fun = lambda x: -self.run(x, _scaled_y=_scaled_y)
        else:
            raise NotImplementedError("Only min or max are supported inputs")

        # create random seeds
        seeds = np.random.uniform(1, 1e6, n_restarts).astype(int)

        y_opt_found = []

        for seed in seeds:  # restart n times
            # perform lhs for candidate start points
            x_cand_start = sampling(bounds, n_samples=n_cand, seed=seed)

            # create optimization problem
            opt_prob = OptProb(fun, bounds, ())
            prob = pymgo.problem(opt_prob)

            # init optimization
            algo = pymgo.algorithm(pymgo.sea(gen=1000, seed=seed))
            pop = pymgo.population(prob, n_cand)

            # set candidates starting point
            for i, x_cand in enumerate(x_cand_start):
                pop.set_x(i, x_cand)

            # start optimization
            pop = algo.evolve(pop)

            # get optima for round i
            x_opt = pop.champion_x.flatten()
            y_opt_found.append(self.run(x_opt, _scaled_y=_scaled_y))

        if opt == "min":
            return np.array(y_opt_found).min()
        else:
            return np.array(y_opt_found).max()

    def _scaler_y(self, y, bounds: Tuple[float, float] = (0, 10)):
        """
        Scale y to the defined bounds

        Parameters
        ----------
        y: float or np.ndarray

        bounds: tuple

        Returns
        -------
        y_scaled: float or np.ndarray
        """
        if self._ymin is None:  # find minima if not given
            self._ymin = self.find_opt(_scaled_y=False)

        if self._ymax is None:  # find minima if not given
            self._ymax = self.find_opt("max", _scaled_y=False)

        range = bounds[1] - bounds[0]

        assert range > 0

        scale = range / (self._ymax - self._ymin)
        y_shifted = y - self._ymin + bounds[0]  # shift data

        return scale * y_shifted  # scale data

    def run(
        self, x: ARRAY_LIKE_2D, _scaled_y: bool = True, **fun_param
    ) -> Union[float, np.ndarray]:
        """
        Evaluate benchmark function

        Parameters
        ----------
        x: arr_like, shape=(n_samples, n_features)
           input param
        fun_param: dict
            additional parameters to pass to function if executed

        Returns
        -------
        res: float or np.ndarray, shape=(n_samples)
            return benchmark function evaluation
        """
        x: np.ndarray = np.ascontiguousarray(x)

        if self._normalize_x:
            x = rescale_from_unity(x, bounds=self._bounds)

        param = self._fun_param.copy()
        param.update(fun_param)

        y = self._fun(x, **param)
        if self._scale_y and _scaled_y:
            y = self._scaler_y(y)
        return y


class AdjBenchmarkFun(BenchmarkFun):
    """
    Benchmark with adjustable dimensionality.
    """

    def __init__(
        self,
        fun: FUNCTION,
        bounds: BOUNDS,
        dim: int,
        normalize_x: bool = False,
        scale_y: bool = False,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ):

        super(AdjBenchmarkFun, self).__init__(
            fun,
            bounds,
            normalize_x=normalize_x,
            scale_y=scale_y,
            y_min=y_min,
            y_max=y_max,
        )

        self._dim: int = dim
        self._bounds: BOUNDS = bounds * self._dim
        self._norm_bounds: BOUNDS = [[0, 1]] * self._dim

    def change_dim(self, dim: int):
        self._dim = dim
        self._bounds = [self._bounds[0]] * self._dim
        self._norm_bounds = [[0, 1]] * self._dim

    def run(self, x: ARRAY_LIKE_2D, _scaled_y: bool = True, **fun_param):

        x: np.ndarray = np.ascontiguousarray(x)

        if self._normalize_x:
            x = rescale_from_unity(x, bounds=self._bounds)

        param = self._fun_param.copy()
        param.update(fun_param)

        y = self._fun(x, self._dim, **param)

        if self._scale_y and _scaled_y:
            y = self._scaler_y(y)
        return y


# 1D - Functions
def hump_single(x) -> np.ndarray:
    """
    Single Hump function, R^1

    Input Domain
    ------------
    -1.5 <= x <= 5
    """
    x = np.asarray_chkfinite(x)
    if np.min(x) >= -1.5 and np.max(x) <= 5:
        res = (
            3 * x
            - (0.05 / ((x - 4.75) ** 2 + 0.004))
            - (0.09 / ((x - 4.45) ** 2 + 0.005))
            - 6
        )
        return res.flatten()
    else:
        raise ValueError("Out of range.")


def hump_two(x) -> np.ndarray:
    """
    Single Hump function, R^1

    Input Domain
    ------------
    -0.5 <= x <= 5
    """
    x = np.asarray_chkfinite(x)
    if np.min(x) >= -0.5 and np.max(x) <= 5:
        res = (
            5 * x
            + (0.05 / ((x - 4.5) ** 2 + 0.002))
            - (0.5 / ((x - 3.5) ** 2 + 0.03))
            - 6
        )
        return res.flatten()
    else:
        raise ValueError("Out of range.")


def gram_lee(x) -> np.ndarray:
    """
    Gramacy and Lee function, R^1

    Input Domain
    ------------
    -1.5 <= x <= 1
    """
    x = np.asarray_chkfinite(x)
    if np.min(x) >= -1.5 and np.max(x) <= 1:

        res = (60 * np.sin(6 * np.pi * x)) / (2 * np.cos(x)) + np.power(x - 1, 4)
        return res.flatten()
    else:
        raise ValueError("Out of range.")


# 2D - Functions
def bl_fun(x) -> np.ndarray:
    """
    Beker&Logan function (BL), R^2

    Input Domain
    ------------
    -10 <= x1 <= 10
    -10 <= x2 <= 10
    """
    x = np.asarray_chkfinite(x)

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    if (
        np.min(x1) >= -10
        and np.max(x1) <= 10
        and np.min(x2) >= -10
        and np.max(x2) <= 10
    ):
        return ((np.abs(x1) - 5) ** 2 + (np.abs(x2) - 5) ** 2).flatten()
    else:
        raise ValueError("Out of range.")


def egg(x):
    """
    Eggholder-function, R^2

    Input Domain
    ------------
    -512 <= x1 <= 512
    -512 <= x2 <= 512
    """
    x = np.asarray_chkfinite(x)

    # bounds
    lb1 = -512
    lb2 = -512
    up1 = 512
    up2 = 512

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    if (
        np.min(x1) >= lb1
        and np.max(x1) <= up1
        and np.min(x2) >= lb2
        and np.max(x2) <= up2
    ):

        fun_val = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(
            np.sqrt(np.abs(x1 - (x2 + 47)))
        )
        return fun_val.flatten()
    else:
        raise ValueError("Out of range.")


def dropwave(x):
    """
    DropWave-function, R^2

    Input Domain
    ------------
    -0.6 <= x <= 0.9
    """
    x = np.asarray_chkfinite(x)

    if (x < -0.6).any() or (x > 0.9).any():
        raise ValueError("Out of range.")

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    fun_val = -(1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (
        0.5 * (x1**2 + x2**2) + 2
    )
    return fun_val.flatten()


def himm(x):
    """
    Himmelblau-function, R^2

    Input Domain
    ------------
    -6 <= x1 <= 6
    -6 <= x2 <= 6

    Local Max:
    f(x1=-0.27085, x2=-0.923039) = 181.617

    Local Min:
    f(x1=3.0, x2=2.0) = 0.0
    f(x1=-2.805118, x2=3.131312) = 0.0
    f(x1=-3.779310, x2=-3.283186) = 0.0
    f(x1=3.584428, x2=-1.848126) = 0.0
    """
    x = np.asarray_chkfinite(x)

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    if np.min(x1) >= -6 and np.max(x1) <= 6 and np.min(x2) >= -6 and np.max(x2) <= 6:
        return ((x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2).flatten()
    else:
        raise ValueError("Out of range.")


def branin(x, **kwargs):
    """
    Branin-Hoo-function, R^2

    Input Domain
    ------------
    -5 <= x1 <= 10
    0 <= x2 <= 15
    """
    x = np.asarray_chkfinite(x)

    # bounds
    lb1 = -5
    lb2 = 0
    up1 = 10
    up2 = 15

    # recommended parameters
    pi = np.pi
    a = kwargs.pop("a", 1)
    b = kwargs.pop("b", 5.1 / (4 * np.power(pi, 2)))
    c = kwargs.pop("c", 5 / pi)
    r = kwargs.pop("r", 6)
    s = kwargs.pop("s", 10)
    t = kwargs.pop("t", 1 / (8 * pi))

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    if (
        np.min(x1) >= lb1
        and np.max(x1) <= up1
        and np.min(x2) >= lb2
        and np.max(x2) <= up2
    ):

        fun_val = (
            a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        )

        return fun_val.flatten()
    else:
        raise ValueError("Out of range.")


# 3D - Functions
def ishi(x, **kwargs):
    """
    Ishigami-function, R^3

    Input Domain
    ------------
    -pi <= x <= pi
    """
    x = np.asarray_chkfinite(x)

    # recommended parameters
    a = kwargs.pop("a", 7)
    b = kwargs.pop("b", 0.1)

    # bounds
    if (x < -np.pi).any() or (np.pi < x).any():
        raise ValueError("Out of range.")

    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

    fun_value = (
        np.sin(x1) + a * np.power(np.sin(x2), 2) + b * np.power(x3, 4) * np.sin(x1)
    )

    return fun_value.flatten()


# 6D - Functions
def hart6d(x):
    """
    6D-Hartmann-function, R^6

    Input Domain
    ------------
    0 <= x <= 1
    """
    x = np.asarray_chkfinite(x)

    if (x < 0).any() or (x > 1).any():
        raise ValueError("Out of range.")

    # param
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    def h_fun(x_k) -> float:
        outer = 0
        for i in range(4):
            inner = 0
            for j in range(6):
                inner += A[i, j] * (x_k[j] - P[i, j]) ** 2
            new = alpha[i] * np.exp(-inner)
            outer += new
        return -outer

    if len(x.shape) == 1:
        fun_val = h_fun(x)
    else:
        fun_val = list()
        for x_i in x:
            fun_val.append(h_fun(x_i))
        fun_val = np.array(fun_val)

    return fun_val.flatten()


# nD - Functions
def rose(x, dim: int):
    """
    Rosenbrock function, R^n

    Input Domain
    ------------
    -5 <= x <= 5
    """
    x = np.asarray_chkfinite(x)

    if (x < -5).any() or (x > 5).any():
        raise ValueError("Out of range.")

    if len(x.shape) == 1:
        sum_rb = 0
        for k in range(dim - 1):
            sum_rb += 100 * (x[k + 1] - x[k] ** 2) ** 2 + (x[k] - 1) ** 2
    else:
        sum_rb = np.zeros(x.shape[0])
        for k in range(dim - 1):
            sum_rb = sum_rb + (
                100 * (x[:, k + 1] - x[:, k] ** 2) ** 2 + (x[:, k] - 1) ** 2
            )

    return sum_rb.flatten()


def micha(x, dim: int, **kwargs):
    """
    Michalewicz function, R^n

    Input Domain
    ------------
    0 <= x <= pi
    """
    x = np.asarray_chkfinite(x)

    if (x < 0).any() or (x > np.pi).any():
        raise ValueError("Out of range.")

    # recommended parameters
    m = kwargs.pop("m", 10)

    if len(x.shape) == 1:
        if len(x) > dim:
            raise ValueError("Given input dimensionality exceeds %i" % dim)

        tmpi = np.arange(1, x.shape[0] + 1)
        fun_val = -np.sum(
            np.sin(x) * (np.power(np.sin(np.power(x, 2) * tmpi / np.pi), m))
        )
    else:
        if x.shape[1] > dim:
            raise ValueError("Given input dimensionality exceeds %i" % dim)
        tmpi = np.arange(1, x.shape[1] + 1)
        fun_val = -np.sum(
            np.sin(x) * (np.power(np.sin(np.power(x, 2) * tmpi / np.pi), m)), axis=1
        )

    return fun_val.flatten()


def ackley(x, dim: int, **kwargs):
    """
    Ackley's function, R^n

    Input Domain
    ------------
    -5 <= x <= 5
    """
    x = np.asarray_chkfinite(x)

    if (x < -5).any() or (x > 5).any():
        raise ValueError("Out of range.")

    # recommended parameters
    a = kwargs.pop("a", 20)
    b = kwargs.pop("b", 0.2)
    c = kwargs.pop("c", 2 * np.pi)

    if len(x.shape) == 1:
        sum_cos = np.sum(np.cos(c * x))
        sum_x = np.sum(x**2)
    else:
        sum_cos = np.sum(np.cos(c * x), axis=1)
        sum_x = np.sum(x**2, axis=1)

    fun_val = (
        -a * np.exp(-b * np.sqrt(sum_x / dim)) - np.exp(sum_cos / dim) + a + np.exp(1)
    )

    return fun_val.flatten()


def schwefel(x, dim: int):
    """
    Rosenbrock function, R^n

    Input Domain
    ------------
    -500 <= x <= 500
    """
    x = np.array(x)

    if (x < -500).any() or (x > 500).any():
        raise ValueError("Out of range.")

    x = np.asarray_chkfinite(x)

    if len(x.shape) == 1:
        fun_val = 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        fun_val = 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

    return fun_val.flatten()


def stybli(x, dim: int):
    """
    Styblinski-Tang function, R^n

    Input Domain
    ------------
    -5 <= x <= 5
    """
    x = np.array(x)

    if (x < -5).any() or (x > 5).any():
        raise ValueError("Out of range.")

    x = np.asarray_chkfinite(x)

    if len(x.shape) == 1:
        fun_val = 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x)
    else:
        fun_val = 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x, axis=1)

    return fun_val.flatten()


# Create benchmark problems
# 1D functions
HumpSingle_norm = BenchmarkFun(
    hump_single, [[-1.5, 5]], normalize_x=True, y_min=-11.1819
)
HumpTwo_norm = BenchmarkFun(hump_two, [[-0.5, 5]], normalize_x=True, y_min=-8.5282)
GramLee_norm = BenchmarkFun(gram_lee, [[-1.5, 1]], normalize_x=True, y_min=-173.3625)
# 2D functions
BekerLogan_norm = BenchmarkFun(bl_fun, [[-10, 10]] * 2, normalize_x=True, y_min=0)
Egg_norm = BenchmarkFun(egg, [[-512, 512]] * 2, normalize_x=True, y_min=-959.6406)
Himm_norm = BenchmarkFun(himm, [[-6, 6]] * 2, normalize_x=True, y_min=0)
Branin_norm = BenchmarkFun(
    branin, [[-5, 10], [0, 15]], normalize_x=True, y_min=0.397887
)
DropWave_norm = BenchmarkFun(dropwave, [[-0.6, 0.9]] * 2, normalize_x=True, y_min=-1)
# 3D functions
Ishi_norm = BenchmarkFun(ishi, [[-np.pi, np.pi]] * 3, normalize_x=True, y_min=-10.7342)
# 6D functions
Hart6d_norm = BenchmarkFun(hart6d, [[0, 1]] * 6, normalize_x=True, y_min=-3.32237)
# nD functions
Rose3d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=3, normalize_x=True, y_min=0)
Rose4d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=4, normalize_x=True, y_min=0)
Rose6d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=6, normalize_x=True, y_min=0)
Rose8d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=8, normalize_x=True, y_min=0)
Rose16d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=16, normalize_x=True, y_min=0)
Rose32d_norm = AdjBenchmarkFun(rose, [[-5, 5]], dim=32, normalize_x=True, y_min=0)
Ackley3d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=3, normalize_x=True, y_min=0)
Ackley4d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=4, normalize_x=True, y_min=0)
Ackley6d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=6, normalize_x=True, y_min=0)
Ackley8d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=8, normalize_x=True, y_min=0)
Ackley16d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=16, normalize_x=True, y_min=0)
Ackley32d_norm = AdjBenchmarkFun(ackley, [[-5, 5]], dim=32, normalize_x=True, y_min=0)
Micha2d_norm = AdjBenchmarkFun(
    micha, [[0, np.pi]], dim=2, normalize_x=True, y_min=-1.8066
)
Micha3d_norm = AdjBenchmarkFun(
    micha, [[0, np.pi]], dim=3, normalize_x=True, y_min=-1.8066
)
Micha4d_norm = AdjBenchmarkFun(
    micha, [[0, np.pi]], dim=4, normalize_x=True, y_min=-3.7042
)
Micha6d_norm = AdjBenchmarkFun(
    micha, [[0, np.pi]], dim=6, normalize_x=True, y_min=-1.8066
)
Micha8d_norm = AdjBenchmarkFun(
    micha, [[0, np.pi]], dim=8, normalize_x=True, y_min=-7.6488
)
Schwefel2d_norm = AdjBenchmarkFun(
    schwefel,
    [[-500, 500]],
    dim=2,
    normalize_x=True,
    scale_y=False,
    y_min=0,
    y_max=1675.9138701218737,
)
Stybli2d_norm = AdjBenchmarkFun(
    stybli, [[-5, 5]], dim=2, normalize_x=True, y_min=-78.33233141
)
Stybli4d_norm = AdjBenchmarkFun(
    stybli, [[-5, 5]], dim=4, normalize_x=True, y_min=-156.66396
)
Stybli8d_norm = AdjBenchmarkFun(
    stybli, [[-5, 5]], dim=8, normalize_x=True, y_min=-313.32932563
)

benchmarks: Dict[str, BenchmarkFun] = {
    "HumpSingle_norm": HumpSingle_norm,
    "HumpTwo_norm": HumpTwo_norm,
    "GramLee_norm": GramLee_norm,
    "BekerLogan_norm": BekerLogan_norm,
    "Himm_norm": Himm_norm,
    "Rose3d_norm": Rose3d_norm,
    "Rose4d_norm": Rose4d_norm,
    "Rose6d_norm": Rose6d_norm,
    "Rose8d_norm": Rose8d_norm,
    "Rose16d_norm": Rose16d_norm,
    "Rose32d_norm": Rose32d_norm,
    "Egg_norm": Egg_norm,
    "Schwefel2d_norm": Schwefel2d_norm,
    "Branin_norm": Branin_norm,
    "Hart6d_norm": Hart6d_norm,
    "Ackley3d_norm": Ackley3d_norm,
    "Ackley4d_norm": Ackley4d_norm,
    "Ackley6d_norm": Ackley6d_norm,
    "Ackley8d_norm": Ackley8d_norm,
    "Ackley16d_norm": Ackley16d_norm,
    "Ackley32d_norm": Ackley32d_norm,
    "Ishi_norm": Ishi_norm,
    "Micha2d_norm": Micha2d_norm,
    "Micha3d_norm": Micha3d_norm,
    "Micha4d_norm": Micha4d_norm,
    "Micha6d_norm": Micha6d_norm,
    "Micha8d_norm": Micha8d_norm,
    "DropWave_norm": DropWave_norm,
    "Stybli4d_norm": Stybli4d_norm,
    "Stybli8d_norm": Stybli8d_norm,
}

BENCHMARKS_1D: Dict[str, BenchmarkFun] = {
    "HumpSingle_norm": HumpSingle_norm,
    "HumpTwo_norm": HumpTwo_norm,
    "GramLee_norm": GramLee_norm,
}

BENCHMARKS_2D: Dict[str, BenchmarkFun] = {
    "BekerLogan_norm": BekerLogan_norm,
    "Egg_norm": Egg_norm,
    "Himm_norm": Himm_norm,
    "Branin_norm": Branin_norm,
    "DropWave_norm": DropWave_norm,
    "Micha2d_norm": Micha2d_norm,
    "Schwefel2d_norm": Schwefel2d_norm,
}

BENCHMARKS_3D: Dict[str, BenchmarkFun] = {
    "Ishi_norm": Ishi_norm,
    "Rose3d_norm": Rose3d_norm,
    "Micha3d_norm": Micha3d_norm,
    "Ackley3d_norm": Ackley3d_norm,
}

BENCHMARKS_4D: Dict[str, BenchmarkFun] = {
    "Ackley4d_norm": Ackley4d_norm,
    "Micha4d_norm": Micha4d_norm,
    "Rose4d_norm": Rose4d_norm,
    "Stybli4d_norm": Stybli4d_norm,
}

BENCHMARKS_6D: Dict[str, BenchmarkFun] = {
    "Hart6d_norm": Hart6d_norm,
    "Rose6d_norm": Rose6d_norm,
    "Micha6d_norm": Micha6d_norm,
    "Ackley6d_norm": Ackley6d_norm,
}

BENCHMARKS_8D: Dict[str, BenchmarkFun] = {
    "Ackley8d_norm": Ackley8d_norm,
    "Micha8d_norm": Micha8d_norm,
    "Rose8d_norm": Rose8d_norm,
    "Stybli8d_norm": Stybli8d_norm,
}

BENCHMARKS_HIGH: Dict[str, BenchmarkFun] = {
    "Rose16d_norm": Rose16d_norm,
    "Rose32d_norm": Rose32d_norm,
    "Ackley16d_norm": Ackley16d_norm,
    "Ackley32d_norm": Ackley32d_norm,
}

ALL_BENCHMARK_NAMES: List[str] = (
    [*BENCHMARKS_1D]
    + [*BENCHMARKS_2D]
    + [*BENCHMARKS_3D]
    + [*BENCHMARKS_4D]
    + [*BENCHMARKS_6D]
    + [*BENCHMARKS_8D]
)

ALL_BENCHMARK_NAMES_GROUPED: List[List[str]] = [
    [*BENCHMARKS_1D],
    [*BENCHMARKS_2D],
    [*BENCHMARKS_3D],
    [*BENCHMARKS_4D],
    [*BENCHMARKS_6D],
    [*BENCHMARKS_8D],
]
