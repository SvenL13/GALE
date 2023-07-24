"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Adaptive Sampling Base
"""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pygmo.core as pg
    PYGMO_AVAILABLE = True
except ImportError:
    PYGMO_AVAILABLE = False
from scipy.optimize import differential_evolution
from skopt.sampler import Lhs

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D, BOUNDS
from gale.doe.space import Space
from gale.doe.utils import edge_sampling
from gale.models import SurrogateRegressionModel, cook_model
from gale.utils import closest_point, ensure_2d


class AdaptiveSampling(ABC):
    """
    Base class for adaptive sampling

    Parameters
    ----------
    bounds: Bounds
        Dimensions of input space / design space,
        e.g. N_D bounds: [[lw_d1, up_d1], [lw_d2, up_d2], ..., [lw_dN, up_dN]]
    n_init: int, optional(default=None)
        default sampling number for LHS, if None -> 10*m is used, where m is the dimension
        of input space
    model: str, optional(default="GP)
        specify the used surrogate model ("GP", "BNN")
    model_param: dict, optional(default=None)
        additional arguments used to creating the surrogate model
    init_sampling: Union[np.ndarray, Literal["LHS"]], optional (default="lhs")
        sampling method used to create initial samples, optional samples can be
        passed directly
    lhs_crit: Optional[Literal["correlation", "maximin", "ratio"]], optional (default="maximin")
        optimization criterion for LHS sampling, if None -> no optimization of LHS
        is performed
    lhs_append_bounds: bool, optional (default=False)
        append outer bounds of the design domain to the initial samples
    optimizer: Optional[Literal["pygmo", "diff_evo"]], optional (default=None)
        used optimizer for optimizing the acquisition function, if None -> default
        optimizer is selected
    rnd_state: int, optional (default=None)
        seed
    n_jobs: int, optional (default=None)
        max no. of parallelization used e.g. in optimization, if None -> max. no. of
        cpu_cores is used
    verbose: bool, optional (default=False)
        plot information during iterations
    """

    name: str = "Adaptive Base Class"
    short_name: str = "ABC"
    optimizer_used: bool = False
    dtype = np.float64

    def __init__(
        self,
        bounds: BOUNDS,
        n_init: int = None,
        model: Literal["GP", "SGP", "BNN"] = "GP",
        model_param: dict = None,
        init_sampling: Union[np.ndarray, Literal["LHS"]] = "LHS",
        lhs_crit: Optional[Literal["correlation", "maximin", "ratio"]] = "maximin",
        rnd_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        optimizer: Optional[Literal["pygmo", "diff_evo"]] = None,
        lhs_append_bounds: bool = False,
        fit_model_every: int = 1,
        n_max: int = None,
    ):

        assert optimizer in ("pygmo", "diff_evo") or optimizer is None

        # general parameters
        self.verbose: bool = verbose
        self._rnd_state: Optional[int] = rnd_state  # seed for replication
        self.iter_count: int = 1  # no. of iterations
        self.opt_info = None
        self.auto_stop: bool = False  # termination condition is used
        self.n_max: Optional[int] = n_max

        # track param
        self.tracking: list = []  # holds information for all iterations
        self.tracking_i: dict = {}  # track param for individual iterations

        # create design space
        self.bounds: BOUNDS = bounds  # bounds for input w_o transformation
        self._design_space: Space = Space(bounds)

        # init model
        self.surr_model: SurrogateRegressionModel
        self.surrogate_name: str = model

        model_param: dict = {} if model_param is None else model_param

        self.surr_model, self.model_class = cook_model(
            model,
            rnd_state=rnd_state,
            return_class=True,
            **model_param
        )
        self.fit_model_every: int = max([1, int(fit_model_every)])
        if not hasattr(self.surr_model, "_update") and self.fit_model_every > 1:
            raise ValueError("Model doesn't implement an update method.")
        self._last_trained_iter = 0

        # optimizer parameters
        if n_jobs is None:
            self.n_jobs: int = os.cpu_count()
        else:
            self.n_jobs = n_jobs

        if self.surrogate_name == "BNN":
            if optimizer is None:
                optimizer = "diff_evo"
            if optimizer != "diff_evo":
                raise NotImplementedError("Optimizer not supported for BNN")
        else:
            if optimizer is None:
                optimizer = "pygmo"
        self.optimizer_name: Literal["pygmo", "diff_evo"] = optimizer

        # init response variables
        self.y_observed: ARRAY_LIKE_1D = (
            list()
        )  # observed objective response (e.g. experiments)
        self.X_observed: ARRAY_LIKE_2D = list()

        # initial sampling number
        if isinstance(n_init, int) and n_init > 0:
            self.n_init: int = n_init
        else:  # init sampling number is recommended to set to 10*ndim
            self.n_init: int = 10 * self._design_space.n_dims

        # initial sampling
        if isinstance(init_sampling, np.ndarray):  # init samples directly given
            init_sampling = ensure_2d(init_sampling)
            self.X_init: ARRAY_LIKE_2D = self._design_space.transform(init_sampling)
            self.init_sampling_name: str = "Given samples"
            self.lhs_crit: Optional[Literal["correlation", "maximin", "ratio"]] = None
            self.lhs_appnd_bounds: bool = False
        else:
            self.init_sampling_name = init_sampling
            self.lhs_crit = lhs_crit
            self.lhs_appnd_bounds = lhs_append_bounds
            self.X_init: ARRAY_LIKE_2D = self._init_sampling(
                self.n_init, lhs_crit, append_edges=lhs_append_bounds
            )

        # create initial candidate points (optional, depending on used adaptive sampling
        # method)
        self.X_cand: Optional[ARRAY_LIKE_2D] = None
        self.aq_cand: Optional[ARRAY_LIKE_2D] = None

    def __str__(self):
        res = "\n####################\n"
        res += "Start Adaptive Sampling:\n"
        res += "Sampling Strategy: %s (%s)\n\n" % (
            str(self.__class__.name),
            str(self.__class__.short_name),
        )
        res += "Given Parameters:\n"
        res += "------------------\n"
        res += "Seed: %i\n" % self._rnd_state
        res += "Verbose: %r\n" % self.verbose
        res += "Init. Sampling Number: %i\n" % self.n_init
        res += "Bounds: %r\n" % self.bounds
        res += "Init. Sampling: %s\n" % self.init_sampling_name
        res += "Sampling on bounds: %s\n" % self.lhs_appnd_bounds
        if self.init_sampling_name == "LHS":
            res += "Init. Sampling - LHS optimization: %s\n" % self.lhs_crit
        if self.__class__.optimizer_used:
            res += "Optimizer: %s\n" % self.optimizer_name
            res += "No. of parallelization for optimization: %i\n" % self.n_jobs
        else:
            res += "Optimizer: False\n"
        if isinstance(self.X_cand, np.ndarray):
            res += "No. of used candidate points: %i\n" % self.X_cand.shape[0]
        res += "\n"
        res += "Used Surrogate Model: %s\n" % self.surrogate_name
        res += "------------------\n"
        res += "####################\n"
        return res

    @property
    def param(self) -> dict:

        param = {
            "verbose": self.verbose,
            "rnd_state": self._rnd_state,
            "iter_count": self.iter_count,
            "opt_info": self.opt_info,
            "auto_stop": self.auto_stop,
            "tracking": self.tracking,
            "tracking_i": self.tracking_i,
            "bounds": self.bounds,
            "design_space": self._design_space,
            "n_jobs": self.n_jobs,
            "optimizer_name": self.optimizer_name,
            "y_observed": self.y_observed,
            "X_observed": self.X_observed,
            "model_name": self.surrogate_name,
            "n_init": self.n_init,
            "init_sampling_name": self.init_sampling_name,
            "lhs_crit": self.lhs_crit,
            "lhs_append_bounds": self.lhs_appnd_bounds,
            "X_init": self.X_init,
            "X_cand": self.X_cand,
        }
        return param.copy()

    def _init_sampling(
        self,
        n_samples: int,
        lhs_crit: Optional[Literal["correlation", "maximin", "ratio"]] = "maximin",
        append_edges: bool = False,
    ) -> np.ndarray:
        """
        Sample the initial design points with LHS from the design space

        Parameters
        ----------
        n_samples: int
            number of samples to create
        lhs_crit: str, optional (default="maximin")
            optimization passed if LHS is used
        append_edges: bool, optional (default=False)
            create design points at upper and lower bounds and append to init samples

        Returns
        -------
        samples: np.ndarray, shape=(n_samples, n_features)
            init design samples
        """
        assert isinstance(n_samples, int)
        assert n_samples > 0

        # create initial LHS design, X e R^(m x u)
        lhs = Lhs(criterion=lhs_crit, iterations=1000)

        if append_edges:
            assert (
                n_samples > 2**self._design_space.n_dims
            ), "n_samples should be greater than 2**n_dim"
            n_init = n_samples - self._design_space.n_dims**2
        else:
            n_init = n_samples

        # generate samples
        X_samples = lhs.generate(
            self._design_space.transformed_bounds,
            n_init,
            random_state=self._rnd_state,
        )

        # initial sampling on bounds
        if append_edges:
            idx = np.min([self._design_space.n_dims**2, n_samples])
            X_edges = edge_sampling(
                self._design_space.transformed_bounds, self._design_space.n_dims
            )
            X_samples = np.concatenate([X_samples, X_edges[idx]], axis=0)

        _, count = np.unique(X_samples, axis=0, return_counts=True)
        if np.any(count > 1):
            warnings.warn("DSpace - draw: X_samples includes duplicates.")

        return np.array(X_samples, dtype=self.dtype)

    def _gen_cand_points(
        self,
        generator: Literal["ff", "random", "lhs"] = "lhs",
        n_samples: int = None,
        lhs_crit: Optional[Literal["correlation", "maximin", "ratio"]] = "maximin",
        lhs_opt_iter: int = 1000,
        append_edges: bool = False,
    ) -> np.ndarray:
        """
        Generate Candidate Points depending on selected generator

        Parameters
        ----------
        generator: Literal["ff", "random", "lhs"], optional(default="lhs")
            generator used to create new candidate points
        n_samples: int, optional (default=None)
            no. of candidate points generated,
            if None -> n_samples = 5000 * n_dims is used
        lhs_crit: Union[None, Literal["correlation", "maximin", "ratio"]], optional(default="maximin")
            arg passed to lhs generator, optimization used in lhs design,
            if None -> no optimization is used

        Returns
        -------
        X_cand: np.ndarray, shape=(n_samples, n_features)
            generated candidate points
        """
        if (
            n_samples is None
        ):  # auto selection of no. cand. points depending on design dimensions
            n_samples = self._design_space.n_dims * 5000
        elif not isinstance(n_samples, int):
            raise ValueError("n_init should be integer")

        if generator == "random":  # random sampling
            X_cand = self._design_space.rvs(
                n_samples=n_samples, random_state=self._rnd_state
            )
            X_cand = self._design_space.transform(X_cand)

        elif generator == "lhs":  # latin hypercube design
            lhs = Lhs(criterion=lhs_crit, iterations=lhs_opt_iter)
            X_cand = lhs.generate(
                self._design_space.transformed_bounds,
                n_samples,
                random_state=self._rnd_state,
            )
        else:
            raise ValueError(
                'Generator not supported. Supported types: ["random", "ff", "lhs"]'
            )

        if append_edges is True:
            X_edges = edge_sampling(
                self._design_space.transformed_bounds, self._design_space.n_dims
            )
            X_cand = np.concatenate([X_cand, X_edges], axis=0)

        return np.array(X_cand, dtype=self.dtype)

    def create_result(self) -> dict:
        """
        Create dictionary with results

        Returns
        -------
        res: dict
            dictionary with results
        """
        x_obs = np.array(self.X_observed)
        transformer = None

        y_obs = np.array(self.y_observed).flatten()

        # create dict for observations
        df = pd.DataFrame(self.tracking)

        # write x in individual columns per dimension
        col_names = ["x_obs_%i" % i for i in range(self._design_space.n_dims)]
        df_xobs = pd.DataFrame(df.pop("x_proposed").tolist(), columns=col_names)
        col_names = ["x_prob_%i" % i for i in range(self._design_space.n_dims)]
        df_xprop = pd.DataFrame(df.pop("x_obs").tolist(), columns=col_names)

        # df = pd.DataFrame(self._tracking).set_index("iter")
        df_full = df.join([df_xobs, df_xprop]).set_index("iter")
        df_full["y_obs"] = df_full["y_obs"].astype(float)

        # dict to store results
        res = {
            "model": self.surr_model,
            "conditions": self._design_space,
            "counts": self.iter_count - 1,
            "obs_cond": x_obs,
            "obs_y": y_obs,
            "transformer": transformer,
            "desc": self.__str__(),
            "seed": self._rnd_state,
            "specs": {str(self.__class__.short_name): self.param},
            "optimizer": self.opt_info,
            "tracking_df": df_full,
        }
        return res

    def ask(self) -> List[float]:
        """
        Propose the next point to evaluate the objective response based on used method

        Returns
        -------
        next_x: list, shape=(n_features)
            next condition to evaluate
        """
        n_observed = len(self.y_observed)

        if n_observed < self.n_init:  # use initial sampling for first points

            # propose next point init sampling
            next_x: List[float] = self.X_init[n_observed]

        else:  # use adaptive approach to propose next point
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                next_x = np.array(self.ask_()).tolist()

        # print warning if observation was done before
        if self.X_observed:
            _, _, min_delta_x = closest_point(
                np.array(next_x), self.X_observed, return_dist=True
            )
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated at this point before.")

        next_x = self._design_space.inverse_transform(next_x)[0]

        next_x = np.array(next_x).astype(float).tolist()

        # track results (w_o transformation)
        self.tracking_i["x_proposed"] = next_x

        return next_x

    @abstractmethod
    def ask_(self) -> Union[List[float], np.ndarray]:
        """
        Placeholder for implementation of adaptive approach, should return next_x

        Returns
        -------
        next_x: list, shape=(n_features)
            next condition to evaluate
        """
        pass

    def tell(self, obs_x: tuple, obs_y: tuple) -> Tuple[dict, bool]:
        """
        Return observation to the adaptive method

        Parameters
        ----------
        obs_x: tuple, shape=(n_features)
            condition that was observed
        obs_y: tuple, shape=(n_outputs)
            System response that was observed

        Returns
        -------
        res: dict
            dictionary with results
        terminate: bool
            if termination condition is met -> return True
        """
        obs_x = np.ascontiguousarray(obs_x).ravel()
        obs_y = np.ascontiguousarray(obs_y).ravel()

        if self.verbose:
            print("\nObserved point in iteration %i" % self.iter_count)
            print("Observed X:", obs_x)
            print("Observed y:", obs_y)

        # track results (before transformation)
        self.tracking_i["y_obs"] = obs_y
        self.tracking_i["x_obs"] = obs_x
        self.tracking_i["iter"] = self.iter_count

        obs_x = self._design_space.transform(obs_x)[0]

        # update observed datasets
        self.y_observed.append(obs_y)
        self.X_observed.append(obs_x)

        if len(self.y_observed) >= self.n_init:

            # fit surrogate model after n_init points are observed
            diff_iter = self.iter_count - self._last_trained_iter
            if diff_iter >= self.fit_model_every:
                self.surr_model.fit(
                    np.array(self.X_observed), np.array(self.y_observed)
                )
                self._last_trained_iter = self.iter_count
            else:
                self.surr_model.update(
                    np.array(self.X_observed), np.array(self.y_observed)
                )

        if self.auto_stop:
            # evaluate if stopping criterion is met
            terminate: bool = self.stopping_crit()
        else:
            terminate = False

        # increment iteration counter
        self.iter_count += 1

        # append tracking for iteration
        self.tracking.append(self.tracking_i.copy())

        # create result
        res = self.create_result()

        return res, terminate

    def return_aq(self, x: ARRAY_LIKE_2D):
        """
        Evaluate acquisition function at x if optimizer is used, otherwise aq_values at
        x_candidate points are returned together with x_cand

        Parameters
        ----------
        x: arr_like, shape=(n_samples, n_features)
            input points

        Returns
        -------
        aq_values: np.ndarray, shape=(n_samples)
            acquisition value at x
        x_cand: np.ndarray, optional shape=(n_samples, n_features)
            candidate points, only returned if no optimizer is used in method
        """
        if self.optimizer_used:
            x = self._design_space.transform(x)

            return self._return_aq(x)
        else:
            x_cand, aq_values = self._return_aq(None)
            x_cand = self._design_space.inverse_transform(x_cand)
            return x_cand, aq_values

    @abstractmethod
    def _return_aq(self, x) -> np.ndarray:
        """
        Placeholder for function that returns the acquisition function.
        should return value of acquisition function at point x

        if no optimizer is used in method, the fun should return the candidate points
        together with aq value at each point
        """
        pass

    def stopping_crit(self) -> bool:
        """
        Implement stopping criteria for adaptive sampling process

        Returns
        -------
        stop: bool
            stop adaptive sampling process if stop=True
        """
        pass

    def _optimize(
        self,
        fun: Callable[[ARRAY_LIKE_2D], ARRAY_LIKE_1D],
        optimizer: Literal["pygmo", "diff_evo"] = "pygmo",
        args: tuple = (),
    ) -> List[float]:
        """
        Optimizer for finding the minimum of the acquisition function

        Optimizers
        ----------
        "diff_evo" -> scipy.optimize.differential_evolution
            Differential Evolution is stochastic in nature (does not use gradient methods)
            to find the minimum, and can search large areas of candidate space, but often
            requires larger numbers of function evaluations than conventional
            gradient-based techniques.
        "pygmo" -> pygmo.sea
            (N+1)-ES simple evolutionary algorithm

        Parameters
        ----------
        fun: fun
            function to be minimized, should take x as input
        optimizer: str, optional (default="pygmo")
            used optimizer, one of ("diff_evo", "pygmo")
        args: tuple
            additional arguments to pass to function while optimizing

        Returns
        -------
        next_x: list, shape=(n_features)
            minimum of fun
        """
        bounds = self._design_space.transformed_bounds

        if optimizer == "diff_evo":

            result = differential_evolution(
                fun, args=args, bounds=bounds, seed=self._rnd_state
            )
            next_x = result.x

            # optimizer info
            self.opt_info = differential_evolution.__dict__

        elif optimizer == "pygmo":

            if PYGMO_AVAILABLE is False:
                raise ValueError("pygmo installation not found.")

            n_candidates = 50 * self._design_space.n_dims
            # perform lhs for candidate start points
            x_cand_start = self._init_sampling(n_samples=n_candidates)

            # create optimization problem
            opt_prob = OptProb(fun, bounds, args)
            prob = pg.problem(opt_prob)

            # init optimization
            algo = pg.algorithm(pg.sea(gen=1000, seed=self._rnd_state))
            pop = pg.population(prob, n_candidates)

            # set candidates starting point
            for i, x_cand in enumerate(x_cand_start):
                pop.set_x(i, x_cand)

            # start optimization
            pop = algo.evolve(pop)

            # propose candidate
            next_x = pop.champion_x.flatten()
        else:
            raise ValueError(
                "Unknown optimizer, should be one of ['diff_evo', 'pygmo']"
            )

        return next_x.tolist()


class OptProb:
    """
    Optimization problem wrapper class for the pygmo optimizer implementation

    Parameters
    ----------
    fun: Callable
        function that is optimized
    bounds: tuple
        problem bounds
    args: tuple
        additional arguments to pass to fun
    """

    def __init__(self, fun, bounds: tuple, args: tuple):
        self._fun = fun
        self._args = args

        self._observer = {"x": list(), "fit_val": list()}

        # create bounds
        lb = list()
        up = list()
        for bound in bounds:
            lb.append(bound[0])
            up.append(bound[1])
        self._bounds = (lb, up)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        fit_val = self._fun(x, *self._args)

        self._observer["x"].append(x)
        self._observer["fit_val"].append(fit_val)

        return np.array(fit_val).flatten()

    def get_bounds(self) -> tuple:
        return self._bounds

    def get_observations(self) -> dict:
        return self._observer.copy()
