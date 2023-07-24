"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Sequential Sampling
"""
from timeit import default_timer as timer
from typing import Callable, Literal, Optional, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gale.doe.acquisition.adaptive_sampling import AdaptiveSampling
from gale.doe.utils import sampling
from gale.models import mean_relative_error
from gale.utils import print_progress


class SeqED:
    """
    Sequential experiment design class

    Parameters
    ----------
    fun: Callable(array-like[n_samples, n_features]) -> float
        expensive function that is adaptive sampled, fun should take array-like of
        shape=(n_samples, n_features) and return float
    adaptive_method: AdaptiveSampling
        strategy that is used to propose new sampling points
    n_calls: int
        max no. of iterations used
    callback: Callable
        callback is called within each iteration and should take the result dict as input
    eval_performance: bool
        evaluate the model performance based on a set of test samples
    true_fun: BenchmarkFun, optional (default=None)
        GT function used for evaluating model performance, this is useful if 'fun' is
        noisy, if bench_fun=None -> 'fun' is used for benchmark
    bench_sampling: Union[Literal["rnd", "lhs"], np.ndarray]
        method used to create test samples for evaluating the model performance,
        array can be passed with test samples instead of using a sampling method
    verbose: bool, optional (default=True)
        print progress and information
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        adaptive_method: AdaptiveSampling,
        n_calls: int,
        callback: Callable = None,
        copy_x: bool = True,
        eval_performance: bool = False,
        true_fun: Callable = None,
        bench_sampling: Union[Literal["rnd", "lhs"], np.ndarray] = "lhs",
        verbose: bool = True,
    ):

        assert n_calls >= 1

        # define adaptive sampling method
        self._adaptive_method: AdaptiveSampling = adaptive_method
        self._callback: Optional[Callable] = callback

        self._fun: Callable[
            [np.ndarray], float
        ] = fun  # call to function that is efficiently learned
        self.verbose: bool = verbose

        # if True -> next_x is always evaluated at same position, if False it is assumed that it can't be ensured
        # that x_next = x_observed -> fun should return y_obs together with x_obs is such cases
        self._copy_x: bool = copy_x

        # parameters
        self._iter_count: int = 0
        self.n_max: int = int(n_calls)  # max no. of calls
        self.n_init: int = self._adaptive_method.n_init
        self._next_x = None

        # set max number of calls
        self._adaptive_method.n_max = int(n_calls)

        if self.n_init >= self.n_max:
            raise ValueError(
                "SeqEd - Initial sampling number should be lower than max number of iterations "
                "(n_init > n_max)"
            )

        self.bounds = self._adaptive_method.bounds
        self.f = None

        # tracking of results
        self._result = dict()
        self._ask_time = list()  # time it takes to propose a new point
        self._tell_time = list()  # time it takes to process the observed data
        self._fun_time = list()  # time it takes to evaluate the expensive function

        # benchmark & metrics
        self.eval_performance: bool = eval_performance
        self.benchmark: Optional[Callable] = true_fun

        # sampling
        if isinstance(bench_sampling, np.ndarray):
            # sampling points are directly passed
            self._bench_sampling: np.ndarray = bench_sampling
        elif bench_sampling in ["rnd", "lhs"]:
            self._bench_sampling: Literal[
                "rnd", "lhs"
            ] = bench_sampling  # one of "rnd", "lhs"
        else:
            raise ValueError(
                "SeqED - Benchmark sampling method (bench_sampling) should be one of: 'rnd', 'lhs'"
            )

        self._bench_num_samples: int = 100000

        # list with metrics to evaluate, fun should take y_true and y_pred as input
        self._metrics = [
            {"name": "MSE", "fun": mean_squared_error, "args": None},
            {"name": "MAE", "fun": mean_absolute_error, "args": None},
            {"name": "R2", "fun": r2_score, "args": None},
            {"name": "MRE", "fun": mean_relative_error, "args": None},
            {"name": "SurrPred", "fun": lambda y_true, y_pred: y_pred, "args": None},
        ]
        self._metrics_res = [
            {"name": "MSE", "full_name": "Mean Square Error", "results": list()},
            {"name": "MAE", "full_name": "Mean Absolute Error", "results": list()},
            {"name": "R2", "full_name": "R2 Score", "results": list()},
            {"name": "MRE", "full_name": "Mean Relative Error", "results": list()},
            {
                "name": "SurrPred",
                "full_name": "Surrogate Prediction",
                "results": list(),
            },
        ]

    @property
    def param(self) -> dict:
        param = {
            "verbose": self.verbose,
            "iter_count": self._iter_count,
            "n_max": self.n_max,
            "n_init": self.n_init,
            "next_x": self._next_x,
            "benchmark": self.eval_performance,
            "bounds": self.bounds,
        }
        return param

    def run(
        self,
        n_calls: int = None,
        serializable_result: bool = False,
        path_save: str = None,
    ) -> dict:
        """
        Start the adaptive sampling process,
        in each iteration a new sampling point is proposed based on the adaptive method,
        then the expensive function is evaluated at this point,
        at last the result from the evaluated is returned to the adaptive method.

        Parameters
        ----------
        n_calls: int
            maximum number of evaluations used
        serializable_result: bool
            if True a serializable result is returned
        path_save: str, optional(default=None)
            path to save the model if it isn't serializable

        Returns
        -------
        result: dict
            results form the adaptive sampling process
        """
        if not isinstance(n_calls, int):
            n_calls = self.n_max
        elif n_calls <= 0:
            n_calls = self.n_max

        if self.verbose:
            print(self._adaptive_method)

        # adaptive sampling loop
        for i in range(1, n_calls + 1):

            self._iter_count = i

            # 1 - propose new sampling point
            start = timer()
            self._next_x = self._adaptive_method.ask()  # get new condition to evaluate
            self._ask_time.append(timer() - start)  # keep track of ask time [s]

            # 2 - evaluate function that is learned
            start = timer()

            if self._copy_x:  # it is assumed that next_x = obs_x
                y_obs = self._fun(np.ascontiguousarray(self._next_x))
                x_obs = self._next_x
            else:
                y_obs, x_obs = self._fun(np.ascontiguousarray(self._next_x))

            self._fun_time.append(timer() - start)

            # 3 - return observation to adaptive sampling method
            start = timer()
            self._result, terminate = self._adaptive_method.tell(x_obs, y_obs)
            self._tell_time.append(timer() - start)  # keep track of ask time [s]

            # update result
            self._result["specs"]["SeqED"] = self.param
            self._result["ask_time"] = self._ask_time
            self._result["fun_time"] = self._fun_time
            self._result["tell_time"] = self._tell_time
            self._result["metrics"] = self._metrics_res

            # callback with results
            if self._callback is not None:
                self._callback(self._result)

            # conduct benchmark
            if (
                self.eval_performance and i >= self.n_init
            ):  # evaluate performance after init points are observed
                self._eval_model()

            # 4 - stop if termination condition is met
            if terminate:
                break

            # print progress
            if self.verbose:
                print_progress(i / self.n_max)

        return self.result(serializable_result, path_save)

    def _eval_model(self):
        """
        Keep track of improvement

        Used Metrics:
            - Mean Squared Error (MSE)
            - Mean Absolute Error (MAE)
            - R^2 Score
            - MRE Score
        """
        model = self._result.get("model", None)

        if model is not None:

            if isinstance(
                self._bench_sampling, np.ndarray
            ):  # sampling points are passed
                X_test = self._bench_sampling
            elif isinstance(self._bench_sampling, str):
                X_test = sampling(
                    self.bounds,
                    method=self._bench_sampling,
                    n_samples=self._bench_num_samples,
                )
            else:
                raise ValueError(
                    "SeqED - Benchmark sampling method (bench_sampling) should be one of: "
                    "'rnd', 'lhs'"
                )

            # evaluate true function
            if self.benchmark is not None:
                y_true = self.benchmark.run(X_test)  # eval benchmark
                y_min = self.benchmark.ymin  # get GT y_min
            else:
                y_true = self._fun(X_test)
                y_min = None

            y_pred = model.predict(X_test)

            bench_call_out = dict()
            for metric, metric_res in zip(
                self._metrics, self._metrics_res
            ):  # iterate over specified metrics
                # calc metric
                if metric["name"] == "MRE" and y_min is not None:
                    metric_val = metric["fun"](y_true, y_pred, y_min)
                elif metric["name"] == "SurrPred":
                    metric_val = metric["fun"](y_true, y_pred)
                    metric_val = metric_val.astype(np.float32).flatten()
                else:
                    metric_val = metric["fun"](y_true, y_pred)

                # print metric
                if self.verbose and metric["name"] != "SurrPred":
                    print(
                        "%s in iteration %i: %.2f"
                        % (metric["name"], self._iter_count, metric_val)
                    )

                # append data
                metric_res["results"].append(metric_val)

                # output for bench call
                bench_call_out[metric_res["name"]] = metric_val

    def result(self, serializable: bool = False, path: str = None) -> dict:
        """
        Return result from last run
        """
        if self._adaptive_method.surrogate_name == "BNN":

            res = self._result

            if serializable:  # return serializable result object
                model = res.pop("model")
                bnn_param, save_path = model.save(path)

                res["model"] = {
                    "bnn_param": bnn_param,
                    "bnn_save_path": save_path,
                }
            return res

        else:
            return self._result.copy()
