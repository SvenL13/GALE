"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Run Experiments
"""
import argparse
import json
import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from gale.doe import adaptive_sampling
from gale.experiments.bench_functions import (
    ALL_BENCHMARK_NAMES,
    ALL_BENCHMARK_NAMES_GROUPED,
    BENCHMARKS_1D,
    BENCHMARKS_2D,
    BENCHMARKS_3D,
    BENCHMARKS_4D,
    BENCHMARKS_6D,
    BENCHMARKS_8D,
    BENCHMARKS_HIGH,
)
from gale.experiments.experiment import ex
from gale.experiments.utils import (
    concat_results,
    create_agg_plots,
    create_plts,
    create_summary,
)

BENCHMARK_DIMS = Literal["1D", "2D", "3D", "4D", "6D", "8D", "High"]
BENCHMARK_CONFIG = Union[Literal["all"], List[BENCHMARK_DIMS], List[str]]


def _get_benchmark_config(
    config: BENCHMARK_CONFIG,
) -> Tuple[List[str], Optional[List[List[str]]]]:
    """
    Create list with benchmarks for experiment
    """
    lookup = {
        "1D": [*BENCHMARKS_1D],
        "2D": [*BENCHMARKS_2D],
        "3D": [*BENCHMARKS_3D],
        "4D": [*BENCHMARKS_4D],
        "6D": [*BENCHMARKS_6D],
        "8D": [*BENCHMARKS_8D],
        "High": [*BENCHMARKS_HIGH],
    }

    if config == "all":
        benchmarks: list = ALL_BENCHMARK_NAMES
        benchmarks_grouped: Optional[List[List[str]]] = ALL_BENCHMARK_NAMES_GROUPED

    elif isinstance(config, list):

        benchmarks = []
        benchmarks_grouped = []

        if (
            config[0] + "_norm" in ALL_BENCHMARK_NAMES
        ):  # is list of single benchmarks to conduct

            for c in config:
                if c + "_norm" in ALL_BENCHMARK_NAMES:
                    benchmarks.append(c + "_norm")
                else:
                    raise ValueError("Unknown benchmark given: %s" % c)

            benchmarks_grouped = None

        elif config[0] in lookup.keys():  # is list of benchmark dims

            for c in config:
                if c in [*lookup]:
                    benchmarks += lookup[c]
                    benchmarks_grouped.append(lookup[c])
                else:
                    raise ValueError("Unknown benchmark given: %s" % c)
        else:
            raise ValueError("Unknown benchmark given: %s" % config[0])
    else:
        raise ValueError(
            "Unknown benchmark given, should be 'all' or "
            "List of benchmarks/benchmark dims"
        )
    return benchmarks, benchmarks_grouped


def _get_paths(model: Literal["GP", "BNN", "SGP"], path: str = None) -> Tuple[str, str, str]:
    assert model in ("GP", "BNN", "SGP")

    # paths
    if path is None:
        path_base = (
            os.path.dirname(os.path.abspath(__file__)) + r"\results\%s" % model.lower()
        )
    else:
        path_base = path + r"\%s" % model.lower()

    # path where experimental results are stored
    path_exp = path_base + r"\experiments"
    # path to store plots, summary and intersections
    path_save = path_base + r"\plots"
    path_summary = path_base + r"\summary"

    for p in (path_base, path_exp, path_save, path_summary):
        if not os.path.exists(p):
            os.makedirs(p)

    return path_exp, path_save, path_summary


def _load_config(path_config: str) -> dict:
    with open(path_config) as f:
        config: dict = json.load(f)

    return config


def run_ex(
    model: Literal["GP", "BNN", "SGP"],
    bench_config: BENCHMARK_CONFIG,
    n_jobs: int,
    model_param: Optional[dict] = None,
    strategies: List[str] = "all",
    log_path: str = None,
    repetitions: int = 10,
    fit_model_every: int = 1,
    save_pred: bool = False,
    save_pred_sep: bool = False,
):
    assert model in ("GP", "BNN", "SGP")
    assert isinstance(save_pred, bool)
    assert isinstance(save_pred_sep, bool)
    assert isinstance(n_jobs, int)
    assert isinstance(repetitions, int) and repetitions > 0
    assert isinstance(fit_model_every, int) and fit_model_every > 0
    assert isinstance(strategies, list) or strategies == "all"
    if strategies == "all":
        strategies = [*adaptive_sampling]
    else:
        assert np.isin(strategies, [*adaptive_sampling]).all()

    benchmarks, _ = _get_benchmark_config(bench_config)
    path_exp, *_ = _get_paths(model, log_path)

    # run experiment
    for strategy in strategies:
        for bench_key in benchmarks:

            if model_param is not None:
                model_param_: Optional[dict] = model_param[
                    bench_key.replace("_norm", "")
                ]
            else:
                model_param_ = None

            ex.run(
                config_updates={
                    "bench_name": bench_key,
                    "model": model,
                    "model_param": model_param_,
                    "n_jobs": n_jobs,
                    "save_pred": save_pred,
                    "repetitions": repetitions,
                    "save_pred_sep": save_pred_sep,
                    "path_store": path_exp,
                    "adaptive_methods": [strategy],
                },
                meta_info={"comment": "NEW RUNS - Round 3"},
            )


def eval_ex(
    model: Literal["GP", "BNN", "SGP"],
    bench_config: BENCHMARK_CONFIG,
    n_jobs: int,
    strategies: List[str] = "all",
    log_path: str = None,
):
    assert model in ("GP", "BNN", "SGP")
    assert isinstance(n_jobs, int)
    assert isinstance(strategies, list) or strategies == "all"
    if strategies == "all":
        strategies = [*adaptive_sampling]
    else:
        assert np.isin(strategies, [*adaptive_sampling]).all()

    benchmarks, benchmarks_grouped = _get_benchmark_config(bench_config)

    path_exp, path_save, path_summary = _get_paths(model, log_path)

    path_all_results = path_exp + r"\all_results_%s.pkl" % model.lower()

    all_results = concat_results(
        main_path=path_all_results,
        add_bench=benchmarks,
        # add_bench=None,  # just load result
        suffix=["_" + s for s in strategies],
        path_results=path_exp,
        model=model,
        methods_names=strategies,
        filter_strategies=strategies,
    )

    # create individual plot for each benchmark
    for bench_name, bench_res in all_results.items():
        print("Plotting: %s" % bench_name)
        create_plts(
            bench_name=bench_name,
            bench_result=bench_res,
            plt_metrics=["R2"],
            save_plt=True,
            show=False,
            save_table=False,
            save_path=path_save,
        )

    if benchmarks_grouped is not None:
        # create aggregated plots
        for bench in benchmarks_grouped:
            create_agg_plots(
                benchs=bench,
                main_dict=all_results,
                save_plt=True,
                show=False,
                save_as=("jpg", "pdf"),
                agg_metrics=["R2"],
                save_path=path_save,
            )

    create_summary(
        benchs=benchmarks,
        main_dict=all_results,
        save_table=False,
        save_rank=False,
        save_pkl=False,
        save_plt=True,
        n_max_limit={"4": 250, "6": 250, "8": 250},
        save_path=path_summary,
    )


def main(
    model: Literal["GP", "BNN", "SGP"],
    benchmark: BENCHMARK_CONFIG = None,
    strategies: List[str] = "all",
    path_config: str = None,
    n_jobs: int = 1,
    run_experiment: bool = True,
    eval_experiment: bool = True,
    log_path: str = None,
):
    if path_config is not None:
        ex_config = _load_config(path_config)
    else:  # load default
        p_ = os.path.dirname(os.path.abspath(__file__))
        if model == "GP":
            ex_config = _load_config(p_ + "./config/ex_config_gp.json")
        elif model == "SGP":
            ex_config = _load_config(p_ + "./config/ex_config_sgp.json")
        elif model == "BNN":
            ex_config = _load_config(p_ + "./config/ex_config_bnn.json")
        else:
            raise ValueError("Model should be 'GP' or 'BNN'.")

    bench_config = ex_config.pop("bench_config")

    ex_config["eval"]["run_lhs_baseline"] = False

    if benchmark is not None:
        bench_config = benchmark

    if run_experiment:
        run_ex(
            model=model,
            strategies=strategies,
            log_path=log_path,
            bench_config=bench_config,
            n_jobs=n_jobs,
            **ex_config["ex"]
        )

    if eval_experiment:
        eval_ex(
            model=model,
            strategies=strategies,
            log_path=log_path,
            bench_config=bench_config,
            n_jobs=n_jobs,
            **ex_config["eval"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model: select from [GP, BNN]"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        nargs="+",
        default=None,
        help="Benchmark: override benchmarks from the config file, see description",
    )
    parser.add_argument(
        "-p", "--parallel", type=int, default=1, help="Maximum number of running jobs"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Exp. config file to override default, see description",
    )
    parser.add_argument(
        "-logdir",
        type=str,
        default=None,
        help="Directory to which results will be logged (default: ./log)",
    )
    args = parser.parse_args()

    main(
        model=args.model,
        benchmark=args.benchmark,
        path_config=args.config,
        n_jobs=args.parallel,
        log_path=args.logdir,
    )
