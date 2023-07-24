"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Main Experiment
"""
import os
import sys
from datetime import date
from typing import Tuple, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sacred import Experiment, host_info_gatherer
from sacred.observers import FileStorageObserver

# from sacred.observers import MongoObserver
from skopt.sampler import Lhs

from gale.doe import SeqED, adaptive_sampling, sampling
from gale.doe.space import Space
from gale.experiments.bench_functions import benchmarks
from gale.experiments.utils import create_lhs_baseline
from gale.utils import get_hardware


@host_info_gatherer("hardware_specs")
def hardware() -> dict:  # log used hardware for experiment
    return get_hardware()


def get_ex_folder(model: str, bench: str, base_path: str = None) -> str:
    start_date: str = date.today().strftime("%Y%m%d")

    # create experiment folder
    folder_name: str = start_date + "%s_%s" % (model, bench)
    if base_path is None:
        base_path: str = os.path.dirname(os.path.abspath(__file__))

    ex_results_folder = os.path.join(base_path, folder_name)

    if not os.path.exists(ex_results_folder):  # create folder if not existing
        os.makedirs(ex_results_folder)

    return ex_results_folder


# base directory
path_base: str = os.path.dirname(sys.path[0])

# create experiment
ex = Experiment(
    "Bench_GlobalSurr_GP",
    save_git_info=False,
    additional_host_info=[hardware],
    base_dir=path_base,
)

# add source files
for path, subdirs, files in os.walk(path_base):
    for name in files:
        f = os.path.join(path, name)
        _, ext = os.path.splitext(f)
        # only add python files
        if ext == ".py":
            ex.add_source_file(f)

# use local observer
obs_path = path_base + r"\sacred_experiment"
if not os.path.exists(obs_path):
    os.makedirs(obs_path)

ex.observers.append(FileStorageObserver(obs_path))


@ex.config
def config_adaptive():
    """
    Experiment configuration
    """
    # typing in combination with sacred observer causes error,
    # so we put typing in the comments
    model = "GP"  # [Literal["GP", "SGP", "BNN"]]
    model_param = None  # [Optional[dict]]

    bench_name = "GramLee_norm"  # [str]
    bench_dim = benchmarks[bench_name].dim

    path_store = None  # path to store experimental results [Optional[str]]

    settings = {
        1: (10, 40),
        2: (20, 140),
        3: (30, 180),
        4: (40, 250),
        6: (60, 250),
        8: (80, 250),
        16: (160, 1000),
        32: (320, 1200),
    }  # [Dict[int, Tuple[int, int]]]

    # set n_init and n_max depending on dimension
    n_init = settings[bench_dim][0]  # num. of initial samples created with lhs [int]
    n_max = settings[bench_dim][
        1
    ]  # max number of samples (init. sampling + adaptive sampling) [int]

    fit_model_every = 1  # fit model in every n-th iteration

    n_test = 100000  # num. of test samples generated for calculating metrics [int]

    adaptive_methods = [*adaptive_sampling]  # [List[str]]

    repetitions = 10  # repeat each experiment n times [int]
    seed = 1003

    save_pred = (
        False  # if False then prediction from surrogate model is not saved [bool]
    )
    save_pred_sep = False  # save y prediction information separate [bool]

    save_single_results = True  # save results from each method separate [bool]

    n_jobs = 1  # parallelization [int]


@ex.capture
def init_sampling(n_init: int, bench_name: str, _seed: int) -> np.ndarray:
    """
    Create initial samples for experiments
    """
    # get benchmark
    bench = benchmarks[bench_name]

    design = Space(bench.bounds)

    # create initial LHS design, X e R^(m x u)
    lhs = Lhs(criterion="maximin", iterations=1000)

    # generate samples
    samples = lhs.generate(design.transformed_bounds, n_init, random_state=_seed)

    return np.array(samples)


@ex.capture
def lhs_baseline(
    n_max: int,
    repetitions: int,
    bench_name: str,
    X_init_list: List[np.ndarray],
    X_test: np.ndarray,
    _seed: int,
    n_jobs: int,
) -> dict:
    lhs_res = create_lhs_baseline(
        bench_name=bench_name,
        n_max=n_max,
        X_init_list=X_init_list,
        X_test=X_test,
        n_jobs=n_jobs,
        repetitions=repetitions,
    )
    return lhs_res


@ex.capture
def test_adaptive(
    n_init: int,
    init_samples,
    n_max: int,
    bench_name: str,
    method: str,
    _seed: int,
    metric_samples: np.ndarray,
    model: str,
    model_param: dict,
    path_exp: str,
    iter: int = 1,
    fit_model_every_n: int = 1,
) -> Tuple[dict, list]:
    # get benchmark
    bench = benchmarks[bench_name]

    # Evaluate adaptive Method
    adaptive_method = adaptive_sampling[method](
        bounds=bench.bounds,
        n_init=n_init,
        rnd_state=_seed,
        verbose=True,
        init_sampling=init_samples,
        model=model,
        model_param=model_param,
        n_max=n_max,
        fit_model_every=fit_model_every_n,
    )
    seq_exp_design = SeqED(
        fun=bench.run,
        adaptive_method=adaptive_method,
        n_calls=n_max,
        eval_performance=True,
        bench_sampling=metric_samples,
        true_fun=bench,
    )

    # run calculation
    res = seq_exp_design.run(
        serializable_result=True,
        path_save=path_exp + r"\%s_model_%s_%s_%s" % (model, bench_name, method, iter),
    )

    # create artifacts to append
    artif = []

    return res, artif


def run_rep(
    i: int,
    init_samples_rep,
    method: str,
    n_init: int,
    n_max: int,
    seed: int,
    bench_name: str,
    metric_samples,
    model: str,
    path_exp: str,
    model_param: dict,
    fit_model_every: int = 1,
) -> Tuple[dict, list]:
    """
    Run one repetition
    """
    print("\n%s - Repetition: %i\n" % (method, i))
    res_i, artif_i = test_adaptive(
        method=method,
        iter=i,
        init_samples=init_samples_rep[i],
        n_init=n_init,
        n_max=n_max,
        _seed=seed,
        bench_name=bench_name,
        metric_samples=metric_samples,
        model=model,
        path_exp=path_exp,
        model_param=model_param,
        fit_model_every_n=fit_model_every,
    )
    return res_i, artif_i


@ex.automain
def main(
    adaptive_methods: list,
    repetitions: int,
    n_init: int,
    n_max: int,
    bench_name: str,
    _seed: int,
    n_jobs: int,
    model: str,
    n_test: int,
    save_single_results: bool,
    save_pred_sep: bool,
    save_pred: bool,
    path_store: str,
    model_param: dict,
    fit_model_every: int,
):
    if path_store == "" or path_store is None:
        path_store = get_ex_folder(model, bench_name)

    res_dict = dict()

    method_string: str = "".join(adaptive_methods)

    # create n_rep initial samples
    print("Create initial samples")
    init_samples_rep: list = []
    for i in range(repetitions):
        init_samples_rep.append(init_sampling())

    # create initial samples to evaluate metrics
    print("Create test samples")
    # seeds are set in sacred so this should give the same test samples
    X_test = sampling(benchmarks[bench_name].bounds, n_samples=n_test, method="lhs")

    # save as csv
    pkl_path = path_store + r"\TestSamples_%s.pkl" % bench_name
    df = pd.DataFrame(X_test)
    df.to_pickle(pkl_path)
    ex.add_artifact(pkl_path, name="TestSamples_%s" % bench_name)

    print("Start method evaluation")
    for method_name in adaptive_methods:  # iterate over methods

        # parallelize repetitions
        repetition_res = Parallel(n_jobs=n_jobs)(
            delayed(run_rep)(
                i,
                init_samples_rep,
                method_name,
                n_init,
                n_max,
                _seed,
                bench_name,
                X_test,
                model,
                path_store,
                model_param,
                fit_model_every,
            )
            for i in range(repetitions)
        )

        # unpack results
        results_method, artifacts_method = zip(*repetition_res)

        metrics = {}
        for i, (res_i, artif) in enumerate(
            zip(results_method, artifacts_method)
        ):  # repeat experiment i times

            for metric_i in res_i["metrics"]:
                prev = metrics.get(
                    metric_i["name"], list()
                )  # get previous entry if available
                metrics[metric_i["name"]] = prev + [metric_i["results"]]

            # append artifacts from repetitions
            for artif_i in artif:
                ex.add_artifact(artif_i)

        # save y_pred separately
        if save_pred_sep or not save_pred:
            all_dfs = []
            for i, meth_iter in enumerate(results_method):
                y_pred = meth_iter["metrics"].pop(4)
                df = pd.DataFrame(y_pred["results"]).astype("float32")
                df["try"] = i
                all_dfs.append(df)

            # concat dfs
            all_dfs = pd.concat(all_dfs, ignore_index=True)
            # save dfs
            if save_pred_sep:
                df_path = path_store + r"\ypreds_%s_%s_%s.pkl" % (
                    bench_name,
                    method_name,
                    model,
                )
                all_dfs.to_pickle(df_path)
                ex.add_artifact(
                    df_path, name="ypreds_%s_%s_%s" % (bench_name, method_name, model)
                )

        # append results to dict
        res_dict[str(method_name)] = results_method

        # save results for method
        if save_single_results:
            res_path = path_store + r"\Benchmark_%s_%s_%s.pkl" % (
                bench_name,
                model,
                method_name,
            )
            dump(res_dict[str(method_name)], res_path, compress=7)
            ex.add_artifact(
                res_path,
                name="Method_Result_%s_%s_%s"
                % (bench_name, method_name, method_string),
            )

        # calculate mean and std for metrics over repetitions and log them
        mean_list = []

        for metric_name, metric_values in metrics.items():

            # skip
            if metric_name in ["MRE", "MAE", "SurrPred"]:
                continue

            metric_values = np.array(metric_values)

            # calculate stats for metrics
            mean = metric_values.mean(axis=0)
            std = metric_values.std(axis=0)

            mean_list.append(mean)

            for i, (mean, std) in enumerate(zip(mean, std)):
                ex.log_scalar(
                    "Mean %s - %s" % (metric_name, method_name), mean, i + n_init + 1
                )

    # return res_dict
