"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Utilities
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, Literal, Optional, List, Dict, Tuple

import scipy.stats
from joblib import load, dump
from scipy import integrate

from gale._typing import ARRAY_LIKE_1D
from gale.experiments.plot import plot_quantiles, plt_saver, plot_summary
from gale.experiments.bench_functions import (
    benchmarks,
    BENCHMARKS_1D,
    BENCHMARKS_2D,
    BENCHMARKS_3D,
    BENCHMARKS_4D,
    BENCHMARKS_6D,
    BENCHMARKS_8D,
)


def _load_result(
    bench_name: str, path: str, model="GP", subfix="_update"
) -> tuple[dict, str]:
    # load result
    path_res = path + r"\Benchmark_%s_%s%s.pkl" % (bench_name, model, subfix)
    try:
        result = load(path_res)
    except FileNotFoundError:
        path_res = path + r"\Benchmark_%s_GP.pkl" % bench_name
        print("No update found, load default benchmark result: %s" % path_res)
        result = load(path_res)
    return result, path_res


def _extract_quantiles(data, quantiles=(0, 0.25, 0.5, 0.75, 1)):
    """
    calculate quantiles
    """
    return np.quantile(data, quantiles, axis=0)


def calc_integ_r2(
    r2: ARRAY_LIKE_1D, x: ARRAY_LIKE_1D = None, scale: bool = False
) -> float:
    r2 = np.ascontiguousarray(r2)

    if x is not None:
        x = np.ascontiguousarray(x)
        assert x.shape[0] == r2.shape[0]

    # set negative r2 values to 0
    r2[r2 < 0] = 0

    # perform rolling max
    r2 = np.maximum.accumulate(r2)

    if x is None:
        if scale:
            x = np.linspace(0, 1, len(r2))  # use equal spacing between 0 and 1
        else:
            x = np.arange(0, len(r2))  # use equal spacing

    return integrate.simpson(r2, x).item()


def _create_dfs(
    data,
    n_bins: Union[int, str] = "auto",
    n_init: int = 0,
    n_max: Optional[int] = None,
    reducer_bin: Literal["best", "max", "min", "mean", "median"] = "mean",
    metric_names: List[str] = ["MSE", "MAE", "R2", "MRE"],
    drop_lhs: bool = False,
    lhs_full_samples: bool = False,
):
    algos = list(data.keys())

    if drop_lhs:
        _ = algos.pop(algos.index("lhs_baseline"))

    q_dfs = {}
    iqm_dfs = {}

    all_dfs = pd.DataFrame(columns=["algo", "try", "sample"] + metric_names)

    # fix order
    alogs_order = [
        "GGESS",
        "EIGF",
        "GUESS",
        "wMMSE",
        "MMSE",
        "MASA",
        "MEPE",
        "TEAD",
        "DL_ASED",
        "LHS",
    ]
    algos: list = [alg_name for alg_name in alogs_order if alg_name in algos]

    for name in algos:  # iter over methods

        if (
            name == "lhs_baseline" and lhs_full_samples
        ):  # special treatment if lhs has samples from all iterations
            cur_algo_data = data[name]

            for lhs_rep in cur_algo_data:
                for metric in lhs_rep["metrics"]:
                    metric["results"] = metric["results"][
                        (n_init - 1) :
                    ]  # cut initial samples from lhs baseline

            cur_algo_data = cur_algo_data

        else:
            cur_algo_data = data[name]

        dfs = []
        for cur_data in cur_algo_data:
            df = pd.DataFrame({r["name"]: r["results"] for r in cur_data["metrics"]})
            df = df.drop(
                0
            )  # drop first row since the metrics should be identical over the methods

            if n_max is not None:  # limit to n_max iterations
                n_drop_last = (df.shape[0] + n_init) - n_max
                if n_drop_last > 0:
                    df.drop(
                        df.tail(n_drop_last).index, inplace=True
                    )  # drop last n rows
            dfs.append(df)

        # calculate quantiles
        vals = np.array([d[metric_names].values for d in dfs])
        q_dfs[name] = _extract_quantiles(vals)
        iqm_dfs[name] = scipy.stats.trim_mean(vals, proportiontocut=0.25, axis=0)

        for i, d in enumerate(dfs):
            d["algo"] = [name] * d.shape[0]
            d["try"] = int(i)
            d["sample"] = np.arange(d.shape[0]) + 1
            all_dfs = all_dfs.append(d, ignore_index=True)

    # try seaborn violin plot
    grouped = all_dfs.copy()

    # reducers used to merge samples within one bin
    if reducer_bin == "best":
        reducer_bin = []
        for m in metric_names:
            if m == "R2":
                reducer_bin.append("max")
            else:
                reducer_bin.append("min")
    else:
        reducer_bin = [reducer_bin] * len(metric_names)

    grp_by = "sample"
    reducers = {n: "first" for n in grouped.columns if n != grp_by}
    red_dict = dict(zip(metric_names + ["try"], reducer_bin + ["mean"]))
    reducers.update(red_dict)

    n_elem = q_dfs[list(q_dfs.keys())[0]].shape[1]

    # define num. of bins
    if n_bins == "auto":
        rest = n_elem % 3
        if rest == 0:
            bin_size = int(n_elem / 3)
        else:
            rest = n_elem % 4
            if rest == 0:
                bin_size = int(n_elem / 4)
            else:
                bin_size = int(n_elem / 5)

        bins = np.arange(n_init, n_init + n_elem + 1, bin_size)

    else:  # fixed number of bins -> rest is put into first bin
        assert 0 < n_bins <= n_elem

        bin_size = int(n_elem // n_bins)

        rest = n_elem % n_bins
        first_bin_size = bin_size + rest

        # define bins
        bins = np.arange(n_init + first_bin_size, n_init + n_elem + 1, bin_size)
        bins = np.concatenate([[n_init], bins])

    bined = pd.cut(all_dfs[grp_by] + n_init, bins, right=True)

    grouped = (
        grouped.groupby(["algo", "try", bined], sort=False)
        .agg(reducers)
        .reset_index(["algo", "try"], drop=True)
        .reset_index()
    )

    # sort grouped
    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(algos, range(len(algos))))

    # Generate a rank column that will be used to sort
    grouped["algo_rank"] = grouped["algo"].map(sorterIndex)
    grouped = grouped.sort_values(["algo_rank", "try"])
    grouped = grouped.drop(labels="algo_rank", axis=1)  # drop rank column
    grouped = grouped.reset_index(drop=True)

    return q_dfs, grouped, iqm_dfs


def create_lhs_baseline(
    bench_name: str,
    n_max: int,
    repetitions: int = 10,
    X_init_list: Optional[List[np.ndarray]] = None,
    X_test: np.ndarray = None,
    n_jobs: int = 1,
    path: str = None,
    seed: int = 10253,
) -> dict:
    """
    Create the LHS baseline result

    Parameters
    ----------
    bench_name: str

    n_max: int
        maximum number of samples for adaptive sampling
    repetitions: int, optional(default=None)
        num. of repetitions used if X_init_list is None, else num. of repetitions is matched with X_init_list
    X_init_list: List[np.ndarray], optional(default=None)
        prev. observations used in LHS, if None -> complete new LHS is made
    X_test: np.ndarray, optional(default=None)
        test samples, if None -> new samples are generated
    n_jobs: int, optional(default=1)
        parallelization
    path: str, optional(default=None)
        path to store results
    seed: int, optional(default=10253)


    Returns
    -------
    lhs_res: dict
        lhs_baseline results
    """
    raise NotImplementedError()


def _merge_update_results(
    paths: List[str],
    # bench_name: str,
    # path_testsamples: str,
    save: bool = False,
    methods_names: List[str] = None,
    # n_jobs: int = 1,
) -> dict:
    """
    Merge and update results
    """
    if (
        methods_names is not None
    ):  # name methods given as path if result file is not packed as dict
        assert len(methods_names) == len(paths)

    result: dict = {}

    for i, path in enumerate(paths):  # first path in paths should be main result
        print("Load: %s" % path)
        res_i = load(path)

        if (
            not isinstance(res_i, dict) and methods_names is not None
        ):  # use names for dict
            res_i = {str(methods_names[i]): res_i}
        result.update(res_i)

    # save result
    if save:
        new_path = paths[0][:-4] + "_update.pkl"
        dump(result, new_path, compress=7)

    return result


def _process_results(
    bench_name: str,
    n_init: int,
    result: dict,
    save_path: str,
    n_bins: Union[int, str] = "auto",
    limit_n_max: Optional[int] = None,
    reducer_bin: Literal["best", "max", "min", "mean", "median"] = "mean",
    save_table: bool = False,
):
    """
    Load and preprocess benchmark results
    """
    # create dfs
    q_dfs, grouped, iqm_dfs = _create_dfs(
        result,
        n_init=n_init,
        n_bins=n_bins,
        drop_lhs=False,
        reducer_bin=reducer_bin,
        n_max=limit_n_max,
    )

    metric_lookup = {0: "MSE", 1: "MAE", 2: "R2", 3: "MRE"}

    # save metric results after last iteration in table
    df_table = pd.DataFrame(columns=list(q_dfs.keys()))

    for key, values in q_dfs.items():
        # get median for metrics after last iteration
        df_table[key] = values[len(values) // 2, -1, :]

    df_table = df_table.rename(index=metric_lookup)

    if save_table:
        table_name = save_path + r"\MetricTable_%s.xlsx" % bench_name
        df_table.to_excel(table_name)

    return q_dfs, iqm_dfs, grouped, df_table, result


def create_plts(
    bench_name: str,
    save_path: str,
    bench_result: Optional[dict] = None,
    folder_path: Optional[str] = None,
    plt_metrics: List[str] = ["MSE", "MAE", "R2", "MRE"],
    save_plt: bool = False,
    save_as: tuple = ("jpg"),
    show: bool = True,
    bin_size: int = 5,
    save_table: bool = False,
    new_meth_names: dict = {
        "M_EIGF": "M-EIGF",
        "lhs_baseline": "LHS",
        "DL_ASED": "DL-ASED",
    },
):
    # get general information
    bench = benchmarks[bench_name]
    n_init: int = bench.dim * 10

    if bench_result is None:  # get benchmark from main
        bench_result, _ = _load_result(path=folder_path, bench_name=bench_name)

    # load and preprocess results
    q_dfs, iqm_dfs, grouped, df_table, result = _process_results(
        bench_name,
        result=bench_result,
        n_bins=bin_size,
        save_table=save_table,
        n_init=n_init,
        save_path=save_path,
    )

    m_lookup = {"MSE": 0, "MAE": 1, "R2": 2, "MRE": 3}

    # plot metrics
    for metric_name in plt_metrics:

        assert metric_name in m_lookup.keys()

        m_idx = m_lookup[metric_name]

        if metric_name == "R2":
            rolling = "max"
        else:
            rolling = "min"

        params = {
            "baseline_vals": None,
            "grouped_df": grouped,
            "metric_idx": m_idx,
            "start": n_init,
            "show": show,
            "logscale": False,
            "metric_name": metric_name,
            "rolling": rolling,
            "new_meth_names": new_meth_names,
        }

        fig = plot_quantiles(q_dfs, **params)
        if save_plt:
            plt_name = r"\Metric_%s_%s_median" % (bench_name, metric_name)
            plt_saver(fig, save_path, plt_name, types=save_as)
        plt.close(fig)


def create_agg_plots(
    main_dict: dict,
    benchs: List[str],
    save_path: str,
    agg_metrics: Union[str, List[str]] = "R2",
    save_plt: bool = False,
    save_as: tuple = ("jpg"),
    show: bool = True,
    bin_size: int = 5,
):
    """
    Create plot with aggregated results over given benchmarks (same dimension)
    """
    reducer_bin: Literal["best", "max", "min", "mean", "median"] = "mean"
    m_lookup = {"MSE": 0, "MAE": 1, "R2": 2, "MRE": 3}

    result = {}
    prev_dim: Optional[int] = None

    if not isinstance(agg_metrics, list):
        agg_metrics = [agg_metrics]

    for bench_name in benchs:

        if prev_dim is None:
            prev_dim = benchmarks[bench_name].dim

        dim: int = benchmarks[bench_name].dim

        if prev_dim != dim:
            raise ValueError("Benchmark dimensions don't match!")

        print("Processing: %s" % bench_name)
        result_benchmark: dict = main_dict[bench_name]

        for method, val in result_benchmark.items():
            prev_res = result.get(method, [])
            result[method] = prev_res + list(val)

        prev_dim = benchmarks[bench_name].dim

    # get general information
    n_init = prev_dim * 10

    # process all benchmarks
    q_dfs, iqm_dfs, grouped, *_ = _process_results(
        "",
        n_init=n_init,
        result=result,
        n_bins=bin_size,
        save_table=False,
        reducer_bin=reducer_bin,
        limit_n_max=250,
        save_path=save_path,
    )

    for agg_metric in agg_metrics:

        assert agg_metric in m_lookup.keys()

        m_idx = m_lookup[agg_metric]  # get id

        # plt
        magnified_scaler = 0.4
        n_x_zoom = 20
        magnified = False
        if agg_metric == "R2":
            rolling = "max"
            math_metric_name = r"$R^2$"
        else:
            rolling = "min"
            math_metric_name = None

        zoom_param = {}

        new_meth_names = {
            "M_EIGF": "M-EIGF",
            "lhs_baseline": "LHS",
            "DL_ASED": "DL-ASED",
            "M_GGESS": "M-GGESS",
        }

        params = {
            "baseline_vals": None,
            "grouped_df": grouped,
            "metric_idx": m_idx,
            "start": n_init,
            "show": show,
            "logscale": False,
            "metric_name": agg_metric,
            "rolling": rolling,
            "new_meth_names": new_meth_names,
            "math_metric_name": math_metric_name,
            "show_outliers": False,
            "whiskers": 0,
            "magnified": magnified,
            "magnified_scaler": magnified_scaler,
            "n_x_zoom": n_x_zoom,
            "n_x_zoom_end": None,
        }

        params.update(zoom_param)

        fig = plot_quantiles(q_dfs, **params)
        if save_plt:
            plt_name = r"\Metric_%iD_%s_aggregated_%s" % (dim, agg_metric, reducer_bin)
            plt_saver(fig, save_path, plt_name, types=save_as)
        plt.close(fig)

        fig = plot_quantiles(iqm_dfs, first_measure="IQM", **params)
        if save_plt:
            plt_name = r"\Metric_%iD_%s_aggregated_iqm_%s" % (
                dim,
                agg_metric,
                reducer_bin,
            )
            plt_saver(fig, save_path, plt_name, types=save_as)
        plt.close(fig)


def get_bench_dim(bench_name: str) -> int:
    if bench_name in BENCHMARKS_1D.keys():
        return 1
    if bench_name in BENCHMARKS_2D.keys():
        return 2
    if bench_name in BENCHMARKS_3D.keys():
        return 3
    if bench_name in BENCHMARKS_4D.keys():
        return 4
    if bench_name in BENCHMARKS_6D.keys():
        return 6
    if bench_name in BENCHMARKS_8D.keys():
        return 8


def create_summary(
    benchs: list,
    save_path: str,
    path_global_agg_res: Optional[str] = None,
    main_dict: Optional[dict] = None,
    n_max_limit: Optional[Dict[str, int]] = None,
    save_table: bool = False,
    save_rank: bool = False,
    save_plt: bool = False,
    save_pkl: bool = False,
):
    """
    Create summary
    """
    m_lookup = {"MSE": 0, "MAE": 1, "R2": 2, "MRE": 3}
    reducer_iter = [np.min, np.min, np.max, np.min]

    if path_global_agg_res is not None:

        print("Load global aggregated result: %s" % path_global_agg_res)
        result = load(path_global_agg_res)

        # filter benchmarks
        for i, metric_result in enumerate(result):
            use_idx = metric_result["bench"].isin(benchs)
            result[i] = metric_result.loc[use_idx]

    else:
        print("Create global aggregated result from main dict")
        result = [pd.DataFrame(columns=["method", "bench", "run", "value"])] * 4

        for bench_name in benchs:  # iter over benchmarks

            print("Processing: %s" % bench_name)
            if main_dict is not None:  # get benchmark from main
                res_bench: dict = main_dict[bench_name]
            else:
                raise ValueError("'main_dict' not given.")

            for method, val in res_bench.items():  # iter over methods

                method = str(method).replace(
                    "_", "-"
                )  # replace because of latex backend

                for i, repetition in enumerate(val):  # iter over repetitions

                    n_dim: int = repetition["obs_cond"].shape[1]

                    for m_name, m_id in m_lookup.items():  # iter over metrics

                        metric_results = repetition["metrics"][m_id]["results"]

                        if n_max_limit is not None:
                            n_max = n_max_limit.get(str(n_dim), None)
                            if n_max is not None:  # use only until n_max_limit samples
                                n_samples: int = repetition["obs_y"].shape[0]
                                n_del: int = n_samples - n_max
                                if n_del > 0:
                                    del metric_results[-n_del:]

                        best_result = reducer_iter[m_id](metric_results)

                        best_res_dict = {
                            "method": method,
                            "bench": bench_name,
                            "run": i,
                            "value": best_result,
                        }

                        if m_id == m_lookup["R2"]:
                            # calc R2 area and append
                            best_res_dict.update(
                                {"r2_area": calc_integ_r2(metric_results)}
                            )
                            best_res_dict.update(
                                {
                                    "r2_area_scaled": calc_integ_r2(
                                        metric_results, scale=True
                                    )
                                }
                            )

                        # add result
                        result[m_id] = result[m_id].append(
                            best_res_dict, ignore_index=True
                        )

    # save result
    if save_pkl:
        path_pkl = save_path + r"\all_results_agg_test.pkl"
        dump(result, path_pkl)

    # create ranking
    r2_res = result[2]  # use R^2 metric
    # rank methods in each benchmark run
    r2_res["rank"] = r2_res.groupby(["run", "bench"])["value"].rank(
        "dense", ascending=False
    )
    r2_res["rank_area"] = r2_res.groupby(["run", "bench"])["r2_area_scaled"].rank(
        "dense", ascending=False
    )

    # sum rank for methods (1 to n_methods)
    rank_df = []
    group_by = r2_res.groupby("method")
    for name_ in ["rank", "rank_area"]:
        rank_sum = group_by.sum(numeric_only=True).rename(
            {name_: "%s_sum" % name_}, axis=1
        )["%s_sum" % name_]
        rank_mean = group_by.mean(numeric_only=True).rename(
            {name_: "%s_mean" % name_}, axis=1
        )["%s_mean" % name_]
        n_ = r2_res[r2_res["method"] == "GUESS"].shape[0]
        # standard error
        rank_se = (group_by.std(numeric_only=True) / np.sqrt(n_)).rename(
            {name_: "%s_se" % name_}, axis=1
        )["%s_se" % name_]
        rank_df += [rank_sum, rank_mean, rank_se]

    # concat dfs
    rank_df = pd.concat(rank_df, axis=1).sort_values("rank_sum")

    # create ranking for each benchmark
    group_by = r2_res.groupby(["bench", "method"])
    rank_bench_mean: pd.DataFrame = group_by.mean(numeric_only=True).rename(
        {"rank": "rank_mean"}, axis=1
    )["rank_mean"]
    n_ = 10  # repetitions per benchmark
    rank_bench_se: pd.DataFrame = (
        group_by.std(numeric_only=True) / np.sqrt(n_)
    ).rename({"rank": "rank_se"}, axis=1)["rank_se"]
    rank_bench_df: pd.DataFrame = pd.concat([rank_bench_mean, rank_bench_se], axis=1)

    if save_rank:
        table_name = save_path + r"\summary_ranking.xlsx"
        rank_bench_df.to_excel(table_name)

    # get quantiles
    qs = [0, 0.25, 0.5, 0.75, 1]  # quantiles to extract
    res_list = []

    def trim_mean(df: pd.DataFrame, q_=0.25) -> float:
        return float(scipy.stats.trim_mean(a=df.values, proportiontocut=q_, axis=0))

    # iter over metrics
    for metric_name, metric_id in m_lookup.items():
        metric_agg_df = result[metric_id]

        for q in qs:  # calc quantiles
            res_list.append(
                metric_agg_df.groupby(["method"])["value"]
                .quantile(q=q)
                .rename("%s_%s" % (metric_name, str(q)))
            )

        # sample mean
        mean = (
            metric_agg_df.groupby(["method"])["value"]
            .mean()
            .rename("%s_mean" % metric_name)
        )
        # calculate the standard error of the sampling distribution of the sample mean
        n_samples = metric_agg_df[metric_agg_df["method"] == "GUESS"].shape[0]
        std = metric_agg_df.groupby(["method"])["value"].std()
        se = std / np.sqrt(n_samples)
        se = se.rename("%s_se" % metric_name)
        std = std.rename("%s_std" % metric_name)

        iqm = metric_agg_df.groupby(["method"])["value"].apply(trim_mean)
        iqm = iqm.rename("%s_iqm" % metric_name)

        if metric_name == "R2":
            r2_area = metric_agg_df.groupby(["method"])["r2_area_scaled"].mean()
            r2_area_se = metric_agg_df.groupby(["method"])[
                "r2_area_scaled"
            ].std() / np.sqrt(n_samples)
            r2_area_se = r2_area_se.rename("r2_area_scaled_se")
            r2_area_iqm = metric_agg_df.groupby(["method"])["r2_area_scaled"].apply(
                trim_mean
            )
            r2_area_iqm = r2_area_iqm.rename("r2_area_scaled_iqm")

            res_list.append(r2_area)
            res_list.append(r2_area_se)
            res_list.append(r2_area_iqm)

        res_list.append(se)
        res_list.append(std)
        res_list.append(mean)
        res_list.append(iqm)

    result_grouped = pd.concat(res_list, axis=1)

    # append ranks
    result_grouped = pd.concat([result_grouped, rank_df], axis=1)

    # save result grouped_df as excel
    if save_table:
        table_name = save_path + r"\summary_global_agg_test.xlsx"
        result_grouped.to_excel(table_name)

    # create plots
    if save_plt:
        res = r2_res
        rename = {"value": "$R^2$", "r2_area_scaled": "$R^2_{Area}$", "rank": "Rank"}

        fig = plot_summary(
            res,
            x=["value", "r2_area_scaled"],
            y="method",
            rename_ys={"lhs-baseline": "LHS"},
            rename_xs=rename,
            whiskers=0,
            show=False,
            show_points=True,
            show_outliers=False,
            x_limits=[(0.0, 1.0), (0.0, 1.0)],
        )

        plt_name = r"\plot_summary"
        plt_saver(fig, save_path, plt_name, types=("jpg", "pdf"))
        plt.close(fig)

    return result, rank_bench_df, result_grouped


def concat_results(
    main_path: str,
    path_results: str,
    add_bench: Optional[List[str]] = None,
    model: str = "GP",
    suffix: Union[List[str], Tuple[dict, ...]] = ("",),
    save_individual: bool = False,
    save_main_dict: bool = True,
    # create_new_main: bool = False,
    methods_names: List[str] = None,
    filter_strategies: List[str] = None,
) -> dict:
    """
    Preprocess and concatenate experimental results into one dict, default behaviour is to load
    and update a existing one. Pass `create_new_main=True` to create a new one instead.

    Parameters
    ----------
    main_path: str
        path to concatenated exp. results, can be updated with new experiments if file is existing
    path_results: str
        folder where individual exp. results are stored
    add_bench: List[str], optional(default=None)
        determine which benchmarks are added/updated
    model: str
        model name
    suffix: Union[List[str], List[dict]]
        subfix used in the individual result files
    save_individual: bool, optional(default=False)
        save processed results for each benchmark
    save_main_dict: bool, optional(default=True)
        save the created/updated dict containing the experiments
    create_new_main: bool, optional(default=False)
        create new main dict containing the experiments, instead of updating a existing one
    methods_names: List[str], optional(default=None)
        use given str as dict keys

    Returns
    -------
    main_dict: dict
        dict with results
    """

    if os.path.isfile(main_path):
        main_dict: dict = load(main_path)
    else:
        main_dict = {}
        # raise ValueError("given path not valid")

    if add_bench:
        for bench_name in add_bench:

            paths: list = []
            for sub in suffix:  # get paths to data
                if isinstance(sub, dict):

                    try_path = path_results + r"\Benchmark_%s_%s%s.pkl" % (
                        bench_name,
                        model,
                        sub["try"],
                    )

                    if os.path.isfile(try_path):
                        paths.append(try_path)
                    else:
                        except_path = path_results + r"\Benchmark_%s_%s%s.pkl" % (
                            bench_name,
                            model,
                            sub["except"],
                        )
                        paths.append(except_path)
                else:
                    paths.append(
                        path_results
                        + r"\Benchmark_%s_%s%s.pkl" % (bench_name, model, sub)
                    )

            # load and merge results, update lhs (if wanted)
            bench_res = _merge_update_results(
                paths,
                save=save_individual,
                methods_names=methods_names,
            )

            # drop not needed information
            for method_name, method in bench_res.items():
                for rep in method:
                    for key in [
                        "model",
                        "conditions",
                        "transformer",
                        "desc",
                        "specs",
                        "optimizer",
                    ]:
                        rep.pop(key, None)
                    if len(rep["metrics"]) == 5:
                        rep["metrics"].pop(4)  # drop y_prediction

            print("Update: %s in %s" % (bench_res.keys(), bench_name))
            curr_benchmark = main_dict.pop(str(bench_name), {})  # get old information
            curr_benchmark.update(bench_res)  # update method information
            main_dict[str(bench_name)] = curr_benchmark

        # save
        if save_main_dict is True:
            dump(main_dict, main_path, compress=2)

    if isinstance(filter_strategies, (list, tuple)):
        bench_names = list(main_dict.keys())
        for bench in bench_names:
            methods_in_bench = list(main_dict[bench].keys())
            for method in methods_in_bench:
                if method not in filter_strategies:
                    main_dict[bench].pop(method, None)
    return main_dict
