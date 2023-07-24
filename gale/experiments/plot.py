"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Plot
"""
import string
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from seaborn import color_palette

# set font
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# set font sizes
SIZE_DIFFERENCE = 2
SMALL_SIZE = 14
MEDIUM_SIZE = SMALL_SIZE + SIZE_DIFFERENCE
BIGGER_SIZE = MEDIUM_SIZE + SIZE_DIFFERENCE

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # font size of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend font size
plt.rc("figure", titlesize=BIGGER_SIZE)  # font size of the figure title


def plt_saver(fig, path: str, name: str, types="jpg", dpi=600):
    if not isinstance(types, (list, tuple)):
        types = [types]

    for t in types:
        if t == "pgf":
            matplotlib.use("pgf")
            matplotlib.rcParams.update(
                {
                    "pgf.texsystem": "pdflatex",
                    "font.family": "serif",
                    "text.usetex": True,
                    "pgf.rcfonts": False,
                }
            )
            fig.savefig(path + name + ".pgf")
        elif t == "eps":
            fig.savefig(path + name + ".eps", format="eps")
        else:
            fig.savefig(path + name + ".%s" % t, dpi=dpi)


def plot_summary(
    df: pd.DataFrame,
    x: Union[str, List[str]],
    y: str,
    palette="muted",
    rename_ys: dict = None,
    rename_xs: dict = None,
    whiskers: float = 1.5,
    x_limits: Union[Tuple[float, float], List[Tuple[float, float]]] = None,
    show_outliers: bool = True,
    show_points: bool = True,
    show: bool = True,
):

    rename_xs = {} if rename_xs is None else rename_xs

    if rename_ys is not None:
        df = df.copy()
        df = df.replace({y: rename_ys})

    if isinstance(x, str):
        x = [x]

    fig, axs = plt.subplots(figsize=(15, 6), ncols=len(x))

    for i, x_ in enumerate(x):

        ax = axs if len(x) == 1 else axs[i]

        sns.boxplot(
            x=x_,
            y=y,
            data=df,
            saturation=1,
            whis=whiskers,
            showfliers=show_outliers,
            hue=y,
            width=0.6,
            # width=1.2,
            dodge=False,
            palette=palette,
            ax=ax,
        )

        # Add in points to show each observation
        if show_points:
            sns.stripplot(
                x=x_,
                y="method",
                data=df,
                size=4,
                color=".2",
                alpha=0.3,
                linewidth=0,
                ax=ax,
            )

        # Tweak the visual presentation
        minor_ticks = np.arange(0, 1, 0.05)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.grid(True)
        ax.grid(True, "minor", "x", color="k", alpha=0.2, lw=0.4)
        ax.set_yticks([])

        ax.set(ylabel="")
        ax.set(xlabel="%s" % rename_xs.get(x_, x_))

        if x_limits is not None:
            ax.set_xbound(x_limits[i])

        sns.despine(trim=True, left=True)

        # remove legend
        ax.legend([], [], frameon=False)

        ax.set_title(
            "(%s) %s - IQD" % (string.ascii_lowercase[i], rename_xs.get(x_, x_))
        )

    # create legend
    lines, labels = axs[0].get_legend_handles_labels()
    lines = lines.copy()
    legend = fig.legend(
        lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    if show:
        plt.show()
    return fig


def plot_quantiles(
    df_map: dict,
    baseline_vals,
    grouped_df,
    metric_idx=0,
    logscale=False,
    plot_minmax=False,
    start=0,
    show=True,
    palette="muted",
    metric_name="",
    math_metric_name: str = None,
    new_meth_names: dict = None,
    rolling="min",
    y_bound: tuple = None,
    show_outliers: bool = True,
    whiskers: float = 1.5,
    magnified: bool = False,
    magnified_scaler: float = 0.4,
    n_x_zoom: int = None,
    n_x_zoom_end: int = None,
    first_measure: str = "Median",
):
    if math_metric_name is None:
        math_metric_name = metric_name

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # check if single measure is given
    is_single = True if len(list(df_map.values())[0].shape) == 2 else False

    bias = 0
    if logscale:
        bias = _get_min(df_map, metric_idx) - 0.001

    # Inset magnified plot
    if n_x_zoom is None:
        n_x_zoom: int = 20  # iterations shown on zoomed axis
    if n_x_zoom_end is not None:
        n_x_zoom_end -= start
    if is_single:
        n_x: int = list(df_map.values())[0].shape[
            0
        ]  # num. iterations shown in big plot
    else:
        n_x: int = list(df_map.values())[0].shape[
            1
        ]  # num. iterations shown in big plot
    if magnified:
        # determine zoom
        zoom = 1 / (n_x_zoom / n_x)
        axins = zoomed_inset_axes(ax[0], magnified_scaler * zoom, loc=4, borderpad=2)
    else:
        axins = None

    # color depending on which iteration new samples are created
    colors = color_palette(palette, n_colors=len(df_map.keys()))
    # switch 9 and 10-th color
    if len(colors) >= 9:
        c = color_palette(palette, n_colors=10)
        colors[8] = c[9]
        if len(colors) >= 10:
            colors[9] = c[8]
    y_zoom_lower, y_zoom_upper = None, None  # values for zoom y_bounds
    qs, xaxis = None, None

    for method_name, quants, c in zip(
        df_map.keys(), df_map.values(), colors
    ):  # iter over methods

        if new_meth_names is not None:
            if method_name in new_meth_names.keys():
                method_name = new_meth_names[method_name]  # replace with new name

        qs = quants[..., metric_idx]
        if is_single:
            xaxis = np.arange(qs.shape[0]) + 1 + start
        else:
            xaxis = np.arange(qs.shape[1]) + 1 + start

        if rolling == "max":
            acc = np.maximum.accumulate
        elif rolling is None:
            acc = np.array
        else:  # use min accumulation
            acc = np.minimum.accumulate

        if is_single:
            y_val = acc(qs - bias)
        else:
            y_val = acc(qs[len(qs) // 2] - bias)

        p = ax[0].plot(xaxis, y_val, label=method_name, c=c)  # take only min result

        if axins is not None:
            _ = axins.plot(xaxis, y_val, c=c)  # plot on magnified axis
            if n_x_zoom_end is not None:  # end point of zoom
                n_x_zoom_start = n_x_zoom + len(y_val) - n_x_zoom_end
                y_val_zoom = y_val[
                    -n_x_zoom_start:n_x_zoom_end
                ]  # get only values in zoom range
            else:
                y_val_zoom = y_val[-n_x_zoom:]  # get only values in zoom range

            y_val_min: float = np.min(y_val_zoom)
            y_val_max: float = np.max(y_val_zoom)

            if y_zoom_lower is None:
                y_zoom_lower = y_val_min
            elif y_val_min < y_zoom_lower:
                y_zoom_lower = y_val_min
            if y_zoom_upper is None:
                y_zoom_upper = y_val_max
            elif y_val_max > y_zoom_upper:
                y_zoom_upper = y_val_max

        if plot_minmax and not is_single:
            _ = ax[0].plot(
                xaxis,
                acc(qs[0] - bias),
                color=p[0].get_color(),
                linestyle=":",
                label="_nolegend_",
                alpha=0.5,
            )
            _ = ax[0].plot(
                xaxis,
                acc(qs[-1] - bias),
                color=p[0].get_color(),
                linestyle=":",
                label="_nolegend_",
                alpha=0.5,
            )

    if baseline_vals is not None:
        ax[0].plot(
            xaxis,
            np.ones_like(xaxis) * baseline_vals[metric_idx] - bias,
            "--k",
            label="lhs",
        )

    colors_boxplot = {key: c_i for key, c_i in zip(df_map.keys(), colors)}

    # boxplot
    _ = sns.boxplot(
        data=grouped_df,
        x="sample",
        y=metric_name,
        hue="algo",
        ax=ax[1],
        saturation=1,
        showfliers=show_outliers,
        whis=whiskers,
        palette=colors_boxplot,
    )
    ax[1].legend([], [], frameon=False)  # remove legend

    if axins is not None:  # set axis of zoomin
        if n_x_zoom_end is not None:  # end point of zoom
            axins.set_xlim((n_x_zoom_end + start) - n_x_zoom, n_x_zoom_end + start)
        else:
            axins.set_xlim(xaxis[-n_x_zoom], xaxis[-1])
        axins.set_ylim(
            y_zoom_lower - y_zoom_lower * 0.02, y_zoom_upper + y_zoom_upper * 0.01
        )
        mark_inset(
            ax[0], axins, loc1=2, loc2=4, fc="none", ec="0.5"
        )  # loc: 1 ur, 2 ul, 3 bl, 4 br

        # move axins upwards in figure
        pos1 = axins.get_position()  # get the original position
        pos1.y0 = pos1.y0 + 5
        axins.set_position(pos1)  # set a new position

        axins.grid(True, "major", "both", color="k", alpha=0.5, lw=0.6)
        axins.grid(True, "minor", "y", color="k", alpha=0.2, lw=0.4)
        axins.set_axisbelow(True)

    if logscale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

    # create legend
    lines, labels = ax[0].get_legend_handles_labels()
    lines = lines.copy()

    # plot legend below subplots
    legend = fig.legend(
        lines, labels, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=5
    )

    for line in legend.get_lines():
        line.set_linewidth(3)

    for axi in ax:
        axi.grid(True, "major", "both", color="k", alpha=0.5, lw=0.6)
        axi.grid(True, "minor", "y", color="k", alpha=0.2, lw=0.4)
        axi.set_axisbelow(True)

    upper = qs.shape[0] if is_single is True else qs.shape[1]
    ax[0].set_xbound(lower=start + 1, upper=upper + start)

    if y_bound is not None:
        ax[0].set_ybound(lower=y_bound[0], upper=y_bound[1])
        ax[1].set_ybound(lower=y_bound[0], upper=y_bound[1])

    # set labels
    ax[0].set_title("(a) %s - %s" % (math_metric_name, first_measure))
    ax[1].set_title("(b) %s - Average IQD" % math_metric_name)
    ax[0].set_xlabel("Sample")
    ax[1].set_xlabel("Sample")
    ax[0].set_ylabel(math_metric_name)
    ax[1].set_ylabel(math_metric_name)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    if show:
        plt.show()
    return fig


def _get_min(df_map, metric_idx):
    best = None
    for n, quants in df_map.items():
        cur = np.min(quants[0, ..., metric_idx])
        if best is None or cur < best:
            best = cur
    return best
