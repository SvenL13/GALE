"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Utilities
"""
import platform
import sys
from pickle import load
from typing import Tuple, Union

import cpuinfo
import GPUtil
import numpy as np
import psutil
from scipy.spatial import distance

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D, BOUNDS


def split_bounds(bounds: BOUNDS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split bounds into upper and lower bounds

    Parameters
    ----------
    bounds: Bounds
        bounds to split

    Returns
    -------
    lower_bounds: np.ndarray
        lower bounds
    upper_bounds: np.ndarray
        upper bounds
    """
    upper_bounds: list = []
    lower_bounds: list = []

    for dim in bounds:  # iter over dimensions
        lower_bounds.append(dim[0])
        upper_bounds.append(dim[1])

    return np.array(lower_bounds), np.array(upper_bounds)


def closest_point(
    x: ARRAY_LIKE_1D,
    X: ARRAY_LIKE_2D,
    metric: str = "euclidean",
    return_dist: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Find the closest point x_i to a given point x from list of points X, based on given metric

    Parameters
    ----------
    x: np.ndarray, shape=(n_features)
        single point
    X: np.ndarray, shape=(n_samples, n_features)
        list of points
    metric: str
        The distance metric to use. see cdist docs for options, e.g. "euclidean", "minkowski", "mahalanobis", ...
    return_dist

    Returns
    -------
    x_close: np.ndarray
        closest point x_i in X to given x
    x_idx: np.ndarray with int
        index of closest point
    dist_min: float
    """
    x = np.atleast_2d(x)
    X = np.atleast_2d(X)
    dist = distance.cdist(x, X, metric=metric)
    closest_index = dist.argmin()

    if return_dist:
        return X[closest_index], closest_index, dist.min()
    else:
        return X[closest_index], closest_index


def scale_to_unity(x: ARRAY_LIKE_2D, bounds: BOUNDS = None) -> np.ndarray:
    """
    Scale x to unity [0, 1]
    If lw_bound and up_bound are specified these are used instead of lw_bound=min(x) and up_bound=max(x)

    Parameters
    ----------
    x: ARRAY_LIKE_2D
        input values 2d, shape=(samples, features)
    bounds: list or None
        bounds for scaling of data

    Returns
    -------
    scaled_values: np.ndarray, shape=(samples, features)
        scaled values to unit [0, 1]
    """
    x = np.atleast_2d(x)

    if bounds is None:
        lw_bound = np.min(x, axis=1)
        up_bound = np.max(x, axis=1)
    else:
        bounds = np.atleast_2d(bounds)
        lw_bound = bounds[:, 0]
        up_bound = bounds[:, 1]

    if (x > up_bound).any() or (x < lw_bound).any():
        raise ValueError("Input values outside of given bounds")

    if np.any(up_bound < lw_bound):
        raise ValueError("Upper bound below lower bound")

    scale = 1 / (up_bound - lw_bound)
    x_shifted = x - lw_bound  # shift data

    return scale * x_shifted  # scale data


def rescale_from_unity(x: ARRAY_LIKE_2D, bounds: BOUNDS) -> np.ndarray:
    """
    rescale x from unity [0, 1] to default space

    Parameters
    ----------
    x: np.ndarray, shape=(samples, features)
        input in unity space
    bounds: list
        bounds for scaling of data

    Returns
    -------
    scaled_values: np.ndarray, shape=(samples, features)
        rescaled values
    """
    x = np.atleast_2d(x)

    bounds = np.atleast_2d(bounds)
    lw_bound = bounds[:, 0]
    up_bound = bounds[:, 1]

    if (x > 1).any() or (x < 0).any():
        raise ValueError("Input values outside of unity space")

    scale = up_bound - lw_bound
    x_scaled = x * scale  # scale back

    return x_scaled + lw_bound  # shift back


def get_hardware() -> dict:
    """
    Return dict with detailed hardware information
    """
    # get system information
    uname = platform.uname()
    system_info = {
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
        "Machine": uname.machine,
    }

    # get CPU information
    cpufreq = psutil.cpu_freq()
    cpu_info = {
        "Processor": cpuinfo.get_cpu_info()["brand_raw"],
        "Physical cores": psutil.cpu_count(logical=False),
        "Total cores": psutil.cpu_count(logical=True),
        "Max Frequency [Mhz]": cpufreq.max,
        "Min Frequency [Mhz]": cpufreq.min,
        "Current Frequency [Mhz]": cpufreq.current,
        "Total CPU Usage [%]": psutil.cpu_percent(),
    }

    # get memory information
    svmem = psutil.virtual_memory()
    mem_info = {
        "Total [MB]": svmem.total / 1024,
        "Available [MB]": svmem.available / 1024,
        "Used [MB]": svmem.used / 1024,
        "Percentage [%]": svmem.percent,
    }

    # get GPU information
    gpus = GPUtil.getGPUs()
    gpu_info = {}

    if not gpus:
        gpu_info = None

    for i, gpu in enumerate(gpus):
        gpu_i = {
            "ID": gpu.id,
            "Name": gpu.name,
            "Load [%]": gpu.load * 100,
            "Free Memory [MB]": gpu.memoryFree,
            "Used Memory [MB]": gpu.memoryUsed,
            "Total Memory [MB]": gpu.memoryTotal,
            "Temperature [°C]": gpu.temperature,
        }
        entry = "GPU_" + str(i)
        gpu_info[entry] = gpu_i

    return {
        "system_info": system_info,
        "cpu_info": cpu_info,
        "mem_info": mem_info,
        "gpu_info": gpu_info,
    }


def print_progress(progress: float):
    """
    Print progress bar
    :param
    progress: float, 0 <= x <= 1
    """
    bar_length: int = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"

    block = int(round(bar_length * progress))
    sys.stdout.write("\r")
    text = "Simulate: [{0}] {1}% {2}".format(
        "#" * block + "-" * (bar_length - block), round(progress * 100, 2), status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def ensure_2d(x, copy: bool = True) -> np.ndarray:
    """
    Ensure that input x is 2d. If x.shape=(n_features) then a single sample is assumed and X.shape=(1, n_features)
    is returned.

    Parameters
    ----------
    x: arr_like, shape=(n_samples, n_features) or shape=(n_features)
        input
    copy: bool, optional(default=True)
        make copy of x

    Returns
    -------
    X: np.ndarray, shape=(n_samples, n_features)
        2d array
    """
    x = np.array(x)

    if copy:
        x = np.copy(x)

    if len(x.shape) == 1:  # add other dim
        X = np.atleast_2d(x)
    elif len(x.shape) == 2:  # nothing to do
        X = x
    else:
        raise ValueError("Dim of input to high, dim=%i" % len(x.shape))
    return X


def read_from_file(path):
    file = open("%s" % path, "rb")
    data = load(file, encoding="bytes")
    file.close()
    return data
