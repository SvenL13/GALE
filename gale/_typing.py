"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Typing
"""
from typing import Callable, List, Tuple, Union

import numpy as np

ARRAY_LIKE_1D = Union[List[float], Tuple[float, ...], np.ndarray]
ARRAY_LIKE_2D = Union[
    List[List[float]],
    List[Tuple[float, ...]],
    Tuple[List[float], ...],
    Tuple[Tuple[float, ...], ...],
    np.ndarray,
]

BOUNDS = List[List[float]]

FUNCTION = Callable[[np.ndarray, ...], np.ndarray]
