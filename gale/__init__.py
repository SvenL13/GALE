# package information
__name__ = "gale"
__version__ = "0.1"

from . import utils
from . import models
from . import doe
from . import experiments

# module level doc-string
__doc__ = """
GALe - A Python Library for Global Adaptive Learning
====================================================
Author: Sven Lämmle

Contributors: Can Bogoclu

**gale** is a Python package providing tools for using 
adaptive sampling strategies to create globally accurate
surrogate models for single response problems. The package
implements different state of the art methods based on recent
research.

Available subpackages
---------------------
doe
    Core tools for Design of Experiments including acquisition functions
experiments
    Experiments from the corresponding publication
models
    Surrogate models

Utilities
---------
examples
    Basic examples of the package usage

Corresponding Publications
--------------------------
[1] S. Lämmle, C. Bogoclu, K. Cremanns, D. Roos - Gradient and Uncertainty enhanced
    Sequential Sampling for Global Fit. Computer Methods in applied Mechanics and
    Engineering, Volume 415, 2023. https://doi.org/10.1016/j.cma.2023.116226
"""
