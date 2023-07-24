"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models
"""
from .surrogate import SurrogateRegressionModel
from .gp import GPRegressor, gradient_fd

from .regr_metrics import mean_relative_error

from .utils import cook_model
