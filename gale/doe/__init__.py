"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE
"""
from .acquisition import (
    DLASED,
    EI,
    EIGF,
    GGESS,
    MASA,
    MEPE,
    MMSE,
    TEAD,
    adaptive_optimization,
    adaptive_sampling,
    wMMSE,
)
from .seqed import SeqED
from .utils import edge_sampling, sampling
