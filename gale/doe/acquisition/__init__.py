from .dl_ased import DLASED
from .ei import EI
from .eigf import EIGF
from .mmse import MMSE
from .mepe import MEPE
from .wmmse import wMMSE
from .ggess import GGESS
from .masa import MASA
from .tead import TEAD
from .guess import GUESS
from .lhs_baseline import SLHS

# adaptive strategies for global surrogate modelling (single response problems)
adaptive_sampling = {"GGESS": GGESS,
                     "EIGF": EIGF,
                     "wMMSE": wMMSE,
                     "MMSE": MMSE,
                     "MASA": MASA,
                     "MEPE": MEPE,
                     "TEAD": TEAD,
                     "DL_ASED": DLASED,
                     "GUESS": GUESS,
                     "LHS": SLHS,
                     }

# adaptive methods for global optimization
adaptive_optimization = {"EI": EI}
