"""rxmc: Bayesian calibration of reaction models with composable error models.

The 1.0 rewrite is in progress on this branch; see ``docs/groundup_design.md``
for the design and ``docs/recipes.md`` for the supported use cases.
"""

from . import constraint as constraint
from . import covariance as covariance
from . import data as data
from . import diagnostics as diagnostics
from . import likelihood as likelihood
from . import model as model
from . import predictive as predictive
from . import problem as problem
from . import reactions as reactions
from . import terms as terms
from . import transforms as transforms
from . import units as units
from .constraint import Comparison as Comparison
from .constraint import Constraint as Constraint
from .data import Dataset as Dataset
from .data import from_measurement as from_measurement
from .likelihood import Chi2 as Chi2
from .likelihood import Gaussian as Gaussian
from .likelihood import StudentT as StudentT
from .model import Model as Model
from .model import polynomial as polynomial
from .params import Parameter as Parameter
from .problem import Problem as Problem
from .terms import KernelTerm as KernelTerm
from .terms import Term as Term

try:
    from .__version__ import __version__ as __version__
except ImportError:  # pragma: no cover - source checkout without a build
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Chi2",
    "Comparison",
    "Constraint",
    "Dataset",
    "Gaussian",
    "KernelTerm",
    "Model",
    "Parameter",
    "Problem",
    "StudentT",
    "Term",
    "from_measurement",
    "polynomial",
    "constraint",
    "covariance",
    "data",
    "diagnostics",
    "likelihood",
    "model",
    "predictive",
    "problem",
    "reactions",
    "terms",
    "transforms",
    "units",
]
