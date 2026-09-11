"""rxmc: Bayesian calibration of reaction models with composable error models.

The 1.0 rewrite is in progress on this branch; see ``docs/groundup_design.md``
for the design and ``docs/recipes.md`` for the supported use cases.
"""

from . import constraint as constraint
from . import covariance as covariance
from . import data as data
from . import likelihood as likelihood
from . import model as model
from . import problem as problem
from . import reactions as reactions
from . import terms as terms
from . import transforms as transforms
from . import units as units
from .constraint import Comparison as Comparison
from .constraint import Constraint as Constraint
from .data import Dataset as Dataset
from .data import from_measurement as from_measurement
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
    "Comparison",
    "Constraint",
    "Dataset",
    "KernelTerm",
    "Model",
    "Parameter",
    "Problem",
    "Term",
    "from_measurement",
    "polynomial",
    "constraint",
    "covariance",
    "data",
    "likelihood",
    "model",
    "problem",
    "reactions",
    "terms",
    "transforms",
    "units",
]
