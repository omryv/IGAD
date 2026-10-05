__version__ = "1.0.4"



from .detector import IGADDetector
from .curvature import (
    scalar_curvature,
    scalar_curvature_structured,
    scalar_curvature_dirichlet,
    curvature_reliability,
    dirichlet_fisher_inverse,
    fisher_metric,
    third_cumulant_tensor,
)
from .exceptions import ConvergenceError

__all__ = ["IGADDetector", "scalar_curvature", "scalar_curvature_structured",
           "scalar_curvature_dirichlet", "curvature_reliability",
           "dirichlet_fisher_inverse", "fisher_metric", "third_cumulant_tensor",
           "ConvergenceError"]
