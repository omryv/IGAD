"""IGAD: Information-Geometric Anomaly Detection."""
from .curvature import (
    fisher_metric,
    scalar_curvature,
    third_cumulant_tensor,
)
from .detector import IGADDetector
from .exceptions import ConvergenceError

__version__ = "1.0.2"
__all__ = [
    "IGADDetector",
    "scalar_curvature",
    "fisher_metric",
    "third_cumulant_tensor",
    "ConvergenceError",
    "__version__",
]
