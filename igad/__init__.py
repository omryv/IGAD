from .detector import IGADDetector
from .curvature import scalar_curvature, fisher_metric, third_cumulant_tensor
from .exceptions import ConvergenceError

__all__ = ["IGADDetector", "scalar_curvature", "fisher_metric", "third_cumulant_tensor",
           "ConvergenceError"]
__version__ = "0.1.0"
