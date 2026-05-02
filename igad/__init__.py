__version__ = "1.0.1"



from .detector import IGADDetector
from .curvature import scalar_curvature, fisher_metric, third_cumulant_tensor
from .exceptions import ConvergenceError

__all__ = ["IGADDetector", "scalar_curvature", "fisher_metric", "third_cumulant_tensor",
           "ConvergenceError"]
