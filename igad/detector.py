"""
IGAD: Information-Geometric Anomaly Detection.

Anomaly score = R(theta_ref) - R(theta_hat(z))

where R is the scalar curvature of the Fisher-Rao manifold.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Optional

from .curvature import scalar_curvature


class IGADDetector:
    """
    Information-Geometric Anomaly Detector.

    Parameters
    ----------
    family : object
        Exponential family with .log_partition(theta) and .mle(data).
    k_neighbors : int
        Number of neighbors for local parameter estimation.
    use_analytical_T : bool
        If True (default) and the family exposes ``third_cumulant_analytical``,
        use it instead of the finite-difference ``third_cumulant_tensor``.
        Set to False to force the finite-difference path (useful for testing).
    """

    def __init__(self, family, k_neighbors: int = 30, use_analytical_T: bool = True):
        self.family = family
        self.k = k_neighbors
        self.use_analytical_T = use_analytical_T
        self.theta_ref_ = None
        self.R_ref_ = None
        self.tree_ = None
        self.X_train_ = None

    def _scalar_curvature(self, theta: np.ndarray) -> float:
        """Compute scalar curvature, using analytical T if available and requested."""
        T: Optional[np.ndarray] = None
        if self.use_analytical_T and hasattr(self.family, "third_cumulant_analytical"):
            T = self.family.third_cumulant_analytical(theta)
        return scalar_curvature(self.family.log_partition, theta, T=T)

    def fit(self, X: np.ndarray) -> "IGADDetector":
        """Fit reference distribution from training data."""
        X = np.asarray(X, dtype=np.float64)
        self.X_train_ = X
        self.tree_ = KDTree(X)

        self.theta_ref_ = self.family.mle(X)
        self.R_ref_ = self._scalar_curvature(self.theta_ref_)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute IGAD anomaly scores.
        Higher = more anomalous.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            dists, idxs = self.tree_.query(X[i], k=self.k)
            neighbors = self.X_train_[idxs.ravel()]

            try:
                theta_local = self.family.mle(neighbors)
                R_local = self._scalar_curvature(theta_local)
                scores[i] = self.R_ref_ - R_local
            except (np.linalg.LinAlgError, ValueError):
                scores[i] = np.inf

        return scores

    def predict(self, X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
        """Binary anomaly prediction. 1 = anomaly, 0 = normal."""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)
