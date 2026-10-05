"""
IGAD: Information-Geometric Anomaly Detection.

Batch-level scalar curvature anomaly detector.

    score = |R(theta_ref) - R(theta_local)|

where R is the scalar curvature of the Fisher-Rao manifold of an
exponential family. Detects distributional shape shifts that preserve
mean and variance, which are invisible to moment-based detectors.

References
----------
Amari & Nagaoka (2000). Methods of Information Geometry. AMS/Oxford.
Rao (1945). Information and the accuracy attainable in the estimation
    of statistical parameters. Bull. Calcutta Math. Soc.
"""
from typing import Optional

import numpy as np

from .curvature import scalar_curvature


class IGADDetector:
    """
    Information-Geometric Anomaly Detector (batch-level).

    Parameters
    ----------
    family : object
        Exponential family providing ``log_partition(theta)`` and
        ``mle(data)``. May optionally provide
        ``scalar_curvature_analytical(theta)``,
        ``fisher_metric_analytical(theta)`` and
        ``third_cumulant_analytical(theta)`` for closed-form curvature.
    use_analytical : bool, default True
        If True, use the closed-form curvature routes a family exposes.
        Set False to force finite-difference evaluation (useful for
        testing the generic path).

    Attributes
    ----------
    theta_ref_ : ndarray
        Natural parameters of the reference distribution.
    R_ref_ : float
        Scalar curvature evaluated at theta_ref_.

    Notes
    -----
    For best stability, prefer ``fit(theta_ref=...)`` when reference
    parameters are known theoretically. When fitting from data, use
    n >= 1000 to control variance in the estimated R_ref.
    """

    def __init__(self, family, use_analytical: bool = True):
        self.family = family
        self.use_analytical = use_analytical
        self.theta_ref_: Optional[np.ndarray] = None
        self.R_ref_: Optional[float] = None

    def _scalar_curvature(self, theta: np.ndarray) -> float:
        """
        Compute scalar curvature, preferring the cheapest exact route
        available:

        1. ``family.scalar_curvature_analytical`` - O(k), never builds T.
        2. exact ``g`` and ``T`` when the family supplies them, contracted
           by :func:`~igad.curvature.scalar_curvature`.
        3. finite differences - fully generic fallback.

        All three are the same quantity mathematically. They are *not*
        interchangeable numerically: route 3 loses accuracy rapidly as the
        parameter grows (the second and third differences of A are taken
        with fixed steps), so a family that can supply exact g or T should.
        ``use_analytical=False`` forces (3).
        """
        if self.use_analytical and hasattr(self.family, "scalar_curvature_analytical"):
            return float(self.family.scalar_curvature_analytical(theta))

        g: Optional[np.ndarray] = None
        T: Optional[np.ndarray] = None
        if self.use_analytical:
            if hasattr(self.family, "fisher_metric_analytical"):
                g = self.family.fisher_metric_analytical(theta)
            if hasattr(self.family, "third_cumulant_analytical"):
                T = self.family.third_cumulant_analytical(theta)
        return scalar_curvature(self.family.log_partition, theta, g=g, T=T)

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        theta_ref: Optional[np.ndarray] = None,
    ) -> "IGADDetector":
        """
        Fit the reference distribution.

        Provide exactly one of ``X`` (estimate theta_ref via MLE) or
        ``theta_ref`` (use known natural parameters).

        Parameters
        ----------
        X : array-like, optional
            Reference training data. Recommended size n >= 1000.
        theta_ref : array-like, optional
            Known natural parameters of the reference distribution.

        Returns
        -------
        self : IGADDetector
        """
        if (X is None) == (theta_ref is None):
            raise ValueError(
                "Provide exactly one of X or theta_ref."
            )
        if theta_ref is not None:
            self.theta_ref_ = np.asarray(theta_ref, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.theta_ref_ = self.family.mle(X_arr)
        self.R_ref_ = self._scalar_curvature(self.theta_ref_)
        return self

    def score_batch(self, X_batch: np.ndarray) -> float:
        """
        Score a batch of observations.

        Parameters
        ----------
        X_batch : array-like
            Batch from a candidate distribution.

        Returns
        -------
        score : float
            ``|R(theta_ref) - R(theta_local)|``. Higher is more anomalous.

            The absolute value matters: R is not monotone in the shape
            parameters of every family (for a symmetric Dirichlet it rises
            to a ridge near alpha ~ 5 and decays back toward k/2), so the
            signed difference does not order anomalies in either direction.
        """
        if self.theta_ref_ is None:
            raise RuntimeError(
                "Detector not fitted. Call fit() before score_batch()."
            )
        X_arr = np.asarray(X_batch, dtype=np.float64)
        theta_local = self.family.mle(X_arr)
        R_local = self._scalar_curvature(theta_local)
        return float(abs(self.R_ref_ - R_local))
