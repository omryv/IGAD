"""
Concrete exponential families with analytical log-partition functions.
"""

import numpy as np
from scipy.special import gammaln, digamma, polygamma


class GammaFamily:
    """
    Gamma(alpha, beta) in natural parameters theta = (alpha-1, -beta).

    A(theta_1, theta_2) = log Gamma(theta_1+1) - (theta_1+1) log(-theta_2)

    Domain: theta_1 > -1, theta_2 < 0.
    """

    name = "Gamma"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        t1, t2 = theta[0], theta[1]
        alpha = t1 + 1.0
        lam = -t2
        return float(gammaln(alpha) - alpha * np.log(lam))

    @staticmethod
    def to_natural(alpha: float, beta: float) -> np.ndarray:
        return np.array([alpha - 1.0, -beta])

    @staticmethod
    def from_natural(theta: np.ndarray):
        return theta[0] + 1.0, -theta[1]

    @staticmethod
    def fisher_metric_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = theta[0] + 1.0
        lam = -theta[1]
        g = np.array([
            [polygamma(1, alpha), 1.0 / lam],
            [1.0 / lam,          alpha / (lam**2)],
        ])
        return g

    @staticmethod
    def third_cumulant_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = theta[0] + 1.0
        lam = -theta[1]
        T = np.zeros((2, 2, 2))

        # T_{000} = psi''(alpha)
        T[0, 0, 0] = polygamma(2, alpha)
        # T_{001} = T_{010} = T_{100} = d/d(theta_2) g_{00}
        #   g_{00} = psi'(alpha), no dependence on theta_2 => 0
        # (already zero)

        # T_{011} = T_{101} = T_{110} = d/d(theta_1) g_{01}
        #   g_{01} = 1/lam = -1/theta_2, so d/d(theta_1) = 0
        #   BUT g_{01} = 1/lam, and d/d(theta_2)(g_{00}) = 0
        #   Actually T_{ijk} = d^3 A / d theta_i d theta_j d theta_k
        #   T_{011} = d^3 A / d theta_0 d theta_1 d theta_1
        #   A = gammaln(t1+1) - (t1+1)*log(-t2)
        #   dA/dt2 = -(t1+1)/t2 = (t1+1)/lam
        #   d^2A/dt1 dt2 = 1/(-t2) = 1/lam
        #   d^3A/dt1 dt2 dt2 = 0  (no t2 dependence in 1/lam... wait)
        #   d/dt2 (1/lam) = d/dt2 (-1/t2) = 1/t2^2 = 1/lam^2
        #   So T_{011} = +1/lam^2
        T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = 1.0 / (lam**2)

        # T_{111} = d^3 A / d theta_2^3
        #   dA/dt2 = (t1+1)/(-t2) = alpha/lam
        #   d^2A/dt2^2 = alpha/t2^2 = alpha/lam^2
        #   d^3A/dt2^3 = -2*alpha/t2^3 = -2*alpha/(-lam)^3 = 2*alpha/lam^3
        T[1, 1, 1] = 2.0 * alpha / (lam**3)

        return T

    @staticmethod
    def mle(data: np.ndarray) -> np.ndarray:
        """MLE for Gamma from positive data via Newton iteration."""
        x = np.asarray(data).ravel()
        x = x[x > 0]
        mean_x = np.mean(x)
        mean_log_x = np.mean(np.log(x))
        s = np.log(mean_x) - mean_log_x

        alpha = 0.5 / s if s > 0 else 1.0
        for _ in range(50):
            f = np.log(alpha) - digamma(alpha) - s
            fp = 1.0 / alpha - polygamma(1, alpha)
            step = f / fp
            alpha = max(alpha - step, 1e-8)
            if abs(step) < 1e-12:
                break

        beta = alpha / mean_x
        return GammaFamily.to_natural(alpha, beta)


def _inv_digamma(y: float) -> float:
    """Numerical inverse of digamma via Newton's method (Minka 2000)."""
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0 / (y + digamma(1.0))
    for _ in range(50):
        x -= (digamma(x) - y) / polygamma(1, x)
        if x <= 0:
            x = 1e-8
    return x


class DirichletFamily:
    """
    Dirichlet(alpha_1, ..., alpha_k) in natural parameters theta_i = alpha_i - 1.

    A(theta) = sum_i gammaln(theta_i + 1) - gammaln(sum_i (theta_i + 1))
             = sum_i gammaln(alpha_i) - gammaln(alpha_0)

    Domain: theta_i > -1  for all i  (i.e., alpha_i > 0).

    This is a (k-1)-dimensional exponential family embedded in R^k.
    The manifold has non-constant scalar curvature for k >= 3.

    Key structural property: for k >= 3, mean + marginal variances do NOT
    determine alpha uniquely. Pure concentration-profile shifts are detectable
    via curvature even when lower-order moments match.

    References:
        - Minka, T. (2000). Estimating a Dirichlet distribution. Technical report.
        - Amari & Nagaoka (2000). Methods of Information Geometry.
    """

    name = "Dirichlet"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        alpha = theta + 1.0
        return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

    @staticmethod
    def to_natural(alpha: np.ndarray) -> np.ndarray:
        return np.asarray(alpha, dtype=np.float64) - 1.0

    @staticmethod
    def from_natural(theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=np.float64) + 1.0

    @staticmethod
    def fisher_metric_analytical(theta: np.ndarray) -> np.ndarray:
        alpha = np.asarray(theta, dtype=np.float64) + 1.0
        alpha0 = alpha.sum()
        k = len(alpha)
        tri_alpha = polygamma(1, alpha)   # shape (k,)
        tri_alpha0 = polygamma(1, alpha0) # scalar
        g = -tri_alpha0 * np.ones((k, k))
        np.fill_diagonal(g, tri_alpha - tri_alpha0)
        return g

    @staticmethod
    def mle(data: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
        """
        MLE for Dirichlet via fixed-point iteration (Minka 2000).

        Parameters
        ----------
        data : array of shape (n, k), rows are simplex observations (sum to 1).
        max_iter : maximum fixed-point iterations.
        tol : convergence tolerance on alpha change.

        Returns
        -------
        theta : natural parameters (alpha - 1), shape (k,).

        Raises
        ------
        ConvergenceError
            If the fitted alpha does not reproduce E[log x_k] within 1e-4.
        """
        from igad.exceptions import ConvergenceError

        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n, k = data.shape

        data = np.clip(data, 1e-15, 1.0)
        mean_log_x = np.mean(np.log(data), axis=0)   # shape (k,) — sufficient stats

        # Moment-matching initialization
        mean_x = np.mean(data, axis=0)
        var_x  = np.var(data, axis=0)
        eps = 1e-10
        denom = var_x[0] + eps
        alpha0_mom = mean_x[0] * (1.0 - mean_x[0]) / denom - 1.0
        alpha0_mom = max(alpha0_mom, 1.0)
        alpha = mean_x * alpha0_mom
        alpha = np.clip(alpha, 1e-3, None)

        # Fixed-point iteration (Minka 2000)
        for iteration in range(max_iter):
            alpha_old = alpha.copy()
            alpha0 = alpha.sum()
            psi_alpha0 = digamma(alpha0)
            for i in range(k):
                target = psi_alpha0 + mean_log_x[i]
                alpha[i] = _inv_digamma(target)
            alpha = np.clip(alpha, 1e-8, None)
            if np.max(np.abs(alpha - alpha_old)) < tol:
                break

        # ── Constraint 1: MLE Convergence Gate ──────────────────────────────
        # Verify the fitted alpha reproduces the sufficient statistics E[log x_k].
        # If not, raise ConvergenceError — a silent bad fit corrupts all downstream
        # experiments.
        alpha0_hat = alpha.sum()
        expected_log = digamma(alpha) - digamma(alpha0_hat)
        residual = np.abs(expected_log - mean_log_x)
        if np.max(residual) > 1e-4:
            raise ConvergenceError(
                f"DirichletFamily MLE did not converge: "
                f"max sufficient-statistic residual = {np.max(residual):.2e} "
                f"(tolerance 1e-4). alpha_hat = {alpha}"
            )

        return DirichletFamily.to_natural(alpha)


class PoissonFamily:
    """
    Poisson(lam) in natural parameter theta = log(lam).
    A(theta) = exp(theta).
    1D manifold => R = 0 identically. Included for validation only.
    """

    name = "Poisson"

    @staticmethod
    def log_partition(theta: np.ndarray) -> float:
        return float(np.exp(theta[0]))

    @staticmethod
    def to_natural(lam: float) -> np.ndarray:
        return np.array([np.log(lam)])

    @staticmethod
    def mle(data: np.ndarray) -> np.ndarray:
        return PoissonFamily.to_natural(np.mean(data))
