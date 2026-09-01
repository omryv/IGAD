"""
Ground-truth tests for the scalar curvature.

Every other curvature test in this suite compares one implementation to
another, or to a high-precision evaluation of the *same* formula. That
pins arithmetic, not the identity: a sign error or a wrong constant in

    R = 1/4 ( ||T||^2_g - ||S||^2_g )

is invisible to all of them, because every route shares it.

These tests compare R against values known independently of this codebase:

  * the univariate Gaussian Fisher-Rao manifold. Its metric is
    ds^2 = (dmu^2 + 2 dsigma^2) / sigma^2, which under sigma = y/sqrt(2)
    becomes 2 (dmu^2 + dy^2) / y^2 -- the Poincare half-plane scaled by 2.
    Scaling a metric by lambda scales curvature by 1/lambda, so K = -1/2
    and, in two dimensions, R = 2K = -1.

  * the full mean-and-covariance Gaussian family in dimension d, whose
    scalar curvature is -d(d+1)^2/4.

  * one-dimensional families, which are flat: R = 0 identically.

References
----------
Skovgaard, L. T. (1984). A Riemannian geometry of the multivariate normal
    model. Scand. J. Statist. 11(4), 211-223.
Amari & Nagaoka (2000). Methods of Information Geometry, Ch. 2.
"""
import numpy as np
import pytest

from igad.curvature import scalar_curvature
from igad.families import DirichletFamily, PoissonFamily


def _gaussian_1d_log_partition(theta):
    """N(mu, sigma^2) with theta = (mu/sigma^2, -1/(2 sigma^2))."""
    t1, t2 = theta[0], theta[1]
    return float(-t1 ** 2 / (4.0 * t2) - 0.5 * np.log(-2.0 * t2))


def _gaussian_1d_exact(mu, sigma):
    """Exact g and T for the univariate Gaussian, by hand differentiation."""
    t1 = mu / sigma ** 2
    t2 = -1.0 / (2.0 * sigma ** 2)
    g = np.array([[-1.0 / (2.0 * t2), t1 / (2.0 * t2 ** 2)],
                  [t1 / (2.0 * t2 ** 2), -t1 ** 2 / (2.0 * t2 ** 3) + 1.0 / (2.0 * t2 ** 2)]])
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 0.0
    T[0, 0, 1] = T[0, 1, 0] = T[1, 0, 0] = 1.0 / (2.0 * t2 ** 2)
    T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = -t1 / t2 ** 3
    T[1, 1, 1] = 3.0 * t1 ** 2 / (2.0 * t2 ** 4) - 1.0 / t2 ** 3
    return np.array([t1, t2]), g, T


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.0, 2.0), (-3.0, 0.5), (2.5, 3.0)])
def test_univariate_gaussian_has_scalar_curvature_minus_one(mu, sigma):
    """The textbook value. This is the test that pins the sign of R."""
    theta, g, T = _gaussian_1d_exact(mu, sigma)
    R = scalar_curvature(_gaussian_1d_log_partition, theta, g=g, T=T)
    assert R == pytest.approx(-1.0, abs=1e-12), (
        "univariate Gaussian R = %.12f, expected -1 (hyperbolic, K = -1/2). "
        "A value of +1 means the identity is negated." % R
    )


def test_gaussian_curvature_is_negative_not_positive():
    """Guards the sign specifically, in the terms docs/proof.md uses:
    the Gaussian manifold is hyperbolic, so its curvature is negative."""
    theta, g, T = _gaussian_1d_exact(0.0, 1.0)
    assert scalar_curvature(_gaussian_1d_log_partition, theta, g=g, T=T) < 0.0


def _mvn_log_partition_factory(d):
    """A(theta) for N(m, S) with natural parameters (S^-1 m, -1/2 S^-1).

    Theta is packed minimally -- mean, then the diagonal of the precision
    block, then its strict upper triangle -- so the parametrisation is
    d(d+3)/2-dimensional and the Fisher metric is non-singular. Packing the
    full d x d matrix instead would repeat each off-diagonal twice and make
    g singular.
    """
    iu = np.triu_indices(d, 1)

    def A(theta):
        h = np.asarray(theta[:d], dtype=np.float64)
        Th = np.zeros((d, d))
        Th[np.diag_indices(d)] = theta[d:2 * d]
        Th[iu] = theta[2 * d:]
        Th = Th + np.triu(Th, 1).T
        return float(-0.25 * h @ np.linalg.solve(Th, h)
                     - 0.5 * np.log(np.linalg.det(-2.0 * Th)))
    return A


@pytest.mark.parametrize("d", [1, 2, 3])
def test_multivariate_gaussian_matches_the_closed_form(d):
    """R = -d(d+1)^2/4 for the full mean-and-covariance Gaussian family.

    Evaluated through the generic finite-difference path, so the tolerance
    is set by the stencil (~3e-5 relative here), not by the identity.
    """
    iu = np.triu_indices(d, 1)
    S = np.eye(d) + 0.1 * np.ones((d, d))
    Sinv = np.linalg.inv(S)
    Th = -0.5 * Sinv
    theta = np.concatenate([np.zeros(d), np.diag(Th), Th[iu]])
    R = scalar_curvature(_mvn_log_partition_factory(d), theta)
    expected = -d * (d + 1) ** 2 / 4.0
    assert R == pytest.approx(expected, rel=1e-3), (
        "d=%d gave R=%.6f, expected %.6f" % (d, R, expected)
    )


@pytest.mark.parametrize("lam", [0.5, 1.0, 4.0])
def test_one_dimensional_families_are_flat(lam):
    """A 1-D manifold has no curvature. Independent of any convention."""
    theta = PoissonFamily.to_natural(lam)
    assert abs(scalar_curvature(PoissonFamily.log_partition, theta)) < 1e-4


@pytest.mark.parametrize("alpha", [[2.0, 3.0, 4.0], [1.0, 1.0, 1.0], [0.5, 2.0, 5.0]])
def test_dirichlet_curvature_is_negative(alpha):
    """Consequence of the corrected sign, and the quantity IGADDetector
    differences. Recorded so a future reversal is caught here rather than
    in a downstream AUC."""
    theta = DirichletFamily.to_natural(np.array(alpha))
    assert DirichletFamily.scalar_curvature_analytical(theta) < 0.0
