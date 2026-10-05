"""
tests/test_structured_curvature.py

Regression net for the O(k^2) structured curvature route.

The claim under test is *exactness*: scalar_curvature_structured must return
the same number as the general scalar_curvature contraction, for every
parameter point, not merely a close one. The existing suite pins the general
path; these tests pin the fast path to it.
"""

import time

import numpy as np
import pytest

from igad.curvature import (
    scalar_curvature,
    scalar_curvature_structured,
    fisher_metric,
    third_cumulant_tensor,
)
from igad.detector import IGADDetector
from igad.families import DirichletFamily, GammaFamily


ALPHAS = [
    [4.0, 4.0, 4.0],
    [1.5, 4.0, 6.5],
    [0.5, 0.5, 0.5],
    [0.1, 0.2, 0.7],
    [2.0, 3.0, 5.0, 7.0],
    [1.0, 1.0, 1.0, 1.0, 2.0],
    [50.0, 50.0, 50.0],
    [0.05, 3.0, 12.0, 0.4, 6.0],
]


# ─────────────────────────────────────────────────────────────────────────────
# Class 1 - the structural form reproduces the dense tensor
# ─────────────────────────────────────────────────────────────────────────────

class TestThirdCumulantStructure:

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_structure_rebuilds_dense_tensor(self, alpha):
        """T[i,j,k] = c + d_i*delta_{ijk} must match third_cumulant_analytical."""
        theta = DirichletFamily.to_natural(np.array(alpha))
        T_dense = DirichletFamily.third_cumulant_analytical(theta)
        c, d = DirichletFamily.third_cumulant_structure(theta)

        k = len(alpha)
        T_rebuilt = np.full((k, k, k), c, dtype=np.float64)
        idx = np.arange(k)
        T_rebuilt[idx, idx, idx] += d

        np.testing.assert_allclose(T_rebuilt, T_dense, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_structure_shapes(self, alpha):
        theta = DirichletFamily.to_natural(np.array(alpha))
        c, d = DirichletFamily.third_cumulant_structure(theta)
        assert isinstance(c, float)
        assert d.shape == (len(alpha),)


# ─────────────────────────────────────────────────────────────────────────────
# Class 2 - exactness against the general contraction
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredMatchesGeneral:

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_matches_exact_einsum(self, alpha):
        """
        Same metric, same tensor, two contraction routes. Any disagreement
        beyond round-off is an algebra error in the closed form.
        """
        theta = DirichletFamily.to_natural(np.array(alpha))
        g_ana = DirichletFamily.fisher_metric_analytical(theta)
        T_ana = DirichletFamily.third_cumulant_analytical(theta)

        R_general = scalar_curvature(
            DirichletFamily.log_partition, theta, g=g_ana, T=T_ana
        )
        R_struct = DirichletFamily.scalar_curvature_analytical(theta)

        assert abs(R_general - R_struct) <= 1e-9 * max(1.0, abs(R_general)), (
            "structured=%r general=%r alpha=%s" % (R_struct, R_general, alpha)
        )

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_structured_helper_directly(self, alpha):
        """scalar_curvature_structured(g, c, d) is the same as the dense route."""
        theta = DirichletFamily.to_natural(np.array(alpha))
        g_ana = DirichletFamily.fisher_metric_analytical(theta)
        T_ana = DirichletFamily.third_cumulant_analytical(theta)
        c, d = DirichletFamily.third_cumulant_structure(theta)

        R_general = scalar_curvature(
            DirichletFamily.log_partition, theta, g=g_ana, T=T_ana
        )
        R_struct = scalar_curvature_structured(g_ana, c, d)
        assert abs(R_general - R_struct) <= 1e-9 * max(1.0, abs(R_general))

    @pytest.mark.parametrize("alpha", [[4.0, 4.0, 4.0], [1.5, 4.0, 6.5],
                                       [2.0, 3.0, 5.0, 7.0]])
    def test_matches_full_finite_difference_chain(self, alpha):
        """
        End-to-end guard: analytical fast path vs the fully numerical path.
        Tolerance here is set by the finite-difference stencils in
        fisher_metric/third_cumulant_tensor, not by the closed form.
        """
        theta = DirichletFamily.to_natural(np.array(alpha))
        R_fd = scalar_curvature(DirichletFamily.log_partition, theta)
        R_struct = DirichletFamily.scalar_curvature_analytical(theta)
        np.testing.assert_allclose(R_struct, R_fd, rtol=5e-2)

    def test_known_separation_preserved(self):
        """
        The documented Dirichlet detection pair must keep its separation:
        |R([4,4,4]) - R([1.5,4,6.5])| > 0.01  (docs/operational_envelope.md).
        """
        R_ref = DirichletFamily.scalar_curvature_analytical(
            DirichletFamily.to_natural(np.array([4.0, 4.0, 4.0]))
        )
        R_anom = DirichletFamily.scalar_curvature_analytical(
            DirichletFamily.to_natural(np.array([1.5, 4.0, 6.5]))
        )
        assert abs(R_ref - R_anom) > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Class 3 - the point of the change: it scales
# ─────────────────────────────────────────────────────────────────────────────

class TestScaling:

    @pytest.mark.parametrize("k", [16, 32, 64, 128])
    def test_large_k_is_finite_and_fast(self, k):
        """
        The dense route is ~k^6 operations (6.9e10 at k=64) and cannot finish.
        The structured route is ~k^2. A generous wall-clock bound catches any
        regression back to the dense contraction without being flaky.
        """
        theta = DirichletFamily.to_natural(np.full(k, 2.0))
        t0 = time.perf_counter()
        R = DirichletFamily.scalar_curvature_analytical(theta)
        elapsed = time.perf_counter() - t0

        assert np.isfinite(R), "non-finite R at k=%d" % k
        assert elapsed < 10.0, "k=%d took %.2fs - dense path regression?" % (k, elapsed)

    def test_curvature_grows_with_expert_count(self):
        """R scales roughly linearly in k for a uniform Dirichlet."""
        ratios = []
        for k in (8, 16, 32, 64):
            theta = DirichletFamily.to_natural(np.full(k, 2.0))
            ratios.append(DirichletFamily.scalar_curvature_analytical(theta) / k)
        assert max(ratios) - min(ratios) < 0.05, "R/k not stable: %s" % ratios


# ─────────────────────────────────────────────────────────────────────────────
# Class 4 - detector wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectorFastPath:

    def test_detector_prefers_structured_route(self):
        """IGADDetector picks the fast path but reports the same curvature."""
        theta = DirichletFamily.to_natural(np.array([2.0, 3.0, 5.0]))

        fast = IGADDetector(family=DirichletFamily, use_analytical=True)
        slow = IGADDetector(family=DirichletFamily, use_analytical=False)

        R_fast = fast._scalar_curvature(theta)
        R_slow = slow._scalar_curvature(theta)
        np.testing.assert_allclose(R_fast, R_slow, rtol=5e-2)

    def test_gamma_family_unaffected(self):
        """
        GammaFamily exposes no scalar_curvature_analytical, so it routes
        through the dense-T path -- now supplying the exact Fisher metric
        too, rather than re-deriving it by finite differences.
        """
        assert not hasattr(GammaFamily, "scalar_curvature_analytical")
        theta = GammaFamily.to_natural(5.0, 2.0)
        det = IGADDetector(family=GammaFamily, use_analytical=True)
        expected = scalar_curvature(
            GammaFamily.log_partition,
            theta,
            g=GammaFamily.fisher_metric_analytical(theta),
            T=GammaFamily.third_cumulant_analytical(theta),
        )
        assert abs(det._scalar_curvature(theta) - expected) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
