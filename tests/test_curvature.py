import numpy as np
import pytest
from igad.curvature import scalar_curvature, fisher_metric, third_cumulant_tensor
from igad.families import GammaFamily, PoissonFamily

class TestPoissonFlat:
    def test_poisson_curvature_is_zero(self):
        for lam in [0.5, 1.0, 5.0, 20.0]:
            theta = PoissonFamily.to_natural(lam)
            R = scalar_curvature(PoissonFamily.log_partition, theta)
            assert abs(R) < 1e-4, "Poisson R=%s at lam=%s" % (R, lam)

class TestGammaFamily:
    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (5.0, 3.0), (0.5, 2.0)])
    def test_fisher_metric_matches_analytical(self, alpha, beta):
        theta = GammaFamily.to_natural(alpha, beta)
        g_num = fisher_metric(GammaFamily.log_partition, theta)
        g_ana = GammaFamily.fisher_metric_analytical(theta)
        np.testing.assert_allclose(g_num, g_ana, rtol=1e-4)

    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (5.0, 3.0), (1.5, 0.5)])
    def test_T_tensor_matches_analytical(self, alpha, beta):
        theta = GammaFamily.to_natural(alpha, beta)
        T_num = third_cumulant_tensor(GammaFamily.log_partition, theta)
        T_ana = GammaFamily.third_cumulant_analytical(theta)
        np.testing.assert_allclose(T_num, T_ana, rtol=5e-3, atol=1e-6)

    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (5.0, 3.0)])
    def test_T_tensor_is_symmetric(self, alpha, beta):
        theta = GammaFamily.to_natural(alpha, beta)
        T = third_cumulant_tensor(GammaFamily.log_partition, theta)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert abs(T[i,j,k] - T[j,i,k]) < 1e-6
                    assert abs(T[i,j,k] - T[i,k,j]) < 1e-6

    def test_curvature_varies_with_alpha(self):
        R_values = []
        for alpha in [2.0, 5.0, 10.0, 20.0]:
            theta = GammaFamily.to_natural(alpha, 1.0)
            R = scalar_curvature(GammaFamily.log_partition, theta)
            R_values.append(R)
        assert len(set(np.round(R_values, 6))) > 1, "Curvature should vary"

    def test_curvature_is_finite(self):
        for alpha in [0.5, 1.0, 3.0, 10.0]:
            for beta in [0.5, 1.0, 5.0]:
                theta = GammaFamily.to_natural(alpha, beta)
                R = scalar_curvature(GammaFamily.log_partition, theta)
                assert np.isfinite(R), "Non-finite at alpha=%s beta=%s" % (alpha, beta)

    def test_curvature_formula_consistency(self):
        """Verify R = 1/4*(||T||^2 - ||S||^2) by checking components.

        This test re-implements the identity inline, so it pins the
        contraction bookkeeping only -- it cannot detect an error in the
        identity itself, since both sides would move together. The sign and
        magnitude of R are pinned externally in
        tests/test_curvature_ground_truth.py.
        """
        theta = GammaFamily.to_natural(5.0, 2.0)
        g = fisher_metric(GammaFamily.log_partition, theta)
        T = third_cumulant_tensor(GammaFamily.log_partition, theta)
        g_inv = np.linalg.inv(g)
        S = np.einsum("ab,abm->m", g_inv, T)
        S_sq = np.einsum("mn,m,n->", g_inv, S, S)
        T_sq = np.einsum("ia,jb,kc,ijk,abc->", g_inv, g_inv, g_inv, T, T)
        R_manual = 0.25 * (T_sq - S_sq)
        R_func = scalar_curvature(GammaFamily.log_partition, theta, g=g, T=T)
        assert abs(R_manual - R_func) < 1e-10, "Inconsistent: %s vs %s" % (R_manual, R_func)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
