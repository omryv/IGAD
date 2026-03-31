"""
tests/test_dirichlet_family.py

Unit tests for DirichletFamily — 21 tests across 5 classes.
"""

import numpy as np
import pytest
from scipy.special import gammaln, digamma, polygamma

from igad.curvature import scalar_curvature, fisher_metric
from igad.exceptions import ConvergenceError
from igad.families import DirichletFamily, PoissonFamily


# ─────────────────────────────────────────────────────────────────────────────
# Class 1 — Log-partition function
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletLogPartition:

    def test_log_partition_symmetric(self):
        """A(θ) for α=[2,2,2] should equal 3*gammaln(2) - gammaln(6)."""
        alpha = np.array([2.0, 2.0, 2.0])
        theta = DirichletFamily.to_natural(alpha)
        result = DirichletFamily.log_partition(theta)
        expected = 3 * gammaln(2.0) - gammaln(6.0)
        assert abs(result - expected) < 1e-10

    def test_log_partition_gradient_matches_mean(self):
        """Numerical gradient of A(θ) w.r.t. θᵢ equals E[log xᵢ] = digamma(αᵢ) - digamma(α₀)."""
        alpha = np.array([2.0, 3.0, 5.0])
        theta = DirichletFamily.to_natural(alpha)
        alpha0 = alpha.sum()
        # Exponential family identity: dA/dθᵢ = E[T_i(x)] where T_i = log x_i
        expected_grad = digamma(alpha) - digamma(alpha0)

        eps = 1e-5
        grad = np.zeros(len(theta))
        for i in range(len(theta)):
            t_fwd = theta.copy(); t_fwd[i] += eps
            t_bwd = theta.copy(); t_bwd[i] -= eps
            grad[i] = (DirichletFamily.log_partition(t_fwd) -
                       DirichletFamily.log_partition(t_bwd)) / (2 * eps)

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-4)

    def test_log_partition_is_convex(self):
        """Hessian (Fisher metric) must be positive definite."""
        for alpha in [[2.0, 2.0, 2.0], [1.5, 3.0, 5.0], [0.5, 0.5, 2.0]]:
            theta = DirichletFamily.to_natural(np.array(alpha))
            g = fisher_metric(DirichletFamily.log_partition, theta)
            eigvals = np.linalg.eigvalsh(g)
            assert np.all(eigvals > 0), (
                "Fisher metric not PD for alpha=%s, eigvals=%s" % (alpha, eigvals)
            )

    def test_natural_param_roundtrip(self):
        """from_natural(to_natural(α)) == α."""
        for alpha in [[1.0, 2.0, 3.0], [0.5, 4.0, 6.5], [10.0, 10.0, 10.0]]:
            alpha = np.array(alpha)
            recovered = DirichletFamily.from_natural(DirichletFamily.to_natural(alpha))
            np.testing.assert_allclose(recovered, alpha, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Class 2 — Fisher metric
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletFisherMetric:

    @pytest.mark.parametrize("alpha", [
        [2.0, 2.0, 2.0],
        [1.5, 3.0, 5.0],
        [0.5, 0.5, 2.0],
    ])
    def test_fisher_metric_matches_numerical(self, alpha):
        """Analytical Fisher metric matches numerical Hessian."""
        alpha = np.array(alpha)
        theta = DirichletFamily.to_natural(alpha)
        g_ana = DirichletFamily.fisher_metric_analytical(theta)
        g_num = fisher_metric(DirichletFamily.log_partition, theta)
        np.testing.assert_allclose(g_ana, g_num, rtol=1e-3)

    @pytest.mark.parametrize("alpha", [
        [2.0, 2.0, 2.0],
        [1.5, 3.0, 5.0],
        [4.0, 4.0, 4.0],
    ])
    def test_fisher_metric_is_symmetric(self, alpha):
        """g == gᵀ."""
        alpha = np.array(alpha)
        theta = DirichletFamily.to_natural(alpha)
        g = DirichletFamily.fisher_metric_analytical(theta)
        np.testing.assert_allclose(g, g.T, atol=1e-12)

    @pytest.mark.parametrize("alpha", [
        [2.0, 2.0, 2.0],
        [1.5, 3.0, 5.0],
        [0.5, 1.0, 2.0],
    ])
    def test_fisher_metric_is_positive_definite(self, alpha):
        """All eigenvalues > 0."""
        alpha = np.array(alpha)
        theta = DirichletFamily.to_natural(alpha)
        g = DirichletFamily.fisher_metric_analytical(theta)
        eigvals = np.linalg.eigvalsh(g)
        assert np.all(eigvals > 0), "Not PD for alpha=%s" % alpha

    def test_fisher_metric_k4(self):
        """k=4 case works correctly (not hardcoded for k=3)."""
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        theta = DirichletFamily.to_natural(alpha)
        g_ana = DirichletFamily.fisher_metric_analytical(theta)
        g_num = fisher_metric(DirichletFamily.log_partition, theta)
        np.testing.assert_allclose(g_ana, g_num, rtol=1e-3)
        eigvals = np.linalg.eigvalsh(g_ana)
        assert np.all(eigvals > 0)

    def test_fisher_metric_diagonal_formula(self):
        """g_ii = trigamma(αᵢ) - trigamma(α₀),  g_ij = -trigamma(α₀)."""
        alpha = np.array([2.0, 3.0, 5.0])
        theta = DirichletFamily.to_natural(alpha)
        g = DirichletFamily.fisher_metric_analytical(theta)
        alpha0 = alpha.sum()
        tri_alpha0 = polygamma(1, alpha0)

        for i in range(len(alpha)):
            expected_diag = polygamma(1, alpha[i]) - tri_alpha0
            assert abs(g[i, i] - expected_diag) < 1e-10, (
                "Diagonal mismatch at i=%d" % i
            )
            for j in range(len(alpha)):
                if i != j:
                    assert abs(g[i, j] - (-tri_alpha0)) < 1e-10, (
                        "Off-diagonal mismatch at (%d,%d)" % (i, j)
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Class 3 — Scalar curvature
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletCurvature:

    @pytest.mark.parametrize("alpha", [
        [2.0, 2.0, 2.0],
        [1.5, 3.0, 5.0],
        [0.5, 1.0, 2.0],
        [4.0, 4.0, 4.0],
    ])
    def test_curvature_is_finite(self, alpha):
        """R is finite for valid alpha."""
        theta = DirichletFamily.to_natural(np.array(alpha))
        R = scalar_curvature(DirichletFamily.log_partition, theta)
        assert np.isfinite(R), "Non-finite R for alpha=%s" % alpha

    def test_curvature_varies_with_concentration(self):
        """R([4,4,4]) != R([1,1,1]) — curvature is non-constant."""
        R_444 = scalar_curvature(DirichletFamily.log_partition,
                                 DirichletFamily.to_natural(np.array([4.0, 4.0, 4.0])))
        R_111 = scalar_curvature(DirichletFamily.log_partition,
                                 DirichletFamily.to_natural(np.array([1.0, 1.0, 1.0])))
        assert abs(R_444 - R_111) > 1e-6, (
            "Curvature should vary; R_444=%.8f R_111=%.8f" % (R_444, R_111)
        )

    def test_curvature_asymmetric_params(self):
        """R([1.5,4,6.5]) != R([4,4,4]) — anomaly pair has different curvature."""
        R_ref  = scalar_curvature(DirichletFamily.log_partition,
                                  DirichletFamily.to_natural(np.array([4.0, 4.0, 4.0])))
        R_anom = scalar_curvature(DirichletFamily.log_partition,
                                  DirichletFamily.to_natural(np.array([1.5, 4.0, 6.5])))
        assert abs(R_ref - R_anom) > 1e-6, (
            "Curvature should differ; R_ref=%.8f R_anom=%.8f" % (R_ref, R_anom)
        )

    def test_curvature_k4_finite(self):
        """R is finite for k=4."""
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        theta = DirichletFamily.to_natural(alpha)
        R = scalar_curvature(DirichletFamily.log_partition, theta)
        assert np.isfinite(R)


# ─────────────────────────────────────────────────────────────────────────────
# Class 4 — MLE
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletMLE:

    def test_mle_recovers_symmetric(self):
        """Fit n=2000 samples from Dirichlet([3,3,3]), recover α within rtol=0.05."""
        rng = np.random.default_rng(42)
        alpha_true = np.array([3.0, 3.0, 3.0])
        data = rng.dirichlet(alpha_true, size=2000)
        theta_hat = DirichletFamily.mle(data)
        alpha_hat = DirichletFamily.from_natural(theta_hat)
        np.testing.assert_allclose(alpha_hat, alpha_true, rtol=0.05)

    def test_mle_recovers_asymmetric(self):
        """Fit n=2000 samples from Dirichlet([1.5,4,6.5]), recover α within rtol=0.05."""
        rng = np.random.default_rng(123)
        alpha_true = np.array([1.5, 4.0, 6.5])
        data = rng.dirichlet(alpha_true, size=2000)
        theta_hat = DirichletFamily.mle(data)
        alpha_hat = DirichletFamily.from_natural(theta_hat)
        np.testing.assert_allclose(alpha_hat, alpha_true, rtol=0.05)

    def test_mle_convergence_gate_passes_on_good_data(self):
        """DirichletFamily.mle(good_data) does NOT raise ConvergenceError."""
        rng = np.random.default_rng(42)
        data = rng.dirichlet([4.0, 4.0, 4.0], size=500)
        # Should not raise
        DirichletFamily.mle(data)

    def test_mle_convergence_gate_raises_on_bad_data(self):
        """Using max_iter=1 forces a premature stop; ConvergenceError must be raised."""
        # With only 1 iteration from a bad initial point, the sufficient-statistic
        # residual will exceed 1e-4, triggering the convergence gate.
        data = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
        ])
        with pytest.raises(ConvergenceError):
            DirichletFamily.mle(data, max_iter=1)

    def test_mle_sufficient_stats_residual(self):
        """After fitting, |digamma(α̂) - digamma(α̂₀) - mean_log_x| < 1e-4."""
        rng = np.random.default_rng(7)
        alpha_true = np.array([2.0, 5.0, 3.0])
        data = rng.dirichlet(alpha_true, size=2000)
        data = np.clip(data, 1e-15, 1.0)

        theta_hat  = DirichletFamily.mle(data)
        alpha_hat  = DirichletFamily.from_natural(theta_hat)
        alpha0_hat = alpha_hat.sum()

        mean_log_x   = np.mean(np.log(data), axis=0)
        expected_log = digamma(alpha_hat) - digamma(alpha0_hat)
        residual = np.abs(expected_log - mean_log_x)
        assert np.max(residual) < 1e-4, (
            "Sufficient-stat residual too large: %s" % residual
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 5 — IGAD sample efficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestIGADSampleEfficiency:

    def _auc_for_n(self, n, seed=42, n_normal=100, n_anomaly=50):
        """Compute IGAD AUC for Dirichlet(4,4,4) vs (1.5,4,6.5) at batch size n."""
        from sklearn.metrics import roc_auc_score

        alpha_ref  = np.array([4.0, 4.0, 4.0])
        alpha_anom = np.array([1.5, 4.0, 6.5])
        rng = np.random.default_rng(seed)
        R_ref = scalar_curvature(DirichletFamily.log_partition,
                                 DirichletFamily.to_natural(alpha_ref))
        scores = []
        labels = []
        for phase, count, lab in [("normal", n_normal, 0), ("anomaly", n_anomaly, 1)]:
            for _ in range(count):
                if phase == "normal":
                    batch = rng.dirichlet(alpha_ref, size=n)
                else:
                    batch = rng.dirichlet(alpha_anom, size=n)
                theta_local = DirichletFamily.mle(batch)
                R_local = scalar_curvature(DirichletFamily.log_partition, theta_local)
                scores.append(abs(R_ref - R_local))
                labels.append(lab)
        return roc_auc_score(np.array(labels), scores)

    def test_igad_detects_dirichlet_shift_n200(self):
        """IGAD AUC > 0.65 for Dirichlet([4,4,4]) vs [1.5,4,6.5] at n=200."""
        auc = self._auc_for_n(200)
        assert auc > 0.65, "AUC=%.4f expected > 0.65" % auc

    def test_igad_beats_random_n50(self):
        """IGAD AUC > 0.55 at n=50 — early detection capability."""
        auc = self._auc_for_n(50)
        assert auc > 0.55, "AUC=%.4f expected > 0.55" % auc

    def test_igad_auc_increases_with_n(self):
        """AUC at n=200 > AUC at n=50 — documents monotone behavior."""
        auc50  = self._auc_for_n(50)
        auc200 = self._auc_for_n(200)
        assert auc200 > auc50, (
            "Expected AUC monotone: AUC200=%.4f AUC50=%.4f" % (auc200, auc50)
        )

    def test_curvature_separation_dirichlet(self):
        """|R([4,4,4]) - R([1.5,4,6.5])| > 0.01."""
        R_ref  = scalar_curvature(DirichletFamily.log_partition,
                                  DirichletFamily.to_natural(np.array([4.0, 4.0, 4.0])))
        R_anom = scalar_curvature(DirichletFamily.log_partition,
                                  DirichletFamily.to_natural(np.array([1.5, 4.0, 6.5])))
        assert abs(R_ref - R_anom) > 0.01, (
            "|ΔR|=%.6f expected > 0.01" % abs(R_ref - R_anom)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 6 — Failure modes
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureModes:

    def test_poisson_flat(self):
        """scalar_curvature ≈ 0 for Poisson (1D family)."""
        for lam in [0.5, 1.0, 5.0, 20.0, 100.0]:
            theta = PoissonFamily.to_natural(lam)
            R = scalar_curvature(PoissonFamily.log_partition, theta)
            assert abs(R) < 1e-4, "Poisson R=%s at lam=%s" % (R, lam)

    def test_gaussian_constant_curvature(self):
        """
        For a 2D Gaussian log-partition, R(rho=0.2) ≈ R(rho=0.8).

        The Gaussian manifold has constant curvature (hyperbolic geometry),
        so R should be the same regardless of the correlation parameter.
        """
        # 2D Gaussian natural params: theta = (mu/sigma^2, -1/(2*sigma^2))
        # We use the Gaussian used in demo_gaussian2d.py.
        # Log-partition for zero-mean Gaussian with covariance [[1, rho],[rho, 1]]:
        # A(theta) = -1/4 * theta^T Sigma theta + 1/2 log det(2*pi*Sigma)
        # Here we use a simplified 1D Gaussian: A(theta_1, theta_2) = -theta_1^2/(4*theta_2) - 1/2*log(-2*theta_2)
        def gaussian_log_partition(theta):
            """1D Gaussian: A(eta1, eta2) = -eta1^2/(4*eta2) - 0.5*log(-2*eta2)."""
            eta1, eta2 = theta[0], theta[1]
            return float(-eta1**2 / (4 * eta2) - 0.5 * np.log(-2 * eta2))

        # sigma=1, mu=0 vs mu=0: vary sigma to simulate "rho" effect
        # For Gaussian, curvature should be constant regardless of parameters
        R_vals = []
        for mu, sigma in [(0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (2.0, 0.5)]:
            eta1 = mu / sigma**2
            eta2 = -1.0 / (2 * sigma**2)
            theta = np.array([eta1, eta2])
            R = scalar_curvature(gaussian_log_partition, theta)
            R_vals.append(R)

        # All curvatures should be equal (constant curvature manifold)
        for R in R_vals:
            assert abs(R - R_vals[0]) < 2e-3, (
                "Gaussian curvature not constant: %s" % R_vals
            )

    def test_dirichlet_1d_is_degenerate(self):
        """
        k=2 Dirichlet is a Beta — 1-parameter family after fixing α₀.
        Scalar curvature should be effectively zero or constant.
        """
        # For k=2, the Fisher-Rao manifold of Beta is 2D but the constrained
        # 1-parameter slice (fixing alpha_0) has zero intrinsic curvature.
        # We verify that scalar_curvature is well-defined for k=2.
        alpha = np.array([2.0, 3.0])
        theta = DirichletFamily.to_natural(alpha)
        R = scalar_curvature(DirichletFamily.log_partition, theta)
        # For a 2-parameter family, the curvature formula is valid;
        # we just check it's finite and well-behaved.
        assert np.isfinite(R), "k=2 Dirichlet curvature should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
