"""
tests/test_router_common_mirror.py

The router experiments are written against the standard library so that they
can be executed in any environment, including ones where numpy is unavailable.
That buys reproducibility at the cost of a second implementation of the
Dirichlet routines, which could silently drift from `igad`.

These tests close that gap: whenever numpy IS installed, every mirrored routine
in experiments/_router_common.py is checked against the package itself. If the
mirror drifts, CI fails here.
"""

import math

import pytest

np = pytest.importorskip("numpy", reason="mirror is only checkable with numpy present")

from igad.curvature import scalar_curvature
from igad.families import DirichletFamily

from experiments._router_common import (
    digamma, trigamma, tetragamma, mat_inv, sym_eig, spd_log, spd_pow,
    dir_fisher, dir_cumulant_structure, dir_scalar_curvature, dir_mle,
    dir_suff_stat, auc,
)


ALPHAS = [
    [4.0, 4.0, 4.0],
    [1.5, 4.0, 6.5],
    [0.5, 0.5, 0.5],
    [0.1, 0.2, 0.7],
    [2.0, 3.0, 5.0, 7.0],
    [0.5497382] * 4,
    [8.0, 1.0, 1.0, 1.0],
    [50.0, 50.0, 50.0],
]


class TestSpecialFunctions:

    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 2.5, 7.0, 13.0, 60.0])
    def test_digamma(self, x):
        from scipy.special import digamma as sp
        assert abs(digamma(x) - float(sp(x))) < 1e-10

    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 2.5, 7.0, 13.0, 60.0])
    def test_trigamma(self, x):
        from scipy.special import polygamma as sp
        assert abs(trigamma(x) - float(sp(1, x))) < 1e-10 * max(1.0, abs(trigamma(x)))

    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 2.5, 7.0, 13.0, 60.0])
    def test_tetragamma(self, x):
        from scipy.special import polygamma as sp
        want = float(sp(2, x))
        assert abs(tetragamma(x) - want) < 1e-9 * max(1.0, abs(want))


class TestDirichletMirror:

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_fisher_metric(self, alpha):
        theta = DirichletFamily.to_natural(np.array(alpha))
        want = DirichletFamily.fisher_metric_analytical(theta)
        np.testing.assert_allclose(np.array(dir_fisher(alpha)), want, rtol=1e-9)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_cumulant_structure(self, alpha):
        theta = DirichletFamily.to_natural(np.array(alpha))
        c_want, d_want = DirichletFamily.third_cumulant_structure(theta)
        c_got, d_got = dir_cumulant_structure(alpha)
        assert abs(c_got - c_want) < 1e-9 * max(1.0, abs(c_want))
        np.testing.assert_allclose(np.array(d_got), d_want, rtol=1e-9)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_scalar_curvature(self, alpha):
        """The stdlib R must equal igad's, via the package's own general path."""
        theta = DirichletFamily.to_natural(np.array(alpha))
        want = scalar_curvature(
            DirichletFamily.log_partition, theta,
            g=DirichletFamily.fisher_metric_analytical(theta),
            T=DirichletFamily.third_cumulant_analytical(theta),
        )
        got = dir_scalar_curvature(alpha)
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
            "mirror=%r igad=%r alpha=%s" % (got, want, alpha)
        )

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_mle_agrees(self, alpha):
        """Both MLEs must land on the same alpha from the same batch.

        The two start the fixed point from different points, and at high
        concentration the likelihood is nearly flat along alpha_0: the 1e-4
        sufficient-statistic gate both enforce leaves alpha_0 determined only
        to about 1e-4 * alpha_0 (relative). The direction alpha / alpha_0 is
        well determined, so it is held tightly; alpha_0 is held to what the
        gate guarantees.
        """
        rng = np.random.default_rng(7)
        data = rng.dirichlet(np.array(alpha), size=400)
        want = DirichletFamily.from_natural(DirichletFamily.mle(data))
        got = np.array(dir_mle(dir_suff_stat([list(r) for r in data], len(alpha)), len(alpha)))
        np.testing.assert_allclose(got / got.sum(), want / want.sum(), rtol=1e-6)
        assert got.sum() == pytest.approx(want.sum(), rel=1e-4 * want.sum())


class TestLinalgMirror:

    MATS = [
        [[2.0, 0.3], [0.3, 1.1]],
        [[1.4, 0.5, 0.0], [0.5, 0.9, 0.2], [0.0, 0.2, 2.2]],
        [[5.0, -1.0], [-1.0, 0.4]],
    ]

    @pytest.mark.parametrize("M", MATS)
    def test_inverse(self, M):
        np.testing.assert_allclose(np.array(mat_inv(M)), np.linalg.inv(np.array(M)),
                                   rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("M", MATS)
    def test_symmetric_eigenvalues(self, M):
        vals, _ = sym_eig(M)
        want = sorted(np.linalg.eigvalsh(np.array(M)), reverse=True)
        np.testing.assert_allclose(np.array(vals), np.array(want), rtol=1e-9)

    @pytest.mark.parametrize("M", MATS)
    def test_spd_log_roundtrips(self, M):
        """exp(log(M)) == M, checked through the eigen route."""
        L = np.array(spd_log(M))
        w, V = np.linalg.eigh(L)
        back = V @ np.diag(np.exp(w)) @ V.T
        np.testing.assert_allclose(back, np.array(M), rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("M", MATS)
    def test_spd_pow_half(self, M):
        H = np.array(spd_pow(M, 0.5))
        np.testing.assert_allclose(H @ H, np.array(M), rtol=1e-8, atol=1e-10)


class TestAuc:

    def test_matches_sklearn(self):
        roc = pytest.importorskip("sklearn.metrics").roc_auc_score
        rng = np.random.default_rng(3)
        for _ in range(5):
            labels = [0] * 30 + [1] * 30
            scores = list(rng.normal(0, 1, 30)) + list(rng.normal(0.7, 1, 30))
            assert abs(auc(labels, scores) - roc(labels, scores)) < 1e-12

    def test_handles_ties(self):
        assert abs(auc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]) - 0.5) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
