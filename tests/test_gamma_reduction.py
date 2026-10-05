"""
For the Gamma family, IGAD reduces to a re-scaling of the MLE skewness.

R depends on the shape alpha alone (the family is closed under rescaling,
which is an isometry of the Fisher metric) and is strictly monotone in it.
The MLE-skewness control 2/sqrt(alpha_hat) is a function of the same
alpha_hat, so the IGAD score carries no information the control lacks.

These tests pin that, so the claim "IGAD extracts shape information beyond
MLE skewness" cannot return for this family without a test failing.
See RESULTS.md, Hard Case, and experiments/demo_hard.py.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from igad import IGADDetector
from igad.families import GammaFamily

ALPHA_REF, BETA_REF = 8.0, 2.0


def _R(alpha, beta):
    d = IGADDetector(family=GammaFamily())
    return d._scalar_curvature(GammaFamily.to_natural(alpha, beta))


@pytest.mark.parametrize("alpha", [1.5, 2.0, 8.0, 8.7, 30.0])
def test_curvature_does_not_depend_on_rate(alpha):
    values = [_R(alpha, beta) for beta in [0.01, 0.5, 2.0, 50.0, 1e3]]
    np.testing.assert_allclose(values, values[0], rtol=1e-12, atol=0)


def test_curvature_is_strictly_monotone_in_shape():
    alphas = np.linspace(1.5, 50.0, 300)
    R = np.array([_R(a, 1.0) for a in alphas])
    assert np.all(np.diff(R) < 0)


def test_score_is_invariant_to_rescaling_the_batch():
    """Rescaling the data changes beta_hat but not alpha_hat, so the
    score -- like the MLE skewness -- is unchanged."""
    d = IGADDetector(family=GammaFamily()).fit(
        theta_ref=GammaFamily.to_natural(ALPHA_REF, BETA_REF))
    rng = np.random.default_rng(0)
    for _ in range(5):
        batch = rng.lognormal(1.327, 0.343, size=200)
        base = d.score_batch(batch)
        for c in [1e-3, 0.5, 7.0, 1e4]:
            assert d.score_batch(c * batch) == pytest.approx(base, rel=1e-8, abs=1e-15)


def test_score_ranks_identically_to_the_mle_skewness_control():
    """On each side of alpha_ref the IGAD score and the control order
    batches identically: one is a monotone re-scaling of the other."""
    d = IGADDetector(family=GammaFamily()).fit(
        theta_ref=GammaFamily.to_natural(ALPHA_REF, BETA_REF))
    skew_ref = 2.0 / np.sqrt(ALPHA_REF)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        batch = (rng.gamma(ALPHA_REF, 1.0 / BETA_REF, size=200) if i % 2
                 else rng.lognormal(1.327, 0.343, size=200))
        alpha_hat = GammaFamily.mle(batch)[0] + 1.0
        rows.append((alpha_hat, d.score_batch(batch),
                     abs(2.0 / np.sqrt(alpha_hat) - skew_ref)))
    rows = np.array(rows)
    for side in (rows[:, 0] > ALPHA_REF, rows[:, 0] < ALPHA_REF):
        assert side.sum() > 20
        rho, _ = spearmanr(rows[side, 1], rows[side, 2])
        assert rho == pytest.approx(1.0, abs=1e-12)
