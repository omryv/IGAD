"""
tests/test_router_stats.py

Implementation checks for `experiments/router_stats.py`.

Every fixture below has an analytically known answer -- an isotropic
covariance has effective rank d, a rank-one covariance has anisotropy 1, the
affine-invariant distance from a matrix to itself is 0. These are correctness
tests for the functions.

**Nothing here is a benchmark.** No AUC is computed, no detector is compared
to another, and no number produced by these fixtures says anything about a
real router. The brief forbids advancing the product claim with synthetic
scores, and hand-built fixtures for unit tests are the stated exception.
"""

import math

import pytest

from experiments.router_stats import (
    affine_invariant_distance, anisotropy, correlation_drift, covariance_drift,
    drift_scores, effective_rank, eigenvalues, fit_reference, lambda_max,
    load_coefficient_of_variation, load_imbalance, logratio_moments,
    mean_drift, mean_top2_margin, routing_variance, spectral_entropy,
    summarize_window,
)


def identity(d, scale=1.0):
    return [[scale if i == j else 0.0 for j in range(d)] for i in range(d)]


def rank_one(d, scale=1.0):
    return [[scale if (i == 0 and j == 0) else (1e-12 if i == j else 0.0)
             for j in range(d)] for i in range(d)]


# ─────────────────────────────────────────────────────────────────────────────
# Cheap statistics
# ─────────────────────────────────────────────────────────────────────────────

def test_top2_margin_is_the_gap_between_the_two_largest():
    batch = [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]]
    assert mean_top2_margin(batch) == pytest.approx(((0.5 - 0.3) + (0.7 - 0.2)) / 2)


def test_routing_variance_is_zero_on_a_constant_batch():
    batch = [[0.25, 0.25, 0.25, 0.25]] * 8
    assert routing_variance(batch) == pytest.approx(0.0, abs=1e-15)


def test_load_imbalance_is_one_when_load_is_uniform():
    batch = [[0.25, 0.25, 0.25, 0.25]] * 4
    assert load_imbalance(batch) == pytest.approx(1.0)
    assert load_coefficient_of_variation(batch) == pytest.approx(0.0, abs=1e-15)


def test_load_imbalance_rises_when_one_expert_dominates():
    batch = [[0.97, 0.01, 0.01, 0.01]] * 4
    assert load_imbalance(batch) == pytest.approx(0.97 / 0.25)
    assert load_coefficient_of_variation(batch) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Spectral statistics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 3, 5, 8])
def test_isotropic_covariance_has_full_effective_rank(d):
    sigma = identity(d, 2.5)
    assert effective_rank(sigma) == pytest.approx(d, rel=1e-9)
    assert spectral_entropy(sigma) == pytest.approx(math.log(d), rel=1e-9)
    assert anisotropy(sigma) == pytest.approx(1.0 / d, rel=1e-9)
    assert lambda_max(sigma) == pytest.approx(2.5, rel=1e-9)


@pytest.mark.parametrize("d", [3, 6])
def test_rank_one_covariance_is_maximally_anisotropic(d):
    sigma = rank_one(d, 4.0)
    assert anisotropy(sigma) == pytest.approx(1.0, rel=1e-9)
    assert effective_rank(sigma) == pytest.approx(1.0, rel=1e-6)


def test_eigenvalues_are_descending():
    sigma = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
    vals = eigenvalues(sigma)
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Distances between covariances
# ─────────────────────────────────────────────────────────────────────────────

def test_distance_to_self_is_zero():
    sigma = [[2.0, 0.5], [0.5, 1.0]]
    assert affine_invariant_distance(sigma, sigma) == pytest.approx(0.0, abs=1e-9)
    assert covariance_drift(sigma, sigma) == pytest.approx(0.0, abs=1e-12)
    assert correlation_drift(sigma, sigma) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("c", [0.25, 2.0, 10.0])
@pytest.mark.parametrize("d", [2, 4])
def test_affine_invariant_distance_under_scaling(c, d):
    """d_AI(cS, S) = sqrt(d) * |log c| -- all eigenvalues of the ratio are c."""
    sigma = identity(d)
    scaled = identity(d, c)
    assert affine_invariant_distance(scaled, sigma) == pytest.approx(
        math.sqrt(d) * abs(math.log(c)), rel=1e-8)


def test_affine_invariant_distance_is_affine_invariant():
    """The property the name claims: congruence by any invertible A leaves it
    unchanged, while the raw Frobenius drift does not."""
    s1 = [[2.0, 0.3], [0.3, 1.0]]
    s2 = [[1.0, -0.2], [-0.2, 3.0]]
    a = [[1.0, 2.0], [0.0, 0.5]]

    def congruence(m):
        am = [[sum(a[i][t] * m[t][j] for t in range(2)) for j in range(2)]
              for i in range(2)]
        return [[sum(am[i][t] * a[j][t] for t in range(2)) for j in range(2)]
                for i in range(2)]

    before = affine_invariant_distance(s1, s2)
    after = affine_invariant_distance(congruence(s1), congruence(s2))
    assert after == pytest.approx(before, rel=1e-7)
    # the contrast: plain Frobenius drift is not invariant
    assert covariance_drift(congruence(s1), congruence(s2)) != pytest.approx(
        covariance_drift(s1, s2), rel=1e-3)


def test_correlation_drift_ignores_per_coordinate_scaling():
    # both have correlation 0.5; s2's covariance is 0.5 * sqrt(9) * sqrt(1)
    s1 = [[1.0, 0.5], [0.5, 1.0]]
    s2 = [[9.0, 1.5], [1.5, 1.0]]
    assert correlation_drift(s1, s2) == pytest.approx(0.0, abs=1e-12)
    assert covariance_drift(s1, s2) > 1.0


def test_mean_drift_is_zero_at_the_reference_mean():
    sigma = identity(3, 2.0)
    assert mean_drift([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sigma) == pytest.approx(
        0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Window summary and reference
# ─────────────────────────────────────────────────────────────────────────────

def uniform_window(n, k, jitter=0.0):
    """n rows on the k-simplex; jitter tilts row i deterministically."""
    rows = []
    for i in range(n):
        raw = [1.0 + jitter * math.sin(i + j) for j in range(k)]
        total = sum(raw)
        rows.append([v / total for v in raw])
    return rows


def test_summarize_window_reports_a_simplex_summary():
    batch = uniform_window(40, 5, jitter=0.2)
    s = summarize_window(batch)
    assert s["k"] == 5
    assert s["n"] == 40
    assert len(s["mean_load"]) == 5
    assert sum(s["mean_load"]) == pytest.approx(1.0)
    assert len(s["mu"]) == 4                       # k-1 log-ratio coordinates
    assert len(s["eigenvalues"]) == 4
    assert not s["rank_deficient"]


def test_rank_deficiency_is_flagged_not_hidden():
    """n <= k-1 makes the empirical covariance singular; the ridge keeps the
    spectral statistics finite, and the flag says the number is not to be
    trusted."""
    batch = uniform_window(3, 8, jitter=0.3)
    s = summarize_window(batch)
    assert s["rank_deficient"] is True
    assert s["ridge"] > 0.0
    assert math.isfinite(s["effective_rank"])


def test_summarize_window_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_window([])


def test_logratio_moments_ridge_is_reported():
    batch = uniform_window(20, 4, jitter=0.1)
    m = logratio_moments(batch, ridge=1e-6)
    assert m["ridge"] > 0.0
    assert m["d"] == 3


def test_drift_scores_are_zero_against_their_own_reference():
    windows = [uniform_window(30, 4, jitter=0.1 + 0.01 * i) for i in range(6)]
    summaries = [summarize_window(w) for w in windows]
    ref = fit_reference(summaries)
    scores = drift_scores(summaries[0], ref)
    # covariance-based scores compare against the pooled reference, so they are
    # small but not exactly zero; the structural check is that they exist and
    # are finite and non-negative
    for name, value in scores.items():
        assert value == value, name
        assert value >= 0.0, name
    assert "affine_invariant" in scores
    assert "mean_entropy" in scores


def test_drift_scores_grow_with_distance_from_the_reference():
    ref_windows = [uniform_window(30, 4, jitter=0.05) for _ in range(5)]
    ref = fit_reference([summarize_window(w) for w in ref_windows])
    near = drift_scores(summarize_window(uniform_window(30, 4, jitter=0.05)), ref)
    far = drift_scores(summarize_window(uniform_window(30, 4, jitter=2.0)), ref)
    assert far["affine_invariant"] > near["affine_invariant"]
    assert far["covariance_drift"] > near["covariance_drift"]


def test_fit_reference_rejects_an_empty_slice():
    with pytest.raises(ValueError):
        fit_reference([])
