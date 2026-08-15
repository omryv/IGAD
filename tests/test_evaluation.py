"""
tests/test_evaluation.py

Implementation checks for `experiments/evaluation.py`, on cases whose answers
are known without running anything: perfect separation is AUC 1, reversed
scores are AUC 0, all-ties is 0.5, a split must not leak.

The leakage tests are the point of the file. The brief's section 3 is the
easiest thing in the whole protocol to get wrong by accident, and it is not
detectable from the output -- a leaked split produces a plausible number.
"""

import math

import pytest

from experiments.evaluation import (
    LeakageError, achievable_fpr_grid, aggregate_object_scores,
    cross_validated_r2, earliest_warning, minimum_objects_for_auc,
    object_bootstrap_ci, object_kfold, object_train_test_split,
    paired_detector_test, pearson, pr_auc, roc_auc, roc_curve,
    sensitivity_at_fpr, spearman, warning_time_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_roc_auc_endpoints():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


def test_roc_auc_is_undefined_without_both_classes():
    assert math.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_roc_auc_handles_ties_with_average_ranks():
    # one positive and one negative share a score: half credit
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    # four (negative, positive) pairs: 1 + 1 + 0.5 (the tie) + 1, over 4
    assert roc_auc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)


def test_pr_auc_endpoints():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert math.isnan(pr_auc([1, 1], [0.1, 0.2]))
    # a useless detector's average precision approaches the prevalence
    labels = [1] * 10 + [0] * 90
    scores = [0.5] * 100
    assert pr_auc(labels, scores) == pytest.approx(0.10, abs=1e-9)


def test_pr_auc_is_the_metric_that_notices_imbalance():
    """ROC AUC and PR AUC diverge when positives are rare -- which is why the
    brief asks for both."""
    labels = [1] * 5 + [0] * 95
    # detector ranks all 5 positives just above the bulk, but 20 negatives beat them
    scores = [0.6] * 5 + [0.9] * 20 + [0.1] * 75
    assert roc_auc(labels, scores) > 0.75
    assert pr_auc(labels, scores) < 0.25


def test_sensitivity_at_fpr():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert sensitivity_at_fpr(labels, scores, 0.0) == pytest.approx(1.0)
    # a detector that puts one negative on top cannot reach TPR 1 at FPR 0
    scores_bad = [0.1, 0.2, 0.3, 0.95, 0.5, 0.6, 0.7, 0.8]
    assert sensitivity_at_fpr(labels, scores_bad, 0.0) == pytest.approx(0.0)
    assert sensitivity_at_fpr(labels, scores_bad, 0.25) == pytest.approx(1.0)


def test_achievable_fpr_grid_exposes_resolution_limits():
    """With 20 negatives, a 1% FPR target does not exist -- the report should
    say so rather than quote the 0% value."""
    grid = achievable_fpr_grid(20)
    assert 0.05 in grid
    assert not any(abs(v - 0.01) < 1e-12 for v in grid)
    assert achievable_fpr_grid(0) == []


def test_roc_curve_starts_at_origin_and_reaches_one():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    curve = roc_curve(labels, scores)
    assert curve[0][:2] == (0.0, 0.0)
    assert curve[-1][:2] == (1.0, 1.0)


def test_correlations():
    xs = [1, 2, 3, 4, 5]
    assert pearson(xs, [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert pearson(xs, [10, 8, 6, 4, 2]) == pytest.approx(-1.0)
    # monotone but not linear: Spearman 1, Pearson less
    ys = [1, 2, 4, 8, 16]
    assert spearman(xs, ys) == pytest.approx(1.0)
    assert pearson(xs, ys) < 0.99


# ─────────────────────────────────────────────────────────────────────────────
# Early warning
# ─────────────────────────────────────────────────────────────────────────────

def test_earliest_warning_requires_the_threshold_to_hold_afterwards():
    fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    # touches 0.85 at 0.1 then collapses -- that is not a warning at 0.1
    flaky = [0.5, 0.85, 0.55, 0.6, 0.9, 0.95]
    assert earliest_warning(fracs, flaky, 0.80) == pytest.approx(0.6)
    # monotone
    good = [0.6, 0.7, 0.82, 0.88, 0.9, 0.95]
    assert earliest_warning(fracs, good, 0.80) == pytest.approx(0.2)
    # never
    assert earliest_warning(fracs, [0.5] * 6, 0.80) is None


def test_warning_time_table_compares_detectors():
    fracs = [0.1, 0.2, 0.4, 1.0]
    table = warning_time_table(
        {"entropy": [0.5, 0.6, 0.7, 0.85],
         "covariance": [0.6, 0.84, 0.9, 0.95],
         "useless": [0.5, 0.5, 0.5, 0.5]},
        fracs, 0.80)
    assert table["covariance"] == pytest.approx(0.2)
    assert table["entropy"] == pytest.approx(1.0)
    assert table["useless"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Object-level protocol
# ─────────────────────────────────────────────────────────────────────────────

def test_aggregate_object_scores_collapses_windows_to_objects():
    rows = [("a", 0.1), ("a", 0.9), ("b", 0.4), ("b", 0.2)]
    assert aggregate_object_scores(rows, "max") == {"a": 0.9, "b": 0.4}
    assert aggregate_object_scores(rows, "mean") == {"a": pytest.approx(0.5),
                                                     "b": pytest.approx(0.3)}
    assert aggregate_object_scores(rows, "first") == {"a": 0.1, "b": 0.4}
    with pytest.raises(ValueError):
        aggregate_object_scores(rows, "median")


def test_split_never_puts_an_object_on_both_sides():
    ids = ["obj-%02d" % i for i in range(20)]
    train, test = object_train_test_split(ids, 0.3, seed=1)
    assert set(train) & set(test) == set()
    assert set(train) | set(test) == set(ids)
    assert len(test) == 6


def test_split_keeps_objects_sharing_a_conditioning_together():
    """Two seeds of the same input image are not independent."""
    ids = ["obj-%02d" % i for i in range(20)]
    groups = {i: "img-%d" % (int(i.split("-")[1]) // 2) for i in ids}
    train, test = object_train_test_split(ids, 0.3, seed=3, groups=groups)
    train_groups = {groups[i] for i in train}
    test_groups = {groups[i] for i in test}
    assert train_groups & test_groups == set()


def test_kfold_is_a_partition():
    ids = ["obj-%02d" % i for i in range(23)]
    folds = object_kfold(ids, k=5, seed=2)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))
    assert len(folds) == 5


def test_kfold_respects_groups():
    ids = ["obj-%02d" % i for i in range(24)]
    groups = {i: "img-%d" % (int(i.split("-")[1]) // 3) for i in ids}
    folds = object_kfold(ids, k=4, seed=5, groups=groups)
    seen = {}
    for fi, fold in enumerate(folds):
        for i in fold:
            seen.setdefault(groups[i], set()).add(fi)
    assert all(len(v) == 1 for v in seen.values())


def test_bootstrap_ci_resamples_objects_and_brackets_the_point():
    labels = [0] * 20 + [1] * 20
    scores = [0.1 + 0.01 * i for i in range(20)] + [0.4 + 0.01 * i for i in range(20)]
    objects = ["obj-%02d" % i for i in range(40)]
    ci = object_bootstrap_ci(objects, labels, scores, n_boot=400, seed=7)
    assert ci["n_objects"] == 40
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0.0 <= ci["lo"] <= 1.0 and 0.0 <= ci["hi"] <= 1.0


def test_bootstrap_rejects_non_parallel_inputs():
    with pytest.raises(ValueError):
        object_bootstrap_ci(["a", "b"], [0, 1], [0.1], n_boot=10)


def test_paired_test_on_identical_detectors_finds_no_difference():
    labels = [0] * 15 + [1] * 15
    scores = [0.1 * i for i in range(30)]
    objects = ["obj-%02d" % i for i in range(30)]
    out = paired_detector_test(objects, labels, scores, list(scores),
                               n_boot=300, seed=11)
    assert out["difference"] == pytest.approx(0.0)
    assert out["lo"] == pytest.approx(0.0)
    assert out["hi"] == pytest.approx(0.0)
    assert out["p_two_sided"] == pytest.approx(1.0)


def test_paired_test_detects_a_real_difference():
    labels = [0] * 25 + [1] * 25
    good = [0.0 + 0.01 * i for i in range(25)] + [1.0 + 0.01 * i for i in range(25)]
    useless = [0.5] * 50
    objects = ["obj-%02d" % i for i in range(50)]
    out = paired_detector_test(objects, labels, good, useless,
                               n_boot=400, seed=13)
    assert out["difference"] == pytest.approx(0.5)
    assert out["lo"] > 0.0
    assert out["p_two_sided"] < 0.05


def test_cross_validated_r2_on_a_perfect_linear_relationship():
    xs = [float(i) for i in range(20)]
    ys = [3.0 * x + 1.0 for x in xs]
    folds = object_kfold(list(range(20)), k=4, seed=0)
    r2 = cross_validated_r2(xs, ys, folds)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_cross_validated_r2_is_not_fooled_by_noise():
    xs = [float(i) for i in range(30)]
    ys = [1.0 if i % 2 else -1.0 for i in range(30)]     # unrelated to xs
    folds = object_kfold(list(range(30)), k=5, seed=0)
    assert cross_validated_r2(xs, ys, folds) < 0.2


def test_sample_size_planner_scales_with_effect_size():
    easy = minimum_objects_for_auc(0.90)
    hard = minimum_objects_for_auc(0.60)
    assert easy["n_objects"] < hard["n_objects"]
    assert hard["n_objects"] > 100
    assert easy["n_failures"] >= 2 and easy["n_successes"] >= 2
