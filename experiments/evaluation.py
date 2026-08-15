"""
experiments/evaluation.py

Evaluation metrics and the object-level statistical protocol for the MoE
early-warning experiment (brief sections 3, 5, 10, 11).

**No synthetic data is generated here.** These functions consume real labels
and real scores. `tests/test_evaluation.py` checks them against cases with
analytically known answers.

The one thing this module exists to enforce
-------------------------------------------
The independent statistical unit is the **generated object**, not the token.
One generated object contributes on the order of 10^6 router vectors; treating
those as independent inflates every confidence interval by roughly the square
root of that, and would let a detector look significant on a single object.

So every resampling routine here takes objects, not rows:

    object_bootstrap_ci     resamples object IDs, never windows
    object_train_test_split splits on object IDs, and refuses to leak
    paired_detector_test    compares two detectors on the same objects
    grouped_split           keeps objects sharing a conditioning ID together

Token windows may be used to *construct* an object's score -- that is what
`aggregate_object_scores` is for -- but never to count toward n.
"""

import math
import random

FAILED = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def roc_auc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties. 1 = anomaly."""
    pos = sum(1 for v in labels if v)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return FAILED
    ranks = _ranks(scores)
    rsum = sum(r for r, l in zip(ranks, labels) if l)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def roc_curve(labels, scores):
    """(fpr, tpr, threshold) triples, ordered by decreasing threshold."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    pos = sum(1 for v in labels if v)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return []
    tp = fp = 0
    out = [(0.0, 0.0, float("inf"))]
    i = 0
    while i < len(order):
        thr = scores[order[i]]
        while i < len(order) and scores[order[i]] == thr:
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        out.append((fp / neg, tp / pos, thr))
    return out


def pr_auc(labels, scores):
    """Average precision: sum over thresholds of (recall increment) * precision.

    Reported alongside ROC AUC because the brief's failure rates are expected
    to be imbalanced (10-20%), where ROC AUC flatters a detector that PR AUC
    does not.
    """
    pos = sum(1 for v in labels if v)
    if pos == 0 or pos == len(labels):
        return FAILED
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    prev_recall = 0.0
    total = 0.0
    i = 0
    while i < len(order):
        thr = scores[order[i]]
        while i < len(order) and scores[order[i]] == thr:
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        total += (recall - prev_recall) * precision
        prev_recall = recall
    return total


def sensitivity_at_fpr(labels, scores, max_fpr):
    """Best TPR achievable without exceeding `max_fpr`.

    The brief asks for 1%, 5% and 10%. With object-level n in the hundreds,
    1% FPR may be finer than the data can resolve -- `achievable_fpr_grid`
    reports the FPR values that actually exist for a given negative count, so
    a report can say "1% is below the resolution of this sample" instead of
    quoting a number that is really 0%.
    """
    curve = roc_curve(labels, scores)
    if not curve:
        return FAILED
    best = 0.0
    for fpr, tpr, _ in curve:
        if fpr <= max_fpr + 1e-12:
            best = max(best, tpr)
    return best


def achievable_fpr_grid(n_negative):
    """The FPR values a sample of this many negatives can actually express."""
    if n_negative <= 0:
        return []
    return [i / n_negative for i in range(n_negative + 1)]


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return FAILED
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else FAILED


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def cross_validated_r2(xs, ys, folds, objects=None):
    """Leave-one-fold-out R^2 of a univariate linear fit, split by object.

    `folds` is a list of lists of indices; use `object_kfold` to build them so
    the split is on objects rather than on rows.
    """
    if not folds:
        return FAILED
    ss_res = ss_tot = 0.0
    my_all = sum(ys) / len(ys)
    for held in folds:
        held = set(held)
        tr = [i for i in range(len(xs)) if i not in held]
        te = sorted(held)
        if len(tr) < 2 or not te:
            continue
        mx = sum(xs[i] for i in tr) / len(tr)
        my = sum(ys[i] for i in tr) / len(tr)
        den = sum((xs[i] - mx) ** 2 for i in tr)
        slope = (sum((xs[i] - mx) * (ys[i] - my) for i in tr) / den) if den else 0.0
        intercept = my - slope * mx
        for i in te:
            pred = intercept + slope * xs[i]
            ss_res += (ys[i] - pred) ** 2
            ss_tot += (ys[i] - my_all) ** 2
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Early warning (brief section 11)
# ─────────────────────────────────────────────────────────────────────────────

def earliest_warning(prefix_fractions, aucs, threshold=0.80):
    """Smallest prefix fraction at which AUC first reaches `threshold` and
    stays there for every later prefix.

    The "and stays there" clause matters: a detector that touches 0.80 at 10%
    of generation, falls to 0.6 at 20%, and recovers at 80% has not given a
    usable warning at 10%. Returns None if the threshold is never met.
    """
    pairs = sorted(zip(prefix_fractions, aucs))
    for i, (frac, auc) in enumerate(pairs):
        if auc is None or auc != auc or auc < threshold:
            continue
        if all(a is not None and a == a and a >= threshold
               for _, a in pairs[i:]):
            return frac
    return None


def warning_time_table(detector_aucs, prefix_fractions, threshold=0.80):
    """detector -> earliest prefix meeting `threshold`. None = never."""
    return {name: earliest_warning(prefix_fractions, aucs, threshold)
            for name, aucs in detector_aucs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Object-level protocol (brief section 3)
# ─────────────────────────────────────────────────────────────────────────────

class LeakageError(Exception):
    """Raised when a split would put one object on both sides."""


def aggregate_object_scores(rows, how="max"):
    """Collapse many window scores for one object into one object-level score.

    `rows` is a sequence of (object_id, score). Returns {object_id: score}.
    This is the only sanctioned way for token windows to influence the
    analysis: they build an object's score, and then the object is one
    observation.
    """
    grouped = {}
    for obj, score in rows:
        grouped.setdefault(obj, []).append(score)
    reducers = {"max": max, "mean": lambda v: sum(v) / len(v),
                "last": lambda v: v[-1], "first": lambda v: v[0]}
    if how not in reducers:
        raise ValueError("unknown aggregation %r; expected one of %s"
                         % (how, sorted(reducers)))
    return {obj: reducers[how](vals) for obj, vals in grouped.items()}


def object_train_test_split(object_ids, test_fraction=0.3, seed=0, groups=None):
    """Split on objects. `groups` maps object -> conditioning id.

    When `groups` is given, all objects sharing a conditioning id land on the
    same side: two seeds of the same input image are not independent, and
    splitting between them leaks the conditioning into the test set.
    """
    rng = random.Random(seed)
    ids = sorted(set(object_ids))
    if groups is None:
        units = [[i] for i in ids]
    else:
        by_group = {}
        for i in ids:
            by_group.setdefault(groups[i], []).append(i)
        units = [sorted(v) for _, v in sorted(by_group.items())]
    rng.shuffle(units)
    n_test = max(1, int(round(test_fraction * len(units)))) if units else 0
    test = [i for unit in units[:n_test] for i in unit]
    train = [i for unit in units[n_test:] for i in unit]
    overlap = set(train) & set(test)
    if overlap:
        raise LeakageError("objects on both sides: %s" % sorted(overlap)[:5])
    return sorted(train), sorted(test)


def object_kfold(object_ids, k=5, seed=0, groups=None):
    """k folds of object ids, keeping grouped objects together."""
    rng = random.Random(seed)
    ids = sorted(set(object_ids))
    if groups is None:
        units = [[i] for i in ids]
    else:
        by_group = {}
        for i in ids:
            by_group.setdefault(groups[i], []).append(i)
        units = [sorted(v) for _, v in sorted(by_group.items())]
    rng.shuffle(units)
    folds = [[] for _ in range(k)]
    for idx, unit in enumerate(units):
        folds[idx % k].extend(unit)
    return [sorted(f) for f in folds]


def object_bootstrap_ci(objects, labels, scores, statistic=roc_auc,
                        n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI, resampling **objects** with replacement.

    `objects`, `labels` and `scores` are parallel sequences with one entry per
    object. Resampling rows instead of objects is the pseudo-replication the
    brief forbids, and this function has no way to do it.
    """
    n = len(objects)
    if not (n == len(labels) == len(scores)):
        raise ValueError("objects, labels and scores must be parallel")
    rng = random.Random(seed)
    point = statistic(labels, scores)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = statistic([labels[i] for i in idx], [scores[i] for i in idx])
        if v == v:
            draws.append(v)
    if not draws:
        return {"point": point, "lo": FAILED, "hi": FAILED,
                "n_objects": n, "n_effective_draws": 0}
    draws.sort()
    lo = draws[int(alpha / 2 * (len(draws) - 1))]
    hi = draws[int((1 - alpha / 2) * (len(draws) - 1))]
    return {"point": point, "lo": lo, "hi": hi, "n_objects": n,
            "n_effective_draws": len(draws), "alpha": alpha}


def paired_detector_test(objects, labels, scores_a, scores_b,
                         statistic=roc_auc, n_boot=2000, seed=0):
    """Paired bootstrap on the *difference* between two detectors.

    Both detectors are evaluated on the same resampled objects, so the
    comparison is not contaminated by which objects happened to be drawn. The
    reported p-value is two-sided against the null that the difference is zero.
    """
    n = len(objects)
    rng = random.Random(seed)
    point = statistic(labels, scores_a) - statistic(labels, scores_b)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        a = statistic(la, [scores_a[i] for i in idx])
        b = statistic(la, [scores_b[i] for i in idx])
        if a == a and b == b:
            diffs.append(a - b)
    if not diffs:
        return {"difference": point, "lo": FAILED, "hi": FAILED,
                "p_two_sided": FAILED, "n_objects": n}
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    centred = [d - point for d in diffs]
    extreme = sum(1 for d in centred if abs(d) >= abs(point))
    return {"difference": point, "lo": lo, "hi": hi,
            "p_two_sided": (extreme + 1) / (len(centred) + 1),
            "n_objects": n, "n_effective_draws": len(diffs)}


def minimum_objects_for_auc(auc_alt, auc_null=0.5, power=0.8, alpha=0.05,
                            prevalence=0.15):
    """Rough object count needed to distinguish `auc_alt` from `auc_null`.

    Uses the Hanley-McNeil variance approximation. Deliberately reported as an
    order-of-magnitude planning number, not a promise: it assumes objects are
    independent and the AUC estimate is normal.
    """
    def hanley_var(a, n_pos, n_neg):
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        return (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)

    z_alpha = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.959963984540054
    z_beta = 0.8416212335729143 if abs(power - 0.8) < 1e-9 else 0.8416212335729143
    n = 10
    for _ in range(10000):
        n_pos = max(int(round(n * prevalence)), 2)
        n_neg = max(n - n_pos, 2)
        se = math.sqrt(hanley_var(auc_alt, n_pos, n_neg)
                       + hanley_var(auc_null, n_pos, n_neg))
        if se > 0 and abs(auc_alt - auc_null) / se >= z_alpha + z_beta:
            return {"n_objects": n, "n_failures": n_pos,
                    "n_successes": n_neg, "prevalence": prevalence,
                    "auc_alt": auc_alt, "auc_null": auc_null,
                    "power": power, "alpha": alpha}
        n += 10
    return {"n_objects": None, "note": "no n below 100000 reaches this power"}
