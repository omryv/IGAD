"""
experiments/router_stats.py

Router statistics for the MoE early-warning experiment: the cheap operational
baselines the brief lists in section 6, and the structure-aware candidates it
lists in section 7.

**No synthetic data is generated here.** These are functions that consume real
router traces. `tests/test_router_stats.py` exercises them on small
hand-constructed fixtures whose correct answers are known analytically -- that
is implementation checking, not a benchmark, and no score from it means
anything about a real router.

Shape of the API
----------------
The brief's section 9 asks for reference states at several resolutions (per
layer, per timestep, per category), which means a statistic and a *reference*
have to be separable. So:

    summarize_window(batch)          -> dict of statistics for one window
    fit_reference(summaries)         -> a reference state, pooled over windows
    drift_scores(summary, reference) -> dict of name -> score

`drift_scores` is the only thing that produces a number a detector can rank on,
and it always needs a reference. That is deliberate: "mean entropy is 1.31" is
not a detection, and building the API so it cannot be mistaken for one keeps
section 6's baselines honest competitors rather than decoration.

Every score is oriented so that **larger means more anomalous**.

Two statistics the brief explicitly asks not to double-count
------------------------------------------------------------
Fisher-Rao geodesic distance on the fixed-mean covariance manifold is
`d_FR = (1/sqrt 2) * d_AI`, verified to 4.2e-7 in
`demo_router_logistic_normal_geometry.py`. They are the same detector with the
same ranking and the same AUC. Only `affine_invariant` is provided; the
geometry is the *reason* it is the natural statistic, not a second entry.

Scalar curvature is not here either. It lost to `affine_invariant` at every
window in the previous pass and is close to a relabelling of concentration
(rho(R, alpha_0) = 0.841). `igad` still computes it -- now in O(k) -- but it is
not a candidate monitor.
"""

import math

from experiments._router_common import (
    SingularMatrix, empirical_cov, frob, matmul, mean_entropy, mean_load,
    mean_max_prob, spd_log, spd_pow, sym_eig, to_logratio, trace,
)

TINY = 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 -- cheap operational diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def mean_top2_margin(batch):
    """E[p_(1) - p_(2)] -- how decisively the top expert wins."""
    total = 0.0
    for row in batch:
        top2 = sorted(row, reverse=True)[:2]
        total += top2[0] - (top2[1] if len(top2) > 1 else 0.0)
    return total / len(batch)


def routing_variance(batch):
    """tr(Cov(p)) on the raw simplex -- total routing variance."""
    _, cov = empirical_cov(batch)
    return trace(cov)


def load_imbalance(batch):
    """max_i load_i / mean_i load_i -- the classic MoE load-balancing statistic.

    NOTE: under **expert-choice** routing this is uniform by construction and
    carries no information. `experiments/trace_schema.py` records
    `routing_mode` for exactly this reason, and any report that includes this
    baseline must say which mode produced it.
    """
    load = mean_load(batch)
    mean = sum(load) / len(load)
    return max(load) / mean if mean > 0 else float("inf")


def load_coefficient_of_variation(batch):
    """sd(load) / mean(load) -- imbalance that uses every expert, not just the
    largest. Same expert-choice caveat as `load_imbalance`."""
    load = mean_load(batch)
    k = len(load)
    mean = sum(load) / k
    if mean <= 0:
        return float("inf")
    var = sum((v - mean) ** 2 for v in load) / k
    return math.sqrt(var) / mean


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 -- structure-aware statistics, in log-ratio coordinates
# ─────────────────────────────────────────────────────────────────────────────

def logratio_moments(batch, ridge=1e-9):
    """(mu, Sigma) of y_i = log(p_i / p_k), the identifiable simplex chart.

    A ridge of `ridge * tr(Sigma) / d` is added to the diagonal. With n rows
    and d = k-1 coordinates the empirical covariance is singular whenever
    n <= d, and every spectral statistic below then divides by zero. The ridge
    is reported in the returned dict so it can never be silently forgotten.
    """
    ys = [to_logratio(row) for row in batch]
    mu, cov = empirical_cov(ys)
    d = len(mu)
    tr = trace(cov)
    lam = ridge * (tr / d if d and tr > 0 else 1.0)
    for i in range(d):
        cov[i][i] += lam
    return {"mu": mu, "sigma": cov, "n": len(ys), "d": d, "ridge": lam,
            "rank_deficient": len(ys) <= d}


def eigenvalues(sigma):
    """Descending eigenvalues of a symmetric matrix."""
    vals, _ = sym_eig(sigma)
    return vals


def lambda_max(sigma):
    return eigenvalues(sigma)[0]


def anisotropy(sigma):
    """lambda_max / tr(Sigma) in [1/d, 1]. 1/d is isotropic, 1 is rank one."""
    vals = eigenvalues(sigma)
    tr = sum(vals)
    return vals[0] / tr if tr > 0 else float("nan")


def spectral_entropy(sigma):
    """H of the eigenvalue spectrum normalised to a probability vector."""
    vals = [max(v, 0.0) for v in eigenvalues(sigma)]
    tr = sum(vals)
    if tr <= 0:
        return float("nan")
    h = 0.0
    for v in vals:
        p = v / tr
        if p > TINY:
            h -= p * math.log(p)
    return h


def effective_rank(sigma):
    """exp(H(normalised spectrum)) -- a continuous count of active directions."""
    h = spectral_entropy(sigma)
    return math.exp(h) if h == h else float("nan")


def covariance_drift(sigma, sigma_ref):
    """||Sigma - Sigma_ref||_F. Scale-dependent; included because it is the
    obvious thing to try, and a structure-aware detector should have to beat
    it as well as beating entropy."""
    d = len(sigma)
    return frob([[sigma[i][j] - sigma_ref[i][j] for j in range(d)]
                 for i in range(d)])


def affine_invariant_distance(sigma, sigma_ref):
    """|| log( Sigma_ref^{-1/2} Sigma Sigma_ref^{-1/2} ) ||_F.

    Invariant under any common invertible linear reparameterisation of the
    log-ratio coordinates, which a raw Frobenius difference is not. This is
    also `sqrt(2) * d_FR`, the Fisher-Rao geodesic distance on the fixed-mean
    covariance manifold -- the same detector, so it is entered once.
    """
    half_inv = spd_pow(sigma_ref, -0.5)
    middle = matmul(matmul(half_inv, sigma), half_inv)
    return frob(spd_log(middle))


def correlation_matrix(sigma):
    d = len(sigma)
    sd = [math.sqrt(sigma[i][i]) if sigma[i][i] > 0 else TINY for i in range(d)]
    return [[sigma[i][j] / (sd[i] * sd[j]) for j in range(d)] for i in range(d)]


def correlation_drift(sigma, sigma_ref):
    """Frobenius drift of the correlation structure, with the marginal
    variances divided out -- separates "the experts became more variable" from
    "the experts changed how they co-vary"."""
    a, b = correlation_matrix(sigma), correlation_matrix(sigma_ref)
    d = len(a)
    return frob([[a[i][j] - b[i][j] for j in range(d)] for i in range(d)])


def mean_drift(mu, mu_ref, sigma_ref):
    """Mahalanobis distance between log-ratio means under the reference."""
    d = len(mu)
    diff = [mu[i] - mu_ref[i] for i in range(d)]
    inv = spd_pow(sigma_ref, -1.0)
    return math.sqrt(max(sum(diff[i] * inv[i][j] * diff[j]
                             for i in range(d) for j in range(d)), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Window summary, reference state, drift scores
# ─────────────────────────────────────────────────────────────────────────────

CHEAP_STATISTICS = ("mean_entropy", "mean_max_prob", "mean_top2_margin",
                    "routing_variance", "load_imbalance", "load_cv")
STRUCTURE_STATISTICS = ("lambda_max", "anisotropy", "spectral_entropy",
                        "effective_rank")


def summarize_window(batch, ridge=1e-9):
    """Every statistic of one window of pre-top-k routing vectors.

    `batch` is a sequence of full simplex rows -- pre-top-k, summing to 1. A
    post-top-k window has k - top_k structural zeros, the log-ratio transform
    then produces log(1e-15) entries, and every covariance statistic describes
    the masking rather than the routing.
    `experiments.trace_schema.verify_pre_topk_capture` is the check for that,
    and it should be run at capture time, not here.
    """
    if not batch:
        raise ValueError("empty window")
    moments = logratio_moments(batch, ridge)
    sigma = moments["sigma"]
    return {
        "n": len(batch),
        "k": len(batch[0]),
        "mean_load": mean_load(batch),
        "mean_entropy": mean_entropy(batch),
        "mean_max_prob": mean_max_prob(batch),
        "mean_top2_margin": mean_top2_margin(batch),
        "routing_variance": routing_variance(batch),
        "load_imbalance": load_imbalance(batch),
        "load_cv": load_coefficient_of_variation(batch),
        "mu": moments["mu"],
        "sigma": sigma,
        "ridge": moments["ridge"],
        "rank_deficient": moments["rank_deficient"],
        "eigenvalues": eigenvalues(sigma),
        "lambda_max": lambda_max(sigma),
        "anisotropy": anisotropy(sigma),
        "spectral_entropy": spectral_entropy(sigma),
        "effective_rank": effective_rank(sigma),
    }


def fit_reference(summaries):
    """Pool window summaries into a reference state.

    The brief's section 9 warns that a single pooled reference manufactures
    false anomalies when the router legitimately behaves differently by layer
    or by generation timestep. This function therefore fits *one* reference
    from whatever slice it is given; the caller is responsible for slicing by
    (layer, timestep, category) and calling it once per cell.
    `reference_key` below is the intended grouping helper.
    """
    if not summaries:
        raise ValueError("no summaries to fit a reference from")
    d = len(summaries[0]["mu"])
    n = len(summaries)
    mu = [sum(s["mu"][i] for s in summaries) / n for i in range(d)]
    sigma = [[sum(s["sigma"][i][j] for s in summaries) / n for j in range(d)]
             for i in range(d)]
    scalars = {}
    for name in CHEAP_STATISTICS + STRUCTURE_STATISTICS:
        vals = [s[name] for s in summaries if s[name] == s[name]]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)
        scalars[name] = {"mean": m, "sd": math.sqrt(var)}
    return {"n_windows": n, "d": d, "mu": mu, "sigma": sigma,
            "scalars": scalars}


def reference_key(record, by=("layer", "step_bucket")):
    """Grouping key for section 9's per-layer / per-timestep references."""
    return tuple(record[field] for field in by)


def drift_scores(summary, reference):
    """name -> score, larger = more anomalous.

    Scalar statistics become |z| against the reference distribution of that
    statistic, so a cheap baseline and a structure-aware statistic are on the
    same footing and can be compared by AUC without either being advantaged by
    its units.
    """
    scores = {}
    for name in CHEAP_STATISTICS + STRUCTURE_STATISTICS:
        ref = reference["scalars"].get(name)
        if ref is None or summary.get(name) is None:
            continue
        sd = ref["sd"]
        scores[name] = (abs(summary[name] - ref["mean"]) / sd
                        if sd > TINY else 0.0)
    sigma, sigma_ref = summary["sigma"], reference["sigma"]
    scores["covariance_drift"] = covariance_drift(sigma, sigma_ref)
    scores["correlation_drift"] = correlation_drift(sigma, sigma_ref)
    try:
        scores["affine_invariant"] = affine_invariant_distance(sigma, sigma_ref)
        scores["mean_drift"] = mean_drift(summary["mu"], reference["mu"],
                                          sigma_ref)
    except SingularMatrix:
        # a reference with a non-positive eigenvalue cannot define these; say
        # so rather than substituting a number
        scores["affine_invariant"] = float("nan")
        scores["mean_drift"] = float("nan")
    return scores


DETECTOR_FAMILIES = {
    "cheap": list(CHEAP_STATISTICS),
    "structure": list(STRUCTURE_STATISTICS) + [
        "covariance_drift", "correlation_drift", "affine_invariant",
        "mean_drift"],
}
