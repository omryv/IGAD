"""
experiments/demo_moe_router.py

Does scalar curvature earn its place in an MoE router monitor?

Setup: every condition below has EXACTLY uniform mean expert load (1/k), so an
auxiliary load-balancing loss -- a first-moment constraint on expert load --
cannot separate any of them. What varies is alpha_0, the router's per-token
decisiveness:

    small alpha_0  ->  each token routed decisively (draws near simplex vertices)
    large alpha_0  ->  each token routed mushily    (draws near simplex centre)

The axis swept is the monitoring WINDOW (tokens per batch), because sample
efficiency is IGAD's actual claim: "AUC > 0.7 at n = 50-100 where MMD requires
n = 200-300" (docs/operational_envelope.md). At large windows every method
saturates at AUC 1.0 and the comparison is vacuous.

Four detectors compete on identical batches:

    IGAD      |R(alpha_hat) - R(alpha_ref)|      structured O(k^2) curvature
    MLE-a0    |alpha_0_hat - alpha_0_ref|        SAME MLE, curvature discarded
    Entropy   |H_bar(batch) - H_bar(ref)|        mean per-token routing entropy
    MMD       RBF-kernel MMD^2 vs reference pool

MLE-a0 is the control that matters, and it mirrors the MLE-skewness control in
demo_hard.py. IGAD beating Entropy proves only that fitting a Dirichlet beats a
single scalar summary. IGAD only vindicates the *geometry* if it also beats
MLE-a0, which pays for the identical MLE and then ignores the curvature tensor.

IMPORTANT -- feed the PRE-top-k softmax. Post-top-k weights are mostly exact
zeros; DirichletFamily.mle clips them to 1e-15, so log(x) = -34.5 dominates the
sufficient statistics and the resulting fit describes the top-k masking rather
than the routing behaviour.
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from igad.exceptions import ConvergenceError
from igad.families import DirichletFamily


# ── Defaults ─────────────────────────────────────────────────────────────────
# The repository carries no MoE configuration, so these stand in for a sparse
# backbone. Override on the command line to match a real run.
K_EXPERTS = 64
ALPHA_I_REF = 0.5           # reference decisiveness; alpha_0 = k * ALPHA_I_REF
ALPHA_I_ANOM = 0.52         # drifted decisiveness (calibrated to avoid saturation)
WINDOWS = [8, 16, 32, 64]   # tokens per monitoring window
N_BATCHES = 30              # normal batches (and the same number of anomalies)
SEEDS = [11, 23, 42]
MMD_POOL = 64               # reference rows for MMD; the estimator is O(n^2 k)
FAILED_FIT = 1e12           # finite sentinel for a non-converged MLE


# ── Detectors ────────────────────────────────────────────────────────────────

def mean_entropy(batch: np.ndarray) -> float:
    """Mean per-token routing entropy in nats."""
    x = np.clip(batch, 1e-12, 1.0)
    return float(np.mean(-np.sum(x * np.log(x), axis=1)))


def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """MMD^2 with an RBF kernel, median-heuristic bandwidth (as in demo_dirichlet)."""
    XY = np.vstack([X, Y])
    d2_all = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2_all[d2_all > 0]) + 1e-8

    def rbf(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * sigma2))

    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean())


# ── One (window, seed) cell ──────────────────────────────────────────────────

def evaluate(k, window, a_ref, a_anom, n_batches, mmd_pool, seed):
    rng = np.random.default_rng(seed)

    alpha_ref = np.full(k, a_ref)
    alpha_anom = np.full(k, a_anom)
    theta_ref = DirichletFamily.to_natural(alpha_ref)

    R_ref = DirichletFamily.scalar_curvature_analytical(theta_ref)
    a0_ref = float(alpha_ref.sum())

    ref_pool = rng.dirichlet(alpha_ref, size=max(window, mmd_pool))
    H_ref = mean_entropy(ref_pool)
    mmd_ref = ref_pool[:mmd_pool]

    scores = {"IGAD": [], "MLE-a0": [], "Entropy": [], "MMD": []}
    labels = []

    for is_anom in (0, 1):
        alpha = alpha_anom if is_anom else alpha_ref
        for _ in range(n_batches):
            batch = rng.dirichlet(alpha, size=window)
            try:
                theta_hat = DirichletFamily.mle(batch)
                alpha_hat = DirichletFamily.from_natural(theta_hat)
                R_hat = DirichletFamily.scalar_curvature_analytical(theta_hat)
                scores["IGAD"].append(abs(R_ref - R_hat))
                scores["MLE-a0"].append(abs(float(alpha_hat.sum()) - a0_ref))
            except (ConvergenceError, np.linalg.LinAlgError):
                # A failed fit is maximally anomalous. Use a large finite
                # sentinel rather than inf: roc_auc_score rejects non-finite
                # scores, and only the ranking matters here.
                scores["IGAD"].append(FAILED_FIT)
                scores["MLE-a0"].append(FAILED_FIT)
            scores["Entropy"].append(abs(mean_entropy(batch) - H_ref))
            scores["MMD"].append(mmd_rbf(batch, mmd_ref))
            labels.append(is_anom)

    return {m: roc_auc_score(labels, s) for m, s in scores.items()}


# ── Experiment ───────────────────────────────────────────────────────────────

def run(k, a_ref, a_anom, windows, n_batches, seeds, mmd_pool):
    methods = ["IGAD", "MLE-a0", "Entropy", "MMD"]

    print("=" * 78)
    print("MoE router monitor — uniform mean load throughout, window sweep")
    print("=" * 78)
    print("  experts k         : %d" % k)
    print("  alpha_i           : %.3f -> %.3f  (alpha_0 %.1f -> %.1f)"
          % (a_ref, a_anom, a_ref * k, a_anom * k))
    print("  R(ref)            : %.6f" % DirichletFamily.scalar_curvature_analytical(
        DirichletFamily.to_natural(np.full(k, a_ref))))
    print("  batches           : %d vs %d per (window, seed)" % (n_batches, n_batches))
    print("  seeds             : %s" % seeds)
    print("  mean expert load  : 1/%d in EVERY condition — load-balance loss is blind"
          % k)
    print()
    print("  mean AUC over seeds (std in parentheses)")
    print("  %-8s %-16s %-16s %-16s %-16s" % ("window", *methods))
    print("  " + "-" * 74)

    for window in windows:
        per_seed = [evaluate(k, window, a_ref, a_anom, n_batches, mmd_pool, s)
                    for s in seeds]
        cells = []
        for m in methods:
            v = np.array([r[m] for r in per_seed])
            cells.append("%.4f (%.3f)" % (v.mean(), v.std()))
        print("  %-8d %-16s %-16s %-16s %-16s" % (window, *cells))

    print()
    print("  Curvature earns its place only if IGAD beats MLE-a0, which pays for")
    print("  the same Dirichlet MLE and then throws the curvature tensor away.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=K_EXPERTS)
    p.add_argument("--alpha-ref", type=float, default=ALPHA_I_REF)
    p.add_argument("--alpha-anom", type=float, default=ALPHA_I_ANOM)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    p.add_argument("--batches", type=int, default=N_BATCHES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--mmd-pool", type=int, default=MMD_POOL)
    a = p.parse_args()
    run(a.experts, a.alpha_ref, a.alpha_anom, a.windows,
        a.batches, a.seeds, a.mmd_pool)


if __name__ == "__main__":
    main()
