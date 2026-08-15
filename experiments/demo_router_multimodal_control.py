"""
experiments/demo_router_multimodal_control.py

Phase 0 reproduction: multimodal routing vs its own best-fit single Dirichlet.

Condition B is an equal mixture of Dir(8,1,1,1) and its permutations.
Condition A is *defined* as Dir(alpha_hat), where alpha_hat is the population
best single-Dirichlet fit to that mixture. The Dirichlet MLE matches E[log x],
so the two conditions share the aggregate mean AND the sufficient statistic.

That makes the negative result structural rather than empirical:

    alpha_hat_A = alpha_hat_B   =>   R(alpha_hat_A) = R(alpha_hat_B)

because R is a deterministic function of the fitted parameters. Any statistic
of the form f(alpha_hat) inherits the same blindness -- MLE-a0 included.
Entropy and MMD are not functions of alpha_hat and are therefore not bound by it.

Regenerates docs/router_geometry.html stage 3. Raw output is written to
experiments/results/ before any figure is drawn.

    python -m experiments.demo_router_multimodal_control
"""

import argparse
import math
import random

from experiments._router_common import (
    ConvergenceError, FAILED_FIT, auc, cohens_d, digamma, dir_mle,
    dir_scalar_curvature, dir_suff_stat, mean_entropy, mean_load, mean_max_prob,
    mean_sd, mmd_rbf, print_table, sample_dirichlet, sample_dirichlet_mixture,
    save_results,
)

K = 4
MODES = [[8., 1., 1., 1.], [1., 8., 1., 1.], [1., 1., 8., 1.], [1., 1., 1., 8.]]
WINDOWS = [50, 100, 200, 400]
SEEDS = [11, 23, 42]
N_BATCHES = 40
MMD_POOL = 64

METHODS = ["IGAD", "MLE-a0", "Entropy", "MMD"]


def mixture_suff_stat():
    """Exact E[log x_i] under the equal mixture, by symmetry of the modes."""
    a0 = sum(MODES[0])
    hot, cold = MODES[0][0], MODES[0][1]
    p_hot = 1.0 / K
    return p_hot * (digamma(hot) - digamma(a0)) + (1 - p_hot) * (digamma(cold) - digamma(a0))


def best_single_dirichlet():
    """Solve psi(a) - psi(K a) = E[log x] for the symmetric population MLE."""
    target = mixture_suff_stat()
    lo, hi = 1e-6, 50.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (digamma(mid) - digamma(K * mid)) < target:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    resid = abs((digamma(a) - digamma(K * a)) - target)
    return [a] * K, target, resid


def expected_entropy(alpha):
    """E[H(x)] for x ~ Dir(alpha), in nats."""
    a0 = sum(alpha)
    return -sum((a / a0) * (digamma(a + 1.0) - digamma(a0 + 1.0)) for a in alpha)


def sample_condition(multimodal, n, rng, alpha_hat):
    if multimodal:
        return sample_dirichlet_mixture(MODES, n, rng)
    return sample_dirichlet(alpha_hat, n, rng)


def evaluate(window, n_batches, seed, mmd_pool, alpha_hat):
    rng = random.Random(seed)
    R_ref = dir_scalar_curvature(alpha_hat)
    a0_ref = sum(alpha_hat)

    ref_pool = sample_condition(False, max(window, mmd_pool), rng, alpha_hat)
    H_ref = mean_entropy(ref_pool)
    mmd_ref = ref_pool[:mmd_pool]

    scores = {m: [] for m in METHODS}
    labels = []
    for is_anom in (0, 1):
        for _ in range(n_batches):
            b = sample_condition(bool(is_anom), window, rng, alpha_hat)
            try:
                ahat = dir_mle(dir_suff_stat(b, K), K)
                scores["IGAD"].append(abs(R_ref - dir_scalar_curvature(ahat)))
                scores["MLE-a0"].append(abs(sum(ahat) - a0_ref))
            except (ConvergenceError, ZeroDivisionError, OverflowError, ValueError):
                scores["IGAD"].append(FAILED_FIT)
                scores["MLE-a0"].append(FAILED_FIT)
            scores["Entropy"].append(abs(mean_entropy(b) - H_ref))
            scores["MMD"].append(mmd_rbf(b[:mmd_pool], mmd_ref))
            labels.append(is_anom)

    return ({m: auc(labels, s) for m, s in scores.items()},
            {m: cohens_d(labels, s) for m, s in scores.items()})


def run(windows, seeds, n_batches, mmd_pool, check_n):
    alpha_hat, target, resid = best_single_dirichlet()
    rng = random.Random(20260815)

    mix = sample_dirichlet_mixture(MODES, check_n, rng)
    coh = sample_dirichlet(alpha_hat, check_n, rng)

    payload = {
        "experiment": "multimodal_control",
        "k": K, "modes": MODES, "seeds": seeds, "n_batches": n_batches,
        "mmd_pool": mmd_pool, "methods": METHODS,
        "E_log_x_mixture": target,
        "alpha_hat": alpha_hat, "alpha_hat_0": sum(alpha_hat),
        "alpha_hat_residual": resid,
        "R_alpha_hat": dir_scalar_curvature(alpha_hat),
        "R_single_mode": dir_scalar_curvature(MODES[0]),
        "E_entropy_mixture": sum(expected_entropy(m) for m in MODES) / len(MODES),
        "E_entropy_coherent": expected_entropy(alpha_hat),
        "ln_k": math.log(K),
        "empirical_check_n": check_n,
        "empirical": {
            "mean_load_mixture": mean_load(mix),
            "mean_load_coherent": mean_load(coh),
            "mean_entropy_mixture": mean_entropy(mix),
            "mean_entropy_coherent": mean_entropy(coh),
            "mean_max_prob_mixture": mean_max_prob(mix),
            "mean_max_prob_coherent": mean_max_prob(coh),
        },
        "cells": [],
    }

    print("E[log x_i] under the mixture (exact) = %.9f" % target)
    print("best single-Dirichlet fit  alpha_hat = (%.6f, ...)  alpha_hat_0 = %.6f"
          % (alpha_hat[0], sum(alpha_hat)))
    print("  residual on the sufficient statistic = %.2e" % resid)
    print("  R(alpha_hat) = %.6f      R(one mode) = %.6f"
          % (payload["R_alpha_hat"], payload["R_single_mode"]))
    print()
    print("Population entropies (exact, nats):  mixture %.6f   coherent %.6f   diff %.6f"
          % (payload["E_entropy_mixture"], payload["E_entropy_coherent"],
             abs(payload["E_entropy_mixture"] - payload["E_entropy_coherent"])))
    print("Entropy is NOT a function of alpha_hat, so the construction does not bind it.")
    print()
    e = payload["empirical"]
    print("Empirical check at n = %d:" % check_n)
    print("  mean load  mixture  %s" % [round(v, 4) for v in e["mean_load_mixture"]])
    print("  mean load  coherent %s" % [round(v, 4) for v in e["mean_load_coherent"]])
    print("  mean H     %.4f vs %.4f    mean max-prob %.4f vs %.4f"
          % (e["mean_entropy_mixture"], e["mean_entropy_coherent"],
             e["mean_max_prob_mixture"], e["mean_max_prob_coherent"]))
    print()

    rows = []
    for w in windows:
        per_seed = [evaluate(w, n_batches, s, mmd_pool, alpha_hat) for s in seeds]
        cell = {"window": w, "auc": {}, "auc_sd": {}, "cohens_d": {}}
        row = [w]
        for m in METHODS:
            mu, sd = mean_sd([a[m] for a, _ in per_seed])
            dmu, _ = mean_sd([d[m] for _, d in per_seed])
            cell["auc"][m], cell["auc_sd"][m], cell["cohens_d"][m] = mu, sd, dmu
            row.append("%.4f (%.3f)" % (mu, sd))
        payload["cells"].append(cell)
        rows.append(row)

    print_table("AUC by window - mean over seeds %s (sd)" % seeds,
                ["window"] + METHODS, rows)

    gap = [abs(c["auc"]["IGAD"] - c["auc"]["MLE-a0"]) for c in payload["cells"]]
    payload["max_igad_vs_a0_gap"] = max(gap)
    print("max |AUC(IGAD) - AUC(MLE-a0)| across windows = %.4f" % max(gap))
    print("Both are functions of the same alpha_hat; the residual gap is estimator noise.")
    print()

    path = save_results("multimodal_control", payload)
    print("raw results -> %s" % path)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batches", type=int, default=N_BATCHES)
    p.add_argument("--mmd-pool", type=int, default=MMD_POOL)
    p.add_argument("--check-n", type=int, default=4000)
    a = p.parse_args()
    run(a.windows, a.seeds, a.batches, a.mmd_pool, a.check_n)


if __name__ == "__main__":
    main()
