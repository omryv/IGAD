"""
experiments/demo_router_fixed_a0_anisotropy.py

Phase 0 reproduction: fixed-alpha_0 anisotropy.

Reference Dir(4,4,4,4) vs anomaly Dir(7,5,3,1). Both have alpha_0 = 16, so a
detector reading only concentration is blind by construction. The question is
whether curvature's win over that control marks out territory of its own.

Regenerates every number published in docs/router_geometry.html stage 2 and in
docs/moe_router.md. Raw output is written to experiments/results/ before any
figure is drawn.

    python -m experiments.demo_router_fixed_a0_anisotropy
"""

import argparse
import random

from experiments._router_common import (
    ConvergenceError, FAILED_FIT, auc, cohens_d, dir_mle, dir_scalar_curvature,
    dir_suff_stat, l1, mean_entropy, mean_load, mean_sd, mmd_rbf, print_table,
    sample_dirichlet, save_results,
)

K = 4
ALPHA_REF = [4.0, 4.0, 4.0, 4.0]
ALPHA_ANOM = [7.0, 5.0, 3.0, 1.0]
WINDOWS = [25, 50, 100, 200]
SEEDS = [11, 23, 42]
N_BATCHES = 40
MMD_POOL = 64

METHODS = ["IGAD", "MLE-a0", "MeanLoad", "Entropy", "MMD"]


def evaluate(window, n_batches, seed, mmd_pool):
    rng = random.Random(seed)
    R_ref = dir_scalar_curvature(ALPHA_REF)
    a0_ref = sum(ALPHA_REF)

    ref_pool = sample_dirichlet(ALPHA_REF, max(window, mmd_pool), rng)
    H_ref = mean_entropy(ref_pool)
    load_ref = mean_load(ref_pool)
    mmd_ref = ref_pool[:mmd_pool]

    scores = {m: [] for m in METHODS}
    labels = []
    for is_anom in (0, 1):
        alpha = ALPHA_ANOM if is_anom else ALPHA_REF
        for _ in range(n_batches):
            b = sample_dirichlet(alpha, window, rng)
            try:
                ahat = dir_mle(dir_suff_stat(b, K), K)
                scores["IGAD"].append(abs(R_ref - dir_scalar_curvature(ahat)))
                scores["MLE-a0"].append(abs(sum(ahat) - a0_ref))
            except (ConvergenceError, ZeroDivisionError, OverflowError, ValueError):
                scores["IGAD"].append(FAILED_FIT)
                scores["MLE-a0"].append(FAILED_FIT)
            scores["MeanLoad"].append(l1(mean_load(b), load_ref))
            scores["Entropy"].append(abs(mean_entropy(b) - H_ref))
            scores["MMD"].append(mmd_rbf(b[:mmd_pool], mmd_ref))
            labels.append(is_anom)

    return ({m: auc(labels, s) for m, s in scores.items()},
            {m: cohens_d(labels, s) for m, s in scores.items()})


def run(windows, seeds, n_batches, mmd_pool):
    R_ref = dir_scalar_curvature(ALPHA_REF)
    R_anom = dir_scalar_curvature(ALPHA_ANOM)
    a0 = sum(ALPHA_REF)

    payload = {
        "experiment": "fixed_a0_anisotropy",
        "k": K, "alpha_ref": ALPHA_REF, "alpha_anom": ALPHA_ANOM,
        "alpha_0": a0, "seeds": seeds, "n_batches": n_batches,
        "mmd_pool": mmd_pool, "methods": METHODS,
        "R_ref": R_ref, "R_anom": R_anom, "abs_dR": abs(R_ref - R_anom),
        "rel_dR_pct": 100.0 * abs(R_ref - R_anom) / abs(R_ref),
        "mean_ref": [a / a0 for a in ALPHA_REF],
        "mean_anom": [a / a0 for a in ALPHA_ANOM],
        "cells": [],
    }

    print("R(ref)   = %.6f   alpha=%s" % (R_ref, ALPHA_REF))
    print("R(anom)  = %.6f   alpha=%s" % (R_anom, ALPHA_ANOM))
    print("|dR|     = %.6f  (%.3f%% of R_ref)"
          % (payload["abs_dR"], payload["rel_dR_pct"]))
    print("alpha_0  = %.1f in both -> MLE-a0 blind by construction" % a0)
    print("mean     = %s  vs  %s -> mean load is NOT blind"
          % ([round(v, 4) for v in payload["mean_ref"]],
             [round(v, 4) for v in payload["mean_anom"]]))
    print()

    rows = []
    for w in windows:
        per_seed = [evaluate(w, n_batches, s, mmd_pool) for s in seeds]
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

    drows = [[c["window"]] + ["%+.2f" % c["cohens_d"][m] for m in METHODS]
             for c in payload["cells"]]
    print_table("Effect size (Cohen's d) on the raw detector scores",
                ["window"] + METHODS, drows)

    path = save_results("fixed_a0_anisotropy", payload)
    print("raw results -> %s" % path)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batches", type=int, default=N_BATCHES)
    p.add_argument("--mmd-pool", type=int, default=MMD_POOL)
    a = p.parse_args()
    run(a.windows, a.seeds, a.batches, a.mmd_pool)


if __name__ == "__main__":
    main()
