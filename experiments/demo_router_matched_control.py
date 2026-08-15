"""
experiments/demo_router_matched_control.py

Parts B, C, D1 and F — does synthetic router structure survive cheap controls,
can a richer model measure it, and does curvature add anything?

SCOPE LIMIT, STATED UP FRONT
    No MoE checkpoint, no 3D generator and no mesh tooling exist in this
    environment (torch/numpy/trimesh absent, PyPI and the Ubuntu archives
    return 403). Every router vector here is SYNTHETIC. That means this file
    can establish a POSSIBILITY result -- that structure surviving cheap
    controls can exist and can be measured -- and it cannot establish anything
    about real routers or about 3D-generation quality. Part E of the brief is
    untested, and the decision table records it as untested rather than
    substituting synthetic AUC for it.

CONSTRUCTION
    k experts, logits z ~ N(mu, Sigma), p = softmax(z).
    Condition A: Sigma isotropic in the gauge-free subspace, mu = 0, so
                 E[p] = 1/k exactly by exchange symmetry.
    Condition B: Sigma anisotropic with tr Sigma matched to A. mu is then
                 tuned numerically until E[p] = 1/k, and the scale of Sigma is
                 tuned until mean entropy matches A.
    Residual mismatch on every cheap statistic is measured and reported. A
    comparison is not called matched without that number.

    python -m experiments.demo_router_matched_control
"""

import argparse
import math
import random

from experiments._router_common import (
    ConvergenceError, FAILED_FIT, auc, cohens_d, dir_mle, dir_scalar_curvature,
    dir_suff_stat, empirical_cov, frob, l1, mat_inv, matmul, mean_entropy,
    mean_l2_mass, mean_load, mean_max_prob, mean_sd, mmd_rbf, print_table,
    sample_mvn, save_results, softmax, spd_log, spd_pow, sym_eig, to_logratio,
    trace,
)

K = 8
D = K - 1                      # identifiable log-ratio dimension
TUNE_N = 60000
WINDOWS = [32, 64, 128, 256]
SEEDS = [11, 23, 42, 57, 91]
N_BATCHES = 40
MMD_POOL = 48

METHODS = [
    "MeanLoad", "Entropy", "MaxProb", "L2Mass", "TraceCov",      # cheap scalars
    "LambdaMax", "SpectralEnt", "AffineInv",                     # structure-aware
    "MMD",                                                        # nonparametric
    "Dirichlet-a0", "IGAD-R",                                     # same-fit controls
]


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def iso_cov(scale):
    """Isotropic in the gauge-free subspace: scale * (I - J/K)."""
    return [[scale * ((1.0 if i == j else 0.0) - 1.0 / K) for j in range(K)]
            for i in range(K)]


def aniso_cov(scale, decay):
    """
    Anisotropic: eigenvalues decaying geometrically along a fixed orthonormal
    basis of the gauge-free subspace, then rescaled so the trace matches.
    """
    basis = gauge_free_basis()
    lam = [decay ** i for i in range(D)]
    s = sum(lam)
    lam = [scale * v / s * D / D for v in lam]
    C = [[0.0] * K for _ in range(K)]
    for a in range(D):
        v = basis[a]
        for i in range(K):
            for j in range(K):
                C[i][j] += lam[a] * v[i] * v[j]
    # rescale so trace equals `scale` * (D/D) * ... match to iso trace
    t_target = trace(iso_cov(scale))
    t_now = trace(C)
    f = t_target / t_now
    return [[C[i][j] * f for j in range(K)] for i in range(K)]


def gauge_free_basis():
    """Orthonormal basis of {v : sum v_i = 0} in R^K (Helmert)."""
    B = []
    for a in range(1, K):
        v = [1.0 / math.sqrt(a * (a + 1.0))] * a + [-a / math.sqrt(a * (a + 1.0))] \
            + [0.0] * (K - a - 1)
        B.append(v)
    return B


def jitter(C, eps=1e-9):
    """Ridge for numerical PD-ness. Dimension is taken from C, which is K x K
    for logit covariance but (K-1) x (K-1) in log-ratio coordinates."""
    n = len(C)
    return [[C[i][j] + (eps if i == j else 0.0) for j in range(n)] for i in range(n)]


def sample_router(mu, cov, n, rng):
    return [softmax(z) for z in sample_mvn(mu, jitter(cov), n, rng)]


def population_stats(mu, cov, n, rng):
    P = sample_router(mu, cov, n, rng)
    _, C = empirical_cov(P)
    return {
        "mean_load": mean_load(P),
        "entropy": mean_entropy(P),
        "max_prob": mean_max_prob(P),
        "l2_mass": mean_l2_mass(P),
        "trace_cov": trace(C),
    }, P


def tune_condition_b(scale_a, decay, rng, rounds=14, tune_n=TUNE_N):
    """
    Tune mu_B so E[p] = 1/K, and the scale of Sigma_B so mean entropy matches A.
    Both by fixed-point iteration on Monte-Carlo estimates.
    """
    cov_a = iso_cov(scale_a)
    ref, _ = population_stats([0.0] * K, cov_a, tune_n, random.Random(1))
    target_H = ref["entropy"]

    mu = [0.0] * K
    scale_b = scale_a
    for r in range(rounds):
        cov_b = aniso_cov(scale_b, decay)
        st, _ = population_stats(mu, cov_b, tune_n, random.Random(1000 + r))
        # multiplicative correction on the logit scale, then re-centre (gauge)
        mu = [mu[i] - math.log(st["mean_load"][i] * K) for i in range(K)]
        m = sum(mu) / K
        mu = [v - m for v in mu]
        # entropy is monotone decreasing in logit spread; nudge the scale
        st2, _ = population_stats(mu, cov_b, tune_n, random.Random(2000 + r))
        err = st2["entropy"] - target_H
        scale_b *= math.exp(0.9 * err)          # H too high -> spread more
    return mu, aniso_cov(scale_b, decay), scale_b, ref, target_H


# ─────────────────────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────────────────────

def logratio_cov(P):
    Y = [to_logratio(p) for p in P]
    _, C = empirical_cov(Y)
    return C


def spectrum(C):
    vals, _ = sym_eig(C)
    vals = [max(v, 1e-15) for v in vals]
    t = sum(vals)
    q = [v / t for v in vals]
    ent = -sum(v * math.log(v) for v in q)
    return vals, ent


def affine_invariant(C_ref_isqrt, C):
    return frob(spd_log(matmul(matmul(C_ref_isqrt, C), C_ref_isqrt)))


def score_batch(P, ref):
    """All detector scores for one window, as distances from the reference fit."""
    s = {}
    s["MeanLoad"] = l1(mean_load(P), ref["mean_load"])
    s["Entropy"] = abs(mean_entropy(P) - ref["entropy"])
    s["MaxProb"] = abs(mean_max_prob(P) - ref["max_prob"])
    s["L2Mass"] = abs(mean_l2_mass(P) - ref["l2_mass"])
    _, Cp = empirical_cov(P)
    s["TraceCov"] = abs(trace(Cp) - ref["trace_cov"])

    C = logratio_cov(P)
    try:
        vals, ent = spectrum(C)
        s["LambdaMax"] = abs(vals[0] - ref["lambda_max"])
        s["SpectralEnt"] = abs(ent - ref["spectral_entropy"])
        s["AffineInv"] = affine_invariant(ref["C_isqrt"], jitter(C, 1e-10))
    except Exception:
        s["LambdaMax"] = s["SpectralEnt"] = s["AffineInv"] = FAILED_FIT

    try:
        ahat = dir_mle(dir_suff_stat(P, K), K)
        s["Dirichlet-a0"] = abs(sum(ahat) - ref["alpha0"])
        s["IGAD-R"] = abs(dir_scalar_curvature(ahat) - ref["R"])
        s["_alpha0_raw"] = sum(ahat)
        s["_R_raw"] = dir_scalar_curvature(ahat)
    except (ConvergenceError, ZeroDivisionError, OverflowError, ValueError):
        s["Dirichlet-a0"] = s["IGAD-R"] = FAILED_FIT
        s["_alpha0_raw"] = s["_R_raw"] = float("nan")
    return s


def build_reference(mu_a, cov_a, n, rng):
    P = sample_router(mu_a, cov_a, n, rng)
    _, Cp = empirical_cov(P)
    C = logratio_cov(P)
    vals, ent = spectrum(C)
    ref = {
        "mean_load": mean_load(P), "entropy": mean_entropy(P),
        "max_prob": mean_max_prob(P), "l2_mass": mean_l2_mass(P),
        "trace_cov": trace(Cp), "lambda_max": vals[0], "spectral_entropy": ent,
        "C_isqrt": spd_pow(jitter(C, 1e-10), -0.5),
    }
    ahat = dir_mle(dir_suff_stat(P, K), K)
    ref["alpha0"] = sum(ahat)
    ref["R"] = dir_scalar_curvature(ahat)
    return ref, P


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def evaluate(window, seed, mu_a, cov_a, mu_b, cov_b, n_batches, mmd_pool,
             permute_labels=False, shuffle_experts=False):
    rng = random.Random(seed)
    ref, ref_pool = build_reference(mu_a, cov_a, max(window, 400), rng)
    mmd_ref = ref_pool[:mmd_pool]

    scores = {m: [] for m in METHODS}
    raw_a0, raw_R, labels = [], [], []
    for is_anom in (0, 1):
        mu, cov = (mu_b, cov_b) if is_anom else (mu_a, cov_a)
        for _ in range(n_batches):
            P = sample_router(mu, cov, window, rng)
            if shuffle_experts:
                perm = list(range(K))
                rng.shuffle(perm)
                P = [[p[perm[i]] for i in range(K)] for p in P]
            s = score_batch(P, ref)
            s["MMD"] = mmd_rbf(P[:mmd_pool], mmd_ref)
            for m in METHODS:
                scores[m].append(s[m])
            raw_a0.append(s["_alpha0_raw"])
            raw_R.append(s["_R_raw"])
            labels.append(is_anom)

    if permute_labels:
        rng.shuffle(labels)

    out_auc = {m: auc(labels, scores[m]) for m in METHODS}
    out_d = {m: cohens_d(labels, scores[m]) for m in METHODS}
    pairs = [(a, r) for a, r in zip(raw_a0, raw_R) if a == a and r == r]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 8 else float("nan")
    return out_auc, out_d, rho


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batches", type=int, default=N_BATCHES)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--decay", type=float, default=0.45)
    p.add_argument("--tune-n", type=int, default=TUNE_N)
    a = p.parse_args()

    print("=" * 78)
    print("CONSTRUCTION AND MATCHING (k=%d experts, log-ratio dim %d)" % (K, D))
    print("=" * 78)
    rng = random.Random(5)
    mu_a, cov_a = [0.0] * K, iso_cov(a.scale)
    mu_b, cov_b, scale_b, ref_stats, target_H = tune_condition_b(
        a.scale, a.decay, rng, tune_n=a.tune_n)

    st_a, _ = population_stats(mu_a, cov_a, a.tune_n, random.Random(77))
    st_b, _ = population_stats(mu_b, cov_b, a.tune_n, random.Random(78))

    rows, mism = [], {}
    for key, label in (("entropy", "mean entropy"), ("max_prob", "E[max_i p_i]"),
                       ("l2_mass", "E[||p||_2^2]"), ("trace_cov", "tr Cov(p)")):
        d = abs(st_a[key] - st_b[key])
        rel = d / max(1e-12, abs(st_a[key]))
        mism[key] = {"A": st_a[key], "B": st_b[key], "abs": d, "rel": rel}
        rows.append([label, "%.6f" % st_a[key], "%.6f" % st_b[key],
                     "%.2e" % d, "%.2f%%" % (100 * rel)])
    ml = max(abs(x - y) for x, y in zip(st_a["mean_load"], st_b["mean_load"]))
    mism["mean_load_max_abs"] = ml
    rows.insert(0, ["max_i |E_A p_i - E_B p_i|", "1/%d = %.6f" % (K, 1.0 / K),
                    "-", "%.2e" % ml, "%.2f%%" % (100 * ml * K)])
    print_table("RESIDUAL MISMATCH after tuning (n=%d per estimate)" % a.tune_n,
                ["statistic", "condition A", "condition B", "|diff|", "rel"], rows)

    ca, cb = logratio_cov(sample_router(mu_a, cov_a, 20000, random.Random(3))), \
             logratio_cov(sample_router(mu_b, cov_b, 20000, random.Random(4)))
    va, _ = spectrum(ca)
    vb, _ = spectrum(cb)
    print("  log-ratio covariance eigenvalues")
    print("    A: %s" % " ".join("%.3f" % v for v in va))
    print("    B: %s" % " ".join("%.3f" % v for v in vb))
    print("    lambda_max/trace   A %.3f   B %.3f"
          % (va[0] / sum(va), vb[0] / sum(vb)))
    print()

    payload = {"k": K, "d": D, "scale_a": a.scale, "scale_b": scale_b,
               "decay": a.decay, "mu_b": mu_b, "seeds": a.seeds,
               "n_batches": a.batches, "windows": a.windows,
               "mismatch": mism, "eig_A": va, "eig_B": vb, "cells": [],
               "controls": {}}

    # main sweep
    rows = []
    rhos = []
    for w in a.windows:
        per = [evaluate(w, s, mu_a, cov_a, mu_b, cov_b, a.batches, MMD_POOL)
               for s in a.seeds]
        cell = {"window": w, "auc": {}, "auc_sd": {}, "cohens_d": {}}
        row = [w]
        for m in METHODS:
            mu_, sd_ = mean_sd([x[0][m] for x in per])
            dd, _ = mean_sd([x[1][m] for x in per])
            cell["auc"][m], cell["auc_sd"][m], cell["cohens_d"][m] = mu_, sd_, dd
            row.append("%.3f±%.3f" % (mu_, sd_))
        rhos += [x[2] for x in per]
        payload["cells"].append(cell)
        rows.append(row)
    print_table("AUC by window -- mean ± sd over %d seeds, %d vs %d batches"
                % (len(a.seeds), a.batches, a.batches), ["win"] + METHODS, rows)

    r_mean, r_sd = mean_sd(rhos)
    payload["spearman_R_vs_alpha0"] = {"mean": r_mean, "sd": r_sd}
    print("  D1  Spearman rho( R(alpha_hat), alpha_hat_0 ) = %.4f ± %.4f  over %d cells"
          % (r_mean, r_sd, len(rhos)))
    print("      |rho| ~ 1 means curvature is a monotone relabelling of concentration")
    print("      inside this subfamily, hence identical AUC by construction.")
    print()

    # F controls at the largest window
    w = a.windows[-1]
    f5 = [evaluate(w, s, mu_a, cov_a, mu_b, cov_b, a.batches, MMD_POOL,
                   permute_labels=True) for s in a.seeds[:3]]
    f1 = [evaluate(w, s, mu_a, cov_a, mu_b, cov_b, a.batches, MMD_POOL,
                   shuffle_experts=True) for s in a.seeds[:3]]
    base = payload["cells"][-1]["auc"]
    rows = []
    for m in METHODS:
        m5, _ = mean_sd([x[0][m] for x in f5])
        m1, _ = mean_sd([x[0][m] for x in f1])
        rows.append([m, "%.3f" % base[m], "%.3f" % m5, "%.3f" % m1])
    payload["controls"]["F5_label_permutation"] = {
        m: mean_sd([x[0][m] for x in f5])[0] for m in METHODS}
    payload["controls"]["F1_expert_shuffle"] = {
        m: mean_sd([x[0][m] for x in f1])[0] for m in METHODS}
    print_table("F CONTROLS at window %d  (F5 must collapse to ~0.50; "
                "F1 must leave permutation-invariant detectors unchanged)" % w,
                ["detector", "baseline AUC", "F5 labels permuted",
                 "F1 experts shuffled"], rows)

    path = save_results("router_matched_control", payload)
    print("raw results -> %s" % path)


if __name__ == "__main__":
    main()
