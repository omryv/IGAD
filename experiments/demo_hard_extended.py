"""
experiments/demo_hard_extended.py

Re-run the Gamma(8,2) vs LogNormal experiment from demo_hard.py with two
additional baselines: MMD (RBF kernel, median heuristic) and Wasserstein (1D).
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_auc_score
from scipy.stats import skew as sp_skew
from scipy.stats import wasserstein_distance

from igad.curvature import scalar_curvature
from igad.families import GammaFamily


# ── Distribution parameters (same as demo_hard.py) ──────────────────────────
ALPHA_REF, BETA_REF = 8.0, 2.0
REF_MEAN  = ALPHA_REF / BETA_REF        # 4.0
REF_VAR   = ALPHA_REF / BETA_REF**2     # 2.0
REF_SKEW  = 2.0 / math.sqrt(ALPHA_REF) # 0.707

SIG2   = math.log(1 + REF_VAR / REF_MEAN**2)
SIG_LN = math.sqrt(SIG2)
MU_LN  = math.log(REF_MEAN) - SIG2 / 2

REF_SAMPLE_SIZE = 500  # fixed reference pool per seed for MMD/Wasserstein


# ── MMD ──────────────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """Maximum Mean Discrepancy with RBF kernel, median-heuristic bandwidth."""
    XY = np.vstack([X, Y])
    dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(dists[dists > 0]) + 1e-8

    def rbf(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * sigma2))

    m, n_b = len(X), len(Y)
    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (Kxx.sum() / (m * (m - 1)) +
            Kyy.sum() / (n_b * (n_b - 1)) -
            2.0 * Kxy.mean())
    return float(mmd2)


# ── One-seed evaluation ───────────────────────────────────────────────────────

def _scores_one_seed(seed, batch_size, n_normal=100, n_anomaly=50):
    """
    Run one seed. Returns dict of AUC values.

    Methods:
      igad      — |R_ref - R_local| (curvature deviation)
      skew_mle  — |skew_mle(batch) - skew_ref|  [CONTROL: same MLE, no geometry]
      mmd       — MMD^2 between batch and fixed reference sample
      wass      — Wasserstein_1 between batch and fixed reference sample
      skew_raw  — |scipy.stats.skew(batch) - skew_ref|
      mean      — |mean(batch) - ref_mean| / sqrt(ref_var)
      var       — |var(batch)  - ref_var|
    """
    rng     = np.random.default_rng(seed)
    rng_ref = np.random.default_rng(seed + 10000)  # independent RNG for reference pool

    theta_ref = GammaFamily.to_natural(ALPHA_REF, BETA_REF)
    R_ref     = scalar_curvature(GammaFamily.log_partition, theta_ref)

    # Fixed reference sample drawn ONCE per seed (size=500)
    ref_sample = rng_ref.gamma(ALPHA_REF, 1.0 / BETA_REF, size=REF_SAMPLE_SIZE)
    ref_1d = ref_sample.reshape(-1, 1)

    igad_scores     = []
    skew_mle_scores = []
    mmd_scores      = []
    wass_scores     = []
    skew_raw_scores = []
    mean_scores     = []
    var_scores      = []
    labels          = []

    for phase, count, lab in [("normal", n_normal, 0),
                               ("anomaly", n_anomaly, 1)]:
        for _ in range(count):
            if phase == "normal":
                batch = rng.gamma(ALPHA_REF, 1.0 / BETA_REF, size=batch_size)
            else:
                batch = rng.lognormal(MU_LN, SIG_LN, size=batch_size)
                batch = batch[batch > 0]

            # ── IGAD ──────────────────────────────────────────────────────
            theta_local = GammaFamily.mle(batch)
            R_local     = scalar_curvature(GammaFamily.log_partition, theta_local)
            igad_scores.append(abs(R_ref - R_local))

            # ── MLE skewness (CONTROL) ─────────────────────────────────────
            alpha_mle = theta_local[0] + 1.0
            skew_mle  = 2.0 / math.sqrt(alpha_mle)
            skew_mle_scores.append(abs(skew_mle - REF_SKEW))

            # ── MMD ────────────────────────────────────────────────────────
            batch_1d = batch.reshape(-1, 1)
            mmd_scores.append(mmd_rbf(batch_1d, ref_1d))

            # ── Wasserstein ────────────────────────────────────────────────
            wass_scores.append(wasserstein_distance(batch, ref_sample))

            # ── Raw skewness ───────────────────────────────────────────────
            skew_raw_scores.append(abs(sp_skew(batch) - REF_SKEW))

            # ── Mean / variance ───────────────────────────────────────────
            mean_scores.append(abs(np.mean(batch) - REF_MEAN) / math.sqrt(REF_VAR))
            var_scores.append(abs(np.var(batch)   - REF_VAR))

            labels.append(lab)

    labels = np.array(labels)
    return {
        "igad":     roc_auc_score(labels, igad_scores),
        "skew_mle": roc_auc_score(labels, skew_mle_scores),
        "mmd":      roc_auc_score(labels, mmd_scores),
        "wass":     roc_auc_score(labels, wass_scores),
        "skew_raw": roc_auc_score(labels, skew_raw_scores),
        "mean":     roc_auc_score(labels, mean_scores),
        "var":      roc_auc_score(labels, var_scores),
    }


def run_hard_extended():
    ln_mean = math.exp(MU_LN + SIG2 / 2)
    ln_var  = (math.exp(SIG2) - 1) * math.exp(2 * MU_LN + SIG2)
    ln_skew = (math.exp(SIG2) + 2) * math.sqrt(math.exp(SIG2) - 1)

    print("=" * 65)
    print("IGAD Extended Hard Test: Gamma(8,2) vs LogNormal + MMD + Wasserstein")
    print("=" * 65)
    print("Reference : Gamma(%.0f, %.0f)  mean=%.3f  var=%.3f  skew=%.3f"
          % (ALPHA_REF, BETA_REF, REF_MEAN, REF_VAR, REF_SKEW))
    print("Anomaly   : LogNormal(mu=%.3f, sigma=%.3f)" % (MU_LN, SIG_LN))
    print("            mean=%.3f  var=%.3f  skew=%.3f" % (ln_mean, ln_var, ln_skew))
    print("Note: mean AND variance are matched. Only higher-order structure differs.")
    print()

    SEEDS      = [42, 7, 123, 999, 2024]
    BATCH_SIZE = 200

    all_results = [_scores_one_seed(s, BATCH_SIZE) for s in SEEDS]

    methods = [
        ("IGAD (curvature)",          "igad"),
        ("MLE skewness [CONTROL]",    "skew_mle"),
        ("MMD (RBF, median BW)",      "mmd"),
        ("Wasserstein (1D)",           "wass"),
        ("Raw skewness",              "skew_raw"),
        ("Mean shift   [BLIND]",      "mean"),
        ("Variance shift [BLIND]",    "var"),
    ]

    print("%-30s  %8s  %8s" % ("Method", "Mean AUC", "± Std"))
    print("-" * 50)

    summary = {}
    for label, key in methods:
        vals = [r[key] for r in all_results]
        mu, sd = np.mean(vals), np.std(vals)
        summary[key] = (mu, sd)
        print("%-30s  %8.4f  %8.4f" % (label, mu, sd))

    print()
    igad_mu,    igad_sd    = summary["igad"]
    control_mu, control_sd = summary["skew_mle"]
    gap = igad_mu - control_mu
    print("Gap (IGAD − MLE skewness): %+.4f" % gap)
    if gap > 0:
        print("→ Curvature geometry adds signal BEYOND MLE efficiency alone.")
    else:
        print("→ MLE efficiency explains the advantage. Geometry adds nothing here.")
    print()


if __name__ == "__main__":
    run_hard_extended()
