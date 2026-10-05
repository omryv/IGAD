# experiments/demo_gaussian2d.py
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from igad.curvature import scalar_curvature


# ── Zero-mean bivariate Gaussian family ──────────────────────────────────────
# Natural parameters: theta = (Theta_11, Theta_12, Theta_22)
# Theta = Sigma^{-1} = precision matrix
# log-partition: A(theta) = -1/2 * log(t0*t2 - t1^2) + log(2*pi)
#
# This is a d=3 family. Mean+marginal_variance do NOT determine theta uniquely.
# The correlation rho is a free parameter - this is where IGAD has an edge.

def _log_partition(theta):
    t0, t1, t2 = theta
    det = t0 * t2 - t1 ** 2
    if det <= 0:
        return np.inf
    return -0.5 * np.log(det) + np.log(2 * np.pi)


def _mle(data):
    """Precision MLE for zero-mean Gaussian: Theta = (S)^{-1}."""
    S = (data.T @ data) / len(data)
    det_S = S[0, 0] * S[1, 1] - S[0, 1] ** 2
    if det_S < 1e-12:
        raise ValueError("Singular sample covariance")
    P = np.array([[S[1, 1], -S[0, 1]],
                  [-S[1, 0],  S[0, 0]]]) / det_S
    return np.array([P[0, 0], P[0, 1], P[1, 1]])


def _to_natural(sigma1_sq, rho, sigma2_sq):
    """Covariance parameters → natural parameters."""
    cov = rho * math.sqrt(sigma1_sq * sigma2_sq)
    S = np.array([[sigma1_sq, cov], [cov, sigma2_sq]])
    det_S = S[0, 0] * S[1, 1] - S[0, 1] ** 2
    P = np.array([[S[1, 1], -S[0, 1]],
                  [-S[1, 0],  S[0, 0]]]) / det_S
    return np.array([P[0, 0], P[0, 1], P[1, 1]])


def _rho_from_theta(theta):
    """Extract correlation from natural parameters."""
    t0, t1, t2 = theta
    # Sigma = Theta^{-1}
    det = t0 * t2 - t1 ** 2
    sigma12 = -t1 / det
    sigma11 = t2 / det
    sigma22 = t0 / det
    return sigma12 / math.sqrt(sigma11 * sigma22)


# ── Experiment parameters ─────────────────────────────────────────────────────
# Reference:  N(0, Sigma_ref),  rho=0.2
# Anomaly:    N(0, Sigma_anom), rho=0.8
# Both:       mean=(0,0), marginal variances=(1,1)
# Difference: ONLY correlation
RHO_REF  = 0.2
RHO_ANOM = 0.8
S1, S2   = 1.0, 1.0   # marginal variances


def _sample_ref(rng, n):
    cov  = RHO_REF * math.sqrt(S1 * S2)
    Sigma = np.array([[S1, cov], [cov, S2]])
    return rng.multivariate_normal([0, 0], Sigma, size=n)


def _sample_anom(rng, n):
    cov  = RHO_ANOM * math.sqrt(S1 * S2)
    Sigma = np.array([[S1, cov], [cov, S2]])
    return rng.multivariate_normal([0, 0], Sigma, size=n)


def _scores_one_seed(seed, batch_size, n_normal=100, n_anomaly=50):
    rng = np.random.default_rng(seed)

    theta_ref = _to_natural(S1, RHO_REF, S2)
    R_ref     = scalar_curvature(_log_partition, theta_ref)

    igad_scores    = []
    rho_mle_scores = []   # CONTROL: same MLE, no geometry
    rho_raw_scores = []   # model-free sample correlation
    mean_scores    = []   # should be blind
    var_scores     = []   # should be blind
    labels         = []

    for phase, count, lab in [("normal", n_normal, 0),
                               ("anomaly", n_anomaly, 1)]:
        for _ in range(count):
            batch = (_sample_ref(rng, batch_size)
                     if phase == "normal"
                     else _sample_anom(rng, batch_size))

            # ── IGAD ──────────────────────────────────────────────────────
            theta_local = _mle(batch)
            R_local     = scalar_curvature(_log_partition, theta_local)
            igad_scores.append(abs(R_ref - R_local))

            # ── MLE correlation [CONTROL] ──────────────────────────────────
            # Extracts rho from the SAME MLE theta_local - no curvature used.
            # If IGAD ≈ this: geometry adds nothing.
            # If IGAD > this: curvature tensor is doing real work.
            rho_mle = _rho_from_theta(theta_local)
            rho_mle_scores.append(abs(rho_mle - RHO_REF))

            # ── Raw (model-free) correlation ───────────────────────────────
            rho_raw = np.corrcoef(batch[:, 0], batch[:, 1])[0, 1]
            rho_raw_scores.append(abs(rho_raw - RHO_REF))

            # ── Mean shift (should be BLIND) ───────────────────────────────
            mean_scores.append(np.linalg.norm(np.mean(batch, axis=0)))

            # ── Marginal variance shift (should be BLIND) ──────────────────
            var_scores.append(
                abs(np.var(batch[:, 0]) - S1) +
                abs(np.var(batch[:, 1]) - S2))

            labels.append(lab)

    labels = np.array(labels)
    return {
        "igad":    roc_auc_score(labels, igad_scores),
        "rho_mle": roc_auc_score(labels, rho_mle_scores),
        "rho_raw": roc_auc_score(labels, rho_raw_scores),
        "mean":    roc_auc_score(labels, mean_scores),
        "var":     roc_auc_score(labels, var_scores),
    }


def run_gaussian2d_demo():
    print("=" * 65)
    print("IGAD Gaussian-2D: d=3 family, correlation-only anomaly")
    print("=" * 65)
    print("Reference : N(0, Sigma)  rho=%.1f  marginal_var=(1,1)" % RHO_REF)
    print("Anomaly   : N(0, Sigma)  rho=%.1f  marginal_var=(1,1)" % RHO_ANOM)
    print("Mean detectors : BLIND (both have mean=0)")
    print("Variance detectors: BLIND (both have var=1)")
    print("Only correlation differs: %.1f vs %.1f" % (RHO_REF, RHO_ANOM))

    theta_ref = _to_natural(S1, RHO_REF, S2)
    R_ref     = scalar_curvature(_log_partition, theta_ref)
    theta_anom = _to_natural(S1, RHO_ANOM, S2)
    R_anom     = scalar_curvature(_log_partition, theta_anom)
    print()
    print("R(reference rho=0.2) = %.6f" % R_ref)
    print("R(anomaly   rho=0.8) = %.6f" % R_anom)
    print("|R_ref - R_anom|     = %.6f" % abs(R_ref - R_anom))

    # ── Part 1: 5 seeds, batch_size=200 ──────────────────────────────────
    SEEDS      = [42, 7, 123, 999, 2024]
    BATCH_SIZE = 200

    print()
    print("─" * 65)
    print("Part 1 - AUC-ROC over %d seeds  (batch_size=%d)"
          % (len(SEEDS), BATCH_SIZE))
    print("─" * 65)
    print("%-30s  %6s  %6s  %6s  %6s  %6s"
          % ("Method", *["s%d" % s for s in SEEDS]))
    print("-" * 65)

    all_results = [_scores_one_seed(s, BATCH_SIZE) for s in SEEDS]

    methods = [
        ("IGAD (curvature)",         "igad"),
        ("MLE correlation [CONTROL]","rho_mle"),
        ("Raw correlation",          "rho_raw"),
        ("Mean shift  [BLIND]",      "mean"),
        ("Variance shift  [BLIND]",  "var"),
    ]

    summary = {}
    for label, key in methods:
        vals = [r[key] for r in all_results]
        summary[key] = (np.mean(vals), np.std(vals))
        row = "%-30s  " % label
        row += "  ".join("%6.4f" % v for v in vals)
        print(row)

    print()
    print("%-30s  %8s  %8s" % ("Method", "Mean AUC", "± Std"))
    print("-" * 52)
    for label, key in methods:
        mu, sd = summary[key]
        print("%-30s  %8.4f  %8.4f" % (label, mu, sd))

    igad_mu,   igad_sd   = summary["igad"]
    ctrl_mu,   ctrl_sd   = summary["rho_mle"]
    gap = igad_mu - ctrl_mu
    print()
    print("Gap (IGAD − MLE-correlation): %+.4f" % gap)
    if gap > 0:
        print("→ Curvature geometry adds signal BEYOND MLE efficiency alone.")
        print("→ d=3 result: mean+variance detectors are BLIND. IGAD sees it.")
    else:
        print("→ MLE efficiency explains the advantage.")

    # ── Part 2: scaling ────────────────────────────────────────────────────
    print()
    print("─" * 65)
    print("Part 2 - Scaling with batch size  (seed=42)")
    print("─" * 65)
    print("%-6s  %8s  %10s  %10s  %8s"
          % ("n", "IGAD", "MLE-corr", "Raw-corr", "gap"))
    print("-" * 55)
    for bs in [100, 200, 500, 1000]:
        r = _scores_one_seed(42, bs)
        print("%-6d  %8.4f  %10.4f  %10.4f  %+8.4f"
              % (bs, r["igad"], r["rho_mle"], r["rho_raw"],
                 r["igad"] - r["rho_mle"]))

    # ── Plot ────────────────────────────────────────────────────────────────
    rng_plot   = np.random.default_rng(42)
    theta_ref  = _to_natural(S1, RHO_REF, S2)
    R_ref      = scalar_curvature(_log_partition, theta_ref)

    ig_n, ig_a, cm_n, cm_a, cr_n, cr_a = [], [], [], [], [], []
    for phase, count in [("n", 100), ("a", 50)]:
        for _ in range(count):
            batch = (_sample_ref(rng_plot, 200)
                     if phase == "n"
                     else _sample_anom(rng_plot, 200))
            theta_l  = _mle(batch)
            R_l      = scalar_curvature(_log_partition, theta_l)
            ig = abs(R_ref - R_l)
            cm = abs(_rho_from_theta(theta_l) - RHO_REF)
            cr = abs(np.corrcoef(batch[:,0], batch[:,1])[0,1] - RHO_REF)
            if phase == "n":
                ig_n.append(ig); cm_n.append(cm); cr_n.append(cr)
            else:
                ig_a.append(ig); cm_a.append(cm); cr_a.append(cr)

    lab_plot = [0]*100 + [1]*50
    auc_ig = roc_auc_score(lab_plot, ig_n + ig_a)
    auc_cm = roc_auc_score(lab_plot, cm_n + cm_a)
    auc_cr = roc_auc_score(lab_plot, cr_n + cr_a)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Gaussian-2D: rho=0.2 (normal) vs rho=0.8 (anomaly) - mean+var BLIND",
        fontsize=11)

    for ax, n_s, a_s, title, xlabel in [
        (axes[0], ig_n, ig_a,
         "IGAD  (AUC=%.3f)" % auc_ig,
         "|R_ref − R_local|"),
        (axes[1], cm_n, cm_a,
         "MLE correlation CONTROL  (AUC=%.3f)" % auc_cm,
         "|rho_MLE − rho_ref|"),
        (axes[2], cr_n, cr_a,
         "Raw correlation  (AUC=%.3f)" % auc_cr,
         "|corr_raw − rho_ref|"),
    ]:
        ax.hist(n_s, bins=25, alpha=0.6, label="Normal",  density=True)
        ax.hist(a_s, bins=12, alpha=0.6, label="Anomaly", density=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig("docs/figures/exp3_gaussian2d_correlation.png", dpi=150)
    print()
    print("Plot saved to docs/figures/exp3_gaussian2d_correlation.png")
    print()
    print("Interpretation key:")
    print("  Mean/Variance AUC ≈ 0.50  →  those detectors are completely blind")
    print("  IGAD > MLE-correlation     →  geometry adds signal beyond MLE")


if __name__ == "__main__":
    run_gaussian2d_demo()