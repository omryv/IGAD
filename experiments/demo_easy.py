import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from experiments.demo_hard import gamma_scalar_curvature
from igad.families import GammaFamily


def run_demo(seed=42):
    rng = np.random.default_rng(seed)
    print("=" * 60)
    print("IGAD Demo: Skewness-Based Anomaly Detection")
    print("=" * 60)

    alpha_ref, beta_ref = 9.0, 3.0
    theta_ref = GammaFamily.to_natural(alpha_ref, beta_ref)
    R_ref = gamma_scalar_curvature(theta_ref)
    print("Reference: Gamma(9,3) mean=3.00 skew=0.667")
    print("  R_ref = %.6f" % R_ref)

    n_batches_normal = 100
    n_batches_anomaly = 50
    batch_size = 200

    scores = []
    labels = []

    print("Computing curvature for %d normal batches..." % n_batches_normal)
    for b in range(n_batches_normal):
        batch = rng.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
        theta_local = GammaFamily.mle(batch)
        R_local = gamma_scalar_curvature(theta_local)
        scores.append(abs(R_ref - R_local))
        labels.append(0)

    alpha_anom, beta_anom = 1.5, 0.5
    print("Computing curvature for %d anomalous batches..." % n_batches_anomaly)
    print("Anomaly:  Gamma(1.5,0.5) mean=3.00 skew=1.633")
    for b in range(n_batches_anomaly):
        batch = rng.gamma(alpha_anom, 1.0 / beta_anom, size=batch_size)
        theta_local = GammaFamily.mle(batch)
        R_local = gamma_scalar_curvature(theta_local)
        scores.append(abs(R_ref - R_local))
        labels.append(1)

    scores = np.array(scores)
    labels = np.array(labels)
    auc_igad = roc_auc_score(labels, scores)

    # Baseline: batch mean shift
    scores_mean = []
    rng2 = np.random.default_rng(seed)
    ref_mean = alpha_ref / beta_ref
    ref_std = np.sqrt(alpha_ref) / beta_ref
    for b in range(n_batches_normal):
        batch = rng2.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
        scores_mean.append(abs(np.mean(batch) - ref_mean) / ref_std)
    for b in range(n_batches_anomaly):
        batch = rng2.gamma(alpha_anom, 1.0 / beta_anom, size=batch_size)
        scores_mean.append(abs(np.mean(batch) - ref_mean) / ref_std)
    auc_mean = roc_auc_score(labels, scores_mean)

    # Baseline: batch variance shift
    scores_var = []
    rng3 = np.random.default_rng(seed)
    ref_var = alpha_ref / (beta_ref * beta_ref)
    for b in range(n_batches_normal):
        batch = rng3.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
        scores_var.append(abs(np.var(batch) - ref_var))
    for b in range(n_batches_anomaly):
        batch = rng3.gamma(alpha_anom, 1.0 / beta_anom, size=batch_size)
        scores_var.append(abs(np.var(batch) - ref_var))
    auc_var = roc_auc_score(labels, scores_var)

    # Baseline: batch skewness shift
    from scipy.stats import skew as sp_skew
    scores_skew = []
    rng4 = np.random.default_rng(seed)
    ref_skew = 2.0 / np.sqrt(alpha_ref)
    for b in range(n_batches_normal):
        batch = rng4.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
        scores_skew.append(abs(sp_skew(batch) - ref_skew))
    for b in range(n_batches_anomaly):
        batch = rng4.gamma(alpha_anom, 1.0 / beta_anom, size=batch_size)
        scores_skew.append(abs(sp_skew(batch) - ref_skew))
    auc_skew = roc_auc_score(labels, scores_skew)

    # Results
    print()
    print("%-30s %10s" % ("Method", "AUC-ROC"))
    print("-" * 42)
    for name, auc in [("IGAD (curvature)", auc_igad),
                      ("Batch mean shift", auc_mean),
                      ("Batch variance shift", auc_var),
                      ("Batch skewness shift", auc_skew)]:
        print("%-30s %10.4f" % (name, auc))

    # Diagnostics
    print()
    print("Curvature diagnostics:")
    print("  R(reference)       = %.6f" % R_ref)
    theta_anom = GammaFamily.to_natural(alpha_anom, beta_anom)
    R_anom = gamma_scalar_curvature(theta_anom)
    print("  R(anomaly truth)   = %.6f" % R_anom)
    print("  |R_ref - R_anom|   = %.6f" % abs(R_ref - R_anom))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_scores = scores[:n_batches_normal]
    a_scores = scores[n_batches_normal:]
    axes[0].hist(n_scores, bins=30, alpha=0.6, label="Normal", density=True)
    axes[0].hist(a_scores, bins=15, alpha=0.6, label="Anomaly", density=True)
    axes[0].set_title("IGAD Score (AUC=%.3f)" % auc_igad)
    axes[0].set_xlabel("|R_ref - R_local|")
    axes[0].legend()

    n_sk = scores_skew[:n_batches_normal]
    a_sk = scores_skew[n_batches_normal:]
    axes[1].hist(n_sk, bins=30, alpha=0.6, label="Normal", density=True)
    axes[1].hist(a_sk, bins=15, alpha=0.6, label="Anomaly", density=True)
    axes[1].set_title("Skewness Shift (AUC=%.3f)" % auc_skew)
    axes[1].set_xlabel("|skew - skew_ref|")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("docs/figures/exp1_easy_gamma_vs_gamma.png", dpi=150)
    print()
    print("Plot saved to docs/figures/exp1_easy_gamma_vs_gamma.png")


if __name__ == "__main__":
    run_demo()
