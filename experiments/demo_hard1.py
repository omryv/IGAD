import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from experiments.demo_hard import gamma_scalar_curvature
from igad.families import GammaFamily
from scipy.stats import skew as sp_skew


def run_hard_demo(seed=42):
    rng = np.random.default_rng(seed)
    print("=" * 60)
    print("IGAD Hard Test: Same Mean AND Variance, Different Skewness")
    print("=" * 60)

    # Reference: Gamma(alpha, beta) with mean=m, var=v
    # mean = alpha/beta, var = alpha/beta^2
    # So: beta = mean/var, alpha = mean*beta = mean^2/var

    # We fix mean=4, var=2
    # Reference: alpha=8, beta=2  -> skew = 2/sqrt(8) = 0.707
    # To get same mean and var but different shape, we use a
    # MIXTURE that has mean=4, var=2 but different skewness.
    # Or: we use a slightly different approach.

    # Actually, within Gamma family, mean and var determine alpha,beta uniquely.
    # So we need to go outside Gamma for the anomaly.
    # Use a shifted/scaled distribution: LogNormal with matched mean and var.

    # Reference: Gamma(8, 2) -> mean=4, var=2, skew=2/sqrt(8)=0.707
    alpha_ref, beta_ref = 8.0, 2.0
    ref_mean = alpha_ref / beta_ref  # 4.0
    ref_var = alpha_ref / (beta_ref ** 2)  # 2.0
    ref_skew = 2.0 / np.sqrt(alpha_ref)  # 0.707

    # Anomaly: LogNormal(mu_ln, sigma_ln) with same mean=4, var=2
    # LogNormal: mean = exp(mu + sigma^2/2), var = (exp(sigma^2)-1)*exp(2mu+sigma^2)
    # Skewness = (exp(sigma^2)+2)*sqrt(exp(sigma^2)-1)
    # Solve: exp(mu+sigma^2/2) = 4, (exp(sigma^2)-1)*16 = 2  (since mean^2=16 for lognormal var formula)
    # From var: exp(sigma^2)-1 = 2/16 = 0.125 -> exp(sigma^2)=1.125 -> sigma^2=ln(1.125)
    import math
    sig2 = math.log(1.125)
    sig_ln = math.sqrt(sig2)
    mu_ln = math.log(4) - sig2 / 2
    # Verify
    ln_mean = math.exp(mu_ln + sig2 / 2)
    ln_var = (math.exp(sig2) - 1) * math.exp(2 * mu_ln + sig2)
    ln_skew = (math.exp(sig2) + 2) * math.sqrt(math.exp(sig2) - 1)

    print("Reference: Gamma(8,2)")
    print("  mean=%.3f, var=%.3f, skew=%.3f" % (ref_mean, ref_var, ref_skew))
    print("Anomaly:   LogNormal(mu=%.3f, sigma=%.3f)" % (mu_ln, sig_ln))
    print("  mean=%.3f, var=%.3f, skew=%.3f" % (ln_mean, ln_var, ln_skew))

    theta_ref = GammaFamily.to_natural(alpha_ref, beta_ref)
    R_ref = gamma_scalar_curvature(theta_ref)
    print("  R_ref = %.6f" % R_ref)

    n_normal = 100
    n_anomaly = 50
    batch_size = 200

    scores_igad = []
    scores_mean_test = []
    scores_var_test = []
    scores_skew_test = []
    labels = []

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    rng3 = np.random.default_rng(seed)
    rng4 = np.random.default_rng(seed)

    print("\nComputing %d normal + %d anomalous batches..." % (n_normal, n_anomaly))
    for phase in ["normal", "anomaly"]:
        count = n_normal if phase == "normal" else n_anomaly
        lab = 0 if phase == "normal" else 1
        for b in range(count):
            if phase == "normal":
                b1 = rng1.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
                b2 = rng2.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
                b3 = rng3.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
                b4 = rng4.gamma(alpha_ref, 1.0 / beta_ref, size=batch_size)
            else:
                b1 = rng1.lognormal(mu_ln, sig_ln, size=batch_size)
                b2 = rng2.lognormal(mu_ln, sig_ln, size=batch_size)
                b3 = rng3.lognormal(mu_ln, sig_ln, size=batch_size)
                b4 = rng4.lognormal(mu_ln, sig_ln, size=batch_size)

            # IGAD: fit Gamma MLE, compute curvature
            b1_pos = b1[b1 > 0]
            theta_local = GammaFamily.mle(b1_pos)
            R_local = gamma_scalar_curvature(theta_local)
            scores_igad.append(abs(R_ref - R_local))

            # Mean shift
            scores_mean_test.append(abs(np.mean(b2) - ref_mean) / np.sqrt(ref_var))

            # Variance shift
            scores_var_test.append(abs(np.var(b3) - ref_var))

            # Skewness shift
            scores_skew_test.append(abs(sp_skew(b4) - ref_skew))

            labels.append(lab)

    labels = np.array(labels)
    auc_igad = roc_auc_score(labels, scores_igad)
    auc_mean = roc_auc_score(labels, scores_mean_test)
    auc_var = roc_auc_score(labels, scores_var_test)
    auc_skew = roc_auc_score(labels, scores_skew_test)

    print("\n%-30s %10s" % ("Method", "AUC-ROC"))
    print("-" * 42)
    for name, auc in [("IGAD (curvature)", auc_igad),
                      ("Batch mean shift", auc_mean),
                      ("Batch variance shift", auc_var),
                      ("Batch skewness shift", auc_skew)]:
        marker = " <-- WINS" if auc == max(auc_igad, auc_mean, auc_var, auc_skew) else ""
        print("%-30s %10.4f%s" % (name, auc, marker))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_ig = scores_igad[:n_normal]
    a_ig = scores_igad[n_normal:]
    axes[0].hist(n_ig, bins=30, alpha=0.6, label="Normal", density=True)
    axes[0].hist(a_ig, bins=15, alpha=0.6, label="Anomaly", density=True)
    axes[0].set_title("IGAD (AUC=%.3f)" % auc_igad)
    axes[0].legend()

    n_sk = scores_skew_test[:n_normal]
    a_sk = scores_skew_test[n_normal:]
    axes[1].hist(n_sk, bins=30, alpha=0.6, label="Normal", density=True)
    axes[1].hist(a_sk, bins=15, alpha=0.6, label="Anomaly", density=True)
    axes[1].set_title("Skewness (AUC=%.3f)" % auc_skew)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("igad_hard_demo.png", dpi=150)
    print("\nPlot saved to igad_hard_demo.png")


if __name__ == "__main__":
    run_hard_demo()
