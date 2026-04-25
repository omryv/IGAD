import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from igad.curvature import scalar_curvature
from igad.families import GammaFamily
from scipy.stats import skew as sp_skew


# ── Distribution parameters ──────────────────────────────────────────────────
ALPHA_REF, BETA_REF = 8.0, 2.0          # Gamma reference
REF_MEAN  = ALPHA_REF / BETA_REF        # 4.0
REF_VAR   = ALPHA_REF / BETA_REF**2     # 2.0
REF_SKEW  = 2.0 / math.sqrt(ALPHA_REF) # 0.707

# LogNormal anomaly: matched mean AND variance
SIG2  = math.log(1 + REF_VAR / REF_MEAN**2)   # ln(1.125)
SIG_LN = math.sqrt(SIG2)                        # 0.343
MU_LN  = math.log(REF_MEAN) - SIG2 / 2         # 1.327


def _verify_lognormal():
    """Confirm mean/var match analytically."""
    ln_mean = math.exp(MU_LN + SIG2 / 2)
    ln_var  = (math.exp(SIG2) - 1) * math.exp(2 * MU_LN + SIG2)
    ln_skew = (math.exp(SIG2) + 2) * math.sqrt(math.exp(SIG2) - 1)
    assert abs(ln_mean - REF_MEAN) < 1e-6, "mean mismatch"
    assert abs(ln_var  - REF_VAR)  < 1e-6, "var mismatch"
    return ln_mean, ln_var, ln_skew


def _scores_one_seed(seed, batch_size, n_normal=100, n_anomaly=50):
    """
    Run one seed. Returns dict of score arrays + label array.

    Baselines:
      igad      — |R_ref - R_local|  (curvature deviation)
      skew_mle  — |skew_mle(batch) - skew_ref|
                  where skew_mle = 2/sqrt(alpha_mle)
                  *** KEY CONTROL: same MLE, no geometry ***
      skew_raw  — |scipy.stats.skew(batch) - skew_ref|
                  model-free sample skewness
      mean      — |mean(batch) - ref_mean| / sqrt(ref_var)
      var       — |var(batch)  - ref_var|
    """
    rng = np.random.default_rng(seed)

    theta_ref = GammaFamily.to_natural(ALPHA_REF, BETA_REF)
    R_ref     = scalar_curvature(GammaFamily.log_partition, theta_ref)

    igad_scores     = []
    skew_mle_scores = []
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
                # lognormal is always positive; the filter below is a no-op guard
                batch = batch[batch > 0]

            # ── IGAD ──────────────────────────────────────────────────────
            theta_local = GammaFamily.mle(batch)
            R_local     = scalar_curvature(GammaFamily.log_partition, theta_local)
            igad_scores.append(abs(R_ref - R_local))

            # ── MLE skewness (CONTROL BASELINE) ───────────────────────────
            # Uses the SAME MLE theta_local but ignores the geometry.
            # If this matches IGAD performance → geometry adds nothing.
            # If IGAD beats this → the curvature tensor is doing real work.
            alpha_mle = theta_local[0] + 1.0   # natural param: theta_0 = alpha - 1
            skew_mle  = 2.0 / math.sqrt(alpha_mle)
            skew_mle_scores.append(abs(skew_mle - REF_SKEW))

            # ── Raw (model-free) skewness ──────────────────────────────────
            skew_raw_scores.append(abs(sp_skew(batch) - REF_SKEW))

            # ── Mean / variance ───────────────────────────────────────────
            mean_scores.append(abs(np.mean(batch) - REF_MEAN) / math.sqrt(REF_VAR))
            var_scores.append(abs(np.var(batch)   - REF_VAR))

            labels.append(lab)

    labels = np.array(labels)
    return {
        "igad":     roc_auc_score(labels, igad_scores),
        "skew_mle": roc_auc_score(labels, skew_mle_scores),
        "skew_raw": roc_auc_score(labels, skew_raw_scores),
        "mean":     roc_auc_score(labels, mean_scores),
        "var":      roc_auc_score(labels, var_scores),
    }


def run_hard_demo():
    ln_mean, ln_var, ln_skew = _verify_lognormal()

    print("=" * 65)
    print("IGAD Hard Test: Matched Mean AND Variance — Geometry vs MLE")
    print("=" * 65)
    print("Reference : Gamma(%.0f, %.0f)  mean=%.3f  var=%.3f  skew=%.3f"
          % (ALPHA_REF, BETA_REF, REF_MEAN, REF_VAR, REF_SKEW))
    print("Anomaly   : LogNormal(mu=%.3f, sigma=%.3f)"
          % (MU_LN, SIG_LN))
    print("            mean=%.3f  var=%.3f  skew=%.3f" % (ln_mean, ln_var, ln_skew))
    print()

    # ── Part 1: single seed, batch_size=200 ──────────────────────────────
    SEEDS      = [42, 7, 123, 999, 2024]
    BATCH_SIZE = 200

    print("─" * 65)
    print("Part 1 — AUC-ROC over %d seeds  (batch_size=%d)"
          % (len(SEEDS), BATCH_SIZE))
    print("─" * 65)
    print("%-28s  %6s  %6s  %6s  %6s  %6s"
          % ("Method", *[f"s{s}" for s in SEEDS]))
    print("-" * 65)

    all_results = [_scores_one_seed(s, BATCH_SIZE) for s in SEEDS]

    methods = [
        ("IGAD (curvature)",        "igad"),
        ("MLE skewness  [CONTROL]", "skew_mle"),
        ("Raw skewness",            "skew_raw"),
        ("Mean shift",              "mean"),
        ("Variance shift",          "var"),
    ]

    summary = {}
    for label, key in methods:
        vals = [r[key] for r in all_results]
        summary[key] = (np.mean(vals), np.std(vals))
        row = "%-28s  " % label
        row += "  ".join("%6.4f" % v for v in vals)
        print(row)

    print()
    print("%-28s  %8s  %8s" % ("Method", "Mean AUC", "± Std"))
    print("-" * 50)
    for label, key in methods:
        mu, sd = summary[key]
        wins = " ◄ IGAD > CONTROL" if key == "skew_mle" else ""
        print("%-28s  %8.4f  %8.4f%s" % (label, mu, sd, wins))

    print()
    igad_mu,    igad_sd    = summary["igad"]
    control_mu, control_sd = summary["skew_mle"]
    gap = igad_mu - control_mu
    print("Gap (IGAD − MLE skewness): %+.4f" % gap)
    if gap > 0:
        print("→ Curvature geometry adds signal BEYOND MLE efficiency alone.")
    else:
        print("→ MLE efficiency explains the advantage. Geometry adds nothing here.")

    # ── Part 2: scaling across batch sizes (single seed) ─────────────────
    print()
    print("─" * 65)
    print("Part 2 — Scaling with batch size  (seed=42)")
    print("─" * 65)
    print("%-6s  %8s  %10s  %10s  %8s"
          % ("n", "IGAD", "MLE-skew", "Raw-skew", "gap"))
    print("-" * 55)
    for bs in [100, 200, 500, 1000]:
        r = _scores_one_seed(42, bs)
        gap_ctrl = r["igad"] - r["skew_mle"]
        gap_raw  = r["igad"] - r["skew_raw"]
        print("%-6d  %8.4f  %10.4f  %10.4f  %+8.4f"
              % (bs, r["igad"], r["skew_mle"], r["skew_raw"], gap_ctrl))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("IGAD Hard Case: Gamma(8,2) vs LogNormal (matched mean+var)",
                 fontsize=12)

    # Plot A: score distributions for seed=42, n=200
    rng_plot   = np.random.default_rng(42)
    theta_ref  = GammaFamily.to_natural(ALPHA_REF, BETA_REF)
    R_ref      = scalar_curvature(GammaFamily.log_partition, theta_ref)
    ig_n, ig_a, sk_n, sk_a, skm_n, skm_a = [], [], [], [], [], []

    for phase, count in [("n", 100), ("a", 50)]:
        for _ in range(count):
            if phase == "n":
                b = rng_plot.gamma(ALPHA_REF, 1.0/BETA_REF, size=200)
            else:
                b = rng_plot.lognormal(MU_LN, SIG_LN, size=200)
                # lognormal is always positive; the filter below is a no-op guard
                b = b[b > 0]
            theta_l  = GammaFamily.mle(b)
            R_l      = scalar_curvature(GammaFamily.log_partition, theta_l)
            ig_score = abs(R_ref - R_l)
            alpha_l  = theta_l[0] + 1.0
            skm_score = abs(2.0/math.sqrt(alpha_l) - REF_SKEW)
            sk_score  = abs(sp_skew(b) - REF_SKEW)
            if phase == "n":
                ig_n.append(ig_score); sk_n.append(sk_score); skm_n.append(skm_score)
            else:
                ig_a.append(ig_score); sk_a.append(sk_score); skm_a.append(skm_score)

    auc_ig  = roc_auc_score([0]*100+[1]*50, ig_n+ig_a)
    auc_sk  = roc_auc_score([0]*100+[1]*50, sk_n+sk_a)
    auc_skm = roc_auc_score([0]*100+[1]*50, skm_n+skm_a)

    axes[0].hist(ig_n,  bins=25, alpha=0.6, label="Normal",  density=True)
    axes[0].hist(ig_a,  bins=12, alpha=0.6, label="Anomaly", density=True)
    axes[0].set_title("IGAD  (AUC=%.3f)" % auc_ig)
    axes[0].set_xlabel("|R_ref − R_local|")
    axes[0].legend()

    axes[1].hist(skm_n, bins=25, alpha=0.6, label="Normal",  density=True)
    axes[1].hist(skm_a, bins=12, alpha=0.6, label="Anomaly", density=True)
    axes[1].set_title("MLE skewness — CONTROL  (AUC=%.3f)" % auc_skm)
    axes[1].set_xlabel("|skew_MLE − skew_ref|")
    axes[1].legend()

    axes[2].hist(sk_n,  bins=25, alpha=0.6, label="Normal",  density=True)
    axes[2].hist(sk_a,  bins=12, alpha=0.6, label="Anomaly", density=True)
    axes[2].set_title("Raw skewness  (AUC=%.3f)" % auc_sk)
    axes[2].set_xlabel("|skew_raw − skew_ref|")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("docs/figures/exp2_hard_gamma_vs_lognormal.png", dpi=150)
    print()
    print("Plot saved to docs/figures/exp2_hard_gamma_vs_lognormal.png")
    print()
    print("Interpretation key:")
    print("  IGAD > MLE-skewness  →  the curvature geometry is doing real work")
    print("  IGAD ≈ MLE-skewness  →  MLE efficiency alone explains the advantage")


if __name__ == "__main__":
    run_hard_demo()