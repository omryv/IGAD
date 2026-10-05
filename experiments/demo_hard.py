"""
experiments/demo_hard.py

Hard Case: Gamma(8, 2) versus LogNormal with matched mean (4.0) and
variance (2.0), scored by IGAD and by the same-fit MLE-skewness control.

    python -m experiments.demo_hard

Two IGAD columns are reported:

  igad     -- curvature from the exact Fisher metric and cumulant tensor,
              the route IGADDetector uses. This is the shipped detector.
  igad_fd  -- curvature by finite differences of the log-partition, the
              route this script used before 1.0.3. Kept only so the
              published +0.053 can be traced to its source.

What this experiment shows (1.0.3): for the Gamma family R depends on the
shape alpha alone, and is strictly monotone in it (pinned by
tests/test_gamma_reduction.py). The MLE-skewness control, 2/sqrt(alpha_hat),
is a function of the same alpha_hat. The IGAD score is therefore a
re-scaling of the MLE skewness and carries no information the control
lacks. With the exact curvature, IGAD scores slightly *below* the control at
every batch size; the earlier +0.053 came from finite-difference error.
"""
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import skew as sp_skew

from experiments._router_common import save_results
from igad.curvature import scalar_curvature
from igad.families import GammaFamily


# ── Distribution parameters ──────────────────────────────────────────────────
ALPHA_REF, BETA_REF = 8.0, 2.0          # Gamma reference
REF_MEAN  = ALPHA_REF / BETA_REF        # 4.0
REF_VAR   = ALPHA_REF / BETA_REF**2     # 2.0
REF_SKEW  = 2.0 / math.sqrt(ALPHA_REF) # 0.707

# LogNormal anomaly: matched mean AND variance
SIG2  = math.log(1 + REF_VAR / REF_MEAN**2)   # ln(1.125)
SIG_LN = math.sqrt(SIG2)                        # 0.343
MU_LN  = math.log(REF_MEAN) - SIG2 / 2         # 1.327

SEEDS       = range(40)
BATCH_SIZES = [100, 200, 500, 1000]


def gamma_scalar_curvature(theta):
    """R from the exact g and T -- the route IGADDetector takes for Gamma."""
    return scalar_curvature(
        GammaFamily.log_partition, theta,
        g=GammaFamily.fisher_metric_analytical(theta),
        T=GammaFamily.third_cumulant_analytical(theta),
    )


def gamma_scalar_curvature_fd(theta):
    """R by finite differences -- the pre-1.0.3 route of this script."""
    return scalar_curvature(GammaFamily.log_partition, theta)


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
    Run one seed. Returns a dict of AUCs.

    Baselines:
      igad - |R_ref - R_local|, exact curvature (the shipped detector)
      igad_fd - |R_ref - R_local|, finite-difference curvature (pre-1.0.3)
      skew_mle - |skew_mle(batch) - skew_ref|, skew_mle = 2/sqrt(alpha_mle)
                  *** KEY CONTROL: same MLE, no geometry ***
      skew_raw - |scipy.stats.skew(batch) - skew_ref|
      mean - |mean(batch) - ref_mean| / sqrt(ref_var)
      var - |var(batch)  - ref_var|
    """
    rng = np.random.default_rng(seed)

    theta_ref = GammaFamily.to_natural(ALPHA_REF, BETA_REF)
    R_ref     = gamma_scalar_curvature(theta_ref)
    R_ref_fd  = gamma_scalar_curvature_fd(theta_ref)

    keys   = ["igad", "igad_fd", "skew_mle", "skew_raw", "mean", "var"]
    scores = {k: [] for k in keys}
    labels = []

    for phase, count, lab in [("normal", n_normal, 0),
                               ("anomaly", n_anomaly, 1)]:
        for _ in range(count):
            if phase == "normal":
                batch = rng.gamma(ALPHA_REF, 1.0 / BETA_REF, size=batch_size)
            else:
                batch = rng.lognormal(MU_LN, SIG_LN, size=batch_size)
                batch = batch[batch > 0]   # safety filter

            theta_local = GammaFamily.mle(batch)
            scores["igad"].append(abs(R_ref - gamma_scalar_curvature(theta_local)))
            scores["igad_fd"].append(abs(R_ref_fd - gamma_scalar_curvature_fd(theta_local)))

            # Same MLE theta_local, no geometry.
            alpha_mle = theta_local[0] + 1.0   # natural param: theta_0 = alpha - 1
            scores["skew_mle"].append(abs(2.0 / math.sqrt(alpha_mle) - REF_SKEW))

            scores["skew_raw"].append(abs(sp_skew(batch) - REF_SKEW))
            scores["mean"].append(abs(np.mean(batch) - REF_MEAN) / math.sqrt(REF_VAR))
            scores["var"].append(abs(np.var(batch) - REF_VAR))
            labels.append(lab)

    return {k: roc_auc_score(labels, v) for k, v in scores.items()}


def _summarise(results, key):
    vals = np.array([r[key] for r in results])
    return float(vals.mean()), float(vals.std())


def _gap(results, a, b):
    """Mean paired gap a - b across seeds, with its standard error."""
    d = np.array([r[a] - r[b] for r in results])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def run_hard_demo():
    ln_mean, ln_var, ln_skew = _verify_lognormal()

    print("=" * 72)
    print("IGAD Hard Test: Matched Mean AND Variance - Geometry vs MLE")
    print("=" * 72)
    print("Reference : Gamma(%.0f, %.0f)  mean=%.3f  var=%.3f  skew=%.3f"
          % (ALPHA_REF, BETA_REF, REF_MEAN, REF_VAR, REF_SKEW))
    print("Anomaly   : LogNormal(mu=%.3f, sigma=%.3f)"
          % (MU_LN, SIG_LN))
    print("            mean=%.3f  var=%.3f  skew=%.3f" % (ln_mean, ln_var, ln_skew))
    print()

    methods = [
        ("IGAD (exact curvature)",       "igad"),
        ("IGAD (finite-diff, pre-1.0.3)", "igad_fd"),
        ("MLE skewness  [CONTROL]",      "skew_mle"),
        ("Raw skewness",                 "skew_raw"),
        ("Mean shift",                   "mean"),
        ("Variance shift",               "var"),
    ]

    payload = {
        "seeds": len(SEEDS),
        "n_normal": 100,
        "n_anomaly": 50,
        "batch_sizes": {},
    }

    for bs in BATCH_SIZES:
        results = [_scores_one_seed(s, bs) for s in SEEDS]
        print("─" * 72)
        print("n = %d   (%d seeds, 100 normal + 50 anomalous batches each)"
              % (bs, len(SEEDS)))
        print("─" * 72)
        print("%-32s  %8s  %8s" % ("Method", "Mean AUC", "± Std"))
        entry = {"auc": {}, "gap_vs_control": {}}
        for label, key in methods:
            mu, sd = _summarise(results, key)
            entry["auc"][key] = {"mean": mu, "std": sd}
            print("%-32s  %8.4f  %8.4f" % (label, mu, sd))
        for key in ["igad", "igad_fd"]:
            g, se = _gap(results, key, "skew_mle")
            entry["gap_vs_control"][key] = {"mean": g, "se": se}
            print("Gap %-28s  %+8.4f  ± %.4f (SE)" % (key + " − control", g, se))
        payload["batch_sizes"][str(bs)] = entry
        print()

    path = save_results("hard_case_exact", payload)
    print("Results saved to %s" % path)
    print()
    print("Reading: for Gamma, R is a strictly monotone function of alpha_hat,")
    print("and so is the MLE skewness. The IGAD score is a re-scaling of the")
    print("control and cannot carry more information than it. A positive gap")
    print("in the finite-difference column is numerical error, not geometry.")

    # ── Plot: score distributions for seed=42, n=200 ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("IGAD Hard Case: Gamma(8,2) vs LogNormal (matched mean+var)",
                 fontsize=12)

    rng_plot   = np.random.default_rng(42)
    theta_ref  = GammaFamily.to_natural(ALPHA_REF, BETA_REF)
    R_ref      = gamma_scalar_curvature(theta_ref)
    ig_n, ig_a, sk_n, sk_a, skm_n, skm_a = [], [], [], [], [], []

    for phase, count in [("n", 100), ("a", 50)]:
        for _ in range(count):
            if phase == "n":
                b = rng_plot.gamma(ALPHA_REF, 1.0/BETA_REF, size=200)
            else:
                b = rng_plot.lognormal(MU_LN, SIG_LN, size=200)
                b = b[b > 0]
            theta_l  = GammaFamily.mle(b)
            ig_score = abs(R_ref - gamma_scalar_curvature(theta_l))
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
    axes[0].set_title("IGAD, exact curvature  (AUC=%.3f)" % auc_ig)
    axes[0].set_xlabel("|R_ref − R_local|")
    axes[0].legend()

    axes[1].hist(skm_n, bins=25, alpha=0.6, label="Normal",  density=True)
    axes[1].hist(skm_a, bins=12, alpha=0.6, label="Anomaly", density=True)
    axes[1].set_title("MLE skewness - CONTROL  (AUC=%.3f)" % auc_skm)
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


if __name__ == "__main__":
    run_hard_demo()
