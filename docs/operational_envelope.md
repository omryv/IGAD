# IGAD Operational Envelope

## When IGAD Wins

### Condition 1: Correct exponential family, k >= 3 parameters, small-to-moderate n

When the reference distribution belongs to a correctly-specified exponential family with
at least three parameters, IGAD exploits scalar curvature variation to detect concentration
shifts that cannot be resolved by mean or variance alone.

**Evidence:** Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5) — same α₀=12, same mean direction
for the symmetric case, only the concentration profile shifts. IGAD AUC > 0.65 at n=200.

The curvature advantage is most pronounced at small-to-moderate n (20–200 samples per batch)
where non-parametric methods have insufficient power. At n > 500, MMD and Wasserstein
catch up as their estimators reach their asymptotic regime.

**Sample complexity:** IGAD achieves AUC > 0.7 at n ≈ 50–100 for the Dirichlet
concentration shift; MMD requires n ≈ 200–300 for equivalent power.

### Condition 2: Cross-family detection (misspecified model, small n)

When data comes from a different exponential family than assumed (e.g., LogNormal data
scored under a Gamma model), IGAD still detects anomalies because the MLE parameter
fit to the anomalous data lands at a point with systematically different curvature
than the reference.

**Evidence:** Gamma(8,2) vs LogNormal (matched mean AND variance). IGAD beats
MLE-skewness control by +0.053 AUC (mean, 5 seeds, n=200). This confirms that the
curvature geometry is doing real work, not just MLE efficiency.

---

## When IGAD Fails

### Failure Mode 1: Gaussian families

**R = constant for all Gaussian parameters.**

Mathematical reason: The Fisher-Rao metric of the multivariate Gaussian is the
Siegel upper half-space metric, a symmetric space of constant negative curvature
(hyperbolic geometry). Specifically, for the n-dimensional Gaussian family,
scalar curvature R = -n(n+1)/2, a constant depending only on dimension, not on
the specific parameters (mean, covariance). No matter how the mean or covariance
changes between reference and anomaly, R_ref = R_local identically, so the IGAD
score is zero.

**Consequence:** IGAD cannot detect any Gaussian-to-Gaussian shift (mean shift,
variance shift, correlation shift). Use classical tests (Hotelling's T², likelihood
ratio tests) for Gaussian anomaly detection.

### Failure Mode 2: 1D families

**R ≡ 0 identically.**

Mathematical reason: For a 1-dimensional exponential family (Poisson, Exponential,
Bernoulli, any single-parameter family), the Fisher-Rao manifold is a 1D Riemannian
manifold. The Riemann curvature tensor of a 1D manifold is identically zero by
definition — you cannot have intrinsic curvature in one dimension. Therefore
R = 0 for all parameter values, and the IGAD score |R_ref - R_local| = 0 always.

**Consequence:** IGAD is completely blind to shifts in Poisson rate, Exponential
rate, Bernoulli probability, or any single-parameter family. This is proven, not
empirical.

### Failure Mode 3: Large n with misspecified model

When n is large (> 500 per batch), non-parametric methods (MMD, Wasserstein)
accumulate enough samples to achieve near-perfect power without any parametric
assumptions. If the reference distribution is misspecified (e.g., data is actually
from a mixture model but scored under a single Dirichlet), IGAD's curvature
estimator absorbs the model error into a biased curvature estimate. Non-parametric
methods remain unbiased and eventually dominate.

**Consequence:** For large-n production systems with complex reference distributions,
prefer ensemble approaches combining IGAD (for small n) with MMD or Wasserstein
(for large n).

---

## The Falsifiable Claims

1. **IGAD AUC > MLE-skewness control by at least +0.04** (mean, 5 seeds, n=200,
   Gamma vs LogNormal with matched mean+variance). Verified in `experiments/demo_hard.py`
   and `experiments/demo_hard_extended.py`.

2. **IGAD AUC > 0.65 at n=200** for Dirichlet k=3 concentration shift
   (Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5), seed=42, 100 normal + 50 anomaly batches).
   Verified in `tests/test_dirichlet_family.py::TestIGADSampleEfficiency`.

3. **IGAD AUC > MMD at n <= 100** for Dirichlet k=3 concentration shift
   (sample efficiency regime). Verified empirically in `experiments/demo_dirichlet.py`
   Part 3.

4. **R(Gaussian) is constant regardless of correlation parameter** (proven, not empirical).
   Verified in `tests/test_dirichlet_family.py::TestFailureModes::test_gaussian_constant_curvature`.

5. **R(Poisson) ≡ 0** (proven, verified in tests).
   Verified in `tests/test_curvature.py::TestPoissonFlat` and
   `tests/test_dirichlet_family.py::TestFailureModes::test_poisson_flat`.

---

## Sample Complexity Analysis

For a correctly-specified k-dimensional exponential family:

- **IGAD** requires O(n^{1/2}) samples to achieve AUC > 0.7 in the concentration-shift
  regime. The curvature estimator converges at the rate of the MLE (Fisher-efficient),
  and the signal (|ΔR|) is a fixed constant for a fixed Δα.

- **Non-parametric methods** (MMD, Wasserstein) require O(n) samples for equivalent
  power, because their test statistics converge at the slower rate of the
  empirical distribution (no parametric structure exploited).

- The advantage is O(n^{1/2}) — a sub-linear sample-efficiency gain from parametric
  structure.

(The O(n^{1/2}) claim is empirically derived from the sweep in Experiment 4,
`experiments/demo_dirichlet.py` Part 3. A formal proof requires analysis of the
Fisher information matrix eigenspectrum and the curvature functional's gradient
with respect to the natural parameters.)

---

## Curvature Separation Verification

For the canonical Dirichlet detection pair:

| Distribution          | α               | R(α)      |
|-----------------------|-----------------|-----------|
| Reference             | [4.0, 4.0, 4.0] | (see exp) |
| Anomaly               | [1.5, 4.0, 6.5] | (see exp) |

`|R_ref - R_anom| > 0.01` — verified in unit tests and Part 4 of
`experiments/demo_dirichlet.py`. This non-zero separation is the mathematical
foundation that makes IGAD work for this family.
