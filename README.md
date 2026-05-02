
---

# IGAD - Information-Geometric Anomaly Detection

**A new path for anomaly detection: treating anomalies as curvature events on a statistical manifold.**

## The Invention

For a century, anomaly detection has measured where data sits and how far it has moved. IGAD proposes that this is only half the picture. Distributions have shape, shape has geometry, and geometry carries information that no moment, no distance, and no density estimate can recover alone. To see the other half of the anomalies in the world, we have to learn to see the shape itself.

"Anomalies do not live only in location and distance. They live in shape."
- Omry Damari, creator of IGAD

IGAD introduces a fundamentally different approach. It treats every fitted distribution as a point on a curved geometric surface — the statistical manifold of an exponential family — and detects anomalies by measuring how the curvature of that surface responds to new data. An anomaly, under IGAD, is not merely a numerical outlier. It is a deformation in the geometry of the distribution itself.

The mathematical core is a single, principled object. IGAD computes the Fisher information metric on the manifold, constructs the third cumulant tensor that encodes higher-order distributional structure, and contracts them into a scalar curvature signal written compactly as ‖T‖²_g. This contraction captures shape information that no individual moment — mean, variance, or skewness, raw or MLE-fitted — can recover on its own. It is, to our knowledge, the first practical anomaly detector built directly on Fisher–Rao scalar curvature with closed-form tensor support for real distributional families.

## Why This Matters

Consider the hardest case in anomaly detection: two distributions that share the same mean and the same variance. A Gamma distribution and a LogNormal distribution can be tuned to match on both moments exactly. To classical detectors, these distributions are nearly invisible to one another. Mean tests fail. Variance tests fail. Even skewness tests, including the MLE-efficient version, struggle to separate them reliably.

IGAD sees the difference. Because the two distributions trace out different curvature on the statistical manifold, the geometric signal survives where moment-based signals collapse. In the regime where this matters most — moderate sample sizes between 200 and 500 observations — IGAD beats the MLE-skewness control by +0.053 AUC, using the identical underlying fit but extracting the curvature tensor that other methods discard.

This is not a marginal improvement on an existing technique. It is evidence that an entire dimension of distributional information has been ignored by the field, and that this dimension can be measured, computed, and used.

## The Falsifiable Claim

IGAD makes a precise scientific claim, not a marketing claim. The contraction ‖T‖²_g extracts shape information not captured by any single moment, raw or MLE-fitted. This claim is supported in the tested regime of n = 200 to 500 observations and remains falsifiable outside that regime. The repository ships with the experiments, the controls, and the failure modes needed to test the claim independently. The Gaussian constant-curvature failure case is documented openly, because a serious geometric method must say where its geometry runs out.

## Using It

The invention is mathematically deep but operationally simple.

```bash
pip install visigence-igad
```

```python
import igad
from igad.families import GammaFamily

detector = igad.IGADDetector(family=GammaFamily())
detector.fit(X_train)

scores = detector.score_samples(X_test)
labels = detector.predict(X_test, contamination=0.10)
```

Closed-form tensor support is included for Gamma and Dirichlet families, with the architecture designed to extend naturally to additional exponential families.

## Verification

The release is validated end-to-end. All 54 tests pass across Python 3.10, 3.11, and 3.12, with reproducible artifacts published on GitHub Actions and the package distributed through PyPI as `visigence-igad`. Validation commit, release commit, and run logs are linked in the verification section below for full auditability.

---
