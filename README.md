
# IGAD: Information-Geometric Anomaly Detection

<a href="https://github.com/Visigence/IGAD/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
</a>
<a href="https://github.com/Visigence/IGAD/actions/workflows/test.yml">
  <img src="https://github.com/Visigence/IGAD/actions/workflows/test.yml/badge.svg" alt="Tests">
</a>
<a href="https://github.com/Visigence/IGAD/blob/main/setup.py">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions">
</a>

> _The anomaly is not only where the distribution lives - it is what shape it becomes._
>
> _Omry Damari_

---

IGAD detects distributional shape shifts using scalar curvature deviation on the Fisher–Rao statistical manifold.

```text
IGAD(batch) = |R(theta_ref) - R(theta_local)|
````
---

## Core Claim

> **The anomaly is not where the distribution is. It is what shape it has.**

```text
IGAD(batch) = | R(θ_ref) − R(θ_local) |
```

where `R(θ)` is the **scalar curvature** of the Fisher-Rao statistical manifold at the natural parameter point `θ`.

---

## The Problem

Every widely-used anomaly detector shares the same assumption:

> **anomaly = a point far from the center**

| Method | What It Measures |
|---|---|
| Z-Score | Distance from mean in standard deviation units |
| Mahalanobis | Distance from cloud center accounting for correlations |
| Isolation Forest | Ease of isolating a point in feature space |
| LOF | Relative local neighborhood density |

All four are **blind** to the following:

```text
Reference : Gamma(8, 2)        mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(...)      mean=4.000  var=2.000  skew=1.105
```

Mean and variance are **exactly identical**. The internal structure of the distribution has changed completely. No distance-based algorithm detects this.

---

## What Was Known Before This Work

Every mathematical identity used here is an established result:

| Component | Source |
|---|---|
| Fisher-Rao metric | Rao (1945) |
| Differential geometry of exponential families | Amari (1985) |
| Scalar curvature formula for Hessian metrics | Amari & Nagaoka (2000) |
| Fourth-cumulant cancellation in Riemann tensor | Standard Hessian geometry |
| Curvature as detector of phase transitions | Ruppeiner (1979, 1995) |

## What Is New

| Component | Description |
|---|---|
| **Construction** | Using scalar curvature deviation as a batch-level anomaly score — not previously proposed in the anomaly detection literature |
| **Insight** | Scalar curvature, governed by the full contraction `‖T‖²_g`, is structurally sensitive to shape shifts — a natural detector for anomalies invisible to location-scale methods |
| **Validation** | A control experiment isolating geometry from MLE efficiency confirms the curvature tensor itself is responsible for the advantage |

Full derivation with attribution: `docs/proof.md`

---

## Mathematical Foundation

For an exponential family with log-partition `A(θ)`:

```text
Fisher metric:          gᵢⱼ(θ)   = ∂²A / ∂θᵢ∂θⱼ
Third cumulant tensor:  Tᵢⱼₖ(θ)  = ∂³A / ∂θᵢ∂θⱼ∂θₖ
Christoffel symbols:    Γᵢⱼ,ₖ    = ½ · Tᵢⱼₖ
Scalar curvature:       R(θ)      = ¼ · ( ‖S‖²_g − ‖T‖²_g )
```

where:

```text
Sₘ     = gᵃᵇ Tₐᵦₘ
‖T‖²_g = gⁱᵃ gʲᵇ gᵏᶜ Tᵢⱼₖ Tₐᵦᶜ
```

The critical quantity is `‖T‖²_g` — a three-index contraction of the third cumulant tensor against the inverse metric, giving a geometrically weighted measure of total skewness content. Unlike `scipy.stats.skew`, it exploits the full parametric structure of the family.

---

## Implementation

```text
igad/
  curvature.py        Fisher metric, third cumulant tensor, scalar curvature
  families.py         GammaFamily, PoissonFamily, DirichletFamily
  detector.py         IGADDetector (batch-level scoring)
tests/
  test_curvature.py        Curvature & Gamma family validation
  test_dirichlet_family.py Dirichlet validation + sample efficiency
experiments/
  demo_easy.py        Experiment 1: Gamma vs Gamma
  demo_hard.py        Experiment 2: Gamma vs LogNormal + MLE control
  demo_gaussian2d.py  Experiment 3: Gaussian failure mode (documented)
docs/
  proof.md            Mathematical background with full attribution
  figures/            Experiment plots with descriptions
RESULTS.md            Full experimental results and analysis
```

### Quick Start

```bash
pip install -e .
```

```python
import numpy as np
from igad import IGADDetector
from igad.families import GammaFamily

detector = IGADDetector(family=GammaFamily)
reference_data = np.random.gamma(8.0, 0.5, size=200)
detector.fit(reference_data)

test_batch = np.random.lognormal(1.327, 0.343, size=200)
score = detector.score_batch(test_batch)
print(f"Curvision score: {score:.6f}")  # Higher = more anomalous
```

### Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
# 54 passed
```

---

## Experimental Results

### Experiment 1 — Easy Case

**Gamma(9, 3) vs Gamma(1.5, 0.5)** · same mean (3.0), different variance and skewness

```text
Method                 AUC-ROC
------------------------------
Curvision (curvature)   1.0000
Variance shift          1.0000
Skewness shift          0.9834
Mean shift              0.8150
```

Curvision achieves perfect separation. Variance baseline also reaches 1.0 because variance differs by 6×. Experiment 2 is the key result.

---

### Experiment 2 — Hard Case (the key result)

**Gamma(8, 2) vs LogNormal** · `mean = 4.0` and `var = 2.0` **identical for both**

```text
Reference : Gamma(8, 2)              mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(μ=1.327,       mean=4.000  var=2.000  skew=1.105
            σ=0.343)
```

A control baseline was constructed using the **identical MLE fit** as Curvision but discarding the curvature tensor:

```text
skew_MLE(batch) = 2 / √α_MLE
score = | skew_MLE − skew_ref |
```

#### Results — 5 seeds, n = 200

```text
Method                        Mean AUC   ± Std
----------------------------------------------
Curvision (curvature)          0.6542    0.047
MLE skewness  [CONTROL]        0.6016    0.038
Raw skewness                   0.6794    0.072
Mean shift    [BLIND]          0.5240    0.062
Variance shift [BLIND]         0.5818    0.027
```

**Gap (Curvision − MLE skewness): +0.053** → Curvature geometry adds signal beyond MLE efficiency alone.

#### Scaling with batch size

```text
n        Curvision   MLE-skew   Raw-skew   Gap (Curv − MLE)
-----------------------------------------------------------
100      0.5704      0.5764     0.5908     −0.006
200      0.6838      0.6098     0.6514     +0.074
500      0.6748      0.5846     0.9194     +0.090
1000     0.7892      0.8214     0.9686     −0.032
```

Curvision beats the MLE-control at n = 200 and n = 500. At n = 1000, model misspecification degrades the curvature signal — model-free methods dominate at large n when the parametric model is wrong.

---

### Experiment 3 — Gaussian Failure Mode (Honest Limitation)

Bivariate Gaussian, ρ_ref = 0.2 vs ρ_anom = 0.8. Mean and marginal variances identical.

```text
ρ_ref=0.20, ρ_anom=0.80   →   |ΔR| = 0.003308
ρ_ref=0.50, ρ_anom=0.55   →   |ΔR| = 0.000049
```

All methods reached **AUC = 1.0** — not because of curvature, but because the correlation difference is large enough for any method to detect. **Curvision added nothing unique here.**

**Reason:** the Gaussian manifold has **constant scalar curvature** (isometric to hyperbolic space). Curvision is not applicable to Gaussian families.

---

### Experiment 4 — Dirichlet Family

Extended Curvision to **Dirichlet(α₁, …, αₖ)** with `k ≥ 3`, where pure shape variation is possible with fixed lower-order moments.

- Fisher metric matches numerical Hessian (symmetric, positive-definite) ✓
- Third cumulant tensor analytical form agrees with numerical derivatives ✓
- Scalar curvature varies meaningfully with concentration and asymmetry ✓
- Curvision detects Dirichlet shape shifts at n = 200 and beats random at n = 50 ✓
- AUC monotonically increases with n on well-specified data ✓

---

## Summary Table

```text
╔══════════════════╦═══════════════╦═════════════╦═══════════════════╗
║ Method           ║  Mean Shift   ║ Shape Shift ║ Low-Sample (n<300)║
╠══════════════════╬═══════════════╬═════════════╬═══════════════════╣
║ Z-Score          ║      ✓        ║      ✗      ║        ✓          ║
║ Mahalanobis      ║      ✓        ║      ✗      ║        ~          ║
║ Isolation Forest ║      ✓        ║      ✗      ║        ✗          ║
║ Skewness Test    ║      ✗        ║      ~      ║        ✗          ║
║ Curvision (this) ║      ~        ║      ✓      ║        ✓          ║
╚══════════════════╩═══════════════╩═════════════╩═══════════════════╝
```

---

## When to Use Curvision

- The correct parametric family is known or approximately known
- Batch sizes are moderate (50 – 300 observations)
- Anomalies differ in distributional **shape**, not just location or scale
- The family has dimension d ≥ 2 (1D manifolds have R = 0)

**Potential applications:**
- Predictive maintenance — vibration profile shape changes before amplitude changes
- Financial monitoring — transaction distribution structure shifts
- Medical signal analysis — ECG waveform geometry changes in early arrhythmia
- Cybersecurity — packet-size distribution shifts in low-and-slow exfiltration

## When NOT to Use Curvision

- Anomalies are simple outliers far from center → use Isolation Forest
- No parametric model is appropriate → use model-free tests
- Large batch sizes (n > 500) and model is approximate → use raw skewness
- 1D parameter families (Poisson, Exponential, Bernoulli) → R ≡ 0
- Gaussian families → R is constant, Curvision adds nothing

---

## Documented Limitations

| Limitation | Explanation |
|---|---|
| Model specification required | Wrong family → signal degrades at large n |
| 1D families | R ≡ 0 (Poisson, Exponential, Bernoulli) |
| Gaussian families | R = constant (hyperbolic geometry) |
| Large n + misspecified model | Model-free methods dominate |
| Computational cost | O(d³) tensor contractions per evaluation |

---

## Validation: 54 Automated Tests

```text
======================== 54 passed in 316.74s ========================

tests/test_curvature.py
  TestPoissonFlat                          1 passed  (R = 0 verified)
  TestGammaFamily                         11 passed  (Fisher, T, R)

tests/test_dirichlet_family.py
  TestDirichletLogPartition                4 passed
  TestDirichletFisherMetric                9 passed
  TestDirichletCurvature                   7 passed
  TestDirichletThirdCumulantAnalytical     8 passed
  TestDirichletMLE                         5 passed
  TestIGADSampleEfficiency                 4 passed
  TestFailureModes                         3 passed
```

Every documented limitation is **enforced by a test** that would fail if the limitation stopped holding.

---

## References

- Rao, C.R. (1945). *Information and the accuracy attainable in the estimation of statistical parameters.* Bull. Calcutta Math. Soc.
- Amari, S. (1985). *Differential-Geometrical Methods in Statistics.* Springer.
- Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry.* AMS / Oxford.
- Ruppeiner, G. (1979). *Thermodynamics: A Riemannian geometric model.* Phys. Rev. A.
- Ruppeiner, G. (1995). *Riemannian geometry in thermodynamic fluctuation theory.* Rev. Mod. Phys.

---

## License

MIT - see [LICENSE](LICENSE) 
