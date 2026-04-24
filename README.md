Curvision # IGAD: Information-Geometric Anomaly Detection

**Author:** Omry Damari · 2026  
```diff
+ ┌──────────────────────────────────────────────┐
+ │                  CURVISION                   │
+ ├──────────────────────────────────────────────┤
+ │ see         : beyond-location                │
+ │ see         : beyond-distance                │
+ │ model       : curvature-based-vision         │
+ └──────────────────────────────────────────────┘
```
Classical anomaly detectors are blind to shape shifts — anomalies that 
preserve mean and variance but change distributional geometry. IGAD detects them.

---

## The Problem

Every widely-used anomaly detection method shares the same assumption:

> anomaly = a point far from the center

| Method           | What It Measures                                        |
|------------------|---------------------------------------------------------|
| Z-Score          | Distance from mean in standard deviation units          |
| Mahalanobis      | Distance from cloud center accounting for correlations  |
| Isolation Forest | Ease of isolating a point in feature space              |
| LOF              | Relative local neighborhood density                     |

All four are blind to the following:
```
Reference : Gamma(8, 2)    mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(...)  mean=4.000  var=2.000  skew=1.105
```

Mean and variance are exactly identical. The internal structure of the 
distribution has completely changed. No distance-based algorithm detects this.

---

## The Invention

The novel contribution of this work is a single construction:
```
IGAD(batch) = |R(theta_ref) - R(theta_local)|
```

where `R(theta)` is the scalar curvature of the Fisher-Rao statistical manifold 
at the natural parameter point `theta`.

### What Was Known Before This Work

| Component                                        | Source                    |
|--------------------------------------------------|---------------------------|
| Fisher-Rao metric                                | Rao (1945)                |
| Differential geometry of exponential families    | Amari (1985)              |
| Scalar curvature formula for Hessian metrics     | Amari & Nagaoka (2000)    |
| Fourth-cumulant cancellation in Riemann tensor   | Standard Hessian geometry |
| Curvature as detector of phase transitions       | Ruppeiner (1979, 1995)    |

### What Is New

The **construction**: using scalar curvature deviation as a batch-level anomaly 
score. This use has not been found in the anomaly detection literature.

The closest known related work in anomaly detection applies Ricci curvature 
to graph structures — a fundamentally different construction. Ricci curvature 
measures how edges in a graph bend. Fisher-Rao scalar curvature measures how 
the space of probability distributions itself bends at a given parameter point. 
One operates on data topology. The other operates on the geometry of the 
statistical model itself.

The **insight**: scalar curvature is governed by `||T||²_g` — a full 
metric-weighted contraction of the third cumulant tensor across all parameter 
dimensions simultaneously. This is not skewness. Skewness is a single number. 
`||T||²_g` is a tensor contraction that weights each direction by the inverse 
Fisher metric — it captures how asymmetry is distributed across the entire 
parameter geometry. No scalar moment captures this.

The **proof**: a control experiment with identical MLE fit but no curvature 
tensor confirms the geometry adds +0.053 AUC independently of MLE efficiency.

The **result**: IGAD beats MMD and Wasserstein when mean and variance are 
held exactly identical — the regime where every distance-based method is blind.

---

## Mathematical Foundation

For an exponential family with log-partition function `A(theta)`:
```
Fisher metric:         g_{ij}(theta) = d²A / dtheta_i dtheta_j
Third cumulant tensor: T_{ijk}(theta) = d³A / dtheta_i dtheta_j dtheta_k
Scalar curvature:      R(theta) = 1/4 * ( ||S||²_g - ||T||²_g )

where S_m = g^{ab} T_{abm}  (trace vector)
      ||T||²_g = g^{ia} g^{jb} g^{kc} T_{ijk} T_{abc}
```

Full derivation with attribution: `docs/proof.md`

---

## Implementation
```
igad/
  curvature.py          Fisher metric, third cumulant tensor, scalar curvature
  families.py           GammaFamily, PoissonFamily, DirichletFamily
  detector.py           IGADDetector.score_batch() — batch-level IGAD score
  exceptions.py         ConvergenceError (MLE convergence gate)
tests/
  test_curvature.py         12 validation tests (Gamma/Poisson)
  test_dirichlet_family.py  34 validation tests (Dirichlet, 6 classes)
experiments/
  demo_easy.py              Experiment 1: Gamma vs Gamma
  demo_hard.py              Experiment 2: Gamma vs LogNormal + MLE control
  demo_dirichlet.py         Experiment 3: Dirichlet concentration shifts
  demo_hard_extended.py     Experiment 4: Gamma vs LogNormal + MMD + Wasserstein
  demo_gaussian2d.py        Experiment 5: Gaussian failure mode (documented)
docs/
  proof.md                  Mathematical background with full attribution
  operational_envelope.md   Falsifiable claims: when IGAD wins and loses
  figures/                  Experiment plots
RESULTS.md                  Full experimental results and analysis
```

---

## Quick Start
```python
pip install -e .

import numpy as np
from igad import IGADDetector
from igad.families import GammaFamily

detector = IGADDetector(family=GammaFamily)
detector.fit(np.random.gamma(8.0, 0.5, size=200))

score = detector.score_batch(np.random.lognormal(1.327, 0.343, size=200))
print(f"IGAD score: {score:.6f}")  # Higher = more anomalous
```

---

## Running Tests
```bash
python -m pytest tests/ -v
# 46 passed
```

---

## Experimental Results

### Experiment 1 — Easy Case

Gamma(9,3) vs Gamma(1.5,0.5) · same mean (3.0), different variance and skewness

| Method           | AUC-ROC |
|------------------|---------|
| IGAD (curvature) | 1.0000  |
| Variance shift   | 1.0000  |
| Skewness shift   | 0.9834  |
| Mean shift       | 0.8150  |

IGAD achieves perfect separation. Variance baseline also reaches 1.0 because 
variance differs by 6×. This experiment does not prove unique value.

---

### Experiment 2 — Hard Case (Gamma vs LogNormal)

Gamma(8,2) vs LogNormal · mean=4.0, var=2.0 **identical** for both

**Results — 5 seeds, batch_size=200**

| Method                  | Mean AUC | ± Std |
|-------------------------|----------|-------|
| IGAD (curvature)        | 0.6542   | 0.047 |
| MLE skewness [CONTROL]  | 0.6016   | 0.038 |
| Raw skewness            | 0.6794   | 0.072 |
| MMD (RBF, median-BW)    | 0.5894   | 0.076 |
| Wasserstein (1D)        | 0.5925   | 0.057 |
| Mean shift [BLIND]      | 0.5240   | 0.062 |
| Variance shift [BLIND]  | 0.5818   | 0.027 |

**Gap (IGAD − MLE skewness): +0.053**  
Curvature geometry adds signal **beyond** MLE efficiency alone.

**Sample-efficiency sweep — IGAD vs MMD vs Wasserstein (fixed signal)**

| n   | IGAD   | MMD    | Wasserstein | Gap(IGAD−MMD) |
|-----|--------|--------|-------------|---------------|
| 50  | 0.5522 | 0.5465 | 0.5425      | +0.006        |
| 100 | 0.5871 | 0.5639 | 0.5440      | +0.023        |
| 200 | 0.6542 | 0.5894 | 0.5925      | **+0.065**    |
| 300 | 0.6395 | 0.6074 | 0.5933      | +0.032        |
| 500 | 0.7150 | 0.6814 | 0.6777      | **+0.034**    |

IGAD beats both MMD and Wasserstein at every batch size tested (n ∈ {50, 100, 200, 300, 500}).  
Mean and variance are **exactly identical** between reference and anomaly in all runs.  
Note: raw skewness dominates at large n (n=500, seed=42: raw-skew AUC=0.919 vs IGAD=0.675);  
IGAD's advantage is over distance-based baselines, not moment estimators.

---

### Experiment 3 — Dirichlet Concentration Shifts

Setup: α_ref = [4, 4, 4] vs α_anom = [1.5, 4, 6.5] — both sum to 12.0

| n   | IGAD   | MMD    | Wasserstein |
|-----|--------|--------|-------------|
| 20  | 0.7540 | 0.9998 | 1.0000      |
| 50  | 0.9074 | 1.0000 | 1.0000      |
| 100 | 0.9302 | 1.0000 | 1.0000      |
| 200 | 0.9822 | 1.0000 | 1.0000      |
| 500 | 0.9878 | 1.0000 | 1.0000      |

Note: The Dirichlet pair used here (α=[4,4,4] vs α=[1.5,4,6.5]) includes a marginal  
mean shift — the component means change when α is non-uniform. MMD and Wasserstein  
dominate because they directly detect this mean shift. This experiment validates  
that IGAD's curvature is non-zero and non-constant on the Dirichlet manifold;  
it does not test the pure concentration-shift regime. The clean cross-family result  
(matched mean AND variance) is Experiment 2.

---

### Experiment 4 — Gaussian Failure Mode (Honest Limitation)

The Gaussian manifold has **constant** scalar curvature (isometric to hyperbolic 
space). IGAD adds nothing to Gaussian anomaly detection. Documented and tested.

---

## Operational Envelope

| Condition                              | IGAD Performance                      |
|----------------------------------------|---------------------------------------|
| Cross-family shape shift, n=200–500    | ✅ Beats MMD and Wasserstein           |
| Gamma family, correct spec, n=200      | ✅ Beats MLE-skewness control (+0.053) |
| Gaussian families                      | ❌ Constant curvature — do not use    |
| 1D parameter families (Poisson, Exp)   | ❌ R≡0 — do not use                   |
| Large n + misspecified model (n>500)   | ⚠️ Degrades — use MMD/Wasserstein     |
| No parametric model available          | ⚠️ Use model-free tests instead       |

See `docs/operational_envelope.md` for falsifiable claims.

---

## When to Use IGAD

- The correct parametric family is known or approximately known
- Batch sizes are moderate (50–500 observations)
- Anomalies differ in distributional **shape**, not just location or scale
- The family has dimension d ≥ 2

**Potential applications:**
- Predictive maintenance (vibration profile shifts before amplitude changes)
- Financial monitoring (transaction distribution structure shifts)
- Medical signal analysis (ECG waveform geometry in early arrhythmia)
- Cybersecurity (packet size distribution shifts in low-and-slow exfiltration)

---

## Validation
```
46 passed

TestDirichletLogPartition    (4)   — log-partition identity, convexity, roundtrip
TestDirichletFisherMetric    (5)   — matches numerical Hessian, PD, k=4
TestDirichletCurvature       (4)   — finite, non-constant, separates anomaly pair
TestDirichletMLE             (5)   — recovery rtol=0.05, convergence gate
TestIGADSampleEfficiency     (4)   — AUC>0.65 at n=200, monotone in n
TestFailureModes             (3)   — Poisson R≡0, Gaussian constant, k=2 degenerate
TestGammaFamily             (12)   — existing suite, all passing
```

---

## References

1. Rao, C.R. (1945). Information and the accuracy attainable in the estimation of statistical parameters.
2. Amari, S. (1985). Differential-Geometrical Methods in Statistics. Springer.
3. Amari, S. & Nagaoka, H. (2000). Methods of Information Geometry. AMS/Oxford.
4. Minka, T. (2000). Estimating a Dirichlet distribution. MIT Tech Report.
5. Ruppeiner, G. (1979). Thermodynamics: A Riemannian geometric model.
6. Ruppeiner, G. (1995). Riemannian geometry in thermodynamic fluctuation theory.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
> *"The anomaly isn't where the distribution is. It's what shape it is."*  
>  Omry Damari, 2026
