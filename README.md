# IGAD
## Curvision
**Information-Geometric Anomaly Detection**


  <a href="https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512">
    <img src="https://img.shields.io/badge/verified%20commit-81dd1eb-6e40c9?logo=github&logoColor=white"
         alt="Verified commit 81dd1eb">
  </a>
</p>

> *The anomaly is not only where the distribution lives — it is what shape it becomes.*
>
> Omry Damari

---

## Current research phase — router structure vs 3D quality

The open question is whether the internal structure of an MoE router gives an
early-warning signal for real 3D-generation quality degradation, beyond expert
load, routing entropy and other cheap diagnostics.

**Status: the hard gate failed and the real-data benchmark was not run.** This
environment has no tensor runtime, no checkpoint, no MoE module to hook, no
accelerator, no mesh library, and no egress to obtain any of them. Rather than
publish another synthetic AUC in its place, the work went where the brief
directs it:

| | |
| --- | --- |
| `python -m experiments.audit_environment` | measures all five gate conditions and exits non-zero |
| [`docs/acquisition_checklist.md`](docs/acquisition_checklist.md) | what to acquire, which candidate models exist, where to hook, what to record, and the statistical protocol |
| [`docs/experiment_plan.md`](docs/experiment_plan.md) | the experiment itself, specified step by step, calling only functions that are already written and tested |
| `experiments/trace_schema.py`, `experiments/quality_schema.py` | tested record contracts — one rejects a post-top-k capture, the other rejects a failure label derived from routing |
| `experiments/router_stats.py`, `experiments/evaluation.py` | the baselines, the structure-aware statistics, and the object-level statistical protocol |
| [`docs/sherman_morrison.md`](docs/sherman_morrison.md) | Dirichlet curvature reduced from O(k³) to **O(k)** — 105 s → 0.33 ms at k=1024 |
| [`docs/numerical_reliability.md`](docs/numerical_reliability.md) | 120-digit arbitration; `error = eps · ρ`, the old cond(g) caveat withdrawn, and a read-only runtime diagnostic |

Every entry in the brief's decision table that depends on real routers or real
3D quality reads **Untested** — not "No", and not a number.
**[`docs/handoff.md`](docs/handoff.md) is the summary**: what is proven here,
what needs external resources, and the single next action.

---
## Repository Status

IGAD is currently a verified research artifact.

The implementation baseline is pinned to commit
[`81dd1eb4540643083854232d9645f6add4150512`](https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512)
and release `IGAD-Ver1.0.0`.

This repository is public for reproducibility, verification, and independent review.  
Future changes should be treated as new research iterations and must be validated by a new GitHub Actions run.
IGAD detects distributional shape shifts using scalar curvature deviation on the Fisher–Rao statistical manifold.

```math
IGAD(batch) = |R(\theta_{ref}) - R(\theta_{local})|
````

---

## Release 1.0.3

IGAD is packaged as `visigence-igad` version `1.0.3`. The import name is
`igad`; the distribution name on PyPI is `visigence-igad`.

```text
Name: visigence-igad
Version: 1.0.3
Author: Omry Damari
Author email: omryv@pm.me
License: MIT
Python: >=3.10
```

Build artifacts:

```text
visigence_igad-1.0.3.tar.gz
visigence_igad-1.0.3-py3-none-any.whl
```

Install from the built wheel:

```bash
python -m pip install dist/visigence_igad-1.0.3-py3-none-any.whl
```

Install from source for development:

```bash
python -m pip install -e ".[dev]"
```

Build locally:

```bash
rm -rf build dist *.egg-info
python -m build
```

Verify the installed package version:

```bash
python - <<'PY'
import igad

print(igad.__version__)
assert igad.__version__ == "1.0.3"
PY
```

The release wheel intentionally excludes `experiments/` and `tests/`; they remain available in the source repository.

---

## Core Claim

> **The anomaly is not where the distribution is. It is what shape it has.**

where `R(θ)` is the **scalar curvature** of the Fisher–Rao statistical manifold at the natural parameter point `θ`.

---

## The Problem

Every widely used anomaly detector shares the same assumption:

> **anomaly = a point far from the center**

| Method           | What It Measures                                       |
| ---------------- | ------------------------------------------------------ |
| Z-Score          | Distance from mean in standard deviation units         |
| Mahalanobis      | Distance from cloud center accounting for correlations |
| Isolation Forest | Ease of isolating a point in feature space             |
| LOF              | Relative local neighborhood density                    |

All four are blind to the following:

```text
Reference : Gamma(8, 2)        mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(...)     mean=4.000  var=2.000  skew=1.105
```

Mean and variance are exactly identical. The internal structure of the distribution has changed completely. Distance-based algorithms do not target this kind of shape shift.

---

## What Was Known Before This Work

Every mathematical identity used here is an established result:

| Component                                      | Source                    |
| ---------------------------------------------- | ------------------------- |
| Fisher–Rao metric                              | Rao (1945)                |
| Differential geometry of exponential families  | Amari (1985)              |
| Scalar curvature formula for Hessian metrics   | Amari & Nagaoka (2000)    |
| Fourth-cumulant cancellation in Riemann tensor | Standard Hessian geometry |
| Curvature as detector of phase transitions     | Ruppeiner (1979, 1995)    |

## What Is New

| Component        | Description                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Construction** | Using scalar curvature deviation as a batch-level anomaly score                                                          |
| **Insight**      | Scalar curvature, governed by the full contraction `‖T‖²_g`, is structurally sensitive to shape shifts                   |
| **Validation**   | A control experiment isolating geometry from MLE efficiency confirms that the curvature tensor itself contributes signal |

Full derivation with attribution: [`docs/proof.md`](docs/proof.md)

---

## Mathematical Foundation

For an exponential family with log-partition `A(θ)`:

```text
Fisher metric:          gᵢⱼ(θ)   = ∂²A / ∂θᵢ∂θⱼ
Third cumulant tensor:  Tᵢⱼₖ(θ)  = ∂³A / ∂θᵢ∂θⱼ∂θₖ
Christoffel symbols:    Γᵢⱼ,ₖ    = ½ · Tᵢⱼₖ
Scalar curvature:       R(θ)      = ¼ · ( ‖T‖²_g − ‖S‖²_g )
```

where:

```text
Sₘ      = gᵃᵇ Tₐᵦₘ
‖T‖²_g  = gⁱᵃ gʲᵇ gᵏᶜ Tᵢⱼₖ Tₐᵦᶜ
```

The critical quantity is `‖T‖²_g`: a three-index contraction of the third cumulant tensor against the inverse metric. It gives a geometrically weighted measure of total skewness content. Unlike `scipy.stats.skew`, it uses the full parametric structure of the family.

---

## Implementation

```text
igad/
  __init__.py         Package version and public exports
  curvature.py        Fisher metric, third cumulant tensor, scalar curvature,
                      O(k) Dirichlet curvature via Sherman-Morrison
  families.py         GammaFamily, PoissonFamily, DirichletFamily
  detector.py         IGADDetector batch-level scoring

tests/
  test_curvature.py        Curvature and Gamma family validation
  test_dirichlet_family.py Dirichlet validation and sample efficiency
  test_sherman_morrison.py O(k) route vs dense routes vs 120-digit reference
  test_highprec.py         The high-precision reference, against identities
  test_trace_schema.py     Router-trace contract, incl. post-top-k rejection
  test_quality_schema.py   3D-quality contract, incl. router-derived labels
  test_router_stats.py     Baselines and structure-aware statistics
  test_evaluation.py       Metrics and the object-level protocol

experiments/
  demo_easy.py             Experiment 1: Gamma vs Gamma
  demo_hard.py             Experiment 2: Gamma vs LogNormal + MLE control
  demo_gaussian2d.py       Experiment 3: Gaussian failure mode
  demo_dirichlet.py        Experiment 4: Dirichlet shape shifts
  demo_moe_router.py       Experiment 5: MoE router monitoring + MLE control

  audit_environment.py     Part 0 hard gate, measured (exits non-zero on fail)
  report_test_status.py    What ran, what could not, and why
  benchmark_sherman_morrison.py  O(k^3) vs O(k^2) vs O(k), measured
  highprec_reliability.py  120-digit arbitration and the reliability boundary
  special_function_accuracy.py   psi/psi'/psi'' vs the 120-digit reference
  make_figures.py          SVG figures, read only from saved JSON

  trace_schema.py          Router-trace record contract + validator
  quality_schema.py        3D-quality record contract + validator
  router_stats.py          Cheap baselines and structure-aware statistics
  evaluation.py            Metrics + object-level bootstrap/splits/paired tests

docs/
  proof.md                 Mathematical background with full attribution
  moe_router.md            Experiment 5 results and operational guards
  sherman_morrison.md      Derivation of the O(k) curvature path
  numerical_reliability.md When R(alpha) can be trusted, and why
  acquisition_checklist.md What the 3D benchmark needs before it can run
  experiment_plan.md       The early-warning experiment, step by step
  handoff.md               What is proven, what needs external resources
  router_geometry.html     Concentration vs geometry: figures from sampled data
  figures/                 Experiment plots with descriptions

RESULTS.md                 Full experimental results and analysis
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

print(f"IGAD score: {score:.6f}")  # Higher = more anomalous
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Experimental Results

### Experiment 1 — Easy Case

**Gamma(9, 3) vs Gamma(1.5, 0.5)** · same mean, different variance and skewness

```text
Method                 AUC-ROC
------------------------------
IGAD (curvature)        1.0000
Variance shift          1.0000
Skewness shift          0.9834
Mean shift              0.8150
```

IGAD achieves perfect separation. Variance baseline also reaches 1.0 because variance differs by 6×. Experiment 2 is the key result.

---

### Experiment 2 — Hard Case

**Gamma(8, 2) vs LogNormal** · `mean = 4.0` and `var = 2.0` are identical for both.

```text
Reference : Gamma(8, 2)              mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(μ=1.327,       mean=4.000  var=2.000  skew=1.105
            σ=0.343)
```

A control baseline was constructed using the identical MLE fit as IGAD but discarding the curvature tensor:

```text
skew_MLE(batch) = 2 / √α_MLE
score = |skew_MLE - skew_ref|
```

#### Results — 5 seeds, n = 200

```text
Method                        Mean AUC   ± Std
----------------------------------------------
IGAD (curvature)               0.6542    0.047
MLE skewness [CONTROL]         0.6016    0.038
Raw skewness                   0.6794    0.072
Mean shift [BLIND]             0.5240    0.062
Variance shift [BLIND]         0.5818    0.027
```

**Gap: IGAD − MLE skewness = +0.053.**

This indicates that curvature geometry adds signal beyond MLE efficiency alone.

#### Scaling with batch size

```text
n        IGAD      MLE-skew   Raw-skew   Gap (IGAD − MLE)
----------------------------------------------------------
100      0.5704    0.5764     0.5908     −0.006
200      0.6838    0.6098     0.6514     +0.074
500      0.6748    0.5846     0.9194     +0.090
1000     0.7892    0.8214     0.9686     −0.032
```

IGAD beats the MLE control at n = 200 and n = 500. At n = 1000, model misspecification degrades the curvature signal, and model-free methods dominate.

---

### Experiment 3 — Gaussian Failure Mode

Bivariate Gaussian, `ρ_ref = 0.2` vs `ρ_anom = 0.8`. Mean and marginal variances are identical.

```text
ρ_ref=0.20, ρ_anom=0.80   →   |ΔR| = 0.003308
ρ_ref=0.50, ρ_anom=0.55   →   |ΔR| = 0.000049
```

All methods reached **AUC = 1.0** — not because of curvature, but because the correlation difference is large enough for any method to detect. IGAD adds no unique value here.

**Reason:** the Gaussian manifold has constant scalar curvature. IGAD is not applicable to Gaussian families.

---

### Experiment 4 — Dirichlet Family

IGAD extends to **Dirichlet(α₁, …, αₖ)** with `k ≥ 3`, where pure shape variation is possible with fixed lower-order moments.

* Fisher metric matches numerical Hessian.
* Third cumulant tensor analytical form agrees with numerical derivatives.
* Scalar curvature varies meaningfully with concentration and asymmetry.
* IGAD detects Dirichlet shape shifts at n = 200 and beats random at n = 50.
* AUC monotonically increases with n on well-specified data.

---

## Summary

```text
╔══════════════════╦═══════════════╦═════════════╦═══════════════════╗
║ Method           ║  Mean Shift   ║ Shape Shift ║ Low-Sample (n<300)║
╠══════════════════╬═══════════════╬═════════════╬═══════════════════╣
║ Z-Score          ║      ✓        ║      ✗      ║        ✓          ║
║ Mahalanobis      ║      ✓        ║      ✗      ║        ~          ║
║ Isolation Forest ║      ✓        ║      ✗      ║        ✗          ║
║ Skewness Test    ║      ✗        ║      ~      ║        ✗          ║
║ IGAD             ║      ~        ║      ✓      ║        ✓          ║
╚══════════════════╩═══════════════╩═════════════╩═══════════════════╝
```

---

## When to Use IGAD

* The correct parametric family is known or approximately known.
* Batch sizes are moderate: 50–300 observations.
* Anomalies differ in distributional shape, not only location or scale.
* The family has dimension `d ≥ 2`; 1D manifolds have `R = 0`.

Potential applications:

* Predictive maintenance: vibration profile shape changes before amplitude changes.
* Financial monitoring: transaction distribution structure shifts.
* Medical signal analysis: ECG waveform geometry changes in early arrhythmia.
* Cybersecurity: packet-size distribution shifts in low-and-slow exfiltration.

## When Not to Use IGAD

* Anomalies are simple outliers far from center; use Isolation Forest or similar.
* No parametric model is appropriate; use model-free tests.
* Batch sizes are large and the model is approximate; raw shape statistics may dominate.
* The family is 1D: Poisson, Exponential, Bernoulli.
* The family is Gaussian; scalar curvature is constant.

---

## Documented Limitations

| Limitation                      | Explanation                                 |
| ------------------------------- | ------------------------------------------- |
| Model specification required    | Wrong family can degrade signal at large n  |
| 1D families                     | `R ≡ 0` for Poisson, Exponential, Bernoulli |
| Gaussian families               | `R` is constant under the relevant geometry |
| Large n with misspecified model | Model-free methods can dominate             |
| Computational cost              | `O(d⁶)` literal / `O(d⁴)` pairwise for a general family; **`O(k)` for Dirichlet** |
| Numerical cancellation          | ~`16 − log₁₀ ρ` digits survive; `curvature_reliability` reports ρ |

---

## Validation — Automated Tests

**Current status is measured, not transcribed.** Run:

```bash
python -m experiments.report_test_status
```

It classifies every test into three exclusive buckets — *verified locally*
(executed here and passed), *skipped, no dependency*, and *not collected* —
and writes `experiments/results/test_status.json`. Only the first bucket may
be described as verified.

CI is reported separately. Every GitHub Actions run on this repository,
including runs on `main`, currently ends in 1–3 seconds with `runner_id: 0`
and no executed steps: the jobs never receive a runner. That is **CI
infrastructure unavailability**, not a code result in either direction, and
it predates the current work.

The block below is the historical record of the pinned `IGAD-Ver1.0.0`
validation run, kept for the release citation. It is not a current status.

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

Every documented limitation is enforced by a test that would fail if the limitation stopped holding.

---

## References

* Rao, C.R. (1945). *Information and the accuracy attainable in the estimation of statistical parameters.* Bull. Calcutta Math. Soc.
* Amari, S. (1985). *Differential-Geometrical Methods in Statistics.* Springer.
* Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry.* AMS / Oxford.
* Ruppeiner, G. (1979). *Thermodynamics: A Riemannian geometric model.* Phys. Rev. A.
* Ruppeiner, G. (1995). *Riemannian geometry in thermodynamic fluctuation theory.* Rev. Mod. Phys.

---

## License

MIT [LICENSE](LICENSE).
