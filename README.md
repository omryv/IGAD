# IGAD
**Information-Geometric Anomaly Detection**

<p>
  <a href="https://github.com/omryv/IGAD/actions/workflows/test.yml">
    <img src="https://github.com/omryv/IGAD/actions/workflows/test.yml/badge.svg?branch=main"
         alt="Tests">
  </a>
  <a href="https://github.com/omryv/IGAD/actions/runs/37272023848">
    <img src="https://img.shields.io/badge/validated%20commit-3b5c822-b2f12a?logo=github&logoColor=white"
         alt="Validated commit 3b5c822">
  </a>
</p>

> *The anomaly is not only where the distribution lives, it is what shape it becomes.*
>
> Omry Damari

---

## Current research phase: router structure vs 3D quality

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
| `experiments/trace_schema.py`, `experiments/quality_schema.py` | tested record contracts: one rejects a post-top-k capture, the other rejects a failure label derived from routing |
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

The current code is validated by GitHub Actions at commit
[`3b5c822b2f12acc5178d9ad66196775fa9af880b`](https://github.com/omryv/IGAD/commit/3b5c822b2f12acc5178d9ad66196775fa9af880b)
([run 37272023848](https://github.com/omryv/IGAD/actions/runs/37272023848)):
480 tests passing on Python 3.10, 3.11 and 3.12, with a clean dependency audit.
The original 1.0.0 baseline is commit
[`6fbd779bc70b50f53238fd0cd469b7e8d9451a69`](https://github.com/omryv/IGAD/commit/6fbd779bc70b50f53238fd0cd469b7e8d9451a69)
(release `IGAD-Ver1.0.0`).

This repository is public for reproducibility, verification, and independent review.  
Future changes should be treated as new research iterations and must be validated by a new GitHub Actions run.
IGAD detects distributional shape shifts using scalar curvature deviation on the Fisher–Rao statistical manifold.

```math
IGAD(batch) = |R(\theta_{ref}) - R(\theta_{local})|
````

> **Correction (1.0.3) — the Hard Case result is withdrawn.** Earlier versions
> of this README reported that IGAD beats a same-fit MLE-skewness control by
> +0.053 AUC on Gamma vs LogNormal, and concluded that `‖T‖²_g` extracts shape
> information no single moment captures. That result came from finite-difference
> error in the experiment script. With the exact curvature the shipped detector
> uses, IGAD scores **below** the control at every batch size, and it must: for
> the Gamma family `R` is a monotone function of the fitted shape α̂ alone, so
> the IGAD score is a re-scaling of the MLE skewness `2/√α̂`. Details:
> [Experiment 2](#experiment-2--hard-case) and `RESULTS.md`.

---

## Release 1.0.3

IGAD is packaged as `visigence-igad` version `1.0.3`. The import name is
`igad`; the distribution name on PyPI is `visigence-igad`.

```text
Name: visigence-igad
Version: 1.0.3
Author: Omry Damari
Author email: omryv@pm.me
License: Apache-2.0
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
| **Validation**   | A same-fit control experiment was built to isolate geometry from MLE efficiency. Run with exact curvature, it shows **no** geometric contribution for Gamma (see Experiment 2) |

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

The critical quantity is `‖T‖²_g`: a three-index contraction of the third cumulant tensor against the inverse metric. It gives a geometrically weighted measure of total skewness content. Unlike `scipy.stats.skew`, it uses the full parametric structure of the family — which also means it is a function of the fitted parameters only. For a two-parameter scale family such as Gamma, `R` depends on the fitted shape alone, and so carries the same information as the MLE shape estimate.

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
  test_gamma_reduction.py  Gamma: R depends on alpha alone; IGAD ranks as MLE skewness
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

IGAD achieves perfect separation. Variance baseline also reaches 1.0 because variance differs by 6×. Experiment 2 was designed to isolate the geometric contribution; it does not find one.

---

### Experiment 2 — Hard Case

**Gamma(8, 2) vs LogNormal** · `mean = 4.0` and `var = 2.0` are identical for both.

```text
Reference : Gamma(8, 2)              mean=4.000  var=2.000  skew=0.707
Anomaly   : LogNormal(μ=1.327,       mean=4.000  var=2.000  skew=1.105
            σ=0.343)
```

A control baseline uses the identical MLE fit as IGAD but discards the curvature tensor:

```text
skew_MLE(batch) = 2 / √α_MLE
score = |skew_MLE - skew_ref|
```

**Why IGAD cannot beat this control.** For Gamma, `R` depends on the shape α
alone (rescaling the data is an isometry of the Fisher metric) and is strictly
monotone in it. `skew_MLE` is a function of the same α̂. So on either side of
the reference, the IGAD score and the control rank batches identically; they
differ only in how `|·|` weighs one side against the other.
`tests/test_gamma_reduction.py` pins this.

#### Results — 40 seeds, exact curvature (`python -m experiments.demo_hard`)

```text
n       IGAD     MLE-skew   Raw-skew   Gap (IGAD − MLE) ± SE
-------------------------------------------------------------
100     0.5513   0.5686     0.5959     −0.0172 ± 0.0012
200     0.5981   0.6146     0.7016     −0.0165 ± 0.0009
500     0.6876   0.7013     0.8760     −0.0136 ± 0.0006
1000    0.8125   0.8232     0.9736     −0.0107 ± 0.0005
```

IGAD scores below the MLE-skewness control at every batch size, and model-free
raw skewness beats it from n = 200 upward. Mean and variance tests stay near
chance (0.49–0.58), as designed.

#### What happened to the published +0.053

Before 1.0.3 the experiment script computed curvature by finite differences,
not with the exact Fisher metric and cumulant tensor the detector uses. Near
α = 8 that error (2–9 × 10⁻³, varying with the rate β) is 10–30× the true
curvature difference between the two classes (3 × 10⁻⁴). On the original five
seeds at n = 200, every baseline reproduces exactly and only IGAD changes:
0.6542 with finite differences, **0.5785** with exact curvature, against the
control's 0.6016. The single-seed scaling table that showed a +0.074 / +0.090
advantage at n = 200 / 500 came from the same error.

---

### Experiment 3 — Gaussian Failure Mode

Bivariate Gaussian, `ρ_ref = 0.2` vs `ρ_anom = 0.8`. Mean and marginal variances are identical.

```text
ρ_ref=0.20, ρ_anom=0.80   →   |ΔR| = 0.003308
ρ_ref=0.50, ρ_anom=0.55   →   |ΔR| = 0.000049
```

All methods reached **AUC = 1.0** — not because of curvature, but because the correlation difference is large enough for any method to detect. IGAD adds no unique value here.

**Reason:** the Gaussian manifold has constant scalar curvature. IGAD is not applicable to Gaussian families.

The nonzero `|ΔR|` values above are themselves finite-difference error: this family's scalar curvature is exactly −2 everywhere, so computed exactly every IGAD score here is zero.

---

### Experiment 4 — Dirichlet Family

IGAD extends to **Dirichlet(α₁, …, αₖ)** with `k ≥ 3`. Note that the mean vector and any one marginal variance determine α uniquely, so a Dirichlet has no "same mean and variance, different shape" alternative.

* Fisher metric matches numerical Hessian.
* Third cumulant tensor analytical form agrees with numerical derivatives.
* Scalar curvature varies meaningfully with concentration and asymmetry.
* On Dirichlet(4,4,4) vs Dirichlet(1.5,4,6.5), with the exact O(k) curvature, IGAD reaches AUC 0.993 at n = 20 and 1.000 from n = 100, tying MMD and Wasserstein from n = 100. That anomaly also shifts the marginal means, and no same-fit control was run, so this does not show a geometric advantage.

---

## Summary

```text
╔══════════════════╦═══════════════╦═════════════╦═══════════════════╗
║ Method           ║  Mean Shift   ║ Shape Shift ║ Low-Sample (n<300)║
╠══════════════════╬═══════════════╬═════════════╬═══════════════════╣
║ Z-Score          ║      ✓        ║      ✗      ║        ✓          ║
║ Mahalanobis      ║      ✓        ║      ✗      ║        ~          ║
║ Isolation Forest ║      ✓        ║      ✗      ║        ✗          ║
║ Skewness Test    ║      ✗        ║      ✓      ║        ~          ║
║ IGAD             ║      ~        ║      ~      ║        ~          ║
╚══════════════════╩═══════════════╩═════════════╩═══════════════════╝
```

On the Gamma hard case, raw sample skewness beats IGAD from n = 200 upward, and
IGAD is a re-scaled MLE skewness that scores slightly below it.

---

## When to Use IGAD

* The correct parametric family is known or approximately known.
* Batch sizes are moderate: 50–300 observations.
* Anomalies differ in distributional shape, not only location or scale.
* The family has dimension `d ≥ 2`; 1D manifolds have `R = 0`.

Potential applications (none has been tested on real data):

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
* The family is Gamma, or another two-parameter scale family: `R` depends on the fitted shape alone, so use the MLE shape estimate (or a likelihood-ratio test) directly.

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

### Current CI result

| Field | Value |
| --- | --- |
| Workflow | [`Tests`](https://github.com/omryv/IGAD/actions/workflows/test.yml) (`.github/workflows/test.yml`) |
| Commit | [`3b5c822b2f12acc5178d9ad66196775fa9af880b`](https://github.com/omryv/IGAD/commit/3b5c822b2f12acc5178d9ad66196775fa9af880b) |
| Run | [37272023848](https://github.com/omryv/IGAD/actions/runs/37272023848), 2026-10-05 |
| Python | 3.10, 3.11, 3.12 — all three jobs passed |
| Tests | **480 passed**, 0 failed |
| Dependency audit | `pip-audit`: no known vulnerabilities |
| Test reports | HTML and JSON artifacts attached to the run, one per Python version |

Every push and pull request runs the same workflow; the badge at the top of
this README shows the state of `main`.

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

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Omry Damari. Releases up to and including 1.0.2 were published under the MIT License.
