
# IGAD - Information-Geometric Anomaly Detection

**IGAD** detects distributional anomalies by measuring scalar curvature on the statistical manifold of an exponential family.

Instead of comparing only mean, variance, or skewness, IGAD contracts the full third cumulant tensor against the Fisher–Rao metric. The result is a geometric anomaly score that can expose shape differences not captured by a single moment.

---

<p align="center">
  <a href="https://github.com/Visigence/IGAD/actions/workflows/test.yml">
    <img src="https://github.com/Visigence/IGAD/actions/workflows/test.yml/badge.svg"
         alt="CI workflow status">
  </a>

  <img src="https://img.shields.io/badge/release-IGAD--Ver1.0.0-blue?logo=git&logoColor=white"
       alt="Release IGAD-Ver1.0.0">

  <a href="PASTE_EXACT_GITHUB_ACTIONS_RUN_URL_HERE">
    <img src="https://img.shields.io/badge/GitHub%20validated-54%2F54%20tests-brightgreen?logo=github&logoColor=white"
         alt="GitHub validated: 54/54 tests passed">
  </a>

  <a href="https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512">
    <img src="https://img.shields.io/badge/verified%20commit-81dd1eb-6e40c9?logo=github&logoColor=white"
         alt="Verified commit 81dd1eb">
  </a>

  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab?logo=python&logoColor=white"
       alt="Python 3.10 | 3.11 | 3.12">

  <img src="https://img.shields.io/badge/license-MIT-blue"
       alt="License: MIT">
</p>

---

## Verified Release Evidence

| Field | Value |
|------|-------|
| Release | `IGAD-Ver1.0.0` |
| Verified commit | [`81dd1eb4540643083854232d9645f6add4150512`](https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512) |
| Validation environment | GitHub-hosted Linux Actions runners |
| Python versions | 3.10, 3.11, 3.12 |
| Test result | 54/54 tests passed |
| Evidence | [GitHub Actions validation run](https://github.com/Visigence/IGAD/actions/runs/25236119831) |

The validation run produced separate test-report artifacts for each Python runtime.

---

## Install

```bash
pip install igad
````

---

## Documentation

* [Experimental Results](../RESULTS.md)
* [Operational Envelope](operational_envelope.md)
* [Mathematical Proof](proof.md)

---

## Core Claim

IGAD tests whether scalar curvature contains anomaly signal beyond ordinary moment comparisons.

The falsifiable claim is:

> In matched mean/variance regimes, the curvature contraction `‖T‖²_g` can recover shape information that is not captured by MLE-derived skewness alone.

The current evidence supports this claim in the tested regime `n = 200–500`.

---

## Experiment Figures

All figures are reproducible from the repository experiments.

### Experiment 1 — Easy Case

Gamma vs Gamma with different variance.

![Experiment 1 — Easy Gamma vs Gamma](figures/exp1_easy_gamma_vs_gamma.png)

In this regime, IGAD reaches perfect separation, but variance shift also succeeds. This experiment verifies correctness, not geometric advantage.

---

### Experiment 2 — Hard Case

Gamma vs LogNormal with matched mean and matched variance.

![Experiment 2 — Hard Gamma vs LogNormal](figures/exp2_hard_gamma_vs_lognormal.png)

This is the key test. The distributions share first and second moments, so the anomaly signal must come from higher-order shape structure.

---

### Experiment 3 — Gaussian 2D Correlation

Two-dimensional Gaussian structure with correlation shift.

![Experiment 3 — Gaussian 2D Correlation](figures/exp3_gaussian2d_correlation.png)

This experiment checks whether the method detects geometric structure in a multivariate setting.

---

### Experiment 4 — Dirichlet Sample Efficiency

Dirichlet-family sample-efficiency evaluation.

![Experiment 4 — Dirichlet Sample Efficiency](figures/exp4_dirichlet_sample_efficiency.png)

This experiment measures how the curvature signal behaves as sample size changes.

---

## Known Limitations

IGAD is not a universal anomaly detector.

* It requires a suitable exponential-family model.
* One-dimensional flat families such as Poisson, Exponential, and Bernoulli have scalar curvature `R = 0`.
* Under model misspecification, model-free methods can dominate.
* The tensor contraction cost scales as `O(d³)` per evaluation.

---

## Repository

Source code, tests, workflows, and reproducibility scripts are available at:

[github.com/Visigence/IGAD](https://github.com/Visigence/IGAD)

