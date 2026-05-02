## Verified

<p>
  <a href="https://github.com/Visigence/IGAD/actions/runs/25236119831">
    <img src="https://img.shields.io/badge/GitHub%20Actions-validated%2054%2F54-brightgreen?logo=github&logoColor=white"
         alt="GitHub Actions validated: 54/54 tests passed">
  </a>
  <a href="https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512">
    <img src="https://img.shields.io/badge/validated%20commit-81dd1eb-6e40c9?logo=github&logoColor=white"
         alt="Validated commit 81dd1eb">
  </a>
</p>

IGAD-VER-1.0.0 was validated on GitHub Actions at commit
[`81dd1eb4540643083854232d9645f6add4150512`](https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512).
The PyPI release was built from commit
[`156160d59f288d11451895ecb8c234ff7ef9c895`](https://github.com/Visigence/IGAD/commit/156160d59f288d11451895ecb8c234ff7ef9c895).
The validation run executed on GitHub-hosted Linux runners and completed successfully across Python 3.10, 3.11, and 3.12, producing separate test-report artifacts for each runtime.

[View the exact GitHub Actions validation run](https://github.com/Visigence/IGAD/actions/runs/25236119831).

---

## What is IGAD?

IGAD detects anomalies by measuring scalar curvature on the statistical manifold of an exponential family. Instead of comparing moments directly, it contracts the full third cumulant tensor against the Fisher–Rao metric, extracting shape information that no single moment captures.

---

## Experimental Results

All results are reproducible via `experiments/`.

### Easy Case — Gamma vs Gamma

Different variance.

| Method | AUC-ROC |
|--------|---------|
| IGAD (curvature) | 1.0000 |
| Variance shift | 1.0000 |
| Skewness shift | 0.9834 |
| Mean shift | 0.8150 |

IGAD achieves perfect separation. Variance shift matches it, so this regime does not isolate the geometric advantage.

### Hard Case — Matched Mean and Variance

Same mean = 4.0. Same variance = 2.0. Gamma vs LogNormal.  
Evaluation: n = 200, 5 seeds.

| Method | Mean AUC | ± Std |
|--------|----------|-------|
| IGAD (curvature) | 0.6542 | 0.0469 |
| MLE skewness (control) | 0.6016 | 0.0382 |
| Raw skewness | 0.6794 | 0.0722 |
| Mean shift | 0.5240 | 0.0618 |
| Variance shift | 0.5818 | 0.0266 |

IGAD beats the MLE-skewness control by +0.053, showing that scalar curvature contributes signal beyond MLE efficiency alone.

### Scaling with Batch Size

Seed = 42.

| n | IGAD | MLE-skew | Gap |
|---|------|----------|-----|
| 100 | 0.5704 | 0.5764 | −0.006 |
| 200 | 0.6838 | 0.6098 | +0.074 |
| 500 | 0.6748 | 0.5846 | +0.090 |
| 1000 | 0.7892 | 0.8214 | −0.032 |

The geometric advantage is strongest at n = 200–500. At n = 1000, model misspecification degrades the curvature signal.

---

## Honest Limitations

- Model specification required: IGAD needs a correctly chosen exponential family.
- 1D families such as Poisson, Exponential, and Bernoulli are flat, with R = 0.
- Model-free methods can dominate at n > 500 under misspecification.
- Computational cost is O(d³) tensor contractions per evaluation.

---

## Falsifiable Claim

IGAD's advantage over MLE-derived skewness, using the identical MLE fit but discarding the curvature tensor, shows that the full contraction `‖T‖²_g` extracts shape information not captured by any single moment, raw or MLE-fitted.

This claim is supported in the tested regime n = 200–500 and remains falsifiable outside that regime.

---

## Verified Test Suite

- Validated commit: [`81dd1eb4540643083854232d9645f6add4150512`](https://github.com/Visigence/IGAD/commit/81dd1eb4540643083854232d9645f6add4150512)
- Release commit: [`156160d59f288d11451895ecb8c234ff7ef9c895`](https://github.com/Visigence/IGAD/commit/156160d59f288d11451895ecb8c234ff7ef9c895)
- Release tag: `IGAD-VER-1.0.0`
- Python versions: 3.10, 3.11, 3.12
- Result: 54/54 tests passed
- Evidence: [GitHub Actions validation run](https://github.com/Visigence/IGAD/actions/runs/25236119831)

---

## Install

```bash
pip install visigence-igad
```
