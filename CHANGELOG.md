# Changelog

All notable changes to IGAD are documented in this file.

## 1.0.3 — Curvature sign correction, and the 1.0.2 detector restored

### Fixed: the sign of the scalar curvature

`scalar_curvature` and every route that shares its identity returned the
**negative** of the Levi-Civita scalar curvature. The identity is now

```
R = 1/4 ( ||T||^2_g - ||S||^2_g )
```

where it previously read `1/4 ( ||S||^2_g - ||T||^2_g )`.

The error came from `docs/proof.md` section 3, which asserted that the whole
derivative-of-Christoffel term cancels by antisymmetry in `(i, j)`. Only its
fourth-cumulant half does. The other half,
`1/2 (d_i g^{lm}) T_{jkm}` with `d_i g^{lm} = -g^{la} g^{mb} T_{iab}`, is
quadratic in `T`, survives the antisymmetrisation, and flips the sign.
Section 3 now derives both halves.

**What changes for callers.** Every reported `R` changes sign: the univariate
Gaussian Fisher-Rao manifold now returns the textbook `-1` instead of `+1`,
and Dirichlet curvatures are negative. Anomaly scores are **unaffected** —
`score_batch` returns `|R_ref - R_local|`, in which a global sign cancels.

The sign is now pinned by `tests/test_curvature_ground_truth.py` against
values independent of this codebase: `R = -1` for the univariate Gaussian,
`R = -d(d+1)^2/4` for the full mean-and-covariance Gaussian family, and
`R = 0` for one-dimensional families. No previous test could catch a sign
error, because every one compared the formula to itself or to a
high-precision evaluation of the same formula.

The same correction is applied to the standard-library mirror
(`experiments/_router_common.py`) and the high-precision reference
(`experiments/_highprec.py`) so all three remain in agreement.

### Fixed: `IGADDetector` regressed to the pre-1.0.2 point-level API

`main` had reverted to the k-NN point-level detector that 1.0.2 replaced,
while PyPI continued to serve 1.0.2. The batch-level API is restored:
`score_batch`, and `fit(X=...)` or `fit(theta_ref=...)`.

Restoring it fixes, in one change, defects that were present in the
point-level detector:

- the score was the **signed** `R_ref - R_local` while the documented and
  published formula is `|R_ref - R_local|`. Since `R` is not monotone in
  concentration, the signed form scored below chance on diffuse anomalies;
- `_scalar_curvature` never passed a family's exact
  `fisher_metric_analytical`, so non-Dirichlet families were scored from a
  finite-difference metric that is sign-wrong at large concentration;
- `k_neighbors > len(X_train)` raised an uncaught `IndexError`;
- an `np.inf` failure sentinel made `np.percentile` return `NaN`, so
  `predict()` reported zero anomalies — the guard failed open;
- `fit()` validated nothing: `n=0` produced a `NaN` reference silently;
- attributes initialised to `None` carried no `Optional` annotation.

`_scalar_curvature` keeps the O(k) Dirichlet dispatch added after 1.0.2, so
the closed-form route is still preferred where a family provides it.

### Added

- `tests/test_curvature_ground_truth.py`: curvature against externally
  known values, including the sign.
- `tests/test_detector_e2e.py`: restored from 1.0.2, the end-to-end Hard
  Case benchmark whose absence let the detector regression through.

### Note on versioning

1.0.0 was declared here while PyPI already served 1.0.0, 1.0.1 and 1.0.2, so
a release from this tree could not have been published. The version now
moves past the published one.

## 1.0.2 — Detector API alignment

### Breaking change

`IGADDetector` is now a batch-level detector matching the algorithm
validated in RESULTS.md. The previous point-level interface
(`score_samples`, `predict`) has been replaced with `score_batch`.

### Migration

Old API (1.0.1):

```python
detector.score_samples(X)
detector.predict(X, contamination=0.1)
```

New API (1.0.2):

```python
detector = IGADDetector(family=GammaFamily()).fit(
    theta_ref=GammaFamily.to_natural(8.0, 2.0)
)
score = detector.score_batch(X_batch)
```

`fit` now accepts either `X` (training data, n >= 1000 recommended)
or `theta_ref` (known natural parameters). Exactly one is required.

### Why

The 1.0.1 detector evaluated curvature on local k-NN neighborhoods,
which deviated from the batch-level algorithm responsible for the
RESULTS.md figures and produced AUC near 0.5 on the canonical Hard Case.
The 1.0.2 detector reproduces RESULTS.md within reported variance.

### Added

- `tests/test_detector_e2e.py`: end-to-end tests against the
  canonical Gamma vs LogNormal benchmark with mean and std bounds.
