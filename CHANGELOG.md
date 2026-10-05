# Changelog

All notable changes to IGAD are documented in this file.

## 1.0.4 — Hard Case result withdrawn

### Retracted: "IGAD beats the MLE-skewness control by +0.053 AUC"

The Hard Case (Gamma(8,2) vs LogNormal, matched mean and variance) was the
evidence for the project's central claim: that `‖T‖²_g` extracts shape
information no single moment captures, raw or MLE-fitted. That claim is
withdrawn for the Gamma family.

**Why it cannot hold.** For Gamma, `R` depends on the shape α alone and is
strictly monotone in it. The control `2/√α̂` is a function of the same α̂, so
the IGAD score is a re-scaling of the control: on either side of the reference
the two order batches identically.

**What the experiment actually showed.** `experiments/demo_hard.py` computed
curvature by finite differences of the log-partition, not with the exact
Fisher metric and cumulant tensor `IGADDetector` uses. Near α = 8 the
finite-difference error (2–9 × 10⁻³, varying with the rate β) is 10–30× the
true curvature difference between the classes (3 × 10⁻⁴). With exact curvature,
over 40 seeds, IGAD scores 0.011–0.017 AUC **below** the control at every
n from 100 to 1000; on the original five seeds, 0.5785 instead of the published
0.6542, against the control's 0.6016.

**The shipped detector is unchanged.** It always used the exact route; only
the experiment scripts and the claims built on them were wrong.

### Changed

- `experiments/demo_hard.py`: exact curvature, 40 seeds at every batch size,
  paired gaps with standard errors, results saved to
  `experiments/results/hard_case_exact.json`. The finite-difference column is
  kept, labelled, so the old number can be traced.
- `demo_hard_extended.py`, `demo_hard1.py`, `demo_easy.py`: exact Gamma
  curvature. `demo_dirichlet.py`: the exact O(k) closed form. With it IGAD
  improves on Dirichlet (AUC 0.993 at n = 20, was 0.754) and ties MMD and
  Wasserstein from n = 100; the published "WINS at small n" was not supported
  either before or after.
- `README.md`, `RESULTS.md`, `docs/index.md`, `docs/operational_envelope.md`,
  `docs/figures/README.md`: claims corrected. Also corrected: the Dirichlet
  rationale (mean and one marginal variance *do* determine α uniquely), the
  Gaussian 2D separation (an artefact of finite-difference error; the true
  `R` is constant), the stale-sign Dirichlet `R` values, and the cost figure
  (O(d⁶) literal / O(d⁴) pairwise, not O(d³)).
- Figures `exp1`, `exp2` and `exp4` regenerated.
- Repository links (README, docs, `pyproject.toml` project URLs) point to
  `github.com/omryv/IGAD`, where the repository now lives.

### History rewritten

Commit authorship and attribution trailers were normalised across the whole
history, and every tag was moved to its rewritten commit. File contents are
unchanged apart from one machine-local path in
`experiments/results/test_status.json`; every rewritten commit and tag has a
byte-identical tree to the original. Hashes cited externally map as follows:

| Original | Rewritten | Cited by |
| --- | --- | --- |
| `81dd1eb` | `6fbd779` | validation run 25236119831, Zenodo v1, PyPI 1.0.2 |
| `156160d` | `2b31d5f` | release tag `IGAD-VER-1.0.0`, Zenodo v1 |

### Licence

Relicensed from MIT to the Apache License 2.0 (`LICENSE`, `NOTICE`),
copyright Omry Damari. Releases up to and including 1.0.3 remain available
under the MIT License they were published with.

### Added

- `tests/test_gamma_reduction.py`: pins that Gamma `R` is independent of the
  rate, strictly monotone in the shape, that the score is invariant to
  rescaling the batch, and that it ranks batches identically to the MLE
  skewness on either side of the reference.

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
