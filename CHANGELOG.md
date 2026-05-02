# Changelog

All notable changes to IGAD are documented in this file.

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


