"""
End-to-end tests for IGADDetector.

Verifies the canonical Hard Case from RESULTS.md:
Gamma(alpha=8, beta=2) versus LogNormal with matched mean=4.0, var=2.0.
The two distributions are indistinguishable to mean and variance tests
but separable via Fisher-Rao scalar curvature -- though for Gamma the
curvature is a function of the fitted shape alone, so this is no better
than the MLE skewness (see tests/test_gamma_reduction.py).

Restored from release 1.0.2, which introduced the batch-level detector.
These tests are what would have caught the regression to point-level
k-NN scoring: that detector produced AUC near 0.5 on this benchmark.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from igad import IGADDetector
from igad.families import GammaFamily


class TestIGADDetectorEndToEnd:

    def _run_hard_case(
        self,
        fit_mode: str,
        n_seeds: int = 5,
        batch_size: int = 200,
        n_per_class: int = 100,
    ):
        aucs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            detector = IGADDetector(family=GammaFamily())
            if fit_mode == "theta":
                detector.fit(theta_ref=GammaFamily.to_natural(8.0, 2.0))
            elif fit_mode == "data":
                X_train = rng.gamma(8.0, 0.5, size=5000)
                detector.fit(X=X_train)
            else:
                raise ValueError(fit_mode)
            scores, labels = [], []
            for _ in range(n_per_class):
                batch = rng.gamma(8.0, 0.5, size=batch_size)
                scores.append(detector.score_batch(batch))
                labels.append(0)
            for _ in range(n_per_class):
                batch = rng.lognormal(1.327, 0.343, size=batch_size)
                scores.append(detector.score_batch(batch))
                labels.append(1)
            aucs.append(roc_auc_score(labels, scores))
        return aucs

    def test_hard_case_with_theta_ref(self):
        aucs = self._run_hard_case(fit_mode="theta")
        mean, std = float(np.mean(aucs)), float(np.std(aucs))
        assert mean > 0.60, "Mean AUC %.4f < 0.60" % mean
        assert std < 0.06, "Std %.4f >= 0.06 (unstable)" % std

    def test_hard_case_with_large_training(self):
        aucs = self._run_hard_case(fit_mode="data")
        mean = float(np.mean(aucs))
        assert mean > 0.60, "Mean AUC %.4f < 0.60" % mean

    def test_fit_requires_exactly_one_argument(self):
        d = IGADDetector(family=GammaFamily())
        with pytest.raises(ValueError):
            d.fit()
        with pytest.raises(ValueError):
            d.fit(
                X=np.array([1.0, 2.0]),
                theta_ref=GammaFamily.to_natural(2.0, 1.0),
            )

    def test_score_batch_requires_fit(self):
        d = IGADDetector(family=GammaFamily())
        with pytest.raises(RuntimeError):
            d.score_batch(np.array([1.0, 2.0, 3.0]))

    def test_score_is_non_negative_and_sign_convention_independent(self):
        """`score_batch` returns |R_ref - R_local|, so it is unchanged by the
        overall sign convention of R (docs/proof.md section 4) -- while R_ref_
        itself is negative under the Levi-Civita convention."""
        d = IGADDetector(family=GammaFamily())
        d.fit(theta_ref=GammaFamily.to_natural(8.0, 2.0))
        assert d.R_ref_ < 0.0
        rng = np.random.default_rng(0)
        for _ in range(5):
            assert d.score_batch(rng.gamma(8.0, 0.5, size=200)) >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
