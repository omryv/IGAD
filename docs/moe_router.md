# IGAD as an MoE Router Monitor

Sparse Mixture-of-Experts backbones route each token through a softmax over
`k` experts. Those router rows are simplex-valued, which is exactly
`DirichletFamily`'s domain — the one place in a diffusion-transformer stack
where IGAD's assumptions are satisfied rather than assumed.

This note records what happened when that idea was tested against the
detectors it would have to displace.

---

## Why the router and nothing else

| Pipeline stage | Applicable | Reason |
| --- | --- | --- |
| Image / text conditioning | Marginally | Positive-support stats via `GammaFamily` |
| Noised latents | **No** | Gaussian by construction |
| **Router weights** | **Yes** | Simplex-valued, `k >= 3`, non-constant `R` |
| Latent tokens | **No — provably** | Gaussian manifold, `R = -n(n+1)/2` is constant |
| Decoded 3D output | Yes | Positive-support mesh stats via `GammaFamily` |

Latent-token drift is the most tempting thing to monitor and the one IGAD
cannot see at all: `R` does not depend on the Gaussian parameters, so
`R_ref - R_local` is identically zero no matter how far the latents move
(`operational_envelope.md`, Failure Mode 1). Anything scalar — loss curves,
gradient norms, per-expert load as a single number — is a 1D family with
`R = 0` (Failure Mode 2).

---

## The experiment

`experiments/demo_moe_router.py`.

Every condition holds mean expert load at exactly `1/k`. An auxiliary
load-balancing loss is a first-moment constraint on that load, so it cannot
separate any two conditions here — the whole comparison lives in its blind
spot. What varies is `alpha_0`, the router's per-token decisiveness: small
`alpha_0` routes each token decisively, large `alpha_0` routes it mushily.

The swept axis is the monitoring **window**, because sample efficiency is
IGAD's actual claim. At windows of a few hundred tokens every method reaches
AUC 1.0 and the comparison is vacuous.

Four detectors, identical batches:

| Detector | Statistic | Cost |
| --- | --- | --- |
| `IGAD` | `\|R(alpha_hat) - R(alpha_ref)\|` | MLE + `O(k^2)` curvature |
| `MLE-a0` | `\|alpha_0_hat - alpha_0_ref\|` | MLE, curvature discarded |
| `Entropy` | `\|H_bar(batch) - H_bar(ref)\|` | `O(k)` per token, no MLE |
| `MMD` | RBF MMD² vs reference pool | `O(n^2 k)` |

`Entropy` is the incumbent — most MoE training stacks already log it.
`MLE-a0` is the control that decides the question, mirroring the
MLE-skewness control in `demo_hard.py`: it pays for the *same* Dirichlet fit
and then throws the curvature tensor away. Beating `Entropy` only shows that
fitting `k` sufficient statistics beats one scalar summary. Only beating
`MLE-a0` would vindicate the geometry.

---

## Results

Mean AUC over seeds `[11, 23, 42]`, 30 normal vs 30 anomalous batches per
cell, standard deviation in parentheses.

### k = 64 experts, `alpha_i` 0.50 → 0.52

| window | IGAD | MLE-a0 | Entropy | MMD |
| ---: | --- | --- | --- | --- |
| 8 | **0.6544** (0.054) | 0.6267 (0.077) | 0.6093 (0.058) | 0.4907 (0.090) |
| 16 | **0.7922** (0.038) | 0.7830 (0.052) | 0.6396 (0.147) | 0.4974 (0.051) |
| 32 | 0.8481 (0.080) | **0.8707** (0.064) | 0.6244 (0.185) | 0.5444 (0.048) |
| 64 | 0.8989 (0.065) | **0.9215** (0.042) | 0.5922 (0.168) | 0.4796 (0.028) |

### k = 8 experts, `alpha_i` 0.50 → 0.60

| window | IGAD | MLE-a0 | Entropy | MMD |
| ---: | --- | --- | --- | --- |
| 8 | 0.7715 (0.033) | **0.7911** (0.036) | 0.5930 (0.087) | 0.4830 (0.023) |
| 16 | 0.8237 (0.047) | **0.8270** (0.051) | 0.6222 (0.091) | 0.5119 (0.032) |
| 32 | 0.9267 (0.026) | **0.9385** (0.030) | 0.7789 (0.045) | 0.5385 (0.082) |
| 64 | 0.9952 (0.003) | **0.9981** (0.003) | 0.9048 (0.059) | 0.5707 (0.067) |

---

## The structural limitation

This is the part that matters most, and it is not an empirical finding that
better tuning could overturn.

IGAD's score is a deterministic function of the fitted parameters. Therefore

```
alpha_hat_A = alpha_hat_B   =>   R(alpha_hat_A) = R(alpha_hat_B)
```

Two populations that induce the same fitted Dirichlet are **exactly**
indistinguishable to scalar curvature — not weakly separated, not separated
with poor power, but identical in score. The same holds for every other
statistic of the form `f(alpha_hat)`, `MLE-a0` included.

That is a limitation of the *representation*, not of the comparison. The
Dirichlet MLE matches `E[log x]`, so it is enough for two populations to agree
on that one sufficient statistic for curvature to be blind to every other way
they differ. `experiments/demo_router_multimodal_control.py` builds exactly
such a pair — a four-mode mixture and its own best-fit single Dirichlet — and
measures the consequence: IGAD and `MLE-a0` agree to within 0.0075 AUC at every
window tested, because they are two functions of one number.

The practical reading: choosing curvature commits you to whatever the fit
preserves. Any structure the family discards is gone before the geometry is
evaluated, and no amount of geometric machinery downstream can recover it.

---

## Verdict

**The Dirichlet fit earns its place. The curvature tensor does not.**

1. **IGAD beats the incumbent decisively.** Against `Entropy` the margin is
   large and consistent — `+0.15` AUC at k=64/window=16, `+0.09` at
   k=8/window=64. Parametric monitoring of router distributions is worth
   doing.

2. **But the gain is the MLE, not the geometry.** Averaged over all eight
   cells, IGAD scores 0.839 and `MLE-a0` scores 0.845. IGAD leads in two of
   eight cells, both by less than one seed-standard-deviation; `MLE-a0` leads
   in the other six. Tracking the fitted concentration `alpha_0_hat` and
   discarding the curvature tensor entirely is as good or better, and it is
   `O(k)` after the MLE rather than `O(k^2)` with a matrix inverse.

   This is the predictable consequence of `R` being a near-monotone function
   of `alpha_0` for uniform `alpha` in this regime: `|dR|` and `|d alpha_0|`
   are close to the same statistic, and the cheaper one wins on variance.

3. **MMD is at chance throughout** (0.48–0.57), sample-starved at every
   window tested. Consistent with the repository's own claim that MMD needs
   `n` of 200–300 for comparable power.

The honest recommendation is a Dirichlet-MLE router monitor tracking
`alpha_0_hat`. Curvature is worth keeping as a secondary channel only
because `scalar_curvature_structured` made it nearly free — not because it
was shown to carry signal the concentration does not.

---

## Practical guards

- **Feed the pre-top-k softmax.** Post-top-k weights are mostly exact zeros.
  `DirichletFamily.mle` clips to `1e-15`, so `log x = -34.5` dominates the
  sufficient statistics and the fit describes the masking, not the routing.
- **Window must exceed expert count.** A `k`-parameter Dirichlet fit from
  `n < k` rows is underdetermined. With 64 experts the useful floor is well
  above the 50–300 batch range the rest of the repository validates.
- **Conditioning tracks confidence, not width.** `cond(g)` runs ~1.7 at
  `alpha_i = 0.5`, ~8.5 at `alpha_i = 4`, ~102 at `alpha_i = 50`, and is
  essentially independent of `k`. Wide expert counts are fine; very confident
  routers are the numerically awkward case. The determinant scale factor is
  `1 - psi'(alpha_0) * sum_i 1/psi'(alpha_i)`.
- **`R` saturates and is non-monotonic above `alpha_i ~ 2`.** For k=32,
  `alpha_0` from 64 to 512 — an eightfold change in decisiveness — maps into a
  ~3.5% band of `R`. Sparse routers (`alpha_i < 1`) sit in the steep,
  well-behaved region; mushy routers do not.

---

## Would a richer family rescue the geometry?

The obvious response to the structural limitation is to stop projecting through
a single Dirichlet and fit something that can represent directional covariance
— a logistic-normal model, i.e. a Gaussian on the identifiable log-ratio
coordinates `y_i = log(x_i / x_k)`.

That family does preserve more. But before benchmarking it, ask what its
geometry can contain. `experiments/demo_router_logistic_normal_geometry.py`
answers this numerically, and the answer closes the route:

1. **Scalar curvature is constant.** The affine group `y -> Ay + b` acts
   transitively on `(mu, Sigma)` and acts by isometries of the Fisher metric,
   so the manifold is homogeneous and every curvature invariant is
   parameter-independent. Measured with the repository's own formula,
   `R = d(d+1)^2/4` — depending only on the data dimension, never on the
   parameters. Residual scatter across parameter points scales as `h^2` with
   the finite-difference step (ratios 3.92, 3.95, 2.88 for a doubled step),
   which is truncation error rather than parameter dependence. A detector built
   on it would score identically zero on every input.

2. **The Fisher metric block-diagonalises**, recovered from the KL divergence
   to 1e-9: the mean block is `Sigma^-1` (Mahalanobis), the covariance block is
   `1/2 tr(S^-1 dS S^-1 dS)` (affine-invariant), and the cross block is exactly
   zero.

3. **Fisher-Rao distance is a cheap control in disguise.** Integrating metric
   length along the geodesic on the fixed-mean submanifold gives

   ```
   d_FR(S_ref, S_hat) = (1/sqrt 2) * ||log(S_ref^-1/2 S_hat S_ref^-1/2)||_F
   ```

   to a relative error of 4.2e-7. The right-hand side is the affine-invariant
   covariance distance — a standard fitted-parameter summary. AUC is invariant
   under monotone transforms, so the "information-geometric" statistic and the
   "cheap control" produce *identical* rankings and identical AUC by
   construction.

Together these say the geometry of this family decomposes entirely into a
Mahalanobis term on the mean and an affine-invariant term on the covariance —
both already standard statistics. There is no residual for geometry to occupy,
so the matched-control benchmark was not run: its outcome is fixed in advance.

---

## Reproducing

```bash
python -m experiments.demo_moe_router                      # numpy; defaults above
python -m experiments.demo_moe_router --experts 8 \
    --alpha-ref 0.5 --alpha-anom 0.6 --windows 8 16 32 64

# stdlib only, no dependencies; each writes JSON to experiments/results/
python -m experiments.demo_router_fixed_a0_anisotropy
python -m experiments.demo_router_multimodal_control
python -m experiments.demo_router_logistic_normal_geometry
```

The three `demo_router_*` scripts import only the standard library, so they run
in any environment and were executed to produce every number quoted above and
in `docs/router_geometry.html`. Raw output is written to
`experiments/results/*.json` before any figure is drawn.

`tests/test_router_common_mirror.py` cross-checks every mirrored Dirichlet,
special-function and linear-algebra routine in `experiments/_router_common.py`
against `igad` and scipy whenever numpy is installed, so the stdlib mirror
cannot drift from the package unnoticed.

**Provenance.** `demo_moe_router.py` is the one script here that requires numpy
and has therefore not been executed — numpy could not be installed in the
authoring environment (PyPI and the Ubuntu archives both returned 403). Its
numbers came from the stdlib mirror that the `demo_router_*` scripts now make
first-class. Nothing in this document has been through a GitHub Actions run;
per the repository's stated policy this is a new research iteration requiring
its own validated run.
