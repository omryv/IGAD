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

## Reproducing

```bash
python -m experiments.demo_moe_router                      # defaults above
python -m experiments.demo_moe_router --experts 8 \
    --alpha-ref 0.5 --alpha-anom 0.6 --windows 8 16 32 64
```

**Provenance.** The tables above were produced by a standard-library mirror
of this script, because numpy could not be installed in the authoring
sandbox (PyPI and the Ubuntu archives both returned 403). The curvature
routine in that mirror was verified against the naive `O(k^6)` contraction
to a relative error of 1e-13 across eight parameter points. These numbers
have **not** been through a GitHub Actions run; re-running the committed
script is the confirmation step, and per the repository's stated policy this
counts as a new research iteration requiring its own validated run.
