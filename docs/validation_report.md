# Validation Report - Router Structure, Distributional Modelling, Scalar Curvature

Three claims were treated as hypotheses to falsify, not conclusions to defend.

---

> ## Superseded in Part A
>
> **Two conclusions in this report's Part A are withdrawn.** A later pass built
> a 120-digit reference (`experiments/_highprec.py`) and measured what Part A
> could only infer from comparing two float64 routes to each other.
>
> 1. **"The residual sits in the shared, ill-conditioned `g⁻¹`."** Withdrawn.
>    The dominant error was the standard-library `trigamma`/`tetragamma`, which
>    were accurate to only ~1e-12. Because every curvature route consumes the
>    same values, that error cancelled exactly in the route-versus-route
>    comparison this report used, and stayed invisible: several cases listed
>    below as passing at 1e-14 were in fact 6e-11 from the truth. Corrected,
>    they are accurate to ~1e-15.
> 2. **"Do not trust R beyond ~1e-11 relative when cond(g) ≳ 1e3."** Withdrawn.
>    Conditioning of `g` does not order the error. The cancellation ratio ρ does,
>    with R² = 0.986 and slope 1.00 against R² = 0.673 for cond(g); holding ρ
>    fixed while moving cond(g) by fifteen orders of magnitude changes the error
>    by 9%. The replacement rule is `error ≈ eps · ρ · kᵖ`.
>
> Part A's complexity table is also superseded: the O(k³) matrix inverse it
> identified as the bottleneck has been removed, and the complete path is now
> O(k) in time and memory.
>
> See `docs/numerical_reliability.md` and `docs/sherman_morrison.md`. The rest
> of this report - Parts B, C, D, F, and the scope limit - stands as written.

---

## Scope limit, stated before any result

Part E of the brief - *does router structure predict 3D-generation degradation
earlier than ordinary MoE diagnostics* - **was not executed**. This environment
has no MoE checkpoint, no 3D generator, no mesh tooling, and no network to
obtain them: `torch`, `numpy`, `scipy`, `trimesh` and `open3d` are all absent,
and PyPI and the Ubuntu archives both return 403. There are no `.pt`,
`.safetensors`, `.obj` or `.ply` files anywhere in the repository.

That scope limit was later re-checked by machine rather than by hand:
`python -m experiments.audit_environment` evaluates all five gate conditions,
verifies file contents rather than trusting suffixes, and exits non-zero. It
confirms this paragraph. What is missing, and how to acquire it, is written up
in `docs/acquisition_checklist.md`.

Every router vector below is **synthetic**. That supports a *possibility*
result - structure surviving cheap controls can exist and can be measured - and
supports nothing whatsoever about real routers or real 3D quality. Per the
brief's own rule (*real 3D quality before synthetic AUC*), the product verdict
is recorded as **Untested**, not inferred from synthetic AUC.

---

## Part A - is the O(k²) implementation correct?

### A1 exactness

Structured vs dense contraction, 13 parameter points
(`experiments/validate_curvature_implementation.py`):

| regime | cases | worst relative error | verdict |
| --- | --- | --- | --- |
| symmetric, asymmetric, low concentration | 9 | 2.0e−13 | pass |
| high concentration (α ≈ 200–500) | 2 | 1.9e−11 | over target |
| ill-conditioned (cond g ≈ 1e13–1e14) | 2 | 9.8e−09 | over target |

Nine of thirteen meet the 1e−11 target, including every well-conditioned case.
The four misses were investigated rather than excused:

- **Summation order ruled out.** Re-running the dense route with `math.fsum`
  (exactly rounded) changes the disagreement by less than a factor of 1.2. It
  is not k⁶ accumulation error.
- ~~**Catastrophic cancellation ruled out.** `‖S‖²_g` and `‖T‖²_g` differ by
  enough that only 0.3–0.9 decimal digits are lost in the subtraction.~~
  **Withdrawn.** This measured only the *final* subtraction. The larger
  cancellation happens one level down, among the three terms that build
  `‖T‖²_g`: at α = (200, 200, 200) they are +3.6e5, −7.2e5 and +3.6e5,
  summing to −190. Counting every intermediate gives a cancellation ratio
  ρ = 1.2e5 for that point, and ρ explains the error with R² = 0.986.

~~The residual therefore sits in the shared, ill-conditioned `g⁻¹` and the
polygamma inputs, which the two contraction orders amplify differently.
Neither route is "the truth" in float64 at cond(g) ≈ 1e14. **Operational
caveat: do not trust R beyond ~1e−11 relative when cond(g) ≳ 1e3, whichever
route computes it.**~~

**Both sentences are withdrawn**; see the banner at the top of this report.
The table above is also stale, because it measured the routes against each
other rather than against the truth, and because the special functions have
since been corrected. The current numbers are in `docs/numerical_reliability.md`
and `experiments/results/highprec_reliability.json`.

### A2 complexity - measured, not extrapolated

Wall clock for one evaluation; a route is abandoned once it exceeds 20 s.

| k | dense-naive O(k⁶) | dense-pairwise O(k⁴) | structured O(k²) |
| ---: | --- | --- | --- |
| 3 | 0.0002 s | 0.0001 s | 0.0000 s |
| 8 | 0.0272 s | 0.0020 s | 0.0002 s |
| 16 | 1.5849 s | 0.0209 s | 0.0007 s |
| 32 | 96.02 s | 0.2808 s | 0.0043 s |
| 64 | skipped | 4.4675 s | 0.0314 s |
| 128 | skipped | 76.28 s | 0.2598 s |
| 256 | skipped | skipped | **2.0367 s** |

Measured scaling per doubling: dense-naive 58–61× (k⁶ predicts 64×),
dense-pairwise 10–17× (k⁴ predicts 16×), structured 4–8×. **The structured
route's own scaling drifts above k² at large k because the O(k³) matrix
inverse, not the contraction, dominates there** - worth stating precisely
rather than claiming a clean k².

**Engineering claim validated.** Realistic expert counts became practical:
k=64 goes from infeasible (naive) / 4.47 s (pairwise) to 0.031 s, and k=256 is
reachable at all for the first time. Agreement across surviving routes stays
between 1e−14 and 1e−7.

**Superseded.** The bottleneck this paragraph correctly identified - the O(k³)
matrix inverse - has since been removed. The Dirichlet Fisher metric is
diagonal-plus-rank-one, so Sherman–Morrison gives `g⁻¹` in closed form, and
substituting the factored inverse into the contraction makes the complete path
O(k) in both time and memory. k=1024 goes from 105 s to 0.33 ms; k=131 072 is
now reachable. See `docs/sherman_morrison.md` and the tables in `RESULTS.md`.

The accumulation ordering also turned out to matter for accuracy, not only
speed: in units of eps·ρ the literal six-index route grows as k^7.67, the
pairwise route as k^2.46, and the closed form as k^0.30
(`validate_curvature_implementation.py`, section A3).

---

## Parts B, C, D - matched-control experiment

`experiments/demo_router_matched_control.py`. k=8 experts, logits
`z ~ N(mu, Sigma)`, `p = softmax(z)`. Condition A isotropic in the gauge-free
subspace; condition B anisotropic, with `mu` tuned until `E[p] = 1/k` and the
covariance scale tuned until mean entropy matches.

### Residual mismatch (reported, not assumed)

| statistic | A | B | \|diff\| | rel |
| --- | --- | --- | --- | --- |
| max_i \|E_A p_i − E_B p_i\| | 0.125000 | - | 1.55e−03 | 1.24% |
| mean entropy | 1.318145 | 1.323513 | 5.37e−03 | 0.41% |
| E[max_i p_i] | 0.519621 | 0.509366 | 1.03e−02 | 1.97% |
| E[‖p‖²₂] | 0.379103 | 0.381468 | 2.37e−03 | 0.62% |
| tr Cov(p) | 0.254100 | 0.256473 | 2.37e−03 | 0.93% |

All within 2%. Log-ratio covariance eigenvalues - A:
`23.99 3.07 3.04 3.02 2.98 2.97 2.92`; B: `20.02 8.84 4.11 1.82 1.31 0.82 0.37`.
Same trace, very different spectrum.

### Detection - mean AUC ± sd over 5 seeds, 30 vs 30 batches

| win | MeanLoad | Entropy | MaxProb | L2Mass | TraceCov | LambdaMax | SpectralEnt | **AffineInv** | MMD | Dirichlet-α₀ | IGAD-R |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.502±.067 | 0.619±.102 | 0.622±.091 | 0.606±.101 | 0.626±.103 | 0.521±.120 | 0.721±.117 | **1.000±.000** | 0.734±.066 | 0.672±.098 | 0.784±.051 |
| 64 | 0.446±.057 | 0.618±.048 | 0.659±.060 | 0.630±.057 | 0.639±.071 | 0.593±.153 | 0.721±.068 | **1.000±.000** | 0.757±.071 | 0.704±.106 | 0.840±.072 |
| 128 | 0.437±.042 | 0.557±.066 | 0.621±.078 | 0.531±.075 | 0.530±.073 | 0.691±.233 | 0.766±.125 | **1.000±.000** | 0.823±.034 | 0.744±.076 | 0.952±.029 |
| 256 | 0.434±.080 | 0.539±.056 | 0.658±.038 | 0.526±.050 | 0.530±.047 | 0.810±.247 | 0.862±.126 | **1.000±.000** | 0.738±.088 | 0.776±.039 | 0.995±.006 |

`MaxProb`'s 0.62–0.66 is not evidence of structure - it tracks the 1.97%
residual mismatch on exactly that statistic. Read it as a matching artefact.

### D1 - is curvature a relabelling of concentration?

Spearman `rho(R(alpha_hat), alpha_hat_0) = 0.841 ± 0.055` over 20 cells.
Strong monotone dependence, though not the ≈1 seen in the symmetric subfamily.
Curvature is not *only* concentration here - and it still loses to `AffineInv`
at every window.

### F controls (window 256)

| detector | baseline | F5 labels permuted | F1 experts shuffled |
| --- | --- | --- | --- |
| MeanLoad | 0.434 | 0.506 | 0.477 |
| Entropy | 0.539 | 0.580 | 0.588 |
| SpectralEnt | 0.862 | 0.560 | 0.865 |
| AffineInv | 1.000 | 0.525 | 1.000 |
| IGAD-R | 0.995 | 0.541 | 0.991 |

F5 collapses every detector to near chance - no label leakage. F1 leaves the
permutation-invariant detectors unchanged (AffineInv 1.000→1.000, SpectralEnt
0.862→0.865, IGAD-R 0.995→0.991), as required.

---

## The sharpest finding

The winning detector, `AffineInv`, **is the Fisher-Rao geodesic distance** -
`d_FR = (1/√2)·‖log(Σ_ref^{−1/2} Σ̂ Σ_ref^{−1/2})‖_F`, verified to 4.2e−7 in
`demo_router_logistic_normal_geometry.py`. The best-performing statistic in
this study is an information-geometric one.

And that is exactly why geometry earns no operational credit: because
`d_FR` is a *constant multiple* of the ordinary affine-invariant covariance
distance, the two are the same detector with the same ranking and the same AUC.
The geometry supplies a good **interpretation** of why that statistic is the
natural one. It supplies no **operational advantage**, because a practitioner
who never heard of Fisher-Rao computes the identical number.

---

## Decision table

| Question | Evidence | Verdict |
| --- | --- | --- |
| Is there router structure beyond load/entropy? | MeanLoad 0.434–0.502, Entropy 0.539–0.619, AffineInv 1.000±0.000, all under ≤2% mismatch | **Yes** (synthetic only) |
| Can richer distributional modelling capture it? | log-ratio covariance: AffineInv 1.000 at every window; single Dirichlet reaches only α̂₀ 0.672–0.776 | **Yes** (synthetic only) |
| Does scalar curvature add information? | IGAD-R 0.784–0.995 never exceeds AffineInv 1.000; ρ(R, α̂₀)=0.841 | **No** |
| Does covariance structure predict 3D quality? | no checkpoint, no 3D pipeline, no network | **Untested** |
| Does it predict quality earlier than entropy/load? | AffineInv 1.000 at n=32 where entropy never exceeds 0.62 at any n - but on synthetic data | **Yes** (synthetic) / **Untested** (3D) |
| Is geometry better than cheap same-fit controls? | the best detector *is* d_FR, and d_FR = (1/√2)·affine-invariant - identical rankings | **No** |

No prose above contradicts this table.

---

## Verdict on the three claims

- **Claim 1 - survives.** Mean load sits at chance (0.434–0.502) and entropy
  near chance (0.539–0.619) while a structure-aware statistic reaches 1.000.
  This meets the stated acceptance criterion. Scope: synthetic routers.
- **Claim 2 - survives.** The log-ratio covariance retains what the single
  Dirichlet destroys, and it does not merely track entropy or total variance
  (both of which are matched to under 1%). Scope: synthetic routers.
- **Claim 3 - fails.** Curvature never beats a control derived from the same
  or cheaper fit, at any window, across five seeds.

**Hypothesis selected: H₃** - a richer statistical model is useful, scalar
curvature adds nothing. H₁ is rejected (structure exists beyond cheap
statistics). H₄ is rejected (geometry does not beat same-fit controls).

## Product verdict, kept separate

**Untested.** The operational question - whether router structure gives earlier
warning of 3D-generation degradation than cheap MoE diagnostics - requires a
checkpoint and a 3D quality pipeline that do not exist here. The synthetic
early-detection pattern (AffineInv saturating at n=32 where entropy never
resolves) is the pattern the brief describes as a product win, but it is a
pattern in data I generated, and it is not evidence about a real generator.

The honest next step is not more synthetic work. It is one MoE checkpoint, the
pre-top-k softmax logged per layer and timestep, and any real 3D quality label.

---

## Scientifically interesting vs operationally useful

- **Scientifically interesting:** Fisher-Rao geometry identifies the right
  statistic for covariance drift, and explains *why* the affine-invariant
  distance is the natural choice rather than an ad-hoc one.
- **Operationally useful:** the affine-invariant covariance distance -
  computable without any geometric machinery, and identical to the geometric
  answer up to 1/√2.

Scalar curvature is in neither column for this problem.

---

## Reproducing

```bash
python -m experiments.validate_curvature_implementation   # Part A
python -m experiments.demo_router_matched_control         # Parts B, C, D1, F
python -m experiments.demo_router_logistic_normal_geometry
python -m experiments.demo_router_fixed_a0_anisotropy
python -m experiments.demo_router_multimodal_control
```

Standard library only; explicit seeds throughout; raw results written to
`experiments/results/*.json` before any figure.

**Not delivered:** plots generated from the saved JSON. The raw results are
committed and sufficient to produce them, but the figures themselves were not
built in this pass and should not be assumed to exist.

**Statistical caveat:** uncertainty is over 5 generation seeds with 30 vs 30
batches per cell. The brief asks for object-level resampling to avoid
token-level pseudo-replication; with synthetic data each batch is an
independent draw, so that hazard does not arise here - but it will the moment
real router traces are used, where tokens within one generated object are not
independent.
