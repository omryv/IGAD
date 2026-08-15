# Handoff

The research question is:

> Does hidden MoE-router structure predict real 3D-generation degradation
> early — earlier or more accurately than expert load and routing entropy?

**It has not been tested, and nothing in this repository claims otherwise.**
The environment cannot load a model, so the work went into the two things that
*can* be finished without one: the mathematics and implementation the
experiment depends on, and the experiment itself, specified and scaffolded to
the point where only a checkpoint is missing.

Reproduce the whole status in one command:

```bash
python -m experiments.audit_environment        # exits non-zero: the gate fails
python -m experiments.report_test_status       # what ran, what could not, why
```

---

## What is proven

Everything in this section was established *in this environment*, by code that
executed here. Each line names the artifact that reproduces it.

### Exact complexity

**Dirichlet scalar curvature is O(k) in time and O(k) in memory, exactly.**
The Fisher metric is diagonal-plus-rank-one, `g = D − c·11ᵀ`, so
Sherman–Morrison gives `g⁻¹ = D⁻¹ + β·uuᵀ` in closed form. Substituting the
*factored* inverse into the structured contraction collapses every remaining
sum to a single pass over the experts; the one term that looks irreducibly
quadratic, `Σ_{i,a} d_i d_a (g⁻¹)_{ia}³`, factorises off the diagonal into the
square of a single sum. No k×k object is ever allocated. Derivation:
`docs/sherman_morrison.md`.

This is the preferred Dirichlet path:
`DirichletFamily.scalar_curvature_analytical` — the name every existing caller
including `IGADDetector` already uses — now dispatches to it.
`scalar_curvature_via_inverse` retains the O(k³) route as an independent
cross-check. The generic finite-difference route for other families is
untouched.

### Runtime scaling, measured

`experiments/results/sherman_morrison_benchmark.json`, standard-library Python,
minimum over repeats.

| route | predicted | measured slope (k ≥ 64) | memory slope | largest k reached |
| --- | --- | ---: | ---: | ---: |
| dense-inverse | O(k³) | 2.99 | 2.01 | 1 024 |
| sm-matrix | O(k²) | 2.12 | 2.03 | 2 048 |
| **sm-closed** | **O(k)** | **1.02** | **1.01** | **131 072** |

At k = 1024: **101.67 s → 0.34 ms**, a 299 000× speedup. Peak allocation at k = 512:
**20.9 MiB → 14.3 KiB**.

### Numerical accuracy

**The dense and structured formulae are the same number.** Four independent
high-precision routes — literal six-index contraction, pairwise dense
contraction, structured collapse, Sherman–Morrison — agree to **5.2e−145** at
160 working digits, and the reference does not move when recomputed at 200
digits. Every float64 disagreement is therefore numerical, not a difference of
formula. (`experiments/highprec_reliability.py`, section C1.)

**The high-precision oracle is validated against identities it does not use.**
ψ, ψ′ and ψ″ in `decimal` at 160 digits, checked against 45 identities —
recurrence, Legendre duplication, reflection, ψ′(1) = π²/6, ψ′(½) = π²/2, and
self-consistency under raising both the term count and the shift target.
**Worst residual 1.6e−157.** (`experiments/_highprec.py`,
`tests/test_highprec.py`.)

**The float64 special functions are fixed and bounded.**
`experiments/results/special_function_accuracy.json`, 86 arguments spanning
1e−6 to 7e6, against the 120-digit reference:

| function | worst abs | worst rel | worst adjusted ulp |
| --- | --- | --- | ---: |
| digamma | 4.00e−14 | 7.91e−16 | 4.96 |
| trigamma | 3.71e−05 | 9.27e−16 | 4.86 |
| tetragamma | 7.54e−17 | 3.89e−16 | 2.72 |

One documented exception: ψ has a root at x ≈ 1.4616321, where the recurrence
sum and the asymptotic tail cancel 185×. Raw ulp error there reaches 118 while
absolute error stays at 8.2e−16 — under 1 ulp of the pre-cancellation
magnitude, and no recurrence-based ψ can do better without a
root-centred expansion. ψ is not on the curvature path; it enters only
`inv_digamma` and the Dirichlet MLE gate, whose tolerance is 1e−4.

Before this pass the stdlib ψ′/ψ″ were accurate to only ~1e−12 (≈4500 ulp).
Because every curvature route consumes the same values, that error cancelled
*exactly* in route-versus-route comparisons: structured and dense agreed to
1e−14 while both sat 6.3e−11 from the truth.

### The error law, and the runtime diagnostic

**Cancellation governs the error; conditioning does not.** Regressing
log₁₀(error) over 66 parameter points with k ≤ 8 (so k cannot confound):

| predictor | slope | intercept | R² |
| --- | ---: | ---: | ---: |
| **log₁₀ ρ** (cancellation ratio) | **1.00** | **−16.09** | **0.985** |
| log₁₀ cond(g) | 0.48 | −14.57 | 0.657 |

The intercept recovers log₁₀(eps) = −15.95, so the law is `error = eps · ρ`
with no free parameter. Partial correlations: once ρ is known, cond(g) explains
+0.08 of the residual; once cond(g) is known, ρ still explains +0.57. Two
controlled pairs make it concrete — holding ρ fixed while moving cond(g) by
**fifteen orders of magnitude** changes the error by 9%; holding cond(g) fixed
while moving ρ by six orders moves the error by six orders. Sweeps cover k, α₀,
anisotropy and conditioning separately.

**Size is a second, weaker factor.** In units of eps·ρ, error grows as k^1.15
for the closed form against k^2.05–2.14 for the routes that build a k×k matrix,
and k^7.67 for the literal six-index contraction. The O(k) route is therefore
also the *most accurate* route — 60× better than dense-inverse at k = 256.

**The bound holds and the diagnostic is conservative.**
`8 · max(eps·ρ·kᵖ, input-rounding floor, special-function error)` bounded the
measured error on **80 of 80** points for all three routes. The float64 ρ̂ is
within 2× of the exact ρ on **80 of 80**. `curvature_reliability(...)` reports
ρ, the surviving-digit estimate and the Sherman–Morrison denominator from the
same O(k) pass; it is **read-only** — it returns a value bit-identical to the
plain route and never modifies R. Its digit estimate was **conservative on 80
of 80 points**, worst over-promise 0.00 digits, minimum margin 0.39 digits.

The earlier operational caveat — *"do not trust R beyond 1e−11 relative when
cond(g) ≳ 1e3"* — **is withdrawn**, and every place it appeared in the
repository has been corrected or struck through with the reason.

### Structural mathematical results that narrow future work

- **Fisher–Rao distance and the affine-invariant covariance distance are one
  detector.** On the fixed-mean covariance manifold `d_FR = (1/√2)·d_AI`,
  verified to 4.2e−7. Identical rankings, identical AUC. They are entered once,
  not as competitors.
- **Under expert-choice routing the load baseline is degenerate** — uniform by
  construction. `trace_schema` records `routing_mode` so a report cannot quote
  a win over it without saying so.
- **Scalar curvature is not a detector candidate.** It lost to `affine_invariant`
  at every window in the previous pass and is close to a relabelling of
  concentration (ρ(R, α₀) = 0.841). It remains a validated numerical routine,
  now O(k). No attempt was made to rescue it.

### Tests actually executed

`python -m experiments.report_test_status` →
`experiments/results/test_status.json`. Three exclusive buckets:

- **verified locally** — executed here and passed;
- **skipped, no dependency** — could not run because a package is absent;
- **not collected** — the module could not be imported at all.

Only the first bucket may be described as verified. The counts are in the JSON
rather than transcribed here, so they cannot go stale.

### Contracts that reject bad captures

Both are tested against fixtures whose correct answers are known; neither
contains or generates data.

- `experiments/trace_schema.py` — router traces. Rejects a **post-top-k**
  capture (exactly `k − top_k` hard zeros is the signature; softmax output has
  none), a manifest without `pre_topk_verified`, and a dataset with fewer than
  two objects.
- `experiments/quality_schema.py` — 3D quality. Rejects a failure label whose
  provenance is **router-derived**, a manifest whose thresholds were not
  registered before the router analysis, a reference-based metric with no
  reference asset, and a record carrying only labels and no measurement.

### Analysis code ready for real traces

`experiments/router_stats.py` and `experiments/evaluation.py`: the cheap
baselines of brief §6, the structure-aware statistics of §7, ROC/PR AUC,
sensitivity at fixed FPR with an achievable-FPR grid, correlations,
cross-validated R², earliest-warning time with the "and stays there" rule, and
the object-level protocol of §3 — bootstrap CI, train/test split, k-fold,
paired detector test, all resampling **objects**, with group-aware splitting so
that two seeds of one conditioning image cannot straddle a split.

Every function consumes real traces. None generates any. The unit tests use
hand-built fixtures with analytically known answers, and no number from them
says anything about a real router.

---

## What requires external resources

Nothing below can be produced here, and no substitute for it was published.

| blocked | why | acceptance test |
| --- | --- | --- |
| **Tensor runtime** | no `torch`/`jax`/`tensorflow`; `pypi.org` returns 403 from the egress proxy | `audit_environment` reports condition 1 met |
| **Accelerator** | no `nvidia-smi`, no `/dev/nvidia*`, no CUDA-capable runtime | `torch.cuda.is_available()` |
| **A real MoE router** | no checkpoint on disk (91 weight-suffixed files, 0 over 10 MiB — package-manager caches); `huggingface.co` returns 403 | `trace_schema.verify_pre_topk_capture` passes on real hook output |
| **A 3D generation pipeline** | there is no model that is both 3D and token-level MoE with released weights. TRELLIS.2-4B is 3D but **dense**; Nucleus-Image is real MoE but **2D**. Neither, alone or together, tests the hypothesis. | a generator that is both, or an upcycled one |
| **Real quality labels** | no mesh library (`trimesh`, `open3d`, `pytorch3d`, …); 0 of 20 mesh-suffixed files on disk survive a header check | `quality_schema.validate_file` passes on measured records |
| **NumPy-dependent verification** | `numpy`/`scipy` unavailable, so the `igad` package code paths and 3 test modules **have never executed**. The stdlib mirror is pinned to them by `tests/test_router_common_mirror.py` — but only where numpy is present. | `report_test_status` shows an empty "not collected" bucket |
| **A working CI runner** | every GitHub Actions run on this repository, on `main` as well as on branches, ends in 1–3 seconds with `runner_id: 0` and no executed steps. That is **CI infrastructure unavailable**, not a code test failure — it predates this work and is unaffected by it. | a run that reaches the "Run tests" step |

The two scripts deliberately not written are `capture_router_traces.py` and
`evaluate_3d_quality.py` — the only two that must call a model and a mesh
library. Everything they would call is built and tested; what is missing is the
binding to a specific checkpoint's module names, which cannot be written
correctly against a runtime that will not import.

Until real router traces and real 3D quality exist, the product verdict is
**Untested** — not "No", and not a number.

### The single next external action

> **Provide one runnable MoE checkpoint that exposes pre-top-k routing
> probabilities and generates 3D outputs, together with a corresponding real
> 3D quality target. Then execute the prepared early-warning pipeline in
> `docs/experiment_plan.md`.**

If no model is both, the fallback in order of preference is: upcycle a dense 3D
generator to MoE (`docs/acquisition_checklist.md`, Path B — the recipe and its
documented failure modes are cited there), or fine-tune an open MoE language
backbone on mesh tokens (Path C). Using Nucleus-Image to rehearse the
instrumentation, or TRELLIS.2 to rehearse the quality tooling, is worth doing
first — but a result from either is a result about that model, not about the
hypothesis.
