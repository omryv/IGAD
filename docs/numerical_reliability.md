# Numerical reliability of R(alpha)

Part 1.2 asked three things: prove that the dense and structured formulae are
the same number, decide whether the float64 disagreements are "conditioning",
and produce a documented reliability boundary.

Reproduce everything here:

```bash
python -m experiments.highprec_reliability   # 80 parameter points, 120 digits
```

Raw output: `experiments/results/highprec_reliability.json`.

Sections C1–C3 use 66 parameter points that hold `k <= 8`, so that expert count
cannot confound the rho-versus-cond(g) comparison - a concentration sweep, a
spread sweep that drives cond(g) to 1e28, and an anisotropy sweep that holds
both `k` and `alpha_0` fixed and varies only the shape of the concentration
profile. C5 adds a 14-point sweep that varies `k` from 3 to 256 and nothing
else. C4 and C6 check the bound and the digit estimate on all 80.

---

## 1. The reference

`experiments/_highprec.py` evaluates psi, psi' and psi'' with the standard
library's `decimal` at 160 working digits: upward recurrence to lift the
argument past 350, then Euler–Maclaurin with 50 exact Bernoulli terms
(computed as `Fraction`). The first omitted term of that asymptotic series is
below 1e-180.

The implementation is checked against **45 identities it does not use** -
the recurrence, Legendre duplication, the reflection formula, psi'(1) = pi²/6
and psi'(1/2) = pi²/2 (with pi from Machin), and self-consistency under raising
both the term count and the shift target. **Worst residual: 1.6e-157.**
(`validate_special_functions`, pinned by `tests/test_highprec.py`.)

Four independent curvature routes are then built on top: the literal six-index
contraction, a pairwise contraction of the same materialised tensor, the
structured `c + d_i delta_ijk` collapse, and the Sherman–Morrison form. The
first two know nothing about the tensor's structure.

---

## 2. Result C1 - dense and structured are the same number

| route | worst disagreement vs Sherman–Morrison, over 66 parameter points |
| --- | --- |
| literal six-index contraction | 5.2e-145 |
| pairwise dense contraction | 5.2e-145 |
| structured collapse | 2.7e-145 |
| reference recomputed at 200 digits | 5.2e-145 |

**They agree to 145 decimal places.** The structured closed form in
`igad.curvature.scalar_curvature_structured` is not an approximation of the
dense contraction, and neither is the O(k) Sherman–Morrison form. Every
disagreement observed in float64 is therefore purely numerical, and one of the
two float64 answers is simply wrong - a fact that no comparison *between* them
could establish.

---

## 3. Result C2 - where the float64 error actually comes from

Because every route accepts its polygamma inputs explicitly, the error splits
three ways and each part is measured:

| component | what it is | worst over 66 points |
| --- | --- | --- |
| input-rounding floor | exact evaluation of psi values already rounded to double - the floor no float64 pipeline can beat | 2.0e-09 |
| special functions | what the float64 psi'/psi'' add on top | 7.0e-09 |
| arithmetic | what each route's own operations add | 2.7e-01 |

### A correction to the previous report

`docs/validation_report.md` recorded four parameter points over the 1e-11
target and attributed the residual to "the shared, ill-conditioned `g^-1` and
the polygamma inputs". Half of that was right for the wrong reason, and the
diagnosis missed a larger, simpler problem.

The stdlib `trigamma`/`tetragamma` lifted their argument only to 12 and
truncated after three or four Bernoulli terms, leaving a relative error near
**1e-12** - roughly 4500 ulp. That propagated into `R(alpha)` as ~6e-11 on
*well-conditioned* points, including several the report listed as passing:

| case | cond(g) | error vs truth, old psi | error vs truth, corrected psi |
| --- | --- | --- | --- |
| symmetric k=3 | 1.2e+01 | 6.3e-11 | 9.4e-16 |
| asymmetric k=3 | 5.2e+01 | 6.5e-11 | 7.2e-15 |
| strongly asymmetric k=5 | 5.8e+04 | 4.6e-11 | 2.1e-14 |

This was invisible to the old check because **every curvature route consumes
the same psi values, so the shared error cancels exactly when routes are
compared to each other.** Structured-vs-dense agreed to 1e-14 while both were
6e-11 from the truth. That is the concrete reason the brief forbids arbitrating
between two float64 implementations with a third float64 implementation.

Raising the shift target to 30 and carrying six Bernoulli terms fixes it.
Measured against the 120-digit reference at 86 arguments spanning 1e-6 to
7e6 (`experiments/special_function_accuracy.py`): worst relative error
7.9e-16 for psi, 9.3e-16 for psi', 3.9e-16 for psi''. One documented
exception -- psi has a root at x ~ 1.4616321, where the recurrence sum and
the asymptotic tail cancel 93x; raw ulp error reaches 118 at x = 1.5 while
absolute error stays at 8.2e-16, and no recurrence-based psi avoids that.
psi is not on the curvature path.

---

## 4. Result C3 - it is cancellation, not conditioning

Define the **cancellation ratio**

```
rho = max( |S^2|, |T^2|, |term_a|, |term_b|, |term_c| ) / |T^2 - S^2|
```

That is, the largest intermediate magnitude divided by the final difference, where
`R = (T^2 - S^2)/4`. Regressing `log10(arithmetic error)` on each candidate
predictor across all 66 points:

| predictor | slope | intercept | R² | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| **log10 rho** | **0.998** | **−16.092** | **0.985** | 0.993 | 0.964 |
| log10 cond(g) | 0.476 | −14.574 | 0.657 | 0.810 | 0.793 |

The rho fit has **slope 1.00 and intercept −16.09**, and `log10(eps) = −15.95`
for binary64. So the fitted law is

```
relative error  =  eps * rho
```

with no free parameter - the intercept recovers machine epsilon to within 0.12
in the exponent.

Partial correlations settle the attribution:

- correlation of `log10 cond(g)` with the residual after rho: **+0.067** -
  once rho is known, cond(g) explains essentially nothing;
- correlation of `log10 rho` with the residual after cond(g): **+0.576** -
  once cond(g) is known, rho still explains a great deal.

Two controlled pairs make it concrete:

| | case | rho | cond(g) | measured error |
| --- | --- | --- | --- | --- |
| **rho held fixed** | `conc k=5 a=100000` | 6.3e+09 | 2.5e+05 | 8.7e-07 |
| | `spread k=5 m=5` | 4.8e+09 | 1.1e+20 | 9.5e-07 |
| **cond held fixed** | `strongly asymmetric k=5` | 5.3e+01 | 5.8e+04 | 2.1e-14 |
| | `conc k=3 a=10000` | 3.0e+08 | 3.0e+04 | 6.9e-08 |

Moving cond(g) by **fifteen orders of magnitude** at fixed rho changes the
error by 9%. Moving rho by six orders at fixed cond(g) changes the error by six
orders.

**The operational caveat in the previous report - "do not trust R beyond ~1e-11
relative when cond(g) >~ 1e3" - is withdrawn.** It is simultaneously too
strict (well-conditioned-by-rho points at cond(g) = 5.8e4 are accurate to
2e-14) and too permissive (`conc k=3 a=10000` has cond(g) = 3.0e4 and is only
accurate to 7e-8).

### Why cancellation and not conditioning

For a Dirichlet parameter the Sherman–Morrison denominator

```
w = 1 - psi'(alpha_0) * sum_i 1/psi'(alpha_i)
```

equals `det(g) / prod_i psi'(alpha_i)` and is strictly positive. For large
alpha it behaves like `(1/2a)(1 - 1/k)`, so `1/w` - which multiplies every
intermediate in the contraction - grows linearly with concentration. The
resulting terms are large and nearly cancel. Conditioning of `g` and this
cancellation are related but not the same quantity, and it is the cancellation
that the arithmetic sees.

---

## 5. Result C5 - the second factor: how much the contraction order costs

`eps * rho` accounts for the cancellation in the *expression*. It does not
account for the rounding that accumulates across the route's own operations.
Section C5 isolates that by sweeping k at a fixed `alpha_i = 2`, where rho
barely moves (13.0 down to 1.30):

| route | error / (eps · rho) at k=256 | fitted growth |
| --- | ---: | --- |
| dense-inverse | 13 481 | k^2.05 |
| sm-matrix | 13 573 | k^2.14 |
| **sm-closed** | **227** | **k^1.15** |

**The O(k) route is also the most accurate route**, by about a factor of k. It
touches O(k) intermediate values where the others touch O(k²) or run O(k³)
elimination steps, so it accumulates proportionally less rounding. That was not
the goal of the rewrite; it falls out of it.

The same effect, measured over the contraction orders in
`experiments/validate_curvature_implementation.py` (section A3, `alpha_i = 2`,
k up to 16):

| contraction | fitted growth | error / (eps · rho) at k=16 |
| --- | --- | ---: |
| literal six-index, O(k⁶) | **k^7.67** | 802 260 |
| pairwise, O(k⁴) | k^2.46 | 128 |
| structured collapse | k^0.30 | 18 |

The literal contraction sums k⁶ terms into a single accumulator, with heavy
cancellation among them. It is the least accurate route as well as the
slowest - a second, independent reason to prefer the closed form.

---

## 6. Result C4 - the boundary, and a guard rail a caller can evaluate

The bound

```
relative error  <=  8 * max( eps * rho * k^p,
                             input-rounding floor,
                             special-function error )
```

with `p = 1` for `sm-closed` and `p = 2` for the two matrix routes,
**held on 80 of 80 points** for all three routes (worst error/bound ratio
0.143, 0.086 and 0.086 respectively).

Surviving digits, banded by rho (each of the 80 points counted once):

| rho | points | worst relative error | significant digits of R that survive |
| --- | ---: | --- | --- |
| 1 – 10² | 48 | 3.0e-14 | ~13.5 |
| 10² – 10⁴ | 7 | 3.1e-13 | ~12.5 |
| 10⁴ – 10⁶ | 8 | 7.2e-11 | ~10.1 |
| 10⁶ – 10⁸ | 7 | 3.0e-09 | ~8.5 |
| 10⁸ – 10¹⁰ | 6 | 9.5e-07 | ~6.0 |
| 10¹⁰ – 10¹² | 2 | 1.4e-05 | ~4.8 |
| ≥ 10¹² | 2 | 2.7e-01 | ~0.6 |

The rule of thumb is exactly `16 - log10(rho)` significant digits. Past
rho ≈ 10¹⁵ there is nothing left: at `alpha = (1e-7, 1, 1e7)` the float64 value
of R is wrong in the first digit, and **no float64 implementation can do
better** - the input-rounding floor alone is 2.0e-09 there, because psi' and
psi'' cannot be represented exactly as doubles.

### rho is computable in float64, in O(k), alongside R

`dir_curvature_sm_closed_diagnostic(tri, tri0, tet, tet0)` returns
`(R, rho_hat)` from the same single pass. Measured against the exact
120-digit rho:

**`rho_hat` is within 2× on 80 of 80 points.**

So the guard rail is usable at runtime, not just in hindsight:

```python
from igad import curvature_reliability

info = curvature_reliability(tri, tri0, tet, tet0)
info["R"]                     # identical to scalar_curvature_dirichlet
info["cancellation_ratio"]    # rho
info["trusted_digits"]        # -log10(8 * eps * rho * k), clipped at 0
info["relative_error_bound"]
info["sm_denominator"]        # w; small exactly when the router is concentrated
```

It is **read-only**: the `R` it returns is bit-identical to the plain route,
and nothing about the computation changes because the diagnostic was asked
for. `tests/test_sherman_morrison.py` pins both that identity and the
conservatism below.

### Is the digit estimate calibrated? (C6)

A guard rail that over-promises is worse than none, so the estimate uses the
same safety factor 8 that section C4 validated. Measured against the digits
that actually survive, using the float64 `rho_hat` a caller would really have:

**Conservative on 80 of 80 points; worst over-promise 0.00 digits.** Margins
(actual minus promised): minimum 0.39, median 1.42, maximum 3.05 digits. A
large positive margin is the safe direction - the estimate is pessimistic by
about one to three digits, and never optimistic.

---

## 7. What this means for the tests

`tests/test_sherman_morrison.py` asserts accuracy against `64 * eps * rho * k^p`
rather than a flat tolerance. A flat 1e-11 target is not a property any
implementation can satisfy: at rho = 9.1e14 it is unreachable in float64, and
at rho = 2 it is 1e5 times weaker than what is actually achieved. Tolerances
that ignore rho either pass vacuously or fail on points where every possible
implementation fails.

---

## 8. Figures

Generated from the committed JSON by `python -m experiments.make_figures`:

| figure | shows |
| --- | --- |
| `figures/error_vs_cancellation.svg` | measured error against rho, with the `eps * rho` line - Spearman +0.960 |
| `figures/error_vs_condition.svg` | the same errors against cond(g) - the null result |
| `figures/error_decomposition.svg` | the three error components, ordered by rho: only the arithmetic term tracks it |
