# Dirichlet scalar curvature in O(k)

Part 1.1 of the brief asked for a genuinely O(k²) end-to-end curvature path,
by exploiting the Dirichlet Fisher structure through a Sherman–Morrison
inverse. The structure gives more than that: once the factored inverse is
substituted into the contraction, **no k × k object needs to exist at all**,
and the complete path is O(k) in both time and memory.

This document derives that, and records what was measured.

---

## 1. The two structures

For `Dirichlet(alpha)` in natural parameters `theta_i = alpha_i - 1`, with
`A(theta) = sum_i lgamma(alpha_i) - lgamma(alpha_0)` and `alpha_0 = sum_i alpha_i`:

**Fisher metric** — diagonal plus rank one:

```
g = D - c_g 1 1^T ,      D_ii = psi'(alpha_i) ,   c_g = psi'(alpha_0)
```

**Third cumulant tensor** — constant plus triple diagonal:

```
T_ijk = c + d_i delta_ijk ,   c = -psi''(alpha_0) ,   d_i = psi''(alpha_i)
```

`igad.curvature.scalar_curvature_structured` already exploits the second
structure, collapsing the O(k⁶) contraction

```
R = 1/4 ( ||S||^2_g - ||T||^2_g ) ,     S_m = g^{ab} T_{abm}
```

to O(k²) *given* `M = g^{-1}`. The first structure was not exploited, so `M`
came from a generic inverse and the complete path stayed O(k³). This is the
drift `docs/validation_report.md` reported at large k.

---

## 2. Sherman–Morrison

With `u = D^{-1} 1` (so `u_i = 1 / psi'(alpha_i)`), `U = 1^T u`, and

```
w = 1 - c_g U ,        beta = c_g / w
```

the Sherman–Morrison identity gives

```
g^{-1} = D^{-1} + [ c_g / (1 - c_g 1^T D^{-1} 1) ] D^{-1} 1 1^T D^{-1}
       = D^{-1} + beta u u^T                                            (1)
```

i.e. entrywise `M_ia = delta_ia u_i + beta u_i u_a`.

**`w` is never zero for a valid parameter.** `det(g) = (prod_i D_ii) · w`, and
`g` is the Fisher metric of a minimal exponential family with non-empty
interior, hence positive definite; so `w = det(g) / prod_i psi'(alpha_i) > 0`.
A zero or negative `w` at runtime means the inputs are not a
`(psi'(alpha), psi'(alpha_0))` pair, and both implementations raise rather than
return a number.

Building the matrix in (1) is O(k²) — optimal for an explicit inverse, since
the result has k² entries. That alone answers Part 1.1: `sm-matrix` below is
O(k²) time and memory end to end. The rest of this section removes the matrix.

---

## 3. Collapsing every contraction

Three quantities enter the structured contraction. Each one factorises.

**Column sums.** `r_a = sum_i M_ia = u_a + beta u_a U = u_a (1 + beta U)`, and

```
1 + beta U = 1 + c_g U / w = (w + c_g U) / w = 1 / w
```

since `w = 1 - c_g U`. So

```
r_a = u_a / w ,          s = sum_a r_a = U / w                          (2)
```

**Diagonal.** `M_mm = u_m + beta u_m^2 = u_m (1 + beta u_m)`, hence

```
S_m = c s + u_m (1 + beta u_m) d_m                                      (3)
```

**Quadratic form.** `||S||^2_g = S^T M S = S^T D^{-1} S + beta (u^T S)^2`:

```
||S||^2_g = sum_m S_m^2 u_m + beta ( sum_m S_m u_m )^2                  (4)
```

**Cubed-entry double sum.** The one term that looks irreducibly O(k²) is
`sum_{i,a} d_i d_a M_ia^3`. Split it by whether `i = a`. Off the diagonal
`M_ia = beta u_i u_a` factorises, so the double sum becomes the square of a
single sum:

```
sum_{i != a} d_i d_a beta^3 u_i^3 u_a^3
    = beta^3 [ ( sum_i d_i u_i^3 )^2 - sum_i ( d_i u_i^3 )^2 ]
```

and on the diagonal `M_ii^3 = u_i^3 (1 + beta u_i)^3`, so

```
sum_{i,a} d_i d_a M_ia^3
    = beta^3 [ ( sum_i d_i u_i^3 )^2 - sum_i ( d_i u_i^3 )^2 ]
      + sum_i d_i^2 u_i^3 (1 + beta u_i)^3                              (5)
```

Together with `sum_a d_a r_a^3 = (1/w^3) sum_a d_a u_a^3` from (2):

```
||T||^2_g = c^2 s^3
            + (2c / w^3) sum_a d_a u_a^3
            + beta^3 [ ( sum_i d_i u_i^3 )^2 - sum_i ( d_i u_i^3 )^2 ]
            + sum_i d_i^2 u_i^3 (1 + beta u_i)^3                        (6)
```

Every sum in (2)–(6) is a single pass over the k experts. **The complete
curvature path is O(k) time and O(k) memory**, and exact — no approximation
enters at any step.

---

## 4. Implementations

| name | inverse | contraction | time | memory |
| --- | --- | --- | --- | --- |
| `dense-inverse` | Gauss-Jordan / `np.linalg.inv` | structured | O(k³) | O(k²) |
| `sm-matrix` | equation (1), materialised | structured | O(k²) | O(k²) |
| `sm-closed` | equation (1), factored | equations (2)–(6) | **O(k)** | **O(k)** |

- package (numpy): `igad.curvature.dirichlet_fisher_inverse`,
  `igad.curvature.scalar_curvature_dirichlet`,
  `igad.families.DirichletFamily.scalar_curvature_fast`
- stdlib mirror: `experiments._router_common.dir_fisher_inverse_sm`,
  `dir_curvature_sm_matrix`, `dir_curvature_sm_closed`
- high precision: `experiments._highprec.hp_curvature_sherman_morrison`

`tests/test_sherman_morrison.py` pins all three against each other and against
the 120-digit reference; `tests/test_router_common_mirror.py` pins the stdlib
mirror to the package whenever numpy is installed.

---

## 5. Measured scaling

`python -m experiments.benchmark_sherman_morrison` →
`experiments/results/sherman_morrison_benchmark.json`.

See `RESULTS.md` for the measured table. The numbers are standard-library
Python: absolute timings under numpy are far smaller, but the fitted log-log
slope and the memory growth are what the claim rests on, and those are
implementation-independent.

---

## 6. Numerical caveat

`sm-closed` is not more accurate than `dense-inverse`; it is faster.
Both evaluate the same expression, and that expression cancels — see
`docs/numerical_reliability.md`. The Sherman–Morrison route makes one
cancellation explicit that elimination hides: `w = 1 - c_g U` is computed as a
difference of two nearly equal positive numbers whenever `alpha_0` is large,
and `1/w` is exactly the factor by which the intermediates in (2)–(6) are
inflated. That is a feature for diagnosis — `w` is available in O(k) and tells
the caller how many digits of `R` survive — not a defect introduced by the
rearrangement.
