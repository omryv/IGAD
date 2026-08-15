# Experiment Figures

---

## exp1_easy_gamma_vs_gamma.png

**Experiment 1 — Easy Case**
Reference: Gamma(9,3) | Anomaly: Gamma(1.5,0.5)
- Same mean (3.0), different variance (1.0 vs 6.0) and skewness (0.667 vs 1.633)

Left panel:  IGAD score distribution — normal vs anomaly batches (AUC=1.000)
Right panel: Raw skewness shift distribution (AUC=0.983)

Conclusion: Both IGAD and variance baseline achieve perfect separation.
This experiment shows IGAD works but does not prove unique value,
because the distributions also differ in variance.

---

## exp2_hard_gamma_vs_lognormal.png

**Experiment 2 — Hard Case (the key result)**
Reference: Gamma(8,2) | Anomaly: LogNormal(mu=1.327, sigma=0.343)
- Identical mean (4.0) AND variance (2.0). Only skewness differs (0.707 vs 1.105)

Left panel:   IGAD score — |R_ref - R_local| (AUC=0.684)
Middle panel: MLE skewness CONTROL — same MLE, no geometry (AUC=0.610)
Right panel:  Raw skewness shift (AUC=0.651)

Key finding: IGAD beats the MLE-skewness control by +0.053 (mean over 5 seeds).
Since both use identical MLE fits, the gap is attributable to the curvature
tensor geometry — not MLE statistical efficiency alone.

Mean AUC over 5 seeds (n=200):
  IGAD         : 0.6542 ± 0.047
  MLE skewness : 0.6016 ± 0.038
  Gap          : +0.053

---

## scaling_wallclock.svg

**Part 1.1 — complete Dirichlet curvature path, measured wall clock**

Three routes to the same number, standard-library Python, minimum over repeats:
`dense-inverse` (generic O(k^3) matrix inverse + structured contraction),
`sm-matrix` (Sherman-Morrison inverse materialised, O(k^2)), and `sm-closed`
(factored inverse, nothing k x k ever allocated, O(k)).

Fitted log-log slopes on k >= 64: 2.99, 2.10, 1.03 against predicted 3, 2, 1.
`dense-inverse` reaches k=1024 at 105 s; `sm-closed` reaches k=131 072 at
0.053 s.

Source: `experiments/results/sherman_morrison_benchmark.json`.

---

## scaling_memory.svg

**Part 1.1 — peak allocation per evaluation (tracemalloc)**

Same three routes. Memory slopes 2.01, 2.03, 1.01. At k=512 the peak drops
from 20.9 MiB (`dense-inverse`) to 14.3 KiB (`sm-closed`).

Source: `experiments/results/sherman_morrison_benchmark.json`.

---

## error_vs_cancellation.svg

**Part 1.2 — float64 error against the cancellation ratio**

Each point is one Dirichlet parameter point; the y axis is the relative error
of `R(alpha)` against a 120-digit reference. The dashed line is `eps * rho`,
plotted rather than fitted. Spearman +0.960 over 15 orders of magnitude.

`rho` is the largest intermediate magnitude divided by `|S^2 - T^2|`, and is
returned in O(k) alongside `R` by
`dir_curvature_sm_closed_diagnostic`.

Source: `experiments/results/highprec_reliability.json`.

---

## error_vs_condition.svg

**Part 1.2 — the same errors against cond(g); the null result**

The companion to the figure above, and the reason the earlier operational
caveat ("do not trust R when cond(g) >~ 1e3") was withdrawn. Regression R^2 is
0.673 here against 0.986 for the cancellation ratio, and once rho is known the
partial correlation of cond(g) with what is left is +0.077.

Source: `experiments/results/highprec_reliability.json`.

---

## error_decomposition.svg

**Part 1.2 — where the float64 error comes from**

Parameter points ordered by `rho`, with the error split into the part
attributable to receiving psi values already rounded to double, the part added
by the float64 special functions, and the part added by the route's own
arithmetic. Only the arithmetic term tracks `rho`.

Source: `experiments/results/highprec_reliability.json`.
