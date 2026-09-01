"""
experiments/validate_curvature_implementation.py

Part A — validate the O(k^2) structured curvature route before asking whether
curvature is useful for anything.

A1 EXACTNESS
    Every float64 route -- structured, dense-naive, dense-pairwise -- against
    a 120-digit reference, at every parameter point where the dense path is
    computationally feasible: symmetric, strongly asymmetric, low
    concentration, high concentration, and deliberately ill-conditioned.
    The pass criterion is the measured accuracy law eps * rho * k^p rather
    than a flat tolerance; see docs/numerical_reliability.md.

A3 ACCUMULATION
    How much accuracy each contraction order costs, in units of eps*rho.
    The literal six-index route accumulates as k^7.7 -- it is the least
    accurate route as well as the slowest.

A2 COMPLEXITY
    Measured wall-clock for k in {3, 8, 16, 32, 64, 128, 256} across three
    implementations, not extrapolated from FLOP counts:
      dense-naive      six nested index loops              O(k^6)
      dense-pairwise   three sequential contractions       O(k^4)
      structured       closed form, T never materialised   O(k^2)
    Routes that would exceed the per-point time budget are skipped and
    reported as skipped rather than estimated.

Note on the implementations timed here: these are the standard-library
mirrors. `tests/test_router_common_mirror.py` pins them to `igad` itself
whenever numpy is installed. Absolute numpy timings would be faster; the
claim under test is the SCALING and the exactness, both of which are
implementation-independent.

    python -m experiments.validate_curvature_implementation
"""

import argparse
import math
import time

from experiments._router_common import (
    dir_cumulant_structure, dir_fisher, dir_scalar_curvature, mat_inv,
    print_table, save_results,
)
from experiments._highprec import (
    cancellation_ratio, hp_curvature_sherman_morrison, hp_dirichlet_inputs,
    precision, rel_error, to_decimal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Three routes to the same number
# ─────────────────────────────────────────────────────────────────────────────

def dense_tensor(alpha):
    """Materialise the full (k,k,k) third cumulant tensor."""
    k = len(alpha)
    c, d = dir_cumulant_structure(alpha)
    T = [[[c] * k for _ in range(k)] for _ in range(k)]
    for i in range(k):
        T[i][i][i] += d[i]
    return T


def R_dense_naive(alpha):
    """Literal six-index contraction. O(k^6)."""
    k = len(alpha)
    T = dense_tensor(alpha)
    M = mat_inv(dir_fisher(alpha))
    S = [sum(M[a][b] * T[a][b][m] for a in range(k) for b in range(k))
         for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    T_sq = 0.0
    for i in range(k):
        for j in range(k):
            for kk in range(k):
                t = T[i][j][kk]
                for a in range(k):
                    Mia = M[i][a]
                    for b in range(k):
                        Mjb = M[j][b]
                        for c in range(k):
                            T_sq += Mia * Mjb * M[kk][c] * t * T[a][b][c]
    return 0.25 * (T_sq - S_sq)


def R_dense_pairwise(alpha):
    """Same dense tensor, contracted pairwise. O(k^4) -- the optimize=True route."""
    k = len(alpha)
    T = dense_tensor(alpha)
    M = mat_inv(dir_fisher(alpha))
    S = [sum(M[a][b] * T[a][b][m] for a in range(k) for b in range(k))
         for m in range(k)]
    S_sq = sum(S[m] * M[m][n] * S[n] for m in range(k) for n in range(k))
    U1 = [[[sum(M[i][a] * T[i][j][kk] for i in range(k)) for kk in range(k)]
           for j in range(k)] for a in range(k)]
    U2 = [[[sum(M[j][b] * U1[a][j][kk] for j in range(k)) for kk in range(k)]
           for b in range(k)] for a in range(k)]
    U3 = [[[sum(M[kk][c] * U2[a][b][kk] for kk in range(k)) for c in range(k)]
           for b in range(k)] for a in range(k)]
    T_sq = sum(U3[a][b][c] * T[a][b][c]
               for a in range(k) for b in range(k) for c in range(k))
    return 0.25 * (T_sq - S_sq)


# ─────────────────────────────────────────────────────────────────────────────
# A1 — exactness
# ─────────────────────────────────────────────────────────────────────────────

A1_CASES = [
    ("symmetric k=3",            [4.0, 4.0, 4.0]),
    ("symmetric k=4",            [2.0, 2.0, 2.0, 2.0]),
    ("symmetric k=6",            [1.0] * 6),
    ("asymmetric k=3",           [1.5, 4.0, 6.5]),
    ("strongly asymmetric k=4",  [14.5, 0.5, 0.5, 0.5]),
    ("strongly asymmetric k=5",  [0.05, 3.0, 12.0, 0.4, 6.0]),
    ("low concentration k=3",    [0.05, 0.05, 0.05]),
    ("low concentration k=4",    [0.02, 0.03, 0.02, 0.05]),
    ("high concentration k=3",   [200.0, 200.0, 200.0]),
    ("high concentration k=4",   [500.0, 480.0, 510.0, 495.0]),
    ("ill-conditioned k=3",      [1e-3, 1.0, 5e3]),
    ("ill-conditioned k=4",      [1e-4, 1e-2, 1.0, 1e3]),
    ("near-degenerate k=5",      [1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
]


EPS = 2.0 ** -53
SAFETY = 64.0

# Achievable accuracy is eps * rho * k^p, where rho is the cancellation ratio
# and p depends on how much arithmetic the route does. Section A3 below
# measures p; these are the next integer up from what it reports, and A3 fails
# loudly if a measurement outgrows its constant.
SIZE_EXPONENT = {"structured": 2, "dense_pairwise": 3, "dense_naive": 8}


def run_a1(_unused_tol=None):
    """Each float64 route against a 120-digit reference, not against a sibling.

    This section used to compare the structured route to the dense route and
    pass or fail on a flat 1e-11. Both parts of that were wrong:

    - every route consumes the same psi' and psi'' values, so an error in the
      special functions cancels exactly when routes are compared to each other.
      That hid a ~6e-11 shared error on cases this table reported as passing.
    - a flat tolerance is not a property any implementation can satisfy. The
      achievable accuracy is eps * rho * k^p, where rho is the cancellation
      ratio of the expression; at rho = 9e14 no float64 route reaches 1e-11,
      and at rho = 2 the flat target is 1e5 times weaker than what is achieved.

    See docs/numerical_reliability.md for the measurement behind both points.
    """
    rows, records = [], []
    worst_ratio = 0.0
    for name, alpha in A1_CASES:
        k = len(alpha)
        r_struct = dir_scalar_curvature(alpha)
        r_naive = R_dense_naive(alpha)
        r_pair = R_dense_pairwise(alpha)

        with precision(120):
            parts = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))
            ref = parts["R"]
            rho = cancellation_ratio(parts)
            errs = {"structured": rel_error(to_decimal(r_struct), ref),
                    "dense_naive": rel_error(to_decimal(r_naive), ref),
                    "dense_pairwise": rel_error(to_decimal(r_pair), ref)}
            ref_float = float(ref)

        bounds = {route: max(SAFETY * EPS * rho * k ** p, SAFETY * EPS)
                  for route, p in SIZE_EXPONENT.items()}
        ratios = {route: errs[route] / bounds[route] for route in errs}
        ok = all(r < 1.0 for r in ratios.values())
        worst_ratio = max(worst_ratio, max(ratios.values()))
        cond = condition_number(dir_fisher(alpha))
        rows.append([name, k, "%.9f" % ref_float, "%.1e" % rho,
                     "%.2e" % errs["structured"], "%.2e" % errs["dense_pairwise"],
                     "%.2e" % errs["dense_naive"],
                     "%.2f" % max(ratios.values()), "PASS" if ok else "FAIL"])
        records.append({"case": name, "k": k, "alpha": alpha,
                        "R_reference_hp": repr(ref), "R_reference": ref_float,
                        "R_structured": r_struct, "R_dense_naive": r_naive,
                        "R_dense_pairwise": r_pair,
                        "rel_err_vs_reference": errs,
                        "cancellation_rho": rho, "bounds": bounds,
                        "error_over_bound": ratios,
                        "cond_g": cond, "pass": ok})
    print_table("A1 EXACTNESS -- every float64 route vs a 120-digit reference",
                ["case", "k", "R (reference)", "rho", "rel structured",
                 "rel pairwise", "rel naive", "worst err/bound", "verdict"], rows)
    print("  bound = %.0f * eps * rho * k^p, p = %s"
          % (SAFETY, ", ".join("%s %d" % (r, p)
                               for r, p in sorted(SIZE_EXPONENT.items()))))
    print("  worst error/bound ratio across %d cases: %.3f"
          % (len(A1_CASES), worst_ratio))
    print("  all pass: %s" % all(r["pass"] for r in records))
    print()
    return {"bound_rule": "%.0f * eps * rho * k^p" % SAFETY,
            "size_exponents": SIZE_EXPONENT,
            "worst_error_over_bound": worst_ratio,
            "all_pass": all(r["pass"] for r in records), "cases": records}


# ─────────────────────────────────────────────────────────────────────────────
# A3 — how much accuracy each contraction order costs
# ─────────────────────────────────────────────────────────────────────────────

def run_a3(ks=(3, 4, 5, 6, 7, 8, 10, 12, 14, 16)):
    """Error in units of eps*rho, as k grows at a fixed alpha.

    The cancellation ratio explains how much accuracy the *expression* costs.
    This measures how much the *contraction order* costs on top of it -- a
    route that touches more intermediate values accumulates more rounding.
    The result is the second reason to prefer the closed form: the literal
    six-index contraction is not merely slower, it is far less accurate.
    """
    rows, series = [], {r: [] for r in SIZE_EXPONENT}
    for k in ks:
        alpha = [2.0] * k
        vals = {"structured": dir_scalar_curvature(alpha),
                "dense_pairwise": R_dense_pairwise(alpha),
                "dense_naive": R_dense_naive(alpha)}
        with precision(120):
            parts = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))
            rho = cancellation_ratio(parts)
            excess = {r: rel_error(to_decimal(v), parts["R"]) / max(EPS * rho, EPS)
                      for r, v in vals.items()}
        for r, e in excess.items():
            series[r].append((k, e))
        rows.append([k, "%.2f" % rho] + ["%.1f" % excess[r]
                                         for r in sorted(SIZE_EXPONENT)])
    print_table("A3 ACCUMULATION -- error in units of eps*rho (alpha_i = 2)",
                ["k", "rho"] + sorted(SIZE_EXPONENT), rows)

    fitted, ok = {}, True
    for route in sorted(SIZE_EXPONENT):
        p = _loglog_slope(series[route])
        fitted[route] = p
        within = p is not None and p <= SIZE_EXPONENT[route]
        ok = ok and within
        print("    %-15s measured k^%.2f, bound uses k^%d  %s"
              % (route, p if p is not None else float("nan"),
                 SIZE_EXPONENT[route], "ok" if within else "EXPONENT TOO SMALL"))
    print("  the six-index contraction sums k^6 terms into one accumulator; "
          "that is\n  where its exponent comes from, and it is why the closed "
          "form is also\n  the more accurate route, not only the faster one.")
    print()
    return {"ks": list(ks), "measured_exponents": fitted,
            "declared_exponents": SIZE_EXPONENT, "exponents_valid": ok,
            "excess_over_eps_rho": {r: series[r] for r in series}}


def _loglog_slope(points):
    pts = [(math.log10(k), math.log10(max(e, 1e-3))) for k, e in points if k > 0]
    n = len(pts)
    if n < 2:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return (sum((p[0] - mx) * (p[1] - my) for p in pts) / den) if den else None


def condition_number(M):
    from experiments._router_common import sym_eig
    vals, _ = sym_eig(M)
    lo = min(abs(v) for v in vals)
    return (max(abs(v) for v in vals) / lo) if lo > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# A2 — measured complexity
# ─────────────────────────────────────────────────────────────────────────────

def timed(fn, alpha, budget):
    t0 = time.perf_counter()
    val = fn(alpha)
    dt = time.perf_counter() - t0
    return val, dt


def run_a2(ks, budget):
    rows = []
    records = []
    # A route is abandoned for all larger k once it exceeds the budget.
    alive = {"dense-naive": True, "dense-pairwise": True, "structured": True}
    for k in ks:
        alpha = [2.0] * k
        rec = {"k": k}
        row = [k]
        for name, fn, theo in (("dense-naive", R_dense_naive, "O(k^6)"),
                               ("dense-pairwise", R_dense_pairwise, "O(k^4)"),
                               ("structured", dir_scalar_curvature, "O(k^2)")):
            if not alive[name]:
                row.append("skipped")
                rec[name] = None
                continue
            val, dt = timed(fn, alpha, budget)
            rec[name] = dt
            rec[name + "_R"] = val
            row.append("%.4f s" % dt)
            if dt > budget:
                alive[name] = False
        # agreement among whichever routes ran
        vals = [rec.get(n + "_R") for n in alive if rec.get(n + "_R") is not None]
        rec["max_disagreement"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        row.append("%.2e" % rec["max_disagreement"] if len(vals) > 1 else "-")
        rows.append(row)
        records.append(rec)

    print_table("A2 COMPLEXITY -- measured wall clock, one evaluation "
                "(budget %.0f s, then skipped)" % budget,
                ["k", "dense-naive O(k^6)", "dense-pairwise O(k^4)",
                 "structured O(k^2)", "max disagreement"], rows)

    print("  Observed scaling (ratio of successive timings):")
    for name in ("dense-naive", "dense-pairwise", "structured"):
        seq = [(r["k"], r[name]) for r in records if r.get(name)]
        if len(seq) < 2:
            continue
        parts = []
        for (k0, t0), (k1, t1) in zip(seq, seq[1:]):
            if t0 > 1e-6:
                parts.append("k %d->%d: %.1fx" % (k0, k1, t1 / t0))
        print("    %-16s %s" % (name, "  ".join(parts)))
    print()
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[3, 8, 16, 32, 64, 128, 256])
    p.add_argument("--budget", type=float, default=20.0,
                   help="seconds per evaluation before a route is abandoned")
    a = p.parse_args()

    a1 = run_a1()
    a3 = run_a3()
    a2 = run_a2(a.ks, a.budget)

    print("=" * 78)
    print("PART A VERDICT")
    print("=" * 78)
    print("  A1 exactness  : %s (worst error/bound ratio %.3f, bound = %s)"
          % ("PASS" if a1["all_pass"] else "FAIL",
             a1["worst_error_over_bound"], a1["bound_rule"]))
    reached = max((r["k"] for r in a2 if r.get("structured")), default=0)
    dense_max = max((r["k"] for r in a2 if r.get("dense-naive")), default=0)
    print("  A2 reach      : structured evaluated up to k=%d; dense-naive died at k=%d"
          % (reached, dense_max))
    print()

    save_results("curvature_implementation_validation",
                 {"A1": a1, "A2": a2, "A3": a3,
                  "note": "standard-library mirrors; pinned to igad by "
                          "tests/test_router_common_mirror.py when numpy is present"})


if __name__ == "__main__":
    main()
