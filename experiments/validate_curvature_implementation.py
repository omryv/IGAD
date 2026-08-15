"""
experiments/validate_curvature_implementation.py

Part A — validate the O(k^2) structured curvature route before asking whether
curvature is useful for anything.

A1 EXACTNESS
    scalar_curvature_structured vs the general dense contraction, at every
    parameter point where the dense path is computationally feasible:
    symmetric, strongly asymmetric, low concentration, high concentration,
    and deliberately ill-conditioned. Target |dR| / |R| < 1e-11.

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
import time

from experiments._router_common import (
    dir_cumulant_structure, dir_fisher, dir_scalar_curvature, mat_inv,
    print_table, save_results,
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
    return 0.25 * (S_sq - T_sq)


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
    return 0.25 * (S_sq - T_sq)


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


def run_a1(tol):
    rows = []
    worst = 0.0
    records = []
    for name, alpha in A1_CASES:
        k = len(alpha)
        r_struct = dir_scalar_curvature(alpha)
        r_naive = R_dense_naive(alpha)
        r_pair = R_dense_pairwise(alpha)
        scale = max(1e-300, abs(r_naive))
        abs_err = abs(r_struct - r_naive)
        rel_err = abs_err / scale
        cond = condition_number(dir_fisher(alpha))
        worst = max(worst, rel_err)
        rows.append([name, k, "%.9f" % r_struct, "%.2e" % abs_err,
                     "%.2e" % rel_err, "%.1e" % cond,
                     "PASS" if rel_err < tol else "FAIL"])
        records.append({"case": name, "k": k, "alpha": alpha,
                        "R_structured": r_struct, "R_dense_naive": r_naive,
                        "R_dense_pairwise": r_pair, "abs_err": abs_err,
                        "rel_err": rel_err, "cond_g": cond,
                        "pass": rel_err < tol})
    print_table("A1 EXACTNESS -- structured vs dense-naive (target rel < %.0e)" % tol,
                ["case", "k", "R_structured", "|dR|", "rel", "cond(g)", "verdict"],
                rows)
    print("  worst relative error across %d cases: %.2e" % (len(A1_CASES), worst))
    print("  all pass: %s" % all(r["pass"] for r in records))
    print()
    return {"tolerance": tol, "worst_rel_err": worst,
            "all_pass": all(r["pass"] for r in records), "cases": records}


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
    p.add_argument("--tol", type=float, default=1e-11)
    p.add_argument("--budget", type=float, default=20.0,
                   help="seconds per evaluation before a route is abandoned")
    a = p.parse_args()

    a1 = run_a1(a.tol)
    a2 = run_a2(a.ks, a.budget)

    print("=" * 78)
    print("PART A VERDICT")
    print("=" * 78)
    print("  A1 exactness  : %s (worst relative error %.2e, target < %.0e)"
          % ("PASS" if a1["all_pass"] else "FAIL", a1["worst_rel_err"], a.tol))
    reached = max((r["k"] for r in a2 if r.get("structured")), default=0)
    dense_max = max((r["k"] for r in a2 if r.get("dense-naive")), default=0)
    print("  A2 reach      : structured evaluated up to k=%d; dense-naive died at k=%d"
          % (reached, dense_max))
    print()

    save_results("curvature_implementation_validation",
                 {"A1": a1, "A2": a2,
                  "note": "standard-library mirrors; pinned to igad by "
                          "tests/test_router_common_mirror.py when numpy is present"})


if __name__ == "__main__":
    main()
