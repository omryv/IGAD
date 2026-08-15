"""
experiments/benchmark_sherman_morrison.py

Part 1.1 -- make the Dirichlet curvature path genuinely sub-cubic end to end,
and measure it rather than assert it.

The problem
-----------
`igad.curvature.scalar_curvature_structured` contracts in O(k^2), but it calls
`np.linalg.inv` (the stdlib mirror calls `mat_inv`). A generic inverse is
O(k^3), so the *complete* path was O(k^3) and the contraction's O(k^2) was
never the binding constraint. `docs/validation_report.md` recorded exactly this
symptom -- "the structured route's own scaling drifts above k^2 at large k
because the O(k^3) matrix inverse, not the contraction, dominates there."

The fix
-------
The Dirichlet Fisher metric is diagonal-plus-rank-one,

    g = D - c 1 1^T,   D_ii = psi'(alpha_i),   c = psi'(alpha_0),

so Sherman-Morrison gives the inverse in closed form:

    g^-1 = D^-1 + [ c / (1 - c 1^T D^-1 1) ] D^-1 1 1^T D^-1.

Two routes follow.

  sm-matrix   builds that k x k inverse explicitly, then contracts.
              O(k^2) time, O(k^2) memory -- what Part 1.1 asked for.
  sm-closed   substitutes the factored inverse into the contraction and never
              allocates anything k x k.
              O(k) time, O(k) memory -- strictly better than what was asked.

What is measured
----------------
B1  exactness    all three float64 routes against a >=120-digit reference.
B2  wall clock   k in {8,16,32,64,128,256,512}, repeated, minimum taken.
B3  memory       tracemalloc peak allocation per route per k.
B4  scaling      least-squares log-log slope, fitted on the measured points.

    python -m experiments.benchmark_sherman_morrison

Everything here is standard-library Python. Absolute timings under numpy would
be smaller; the claims under test are the exactness, the scaling exponent and
the memory growth, all of which are implementation-independent.
"""

import argparse
import math
import time
import tracemalloc

from experiments._router_common import (
    dir_curvature_dense_inverse, dir_curvature_sm_closed, dir_curvature_sm_matrix,
    dir_fisher_from, dir_polygamma_inputs, mat_inv, print_table, save_results,
)
from experiments._highprec import (
    hp_condition_number, hp_curvature_sherman_morrison, hp_dirichlet_inputs,
    hp_fisher, precision, rel_error, to_decimal,
)

ROUTES = [
    ("dense-inverse", dir_curvature_dense_inverse, "O(k^3)"),
    ("sm-matrix",     dir_curvature_sm_matrix,     "O(k^2)"),
    ("sm-closed",     dir_curvature_sm_closed,     "O(k)"),
]

# The 13 points from experiments/validate_curvature_implementation.py, so the
# new routes are held to the same cases the old ones were.
B1_CASES = [
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
    ("near-degenerate k=5",      [1e-5] * 5),
    ("realistic router k=8",     [0.6, 0.9, 1.4, 0.3, 2.1, 0.8, 1.1, 0.5]),
    ("realistic router k=16",    [0.4 + 0.1 * i for i in range(16)]),
]


# ─────────────────────────────────────────────────────────────────────────────
# B1 -- exactness against a high-precision reference
# ─────────────────────────────────────────────────────────────────────────────

def run_b1(digits):
    rows, records = [], []
    for name, alpha in B1_CASES:
        f64 = {}
        inputs = dir_polygamma_inputs(alpha)
        for route, fn, _ in ROUTES:
            f64[route] = fn(*inputs)

        with precision(digits):
            hp_inputs = hp_dirichlet_inputs(alpha)
            ref = hp_curvature_sherman_morrison(*hp_inputs)["R"]
            cond = hp_condition_number(hp_fisher(hp_inputs[0], hp_inputs[1]))
            errs = {route: rel_error(to_decimal(v), ref) for route, v in f64.items()}
            ref_float = float(ref)

        rec = {"case": name, "k": len(alpha), "alpha": alpha,
               "R_reference_hp": repr(ref), "R_reference_float": ref_float,
               "cond_g": cond,
               "R_float64": f64, "rel_err_vs_hp": errs}
        records.append(rec)
        rows.append([name, len(alpha), "%.12f" % ref_float, "%.1e" % cond]
                    + ["%.2e" % errs[r] for r, _, _ in ROUTES])

    print_table("B1 EXACTNESS -- float64 routes vs a %d-digit reference" % digits,
                ["case", "k", "R (reference)", "cond(g)"]
                + ["rel err %s" % r for r, _, _ in ROUTES], rows)

    worst = {r: max(rec["rel_err_vs_hp"][r] for rec in records)
             for r, _, _ in ROUTES}
    for route, _, _ in ROUTES:
        print("  worst relative error, %-14s %.2e" % (route, worst[route]))
    # Every route sees the same polygamma inputs, so route-to-route spread is
    # attributable to the arithmetic and not to the special functions.
    spread = max(
        abs(rec["R_float64"][a] - rec["R_float64"][b]) / max(abs(rec["R_reference_float"]), 1e-300)
        for rec in records for a, _, _ in ROUTES for b, _, _ in ROUTES)
    print("  worst route-to-route spread                %.2e" % spread)
    print()
    return {"digits": digits, "cases": records, "worst_rel_err": worst,
            "worst_route_spread": spread}


# ─────────────────────────────────────────────────────────────────────────────
# B2/B3 -- wall clock and peak memory
# ─────────────────────────────────────────────────────────────────────────────

def time_route(fn, inputs, budget, min_repeats=3):
    """Minimum wall clock over repeats; repeat count adapts to the timer."""
    best = float("inf")
    total = 0.0
    n = 0
    while n < min_repeats or (total < 0.05 and n < 2000):
        t0 = time.perf_counter()
        fn(*inputs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        n += 1
        if dt > budget:
            break
    return best, n


def peak_memory(fn, inputs):
    """Peak bytes allocated by one evaluation, above the pre-existing baseline."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn(*inputs)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return max(peak - base, 0)


# A route is not attempted above this k. dense-inverse and sm-matrix both hold
# something k x k in Python objects (~32 bytes per entry), so the ceiling is a
# memory limit rather than a time limit; sm-closed holds only O(k).
MAX_K = {"dense-inverse": 1024, "sm-matrix": 2048, "sm-closed": 2 ** 17}


def run_b2(ks, budget, mem_ks):
    rows, records = [], []
    alive = {r: True for r, _, _ in ROUTES}
    for k in ks:
        alpha = [2.0] * k
        inputs = dir_polygamma_inputs(alpha)
        rec = {"k": k}
        row = [k]
        for route, fn, _ in ROUTES:
            if not alive[route] or k > MAX_K[route]:
                row.append("--")
                continue
            dt, reps = time_route(fn, inputs, budget)
            rec[route + "_seconds"] = dt
            rec[route + "_repeats"] = reps
            rec[route + "_R"] = fn(*inputs)
            if k in mem_ks:
                rec[route + "_peak_bytes"] = peak_memory(fn, inputs)
            row.append("%.6f s" % dt)
            if dt > budget:
                alive[route] = False
        vals = [rec[r + "_R"] for r, _, _ in ROUTES if r + "_R" in rec]
        rec["max_disagreement"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        row.append("%.2e" % rec["max_disagreement"] if len(vals) > 1 else "-")
        rows.append(row)
        records.append(rec)

    print_table("B2 WALL CLOCK -- one evaluation, minimum over repeats "
                "(-- = past this route's ceiling; abandoned above %.0f s)" % budget,
                ["k"] + ["%s %s" % (r, o) for r, _, o in ROUTES]
                + ["max disagreement"], rows)

    mem_rows = [[rec["k"]] + ["%.1f KiB" % (rec[r + "_peak_bytes"] / 1024.0)
                              if r + "_peak_bytes" in rec else "--"
                              for r, _, _ in ROUTES]
                for rec in records if rec["k"] in mem_ks]
    print_table("B3 PEAK MEMORY -- tracemalloc, one evaluation",
                ["k"] + ["%s %s" % (r, o) for r, _, o in ROUTES], mem_rows)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# B4 -- fitted scaling exponents
# ─────────────────────────────────────────────────────────────────────────────

def loglog_slope(points):
    """Least-squares slope of log2(y) on log2(x)."""
    pts = [(math.log2(x), math.log2(y)) for x, y in points if x > 0 and y > 0]
    n = len(pts)
    if n < 2:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else None


def run_b4(records, fit_from):
    rows = []
    fits = {}
    for route, _, theo in ROUTES:
        times = [(r["k"], r.get(route + "_seconds")) for r in records
                 if r.get(route + "_seconds")]
        mems = [(r["k"], r.get(route + "_peak_bytes")) for r in records
                if r.get(route + "_peak_bytes")]
        tail = [p for p in times if p[0] >= fit_from]
        s_all = loglog_slope(times)
        s_tail = loglog_slope(tail)
        s_mem = loglog_slope(mems)
        ratios = "  ".join(
            "%d->%d %.1fx" % (a[0], b[0], b[1] / a[1])
            for a, b in zip(times, times[1:]) if a[1] > 0)
        fits[route] = {"theoretical": theo, "slope_all_k": s_all,
                       "slope_k_ge_%d" % fit_from: s_tail,
                       "memory_slope": s_mem,
                       "k_max_measured": max((p[0] for p in times), default=0)}
        rows.append([route, theo,
                     "%.2f" % s_all if s_all is not None else "-",
                     "%.2f" % s_tail if s_tail is not None else "-",
                     "%.2f" % s_mem if s_mem is not None else "-",
                     ratios])
    print_table("B4 MEASURED SCALING -- least-squares slope of log2(time) on log2(k)",
                ["route", "predicted", "slope (all k)",
                 "slope (k>=%d)" % fit_from, "memory slope",
                 "successive ratios"], rows)
    return fits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--extended-ks", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                   help="extra sizes; each route stops at its own ceiling")
    p.add_argument("--budget", type=float, default=25.0)
    p.add_argument("--fit-from", type=int, default=64,
                   help="smallest k included in the asymptotic slope fit")
    p.add_argument("--digits", type=int, default=120)
    a = p.parse_args()

    b1 = run_b1(a.digits)
    b2 = run_b2(sorted(set(a.ks + a.extended_ks)), a.budget, set(a.ks))
    b4 = run_b4(b2, a.fit_from)

    print("=" * 78)
    print("PART 1.1 VERDICT")
    print("=" * 78)
    for route, _, theo in ROUTES:
        f = b4[route]
        print("  %-14s predicted %-7s measured slope %s (k>=%d), memory slope %s,"
              " reached k=%d"
              % (route, theo,
                 "%.2f" % f["slope_k_ge_%d" % a.fit_from]
                 if f["slope_k_ge_%d" % a.fit_from] is not None else "-",
                 a.fit_from,
                 "%.2f" % f["memory_slope"] if f["memory_slope"] is not None else "-",
                 f["k_max_measured"]))
    print("  B1 exactness: worst relative error vs the %d-digit reference"
          % a.digits)
    for route, _, _ in ROUTES:
        print("      %-14s %.2e" % (route, b1["worst_rel_err"][route]))
    print()

    save_results("sherman_morrison_benchmark",
                 {"B1_exactness": b1, "B2_B3_measurements": b2,
                  "B4_scaling": b4,
                  "config": {"ks": a.ks, "extended_ks": a.extended_ks,
                             "budget_seconds": a.budget, "fit_from": a.fit_from,
                             "reference_digits": a.digits},
                  "note": "standard-library implementations; scaling and "
                          "exactness are implementation-independent, absolute "
                          "timings are not"})


if __name__ == "__main__":
    main()
