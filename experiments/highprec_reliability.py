"""
experiments/highprec_reliability.py

Part 1.2 -- arbitrate the float64 disagreements with a high-precision
reference, and find out what the error actually tracks.

The claim being tested
----------------------
`docs/validation_report.md` reported four Dirichlet parameter points where the
dense and structured float64 routes disagree by more than 1e-11, ruled out
summation order and (on one measurement) cancellation, and concluded:

    "The residual therefore sits in the shared, ill-conditioned g^-1 and the
     polygamma inputs. ... do not trust R beyond ~1e-11 relative when
     cond(g) >~ 1e3, whichever route computes it."

That conclusion was reached by comparing two float64 routes to each other.
This script does not do that. It computes a >=120-digit reference and asks
four separate questions:

C1  Are the dense and structured formulae the same number?
    Four routes -- literal 6-index contraction, pairwise contraction, the
    structured collapse, Sherman-Morrison -- all evaluated at high precision,
    and the reference re-run at a higher precision to prove it has converged.

C2  Where does the float64 error come from?
    Because every route can be fed its polygamma inputs explicitly, the error
    splits three ways and each part is measured, not inferred:
      input-rounding floor  what an *exact* evaluation loses from receiving
                            psi' and psi'' already rounded to double
      special-function      what the float64 trigamma/tetragamma add
      arithmetic            what each route's own operations add

C3  What predicts the arithmetic error -- cond(g), or cancellation?
    Both are computed exactly at high precision and regressed against the
    measured error. The answer decides whether the operational caveat should
    be phrased in terms of cond(g) at all.

C4  A reliability boundary that a caller can actually evaluate.
    Whatever predicts the error has to be computable in float64, in O(k),
    alongside R itself -- otherwise it is a post-hoc explanation and not a
    usable guard rail.

    python -m experiments.highprec_reliability
"""

import argparse
import math

from experiments._router_common import (
    dir_curvature_dense_inverse, dir_curvature_sm_closed,
    dir_curvature_sm_closed_diagnostic, dir_curvature_sm_matrix,
    dir_polygamma_inputs, print_table, save_results,
)
from experiments._highprec import (
    cancellation_ratio, hp_condition_number, hp_curvature_dense_naive,
    hp_curvature_dense_pairwise, hp_curvature_sherman_morrison,
    hp_curvature_structured, hp_dirichlet_inputs, hp_fisher, precision,
    rel_error, to_decimal,
)

EPS = 2.0 ** -53          # unit roundoff for binary64

F64_ROUTES = [
    ("dense-inverse", dir_curvature_dense_inverse),
    ("sm-matrix", dir_curvature_sm_matrix),
    ("sm-closed", dir_curvature_sm_closed),
]

HP_ROUTES = [
    ("hp-dense-naive", hp_curvature_dense_naive),
    ("hp-dense-pairwise", hp_curvature_dense_pairwise),
    ("hp-structured", hp_curvature_structured),
    ("hp-sherman-morrison", hp_curvature_sherman_morrison),
]


def build_cases():
    """Parameter points spanning conditioning and cancellation independently."""
    cases = [
        # the 13 points already on the record
        ("symmetric k=3", [4.0, 4.0, 4.0]),
        ("symmetric k=4", [2.0, 2.0, 2.0, 2.0]),
        ("symmetric k=6", [1.0] * 6),
        ("asymmetric k=3", [1.5, 4.0, 6.5]),
        ("strongly asymmetric k=4", [14.5, 0.5, 0.5, 0.5]),
        ("strongly asymmetric k=5", [0.05, 3.0, 12.0, 0.4, 6.0]),
        ("low concentration k=3", [0.05, 0.05, 0.05]),
        ("low concentration k=4", [0.02, 0.03, 0.02, 0.05]),
        ("high concentration k=3", [200.0, 200.0, 200.0]),
        ("high concentration k=4", [500.0, 480.0, 510.0, 495.0]),
        ("ill-conditioned k=3", [1e-3, 1.0, 5e3]),
        ("ill-conditioned k=4", [1e-4, 1e-2, 1.0, 1e3]),
        ("near-degenerate k=5", [1e-5] * 5),
    ]
    # concentration sweep: raises cancellation while conditioning stays modest
    for k in (3, 5, 8):
        for a in (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5):
            cases.append(("conc k=%d a=%g" % (k, a), [a] * k))
    # spread sweep: raises cond(g) sharply while cancellation stays modest
    for m in range(0, 8):
        cases.append(("spread k=3 m=%d" % m, [10.0 ** -m, 1.0, 10.0 ** m]))
    for m in range(0, 6):
        cases.append(("spread k=5 m=%d" % m,
                      [10.0 ** -m, 10.0 ** (-m / 2.0), 1.0,
                       10.0 ** (m / 2.0), 10.0 ** m]))
    # mixed: concentrated but unequal
    for a in (50.0, 500.0, 5000.0):
        cases.append(("mixed k=4 a=%g" % a, [a, a * 1.05, a * 0.95, a * 1.10]))
    return cases


def build_size_cases(ks=(3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)):
    """Expert-count sweep at a fixed, well-conditioned alpha.

    Section C3 deliberately holds k <= 8 so that k cannot confound the
    rho-versus-cond(g) comparison. This set varies k and nothing else, which
    is what section C5 needs: eps*rho captures the cancellation but not the
    rounding that accumulates across k-term sums, and that second factor is
    only visible when k moves.
    """
    return [("size k=%d" % k, [2.0] * k) for k in ks]


# ─────────────────────────────────────────────────────────────────────────────
# C1 -- are the dense and structured formulae the same number?
# ─────────────────────────────────────────────────────────────────────────────

def run_c1(cases, digits, check_digits):
    rows, records = [], []
    worst = {name: 0.0 for name, _ in HP_ROUTES}
    worst_converge = 0.0
    for case_name, alpha in cases:
        k = len(alpha)
        with precision(digits):
            inputs = hp_dirichlet_inputs(alpha)
            vals = {}
            for route, fn in HP_ROUTES:
                if route == "hp-dense-naive" and k > 6:
                    continue          # O(k^6) in Decimal; the pairwise route covers it
                vals[route] = fn(*inputs)["R"]
            ref = vals["hp-sherman-morrison"]
            errs = {r: rel_error(v, ref) for r, v in vals.items()}
        # the reference itself must not move when the precision is raised
        with precision(check_digits):
            hi = hp_curvature_sherman_morrison(*hp_dirichlet_inputs(alpha))["R"]
        converge = rel_error(to_decimal(ref), hi)

        for r, e in errs.items():
            worst[r] = max(worst[r], e)
        worst_converge = max(worst_converge, converge)
        records.append({"case": case_name, "k": k, "alpha": alpha,
                        "rel_err_between_hp_routes": errs,
                        "rel_change_at_%d_digits" % check_digits: converge})
        rows.append([case_name, k] +
                    ["%.1e" % errs[r] if r in errs else "-" for r, _ in HP_ROUTES] +
                    ["%.1e" % converge])

    print_table("C1 -- do the dense and structured formulae agree at %d digits?"
                % digits,
                ["case", "k"] + [r for r, _ in HP_ROUTES] +
                ["change at %dd" % check_digits], rows)
    for route, _ in HP_ROUTES:
        print("  worst disagreement vs Sherman-Morrison, %-20s %.2e"
              % (route, worst[route]))
    print("  worst change when the reference is recomputed at %d digits: %.2e"
          % (check_digits, worst_converge))
    print()
    return {"digits": digits, "check_digits": check_digits, "cases": records,
            "worst_disagreement": worst, "worst_precision_change": worst_converge}


# ─────────────────────────────────────────────────────────────────────────────
# C2 -- where does the float64 error come from?
# ─────────────────────────────────────────────────────────────────────────────

# Jacobi over Decimal is O(k^3) per sweep; above this k the condition number
# costs more than it is worth, and section C3 -- the only consumer -- holds
# k <= 8 anyway.
MAX_K_COND = 24


def run_c2(cases, digits, title="C2 -- decomposition of the float64 relative error"):
    rows, records = [], []
    for case_name, alpha in cases:
        f64_inputs = dir_polygamma_inputs(alpha)          # library psi values

        with precision(digits):
            exact_inputs = hp_dirichlet_inputs(alpha)
            parts = hp_curvature_sherman_morrison(*exact_inputs)
            R_exact = parts["R"]
            rho = cancellation_ratio(parts)
            cond = (hp_condition_number(hp_fisher(exact_inputs[0], exact_inputs[1]))
                    if len(alpha) <= MAX_K_COND else None)

            # (a) exact psi values rounded to double, then evaluated exactly:
            #     the floor no float64 pipeline fed by doubles can beat.
            rounded = _round_inputs(exact_inputs)
            R_inputround = hp_curvature_sherman_morrison(*rounded)["R"]
            input_floor = rel_error(R_inputround, R_exact)

            # (b) the library's psi values, evaluated exactly: the special
            #     functions' own contribution.
            lifted = _lift_inputs(f64_inputs)
            R_libpsi = hp_curvature_sherman_morrison(*lifted)["R"]
            specfun = rel_error(R_libpsi, R_exact)

            # (c) each float64 route, against the exact evaluation of the same
            #     inputs: the arithmetic's own contribution.
            arithmetic, total = {}, {}
            for route, fn in F64_ROUTES:
                v = to_decimal(fn(*f64_inputs))
                arithmetic[route] = rel_error(v, R_libpsi)
                total[route] = rel_error(v, R_exact)
            R_exact_float = float(R_exact)

        # the same cancellation ratio, estimated in float64 alongside R itself
        _, rho_hat = dir_curvature_sm_closed_diagnostic(*f64_inputs)

        records.append({
            "case": case_name, "k": len(alpha), "alpha": alpha,
            "R_exact": R_exact_float, "cond_g": cond, "cancellation_rho": rho,
            "rho_hat_float64": rho_hat,
            "input_rounding_floor": input_floor,
            "special_function_error": specfun,
            "arithmetic_error": arithmetic, "total_error": total,
            "predicted_floor_eps_rho": EPS * rho,
        })
        rows.append([case_name, len(alpha),
                     "%.1e" % cond if cond is not None else "-", "%.1e" % rho,
                     "%.1e" % input_floor, "%.1e" % specfun]
                    + ["%.1e" % arithmetic[r] for r, _ in F64_ROUTES]
                    + ["%.1e" % total["sm-closed"]])

    print_table(title,
                ["case", "k", "cond(g)", "rho", "input floor", "spec-fn"]
                + ["arith %s" % r for r, _ in F64_ROUTES] + ["total sm-closed"],
                rows)
    return records


def _round_inputs(inputs):
    tri, tri0, tet, tet0 = inputs
    return ([to_decimal(float(v)) for v in tri], to_decimal(float(tri0)),
            [to_decimal(float(v)) for v in tet], to_decimal(float(tet0)))


def _lift_inputs(inputs):
    tri, tri0, tet, tet0 = inputs
    return ([to_decimal(v) for v in tri], to_decimal(tri0),
            [to_decimal(v) for v in tet], to_decimal(tet0))


# ─────────────────────────────────────────────────────────────────────────────
# C3 -- what predicts the error?
# ─────────────────────────────────────────────────────────────────────────────

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((a - mx) ** 2 for a in xs)
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den if den else float("nan")
    intercept = my - slope * mx
    ss_res = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(xs, ys))
    ss_tot = sum((b - my) ** 2 for b in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


FLOOR = 1e-17     # errors below this are exact hits; log10 of 0 is undefined


def run_c3(records, route="sm-closed"):
    xs_rho, xs_cond, ys = [], [], []
    for rec in records:
        err = max(rec["arithmetic_error"][route], FLOOR)
        ys.append(math.log10(err))
        xs_rho.append(math.log10(max(rec["cancellation_rho"], 1.0)))
        xs_cond.append(math.log10(max(rec["cond_g"], 1.0)))

    rows = []
    stats = {}
    for name, xs in (("log10 cancellation rho", xs_rho),
                     ("log10 cond(g)", xs_cond)):
        slope, intercept, r2 = _ols(xs, ys)
        pear = _pearson(xs, ys)
        spear = _pearson(_ranks(xs), _ranks(ys))
        stats[name] = {"slope": slope, "intercept": intercept, "r2": r2,
                       "pearson": pear, "spearman": spear}
        rows.append([name, "%.3f" % slope, "%.3f" % intercept, "%.3f" % r2,
                     "%.3f" % pear, "%.3f" % spear])
    print_table("C3 -- predictors of log10(arithmetic relative error), route=%s"
                % route,
                ["predictor", "slope", "intercept", "R^2", "Pearson", "Spearman"],
                rows)

    # partial check: does cond(g) still explain anything once rho is known?
    slope_r, inter_r, _ = _ols(xs_rho, ys)
    resid = [y - (inter_r + slope_r * x) for x, y in zip(xs_rho, ys)]
    partial = _pearson(xs_cond, resid)
    slope_c, inter_c, _ = _ols(xs_cond, ys)
    resid_c = [y - (inter_c + slope_c * x) for x, y in zip(xs_cond, ys)]
    partial_rho = _pearson(xs_rho, resid_c)
    stats["partial_corr_cond_given_rho"] = partial
    stats["partial_corr_rho_given_cond"] = partial_rho
    print("  correlation of log10 cond(g) with the residual after rho: %+.3f"
          % partial)
    print("  correlation of log10 rho with the residual after cond(g): %+.3f"
          % partial_rho)
    print()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# C4 -- a boundary a caller can evaluate
# ─────────────────────────────────────────────────────────────────────────────

def run_c5(records):
    """How does the error grow with k once rho is divided out?

    eps*rho accounts for the cancellation. What it cannot account for is the
    rounding that accumulates across the k-term sums in the contraction (and,
    for the dense route, across O(k^3) elimination steps). This section
    measures that second factor directly: alpha = 2 for every point, so rho
    barely moves and k is the only thing varying.
    """
    rows, stats = [], {}
    for route, _ in F64_ROUTES:
        xs, ys = [], []
        for rec in records:
            err = max(rec["arithmetic_error"][route], FLOOR)
            excess = err / max(EPS * rec["cancellation_rho"], EPS)
            xs.append(math.log10(rec["k"]))
            ys.append(math.log10(max(excess, 1e-3)))
        slope, intercept, r2 = _ols(xs, ys)
        stats[route] = {"slope_vs_k": slope, "intercept": intercept, "r2": r2}
    for rec in records:
        rows.append([rec["k"], "%.2f" % rec["cancellation_rho"]]
                    + ["%.1f" % (max(rec["arithmetic_error"][r], FLOOR)
                                 / max(EPS * rec["cancellation_rho"], EPS))
                       for r, _ in F64_ROUTES])
    print_table("C5 -- error in units of eps*rho, as k grows (alpha_i = 2 throughout)",
                ["k", "rho"] + ["%s / (eps rho)" % r for r, _ in F64_ROUTES], rows)
    for route, _ in F64_ROUTES:
        s = stats[route]
        print("  %-14s excess ~ k^%.2f  (R^2 %.3f)"
              % (route, s["slope_vs_k"], s["r2"]))
    print("  eps*rho*k is therefore the bound used in C4 and in "
          "tests/test_sherman_morrison.py.")
    print()
    return stats


# From C5: the excess over eps*rho grows like k^1.11 for the O(k) route and
# like k^2.0-2.2 for the two routes that build a k x k matrix. The bound uses
# the next integer up.
SIZE_EXPONENT = {"sm-closed": 1, "sm-matrix": 2, "dense-inverse": 2}


def run_c4(records, route="sm-closed", safety=8.0):
    """Check the bound for every route, and check that rho survives float64."""
    rows = []
    per_route = {}
    for r, _ in F64_ROUTES:
        covered, worst_ratio = 0, 0.0
        for rec in records:
            err = max(rec["total_error"][r], FLOOR)
            bound = _bound(rec, r, safety)
            covered += err <= bound
            worst_ratio = max(worst_ratio, err / bound)
        per_route[r] = {"exponent": SIZE_EXPONENT[r],
                        "cases_within_bound": covered,
                        "worst_error_over_bound": worst_ratio}

    for rec in records:
        err = max(rec["total_error"][route], FLOOR)
        bound = _bound(rec, route, safety)
        rows.append([rec["case"], "%.1e" % rec["cancellation_rho"],
                     "%.1e" % rec["rho_hat_float64"],
                     "%.1e" % err, "%.1e" % bound,
                     "yes" if err <= bound else "NO"])
    print_table("C4 -- is  %g * max(eps*rho*k^p, input floor, spec-fn)  a bound "
                "on the total error of %s (p=%d)?"
                % (safety, route, SIZE_EXPONENT[route]),
                ["case", "rho (exact)", "rho (float64)", "measured error",
                 "bound", "within"], rows)
    for r, _ in F64_ROUTES:
        s = per_route[r]
        print("  %-14s p=%d: bound holds for %d / %d cases, worst ratio %.3f"
              % (r, s["exponent"], s["cases_within_bound"], len(records),
                 s["worst_error_over_bound"]))
    covered = per_route[route]["cases_within_bound"]
    worst_ratio = per_route[route]["worst_error_over_bound"]

    # Is the float64 estimate of rho usable? It only has to be right to within
    # an order of magnitude to tell a caller how many digits survive.
    ratios = [rec["rho_hat_float64"] / rec["cancellation_rho"]
              for rec in records if rec["cancellation_rho"] > 0
              and math.isfinite(rec["rho_hat_float64"])]
    within_2x = sum(1 for r in ratios if 0.5 <= r <= 2.0)
    within_10x = sum(1 for r in ratios if 0.1 <= r <= 10.0)
    print("  float64 rho_hat vs exact rho: within 2x on %d/%d, within 10x on "
          "%d/%d (worst ratio %.2f)"
          % (within_2x, len(ratios), within_10x, len(ratios),
             max(max(ratios), 1.0 / min(ratios))))

    # digits of R that survive, banded by rho -- each case counted once
    print()
    print("  surviving significant decimal digits of R, by cancellation band:")
    bands = [(1e0, 1e2), (1e2, 1e4), (1e4, 1e6), (1e6, 1e8),
             (1e8, 1e10), (1e10, 1e12), (1e12, float("inf"))]
    band_rows = []
    for lo, hi in bands:
        sel = [r for r in records if lo <= r["cancellation_rho"] < hi]
        if not sel:
            continue
        worst = max(r["total_error"][route] for r in sel)
        digits = -math.log10(max(worst, FLOOR))
        band_rows.append({"rho_lo": lo, "rho_hi": hi if hi != float("inf") else None,
                          "n_cases": len(sel), "worst_rel_err": worst,
                          "surviving_digits": digits})
        print("    %-8.0e <= rho < %-8s : %2d cases, worst rel err %.1e, "
              "~%4.1f digits survive"
              % (lo, ("%.0e" % hi) if hi != float("inf") else "inf",
                 len(sel), worst, digits))
    print()
    return {"safety_factor": safety, "cases_within_bound": covered,
            "n_cases": len(records), "worst_error_over_bound": worst_ratio,
            "per_route": per_route, "size_exponent": SIZE_EXPONENT,
            "rho_hat_within_2x": within_2x, "rho_hat_within_10x": within_10x,
            "rho_hat_n": len(ratios), "bands": band_rows}


def _bound(rec, route, safety):
    return safety * max(
        EPS * rec["cancellation_rho"] * rec["k"] ** SIZE_EXPONENT[route],
        rec["input_rounding_floor"], rec["special_function_error"], EPS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digits", type=int, default=120)
    p.add_argument("--check-digits", type=int, default=200)
    p.add_argument("--route", default="sm-closed")
    a = p.parse_args()

    cases = build_cases()
    c1 = run_c1(cases, a.digits, a.check_digits)
    c2 = run_c2(cases, a.digits)
    c3 = run_c3(c2, a.route)
    c5_records = run_c2(build_size_cases(), a.digits,
                        title="C5 inputs -- expert-count sweep at alpha_i = 2")
    c5 = run_c5(c5_records)
    c4 = run_c4(c2 + c5_records, a.route)

    print("=" * 78)
    print("PART 1.2 VERDICT")
    print("=" * 78)
    same = max(c1["worst_disagreement"].values())
    print("  C1  dense == structured at %d digits to %.1e -- they are the same"
          " number, so every float64 disagreement is numerical."
          % (a.digits, same))
    rho_r2 = c3["log10 cancellation rho"]["r2"]
    cond_r2 = c3["log10 cond(g)"]["r2"]
    print("  C3  log10(error) ~ log10(rho): R^2 = %.3f, slope %.2f"
          % (rho_r2, c3["log10 cancellation rho"]["slope"]))
    print("      log10(error) ~ log10(cond g): R^2 = %.3f, slope %.2f"
          % (cond_r2, c3["log10 cond(g)"]["slope"]))
    print("  C5  excess over eps*rho grows as k^%.2f for %s"
          % (c5[a.route]["slope_vs_k"], a.route))
    print("  C4  bound held for %d / %d cases"
          % (c4["cases_within_bound"], c4["n_cases"]))
    print()

    save_results("highprec_reliability",
                 {"C1_route_equivalence": c1,
                  "C2_error_decomposition": c2,
                  "C3_predictors": c3,
                  "C4_boundary": c4,
                  "C5_size_dependence": c5,
                  "C5_size_records": c5_records,
                  "config": {"digits": a.digits, "check_digits": a.check_digits,
                             "eps": EPS, "route": a.route,
                             "max_k_for_condition_number": MAX_K_COND}})


if __name__ == "__main__":
    main()
